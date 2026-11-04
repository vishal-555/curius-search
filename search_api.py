"""
Semantic search API for vishal.fast/reading.

    GET /search?q=...   top 10 articles by meaning: id, title, url, score
    GET /health         200 only when the model and vectors are loaded

Run (one worker; all state lives in this process):
    APP_ENV=dev  uvicorn search_api:app --host 127.0.0.1 --port 8000 --workers 1
    APP_ENV=prod uvicorn search_api:app --host 127.0.0.1 --port 8000 --workers 1

Settings come from environment variables (all optional):
    APP_ENV             prod (default) or dev
    VEC_DIR             folder with ids.npy, vec_*.npy, manifest.json   (data/vectors)
    DB_PATH             SQLite file with a `links` table: id, url, title (data/kaggle/articles.sqlite)
    VEC_DTYPE           float32 (default) or float16 (half the RAM, see compare_fp16.py)
    TORCH_THREADS       threads per embedding (2)
    MAX_CONCURRENT      embeddings at once, also the number of model copies (2)
    RATE_PER_MIN, RATE_PER_DAY   per visitor (12, 300)

Prod: every request must arrive through cloudflared, which sets CF-Connecting-IP. The rate limit
uses that header. Dev: falls back to the connection IP, and CORS allows http://localhost.
"""

import logging
import os
import queue
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).parent
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
log = logging.getLogger("search_api")


@dataclass
class Settings:
    env: str = "prod"
    vec_dir: Path = ROOT / "data" / "vectors"
    db_path: Path = ROOT / "data" / "kaggle" / "articles.sqlite"
    vec_dtype: str = "float32"
    torch_threads: int = 2
    max_concurrent: int = 2
    retry_after_s: int = 2
    queue_wait_s: float = 0.0         # 0 = answer 503 at once when every model copy is busy
    rate_per_min: int = 12
    rate_per_day: int = 300
    cache_size: int = 2000
    max_query_chars: int = 500
    query_max_tokens: int = 192       # 500 characters plus the Qwen3 query instruction
    topk_retrieve: int = 30
    topk_return: int = 10
    dup_cosine: float = 0.995
    prod_origin: str = "https://vishal.fast"
    fp16_chunk: int = 20_000

    @classmethod
    def from_env(cls):
        e = os.environ.get
        return cls(
            env="dev" if e("APP_ENV", "prod").lower() == "dev" else "prod",
            vec_dir=Path(e("VEC_DIR", ROOT / "data" / "vectors")),
            db_path=Path(e("DB_PATH", ROOT / "data" / "kaggle" / "articles.sqlite")),
            vec_dtype=e("VEC_DTYPE", "float32"),
            torch_threads=int(e("TORCH_THREADS", 2)),
            max_concurrent=int(e("MAX_CONCURRENT", 2)),
            rate_per_min=int(e("RATE_PER_MIN", 12)),
            rate_per_day=int(e("RATE_PER_DAY", 300)),
            queue_wait_s=float(e("QUEUE_WAIT_S", 0)),
        )


def clean_query(q, max_chars):
    """Trim, collapse whitespace, cut to max_chars. This is what gets embedded: case matters to the model
    (lowercasing "AI safety" or "Paul Graham" measurably changes which articles come back)."""
    return " ".join(q.split())[:max_chars].strip()


def normalize_query(q, max_chars):
    """clean_query, lowercased. Used only as the cache key."""
    return clean_query(q, max_chars).lower()


def is_curius_url(url):
    host = urlsplit((url or "").strip()).netloc.lower().split(":")[0].removeprefix("www.")
    return host == "curius.app" or host.endswith(".curius.app")


# ---------------------------------------------------------------------------
# Engine: vectors, article metadata, and a pool of model copies
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, settings):
        self.s = settings
        self.ready = False
        self._load_lock = threading.Lock()

    def load(self):
        """Idempotent. Loads vectors, metadata and model copies, then runs one warm-up query."""
        with self._load_lock:
            if self.ready:
                return
            import json

            import torch
            from sentence_transformers import SentenceTransformer

            s = self.s
            t0 = time.time()
            torch.set_num_threads(s.torch_threads)

            manifest = json.loads((s.vec_dir / "manifest.json").read_text())
            if manifest["model"] != MODEL_ID or manifest["dims"] != 1024:
                raise RuntimeError(f"vectors were made with {manifest['model']}, expected {MODEL_ID}")
            ids = np.load(s.vec_dir / "ids.npy")
            dtype = np.float16 if s.vec_dtype == "float16" else np.float32
            parts = sorted(s.vec_dir.glob("vec_*.npy"))
            self.matrix = np.vstack([np.load(p).astype(dtype, copy=False) for p in parts])
            if self.matrix.shape[0] != len(ids) or len(ids) != manifest["count"]:
                raise RuntimeError("vector count does not match ids.npy / manifest.json")

            con = sqlite3.connect(s.db_path)
            meta = {r[0]: (r[1], r[2]) for r in con.execute("SELECT id, title, url FROM links")}
            con.close()
            missing = [int(i) for i in ids if int(i) not in meta]
            if missing:
                raise RuntimeError(f"{len(missing)} vector ids are not in the database, e.g. {missing[:3]}")
            self._m16 = torch.from_numpy(self.matrix) if dtype == np.float16 else None
            self.ids = ids
            self.titles = [meta[int(i)][0] or "" for i in ids]
            self.urls = [meta[int(i)][1] or "" for i in ids]
            self.title_norm = [" ".join(t.lower().split()) for t in self.titles]
            self.exclude = np.array([is_curius_url(u) for u in self.urls], dtype=bool)

            self.pool = queue.Queue()
            for _ in range(s.max_concurrent):
                try:
                    # transformers 5.x calls this "dtype"; 4.x calls it "torch_dtype"
                    m = SentenceTransformer(MODEL_ID, device="cpu", model_kwargs={"dtype": torch.float32})
                except TypeError:
                    m = SentenceTransformer(MODEL_ID, device="cpu", model_kwargs={"torch_dtype": torch.float32})
                m.float()  # belt and braces: newer libraries otherwise load bfloat16, which is 2x slower on CPU
                if next(m.parameters()).dtype != torch.float32:
                    raise RuntimeError(f"model loaded as {next(m.parameters()).dtype}, expected float32")
                m.max_seq_length = s.query_max_tokens
                self.pool.put(m)
            self.search(self.embed("warm up"))  # the first search is slow (cold memory and BLAS threads)
            self.ready = True
            log.info("engine ready: %s vectors (%s), %d model copies, %.0fs",
                     f"{len(ids):,}", s.vec_dtype, s.max_concurrent, time.time() - t0)

    # -- model pool: taking a copy is the concurrency limit --------------------
    def acquire(self, wait_s=0.0):
        """Take a model copy. With wait_s > 0, queue for up to that long first. None means busy."""
        try:
            return self.pool.get(timeout=wait_s) if wait_s > 0 else self.pool.get_nowait()
        except queue.Empty:
            return None

    def release(self, model):
        self.pool.put(model)

    def embed(self, text, model=None):
        own = model is None
        if own:
            model = self.pool.get()
        try:
            return model.encode([text], prompt_name="query", normalize_embeddings=True,
                                convert_to_numpy=True)[0].astype(np.float32)
        finally:
            if own:
                self.pool.put(model)

    # -- search ---------------------------------------------------------------
    def row_f32(self, row):
        return self.matrix[row].astype(np.float32, copy=False)

    def scores(self, qv):
        m = self.matrix
        if m.dtype == np.float32:
            return m @ qv
        import torch

        q = torch.from_numpy(qv)
        out = np.empty(m.shape[0], dtype=np.float32)
        step = self.s.fp16_chunk
        for lo in range(0, m.shape[0], step):
            out[lo:lo + step] = (self._m16[lo:lo + step].float() @ q).numpy()
        return out

    def search(self, qv):
        s = self.s
        scores = self.scores(qv)
        cand = np.argpartition(-scores, s.topk_retrieve)[:s.topk_retrieve]
        cand = cand[np.argsort(-scores[cand])]
        results, kept_by_title = [], {}
        for row in cand:
            if self.exclude[row]:
                continue
            tn = self.title_norm[row]
            if tn:
                v = self.row_f32(row)
                if any(float(v @ self.row_f32(o)) > s.dup_cosine for o in kept_by_title.get(tn, ())):
                    continue  # same title AND near-identical vector: the same article under another URL
                kept_by_title.setdefault(tn, []).append(row)
            results.append({"id": int(self.ids[row]), "title": self.titles[row],
                            "url": self.urls[row], "score": round(float(scores[row]), 4)})
            if len(results) == s.topk_return:
                break
        return results


# ---------------------------------------------------------------------------
# Rate limit and cache
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per visitor, two sliding windows (minute and day). Returns seconds to wait, or 0 if allowed."""

    def __init__(self, per_min, per_day):
        self.per_min, self.per_day = per_min, per_day
        self.hits = {}
        self.lock = threading.Lock()
        self.calls = 0

    def check(self, key, now=None):
        now = time.time() if now is None else now
        with self.lock:
            self.calls += 1
            if self.calls % 2000 == 0:
                self._prune(now)
            dq = self.hits.setdefault(key, deque())
            while dq and dq[0] < now - 86400:
                dq.popleft()
            if len(dq) >= self.per_day:
                return int(dq[0] + 86400 - now) + 1
            recent = [t for t in reversed(dq) if t > now - 60]
            if len(recent) >= self.per_min:
                return int(recent[-1] + 60 - now) + 1
            dq.append(now)
            return 0

    def _prune(self, now):
        for k in [k for k, dq in self.hits.items() if not dq or dq[-1] < now - 86400]:
            del self.hits[k]


class QueryCache:
    def __init__(self, size):
        self.size, self.data, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)

    def clear(self):
        with self.lock:
            self.data.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(settings=None, engine=None):
    s = settings or Settings.from_env()
    eng = engine or Engine(s)
    limiter = RateLimiter(s.rate_per_min, s.rate_per_day)
    cache = QueryCache(s.cache_size)

    @asynccontextmanager
    async def lifespan(app):
        if not eng.ready:
            eng.load()  # blocks start-up, so the server only accepts requests once it is ready
        yield

    prod = s.env == "prod"
    app = FastAPI(title="Curius search", lifespan=lifespan,
                  docs_url=None if prod else "/docs", redoc_url=None, openapi_url=None if prod else "/openapi.json")
    app.state.engine, app.state.limiter, app.state.cache, app.state.settings = eng, limiter, cache, s

    if prod:
        app.add_middleware(CORSMiddleware, allow_origins=[s.prod_origin], allow_methods=["GET"],
                           allow_headers=[], expose_headers=["Retry-After"], allow_credentials=False)
    else:
        app.add_middleware(CORSMiddleware, allow_origin_regex=r"^http://localhost(:\d+)?$",
                           allow_methods=["GET"], allow_headers=[], expose_headers=["Retry-After"], allow_credentials=False)

    def client_ip(request: Request):
        ip = request.headers.get("cf-connecting-ip")
        if ip:
            return ip.strip()
        if not prod:
            return request.client.host if request.client else "unknown"
        return None

    @app.get("/health")
    def health():
        if not eng.ready:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ok", "articles": int(eng.matrix.shape[0]), "vectors": s.vec_dtype}

    @app.get("/search")
    def search(request: Request, q: str = Query(..., min_length=1, max_length=5000)):
        if not eng.ready:
            return JSONResponse({"detail": "loading"}, status_code=503, headers={"Retry-After": "5"})

        ip = client_ip(request)
        if ip is None:
            return JSONResponse({"detail": "missing CF-Connecting-IP"}, status_code=400)
        wait = limiter.check(ip)
        if wait:
            return JSONResponse({"detail": "too many searches, slow down"}, status_code=429,
                                headers={"Retry-After": str(wait)})

        text = clean_query(q, s.max_query_chars)
        key = text.lower()
        if not key:
            return JSONResponse({"detail": "empty query"}, status_code=400)

        t_start = time.perf_counter()
        hit = cache.get(key)
        if hit is not None:
            log.info("search cache=hit q_len=%d results=%d total_ms=%.1f", len(key), len(hit),
                     (time.perf_counter() - t_start) * 1000)
            return {"query": key, "cached": True, "results": hit}

        model = eng.acquire(s.queue_wait_s)
        if model is None:
            return JSONResponse({"detail": "busy, try again shortly"}, status_code=503,
                                headers={"Retry-After": str(s.retry_after_s)})
        try:
            t0 = time.perf_counter()
            qv = eng.embed(text, model=model)
            t1 = time.perf_counter()
            results = eng.search(qv)
            t2 = time.perf_counter()
        finally:
            eng.release(model)

        cache.put(key, results)
        embed_ms, search_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
        log.info("search cache=miss q_len=%d embed_ms=%.0f search_ms=%.0f results=%d total_ms=%.0f",
                 len(key), embed_ms, search_ms, len(results), (t2 - t_start) * 1000)
        body = {"query": key, "cached": False, "results": results}
        if not prod:
            body["timing_ms"] = {"embed": round(embed_ms), "search": round(search_ms, 1)}
        return body

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("search_api:app", host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), workers=1)
