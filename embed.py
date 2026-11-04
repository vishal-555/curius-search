"""
Embed every article in data/curius.db for meaning-based search.

Text per article: title + preview + highlights from everyone who saved it,
cut to EMBED_CHARS. Only new articles, or ones whose text changed, are embedded.

Models:
    qwen    Qwen/Qwen3-Embedding-0.6B, runs on your CPU, free (default)
    openai  text-embedding-3-small via the API; needs OPENAI_API_KEY in .env

Run (inside .venv):
    python embed.py                    # Qwen
    python embed.py --limit 200        # quick speed test
    python embed.py --model openai     # prints the estimated cost first; stops if over --max-cost

Vectors go in the local-only table `embeddings` (not part of schema.sql, never sent to D1).
"""

import argparse
import hashlib
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

DB_FILE = Path(__file__).parent / "data" / "curius.db"
EMBED_CHARS = 2000

QWEN_ID = "Qwen/Qwen3-Embedding-0.6B"
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_PRICE_PER_M = 0.02  # USD per 1M tokens at last check - verify on openai.com/api/pricing
MODEL_KEYS = {"qwen": QWEN_ID, "openai": OPENAI_MODEL}

EMBED_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    link_id     INTEGER NOT NULL,
    model       TEXT NOT NULL,
    dims        INTEGER NOT NULL,
    text_hash   TEXT NOT NULL,
    vector      BLOB NOT NULL,   -- float32, normalized to length 1
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (link_id, model)
) WITHOUT ROWID;
"""


def load_env():
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def connect():
    con = sqlite3.connect(DB_FILE)
    con.executescript(EMBED_SCHEMA)
    return con


def article_texts(con):
    """Yield (link_id, text) for every article that has any text."""
    highlights = defaultdict(list)
    for lid, text in con.execute("SELECT link_id, text FROM highlights ORDER BY link_id, created_at"):
        highlights[lid].append(text.strip())
    for lid, title, snippet in con.execute("SELECT id, title, snippet FROM links"):
        parts = [(title or "").strip(), (snippet or "").strip()]
        hl = " | ".join(dict.fromkeys(h for h in highlights.get(lid, []) if h))
        if hl:
            parts.append("Highlights: " + hl)
        text = "\n\n".join(p for p in parts if p)[:EMBED_CHARS]
        if text:
            yield lid, text


# ---------------------------------------------------------------------------
# Encoders. Each returns (encode_documents, encode_query).
# ---------------------------------------------------------------------------

def qwen_encoder():
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(os.cpu_count() or 4)
    model = SentenceTransformer(QWEN_ID, device="cpu")
    model.max_seq_length = 512  # ~2,000 chars; keeps CPU time bounded

    def docs(texts):
        return model.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)

    def query(text):
        # Qwen3 expects an instruction in front of queries (not documents); the model ships it as "query".
        return model.encode([text], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True)[0]

    return docs, query


class OpenAIEncoder:
    def __init__(self):
        load_env()
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise SystemExit("OPENAI_API_KEY not found. Add it to .env in the project folder.")
        self.tokens = 0

    def _call(self, texts):
        for attempt in range(5):
            r = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": OPENAI_MODEL, "input": texts},
                timeout=120,
            )
            if r.status_code == 200:
                body = r.json()
                self.tokens += body.get("usage", {}).get("total_tokens", 0)
                data = sorted(body["data"], key=lambda d: d["index"])
                v = np.array([d["embedding"] for d in data], dtype=np.float32)
                return v / np.linalg.norm(v, axis=1, keepdims=True)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt * 2)
                continue
            raise SystemExit(f"OpenAI error {r.status_code}: {r.text[:300]}")
        raise SystemExit("OpenAI kept failing; try again later.")

    def docs(self, texts):
        return self._call(texts)

    def query(self, text):
        return self._call([text])[0]


def get_encoder(name):
    if name == "qwen":
        return qwen_encoder()
    enc = OpenAIEncoder()
    return enc.docs, enc.query


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Embed articles for semantic search")
    p.add_argument("--model", choices=list(MODEL_KEYS), default="qwen")
    p.add_argument("--limit", type=int, default=None, help="embed at most N articles")
    p.add_argument("--max-cost", type=float, default=1.00, help="OpenAI: stop if estimate exceeds this (USD)")
    args = p.parse_args()

    con = connect()
    model_key = MODEL_KEYS[args.model]
    done = dict(con.execute("SELECT link_id, text_hash FROM embeddings WHERE model = ?", (model_key,)))
    todo = []
    for lid, text in article_texts(con):
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if done.get(lid) != h:
            todo.append((lid, text, h))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done):,} already embedded with {model_key}, {len(todo):,} to embed")
    if not todo:
        return

    if args.model == "openai":
        est_tokens = sum(len(t) for _, t, _ in todo) / 4
        est_cost = est_tokens / 1e6 * OPENAI_PRICE_PER_M
        print(f"Estimated {est_tokens / 1e6:.2f}M tokens = ${est_cost:.3f}")
        if est_cost > args.max_cost:
            raise SystemExit(f"Over --max-cost ${args.max_cost:.2f}; raise it to continue.")
        enc = OpenAIEncoder()
        encode_docs = enc.docs
    else:
        print(f"Loading {QWEN_ID} (first run downloads ~1.2 GB)...")
        encode_docs, _ = qwen_encoder()

    chunk = 64 if args.model == "qwen" else 256
    t0, n = time.monotonic(), 0
    try:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            vecs = encode_docs([t for _, t, _ in part]).astype(np.float32)
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with con:
                con.executemany(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
                    [(lid, model_key, vecs.shape[1], h, v.tobytes(), now)
                     for (lid, _, h), v in zip(part, vecs)],
                )
            n += len(part)
            rate = n / (time.monotonic() - t0)
            eta = (len(todo) - n) / rate / 60
            print(f"{n:>7,}/{len(todo):,} | {rate:5.1f} articles/s | ETA {eta:5.1f} min")
    except KeyboardInterrupt:
        print("\nStopped - finished chunks are saved; run again to continue.")

    if args.model == "openai":
        print(f"OpenAI tokens used: {enc.tokens:,} = ${enc.tokens / 1e6 * OPENAI_PRICE_PER_M:.4f}")


if __name__ == "__main__":
    main()
