"""
Verify the vectors downloaded from Kaggle (data/vectors) before using them.

  1. files: shapes, dtype, finite, unit length, no zero or duplicate rows
  2. ids: identical to the article ids in the database, in the same order
  3. cross-check: re-embed N random articles here on the CPU in fp32 and compare them with
     the stored GPU (fp16) vectors
  4. search: CPU-embedded queries against the stored vectors, duplicates combined by URL

Run (inside .venv):
    python verify_vectors.py
    python verify_vectors.py --check 40 --queries "my own query" "another one"
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from top_links import normalize_url

ROOT = Path(__file__).parent
VEC_DIR = ROOT / "data" / "vectors"
DB_FILE = ROOT / "data" / "curius.db"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

DEFAULT_QUERIES = [
    "using AI tools to become more ambitious",
    "advice for doing good research as a beginner",
    "how semiconductor companies compete",
    "how to start a startup while in college",
]


def article_text(title, snippet):
    return ((title or "").strip() + "\n" + (snippet or "").strip()).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", type=int, default=30, help="articles to re-embed on the CPU")
    p.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()
    ok = True

    def report(label, passed, detail=""):
        nonlocal ok
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))

    print("1. files")
    manifest = json.loads((VEC_DIR / "manifest.json").read_text())
    chunk_files = sorted(VEC_DIR.glob("vec_*.npy"))
    parts = [np.load(f) for f in chunk_files]
    vecs = np.vstack(parts)
    ids = np.load(VEC_DIR / "ids.npy")
    norms = np.linalg.norm(vecs, axis=1)
    report("chunk count", len(chunk_files) == -(-manifest["count"] // manifest["chunk_size"]),
           f"{len(chunk_files)} files")
    report("shape and dtype", vecs.shape == (manifest["count"], manifest["dims"]) and vecs.dtype == np.float32,
           f"{vecs.shape} {vecs.dtype}")
    report("all values finite", bool(np.isfinite(vecs).all()))
    report("unit length", abs(norms.min() - 1) < 1e-3 and abs(norms.max() - 1) < 1e-3,
           f"{norms.min():.6f} to {norms.max():.6f}")
    report("no all-zero rows", int((np.abs(vecs).sum(axis=1) == 0).sum()) == 0)
    uniq = len(np.unique(vecs[:, :16], axis=0))
    report("no mass duplicate vectors", uniq > 0.97 * len(vecs), f"{uniq:,} distinct of {len(vecs):,} (by first 16 values)")

    print("2. ids against the database")
    con = sqlite3.connect(DB_FILE)
    rows = con.execute("SELECT id, title, snippet, url FROM links ORDER BY id").fetchall()
    expected = [r[0] for r in rows if article_text(r[1], r[2])]
    report("same count", len(expected) == len(ids), f"database {len(expected):,} | vectors {len(ids):,}")
    report("same ids, same order", np.array_equal(np.array(expected, dtype=np.int64), ids))
    report("ids unique", len(np.unique(ids)) == len(ids))
    by_id = {r[0]: r for r in rows}
    row_of = {int(i): k for k, i in enumerate(ids)}

    print(f"3. cross-check: {args.check} articles re-embedded on this CPU (fp32) vs stored GPU vectors")
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID, device="cpu")
    model.float()
    model.max_seq_length = manifest["max_seq_length"]
    rng = np.random.default_rng(7)
    pick = rng.choice(len(ids), size=args.check, replace=False)
    texts = [article_text(by_id[int(ids[k])][1], by_id[int(ids[k])][2]) for k in pick]
    t0 = time.time()
    cpu = model.encode(texts, batch_size=4, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    cos = (cpu * vecs[pick]).sum(axis=1)
    print(f"     {args.check} articles in {time.time() - t0:.0f}s | cosine min {cos.min():.5f} "
          f"mean {cos.mean():.5f} max {cos.max():.5f}")
    report("GPU vectors match CPU vectors (min cosine above 0.99)", cos.min() > 0.99)

    print("4. search with CPU-embedded queries (duplicate URLs combined)")
    times = []
    for q in args.queries:
        t0 = time.perf_counter()
        qv = model.encode([q], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True)[0]
        t1 = time.perf_counter()
        scores = vecs @ qv
        cand = np.argpartition(-scores, 40)[:40]
        cand = cand[np.argsort(-scores[cand])]
        t2 = time.perf_counter()
        times.append((t1 - t0, t2 - t1))
        seen, shown = set(), []
        for k in cand:
            _, title, _, url = by_id[int(ids[k])]
            key = normalize_url(url)
            if key in seen:
                continue
            seen.add(key)
            shown.append((float(scores[k]), title or "(no title)"))
            if len(shown) == args.top:
                break
        print(f"\n  Q: {q}")
        for r, (s, title) in enumerate(shown, 1):
            print(f"    {r}. {s:.3f}  {title[:90]}")
    q_times, s_times = zip(*times[1:] or times)
    print(f"\n  query embedding {np.mean(q_times):.2f}s | search over {len(vecs):,} vectors {np.mean(s_times) * 1000:.0f} ms")
    print("\nOVERALL:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
