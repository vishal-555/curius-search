"""
Builds kaggle_full.ipynb: embeds every article with Qwen3-Embedding-0.6B on a Kaggle GPU.

    python make_kaggle_full_notebook.py

Output (in /kaggle/working/vectors):
    ids.npy              int64, article ids in the order the vectors were made
    vec_000.npy ...      float32 (chunk_size x 1024), unit length, one file per 20,000 articles
    manifest.json        model, dims, counts, settings, checks

Run it with "Save Version > Save & Run All (Commit)". It keeps running with the browser closed,
and a re-run skips every chunk that already exists in the output folder.
"""

import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append(("markdown", text))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md("""# Curius articles: full embedding run (Qwen3-Embedding-0.6B)

Before running:
1. Accelerator: **GPU T4 x2** (one is used). Internet: **On**. Persistence: **Files only**.
2. Input: your private `articles.sqlite` dataset.
3. Run with **Save Version > Save & Run All (Commit)** so it runs without the browser open.
   Expect about 1.5 hours (37 articles/s measured on the test sample).

Output: `/kaggle/working/vectors/` (ids.npy, vec_*.npy, manifest.json). Download it from the notebook's Output tab.""")

code("""
!pip install -q -U sentence-transformers transformers 2>&1 | tail -n 2
""")

code("""
import glob, json, os, sqlite3, time
import numpy as np
import torch
import sentence_transformers, transformers

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_LEN = 1024
BATCH = 32
CHUNK = int(os.environ.get("CHUNK", 20_000))
LIMIT = int(os.environ.get("LIMIT", 0)) or None          # LIMIT is only for local smoke tests
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working/vectors")
HAVE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if HAVE_GPU else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

print("torch", torch.__version__, "| sentence-transformers", sentence_transformers.__version__,
      "| transformers", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0) if HAVE_GPU else "NONE - enable a GPU in Session options")
DB = os.environ.get("DB_PATH") or glob.glob("/kaggle/input/**/articles.sqlite", recursive=True)[0]
print("database:", DB, "| output:", OUT_DIR)
""")

md("## Articles\nText = title + newline + preview. No prompt on articles. Sorted by id so chunk boundaries never move between runs.")
code("""
con = sqlite3.connect(DB)
items = []
for lid, title, snippet in con.execute("SELECT id, title, snippet FROM links ORDER BY id"):
    text = ((title or "").strip() + "\\n" + (snippet or "").strip()).strip()
    if text:
        items.append((lid, text))
if LIMIT:
    items = items[:LIMIT]
ids = np.array([i for i, _ in items], dtype=np.int64)
texts = [t for _, t in items]
n_chunks = (len(texts) + CHUNK - 1) // CHUNK
print(f"{len(texts):,} articles in {n_chunks} chunks of up to {CHUNK:,}")

ids_path = os.path.join(OUT_DIR, "ids.npy")
if os.path.exists(ids_path) and not np.array_equal(np.load(ids_path), ids):
    raise SystemExit("ids.npy in the output folder is from a different article set. "
                     "Delete the old vectors or use a fresh output folder.")
np.save(ids_path, ids)
""")

code("""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_ID, device=DEVICE)
model.half() if HAVE_GPU else model.float()
model.max_seq_length = MAX_LEN
print("model dtype:", next(model.parameters()).dtype, "| device:", DEVICE)
""")

md("""## Embed in chunks
Each chunk is written to a temporary file and renamed when complete, so an interruption never leaves a half-written file.
Vectors are converted to float32 and re-normalized. Any row that is not finite or not unit length is re-embedded in fp32.""")
code("""
def normalize(v):
    v = v.astype(np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def encode(batch_texts, batch_size=BATCH):
    return model.encode(batch_texts, batch_size=batch_size, normalize_embeddings=False,
                        convert_to_numpy=True, show_progress_bar=False)


def repair(vecs, chunk_texts):
    raw_norm = np.linalg.norm(vecs.astype(np.float32), axis=1)
    bad = np.where(~np.isfinite(vecs).all(axis=1) | (np.abs(raw_norm - raw_norm.mean()) > 0.5 * raw_norm.mean()))[0]
    if len(bad) == 0 or not HAVE_GPU:
        return vecs, 0
    model.float()
    fixed = encode([chunk_texts[i] for i in bad], batch_size=4)
    model.half()
    vecs = vecs.astype(np.float32)
    vecs[bad] = fixed
    return vecs, len(bad)


t_start = time.time()
done_articles = 0
repaired_total = 0
for k in range(n_chunks):
    path = os.path.join(OUT_DIR, f"vec_{k:03d}.npy")
    lo, hi = k * CHUNK, min((k + 1) * CHUNK, len(texts))
    if os.path.exists(path):
        print(f"chunk {k + 1}/{n_chunks}: already done, skipping")
        continue
    t0 = time.time()
    chunk_texts = texts[lo:hi]
    vecs = encode(chunk_texts)
    vecs, n_bad = repair(vecs, chunk_texts)
    vecs = normalize(vecs)
    repaired_total += n_bad
    tmp = path + ".tmp.npy"
    np.save(tmp, vecs)
    os.replace(tmp, path)
    done_articles += hi - lo
    dt = time.time() - t0
    elapsed = time.time() - t_start
    left = (len(texts) - hi) / max(done_articles / elapsed, 1e-9) / 60
    print(f"chunk {k + 1}/{n_chunks}: {hi - lo:,} articles in {dt:.0f}s ({(hi - lo) / dt:.1f}/s)"
          f" | re-embedded in fp32: {n_bad} | about {left:.0f} min left")
print(f"\\nfinished in {(time.time() - t_start) / 60:.1f} min")
""")

md("## Checks and manifest")
code("""
parts = [np.load(os.path.join(OUT_DIR, f"vec_{k:03d}.npy")) for k in range(n_chunks)]
allv = np.vstack(parts)
norms = np.linalg.norm(allv, axis=1)
checks = {
    "vectors": int(allv.shape[0]),
    "ids": int(len(ids)),
    "dims": int(allv.shape[1]),
    "dtype": str(allv.dtype),
    "all_finite": bool(np.isfinite(allv).all()),
    "norm_min": float(norms.min()),
    "norm_max": float(norms.max()),
    "duplicate_ids": int(len(ids) - len(set(ids.tolist()))),
    "zero_vectors": int((norms < 0.5).sum()),
    "rows_re_embedded_in_fp32_this_run": int(repaired_total),
}
for k, v in checks.items():
    print(f"{k:<36} {v}")
ok = (checks["vectors"] == checks["ids"] and checks["all_finite"] and checks["duplicate_ids"] == 0
      and checks["zero_vectors"] == 0 and abs(checks["norm_min"] - 1) < 1e-3 and abs(checks["norm_max"] - 1) < 1e-3)
manifest = {
    "model": MODEL_ID, "dims": checks["dims"], "count": checks["vectors"], "chunk_size": CHUNK,
    "max_seq_length": MAX_LEN, "text": "title + newline + preview, no prompt", "normalized": True,
    "metric": "cosine (dot product of unit vectors)", "query_prompt_name": "query", "checks": checks,
}
with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("\\nALL CHECKS PASSED" if ok else "\\nCHECKS FAILED - do not use these vectors, send this output back")
""")

md("## Sanity search\nA last look to confirm the vectors mean something. Queries use `prompt_name=\"query\"`.")
code("""
titles = {r[0]: r[1] for r in con.execute("SELECT id, title FROM links")}
for q in ["interpretability of neural networks", "how to start a startup while in college", "habits of highly productive people"]:
    qv = normalize(model.encode([q], prompt_name="query", convert_to_numpy=True))[0]
    top = np.argsort(-(allv @ qv))[:5]
    print(f"\\nQ: {q}")
    for r, i in enumerate(top, 1):
        print(f"  {r}. {float(allv[i] @ qv):.3f}  {(titles.get(int(ids[i])) or '')[:90]}")
""")


def to_notebook():
    cells = []
    for kind, text in CELLS:
        lines = text.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "kaggle_full.ipynb"
    out.write_text(json.dumps(to_notebook(), indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")
