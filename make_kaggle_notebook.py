"""
Builds kaggle_test.ipynb: the 5-test notebook for Qwen3-Embedding-0.6B on a Kaggle T4 GPU.

    python make_kaggle_notebook.py

Cells are plain strings below so they are easy to edit; the notebook is regenerated from them.
SAMPLE_N and QUERIES can be changed at the top of the notebook after upload.
"""

import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append(("markdown", text))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md("""# Qwen3-Embedding-0.6B test on a 5,000-article sample

Setup (once):
1. Kaggle > Datasets > New Dataset > upload `articles.sqlite`, set it to **Private**.
2. New Notebook > right sidebar: **Add Input** (your dataset), **Session options**: Accelerator = **GPU T4 x2** or **GPU P100**, **Internet = On**.
3. Run all cells top to bottom. Paste the output of the last cell back into the chat.

Tests: 1 schema, 2 token lengths, 3 speed, 4 GPU vs CPU match, 5 search quality.""")

code("""
!pip install -q -U sentence-transformers transformers 2>&1 | tail -n 2
""")

code("""
import glob, os, random, sqlite3, time
import numpy as np
import torch

SAMPLE_N = int(os.environ.get("SAMPLE_N", 5000))
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_LEN = 1024
BATCH = 32
TOTAL_ARTICLES = 202_450
HAVE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if HAVE_GPU else "cpu"

import sentence_transformers, transformers
print("torch", torch.__version__, "| sentence-transformers", sentence_transformers.__version__,
      "| transformers", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0) if HAVE_GPU else "NONE - enable a GPU in Session options")

DB = os.environ.get("DB_PATH") or glob.glob("/kaggle/input/**/articles.sqlite", recursive=True)[0]
print("database:", DB)
""")

md("## Test 1: schema")
code("""
con = sqlite3.connect(DB)
for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    cols = [(r[1], r[2]) for r in con.execute(f"PRAGMA table_info({name})")]
    n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    print(f"{name}: {n:,} rows | columns: {cols}")
""")

md("## Build the sample\nText per article = title + newline + body (the preview). No prompt on articles.")
code("""
rows = con.execute("SELECT id, title, snippet, url FROM links").fetchall()
items = []
for lid, title, snippet, url in rows:
    text = ((title or "").strip() + "\\n" + (snippet or "").strip()).strip()
    if text:
        items.append((lid, text, title or "", url or ""))
random.Random(42).shuffle(items)
sample = items[:SAMPLE_N]
ids = np.array([s[0] for s in sample])
texts = [s[1] for s in sample]
print(f"{len(items):,} articles with text, sampled {len(sample):,}")
print("example:", repr(texts[0][:200]))
""")

code("""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_ID, device=DEVICE)
if HAVE_GPU:
    model.half()  # fp16 on the T4
model.max_seq_length = MAX_LEN
print("model dtype:", next(model.parameters()).dtype, "| device:", DEVICE, "| max_seq_length:", model.max_seq_length)
print("prompts available:", list(model.prompts.keys()))
""")

md("## Test 2: token length per article")
code("""
tok = model.tokenizer
lens = np.array([len(x) for x in tok(texts, truncation=False)["input_ids"]])
print(f"median {np.median(lens):.0f} | p90 {np.percentile(lens, 90):.0f} | p99 {np.percentile(lens, 99):.0f} | max {lens.max()}")
over = (lens > MAX_LEN).mean() * 100
print(f"over {MAX_LEN} tokens: {over:.2f}%   |   over 512: {(lens > 512).mean() * 100:.2f}%")
""")

md("## Test 3: speed (GPU, fp16, no prompt on articles)")
code("""
model.encode(texts[:64], batch_size=BATCH, normalize_embeddings=True)  # warm-up
if HAVE_GPU:
    torch.cuda.synchronize()
t0 = time.time()
vecs = model.encode(texts, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=True).astype(np.float32)
if HAVE_GPU:
    torch.cuda.synchronize()
dt = time.time() - t0
rate = len(texts) / dt
hours = TOTAL_ARTICLES / rate / 3600
print(f"{len(texts):,} articles in {dt:.0f}s = {rate:.1f} articles/s | vectors {vecs.shape} | "
      f"estimated for all {TOTAL_ARTICLES:,}: {hours:.1f} hours")
""")

md("## Test 4: GPU (fp16) vs CPU (fp32) vectors\n10 random articles + the 10 longest in the sample.")
code("""
if HAVE_GPU:
    pick = list(range(10)) + list(np.argsort(-lens)[:10])
    sub = [texts[i] for i in pick]
    gpu_vecs = vecs[pick]  # already computed above on the GPU
    cpu_model = SentenceTransformer(MODEL_ID, device="cpu")
    cpu_model.float()  # newer libraries load the file's own precision (bfloat16); force real fp32
    cpu_model.max_seq_length = MAX_LEN
    print("reference model dtype:", next(cpu_model.parameters()).dtype)
    cpu_vecs = cpu_model.encode(sub, batch_size=4, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    cos = (gpu_vecs * cpu_vecs).sum(axis=1)
    print(f"cosine similarity GPU vs CPU: min {cos.min():.5f} | mean {cos.mean():.5f} | max {cos.max():.5f}")
    min_cos = float(cos.min())
    del cpu_model
else:
    min_cos = None
    print("No GPU in this session, so there is nothing to compare. Skipped.")
""")

md("## Test 5: search quality\nEdit `QUERIES` to 10 things you would really search for. Queries use `prompt_name=\"query\"`.")
code("""
QUERIES = [
    "using AI tools to become more ambitious",
    "how semiconductor companies compete",
    "advice for doing good research as a beginner",
    "essays about taste in software design",
    "how to start a startup while in college",
    "why do language models hallucinate",
    "habits of highly productive people",
    "history of the early internet and web",
    "interpretability of neural networks",
    "how to write clearly",
]
titles = {s[0]: s[2] for s in sample}
urls = {s[0]: s[3] for s in sample}
qvecs = model.encode(QUERIES, prompt_name="query", normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
top1 = []
for q, qv in zip(QUERIES, qvecs):
    scores = vecs @ qv
    best = np.argsort(-scores)[:5]
    top1.append(float(scores[best[0]]))
    print(f"\\nQ: {q}")
    for r, i in enumerate(best, 1):
        print(f"  {r}. {scores[i]:.3f}  {titles[ids[i]][:85]}")
""")

md("## Summary: paste this back into the chat")
code("""
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0) if HAVE_GPU else 'none'} | sample {len(texts):,} | max_seq_length {MAX_LEN}")
print(f"tokens: median {np.median(lens):.0f}, p90 {np.percentile(lens, 90):.0f}, over {MAX_LEN}: {over:.2f}%")
print(f"speed: {rate:.1f} articles/s -> {hours:.1f} h for all {TOTAL_ARTICLES:,}")
print(f"GPU vs CPU min cosine: {min_cos if min_cos is None else round(min_cos, 5)}")
print(f"top-1 score per query: {[round(s, 2) for s in top1]}")
print("-" * 60)
print("Pass 1 (full set under 10 h):", "PASS" if hours < 10 else "FAIL")
print("Pass 2 (min cosine above 0.99):", "n/a" if min_cos is None else ("PASS" if min_cos > 0.99 else "FAIL"))
print("Pass 3 (8 of 10 queries relevant in top 5): judge from the results above")
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
    out = Path(__file__).parent / "kaggle_test.ipynb"
    out.write_text(json.dumps(to_notebook(), indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")
