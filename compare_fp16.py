"""
Compare float16 against float32 vectors for search: results, scores, speed, memory.
Also checks that lowercasing a query (the cache key, also used for the embedding) does not change results.

    python compare_fp16.py
"""

import copy
import dataclasses
import logging
import statistics
import time

import numpy as np
import torch

from search_api import Engine, Settings, clean_query, normalize_query

logging.disable(logging.CRITICAL)

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
CASED = [
    "How Does GPT Work",
    "Rust vs Go for backend services",
    "AI safety and alignment research",
    "Lessons from Paul Graham essays",
    "NYC startup scene",
]


def best_ms(fn, repeats=7):
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main():
    e32 = Engine(Settings(env="dev", max_concurrent=1))
    e32.load()
    e16 = copy.copy(e32)
    e16.matrix = e32.matrix.astype(np.float16)
    e16._m16 = torch.from_numpy(e16.matrix)  # the API builds this at load time when VEC_DTYPE=float16
    e16.s = dataclasses.replace(e32.s, vec_dtype="float16")

    print(f"memory: float32 {e32.matrix.nbytes / 1e6:.0f} MB | float16 {e16.matrix.nbytes / 1e6:.0f} MB\n")
    same_order = same_set = 0
    max_diff = 0.0
    t32, t16 = [], []
    for q in QUERIES:
        qv = e32.embed(clean_query(q, 500))
        r32, r16 = e32.search(qv), e16.search(qv)
        ids32, ids16 = [r["id"] for r in r32], [r["id"] for r in r16]
        diff = max(abs(a["score"] - b["score"]) for a, b in zip(r32, r16))
        max_diff = max(max_diff, diff)
        same_order += ids32 == ids16
        same_set += set(ids32) == set(ids16)
        t32.append(best_ms(lambda: e32.search(qv)))
        t16.append(best_ms(lambda: e16.search(qv)))
        flag = "identical" if ids32 == ids16 else ("same set, order differs" if set(ids32) == set(ids16)
                                                   else f"{len(set(ids32) & set(ids16))}/10 shared")
        print(f"  {flag:<24} max score diff {diff:.4f} | {q}")
        if ids32 != ids16:
            for a, b in zip(r32, r16):
                if a["id"] != b["id"]:
                    print(f"      fp32 {a['score']:.4f} {a['title'][:45]!r}  vs  fp16 {b['score']:.4f} {b['title'][:45]!r}")
    print(f"\nidentical top 10 (same ids, same order): {same_order}/10 | same set: {same_set}/10 | "
          f"largest score difference {max_diff:.4f}")
    print(f"search time (median): float32 {statistics.mean(t32):.0f} ms | float16 {statistics.mean(t16):.0f} ms")

    print("\nlowercasing check (original case vs lowercase query, top 10 ids):")
    overlaps = []
    for q in CASED + [q for q in QUERIES if q != q.lower()]:
        a = [r["id"] for r in e32.search(e32.embed(q))]
        b = [r["id"] for r in e32.search(e32.embed(normalize_query(q, 500)))]
        overlaps.append(len(set(a) & set(b)))
        print(f"  {len(set(a) & set(b)):>2}/10 shared, same order: {a == b!s:<5} | {q}")
    print(f"average shared: {statistics.mean(overlaps):.1f}/10")


if __name__ == "__main__":
    main()
