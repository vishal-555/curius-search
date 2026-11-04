"""
Search articles by meaning, using vectors from embed.py.

Run (inside .venv):
    python search.py "essays about taste in software"
    python search.py "..." --model openai
    python search.py "..." --compare        # Qwen and OpenAI side by side
    python search.py                        # interactive: type queries, empty line to quit
"""

import argparse
import sqlite3

import numpy as np

from embed import DB_FILE, MODEL_KEYS, get_encoder
from top_links import normalize_url


class Index:
    def __init__(self, con, model):
        self.model = model
        rows = con.execute("SELECT link_id, vector FROM embeddings WHERE model = ?",
                           (MODEL_KEYS[model],)).fetchall()
        if not rows:
            raise SystemExit(f"No {model} vectors yet. Run: python embed.py --model {model}")
        self.ids = np.array([r[0] for r in rows])
        self.matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        _, self.encode_query = get_encoder(model)

    def search(self, query, top, con):
        scores = self.matrix @ self.encode_query(query).astype(np.float32)
        results, seen = [], set()
        # The same article can exist under several Curius ids (URL variants); show it once.
        for i in np.argsort(-scores)[:top * 4]:
            lid = int(self.ids[i])
            title, url, _ = describe(con, lid)
            keys = {normalize_url(url), title.strip().lower()}
            if keys & seen:
                continue
            seen |= keys
            results.append((lid, float(scores[i])))
            if len(results) == top:
                break
        return results


def describe(con, link_id):
    title, url = con.execute("SELECT title, url FROM links WHERE id = ?", (link_id,)).fetchone()
    people = con.execute("SELECT count(*) FROM saves WHERE link_id = ?", (link_id,)).fetchone()[0]
    return title or "(no title)", url, people


def show(con, label, results):
    print(f"\n{label}")
    for rank, (lid, score) in enumerate(results, 1):
        title, url, people = describe(con, lid)
        print(f"{rank:>3}. {score:.3f}  [{people} saved] {title[:80]}")
        print(f"            {url[:100]}")


def main():
    p = argparse.ArgumentParser(description="Semantic search over saved articles")
    p.add_argument("query", nargs="?", help="what you're looking for, in your own words")
    p.add_argument("--model", choices=list(MODEL_KEYS), default="qwen")
    p.add_argument("--compare", action="store_true", help="show Qwen and OpenAI results side by side")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    con = sqlite3.connect(DB_FILE)
    models = ["qwen", "openai"] if args.compare else [args.model]
    indexes = [Index(con, m) for m in models]

    def run(q):
        results = [ix.search(q, args.top, con) for ix in indexes]
        for ix, res in zip(indexes, results):
            show(con, f"== {ix.model} ({MODEL_KEYS[ix.model]})", res)
        if len(results) == 2:
            shared = len({l for l, _ in results[0]} & {l for l, _ in results[1]})
            print(f"\nBoth models agree on {shared} of the top {args.top}.")

    if args.query:
        run(args.query)
        return
    while True:
        try:
            q = input("\nSearch (empty to quit): ").strip()
        except EOFError:
            break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()
