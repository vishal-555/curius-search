"""
Find everyone who saved a link, by searching the data you have already fetched.

Searches every data/recent_links_*.jsonl cache (from top_links.py) and
data/links.jsonl (from crawler.py --links). No requests are sent to Curius.

Run:
    python who_saved.py "How to be more ambitious"        # part of the title
    python who_saved.py felixrieseberg.com/how-to-be      # or part of the URL
    python who_saved.py "ambitious" --csv                 # also save a CSV

Matching is case-insensitive. If the text matches several different links,
each one is listed separately so you can tell them apart.
"""

import argparse
import csv
from collections import defaultdict

from crawler import OUT_DIR, iter_jsonl
from top_links import normalize_url


def collect_saves():
    """Yield (user_slug, name, link, title, createdDate) from every local source."""
    for cache in sorted(OUT_DIR.glob("recent_links_*.jsonl")):
        for rec in iter_jsonl(cache):
            for l in rec.get("links", []):
                yield rec["userLink"], rec.get("name") or rec["userLink"], l.get("link"), l.get("title"), l.get("createdDate")
    for l in iter_jsonl(OUT_DIR / "links.jsonl"):
        slug = l.get("userLink", str(l.get("userId")))
        yield slug, slug, l.get("link"), l.get("title"), l.get("createdDate")


def main():
    p = argparse.ArgumentParser(description="Who saved a given Curius link")
    p.add_argument("query", help="part of the title or URL")
    p.add_argument("--csv", action="store_true", help="also write data/who_saved.csv")
    args = p.parse_args()
    q = args.query.lower()

    # normalized url -> {"title": ..., "url": ..., "savers": {slug: (name, earliest date)}}
    matches = defaultdict(lambda: {"title": "", "url": "", "savers": {}})
    for slug, name, link, title, created in collect_saves():
        if not link or (q not in link.lower() and q not in (title or "").lower()):
            continue
        m = matches[normalize_url(link)]
        m["title"] = m["title"] or title or ""
        m["url"] = m["url"] or link
        prev = m["savers"].get(slug)
        if not prev or (created or "") < (prev[1] or ""):
            m["savers"][slug] = (name, created)

    if not matches:
        print("No matches in the local data. Run top_links.py with a longer --days window first.")
        return

    rows = []
    for m in sorted(matches.values(), key=lambda m: len(m["savers"]), reverse=True):
        print(f"\n{m['title']}\n{m['url']}\nSaved by {len(m['savers'])} people:")
        for i, (slug, (name, created)) in enumerate(
                sorted(m["savers"].items(), key=lambda kv: kv[1][1] or ""), 1):
            print(f"  {i:>3}. {name:<28} {(created or '')[:10]}  https://curius.app/{slug}")
            rows.append({"title": m["title"], "url": m["url"], "name": name,
                         "profile": f"https://curius.app/{slug}", "saved": (created or "")[:10]})

    if args.csv:
        out = OUT_DIR / "who_saved.csv"
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
