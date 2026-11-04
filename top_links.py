"""
Top articles saved on Curius in the last N days, ranked by how many people saved them.

How it works:
  1. /api/users/all gives every user's id and lastOnline.
  2. Users not online within the window are skipped (they cannot have saved anything
     new). Use --all-users to check everyone anyway.
  3. For each remaining user, read /users/{id}/links pages (newest first, 30 per page)
     until a link is older than the window.
  4. Group by normalized URL, count distinct savers, print and save the top N.

Run:
    python top_links.py                 # top 20 from the last 7 days
    python top_links.py --days 30 --top 50
    python top_links.py --report-only   # re-rank from the cache without fetching

Results are cached per day in data/recent_links_<date>.jsonl, so a stopped run resumes.
"""

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from crawler import (
    ALL_USERS_EP, DEFAULT_RATE, DEFAULT_WORKERS, LINKS_EP, MAX_LINK_PAGES, MAX_RATE,
    OUT_DIR, NotFound, RateLimiter, append_jsonl, as_list, fetch_json, iter_jsonl,
)

TRACKING_PARAMS = {"ref", "source", "fbclid", "gclid", "si", "s", "mc_cid", "mc_eid"}
DEFAULT_EXCLUDE = ["curius.app"]   # Curius's own pages (onboarding etc.) are not articles


def parse_date(s):
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def normalize_url(url):
    """Same article with different tracking params / www / trailing slash -> one key."""
    p = urlsplit(url.strip())
    host = p.netloc.lower().removeprefix("www.").removeprefix("m.")
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS]
    path = p.path.rstrip("/") or "/"
    return urlunsplit(("https", host, path, urlencode(query), ""))


def is_excluded(url, domains):
    """True if the URL's host is one of `domains` or a subdomain of one."""
    host = urlsplit(url.strip()).netloc.lower().split(":")[0].removeprefix("www.")
    return any(host == d or host.endswith("." + d) for d in domains)


def recent_links_for(user, since, limiter):
    """Runs in a worker thread. Returns (user, links saved since `since`, error)."""
    recent = []
    try:
        for page in range(MAX_LINK_PAGES):
            items = as_list(fetch_json(LINKS_EP.format(user_id=user["id"], page=page), limiter))
            if not items:
                break
            for it in items:
                created = parse_date(it.get("createdDate"))
                if created and created >= since:
                    recent.append({
                        "link": it.get("link"),
                        "title": it.get("title"),
                        "createdDate": it.get("createdDate"),
                        "topics": [t.get("topic") for t in it.get("topics") or [] if isinstance(t, dict)],
                    })
            oldest = parse_date(items[-1].get("createdDate"))
            if oldest and oldest < since:
                break  # newest-first, so everything after this is older
        return user, recent, None
    except (NotFound, RuntimeError) as e:
        return user, recent, str(e)


def fetch_window(since, cache_file, rate, workers, all_users):
    limiter = RateLimiter(rate)
    users = [u for u in as_list(fetch_json(ALL_USERS_EP, limiter)) if isinstance(u, dict) and u.get("id")]
    if not all_users:
        users = [u for u in users if (parse_date(u.get("lastOnline")) or since) >= since]

    done = {rec["userId"] for rec in iter_jsonl(cache_file)}
    todo = [u for u in users if u["id"] not in done]
    print(f"{len(users)} users to check, {len(done)} cached, {len(todo)} to fetch "
          f"({workers} workers at {rate:g} req/s)")

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(recent_links_for, u, since, limiter) for u in todo]
    try:
        for i, fut in enumerate(as_completed(futures), 1):
            user, recent, error = fut.result()
            if error:
                print(f"  skip {user['userLink']}: {error}")
                continue  # not cached, so it is retried next run
            append_jsonl(cache_file, {
                "userId": user["id"], "userLink": user["userLink"],
                "name": f"{user.get('firstName', '')} {user.get('lastName', '')}".strip(),
                "links": recent,
            })
            if recent or i % 50 == 0:
                print(f"{i:>5}/{len(todo)} | {user['userLink']}: {len(recent)} recent")
    except KeyboardInterrupt:
        print("\nStopping - fetched users are cached, run again to resume.")
        pool.shutdown(wait=False, cancel_futures=True)
        raise SystemExit(1)
    pool.shutdown()


def rank(cache_file, since, top_n, exclude):
    savers = defaultdict(dict)            # url -> {userId: name}
    titles = defaultdict(lambda: defaultdict(int))
    raw_url, latest = {}, {}
    topics = defaultdict(lambda: defaultdict(int))

    for rec in iter_jsonl(cache_file):
        for l in rec["links"]:
            created = parse_date(l.get("createdDate"))
            if not l.get("link") or not created or created < since:
                continue
            if is_excluded(l["link"], exclude):
                continue
            key = normalize_url(l["link"])
            savers[key][rec["userId"]] = rec["name"] or rec["userLink"]
            if l.get("title"):
                titles[key][l["title"]] += 1
            raw_url.setdefault(key, l["link"])
            latest[key] = max(latest.get(key, created), created)
            for t in l.get("topics") or []:
                topics[key][t] += 1

    ranked = sorted(savers, key=lambda k: (len(savers[k]), latest[k]), reverse=True)[:top_n]
    rows = []
    for i, key in enumerate(ranked, 1):
        title = max(titles[key], key=titles[key].get) if titles[key] else ""
        names = list(savers[key].values())
        rows.append({
            "rank": i,
            "saves": len(names),
            "title": title,
            "url": raw_url[key],
            "latest_save": latest[key].strftime("%Y-%m-%d"),
            "topics": "; ".join(sorted(topics[key], key=topics[key].get, reverse=True)[:3]),
            "saved_by": ", ".join(names[:5]) + (f" +{len(names) - 5} more" if len(names) > 5 else ""),
        })
    return rows, sum(1 for _ in iter_jsonl(cache_file)), len(savers)


def main():
    p = argparse.ArgumentParser(description="Top Curius links saved in the last N days")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--rate", type=float, default=DEFAULT_RATE)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--all-users", action="store_true",
                   help="also check users whose lastOnline is older than the window")
    p.add_argument("--report-only", action="store_true", help="rank from cache, no fetching")
    p.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE, metavar="DOMAIN",
                   help="domains to leave out of the ranking (default: curius.app); "
                        "pass --exclude with no domains to include everything")
    args = p.parse_args()
    exclude = [d.lower().removeprefix("www.") for d in args.exclude]

    OUT_DIR.mkdir(exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc)
    since = today - dt.timedelta(days=args.days)
    stamp = today.strftime("%Y-%m-%d")
    cache_file = OUT_DIR / f"recent_links_{stamp}_{args.days}d.jsonl"

    if not args.report_only:
        fetch_window(since, cache_file, min(max(args.rate, 0.1), MAX_RATE),
                     max(args.workers, 1), args.all_users)

    rows, n_users, n_links = rank(cache_file, since, args.top, exclude)
    if not rows:
        print("No links found in the window.")
        return

    out = OUT_DIR / f"top_links_{stamp}_{args.days}d.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nTop {len(rows)} links saved since {since:%Y-%m-%d} "
          f"({n_users} users checked, {n_links} distinct links"
          + (f", excluding {', '.join(exclude)}" if exclude else "") + ")\n")
    for r in rows:
        print(f"{r['rank']:>2}. [{r['saves']} saves] {r['title'][:90]}")
        print(f"    {r['url']}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
