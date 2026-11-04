"""
Resumable Curius user crawler.

Step 1: Seed the queue from --seeds, SEED, and /api/users/all (if it returns a list).
Step 2: Snowball through each profile's followingUsers.
Step 3: Save each profile to data/profiles.jsonl as it goes; export data/users.csv.
Optional: --links also saves each user's saved links to data/links.jsonl.

Run:
    pip install requests
    python crawler.py --max-users 20      # small test run first
    python crawler.py                     # resume and keep going
    python crawler.py --rate 5            # requests per second across all workers
    python crawler.py --links             # profiles + saved links (slower)
    python crawler.py --seeds a-b c-d     # add extra starting profiles
    python crawler.py --export-csv        # rebuild users.csv and exit

Stop any time with Ctrl+C - progress is saved and the next run resumes.
"""

import argparse
import csv
import json
import random
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests

# ============================================================================
# CONFIG
# ============================================================================

BASE = "https://curius.app/api"
USER_EP = BASE + "/users/{slug}"                  # returns {"user": {...}}
ALL_USERS_EP = BASE + "/users/all"                # shape unknown; any list is used
LINKS_EP = BASE + "/users/{user_id}/links?page={page}"

SEED = "vishal-waghmode"
MAX_LINK_PAGES = 20   # per user, only with --links
COOKIE = ""           # paste the Cookie header value only if you get 401/403

USER_WRAPPER_KEY = "user"
USER_ID_KEY = "id"
SLUG_KEY = "userLink"
FOLLOWING_KEY = "followingUsers"

CSV_FIELDS = [
    "id", "firstName", "lastName", "userLink", "school", "major",
    "interests", "expertise", "github", "twitter", "website",
    "views", "numFollowers", "numFollowing", "createdDate", "lastOnline", "url",
]

DEFAULT_RATE = 8.0    # requests per second, shared by all workers
MAX_RATE = 10.0       # hard cap - Curius is a small site, be polite
DEFAULT_WORKERS = 8   # Curius answers in ~1.3s and slows down past ~8 parallel requests

# ============================================================================

OUT_DIR = Path(__file__).parent / "data"
PROFILES_FILE = OUT_DIR / "profiles.jsonl"
FAILED_FILE = OUT_DIR / "failed.jsonl"
LINKS_FILE = OUT_DIR / "links.jsonl"
CSV_FILE = OUT_DIR / "users.csv"

MAX_RETRIES = 5


class NotFound(Exception):
    pass


class RateLimiter:
    """One shared request budget for all threads; a 429 pauses everyone."""

    def __init__(self, rate):
        self.interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        time.sleep(max(0.0, at - now))

    def pause(self, seconds):
        with self.lock:
            self.next_at = max(self.next_at, time.monotonic() + seconds)


_local = threading.local()


def session():
    """One requests.Session per thread."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (curius-research-crawler)"
        s.headers["Accept"] = "application/json"
        if COOKIE:
            s.headers["Cookie"] = COOKIE
        _local.session = s
    return s


def fetch_json(url, limiter):
    """GET with retry and backoff on 429 / 5xx / network errors."""
    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            r = session().get(url, timeout=20)
        except requests.RequestException as e:
            wait_s = 2 ** attempt + random.random()
            print(f"  network error ({e.__class__.__name__}), retry in {wait_s:.0f}s")
            time.sleep(wait_s)
            continue

        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            raise PermissionError(f"Got {r.status_code} on {url} - set COOKIE from DevTools.")
        if r.status_code == 404:
            raise NotFound(url)
        if r.status_code == 429 or r.status_code >= 500:
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait_s = float(retry_after)
            else:
                wait_s = 60 if r.status_code == 429 else 2 ** attempt * 5
            print(f"  HTTP {r.status_code}, all workers backing off {wait_s:.0f}s")
            limiter.pause(wait_s)
            continue
        raise RuntimeError(f"HTTP {r.status_code} on {url}")
    raise RuntimeError(f"Gave up after {MAX_RETRIES} retries: {url}")


def as_list(data):
    """Return a bare list, or the first list value inside a dict."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # partial last line from a crash


def load_state():
    """Rebuild the seen set and the discovered frontier from previous runs."""
    seen, discovered = set(), []
    for u in iter_jsonl(PROFILES_FILE):
        seen.add(u[SLUG_KEY])
        discovered.extend(f[SLUG_KEY] for f in u.get(FOLLOWING_KEY) or [] if f.get(SLUG_KEY))
    for rec in iter_jsonl(FAILED_FILE):
        if rec.get("slug"):
            seen.add(rec["slug"])
    return seen, discovered


def fetch_all_users_seeds(limiter):
    try:
        items = as_list(fetch_json(ALL_USERS_EP, limiter))
    except (NotFound, RuntimeError) as e:
        print(f"/users/all unavailable ({e}), relying on the follow graph")
        return []
    slugs = [u[SLUG_KEY] for u in items if isinstance(u, dict) and u.get(SLUG_KEY)]
    print(f"/users/all returned {len(items)} items, {len(slugs)} usable slugs")
    return slugs


def fetch_links(user_id, limiter):
    """Collect all link pages in memory so a crash never leaves half a user's links."""
    links = []
    for page in range(MAX_LINK_PAGES):
        items = as_list(fetch_json(LINKS_EP.format(user_id=user_id, page=page), limiter))
        if not items:
            break
        links.extend(items)
    return links


def fetch_user(slug, with_links, limiter):
    """Runs in a worker thread. Returns (slug, user, links, error)."""
    try:
        data = fetch_json(USER_EP.format(slug=slug), limiter)
        user = data.get(USER_WRAPPER_KEY) if isinstance(data, dict) else None
        if not user:
            raise ValueError("no user object in response")
        links = []
        if with_links:
            try:
                links = fetch_links(user[USER_ID_KEY], limiter)
            except (NotFound, RuntimeError) as e:
                print(f"  links failed for {slug}: {e}")
        return slug, user, links, None
    except NotFound:
        return slug, None, [], "404"
    except (RuntimeError, KeyError, ValueError) as e:
        return slug, None, [], str(e)


def crawl(extra_seeds, max_users, rate, workers, with_links):
    OUT_DIR.mkdir(exist_ok=True)
    limiter = RateLimiter(rate)

    seen, discovered = load_state()
    # Always refetch /users/all (one request) so a resumed run still has the full list.
    seeds = [SEED, *extra_seeds, *fetch_all_users_seeds(limiter)]

    queue, queued = deque(), set()
    for s in seeds + discovered:
        if s and s not in seen and s not in queued:
            queue.append(s)
            queued.add(s)

    if not queue:
        print("Nothing to crawl - the reachable graph is fully crawled. Try --seeds.")
        export_csv()
        return

    print(f"{len(seen)} done, {len(queue)} in queue, {workers} workers at {rate:g} req/s")
    crawled = 0
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = set()

    try:
        while queue or pending:
            # Keep a small backlog of submitted jobs so workers never sit idle.
            while queue and len(pending) < workers * 2 and (
                max_users is None or crawled + len(pending) < max_users
            ):
                slug = queue.popleft()
                queued.discard(slug)
                if slug in seen:
                    continue
                seen.add(slug)  # in memory only; the file is the durable record
                pending.add(pool.submit(fetch_user, slug, with_links, limiter))
            if not pending:
                break

            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                slug, user, links, error = fut.result()
                if error:
                    if error != "404":
                        print(f"  skip {slug}: {error}")
                    append_jsonl(FAILED_FILE, {"slug": slug, "reason": error})
                    continue

                # Links first, then the profile: a profile line marks the user as done.
                for item in links:
                    append_jsonl(LINKS_FILE, {"userId": user[USER_ID_KEY], "userLink": slug, **item})
                user.setdefault(SLUG_KEY, slug)
                append_jsonl(PROFILES_FILE, user)
                crawled += 1

                new = 0
                for f in user.get(FOLLOWING_KEY) or []:
                    link = f.get(SLUG_KEY)
                    if link and link not in seen and link not in queued:
                        queue.append(link)
                        queued.add(link)
                        new += 1

                per_min = crawled / max(time.monotonic() - started, 1e-6) * 60
                eta = (len(queue) + len(pending)) / max(per_min, 1e-6)
                extra = f" | {len(links)} links" if with_links else ""
                print(f"{crawled:>5} this run | {len(queue):>5} queued | {per_min:4.0f}/min"
                      f" | ETA {eta:4.0f} min | +{new} | {slug}{extra}")
    except KeyboardInterrupt:
        print("\nStopping - finished users are saved, run again to resume.")
    except PermissionError as e:
        print(e)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    print(f"This run: {crawled} users. Queue left: {len(queue)}")
    export_csv()


def flatten(v):
    if isinstance(v, list):
        return "; ".join(str(x.get("name", x)) if isinstance(x, dict) else str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return v


def export_csv():
    rows, slugs = [], set()
    for u in iter_jsonl(PROFILES_FILE):
        if u.get(SLUG_KEY) in slugs:
            continue  # duplicate from overlapping runs
        slugs.add(u.get(SLUG_KEY))
        u["numFollowing"] = len(u.get(FOLLOWING_KEY) or [])
        u["url"] = f"https://curius.app/{u.get(SLUG_KEY, '')}"
        rows.append({k: flatten(u.get(k, "")) for k in CSV_FIELDS})
    if not rows:
        print("No profiles yet, skipping CSV.")
        return
    rows.sort(key=lambda r: r["numFollowers"] if isinstance(r["numFollowers"], int) else 0,
              reverse=True)
    # utf-8-sig so Excel on Windows shows non-ASCII names correctly
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} users to {CSV_FILE}")


def main():
    p = argparse.ArgumentParser(description="Resumable Curius user crawler")
    p.add_argument("--seeds", nargs="*", default=[], help="extra starting profile slugs")
    p.add_argument("--max-users", type=int, default=None, help="stop after N users this run")
    p.add_argument("--rate", type=float, default=DEFAULT_RATE,
                   help=f"requests per second across all workers (max {MAX_RATE:g})")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel requests")
    p.add_argument("--links", action="store_true", help="also fetch each user's saved links")
    p.add_argument("--export-csv", action="store_true", help="rebuild users.csv and exit")
    args = p.parse_args()

    if args.export_csv:
        export_csv()
        return
    rate = min(max(args.rate, 0.1), MAX_RATE)
    crawl(args.seeds, args.max_users, rate, max(args.workers, 1), args.links)


if __name__ == "__main__":
    main()
