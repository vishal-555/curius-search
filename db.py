"""
Curius -> SQLite loader. Local first; the same SQL runs on Cloudflare D1.

Commands:
    python db.py init                   # create data/curius.db from schema.sql
    python db.py full                   # every user, every page of saves (resumable)
    python db.py daily                  # only users online since their last sync,
                                        # reading newest saves until one is already stored
    python db.py refresh --days 7       # re-read all saves of users active in the last N days
                                        # (catches deletions and topics added to old saves)
    python db.py refresh --users a-b c-d # re-read just these users ('loaded' = all loaded users)
    python db.py stats                  # row counts and sync progress
    python db.py export-d1              # D1 import files, split by the daily write limit

Flags for full / daily / refresh: --max-users N  --rate 4  --workers 4
Stop any time with Ctrl+C. Each user is saved in one transaction, so nothing is half-written.
"""

import argparse
import datetime as dt
import json
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from crawler import (
    ALL_USERS_EP, DEFAULT_RATE, DEFAULT_WORKERS, LINKS_EP, MAX_RATE, OUT_DIR, USER_EP,
    NotFound, RateLimiter, as_list, fetch_json,
)

DB_FILE = OUT_DIR / "curius.db"
SCHEMA_FILE = Path(__file__).parent / "schema.sql"
EXPORT_DIR = Path(__file__).parent / "d1_export"
MAX_PAGES = 2000  # safety stop: 60,000 saves for one user
SNIPPET_MAX = 1000  # Curius itself stops ~99% of previews here; cut the rest to match

# D1 counts each index entry as an extra row written. Cost per row, per table:
D1_WRITE_COST = {
    "users": 1, "links": 1, "topics": 1, "follows": 1,
    "saves": 2,  # table + saves_by_time index
    "save_topics": 1, "highlights": 1,
}
D1_DAILY_BUDGET = 90_000  # free plan allows 100,000 rows written per day; keep a margin


def now_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def connect():
    OUT_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_FILE)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    return con


def text_or_json(v):
    if v is None or isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def as_int(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    return v if isinstance(v, int) else None


# ---------------------------------------------------------------------------
# Fetching (worker threads)
# ---------------------------------------------------------------------------

def fetch_user_data(user, known, limiter):
    """Fetch one user's profile and saves.

    known: link ids already stored for this user. Reading stops at the first one
           (Curius lists saves newest first). None means read every page.
    Returns (user, profile, items, read_all, error).
    """
    try:
        data = fetch_json(USER_EP.format(slug=user["slug"]), limiter)
        profile = data.get("user") if isinstance(data, dict) else None
        items, read_all = [], True
        for page in range(MAX_PAGES):
            batch = as_list(fetch_json(LINKS_EP.format(user_id=user["id"], page=page), limiter))
            if not batch:
                break
            for it in batch:
                if known is not None and it.get("id") in known:
                    return user, profile, items, False, None
                items.append(it)
        else:
            read_all = False  # hit MAX_PAGES; don't treat missing saves as deleted
        return user, profile, items, read_all and known is None, None
    except NotFound:
        return user, None, [], False, "404"
    except (RuntimeError, ValueError, KeyError) as e:
        return user, None, [], False, str(e)


# ---------------------------------------------------------------------------
# Writing (main thread only)
# ---------------------------------------------------------------------------

UPSERT_BASIC_USER = """
INSERT INTO users (id, slug, first_name, last_name, last_online) VALUES (?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET slug = excluded.slug, first_name = excluded.first_name,
    last_name = excluded.last_name, last_online = excluded.last_online
"""

UPDATE_PROFILE = """
UPDATE users SET school = ?, major = ?, interests = ?, expertise = ?, github = ?, twitter = ?,
    website = ?, views = ?, num_followers = ?, created_at = ?, last_online = coalesce(?, last_online),
    profile_synced_at = ?
WHERE id = ?
"""

UPSERT_LINK = """
INSERT INTO links (id, url, title, snippet, first_saved_by) VALUES (?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET url = excluded.url,
    title = coalesce(excluded.title, links.title),
    snippet = coalesce(excluded.snippet, links.snippet),
    first_saved_by = coalesce(links.first_saved_by, excluded.first_saved_by)
"""

UPSERT_SAVE = """
INSERT INTO saves (user_id, link_id, saved_at, modified_at, favorite, to_read) VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(user_id, link_id) DO UPDATE SET saved_at = excluded.saved_at,
    modified_at = excluded.modified_at, favorite = excluded.favorite, to_read = excluded.to_read
"""

UPSERT_TOPIC = """
INSERT INTO topics (id, user_id, topic, slug, public) VALUES (?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET topic = excluded.topic, slug = excluded.slug, public = excluded.public
"""


def write_user(con, user, profile, items, read_all, synced_at):
    """Store one user's fetch in a single transaction. Returns number of saves removed."""
    uid = user["id"]
    removed = 0
    with con:
        if profile:
            con.execute(UPDATE_PROFILE, (
                profile.get("school"), profile.get("major"),
                text_or_json(profile.get("interests")), text_or_json(profile.get("expertise")),
                profile.get("github"), profile.get("twitter"), profile.get("website"),
                as_int(profile.get("views")), as_int(profile.get("numFollowers")),
                profile.get("createdDate"), profile.get("lastOnline"), synced_at, uid,
            ))
            con.execute("DELETE FROM follows WHERE follower_id = ?", (uid,))
            con.executemany("INSERT OR IGNORE INTO follows VALUES (?, ?)", [
                (uid, f["id"]) for f in profile.get("followingUsers") or [] if f.get("id")
            ])

        for it in items:
            lid = it.get("id")
            if lid is None or not it.get("link"):
                continue
            snippet = (it.get("snippet") or "")[:SNIPPET_MAX] or None
            con.execute(UPSERT_LINK, (lid, it["link"], it.get("title"), snippet, it.get("createdBy")))
            con.execute(UPSERT_SAVE, (
                uid, lid, it.get("createdDate") or synced_at, it.get("modifiedDate"),
                int(bool(it.get("favorite"))), as_int(it.get("toRead")),
            ))
            con.execute("DELETE FROM save_topics WHERE user_id = ? AND link_id = ?", (uid, lid))
            for t in it.get("topics") or []:
                if isinstance(t, dict) and t.get("id") is not None and t.get("topic"):
                    con.execute(UPSERT_TOPIC, (t["id"], t.get("userId", uid), t["topic"],
                                               t.get("slug"), as_int(t.get("public"))))
                    con.execute("INSERT OR IGNORE INTO save_topics VALUES (?, ?, ?)", (uid, lid, t["id"]))
            con.execute("DELETE FROM highlights WHERE user_id = ? AND link_id = ?", (uid, lid))
            for h in it.get("highlights") or []:
                if isinstance(h, dict) and h.get("id") is not None and h.get("highlight"):
                    con.execute("INSERT OR REPLACE INTO highlights VALUES (?, ?, ?, ?, ?, ?)", (
                        uid, lid, h["id"], h["highlight"], text_or_json(h.get("comment")),
                        h.get("createdDate"),
                    ))

        if read_all:
            # Everything Curius lists was just read, so anything else was deleted there.
            con.execute("CREATE TEMP TABLE IF NOT EXISTS keep_ids (link_id INTEGER PRIMARY KEY)")
            con.execute("DELETE FROM keep_ids")
            con.executemany("INSERT OR IGNORE INTO keep_ids VALUES (?)",
                            [(it["id"],) for it in items if it.get("id") is not None])
            removed = con.execute(
                "DELETE FROM saves WHERE user_id = ? AND link_id NOT IN (SELECT link_id FROM keep_ids)",
                (uid,)).rowcount
            for table in ("save_topics", "highlights"):
                con.execute(f"DELETE FROM {table} WHERE user_id = ? "
                            "AND link_id NOT IN (SELECT link_id FROM keep_ids)", (uid,))
            con.execute("UPDATE users SET full_synced_at = ? WHERE id = ?", (synced_at, uid))

        con.execute("UPDATE users SET links_synced_at = ?, sync_error = NULL WHERE id = ?",
                    (synced_at, uid))
    return removed


def mark_error(con, user, error, synced_at):
    with con:
        if error == "404":
            # Profile is gone; don't retry until the user shows up online again.
            con.execute("UPDATE users SET sync_error = '404', links_synced_at = ?, "
                        "full_synced_at = coalesce(full_synced_at, ?) WHERE id = ?",
                        (synced_at, synced_at, user["id"]))
        else:
            con.execute("UPDATE users SET sync_error = ? WHERE id = ?", (error[:500], user["id"]))


# ---------------------------------------------------------------------------
# Sync runs
# ---------------------------------------------------------------------------

def candidates(con, mode, days):
    """Users to process, most recently online first. Each row: id, slug, read_everything."""
    if mode == "full":
        rows = con.execute("SELECT id, slug FROM users WHERE full_synced_at IS NULL "
                           "ORDER BY last_online DESC").fetchall()
        return [(i, s, True) for i, s in rows]
    if mode == "refresh":
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        rows = con.execute("SELECT id, slug FROM users WHERE last_online >= ? "
                           "ORDER BY last_online DESC", (cutoff,)).fetchall()
        return [(i, s, True) for i, s in rows]
    # daily: anyone online since we last checked them, then anyone never fully loaded
    rows = con.execute("""
        SELECT id, slug, full_synced_at IS NULL FROM users
        WHERE full_synced_at IS NULL
           OR (last_online IS NOT NULL AND last_online > coalesce(links_synced_at, ''))
        ORDER BY full_synced_at IS NULL, last_online DESC""").fetchall()
    return [(i, s, bool(never)) for i, s, never in rows]


def sync(mode, max_users, rate, workers, days, slugs=None):
    con = connect()
    limiter = RateLimiter(rate)
    started = now_iso()
    run_id = con.execute("INSERT INTO runs (kind, started_at, status) VALUES (?, ?, 'running')",
                         (mode, started)).lastrowid
    con.commit()

    all_users = [u for u in as_list(fetch_json(ALL_USERS_EP, limiter))
                 if isinstance(u, dict) and u.get("id") and u.get("userLink")]
    with con:
        con.executemany(UPSERT_BASIC_USER, [
            (u["id"], u["userLink"], u.get("firstName"), u.get("lastName"), u.get("lastOnline"))
            for u in all_users
        ])

    if slugs is not None:
        # Named users (or "loaded" = everyone already fully loaded): re-read every page.
        if slugs == ["loaded"]:
            rows = con.execute("SELECT id, slug FROM users WHERE full_synced_at IS NOT NULL").fetchall()
        else:
            marks = ",".join("?" * len(slugs))
            rows = con.execute(f"SELECT id, slug FROM users WHERE slug IN ({marks})", slugs).fetchall()
        todo = [(i, s, True) for i, s in rows]
    else:
        todo = candidates(con, mode, days)
    if max_users is not None:
        todo = todo[:max_users]
    print(f"{len(all_users)} users on Curius, {len(todo)} to process ({mode}, "
          f"{workers} workers at {rate:g} req/s)")

    done = saves_seen = removed_total = 0
    t0 = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=workers)
    pending, queue = set(), list(reversed(todo))
    status = "finished"

    try:
        while queue or pending:
            while queue and len(pending) < workers * 2:
                uid, slug, read_everything = queue.pop()
                known = None
                if not read_everything:
                    known = {r[0] for r in con.execute(
                        "SELECT link_id FROM saves WHERE user_id = ?", (uid,))}
                pending.add(pool.submit(fetch_user_data, {"id": uid, "slug": slug}, known, limiter))

            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                user, profile, items, read_all, error = fut.result()
                if error:
                    mark_error(con, user, error, started)
                    if error != "404":
                        print(f"  skip {user['slug']}: {error}")
                    continue
                # A user's sync time is the run's start, so saves made during the run
                # are picked up next time.
                removed = write_user(con, user, profile, items, read_all, started)
                done += 1
                saves_seen += len(items)
                removed_total += removed
                per_min = done / max(time.monotonic() - t0, 1e-6) * 60
                eta = (len(queue) + len(pending)) / max(per_min, 1e-6)
                gone = f" -{removed} deleted" if removed else ""
                print(f"{done:>5}/{len(todo)} | {per_min:4.0f} users/min | ETA {eta:5.0f} min"
                      f" | {user['slug']}: {len(items)} saves{gone}")
    except KeyboardInterrupt:
        status = "stopped"
        print("\nStopping - every finished user is saved. Run the same command to continue.")
    except PermissionError as e:
        status = "failed"
        print(e)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        with con:
            con.execute("UPDATE runs SET finished_at = ?, users_done = ?, saves_seen = ?, status = ? "
                        "WHERE id = ?", (now_iso(), done, saves_seen, status, run_id))

    print(f"Done: {done} users, {saves_seen} saves read, {removed_total} deleted saves removed.")
    con.close()


# ---------------------------------------------------------------------------
# Stats and D1 export
# ---------------------------------------------------------------------------

def stats():
    con = connect()
    one = lambda sql: con.execute(sql).fetchone()[0]
    print(f"Database: {DB_FILE} ({DB_FILE.stat().st_size / 1e6:.1f} MB)\n")
    print(f"Users            {one('SELECT count(*) FROM users'):>10,}")
    print(f"  fully loaded   {one('SELECT count(*) FROM users WHERE full_synced_at IS NOT NULL'):>10,}")
    print(f"  not loaded yet {one('SELECT count(*) FROM users WHERE full_synced_at IS NULL'):>10,}")
    print(f"  with errors    {one('SELECT count(*) FROM users WHERE sync_error IS NOT NULL'):>10,}")
    for table in ("links", "saves", "highlights", "topics", "save_topics", "follows"):
        print(f"{table:<16} {one(f'SELECT count(*) FROM {table}'):>10,}")
    print("\nRecent runs:")
    for r in con.execute("SELECT kind, started_at, status, users_done, saves_seen FROM runs "
                         "ORDER BY id DESC LIMIT 5"):
        print(f"  {r[1][:16]}  {r[0]:<8} {r[2]:<9} {r[3]:>6} users  {r[4]:>8} saves")
    con.close()


def sql_literal(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("\x00", "").replace("'", "''") + "'"


def export_d1(budget):
    """Write INSERT files sized so each one stays under D1's free daily write limit.

    Uses INSERT OR REPLACE, so re-running a file (after a failed upload) is safe.
    """
    con = connect()
    EXPORT_DIR.mkdir(exist_ok=True)
    for old in EXPORT_DIR.glob("*.sql"):
        old.unlink()
    (EXPORT_DIR / "00_schema.sql").write_text(SCHEMA_FILE.read_text(encoding="utf-8"), encoding="utf-8")

    max_rows_per_stmt, max_stmt_bytes = 50, 90_000  # D1 statement limit is 100 KB
    part, part_cost, part_lines, total_cost, files = 1, 0, [], 0, []

    def flush():
        nonlocal part, part_cost, part_lines
        if part_lines:
            path = EXPORT_DIR / f"part_{part:03d}.sql"
            path.write_text("\n".join(part_lines) + "\n", encoding="utf-8")
            files.append((path.name, part_cost))
            part, part_cost, part_lines = part + 1, 0, []

    for table, cost in D1_WRITE_COST.items():
        cur = con.execute(f"SELECT * FROM {table}")
        cols = ", ".join(d[0] for d in cur.description)
        head = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES\n"
        values, size = [], len(head)
        for row in cur:
            v = "(" + ", ".join(sql_literal(x) for x in row) + ")"
            vb = len(v.encode("utf-8")) + 2
            if values and (len(values) >= max_rows_per_stmt or size + vb > max_stmt_bytes
                           or part_cost + (len(values) + 1) * cost > budget):
                part_lines.append(head + ",\n".join(values) + ";")
                part_cost += len(values) * cost
                total_cost += len(values) * cost
                values, size = [], len(head)
                if part_cost + cost > budget:
                    flush()
            values.append(v)
            size += vb
        if values:
            part_lines.append(head + ",\n".join(values) + ";")
            part_cost += len(values) * cost
            total_cost += len(values) * cost
            if part_cost >= budget:
                flush()
    flush()
    con.close()

    print(f"Wrote {len(files)} data files + 00_schema.sql to {EXPORT_DIR}")
    print(f"Total D1 rows written: {total_cost:,} (indexes included)")
    print(f"At {budget:,} per day on the free plan: {len(files)} day(s).\n")
    for name, c in files:
        print(f"  {name}  {c:>7,} rows written")
    print("\nUpload, one data file per day:")
    print("  npx wrangler d1 execute curius --remote --file=d1_export/00_schema.sql")
    print("  npx wrangler d1 execute curius --remote --file=d1_export/part_001.sql")


def main():
    p = argparse.ArgumentParser(description="Curius -> SQLite loader")
    p.add_argument("command", choices=["init", "full", "daily", "refresh", "stats", "export-d1"])
    p.add_argument("--max-users", type=int, default=None, help="process at most N users this run")
    p.add_argument("--rate", type=float, default=DEFAULT_RATE, help=f"requests/sec (max {MAX_RATE:g})")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--days", type=int, default=7, help="refresh: users online in the last N days")
    p.add_argument("--users", nargs="+", default=None, metavar="SLUG",
                   help="refresh only these users (or 'loaded' for everyone already loaded)")
    p.add_argument("--budget", type=int, default=D1_DAILY_BUDGET,
                   help="export-d1: max D1 rows written per file")
    args = p.parse_args()

    if args.command == "init":
        connect().close()
        print(f"Created {DB_FILE}")
    elif args.command == "stats":
        stats()
    elif args.command == "export-d1":
        export_d1(args.budget)
    else:
        sync(args.command, args.max_users, min(max(args.rate, 0.1), MAX_RATE),
             max(args.workers, 1), args.days, args.users)


if __name__ == "__main__":
    main()
