-- Curius database. Plain SQLite, also valid for Cloudflare D1.
--
-- Keys come from Curius itself:
--   users.id      Curius user id
--   links.id      Curius link id - one per article, shared by everyone who saved it
--   saves         one row per (person, article); saved_at is that person's own save time
--
-- D1 bills every index entry as an extra "row written", so tables that are only
-- looked up by their key are WITHOUT ROWID (the key is the table, no second index).

CREATE TABLE IF NOT EXISTS users (
    id                INTEGER PRIMARY KEY,
    slug              TEXT NOT NULL,
    first_name        TEXT,
    last_name         TEXT,
    school            TEXT,
    major             TEXT,
    interests         TEXT,
    expertise         TEXT,
    github            TEXT,
    twitter           TEXT,
    website           TEXT,
    views             INTEGER,
    num_followers     INTEGER,
    created_at        TEXT,
    last_online       TEXT,
    profile_synced_at TEXT,   -- last time the profile was fetched
    links_synced_at   TEXT,   -- last time new saves were checked
    full_synced_at    TEXT,   -- last time every page of saves was read
    sync_error        TEXT
);

CREATE TABLE IF NOT EXISTS follows (
    follower_id INTEGER NOT NULL,
    followed_id INTEGER NOT NULL,
    PRIMARY KEY (follower_id, followed_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS links (
    id             INTEGER PRIMARY KEY,
    url            TEXT NOT NULL,
    title          TEXT,
    snippet        TEXT,      -- opening text of the article; used later for search
    first_saved_by INTEGER    -- Curius createdBy: the first person to save it
);

CREATE TABLE IF NOT EXISTS saves (
    user_id     INTEGER NOT NULL,
    link_id     INTEGER NOT NULL,
    saved_at    TEXT NOT NULL,
    modified_at TEXT,
    favorite    INTEGER NOT NULL DEFAULT 0,
    to_read     INTEGER,
    PRIMARY KEY (user_id, link_id)
) WITHOUT ROWID;

-- Lets "saved in the last N days" read only recent rows instead of the whole table.
CREATE INDEX IF NOT EXISTS saves_by_time ON saves (saved_at);

CREATE TABLE IF NOT EXISTS topics (
    id      INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    topic   TEXT NOT NULL,
    slug    TEXT,
    public  INTEGER
);

CREATE TABLE IF NOT EXISTS save_topics (
    user_id  INTEGER NOT NULL,
    link_id  INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    PRIMARY KEY (user_id, link_id, topic_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS highlights (
    user_id    INTEGER NOT NULL,
    link_id    INTEGER NOT NULL,
    id         INTEGER NOT NULL,
    text       TEXT NOT NULL,
    comment    TEXT,
    created_at TEXT,
    PRIMARY KEY (user_id, link_id, id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    users_done  INTEGER DEFAULT 0,
    saves_seen  INTEGER DEFAULT 0,
    status      TEXT
);

-- Top articles saved in the last 7 days, by number of people. Curius's own pages excluded.
CREATE VIEW IF NOT EXISTS top_week AS
SELECT l.id AS link_id, l.title, l.url,
       COUNT(*) AS people,
       MAX(s.saved_at) AS latest_save
FROM saves s
JOIN links l ON l.id = s.link_id
WHERE s.saved_at >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-7 days')
  AND l.url NOT LIKE '%://curius.app%'
  AND l.url NOT LIKE '%://www.curius.app%'
GROUP BY l.id
ORDER BY people DESC, latest_save DESC;

-- Every save with the person's name and the article, for ad hoc queries.
CREATE VIEW IF NOT EXISTS saves_detail AS
SELECT s.saved_at,
       trim(coalesce(u.first_name, '') || ' ' || coalesce(u.last_name, '')) AS name,
       u.slug,
       l.title, l.url, s.user_id, s.link_id
FROM saves s
JOIN users u ON u.id = s.user_id
JOIN links l ON l.id = s.link_id;
