# curius-search

Semantic search over every public article saved on [Curius](https://curius.app). Describe what you
want to read in your own words and get the closest articles by meaning. Live at
[vishal.fast/curius](https://vishal.fast/curius).

Everything runs at zero cost: the data lives in local SQLite, the vectors were built on Kaggle's
free GPU, and the API runs on an Oracle Cloud Always Free VM behind a Cloudflare Tunnel.

## How it works

```
curius.app API --db.py--> data/curius.db --update_index.py--> data/vectors/ + articles.sqlite
                                                                    |
                                          web page --fetch--> search_api.py (FastAPI)
```

1. `db.py` loads users, saved links, highlights, topics and follows into `data/curius.db`.
2. Each article is embedded with `Qwen/Qwen3-Embedding-0.6B` (1,024 dims) from `title + "\n" + preview`.
   The first 202,450 were done on Kaggle (`make_kaggle_full_notebook.py`); new ones are added on the
   CPU by `update_index.py`.
3. `search_api.py` keeps every vector in RAM, embeds the query, and returns the top 10 by cosine
   similarity (about 0.5 s on 2 CPU cores, most of it embedding the query).
4. The API returns only article fields (id, title, url, snippet, score), never who saved what.

## Setup

Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux: .venv/bin/pip
.venv/Scripts/pip install -r requirements.txt
```

`data/` is not in git. It holds `curius.db`, `vectors/` (`ids.npy`, `vec_NNN.npy`, `manifest.json`)
and `kaggle/articles.sqlite`.

## Keeping the data fresh

```bash
python db.py daily          # new saves from users active since their last sync (minutes, not hours)
python update_index.py      # embed links that have no vector yet and append them to the index
```

`db.py daily` only reads each active user's newest saves until it reaches one already stored.
`update_index.py` copies new and changed article metadata into the search database, embeds only the
missing links, and writes them as new `vec_NNN.npy` chunks. It can be stopped and resumed, and the
index folder is always consistent (new files are written aside and moved into place, manifest last).
A running API checks the manifest every `RELOAD_CHECK_S` seconds (60) and swaps in the new index
without a restart, then clears its query cache.

Use `python update_index.py --dry-run` to see what would change. `python db.py stats` shows counts and
recent runs.

## Running locally

```bash
APP_ENV=dev .venv/Scripts/python -m uvicorn search_api:app --host 127.0.0.1 --port 8000 --workers 1
python -m http.server 5500 -d web
```

Open <http://localhost:5500/> (use `localhost`, not `127.0.0.1`: dev CORS allows `http://localhost`).
Start-up takes about 30 s while the model and vectors load; `/health` returns 503 until then.

## API

| Endpoint | Returns |
|---|---|
| `GET /search?q=...` | `{"query", "cached", "results": [{"id", "title", "url", "snippet", "score"}]}` |
| `GET /health` | 200 `{"status": "ok", "articles": N}` once loaded, 503 before |

Limits per visitor: 12 searches a minute, 300 a day (429 with `Retry-After`). When every model copy is
busy the API answers 503 with `Retry-After`. Settings are environment variables, listed at the top of
`search_api.py`.

## Tests

```bash
python -m pytest test_update_index.py -v   # index updates and hot reload, synthetic data, seconds
python -m pytest test_api.py -v -s         # full API against the real vectors and model, about a minute
```

## Files

| File | Purpose |
|---|---|
| `db.py`, `schema.sql`, `crawler.py` | Curius loader (`init`, `full`, `daily`, `refresh`, `stats`) |
| `update_index.py` | Incremental embedding and index append |
| `search_api.py` | FastAPI search service |
| `web/index.html` | Search page (plain HTML/CSS/JS) |
| `make_kaggle_*notebook.py`, `kaggle_*.ipynb` | Full embedding run on Kaggle's free GPU |
| `verify_vectors.py`, `query_vector_check.py`, `compare_fp16.py` | Checks used when choosing the setup |
| `embed.py`, `search.py`, `top_links.py`, `who_saved.py` | Early prototypes, kept for reference |
