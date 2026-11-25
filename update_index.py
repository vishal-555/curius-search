"""
Bring the search index up to date after `python db.py daily`.

    python update_index.py              # sync article metadata, embed links that have no vector yet
    python update_index.py --dry-run    # only report what would change
    python update_index.py --limit 500  # embed at most 500 new links this run

Steps:
1. Copy id, url, title, snippet for every link from data/curius.db into the search database
   (data/kaggle/articles.sqlite). New links are added, changed titles and URLs are updated.
2. Embed every link that has no vector yet, exactly like the Kaggle run: text = title + newline +
   preview, no prompt, max_seq_length 1024, fp32 on the CPU, unit length.
3. Append them as new vec_NNN.npy chunks and extend ids.npy and manifest.json.

Safe to stop at any time. Each chunk (--commit-every articles) is written to temporary files and
then moved into place, with manifest.json last, so the folder is always either the old index or
the new one. Run the same command again to continue. A running search_api.py notices the new
manifest and reloads on its own (see RELOAD_CHECK_S there).

Vectors of existing links are not recomputed when their title or preview changes on Curius.
That is rare and the old vector still describes the article.
"""

import argparse
import datetime as dt
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SOURCE_DB = ROOT / "data" / "curius.db"
SEARCH_DB = ROOT / "data" / "kaggle" / "articles.sqlite"
VEC_DIR = ROOT / "data" / "vectors"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DIMS = 1024
MAX_SEQ = 1024  # same as the Kaggle run (manifest.json "max_seq_length")


# ---------------------------------------------------------------------------
# Index folder: manifest.json + ids.npy + vec_NNN.npy
# ---------------------------------------------------------------------------

def build_text(title, snippet):
    """Article text as embedded by make_kaggle_full_notebook.py. Must never change."""
    return ((title or "").strip() + "\n" + (snippet or "").strip()).strip()


def chunk_files(vec_dir, manifest):
    """Chunk file names in order. Older manifests have no "parts" list: use the files on disk."""
    if manifest.get("parts"):
        return list(manifest["parts"])
    return sorted(p.name for p in Path(vec_dir).glob("vec_*.npy"))


def load_index(vec_dir):
    """Returns (manifest, ids). Fails loudly if the folder is inconsistent."""
    vec_dir = Path(vec_dir)
    manifest = json.loads((vec_dir / "manifest.json").read_text())
    if manifest["model"] != MODEL_ID or manifest["dims"] != DIMS:
        raise SystemExit(f"index was made with {manifest['model']} ({manifest['dims']} dims), "
                         f"this script uses {MODEL_ID} ({DIMS} dims)")
    ids = np.load(vec_dir / "ids.npy")
    rows = sum(np.load(vec_dir / f, mmap_mode="r").shape[0] for f in chunk_files(vec_dir, manifest))
    if not (len(ids) == rows == manifest["count"]):
        raise SystemExit(f"index is inconsistent: ids.npy {len(ids)}, vectors {rows}, "
                         f"manifest {manifest['count']}. Restore data/vectors from a backup.")
    if len(np.unique(ids)) != len(ids):
        raise SystemExit("index has duplicate ids")
    return manifest, ids


def _atomic_save_npy(path, arr):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    return tmp


def append_chunk(vec_dir, manifest, ids, new_ids, new_vecs, note=None):
    """Write one new chunk and the extended ids/manifest. Returns (manifest, ids)."""
    vec_dir = Path(vec_dir)
    new_ids = np.asarray(new_ids, dtype=ids.dtype)
    new_vecs = np.asarray(new_vecs, dtype=np.float32)
    if new_vecs.shape != (len(new_ids), DIMS):
        raise ValueError(f"expected {(len(new_ids), DIMS)} vectors, got {new_vecs.shape}")
    if not np.isfinite(new_vecs).all():
        raise ValueError("non-finite values in new vectors")
    norms = np.linalg.norm(new_vecs, axis=1)
    if np.abs(norms - 1).max() > 1e-3:
        raise ValueError(f"vectors are not unit length (norms {norms.min():.4f}..{norms.max():.4f})")
    if np.isin(new_ids, ids).any():
        raise ValueError("some new ids are already in the index")

    parts = chunk_files(vec_dir, manifest)
    last = max((int(p[4:-4]) for p in parts), default=-1)
    name = f"vec_{last + 1:03d}.npy"
    all_ids = np.concatenate([ids, new_ids])

    m = dict(manifest)
    m["parts"] = parts + [name]
    m["count"] = int(len(all_ids))
    m["checks"] = dict(m.get("checks", {}), vectors=int(len(all_ids)), ids=int(len(all_ids)))
    m["updates"] = list(m.get("updates", [])) + [{
        "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "part": name, "added": int(len(new_ids)), **({"note": note} if note else {}),
    }]

    # Write everything to .tmp, then move into place. manifest.json goes last: readers trust it.
    tmp_vec = _atomic_save_npy(vec_dir / name, new_vecs)
    tmp_ids = _atomic_save_npy(vec_dir / "ids.npy", all_ids)
    tmp_man = vec_dir / "manifest.json.tmp"
    tmp_man.write_text(json.dumps(m, indent=2))
    os.replace(tmp_vec, vec_dir / name)
    os.replace(tmp_ids, vec_dir / "ids.npy")
    os.replace(tmp_man, vec_dir / "manifest.json")
    return m, all_ids


# ---------------------------------------------------------------------------
# Article metadata
# ---------------------------------------------------------------------------

def sync_metadata(source_db, search_db, dry_run=False):
    """Copy links from curius.db into the search database. Returns (added, changed)."""
    con = sqlite3.connect(search_db)
    con.execute("CREATE TABLE IF NOT EXISTS links (id INTEGER PRIMARY KEY, url TEXT, title TEXT, snippet TEXT)")
    con.execute("ATTACH DATABASE ? AS src", (str(source_db),))
    added = con.execute("SELECT count(*) FROM src.links s WHERE s.id NOT IN (SELECT id FROM main.links)").fetchone()[0]
    changed = con.execute("""
        SELECT count(*) FROM src.links s JOIN main.links m ON m.id = s.id
        WHERE m.url IS NOT s.url OR m.title IS NOT s.title OR m.snippet IS NOT s.snippet""").fetchone()[0]
    if not dry_run and (added or changed):
        with con:
            con.execute("""
                INSERT INTO main.links (id, url, title, snippet)
                SELECT id, url, title, snippet FROM src.links WHERE true
                ON CONFLICT(id) DO UPDATE SET url = excluded.url, title = excluded.title,
                    snippet = excluded.snippet
                WHERE main.links.url IS NOT excluded.url OR main.links.title IS NOT excluded.title
                   OR main.links.snippet IS NOT excluded.snippet""")
    con.execute("DETACH DATABASE src")
    con.close()
    return added, changed


def pending_articles(search_db, indexed_ids):
    """(id, text) for links in the search database that have no vector, oldest id first."""
    con = sqlite3.connect(search_db)
    rows = con.execute("SELECT id, title, snippet FROM links ORDER BY id").fetchall()
    con.close()
    have = set(int(i) for i in indexed_ids)
    return [(lid, build_text(t, s)) for lid, t, s in rows if lid not in have]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_model(threads):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    try:  # transformers 5.x calls this "dtype"; 4.x calls it "torch_dtype"
        m = SentenceTransformer(MODEL_ID, device="cpu", model_kwargs={"dtype": torch.float32})
    except TypeError:
        m = SentenceTransformer(MODEL_ID, device="cpu", model_kwargs={"torch_dtype": torch.float32})
    m.float()  # newer libraries load bfloat16 by default, which is slower on CPU and not what Kaggle used
    m.max_seq_length = MAX_SEQ
    return m


def embed(model, texts, batch_size):
    # Documents get no prompt (the Kaggle run used none). sentence-transformers sorts by length itself.
    v = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False)
    return v.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    p.add_argument("--limit", type=int, help="embed at most this many new links")
    p.add_argument("--commit-every", type=int, default=1000, help="articles per new chunk file (1000)")
    p.add_argument("--batch", type=int, default=16, help="encode batch size (16)")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="torch threads (default: half the logical CPUs; more oversubscribes and can stall)")
    p.add_argument("--source-db", type=Path, default=SOURCE_DB)
    p.add_argument("--search-db", type=Path, default=SEARCH_DB)
    p.add_argument("--vec-dir", type=Path, default=VEC_DIR)
    a = p.parse_args()

    manifest, ids = load_index(a.vec_dir)
    added, changed = sync_metadata(a.source_db, a.search_db, dry_run=a.dry_run)
    print(f"Metadata: {added:,} new links, {changed:,} changed titles/URLs/previews"
          f"{' (dry run, not written)' if a.dry_run else ''}")
    if a.dry_run:
        con = sqlite3.connect(a.source_db)
        src_ids = [r[0] for r in con.execute("SELECT id FROM links")]
        con.close()
        todo = int((~np.isin(np.array(src_ids), ids)).sum())
        print(f"Index: {len(ids):,} vectors, {todo:,} links would be embedded")
        return

    todo = pending_articles(a.search_db, ids)
    if a.limit is not None:
        todo = todo[:a.limit]
    print(f"Index: {len(ids):,} vectors, {len(todo):,} links to embed")
    if not todo:
        return

    t_load = time.time()
    model = load_model(a.threads)
    print(f"Model loaded in {time.time() - t_load:.0f}s ({a.threads} threads, fp32, max_seq {MAX_SEQ})")

    done, t0 = 0, time.time()
    try:
        for lo in range(0, len(todo), a.commit_every):
            part = todo[lo:lo + a.commit_every]
            vecs = embed(model, [t for _, t in part], a.batch)
            manifest, ids = append_chunk(a.vec_dir, manifest, ids, [i for i, _ in part], vecs,
                                         note="update_index.py, CPU fp32")
            done += len(part)
            rate = done / (time.time() - t0)
            left = (len(todo) - done) / rate / 60 if rate else 0
            print(f"  {done:>6,}/{len(todo):,} | {rate:.1f} articles/s | ETA {left:.0f} min "
                  f"| wrote {manifest['parts'][-1]}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Every finished chunk is saved; run the same command to continue.")
    print(f"Done: {done:,} links embedded, index now {len(ids):,} vectors.")


if __name__ == "__main__":
    main()
