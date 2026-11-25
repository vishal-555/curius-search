"""
Tests for update_index.py and the API's hot reload, on a tiny synthetic index (no model, runs in seconds).

    python -m pytest test_update_index.py -v
"""

import json
import sqlite3

import numpy as np
import pytest

import update_index as ui
from search_api import Engine, Settings, read_index, short_snippet


def unit(rows, seed):
    v = np.random.default_rng(seed).standard_normal((rows, ui.DIMS)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_links_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE links (id INTEGER PRIMARY KEY, url TEXT NOT NULL, title TEXT, snippet TEXT)")
    con.executemany("INSERT INTO links VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()


@pytest.fixture()
def index(tmp_path):
    """Old-style index folder (no "parts" in the manifest), like the Kaggle output: ids 1..5."""
    vec_dir = tmp_path / "vectors"
    vec_dir.mkdir()
    np.save(vec_dir / "ids.npy", np.arange(1, 6, dtype=np.int64))
    np.save(vec_dir / "vec_000.npy", unit(3, 0))
    np.save(vec_dir / "vec_001.npy", unit(2, 1))
    (vec_dir / "manifest.json").write_text(json.dumps(
        {"model": ui.MODEL_ID, "dims": ui.DIMS, "count": 5, "checks": {"vectors": 5, "ids": 5}}))
    search_db = tmp_path / "articles.sqlite"
    make_links_db(search_db, [(i, f"https://example.com/{i}", f"Article {i}", f"preview {i}") for i in range(1, 6)])
    return vec_dir, search_db


def test_build_text_matches_kaggle_format():
    assert ui.build_text(" Title ", " body \n") == "Title\nbody"
    assert ui.build_text(None, "body") == "body"
    assert ui.build_text("Title", None) == "Title"


def test_chunk_files_reads_old_and_new_manifests(index):
    vec_dir, _ = index
    assert ui.chunk_files(vec_dir, {}) == ["vec_000.npy", "vec_001.npy"]
    assert ui.chunk_files(vec_dir, {"parts": ["vec_001.npy"]}) == ["vec_001.npy"]


def test_append_chunk_extends_index(index):
    vec_dir, _ = index
    manifest, ids = ui.load_index(vec_dir)
    m, all_ids = ui.append_chunk(vec_dir, manifest, ids, [10, 11], unit(2, 2), note="test")
    assert m["parts"] == ["vec_000.npy", "vec_001.npy", "vec_002.npy"]
    assert m["count"] == 7 and m["updates"][-1]["added"] == 2
    manifest2, ids2 = ui.load_index(vec_dir)  # re-read from disk: consistent
    assert manifest2 == m and ids2.tolist() == [1, 2, 3, 4, 5, 10, 11]
    assert not list(vec_dir.glob("*.tmp"))


def test_append_chunk_rejects_bad_input(index):
    vec_dir, _ = index
    manifest, ids = ui.load_index(vec_dir)
    with pytest.raises(ValueError, match="already in the index"):
        ui.append_chunk(vec_dir, manifest, ids, [5], unit(1, 3))
    with pytest.raises(ValueError, match="unit length"):
        ui.append_chunk(vec_dir, manifest, ids, [9], unit(1, 3) * 2)
    with pytest.raises(ValueError, match="expected"):
        ui.append_chunk(vec_dir, manifest, ids, [9, 10], unit(1, 3))
    assert ui.load_index(vec_dir)[0]["count"] == 5  # nothing was written


def test_load_index_detects_inconsistency(index):
    vec_dir, _ = index
    np.save(vec_dir / "ids.npy", np.arange(1, 5, dtype=np.int64))
    with pytest.raises(SystemExit, match="inconsistent"):
        ui.load_index(vec_dir)


def test_sync_metadata_and_pending(index, tmp_path):
    _, search_db = index
    source = tmp_path / "curius.db"
    make_links_db(source, [(i, f"https://example.com/{i}", f"Article {i}", f"preview {i}") for i in range(1, 6)]
                  + [(6, "https://new.example/6", "New one", "fresh preview")])
    con = sqlite3.connect(source)
    con.execute("UPDATE links SET title = 'Renamed' WHERE id = 2")
    con.commit()
    con.close()

    assert ui.sync_metadata(source, search_db, dry_run=True) == (1, 1)
    assert ui.sync_metadata(source, search_db) == (1, 1)
    assert ui.sync_metadata(source, search_db) == (0, 0)  # idempotent

    con = sqlite3.connect(search_db)
    assert con.execute("SELECT title FROM links WHERE id = 2").fetchone()[0] == "Renamed"
    con.close()
    pending = ui.pending_articles(search_db, np.arange(1, 6))
    assert pending == [(6, "New one\nfresh preview")]


def test_short_snippet():
    assert short_snippet("  a  b \n c ", 50) == "a b c"
    assert short_snippet(None, 50) == ""
    out = short_snippet("one two three four five", 12)
    assert out == "one two…" and len(out) <= 13


def test_api_hot_reload(index):
    vec_dir, search_db = index
    s = Settings(env="dev", vec_dir=vec_dir, db_path=search_db)
    eng = Engine(s)
    eng.index, eng.ready = read_index(s), True  # what load() does, without the model
    assert eng.reload_if_changed() is False

    new = unit(1, 9)
    con = sqlite3.connect(search_db)
    con.execute("INSERT INTO links VALUES (42, 'https://example.com/42', 'Fresh', 'just saved')")
    con.commit()
    con.close()
    manifest, ids = ui.load_index(vec_dir)
    ui.append_chunk(vec_dir, manifest, ids, [42], new)

    assert eng.reload_if_changed() is True
    assert len(eng.index.ids) == 6
    top = eng.search(new[0])[0]
    assert top["id"] == 42 and top["title"] == "Fresh" and top["snippet"] == "just saved"


def test_api_reload_keeps_old_index_when_folder_is_broken(index):
    vec_dir, search_db = index
    s = Settings(env="dev", vec_dir=vec_dir, db_path=search_db)
    eng = Engine(s)
    eng.index, eng.ready = read_index(s), True
    m = json.loads((vec_dir / "manifest.json").read_text())
    m["count"] = 99
    (vec_dir / "manifest.json").write_text(json.dumps(m))
    assert eng.reload_if_changed() is False
    assert len(eng.index.ids) == 5
    assert eng.reload_if_changed() is False  # same broken manifest: not retried
