"""Tests for listener/db.py — the store, and the id it attaches results to.

The case that matters is RE-ANALYSIS. A first scan of an empty database inserts every
row, so every id is correct and nothing looks wrong; the damage only appears the
second time a file is seen. That is why this went unnoticed through several full
scans.

No pytest, matching the rest of the project: run the file.
"""
import os
import sqlite3
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from listener.db import PROPERTY_COLUMNS, SCHEMA_VERSION, Store  # noqa: E402


def _store(tmp, analyzer="test-1"):
    return Store(os.path.join(tmp, "index.db"), analyzer)


def test_reanalysis_attaches_results_to_the_right_file():
    """The regression that cost a whole index.

    `INSERT ... ON CONFLICT DO UPDATE` leaves lastrowid pointing at the last row
    actually INSERTED, so on the UPDATE branch it names a DIFFERENT file — and it is
    non-zero, so a `if not file_id` fallback never fires. Two files are enough to
    prove it: write A, write B, then re-write A and check A's results are still A's.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        store.record("A.wav", size=1, mtime=1.0, duration=1.0,
                     tags=[("ns", "alpha", 0.9, "m")], properties={"kind": "one_shot"})
        store.record("B.wav", size=2, mtime=2.0, duration=2.0,
                     tags=[("ns", "beta", 0.9, "m")], properties={"kind": "loop"})
        # Re-analysis of A — the UPDATE branch.
        store.record("A.wav", size=1, mtime=1.0, duration=1.0,
                     tags=[("ns", "alpha2", 0.9, "m")], properties={"kind": "one_shot"},
                     embedding=np.arange(4, dtype=np.float32))
        store.commit()
        store.close()

        conn = sqlite3.connect(os.path.join(tmp, "index.db"))
        conn.row_factory = sqlite3.Row
        rows = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM files")}

        tag_of = {p: [t["label"] for t in conn.execute(
            "SELECT label FROM tags WHERE file_id=?", (i,))] for p, i in rows.items()}
        assert tag_of["A.wav"] == ["alpha2"], tag_of
        assert tag_of["B.wav"] == ["beta"], f"B's tags were overwritten: {tag_of}"

        kind_of = {p: conn.execute(
            "SELECT kind FROM properties WHERE file_id=?", (i,)).fetchone()
            for p, i in rows.items()}
        assert kind_of["B.wav"]["kind"] == "loop", "B's properties were clobbered"

        emb = conn.execute("SELECT file_id FROM embeddings").fetchall()
        assert len(emb) == 1 and emb[0]["file_id"] == rows["A.wav"], \
            "the embedding was filed against the wrong file"
        conn.close()


def test_nothing_is_ever_orphaned():
    """Every derived row must point at a file that exists. 12,825 property rows and
    581,789 tags in the real index did not."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        for i in range(5):
            store.record(f"{i}.wav", size=i, mtime=float(i), duration=1.0,
                         tags=[("ns", f"t{i}", 0.5, "m")],
                         properties={"kind": "one_shot"},
                         embedding=np.zeros(4, dtype=np.float32))
        for i in range(5):                       # second pass: all UPDATE branch
            store.record(f"{i}.wav", size=i, mtime=float(i), duration=1.0,
                         tags=[("ns", f"u{i}", 0.5, "m")],
                         properties={"kind": "loop"},
                         embedding=np.ones(4, dtype=np.float32))
        store.commit()
        store.close()
        conn = sqlite3.connect(os.path.join(tmp, "index.db"))
        for table in ("tags", "properties", "embeddings"):
            orphans = conn.execute(
                f"SELECT COUNT(*) FROM {table} x "                   # noqa: S608
                "LEFT JOIN files f ON f.id = x.file_id WHERE f.id IS NULL").fetchone()[0]
            assert orphans == 0, f"{orphans} orphaned rows in {table}"
        assert conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0] == 5
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 5
        conn.close()


def test_embedding_round_trips_exactly():
    """float16 is lossy against float32 input, but storing and reading it back must
    not lose anything further — a truncated blob would be silent."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        vec = np.linspace(-3.0, 3.0, 1280).astype(np.float32)
        store.record("x.wav", size=1, mtime=1.0, embedding=vec)
        store.commit()
        store.close()
        conn = sqlite3.connect(os.path.join(tmp, "index.db"))
        row = conn.execute("SELECT dim, dtype, vector FROM embeddings").fetchone()
        assert row[0] == 1280 and row[1] == "float16"
        back = np.frombuffer(row[2], dtype=np.float16)
        assert back.size == 1280
        assert np.allclose(back.astype(np.float32), vec.astype(np.float16), atol=0)
        conn.close()


def test_a_schema_change_rebuilds_rather_than_lying():
    """CREATE TABLE IF NOT EXISTS does not add columns to an existing table, so
    stamping a new version onto old tables produces a database that misreports its own
    shape."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index.db")
        conn = sqlite3.connect(path)
        conn.executescript(
            "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT UNIQUE);"
            "CREATE TABLE properties(file_id INTEGER PRIMARY KEY, bpm REAL);")
        conn.execute("INSERT INTO meta VALUES('schema_version', '1')")
        conn.execute("INSERT INTO files(path) VALUES('old.wav')")
        conn.commit()
        conn.close()

        store = Store(path, "test-1")            # must migrate, not stamp
        store.record("new.wav", size=1, mtime=1.0, properties={"kind": "loop"})
        store.commit()
        store.close()
        conn = sqlite3.connect(path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(properties)")}
        for column in PROPERTY_COLUMNS:
            assert column in cols, f"{column} missing after migration"
        version = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        assert int(version) == SCHEMA_VERSION
        conn.close()


def test_resumability_skips_only_genuinely_unchanged_files():
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp, analyzer="v1")
        store.record("a.wav", size=10, mtime=100.0)
        store.commit()
        done = store.already_done()
        store.close()
        assert done["a.wav"] == (10, 100.0, "v1")
        # A changed analyzer must NOT look done — this is what makes a re-scan real.
        assert done["a.wav"][2] != "v2"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)
