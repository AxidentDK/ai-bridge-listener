"""SQLite store — the ONLY thing this program hands to the bridge.

The schema is not ours to invent: it is published by the consumer, in the bridge's
``host/sidecar.py`` as ``SCHEMA_SQL``, so that a producer written in any language on
any OS has exactly one file to conform to. It is duplicated here rather than
imported, because the sidecar must not depend on the bridge being installed — but the
duplication is checked, see ``verify_against_bridge``.

Writes are single-threaded on purpose. SQLite takes one writer, decoding is what
actually needs the cores, and a single writer means no lock contention to debug.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: v2 adds the MEASURED half — the `properties` columns that route off one-shot vs
#: loop, and an `embeddings` table. v1 shipped `properties` and never wrote a row to it.
#: v3 adds `bars`: a tempo and its double are not the same tempo, and the bar count is
#: the evidence for which one a file is.
SCHEMA_VERSION = 3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id            INTEGER PRIMARY KEY,
    path          TEXT NOT NULL UNIQUE,
    source_path   TEXT,
    size_bytes    INTEGER,
    mtime         REAL,
    duration_sec  REAL,
    sample_rate   INTEGER,
    channels      INTEGER,
    analyzed_at   TEXT,
    analyzer      TEXT,
    error         TEXT
);
CREATE TABLE IF NOT EXISTS tags (
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    namespace   TEXT NOT NULL,
    label       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    model       TEXT NOT NULL,
    PRIMARY KEY (file_id, namespace, label, model)
);
CREATE TABLE IF NOT EXISTS properties (
    file_id        INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    kind           TEXT,
    onsets         INTEGER,
    bpm            REAL,
    bpm_confidence REAL,
    bars           INTEGER,
    key            TEXT,
    scale          TEXT,
    key_strength   REAL,
    pitch_hz       REAL,
    pitch_midi     INTEGER,
    pitch_confidence REAL,
    attack_ms      REAL,
    decay_ms       REAL,
    stereo_width   REAL,
    stereo_correlation REAL,
    danceability   REAL,
    loudness_lufs  REAL
);
CREATE TABLE IF NOT EXISTS embeddings (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    dim     INTEGER NOT NULL,
    dtype   TEXT NOT NULL,
    vector  BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tags_lookup ON tags(namespace, label, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_files_path  ON files(path);
CREATE INDEX IF NOT EXISTS idx_props_kind  ON properties(kind);
"""

#: Every column ``record`` writes into ``properties``. Named once so the writer and
#: the schema cannot drift apart silently.
PROPERTY_COLUMNS = (
    "kind", "onsets", "bpm", "bpm_confidence", "bars", "key", "scale", "key_strength",
    "pitch_hz", "pitch_midi", "pitch_confidence", "attack_ms", "decay_ms",
    "stereo_width", "stereo_correlation", "danceability", "loudness_lufs",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """Owns the database. One instance, one writer, opened for the whole run."""

    def __init__(self, path: Path, analyzer: str):
        self.path = Path(path)
        self.analyzer = analyzer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        # WAL lets the bridge READ while a long scan is still writing — the whole
        # point of a resumable job you can leave running.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # SQLite disables foreign keys per connection by DEFAULT, so the REFERENCES
        # clauses in the schema were decoration. With them enforced, the lastrowid bug
        # above would have raised "FOREIGN KEY constraint failed" on its very first
        # row instead of quietly writing 580,000 tags against ids that do not exist.
        # A guardrail that is declared but not switched on is worse than none: the
        # schema reads as if something is checking.
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate_if_older()
        self.conn.executescript(SCHEMA_SQL)
        self._set_meta("schema_version", str(SCHEMA_VERSION))
        self._set_meta("analyzer", analyzer)
        self._set_meta("built_at", _utc_now())
        self.conn.commit()

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    # ---- resumability -------------------------------------------------------

    def already_done(self) -> dict[str, tuple[int | None, float | None, str | None]]:
        """{path: (size, mtime, analyzer)} for every file already recorded.

        Read once into memory rather than queried per file: a million point-lookups
        cost far more than a million dict entries, and this runs before any work.
        """
        return {r["path"]: (r["size_bytes"], r["mtime"], r["analyzer"])
                for r in self.conn.execute(
                    "SELECT path, size_bytes, mtime, analyzer FROM files")}

    def _migrate_if_older(self) -> None:
        """Rebuild the derived tables when the schema version has moved on.

        ``CREATE TABLE IF NOT EXISTS`` does NOT add columns to a table that already
        exists — it silently does nothing and the run then stamps the NEW version onto
        the OLD tables, which is a database that lies about its own shape. Everything
        here is derived from audio files that are still on disk, so dropping and
        rebuilding is a re-scan, not data loss. ``meta`` is kept.
        """
        try:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.OperationalError:
            return                                    # brand new database
        if row is None:
            return
        try:
            existing = int(row["value"])
        except (TypeError, ValueError):
            existing = -1
        if existing == SCHEMA_VERSION:
            return
        print(f"  schema {existing} -> {SCHEMA_VERSION}: rebuilding derived tables "
              f"(every file is re-analysed on a version change anyway)")
        for table in ("embeddings", "properties", "tags", "files"):
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")   # noqa: S608
        self.conn.commit()

    # ---- writing ------------------------------------------------------------

    def record(self, path: str, *, size: int, mtime: float, source_path: str | None = None,
               duration: float | None = None, sample_rate: int | None = None,
               channels: int | None = None, error: str | None = None,
               tags: list[tuple[str, str, float, str]] | None = None,
               properties: dict | None = None,
               embedding=None) -> int:
        """Upsert one file and its results. Returns the file id.

        A FAILED file is recorded with ``error`` set rather than left out, so the
        next run skips it instead of hitting the same broken file forever.
        """
        self.conn.execute(
            "INSERT INTO files(path, source_path, size_bytes, mtime, duration_sec, "
            "                  sample_rate, channels, analyzed_at, analyzer, error) "
            "VALUES(?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "  source_path=excluded.source_path, size_bytes=excluded.size_bytes, "
            "  mtime=excluded.mtime, duration_sec=excluded.duration_sec, "
            "  sample_rate=excluded.sample_rate, channels=excluded.channels, "
            "  analyzed_at=excluded.analyzed_at, analyzer=excluded.analyzer, "
            "  error=excluded.error",
            (path, source_path, size, mtime, duration, sample_rate, channels,
             _utc_now(), self.analyzer, error))
        # ALWAYS resolve the id by path. NEVER `cur.lastrowid` here.
        #
        # On an `ON CONFLICT DO UPDATE` that takes the UPDATE branch, SQLite leaves
        # lastrowid at the last row actually INSERTED — a different file's id — and it
        # is non-zero, so a `if not file_id` fallback never fires. Every re-scan then
        # wrote that file's tags, properties and embedding against SOMEONE ELSE'S id:
        #
        #     INSERT a -> lastrowid 1        INSERT b -> lastrowid 2
        #     UPDATE a -> lastrowid 2        <- a's real id is 1
        #
        # Measured in the live index before this was found: 12,825 orphaned property
        # and embedding rows and 581,789 orphaned tags, plus an unknown number
        # silently attached to the WRONG existing file — which is the worse half,
        # because nothing about it looks broken. It only bites on re-analysis, so a
        # first scan of an empty database is always clean and the bug hides until the
        # second run.
        file_id = self.conn.execute(
            "SELECT id FROM files WHERE path=?", (path,)).fetchone()["id"]

        # Replace rather than merge: a re-analysis supersedes the previous verdict.
        self.conn.execute("DELETE FROM tags WHERE file_id=?", (file_id,))
        if tags:
            self.conn.executemany(
                "INSERT OR REPLACE INTO tags(file_id, namespace, label, confidence, "
                "model) VALUES(?,?,?,?,?)",
                [(file_id, ns, label, conf, model) for ns, label, conf, model in tags])
        if properties:
            cols = list(PROPERTY_COLUMNS)
            vals = [properties.get(c) for c in cols]
            self.conn.execute(
                f"INSERT INTO properties(file_id, {', '.join(cols)}) "  # noqa: S608
                f"VALUES({', '.join('?' * (len(cols) + 1))}) "
                f"ON CONFLICT(file_id) DO UPDATE SET "
                + ", ".join(f"{c}=excluded.{c}" for c in cols),
                [file_id, *vals])
        if embedding is not None and len(embedding):
            # float16 halves the footprint for a vector whose values sit well inside
            # its range; 22,100 x 1280 is ~56 MB stored, against ~113 MB at float32.
            # Stored so the drum classifier can be trained and RE-trained without
            # paying the 14-minute extraction again.
            vec = np.asarray(embedding, dtype=np.float16)
            self.conn.execute(
                "INSERT INTO embeddings(file_id, dim, dtype, vector) VALUES(?,?,?,?) "
                "ON CONFLICT(file_id) DO UPDATE SET dim=excluded.dim, "
                "dtype=excluded.dtype, vector=excluded.vector",
                (file_id, int(vec.size), "float16", vec.tobytes()))
        return file_id

    def commit(self) -> None:
        self._set_meta("built_at", _utc_now())
        self.conn.commit()

    def counts(self) -> dict:
        q = self.conn.execute
        return {
            "files_ok": q("SELECT COUNT(*) FROM files WHERE error IS NULL").fetchone()[0],
            "files_failed": q("SELECT COUNT(*) FROM files WHERE error IS NOT NULL").fetchone()[0],
            "tags": q("SELECT COUNT(*) FROM tags").fetchone()[0],
        }

    def close(self) -> None:
        self.commit()
        self.conn.close()


def verify_against_bridge(bridge_sidecar_py: Path) -> list[str]:
    """Compare our DDL with the bridge's published one. Returns a list of problems.

    Cheap insurance against the duplication above drifting: the contract only works
    if both halves agree, and a silent divergence would surface as the bridge
    reporting an unreadable database long after the scan finished.
    """
    problems = []
    try:
        text = bridge_sidecar_py.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"could not read the bridge's sidecar.py: {exc}"]

    for table in ("meta", "files", "tags", "properties"):
        if f"CREATE TABLE IF NOT EXISTS {table}" not in text:
            problems.append(f"bridge schema has no table {table!r}")
    for column in ("source_path", "analyzed_at", "analyzer", "error", "confidence",
                   "namespace", "loudness_lufs"):
        if column not in text:
            problems.append(f"bridge schema is missing column {column!r}")
    if "SCHEMA_VERSION = " in text:
        ver = text.split("SCHEMA_VERSION = ", 1)[1].split("\n", 1)[0].strip()
        if ver != str(SCHEMA_VERSION):
            problems.append(f"schema version mismatch: bridge {ver}, listener {SCHEMA_VERSION}")
    return problems
