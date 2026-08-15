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

SCHEMA_VERSION = 1

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
    bpm            REAL,
    bpm_confidence REAL,
    key            TEXT,
    scale          TEXT,
    key_strength   REAL,
    danceability   REAL,
    loudness_lufs  REAL
);
CREATE INDEX IF NOT EXISTS idx_tags_lookup ON tags(namespace, label, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_files_path  ON files(path);
"""


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

    # ---- writing ------------------------------------------------------------

    def record(self, path: str, *, size: int, mtime: float, source_path: str | None = None,
               duration: float | None = None, sample_rate: int | None = None,
               channels: int | None = None, error: str | None = None,
               tags: list[tuple[str, str, float, str]] | None = None,
               properties: dict | None = None) -> int:
        """Upsert one file and its results. Returns the file id.

        A FAILED file is recorded with ``error`` set rather than left out, so the
        next run skips it instead of hitting the same broken file forever.
        """
        cur = self.conn.execute(
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
        file_id = cur.lastrowid
        if not file_id:
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
            cols = ["bpm", "bpm_confidence", "key", "scale", "key_strength",
                    "danceability", "loudness_lufs"]
            vals = [properties.get(c) for c in cols]
            self.conn.execute(
                f"INSERT INTO properties(file_id, {', '.join(cols)}) "  # noqa: S608
                f"VALUES({', '.join('?' * (len(cols) + 1))}) "
                f"ON CONFLICT(file_id) DO UPDATE SET "
                + ", ".join(f"{c}=excluded.{c}" for c in cols),
                [file_id, *vals])
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
