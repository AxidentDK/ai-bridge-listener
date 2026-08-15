"""Deciding WHAT to analyse — the resumable half.

At a terabyte, a full pass is measured in hours, and nobody completes an eight-hour
job uninterrupted. So the design assumption is that every run is a RESUMED run: work
out what has changed, do only that, and make stopping cheap.

Two sources for the file list:

* **Live's own analysis database** — it already knows the library, so both indexes
  end up describing the same files and the bridge's fallback stays coherent. Note
  only about half of what Live indexes is audio; the rest are presets and racks,
  which are meaningless to analyse and are filtered out here.
* **A directory walk** — for folders Live has never seen.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".w64", ".wv"}

#: Path fragments to skip. These hold .wav CONTAINERS that are not recordings —
#: a wavetable is a synthesis ingredient, and tagging one as audio yields confident
#: nonsense. Extension cannot distinguish them, so the filter is by path.
EXCLUDE_FRAGMENTS = (
    r"\tables",              # Serum / Vital wavetables
    r"\wavetables",
    r"\single cycle",
    r"\impulse",             # convolution IRs — a 1.5 ms cabinet impulse is not a sound
)


def _excluded(path: str, extra: tuple[str, ...] = ()) -> bool:
    # "._Name.wav" is a macOS AppleDouble stub — metadata left behind when a Mac
    # writes to a non-Mac filesystem, carrying no audio. Every one of the 22 failures
    # in the first full scan was one of these.
    if os.path.basename(path).startswith("._"):
        return True
    low = path.lower()
    return any(frag in low for frag in EXCLUDE_FRAGMENTS + extra)


@dataclass(frozen=True)
class Candidate:
    path: str          # the OS-native path, as the bridge will see it
    size: int
    mtime: float

    def unchanged_since(self, recorded: tuple[int | None, float | None, str | None],
                        analyzer: str) -> bool:
        """Skip only if size, mtime AND the analyzer version all still match.

        Including the analyzer is what makes a model upgrade re-analyse everything
        instead of silently leaving old verdicts in place next to new ones.
        """
        size, mtime, ana = recorded
        return (size == self.size
                and mtime is not None and abs(mtime - self.mtime) < 1e-6
                and ana == analyzer)


def _stat(path: str) -> Candidate | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return Candidate(path=path, size=st.st_size, mtime=st.st_mtime)


def from_live_database(db_path: Path, limit: int | None = None) -> list[Candidate]:
    """Candidates taken from Live's own index, so both describe the same library.

    There is no folders table: ``files`` IS the tree. Directories and audio share it,
    linked by ``parent_id``, with drive roots (``C:\\``) sitting at the top. So a path
    is rebuilt by walking parents inside the same table.
    """
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        nodes = {fid: (name, parent) for fid, parent, name in
                 con.execute("SELECT file_id, parent_id, name FROM files")}
    finally:
        con.close()

    out: list[Candidate] = []
    for fid, (name, parent) in nodes.items():
        if not name or Path(name).suffix.lower() not in AUDIO_SUFFIXES:
            continue                                  # presets, racks, directories
        cand = _stat(_resolve(fid, nodes))
        if cand:
            out.append(cand)
            if limit and len(out) >= limit:
                break
    return out


def _resolve(file_id, nodes: dict) -> str:
    """Walk parent_id up to a root to rebuild an absolute path."""
    parts: list[str] = []
    seen = set()
    cur = file_id
    while cur in nodes and cur not in seen:
        seen.add(cur)
        name, parent = nodes[cur]
        parts.append(name)
        cur = parent
    return os.path.normpath(os.path.join(*reversed(parts))) if parts else ""


def from_folder(root: Path, limit: int | None = None,
                exclude: tuple[str, ...] = ()) -> list[Candidate]:
    """Candidates from a directory walk, for folders Live has never indexed."""
    out: list[Candidate] = []
    for dirpath, _dirs, files in os.walk(root):
        if _excluded(dirpath, exclude):
            continue
        for name in files:
            if Path(name).suffix.lower() not in AUDIO_SUFFIXES:
                continue
            full = os.path.join(dirpath, name)
            if _excluded(full, exclude):
                continue
            cand = _stat(full)
            if cand:
                out.append(cand)
                if limit and len(out) >= limit:
                    return out
    return out


def plan(candidates: list[Candidate], done: dict, analyzer: str) -> tuple[list[Candidate], int]:
    """Split candidates into (to_do, skipped). Pure function — trivially testable."""
    todo = [c for c in candidates
            if c.path not in done or not c.unchanged_since(done[c.path], analyzer)]
    return todo, len(candidates) - len(todo)
