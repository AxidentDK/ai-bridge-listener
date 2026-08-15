"""Pick a balanced sample of files with reliable filename ground truth.

Balanced on purpose: taking whatever comes first would over-weight whichever vendor
happens to sort early, and a difference between A and B could then be a difference
between libraries rather than between spectrograms.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from listener.evaluate import RULES  # noqa: E402

DB = Path.home() / ".ai-bridge" / "sound_index.db"
PER_CATEGORY = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("ab_filelist.txt")


def to_wsl(p: str) -> str:
    return f"/mnt/{p[0].lower()}{p[2:]}".replace("\\", "/")


con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
rows = [r[0] for r in con.execute("SELECT path FROM files WHERE error IS NULL")]

compiled = [(name, re.compile(pat, re.I)) for name, pat, _w, _b in RULES]
picked: dict[str, list[str]] = defaultdict(list)
for p in rows:
    hay = f"{os.path.basename(os.path.dirname(p))} {os.path.basename(p)}"
    for name, pattern in compiled:
        if pattern.search(hay):
            if len(picked[name]) < PER_CATEGORY:
                picked[name].append(p)
            break

total = 0
lines = []
for name, _p in compiled:
    n = len(picked[name])
    total += n
    print(f"  {name:<14} {n:>4}")
    lines += [to_wsl(p) for p in picked[name]]

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n{total} files -> {OUT}")
