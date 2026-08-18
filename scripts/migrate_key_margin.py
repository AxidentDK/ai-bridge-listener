"""Bring an existing index in line with the key fixes, WITHOUT re-listening.

    python scripts/migrate_key_margin.py --dry-run
    python scripts/migrate_key_margin.py

WHY THIS IS NOT A RE-SCAN. Two things changed in `shared_dsp.key_from_chroma`, and
neither of them changes a single model output:

  1. a key is refused when its margin over the runner-up is below `MIN_KEY_MARGIN`
  2. keys are spelled by `key_name()` — "Eb major", not the theoretical "D# major"

The margin is already stored, as `key_strength`, and the spelling is a pure rename. So
the whole correction is derivable from what is in the database. Re-running the analyzer
over 29,870 files to recompute values that cannot change would cost hours and carry the
one risk this project has actually been bitten by — a rebuild that silently corrupts the
index. `FEATURE_VERSION` is deliberately NOT bumped for the same reason: the features are
identical, and bumping it would force exactly the pointless re-scan this avoids.

WHAT IT DOES NOT TOUCH: tempo, confidence, bars, pitch, tags, embeddings. The listener
measures with `snap_to_bars=True`, so the tempo-confidence change (which only affects the
non-snapping path the bridge uses) cannot alter a stored row.

Backs the database up first, runs in one transaction, and is idempotent — a second run
finds nothing to do.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "listener"))
from shared_dsp import MIN_KEY_MARGIN, KEY_NAMES_MAJOR, KEY_NAMES_MINOR  # noqa: E402

#: What the old all-sharp table produced, in pitch-class order, so a stored name can be
#: turned back into a pitch class and re-spelled properly.
OLD_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DEFAULT_DB = os.path.join(os.path.expanduser("~"), ".ai-bridge", "sound_index.db")


def renames() -> dict:
    """{(old_name, scale): new_name} for every spelling that actually changes."""
    out = {}
    for pc, old in enumerate(OLD_NAMES):
        for scale, table in (("major", KEY_NAMES_MAJOR), ("minor", KEY_NAMES_MINOR)):
            new = table[pc]
            if new != old:
                out[(old, scale)] = new
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and write nothing")
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"no database at {args.db}", file=sys.stderr)
        return 2

    con = sqlite3.connect(args.db)
    keyed = con.execute(
        "SELECT COUNT(*) FROM properties WHERE key IS NOT NULL AND key <> ''").fetchone()[0]
    weak = con.execute(
        "SELECT COUNT(*) FROM properties WHERE key IS NOT NULL AND key_strength < ?",
        (MIN_KEY_MARGIN,)).fetchone()[0]
    mapping = renames()
    respell = {}
    for (old, scale), new in mapping.items():
        n = con.execute(
            "SELECT COUNT(*) FROM properties WHERE key = ? AND scale = ? "
            "AND key_strength >= ?", (old, scale, MIN_KEY_MARGIN)).fetchone()[0]
        if n:
            respell[(old, scale)] = (new, n)

    print(f"database {args.db}")
    print(f"  files with a key      {keyed}")
    print(f"  below margin {MIN_KEY_MARGIN}     {weak}  "
          f"({100 * weak / max(keyed, 1):.1f}% — these keys were arbitrary)")
    for (old, scale), (new, n) in sorted(respell.items()):
        print(f"  respell {old:>2} {scale:<5} -> {new:<2}  {n} rows")
    total_respell = sum(n for _, n in respell.values())
    print(f"  totals: {weak} cleared, {total_respell} respelled")

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0
    if not weak and not total_respell:
        print("\nnothing to do (already migrated)")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"{args.db}.{stamp}.bak"
    shutil.copy2(args.db, backup)
    print(f"\nbacked up -> {backup}")

    with con:
        # Clear FIRST, then respell: a cleared row must not be respelled on the way out,
        # and doing it in this order means the respell counts above stay true.
        con.execute("UPDATE properties SET key = NULL, scale = NULL, key_strength = NULL "
                    "WHERE key IS NOT NULL AND key_strength < ?", (MIN_KEY_MARGIN,))
        for (old, scale), (new, _n) in respell.items():
            con.execute("UPDATE properties SET key = ? WHERE key = ? AND scale = ?",
                        (new, old, scale))

    after_keyed = con.execute(
        "SELECT COUNT(*) FROM properties WHERE key IS NOT NULL AND key <> ''").fetchone()[0]
    after_weak = con.execute(
        "SELECT COUNT(*) FROM properties WHERE key IS NOT NULL AND key_strength < ?",
        (MIN_KEY_MARGIN,)).fetchone()[0]
    leftover = con.execute(
        "SELECT COUNT(*) FROM properties WHERE key IN ('C#','D#','G#','A#') "
        "AND scale = 'major'").fetchone()[0]
    print(f"\nafter: {after_keyed} keyed ({keyed - after_keyed} removed), "
          f"{after_weak} below margin, {leftover} theoretical major spellings left")
    return 0 if after_weak == 0 and leftover == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
