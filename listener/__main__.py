"""CLI: python -m listener [--folder DIR | --live-db] [--limit N] [--no-tag]"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import scan
from .db import verify_against_bridge
from .run import run

DEFAULT_DB = Path.home() / ".ai-bridge" / "sound_index.db"
LIVE_DB = (Path(os.environ.get("LOCALAPPDATA", "")) / "Ableton" / "Live Database"
           / "Live-files-12300.db")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="listener", description=__doc__)
    p.add_argument("--folder", type=Path, action="append", dest="folders",
                   help="scan this folder instead of Live's index (repeatable)")
    p.add_argument("--exclude", action="append", default=[], metavar="FRAGMENT",
                   help="skip paths containing this (case-insensitive); adds to the "
                        "built-in wavetable exclusions")
    p.add_argument("--live-db", type=Path, default=LIVE_DB,
                   help="Live's analysis database (the default file list)")
    p.add_argument("--out", type=Path, default=DEFAULT_DB, help="sidecar database to write")
    p.add_argument("--limit", type=int, help="stop after N candidates (for trying it out)")
    p.add_argument("--workers", type=int, help="decode processes (default: cores - 1)")
    p.add_argument("--no-tag", action="store_true",
                   help="decode only — measures the pipeline without the model")
    p.add_argument("--check-schema", type=Path, metavar="BRIDGE_SIDECAR_PY",
                   help="verify our DDL still matches the bridge's published one")
    args = p.parse_args(argv)

    if args.check_schema:
        problems = verify_against_bridge(args.check_schema)
        print("\n".join(problems) if problems else "schema matches the bridge")
        return 1 if problems else 0

    if args.folders:
        extra = tuple(e.lower() for e in args.exclude)
        candidates, seen = [], set()
        for folder in args.folders:
            found = scan.from_folder(folder, None, extra)
            fresh = [c for c in found if c.path not in seen]
            seen.update(c.path for c in fresh)
            candidates += fresh
            print(f"  {str(folder)[:58]:<60} {len(fresh):>6} files")
            if args.limit and len(candidates) >= args.limit:
                candidates = candidates[:args.limit]
                break
        source = f"{len(args.folders)} folder(s)"
    else:
        if not args.live_db.exists():
            print(f"Live database not found: {args.live_db}\n"
                  "Pass --folder to scan a directory instead.", file=sys.stderr)
            return 2
        candidates = scan.from_live_database(args.live_db, args.limit)
        source = f"Live index ({args.live_db.name})"

    print(f"source: {source}")
    if not candidates:
        print("no audio files found")
        return 0

    result = run(args.out, candidates, workers=args.workers, tag=not args.no_tag)
    print("\n" + "-" * 56)
    for k, v in result.items():
        print(f"  {k:<16} {v}")
    print(f"  database        {args.out}")
    if not args.no_tag:
        print("\n⚠️  tags are UNVALIDATED — the mel-spectrogram parameters have not "
              "yet been checked against Essentia's own implementation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
