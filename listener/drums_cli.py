"""CLI for the drum classifier: `python -m listener.drums_cli --train|--apply`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import drums

DEFAULT_DB = Path.home() / ".ai-bridge" / "sound_index.db"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--train", action="store_true",
                   help="fit on filename-derived labels and report held-out accuracy")
    p.add_argument("--apply", action="store_true",
                   help="write `drum` tags for the whole index (no audio is decoded)")
    p.add_argument("--min-confidence", type=float, default=drums.MIN_CONFIDENCE)
    args = p.parse_args(argv)

    if not args.db.exists():
        print(f"no index at {args.db}", file=sys.stderr)
        return 2
    if not (args.train or args.apply):
        p.error("give --train, --apply, or both")

    model = None
    if args.train:
        X, y, _ = drums.training_set(args.db)
        if not len(X):
            print("no labelled files found — is the index populated?", file=sys.stderr)
            return 1
        counts = {c: int((y == i).sum()) for i, c in enumerate(drums.CLASSES)}
        print(f"{len(X):,} labelled files")
        print("  " + "  ".join(f"{c}:{n}" for c, n in counts.items()))

        report = drums.evaluate(X, y)
        print(f"\nheld-out accuracy: {100 * report['overall']:.1f}%  "
              f"(train {report['n_train']:,} / test {report['n_test']:,})")
        for name, (acc, n) in sorted(report["per_class"].items(),
                                     key=lambda kv: -kv[1][0]):
            print(f"  {name:8} {100 * acc:5.0f}%   n={n}")
        if report["at_high_confidence"] is not None:
            print(f"  at confidence >= 0.9: "
                  f"{100 * report['at_high_confidence']:.0f}% correct")

        # The shipped model is trained on EVERYTHING, including the held-out fifth —
        # the split exists to estimate accuracy honestly, not to throw data away.
        model = drums.fit(X, y)
        drums.save(model)
        print(f"\nmodel saved to {drums.MODEL_PATH}")

    if args.apply:
        model = model or drums.load()
        if model is None:
            print("no trained model — run --train first", file=sys.stderr)
            return 1
        result = drums.apply_to_index(args.db, model, args.min_confidence)
        print(f"\ntagged {result['tagged']:,} files as drums")
        print(f"  {result['passed_gate']:,} of {result['in_index']:,} files passed the "
              f"gate — a one-shot AND percussive per AudioSet")
        print(f"  {result['skipped_too_long']:,} of those were still too long")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
