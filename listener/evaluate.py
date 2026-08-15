"""How good is the index, actually? Measured, not asserted.

The mel parameters cannot be proven identical to Essentia's without running Essentia,
which needs Linux. But that is a proxy for the question that matters — *are the tags
right?* — and that one can be answered here, at scale.

METHOD. Sample libraries are foldered and named by content: a file called
``Snares/Snare 08.wav`` is a snare. That is weak ground truth — a pack can mislabel,
"kick" appears in "Kickstart" — but across thousands of files the noise averages out
and a real signal shows through. Each rule below says: if the path claims X, at least
one of these labels should appear.

WHAT THIS CANNOT TELL YOU. It measures agreement with human naming, not correctness.
A systematically wrong mel that still separates snares from pads would score well
here. Read it as "is this useful?", never as "is the DSP right?".
"""
from __future__ import annotations

import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

DEFAULT_DB = Path.home() / ".ai-bridge" / "sound_index.db"

#: (name, path regex, acceptable labels, disqualifying labels).
#: The FIRST acceptable label must be the most specific one — strict mode uses only
#: that. Disqualifiers exist because labels are matched as substrings and some
#: contain others: "bass" is inside "Bass drum", so without excluding it a kick
#: tagged `Bass drum` would count as a correctly-identified bass.
RULES: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = [
    ("kick",        r"\bkick|\bbd\b|bass ?drum",  ("bass drum", "kick", "drum"), ()),
    ("snare",       r"\bsnare|\bsd\b|\brim",      ("snare", "drum", "rimshot", "percussion"), ()),
    ("hihat",       r"\bhi.?hat|\bhat\b|\bhh\b",  ("hi-hat", "cymbal", "drum"), ()),
    ("cymbal",      r"\bcrash|\bride\b|cymbal",   ("cymbal", "drum", "hi-hat"), ()),
    ("clap",        r"\bclap|\bsnap\b",           ("clap", "hands", "drum", "percussion"), ()),
    ("tom",         r"\btom\b|\btoms\b",          ("tom", "drum", "percussion"), ()),
    ("percussion",  r"\bperc\b|conga|bongo|shaker|tabla",
                    ("percussion", "drum", "tabla", "wood block", "mallet"), ()),
    ("vocal",       r"\bvocal|\bvox\b|\bchoir|\bvoice",
                    ("speech", "singing", "voice", "vocal", "choir", "chant"), ()),
    ("piano",       r"\bpiano|\bkeys\b|rhodes",   ("piano", "keyboard", "electric piano"), ()),
    ("guitar",      r"\bguitar|\bgtr\b",          ("guitar", "plucked", "string"), ()),
    ("bass",        r"\bbass\b(?! ?drum)",        ("bass guitar", "bass", "synthesizer"),
                    ("bass drum",)),
    ("strings",     r"\bstring|violin|cello",     ("string", "violin", "cello", "bowed"), ()),
    ("brass",       r"\bbrass|trumpet|\bhorn\b",  ("brass", "trumpet", "horn"), ()),
    ("noise",       r"\bnoise\b|\bwhite\b|\bpink\b", ("noise", "static", "pink noise",
                                                      "white noise", "hiss"), ()),
    ("rain",        r"\brain\b|storm|thunder",    ("rain", "thunder", "thunderstorm",
                                                   "water", "wind"), ()),
    ("bell",        r"\bbell\b|\bchime",          ("bell", "chime", "ding", "glockenspiel"), ()),
]


def evaluate(db_path: Path = DEFAULT_DB, min_confidence: float = 0.0,
             strict: bool = False) -> dict:
    """strict=True accepts ONLY each rule's first (most specific) label.

    The lenient reading credits a hi-hat tagged merely `drum`, which answers "did it
    land in the right family". Strict asks "did it name the thing". Both are worth
    knowing, and reporting only the flattering one would be a way of lying with a
    real measurement: a rule that accepts `synthesizer` as evidence of a bass scores
    100% while telling you nothing.
    """
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    labels_by_file: dict[int, set[str]] = defaultdict(set)
    for r in con.execute("SELECT file_id, label FROM tags WHERE confidence >= ?",
                         (float(min_confidence),)):
        labels_by_file[r["file_id"]].add(r["label"].lower())

    compiled = [(name, re.compile(pattern, re.I),
                 wants[:1] if strict else wants, blocked)
                for name, pattern, wants, blocked in RULES]
    hits: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    misses: dict[str, list[str]] = defaultdict(list)

    for row in con.execute("SELECT id, path FROM files WHERE error IS NULL"):
        base = os.path.basename(row["path"])
        parent = os.path.basename(os.path.dirname(row["path"]))
        hay = f"{parent} {base}"
        for name, pattern, wants, blocked in compiled:
            if not pattern.search(hay):
                continue
            total[name] += 1
            got = labels_by_file.get(row["id"], set())
            matched = any(w in label and not any(b in label for b in blocked)
                          for label in got for w in wants)
            if matched:
                hits[name] += 1
            elif len(misses[name]) < 3:
                misses[name].append(base[:44])
            break                      # first matching rule wins; no double counting

    rows = []
    for name, _p, _w, _b in compiled:
        if total[name]:
            rows.append((name, hits[name], total[name],
                         100.0 * hits[name] / total[name], misses[name]))
    overall_hits, overall_total = sum(hits.values()), sum(total.values())
    return {"rows": rows, "hits": overall_hits, "total": overall_total,
            "pct": 100.0 * overall_hits / overall_total if overall_total else 0.0}


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--misses", action="store_true", help="show example disagreements")
    args = p.parse_args(argv)

    lenient = evaluate(args.db, args.min_confidence, strict=False)
    strict = evaluate(args.db, args.min_confidence, strict=True)
    strict_by_name = {r[0]: r for r in strict["rows"]}

    print(f"{'category':<14}{'files':>7}{'named it':>10}{'right family':>14}")
    print("-" * 46)
    for name, hit, tot, pct, miss in sorted(lenient["rows"], key=lambda r: -r[3]):
        s = strict_by_name.get(name)
        s_pct = f"{s[3]:.0f}%" if s else "-"
        print(f"{name:<14}{tot:>7}{s_pct:>10}{pct:>13.0f}%")
        if args.misses and miss:
            for m in miss:
                print(f"                 miss: {m}")
    print("-" * 46)
    print(f"{'OVERALL':<14}{lenient['total']:>7}{strict['pct']:>9.0f}%"
          f"{lenient['pct']:>13.0f}%")
    print("\n'named it'     = the specific label (snare, hi-hat, violin)")
    print("'right family' = any acceptable label, e.g. a hi-hat tagged merely 'drum'")
    print("\nFilenames are weak ground truth: this measures agreement with human "
          "naming,\nnot DSP correctness. Read it as 'is this useful', not 'is the mel "
          "right'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
