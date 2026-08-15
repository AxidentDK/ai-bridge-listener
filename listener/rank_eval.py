"""Rank-based evaluation — how WELL is a sound identified, not merely whether it is.

`evaluate.py` asks "does an acceptable label appear anywhere among this file's ~130
tags?". With 15 audio events plus 25 heads, almost anything relevant appears
somewhere, so that metric saturates: it detects breakage but cannot rank two systems
that both basically work. It scored a corrected spectrogram and a wrong one
identically, to the unit.

This asks the sharper question. For `Snare 08.aif` the correct label went from
`Drum` ranked 1st at 0.96 to `Drum machine` ranked 2nd at 0.15 — a real difference in
how usable the result is, which a presence test scores as a tie.

METRICS

    MRR      mean reciprocal rank of the best acceptable label (1.0 = always first)
    hit@1    fraction where an acceptable label ranks first
    hit@3    ...within the top three
    conf     mean confidence of that label
    found    fraction with any acceptable label at all — i.e. the old metric

RANKING IS WITHIN A NAMESPACE, never across. Confidences are only comparable inside
one head: the two-class softmax heads saturate near 1.0 while multi-label sigmoid
heads sit at 0.3, so a global ranking would put `non_party 0.99` above `Snare drum
0.4` on every file and measure nothing. Each namespace is ranked separately and the
best position across them is taken.
"""
from __future__ import annotations

import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from .evaluate import DEFAULT_DB, RULES

#: AudioSet is an ontology, and these are its interior nodes. They are TRUE of almost
#: any musical sample and identify nothing — "a snare is Music" tells you what you
#: already knew. Measured over 1,647 drum-named files: `Music` outranks the specific
#: answer on 81% of them and is the top label on 70%. Suppressing them is what turns
#: hit@3 ~85% into hit@1.
GENERIC_EVENTS = {
    "music", "musical instrument", "sound effect", "silence", "speech",
    "inside, small room", "inside, large room or hall", "outside, urban or manmade",
    "outside, rural or natural", "electronic music", "cacophony", "noise",
    "environmental noise", "sound reproduction", "background music", "soundtrack music",
}


def _tags_by_namespace(con, min_confidence: float,
                       drop_generic: bool = False) -> dict[int, dict[str, list]]:
    """{file_id: {namespace: [(label, confidence), ...]}}, best confidence first."""
    per_file: dict[int, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list))
    for fid, ns, label, conf in con.execute(
            "SELECT file_id, namespace, label, confidence FROM tags "
            "WHERE confidence >= ? ORDER BY confidence DESC", (float(min_confidence),)):
        if drop_generic and label.lower() in GENERIC_EVENTS:
            continue
        per_file[fid][ns].append((label, conf))
    return per_file


def rank_report(db_path: Path = DEFAULT_DB, min_confidence: float = 0.0,
                drop_generic: bool = False) -> dict:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    tags = _tags_by_namespace(con, min_confidence, drop_generic)
    compiled = [(name, re.compile(pat, re.I), wants, blocked)
                for name, pat, wants, blocked in RULES]

    stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "found": 0, "rr": 0.0, "hit1": 0, "hit3": 0, "conf": 0.0})

    for fid, path in con.execute("SELECT id, path FROM files WHERE error IS NULL"):
        hay = f"{os.path.basename(os.path.dirname(path))} {os.path.basename(path)}"
        for name, pattern, wants, blocked in compiled:
            if not pattern.search(hay):
                continue
            s = stats[name]
            s["n"] += 1
            best_rank, best_conf = None, 0.0
            for _ns, labelled in tags.get(fid, {}).items():
                # labelled is already ordered by confidence within this namespace
                for rank, (label, conf) in enumerate(labelled, start=1):
                    low = label.lower()
                    if any(b in low for b in blocked):
                        continue
                    if any(w in low for w in wants):
                        if best_rank is None or rank < best_rank:
                            best_rank, best_conf = rank, conf
                        break            # best position in THIS namespace
            if best_rank is not None:
                s["found"] += 1
                s["rr"] += 1.0 / best_rank
                s["hit1"] += best_rank == 1
                s["hit3"] += best_rank <= 3
                s["conf"] += best_conf
            break                        # first matching rule wins

    rows = []
    for name, _p, _w, _b in compiled:
        s = stats.get(name)
        if not s or not s["n"]:
            continue
        n, found = s["n"], s["found"]
        rows.append({
            "category": name, "n": n,
            "found": found / n,
            "mrr": s["rr"] / n,                       # over ALL files, not just found
            "hit1": s["hit1"] / n,
            "hit3": s["hit3"] / n,
            "conf": (s["conf"] / found) if found else 0.0,
        })
    tot = sum(r["n"] for r in rows) or 1
    overall = {k: sum(r[k] * r["n"] for r in rows) / tot
               for k in ("found", "mrr", "hit1", "hit3")}
    overall["conf"] = (sum(r["conf"] * r["n"] * r["found"] for r in rows)
                       / max(sum(r["n"] * r["found"] for r in rows), 1e-9))
    return {"rows": rows, "overall": overall, "n": tot}


def _print(title: str, res: dict) -> None:
    print(f"\n{title}")
    print(f"  {'category':<13}{'n':>6}{'MRR':>8}{'hit@1':>8}{'hit@3':>8}"
          f"{'conf':>8}{'found':>8}")
    print("  " + "-" * 57)
    for r in sorted(res["rows"], key=lambda r: -r["mrr"]):
        print(f"  {r['category']:<13}{r['n']:>6}{r['mrr']:>8.3f}{r['hit1']:>8.1%}"
              f"{r['hit3']:>8.1%}{r['conf']:>8.3f}{r['found']:>8.1%}")
    o = res["overall"]
    print("  " + "-" * 57)
    print(f"  {'OVERALL':<13}{res['n']:>6}{o['mrr']:>8.3f}{o['hit1']:>8.1%}"
          f"{o['hit3']:>8.1%}{o['conf']:>8.3f}{o['found']:>8.1%}")


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--compare", type=Path, help="second database to compare against")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--drop-generic", action="store_true",
                   help="ignore AudioSet's interior nodes (Music, Sound effect, ...)")
    args = p.parse_args(argv)

    a = rank_report(args.db, args.min_confidence, args.drop_generic)
    _print(f"A  {args.db.name}" + ("  [generic dropped]" if args.drop_generic else ""), a)
    if not args.compare:
        print("\nMRR 1.0 = the correct label always ranks first within its head.")
        print("'found' is the old presence metric, shown for reference.")
        return 0

    b = rank_report(args.compare, args.min_confidence)
    _print(f"B  {args.compare.name}", b)

    by_name = {r["category"]: r for r in b["rows"]}
    print(f"\nDIFFERENCE (B - A)\n  {'category':<13}{'dMRR':>9}{'dhit@1':>9}"
          f"{'dconf':>9}{'dfound':>9}")
    print("  " + "-" * 49)
    for r in sorted(a["rows"], key=lambda r: r["category"]):
        o = by_name.get(r["category"])
        if not o:
            continue
        print(f"  {r['category']:<13}{o['mrr'] - r['mrr']:>+9.3f}"
              f"{o['hit1'] - r['hit1']:>+9.1%}{o['conf'] - r['conf']:>+9.3f}"
              f"{o['found'] - r['found']:>+9.1%}")
    oa, ob = a["overall"], b["overall"]
    print("  " + "-" * 49)
    print(f"  {'OVERALL':<13}{ob['mrr'] - oa['mrr']:>+9.3f}"
          f"{ob['hit1'] - oa['hit1']:>+9.1%}{ob['conf'] - oa['conf']:>+9.3f}"
          f"{ob['found'] - oa['found']:>+9.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
