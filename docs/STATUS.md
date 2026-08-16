# Where this stands, and what is open

**Purpose of this file:** the other docs record *findings*. None of them recorded **how
a session ended**, so the next session had to reconstruct it from git log, a SQLite
file, and eventually the raw transcript. **Update it at the end of a session, even when
everything is green** — "nothing is blocked" is itself worth writing down.

Last updated: **2026-08-16**, after a long review-and-fix session.

---

## State

| | |
|---|---|
| `ai-bridge-listener` | 28 tests passing (`test_features.py` 23, `test_db.py` 5) |
| `ai-bridge-for-ableton-live` | 125 tests passing |
| Index | **29,870 files, 1,364,217 tags, 1 failure, ZERO orphans** |
| Index analyzer | `mel2+feat4+…+25heads-2` |
| Code analyzer | **`feat5`** — see "the one outstanding re-scan" below |

**Measured accuracy** (filenames as ground truth, octave errors counted as ERRORS):

| | |
|---|---|
| Tempo within 2.5 BPM (n=546) | **69%** (was 57%; octave-out down from 22% to 10%) |
| Pitch class matches a filename note (n=3,671) | **75%** |
| Files with genuinely wide stereo | 8,992 |
| Files whose mono sum partially cancels | 1,490 |

## ⚠️ THE BUG THAT INVALIDATED EVERY PREVIOUS INDEX

`record()` took the file id from `cur.lastrowid` after `INSERT … ON CONFLICT DO
UPDATE`. On the UPDATE branch nothing is inserted, so lastrowid still held the last
`tags` rowid — in the millions, and non-zero, so the `if not file_id` fallback never
fired. **Every re-analysis wrote its tags, properties and embedding against a bogus
id**, while `files.analyzer` was stamped correctly — so the stale verdicts looked fresh
and `plan()` would skip them forever.

Measured before it was caught: **581,789 orphaned tags, 12,825 orphaned properties.**

It only bites on RE-analysis, so a first scan of an empty database is always clean.
That is why it survived several full runs. Two things now prevent it: the id is
resolved by `SELECT … WHERE path=?`, and **`PRAGMA foreign_keys` is ON** (SQLite
disables it per connection by default, so every `REFERENCES` clause was decoration).
`tests/test_db.py` pins it — write A, write B, re-write A, check A's tags are still A's.

## The one outstanding re-scan

The code is on **`feat5`**; the index is on **`feat4`**. The difference is the chroma
fix (a longer window, averaged per pitch class), so **stored `key`/`scale` values are
stale**. Everything else — tags, pitch, tempo, stereo, loudness — is current.

A full re-scan takes about 65 minutes. Nothing is blocked by it.

## ⚠️ How to run a FULL scan

Live's own index is **not** the whole library — it lists ~6,000 files where the library
holds 29,870, and scanning from it silently produces an index a fifth of the size. Pass
the folder roots explicitly:

```
python -m listener \
  --folder "C:/ProgramData/Arturia" --folder "C:/ProgramData/Akai" \
  --folder "C:/ProgramData/Ableton" --folder "<user home>/Documents" \
  --folder "<sample drive>/..."          # every pack root, one --folder each
```

Run it from the project virtualenv (`.ai-bridge/venv` under the user home) —
`onnxruntime` is not in the system Python. ~7.7 files/s with measurement.

## Disk

Three databases are on disk under `~/.ai-bridge/`. Only the first is live:

| file | size | keep? |
|---|---|---|
| `sound_index.db` | 472 MB | **yes** — the current, clean index |
| `sound_index.corrupt-lastrowid.db` | 674 MB | evidence for the bug above; safe to delete |
| `sound_index.v1-backup.db` | 659 MB | pre-`properties` backup; safe to delete |

## The recommendation worth acting on

**The listener and the bridge measure the same things twice, and hand-porting between
them does not work.** Three separate faults tonight existed in both programs at once,
and one of them was a bug fixed in the listener at 01:00 and reproduced verbatim in the
bridge at 03:00 — with the explanatory comment copied across intact. Tempo, chroma, key,
onset detection, envelope and loudness all exist in two versions. They should share one
implementation.

## Open, not started

* **Phase A.5** — branch `phase-a5-tonal-routing`, four-way tonal routing (harmonic /
  chord / inharmonic). 9/9 synthetic, unreliable on real material (1 of 5 real chords
  routed right). The blocker is a per-file overlap: a real note scores 0.685 and a real
  chord 0.681.
* **Phase B** — the drum classifier on the stored embeddings. Everything it needs is in
  the database; nothing has been written.

## The review process

See `docs/REVIEW_LOG.md` in the bridge repo for who found what, and — importantly —
which confident claims were checked and **rejected**. Gemini's musical judgement found
things no measurement of mine would have; verification found the worst bug of the night
and rejected eight claims that would have damaged working code. Neither half is
sufficient alone.
