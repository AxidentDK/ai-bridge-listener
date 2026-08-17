# Where this stands, and what is open

**Purpose of this file:** the other docs record *findings*. None of them recorded **how
a session ended**, so the next session had to reconstruct it from git log, a SQLite
file, and eventually the raw transcript. **Update it at the end of a session, even when
everything is green** — "nothing is blocked" is itself worth writing down.

Last updated: **2026-08-16, end of day** — after the shared-DSP extraction, the drum
classifier, a full conversational review with Gemini, and a drum-tag refresh.

**Nothing is running, nothing is blocked, both repos are green and pushed, and the
index now matches the code** — the `feat7` re-scan completed and verified clean. If you
are picking this up cold, read `docs/PROGRESS.md`'s top two entries and the "Open, not
started" list below and you are current.

---

## State

| | |
|---|---|
| `ai-bridge-listener` | **56 passing** — features 19, shared_dsp 18, drums 11, db 5, sync 3 |
| `ai-bridge-for-ableton-live` | **133 tests passing**, incl. the same 3 sync tests |
| Index | **29,870 files, 1,369,646 tags, 29,869 properties, ZERO orphans** |
| Index analyzer | `mel2+feat7+discogs-effnet-1+25heads-2+yamnet-1(k15,f0.02)` |
| Code analyzer | **`feat7`** — index matches; no re-scan outstanding |
| DSP core | `shared_dsp.py`, byte-identical in both repos, SHA-256 checked by both suites |
| Drum labels | **6,712** in the `drum` namespace — destroyed by the re-scan, restored, and now protected (`EXTERNAL_NAMESPACES`) |

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

## ✅ The re-scan — DONE 2026-08-16, index and code now agree

29,870 files in 3,523 s (**8.48 files/s**), 1 pre-existing failure, 0 skipped. The index
and the code are both on **`feat7`**; there is no outstanding re-scan.

**Verified afterwards rather than assumed**, because the last in-place rewrite of these
tables went silently corrupt for days:

| check | result |
|---|---|
| orphaned tags / properties / embeddings | **0 / 0 / 0** |
| properties per file | **1.0000** (was 1.43 when corrupt) |
| files not on `feat7` | **0** — one analyzer string across all 29,870 |
| errors | 1, unchanged and pre-existing |

That first row is the one that matters: the `lastrowid` bug bit only on RE-analysis,
which is precisely what just ran, so this is the first full re-analysis to prove the fix
under the condition that broke it.

**Did the bass-key fix reach stored data?** 380 of 7,950 keys changed (4.8%). The two
largest intervals are exactly the odd-harmonic corrections the fix predicts — a **major
third down (65)** and a **perfect fifth down (60)**, i.e. files that had been named after
their 5th and 3rd harmonics.

But read that honestly: the remaining ~255 changes are spread fairly evenly across other
intervals, and only 45 of the 380 changed files are named bass/808/sub. So lowering the
chroma floor to A0 does more than fix the square-wave pathology — it adds real
low-frequency content to every chroma and moves borderline cases generally. The
mechanism is supported, not proven exclusive, and 4.8% is more movement than the 2%
measured on bass loops alone.

### ⚠️ That scan's "verified" backup was CORRUPT — how to take one properly

The rollback point taken before the scan was a `Copy-Item` of `sound_index.db`, checked
by comparing SHA-256 of source and destination. **It was malformed** — `PRAGMA
integrity_check` on it reports "Page 24145: never used", and querying it returns
nonsense (a `GROUP BY namespace` yielded three separate `drum` groups).

The hash proved the copy matched the FILE. It could not prove the file was a complete
DATABASE. **The store runs in WAL mode** — deliberately, so the bridge can read during a
scan — and a copy of `sound_index.db` alone leaves the `-wal` behind, so it is a torn
snapshot of a database whose recent pages live in the other file.

**Take backups through SQLite, never with a file copy:**

```
sqlite3.connect("file:sound_index.db?mode=ro", uri=True).execute("VACUUM INTO ?", (dst,))
```

2.6 s for this index, and the result is consistent by construction. Then verify with
`PRAGMA integrity_check` — checking the CONTENT, not the bytes.

Current good backup: **`~/.ai-bridge/sound_index.backup.db`** (453 MB, integrity_check
`ok`, 29,870 files / 1,370,929 tags / 6,712 drum labels). The corrupt one was deleted.

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

## ✅ DONE — the recommendation that was worth acting on

*Left here rather than deleted: it names the problem better than a changelog line, and
the failure mode is one to stay alert to.*

**The listener and the bridge measured the same things twice, and hand-porting between
them did not work.** Three separate faults existed in both programs at once, and one was
a bug fixed in the listener at 01:00 and reproduced verbatim in the bridge at 03:00 —
with the explanatory comment copied across intact.

Fixed on 2026-08-16. `shared_dsp.py` owns onsets, tempo, chroma, key scoring, flatness,
envelope, stereo and loudness; both repos hold the same file byte for byte and both test
suites fail on drift, by SHA-256 over the whole file rather than by substring. The
listener is the source of truth; the bridge's copy is verified, never edited. `prepare()`
is the only way in, so the two programs cannot diverge on pre-processing either — that
was the hole the obvious `(array, rate)` API would have left open.

**If you are about to copy a measurement fix between the repos: don't.** Change
`listener/shared_dsp.py`, copy the whole file, run both suites.

## Open, not started

* **Phase A.5** — branch `phase-a5-tonal-routing`, four-way tonal routing (harmonic /
  chord / inharmonic). 9/9 synthetic, unreliable on real material (1 of 5 real chords
  routed right). The blocker is a per-file overlap: a real note scores 0.685 and a real
  chord 0.681.
* **Agent tool-calling in Live** — the sidecar's search tools have never been exercised
  end to end from an agent inside a real session.
* **rim (44 ± 11) and shaker (35 ± 6)** in the drum classifier, left open ON PURPOSE:
  at 34 and 13 test files nothing proposed for them is falsifiable with this library.
  Not to be tuned until a pack brings the counts up.
* **The 3rd-harmonic story is verified on SYNTHESIS, not on the library.** 2% of real
  bass loops change key with the new floor, but there is no ground truth saying those
  changes are toward the truth. Gemini predicted the change-interval histogram would
  spike at a perfect fifth down; measured on 400 real loops it spikes at a **major
  third** (10 of 23) rather than a fifth (4). Both are odd-harmonic intervals so the
  mechanism survives, but the specific prediction failed and n is small.

## ✅ DONE — Phase B, the drum classifier

Softmax regression over the stored Discogs-EffNet embeddings, labelled from paths
(filename beats folder; contradictions dropped rather than guessed), covering the
distinctions AudioSet has no vocabulary for: kick vs tom vs rim, clap vs snare.

**Refreshed 2026-08-16** after the review found that the 3-second one-shot cap was
excluding 63.8% of rides and 47.7% of crashes — a crash rings 5-8 s and a ride longer.
Retraining with the cap at 15 s and loops filtered by `kind` rather than by duration:

| | before | after |
|---|---|---|
| ride | 131 | **432** |
| crash | 85 | **186** |
| tom | 636 | 841 |
| kick | 1,061 | 1,312 |
| perc | 1,326 | 1,743 |
| **total** | 5,429 | **6,712** |

Stable rather than churned: 5,151 files kept their label, 149 changed it, 1,412 are
newly tagged and 129 lost a tag. Of the newly tagged files whose NAME states a class,
99.4% agree with it — a weak check, since those names also feed training, but it does
establish the new tags are not noise. Held out: 80.0% overall, 93% at confidence >= 0.9.

Re-run any time with `python -m listener.drums_cli --train --apply`; it decodes no audio
and takes seconds.

The thing worth remembering: the first run labelled **13,697 files**, including synth
chords at 0.98 confidence. That is not a threshold to tune — a closed-set softmax must
pick a class and cannot answer "none of the above". Gating on AudioSet percussion events
plus the one-shot property is what makes the number above trustworthy.

## The review process

See `docs/REVIEW_LOG.md` in the bridge repo for who found what, and — importantly —
which confident claims were checked and **rejected**. Gemini's musical judgement found
things no measurement of mine would have; verification found the worst bug of the night
and rejected eight claims that would have damaged working code. Neither half is
sufficient alone.
