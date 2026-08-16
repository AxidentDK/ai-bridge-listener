# Where this stands, and what is open

**Purpose of this file:** the other docs record *findings*. None of them recorded **how
a session ended**, so the next session had to reconstruct it from git log, a SQLite
file, and eventually the raw transcript. **Update it at the end of a session, even when
everything is green** — "nothing is blocked" is itself worth writing down.

Last updated: **2026-08-16, end of day** — after the shared-DSP extraction, the drum
classifier, a full conversational review with Gemini, and a drum-tag refresh.

**Nothing is running, nothing is blocked, both repos are green and pushed.** The one
deliberate loose end is the re-scan, described below. If you are picking this up cold,
read that section and `docs/PROGRESS.md`'s top two entries and you are current.

---

## State

| | |
|---|---|
| `ai-bridge-listener` | **56 passing** — features 19, shared_dsp 18, drums 11, db 5, sync 3 |
| `ai-bridge-for-ableton-live` | **133 tests passing**, incl. the same 3 sync tests |
| Index | **29,870 files, 1,369,646 tags, 29,869 properties, ZERO orphans** |
| Index analyzer | `mel2+feat5+discogs-effnet-1+25heads-2+yamnet-1(k15,f0.02)` |
| Code analyzer | **`feat7`** — see "the one outstanding re-scan" below |
| DSP core | `shared_dsp.py`, byte-identical in both repos, SHA-256 checked by both suites |
| Drum labels | **6,712** files in the `drum` namespace, refreshed 2026-08-16 after the cymbal fix |

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

## The re-scan — STARTED 2026-08-16 ~16:45, running

Kim's instruction was *"build but wait doing the rescan"* through the afternoon; he
released it at the end of the day. Verified running: the analyzer string is being
written as `mel2+feat7+...`, which is the thing that must change or a re-scan compares
identical strings, skips every file and **reports success without re-analysing one**.

**A backup was taken first: `~/.ai-bridge/sound_index.pre-feat7.db`**, verified
byte-identical by SHA-256 before the scan started. The scan rewrites properties and tags
in place, and the last time this table was rewritten it was silently corrupted for days
(see the lastrowid bug above), so a rollback point is cheap insurance rather than
ceremony.

The roots were derived FROM THE INDEX rather than from memory — a count of distinct
top-level paths over all 29,870 files — because taking the list from Live's own database
silently produces an index a fifth of the size:

    C:/ProgramData/Arturia            13,165
    <user home>/Documents              7,147   (Xfer, Synapse, u-he, NI, Vital…)
    C:/ProgramData/Ableton             6,399
    C:/ProgramData/Akai                  926
    <sample drive>/<loop packs>        1,675
    <sample drive>/<free packs>          543

*The section below describes what the scan is fixing; leave it until the scan finishes,
then reduce it to a line saying the index and code agree.*

The code is on **`feat7`**; the index was on **`feat5`**.

`feat6` changes less than the version bump suggests: **the maths is
identical** — proven bit-for-bit against `feat5` on 120 real files — but it now runs on
audio from the shared Kaiser resampler rather than soxr, and a few values move with the
samples (2 of 42 tempos by 0.1 BPM). The one substantive change is `attack_ms`, which
was unstable on sustained one-shots and is now stable; roughly 2% of one-shots have a
stored attack worth distrusting until the scan runs.

`feat7` is the one with real content: the chroma band now starts at A0 instead of A1,
which stops square and saturated basslines being named a perfect fifth above their root.
**Stored `key` values for bass material are wrong until the scan runs** — measured at
about 2% of real bass loops, but wrong in a musically severe way where it lands.

Everything else — tags, key, pitch, tempo, stereo, loudness, drum labels — is current
and correct. **Nothing is blocked by this.** A full re-scan takes about 65 minutes.

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
* **The `feat7` re-scan**, deferred deliberately — see above.
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
