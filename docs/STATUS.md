# Where this stands, and what is open

**Purpose of this file:** the other docs record *findings*. None of them recorded **how
a session ended**, so the next session had to reconstruct it from git log, a SQLite
file, and eventually the raw transcript. **Update it at the end of a session, even when
everything is green** — "nothing is blocked" is itself worth writing down.

Last updated: **2026-08-19, small hours** — key spelling, a key-margin gate, and an index
migration. (Before that: 2026-08-16, the shared-DSP extraction, the drum classifier, a
conversational review with Gemini, and a drum-tag refresh.)

**Nothing is running. Nothing was re-scanned, and nothing needs to be.** `FEATURE_VERSION`
is still `feat7` and the model outputs are unchanged; what changed is what gets DERIVED
from them, and both corrections were derivable from stored columns.

## 2026-08-19 — the key estimator was storing coin-flips

Found from the bridge side, by rendering a piece of music with a known key and measuring
the render (see the bridge's `docs/LIVE_TEST_PLAN.md`).

**Three fixes in `shared_dsp.py`** — edited here, copied to the bridge, `EXPECTED_SHA256`
updated in both. New hash `d8fb9dee735b72eba3392385167ceec10af1e5bf1cf087b7b35355a3b7ac5071`.

1. **`MIN_KEY_MARGIN = 0.05`.** `key_from_chroma` had NO margin gate: it stored a key
   whenever one merely correlated best. A drum-rack preview scored G minor 0.2428 against
   F♯ minor 0.1944 — four candidates bunched between 0.13 and 0.24 — and the winner flipped
   between G and E depending on which resampler produced the signal. **27.5% of every key in
   the index was of that kind.** The threshold is not new: it is the value the bridge already
   used to call a key "ambiguous", now shared instead of duplicated.
2. **`key_name(root, mode)`.** `NOTE_NAMES` is all sharps — right for a pitch, wrong for a
   key. The index held `D# major`, a theoretical key of nine sharps nobody writes. Two
   tables, because the convention differs by mode: pitch class 8 is A♭ major but G♯ minor.
3. **`relative_key()`**, so a consumer can name the relative — a key and its relative share
   every pitch class and chroma cannot separate them even in principle.

**The migration, not a re-scan:** `scripts/migrate_key_margin.py`. The margin was already
stored as `key_strength` and the spelling is a pure rename, so the whole correction is
derivable from the database. Re-running the analyzer over 29,870 files to recompute values
that cannot change would cost hours and carry the one risk this project has actually been
bitten by. Applied 2026-08-19:

| | |
|---|---|
| keys before | 7,997 |
| cleared (margin < 0.05) | **2,198 (27.5%)** |
| respelled | 481 |
| keys after | **5,799** (4,173 major, 1,626 minor), min strength exactly 0.05 |
| backup | `sound_index.db.20260819-005856.bak` |

Idempotent — a second run reports nothing to do.

✅ **The long-standing red test is fixed, and it was the TEST that was wrong.**
`test_measure_reproduces_analyze_field_for_field` had been failing on the key assertion;
fixing keys exposed an older failure underneath — `onsets: 15 vs 18` on
`Perc Kitchen Kit.adg.ogg`.

The cause: the test built `Prepared` from the mel pass's soxr mono while `analyze`
re-derives its own through `prepare`. `analyze` accepts `mono` and **ignores it on
purpose** — that is the whole point of the shared core, both programs measuring one signal
rather than each picking a resampler. So the test was comparing two DIFFERENT signals and
calling it pass-through, and on a percussive file three detections landed either side of a
threshold. It now builds the signal the same way `analyze` does, which is what its own
docstring always said it was checking. How far the two resamplers diverge is a real
question and already has its own test
(`test_owning_the_resampling_moves_the_numbers_only_slightly`, green).

**Both suites are fully green: listener 58, bridge 154.**

## 📌 The octave error is the PRIOR, and the audit proves it

Measured 2026-08-19 over the 546 files whose filenames state a tempo (ground truth: the
people who cut the loops). Overall 68.9% exact, **9.7% octave-out** — which matches the
recorded 10% and hides everything interesting. Broken out by tempo it is not noise at all:

| stated tempo | files | octave-out | rate |
|---|---|---|---|
| 0–90 | 73 | 22 | **30.1%** |
| 90–110 | 147 | 1 | 0.7% |
| 110–130 | 218 | 7 | 3.2% |
| 130–150 | 76 | 9 | 11.8% |
| **150–175** | 28 | 12 | **42.9%** |
| 175+ | 4 | 2 | 50.0% |

**The direction has NO exceptions.** Slow files are doubled (23 of 23); mid and fast files
are halved (30 of 30). Every octave error moves the tempo TOWARD 120 BPM — the prior's
centre. The autocorrelation is not confused; the prior is deciding, and it decides home.

**Two distinct failure modes, from the same cause:**

- **Below ~110 the prior ACTIVELY pushes up.** 75 BPM sits 0.678 octaves from the centre
  and its double 150 sits only 0.322 — the prior prefers the wrong answer outright.
- **At ~170 the prior is NEUTRAL and nothing else votes.** 85 is −0.497 octaves from 120
  and 170 is +0.503: equidistant. With the prior abstaining, the ACF's half-time alias
  wins by default. Bar-fit cannot help either — `Drumloop 12 170BPM.wav` is 2.82 s, which
  is 2 bars at 170 *and* 1 bar at 85, both whole (see `bar_fit`'s own warning).

**This is the fix `_BPM_PRIOR_WIDTH` already names**: *"The real fix is for the
events-per-beat term to carry the octave decision so the prior can be flattened; it is not
strong enough to do that alone yet."* `_EPB_WIDTH = 0.9` octaves is close to flat, which is
why it cannot. The experiment is a two-parameter sweep of (`_EPB_WIDTH`, `_BPM_PRIOR_WIDTH`)
scored **per band on the worst band**, not on the mean — judging on the mean is exactly what
let a 36% collapse on fast material hide behind a headline that barely moved.

### The sweep was run, 2026-08-19. No parameter setting fixes this.

Method: decode the 546 ground-truth files once and cache onsets + flux (they do not depend
on either parameter), then replay `tempo()` across the grid — 546 decodes instead of 546 x
36, 33 seconds of decoding. **Harness check first: it reproduces the stored BPM on 546/546
(100%)**, so the numbers describe the shipped detector and not the scaffolding.

`_EPB_WIDTH` fixed at 0.90, sweeping `_BPM_PRIOR_WIDTH`:

| prior | exact | octave | worst-band exact | worst-band octave | octave per band |
|---|---|---|---|---|---|
| **0.45 (shipped)** | 69.4% | 9.4% | 43% | **43%** | 30, 1, 3, 12, **43** |
| 0.50 | 69.4% | 9.2% | 50% | 36% | 26, 1, 4, 14, 36 |
| **0.60** | 68.8% | 9.6% | **56%** | **29%** | 23, 1, 6, 18, 29 |
| 0.80 | 64.0% | 14.4% | 55% | 29% | 22, 1, 15, 26, 29 |

*(bands 0-90, 90-110, 110-130, 130-150, 150-175)*

**TWO RESULTS, AND THE SECOND IS THE IMPORTANT ONE.**

**1. The comment's proposed fix does not work.** `_BPM_PRIOR_WIDTH` says *"the real fix is
for the events-per-beat term to carry the octave decision so the prior can be flattened"*.
Swept, tightening EPB makes things WORSE in every band — `epb=0.25` scores 61.9% overall
against 68.9%. It cannot carry the decision, and this is now measured rather than hoped.

**2. Widening the prior redistributes the damage; it does not reduce it.** 0.60 halves the
worst band's octave rate (43% → 29%) and lifts its exactness (43% → 56%) — but the OVERALL
octave rate goes 9.4% → 9.6%. The fast band gains about four files and the two mid bands
lose about eleven. The prior does not decide how often the octave is wrong, only **which
music pays for it**. Today that is drum-and-bass and slow material, so mid-tempo does not.

**Recommendation: leave `_BPM_PRIOR_WIDTH` at 0.45 for now.** Unlike the key fixes, changing
it CHANGES STORED BPM VALUES, so it needs a full re-scan to reach the index — and a re-scan
should be spent once, on a discriminator that MEASURES the octave instead of assuming it.
Trading 0.6% overall exactness to move errors sideways does not justify the hours.

### "Resolve tempo by counting bars" — the right principle, and two dead ends

Kim's standing rule is *"tempo octaves are NOT equal; resolve tempo by counting bars."*
Both halves of that were tested on the 546 cached files, and **both failed**. Recorded so
neither is retried:

- **Bar counting as a primary ESTIMATOR** — autocorrelate the onset envelope, take the
  strongest repeat inside the range a bar could occupy (1.26–4.0 s), call it the bar,
  derive the tempo. Recovers **46%** of the octave errors and only **55%** of the files
  the shipped detector already gets right. It breaks on patterns that repeat at a half
  bar or across two (Disco 115 → 152.6, Funk 100 → 133.0).
- **Bar counting as an octave ARBITER** — keep the detector's pulse, and between T and 2T
  prefer the one whose bar correlates better than its own half bar, on the reasoning that
  a real four-beat bar does not repeat at half its length. **Fixes 25 of 54 octave errors
  and breaks 56 of 376 correct ones — net −31 files, 68.9% → 63.2%.**

⚠️ **What this does NOT show.** Both tests used a crude bar detector: raw autocorrelation
of a full-band onset envelope. That is not downbeat tracking. A real downbeat model —
per-band flux, or a beat tracker carrying an explicit bar-level state — is a different and
much stronger thing, and the principle may well hold once it is one. What is now measured
is that **"count the bars" is not a five-line addition to the existing envelope**, which is
what both of these were.

**What a real fix looks like** — none of it needs the prior:
- **Bass-band periodicity.** A 170 BPM DnB loop and its 85 BPM reading differ in where the
  KICK falls, and the sub band carries that cleanly. The full-band flux does not.
- **Spectral change per beat.** Half-time doubles the harmonic events per beat; a chroma or
  band-energy change rate is a different measurement from onset density and is not fooled
  by the same things.
- **Onset-interval bimodality.** A halved reading leaves a strong secondary peak at half the
  chosen interval; its presence is evidence, and it is already computable from the ACF that
  `_metrical_margin` looks at.

The sweep scripts are in the session scratchpad; the cache builder is ~30 lines and worth
re-creating rather than preserving.

If you are picking this up cold, read `docs/PROGRESS.md`'s top two entries and the "Open,
not started" list below and you are current.

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
