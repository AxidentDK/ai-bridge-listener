# Where this stands, and what is open

**Purpose of this file:** the other docs record *findings* — what was learned, what was
fixed, what not to re-chase. None of them recorded **how a session ended**, so the next
session had to reconstruct it from git log, a SQLite file, and eventually the raw
transcript. That is the gap this file closes. **Update it at the end of a session, even
when everything is green** — "nothing is blocked" is itself worth writing down.

Last updated: **2026-08-15** (late).

---

## State

| | |
|---|---|
| `ai-bridge-for-ableton-live` | 60 tools, **103 tests passing**, schema v3 |
| `ai-bridge-listener` | **17 tests passing**, analyzer `mel2+feat2+…` |
| spectrograms | verified against Essentia's own — `MEL_VALIDATION.md` |
| tag quality | hit@1 **57.2%** after demoting AudioSet's interior nodes |

## ⚠️ How to run a FULL scan — this cost real time to rediscover

Live's own index is **not** the whole library. It listed 5,954 files where the original
scan had 22,100, and a re-scan from it silently produced a much smaller index. The
folder roots have to be passed explicitly:

```
python -m listener \
  --folder "C:/ProgramData/Arturia" \
  --folder "C:/ProgramData/Akai" \
  --folder "C:/ProgramData/Ableton" \
  --folder "<user home>/Documents" \
  --folder "<sample drive>/..."          # every pack root, one --folder each
```

Plug in whichever roots hold the packs — plugin factory content under `ProgramData`,
the user's own `Documents`, and any sample drive. The point is that **they must be
listed**; Live's index will not supply them.

Run it from the project's virtualenv (`.ai-bridge/venv` under the user home) —
`onnxruntime` is not in the system Python. Roughly 11 files/s with measurement, so
~45 minutes for a library this size.
It is resumable: files whose size, mtime and **analyzer version** are unchanged are
skipped, so an interrupted scan can simply be re-run.

## Phase A — measured features, DONE and validated

`properties` shipped in schema v1 and had **zero rows**. It now carries, per file:
kind (one-shot/loop by onset density), onsets, bpm + confidence + **bars**, key/scale/
strength for loops, pitch as Hz/MIDI/confidence for one-shots, attack, T60 decay,
stereo width and correlation, and BS.1770-4 loudness. Embeddings are stored as float16
so a classifier can be trained without decoding the library again.

**Validated against filenames, on 28,350 files indexed:**

| | |
|---|---|
| Tempo within 2.5 BPM of a filename-stated BPM (n=546) | **69%** (was 57%) |
| Pitch class matching a note named in the filename (n=3,671) | **74%** |
| Files with genuinely wide stereo (>0.2) | 8,992 |
| Files whose mono sum partially cancels (correlation < 0) | 1,490 |

**Octave errors are counted as errors**, deliberately: 85 and 170 BPM are *not* the same
tempo, because the same pulse written at each differs in note values — the grid, the
swing and every quantise decision differ with it. Kim made that correction; an earlier
"86% allowing an octave" figure was flattering the result.

**Two things about tempo that are easy to get wrong and cost a day between them:**

1. **Bar-fitting does NOT resolve the octave.** 4 bars at 100 BPM is also exactly 8 bars
   at 200. Duration constrains the tempo to a power-of-two family and then says nothing
   about which member. It removes the *non*-octave errors, which is a different job.
2. **Events per beat is what breaks the tie**, and its anchor belongs to the DETECTOR.
   Measured at the true tempo: median 2.10 events/beat, doubling to 4.19 if read at
   half-time. Anchor ~1.5 sloping to ~2.2 at slow tempos. ⚠️ A 16th-note groove has four
   *musical* events per beat — `onset_times` counts only what clears
   `_FLUX_MIN_FRACTION`, so it counts structural impacts. **Change that threshold and
   these numbers must be re-measured**, or the calibration silently becomes a
   coincidence.

Confidence is prominence × margin over the nearest metrical rival. The previous version
measured how *rhythmic* a loop was and was anti-correlated with correctness (53% right
above 0.9 confidence, 65% below it), because a perfectly quantised loop has a perfect
half-time alias.

## The open thread: Phase A.5, on a branch

**Branch `phase-a5-tonal-routing`, deliberately NOT merged.** It splits a tonal
one-shot three ways — harmonic / chord / inharmonic — with chord root and quality from
the unfolded partials.

* **Synthetic: 9/9**, including a *missing fundamental* named correctly.
* **Real material: not good enough.** Of five files Ableton names "Chord", one routes
  as a chord and three land in *inharmonic*.
* **Why it is hard, so nobody re-derives it:** the group separation (Cohen d 1.26)
  hides a per-file overlap exactly where the decision is made — `Bass Guitar Note F`
  scores 0.685 and `Guitar Chord Jazz CMaj9` scores 0.681. A real note and a real
  chord, four thousandths apart.
* **Still to do:** the columns (`tonality`, `chord_root`, `chord_quality`,
  `harmonic_fit`) do not exist in either repo, so nothing is persisted yet.

**Do not re-chase the chroma-density gate.** It was the agreed design and it measures
*backwards*: real single notes come out denser than real chords. Harmonic 3 of a note
is its fifth, harmonic 5 its major third, harmonic 7 its minor seventh — a raw sawtooth
*is* a dominant seventh in chroma space. Folding destroys the distinction before any
threshold sees it.

## Phase B — not started

The drum classifier on the stored embeddings, labelled by folder ∩ filename with
contradictions dropped. It closes the vocabulary hole AudioSet cannot fill (no `kick`
class, no `tom` class). Everything it needs is already in the database.

## The peer review

Gemini (Pro, in the **built-in** browser — never the external Chrome) has reviewed the
architecture with the full picture of both programs. Its contributions that changed the
design: routing-first ordering, width measured above 250 Hz, the YIN guards for 808s,
time-to-peak as a *decision-making* tag, and the four-way tonal split. It has been
wrong twice — assuming CLAP/Essentia at runtime, and the chroma gate — and took both
corrections. See `PLAN.md`.

## ⚠️ The trap that has now been set three times

**Bump the analyzer version when the analysis changes.** If it does not change, a
re-scan compares identical version strings, skips every file, and reports success
having re-analysed none. `decode.MEL_VERSION` and `features.FEATURE_VERSION` both feed
it. Adding a *column* additionally needs `SCHEMA_VERSION` bumped in **both** repos —
`CREATE TABLE IF NOT EXISTS` will not add it, and the run would otherwise stamp the new
version onto the old tables.
