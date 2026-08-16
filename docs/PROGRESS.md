# Progress log

Chronological, append-only. Newest entry at the top.

**What goes where**, so these do not drift into repeating each other:

| document | question it answers | how it is maintained |
|---|---|---|
| `STATUS.md` | *Where are we right now? What is open?* | **rewritten** — always current, never long |
| `PROGRESS.md` (this file) | *What happened, and what did we learn?* | **appended** — never edited once written |
| git log | *What changed in this commit, and why?* | per commit |
| `MEL_VALIDATION.md`, `PLAN.md`, `REVIEW_LOG.md` | one subject in depth | rewritten as the subject develops |

The gap this file fills: a night's worth of findings is otherwise spread across twenty
commit messages, and nobody reconstructs the story from those six months later.

---

## 2026-08-16 — one DSP core for two programs, and a drum classifier that can say "no"

Two days of measurement bugs had one shape in common, so the day was spent on the shape
rather than on more bugs.

### The duplication was the root cause, not a side effect

Three times in one night a fix was made in `listener/features.py` and then hand-ported
into `host/audio_features.py`, and three times the port arrived broken while looking
correct. The clearest case: a tempo fix copied across at 03:00, carrying the paragraph
that explained the bug it no longer fixed. A comment saying "keep these in sync" was
already at the top of both files. Reading a warning is not executing one.

**`listener/shared_dsp.py`** now owns onsets, tempo (autocorrelation + perceptual prior
+ events-per-beat + whole-bar snapping), chroma, Krumhansl key scoring, spectral
flatness, envelope, stereo and BS.1770-4 loudness. `features.py` went 869 → 245 lines,
`audio_features.py` 304 → 180, and `describe.py` lost its own copy of the key profiles
entirely — so the MIDI tier and the audio tier can no longer come to disagree about
what "F minor" means.

**The move is proven, not asserted.** Fed the same mono signal, old and new agree to
0.000e+00 across 120 real library files at seven sample rates — bit-identical, not
"within a tolerance", because a tolerance passes a transposed constant. Writing that
test first is why the swap was allowed to happen at all; a hand-move is precisely the
operation that had already failed three times here.

**Rule 1: the core owns pre-processing.** The obvious API — take `(array, rate)` and
let each caller resample — would have moved the drift rather than removed it, since the
two programs would still each pick their own resampling. `prepare()` is the only way in,
`measure()` raises `TypeError` on a raw array, and the type is the enforcement instead
of a comment asking nicely.

**The sync check is a hash.** `verify_against_bridge`, the schema check this was
modelled on, compares substrings and would pass a renamed column type; it has never
drifted, but not because it would have caught it. `test_shared_dsp_sync.py` — identical
in both repos — compares SHA-256 over the whole file, which as a side effect forces
`shared_dsp.py` to stay self-contained. It earned its keep within the hour: an edit
written through Python on Windows silently became CRLF, and the checker said so, naming
line endings as the likely cause, rather than printing two hashes and leaving a guess.

**Snapping became the caller's decision**, found by the bridge's own tests going red
after the swap. Bar snapping assumes the file is a whole number of bars — true of a
library loop, worth 12 accuracy points there, false for the arbitrary audio the bridge
analyses, where a genuine 90 BPM clip lasting a bar and a half gets pulled to 120. The
maths is shared; the policy is not.

**The first dividend, fixed once instead of twice.** `attack_ms` was `argmax` over a
smoothed envelope, which on a sustained sound decides on numerical noise. `ORGAN9.wav`
has 69 envelope samples within 0.1% of its maximum spread over 1.13 s, with 3e-8
between the top two — so its stored attack swung 409.1 ms → 90.9 ms on any change to
decoding, and about 2% of one-shots are shaped that way. Timing the FIRST arrival at
peak level is both stable under perturbation (27.3 ms across perturbations from 0 to
1e-5) and the musically correct question: attack is when a sound arrives at its level,
not when its single largest sample falls.

`FEATURE_VERSION` → `feat6`. The maths is unchanged, but the signal it measures now
comes from the shared resampler, and a few values move with it (2 of 42 tempos by
0.1 BPM). **The index deliberately stays on `feat5`** — the re-scan is a separate,
deliberate act, not a side effect of a refactor.

### Phase B — a drum classifier with a way out

AudioSet has no vocabulary for the distinctions a producer actually browses by: kick vs
tom vs rim, clap vs snare. So `drums.py` trains a small softmax regression on the stored
Discogs-EffNet embeddings, labelled from paths — filename beating folder, contradictions
dropped rather than guessed.

The first run labelled **13,697 files**, including synth chords at 0.98 confidence. The
fault is structural, not a threshold to tune: a closed-set softmax has to pick a class
and has no way to answer "none of the above". Gating on AudioSet percussion events plus
the one-shot property brought it to **5,429**. The `PERCUSSIVE_EVENTS` list is an
explicit tuple and not a `LIKE` pattern, because `%rum%` matches "inst-**rum**-ent".

### What went wrong on my side, recorded because it will recur

`git add -A` swept an agent's work-in-progress into the drum-gate commit `b77ab1e`, so
that commit contains all 1,040 lines of `shared_dsp.py` and its two test files under a
message about drum gating. It stands as it is: correcting it means rewriting published
history, which requires Kim's explicit yes. The real fix is staging by path.

---

## 2026-08-15 → 16 — the long night: measurement, peer review, and a silently corrupt index

Started as "continue with the sidecar". Ended with ~20 bugs fixed across both repos and
the index rebuilt from scratch, because it turned out to have been quietly wrong.

### What was built

**Phase A — the measured half.** The `properties` table had shipped in the schema and
contained **zero rows**. It now carries, per file: kind (one-shot or loop, by onset
density), onsets, bpm + confidence + bars, key/scale/strength for loops, pitch as
Hz/MIDI/confidence for one-shots, attack, T60 decay, stereo width and correlation, and
BS.1770-4 loudness. Embeddings are stored as float16 so a classifier can be trained
without decoding the library again.

**`live_find_sound` searches meaning, not filenames.** `query` was a bare
`f.path LIKE` — a filename grep wearing a listening result's clothes. It returned files
tagged `Steam` for "cymbal" at relevance 0.0 with nothing saying the audio had never
been consulted. It now matches heard tags AND the path, word by word, and every result
says which in `matched_by`.

**`tools/ask_gemini.py`** — send whole modules to Gemini over the API, replies saved to
disk. Written because pasting 9,200 lines into a browser lost roughly a third of the
messages silently.

### The bug that invalidated every previous index

`record()` took the file id from `cur.lastrowid` after `INSERT … ON CONFLICT DO UPDATE`.
On the UPDATE branch nothing is inserted, so lastrowid still held the last `tags` rowid
— in the millions, and non-zero, so the `if not file_id` fallback never fired.

**Every re-analysis wrote its tags, properties and embedding against a bogus id**, while
`files.analyzer` was stamped correctly. Stale verdicts, marked fresh, which `plan()`
would then skip forever. Measured before it was caught: **581,789 orphaned tags, 12,825
orphaned properties**.

It only bites on RE-analysis, so a first scan of an empty database is always clean —
which is why it survived several full runs. Found by counting rows in two tables and
noticing 42,694 properties for 29,870 files.

Two things now prevent it: the id is resolved by `SELECT … WHERE path=?`, and
**`PRAGMA foreign_keys` is ON** — SQLite disables it per connection, so every
`REFERENCES` clause in the schema had been decoration.

### Measurements, and one that lied

Filenames are the ground truth: a file called `Drumloop 11 170BPM.wav` is a real claim
about itself, and ~546 loops and ~3,671 one-shots state theirs.

| | start of night | end |
|---|---|---|
| Tempo within 2.5 BPM | 57% | **69%** |
| — of which octave errors | 22% | **10%** |
| Pitch class matches filename note | 71% | **75%** |

**The aggregate hid a catastrophe.** Gemini predicted from the prior's multipliers alone
— before any measurement — that a narrow tempo prior would force fast genres into
half-time. Measured: slow 72%, mid 70%, **fast (≥140 BPM) 36%**. Only 58 of 546
ground-truth files are fast, so the headline barely moved while drum-and-bass and trap
broke completely. Widths are now swept **per band** and chosen on the worst one.

Kim's correction underneath that: **85 and 170 BPM are not the same tempo.** The same
pulse written at each differs in note values, so the grid, the swing and every quantise
decision differ with it. Octave errors are counted as errors; forgiving them would read
79% and flatter nobody usefully.

### Bugs worth remembering (all produced plausible output; none raised)

* **A 66 dB window.** One Hann window spanned the whole file in `band_energies`, and a
  Hann window is zero at its edges — the same click measured −35.7 dB at t=0 and
  +30.3 dB at t=2 s. For a one-shot, whose content is at the start, it erased the file.
* **YIN's parabolic interpolation had its sign inverted** — 440 Hz read as 448.9.
* **K-weighting was sampled from hand-derived analogue transfer functions**, wrong by
  3–4 LU and by a different amount at every frequency. Rebuilt from the digital biquads
  the standard specifies; now matches to 12+ significant figures.
* **An onset at sample 0 was invisible, twice over** — the Hann window's zero edge, and
  a median threshold padded with edge values that made a large opening flux its own
  local median. That is where every loop's downbeat sits.
* **Pitch locked onto subsonic subharmonics** at 0.7–0.8 confidence: a stab whose
  strongest partial was E2 reported 27.4 Hz. A confidence floor cannot catch that,
  because the subharmonic genuinely *is* a period of the signal — only the spectrum can.
* **Chroma could not see a bass note.** Three FFT bins for the whole A1–A2 octave;
  twelve pitch classes cannot fit in three bins.
* **Krumhansl answered confidently from a flat histogram** — white noise came back as
  D minor with `ambiguous: False`, because correlation is scale-invariant and cannot
  tell a flat distribution from a peaked one. Fixing the chroma was necessary and not
  sufficient.
* **Chords were named from the lowest pitch class, not the bass note** — an A minor 7
  came back as `C6`. And `add9` could never match: its interval key was written
  `(0, 4, 7, 14 % 12)` while the lookup builds sorted tuples.
* **The one-shot guard stripped genre from 26% of loops**, because a 1-bar loop at
  120 BPM lasts 2.0 s and the threshold was 2.048. One file returned `one_shot: true`
  alongside `kind: loop, bars: 1, bpm: 117.3` in the same reply.

### What we learned

**The same bug lived in both programs three times.** The listener and the bridge measure
some of the same things twice, and hand-porting does not work: the wide-lag tempo fix was
made in the listener at 01:00 and reproduced verbatim in the bridge at 03:00 — *with its
explanatory comment copied across intact*. Reading a warning is not executing it.
Agreed design: one shared DSP core, single source of truth plus a **byte-for-byte
SHA-256 checked copy**, the pattern the SQLite schema already uses successfully. Not yet
built.

**Comments can lie, and a reviewer will believe them.** `sidecar.py` said the sidecar
"runs under WSL" — never true of the shipped design — and Gemini read it and built
advice around two Python environments that do not exist. It had been told the correct
architecture at the start of the session and drifted back because our own source said
otherwise, in writing. A stale "mel parameters NOT YET VALIDATED" banner was removed
from three places for the same reason. In a heavily commented codebase the comments are
load-bearing.

**Peer-review the approach before building it.** Phase A.5 was implemented first and
turned out unreliable on real material — hours parked on a branch. The tempo calibration
was discussed first, and moved the events-per-beat anchor from an intended 2–3 down to a
measured 1.5; the sweep later showed 3.0 scores 49%, a coin flip. Same idea, opposite
outcomes.

**A confident review is not evidence — in either direction.** Gemini found things no
measurement of ours would have. Verification rejected eight of its claims that would
have damaged working code (the K-weighting was already correct to 12 significant
figures; float16 round-trips bit-exactly; the Krumhansl profiles were right). And the
worst bug of the night was found by verification, in a file whose Gemini review had led
with something else entirely. See `REVIEW_LOG.md` in the bridge repo for the
attribution.

### End state

29,870 files, 1,364,217 tags, 1 failure, **zero orphans**. 153 tests green across both
repos. Code is `feat5`, index is `feat4` — the chroma change means stored keys are
stale, so one ~65-minute re-scan is outstanding. Phase A.5 sits on branch
`phase-a5-tonal-routing`, unreliable on real material. Phase B (the drum classifier)
has everything it needs in the database and has not been started.
