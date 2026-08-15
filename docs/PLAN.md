# Agreed build plan — Claude Code + Gemini, 2026-08-15

Kim's instruction was that the two of us should **agree on what to build**, not produce a
review that ends in a list. This is that agreement. Gemini reviewed with the full picture
of both programs — the listener AND the bridge's 60 tools.

## What the review actually found

Three gaps, all **verified in the code before being accepted**, not taken on assertion.

| Gap | Evidence |
|---|---|
| **Stereo is discarded** | `decode.py` sums to mono (`audio.mean(axis=1)`) before anything runs. **17,990 of 22,100 files (81%) are stereo.** No width tag exists. |
| **`properties` is EMPTY** | The table has bpm, bpm_confidence, key, scale, key_strength, danceability, loudness_lufs — and **zero rows**. No tonal or temporal data is indexed at all. |
| **Embeddings are thrown away** | The 1280-dim EffNet vectors feed the 25 heads and are discarded, so any classifier work means a full re-scan. |

The stereo gap was **Gemini's catch**. The other two surfaced while verifying its answers.

Worth stating plainly: **the DSP mostly exists already.** The bridge's
`live_describe_audio` computes crest factor, inter-channel correlation, spectral centroid,
envelope shape and a chroma key — for one file, on demand. The honest framing is not "add
DSP to the listener" but *the same measurements exist twice and are indexed zero times*.

## Corrections issued during the review

* **To Gemini:** it assumed CLAP and Essentia at runtime (neither is used), and called the
  EffNet embeddings 64-dimensional — that is **Live 12's own** embedding size, a different
  system. Ours are **1280**. It accepted both.
* **To Gemini, conceded by it:** it wanted `genre` and `danceability` dropped outright for
  a sample library. Fixing the one-shot/loop discriminator makes that unnecessary — a
  4-bar loop *should* get danceability; a 3-second crash bypasses it on onset count.
* **From Kim, to Claude:** the previously-dropped ML project was **MidiGen — trained on
  MIDI only, never listened to audio**. It is unrelated, so no prior decision stands
  against a small classifier on audio embeddings.
* **From Gemini, to Claude — and it was right:** the proposed ordering was wrong. See below.

## The ordering argument, which Gemini won

Claude proposed: extract properties first, add onset routing second. Gemini overruled it:
that order runs YIN across thousands of atonal breakbeats and writes BPM for 22,100
one-shots. **The cost is not wasted CPU — it is a properties table full of confident junk
that the bridge would later treat as fact.** Onset detection must run FIRST, inside the
same pass, as a *router*.

## PHASE A — one smart indexing pass

Single decode per file. Onset density routes what gets computed.

**Router** — spectral flux onsets (NOT high-frequency content, which over-triggers on
shakers and vinyl crackle), median-filtered moving threshold, count peaks:

    <= 2 onsets   one-shot
       3 onsets   gray area — fall back to the 2.048 s duration rule
    >= 4 onsets   loop

**Routed by type**

| | one-shot | loop |
|---|---|---|
| tonality | YIN fundamental → MIDI note | chroma key + scale + key_strength |
| tempo | skipped | BPM |

The tonality split is Claude's refinement to Gemini's plan: it proposed skipping pitch for
loops entirely; switching the **method** rather than dropping the **question** is better,
since key and scale are exactly what a producer needs before dropping a loop into a
project.

**YIN needs two guards or it fails on 808s** — the transient click is broadband noise and
the second harmonic is often louder than the fundamental. Start analysis **~50 ms in**,
and **low-pass at ~300 Hz** for anything low-flagged.

**Computed for every file, regardless of route**

* **Stereo width — M/S energy ratio above 250 Hz.** Must not be broadband: mono bass with
  wide highs is the standard production shape, and a single number averages it to
  "narrow". Measured **before** the mono downmix.
* **T60 / `decay_time_ms`** — so the bridge knows whether an 808 rings for 400 ms or 3 s.
  Without it, programmed notes overlap and mud the low end.
* **Time-to-peak (attack offset)** — the millisecond where the primary onset peaks. A
  reverse cymbal or chopped vocal may have 40 ms before the hit; indexing this lets the
  bridge shift MIDI placement so the audio lands *on* the grid. **This is the one tag that
  serves decision-making rather than search** — Gemini's contribution, and the item Claude
  was fishing for and could not name.
* **1280-dim EffNet embedding as a float16 BLOB** (~56 MB for the library). **No PCA** —
  1280 features are fine for a linear classifier, and reducing them throws away the
  nuance that separates a tom from a kick.

## PHASE B — the drum classifier

Closes the vocabulary hole AudioSet can never fill (**no `kick` class, no `tom` class**).
A small classifier — logistic regression or a 2-layer MLP — on the stored embeddings.
Seconds to train, kilobytes on disk. Predictions written back as a new namespace.

**Labels by folder ∩ filename**, not folder alone. Folder names are a goldmine but carry
systematic noise — a file in `Cymbals/` called `Hat 01.wav` is a hi-hat, and that exact
case is already in the library. Resolve contradictions with a filename regex; **drop**
conflicts rather than train on them. Better 4,000 clean examples than 10,000 noisy ones.

## BOUNDARY — what must NOT be indexed

**Timbral variance for take selection stays in `live_describe_audio`.** When the bridge
evaluates four takes recorded thirty seconds ago, it needs variance of spectral centroid
and crest factor *over time* — a live measurement. Indexing it would put stale data where
a fresh reading belongs. Static facts about a file get indexed once; comparisons between
takes cannot be.

## ⚠️ The trap most likely to make Phase A silently do nothing

**Bump the analyzer version string.** It has already caught this project twice. If the
analysis changes but the version does not, a re-scan compares identical strings, skips all
22,100 files, and **reports success without re-analysing one of them**. The version must
describe the ANALYSIS, not the models — see `decode.MEL_VERSION`.

## What gets reported back

Whether onset density separates one-shots from loops as cleanly on real material as it
does in theory. If it does not, that gets said — not a tidy success story.
