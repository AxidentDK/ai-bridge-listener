# Mel validation against Essentia — RESOLVED (2026-08-15)

**Status: FIXED and verified. Our spectrograms now match Essentia's.**

Downstream proof, which matters more than any correlation: 640 files, 16 categories,
same models and heads, only the spectrogram differing —

    before   ours 28% / 67%     essentia 29% / 70%
    after    ours 29% / 70%     essentia 29% / 70%      IDENTICAL, every category

Not a wiring artefact: the same harness separated the two arms before the fix.

## The five bugs

Every one changed values while leaving shapes correct, so nothing ever raised.

| model | wrong | right |
|---|---|---|
| MusiCNN | magnitude spectrum | **power** (`type` defaults to power; never overridden) |
| YAMNet | 400-point FFT (201 bins) | **512, frame zero-padded right** (257 bins) |
| YAMNet | Slaney mel | **htkMel** `2595·log10(hz/700+1)` |
| YAMNet | slopes linear in Hz | **slopes linear in mel** (`weighting="warping"`) |
| YAMNet | `ln(x + 0.001)` | **`ln(x + 0.01)`** |

Correlation went 0.876–0.935 → 0.996–0.998 (MusiCNN) and 0.739–0.981 → 0.9989–0.9991
(YAMNet); median ratio 26…2025 → 0.9990 and 1.5 → 1.0028.

**Read the source, don't infer the recipe.** An afternoon went into fitting candidate
logs and inverting ratios against the algorithm's *output*. The answer was in
`src/algorithms/spectral/tensorflowinput{musicnn,vggish}.cpp` — hardcoded, commented,
and unambiguous. The parameters are also NOT in `machinelearning/`, which only holds
the `TensorflowPredict*` wrappers.

## Confirmed irrelevant — do not re-chase

* **zeroPhase** rotates the frame before the FFT. Magnitude spectra are unchanged.
* **shift order** is scale-then-shift, `10000·x + 1` — already what we computed.
* **float32 edge accumulation** matches the source but changed nothing measurable.

## A trap we happen to dodge

Essentia binds `weighting="slaneyMel"` to `hz2mel` (`1127.01048·ln(1+f/700)`), **not**
`hz2melSlaney`. Its "Slaney" weighting is not Slaney's formula. MusiCNN uses
`"linear"` and VGGish `"warping"`, so neither config touches that branch — but a
future config might.

## What this did and did not buy

**+1 point specific, +3 family.** Small, and forecast before the work: the A/B was
run deliberately *first*, so the payoff was known before it was paid for. The value is
not the point gained — it is that the tags are now provably the ones the models were
trained to receive, so every future measurement means something.

## Tools (run inside WSL)

    tools/run_in_wsl.sh        wrapper; silences TensorFlow's CUDA noise
    tools/validate_mel_wsl.py  ours vs Essentia
    tools/ab_mel_test.py       tag the same files both ways — the decisive test
    tools/sweep_melbands_wsl.py  parameter sweep via a monotonicity test
    tools/bisect_mel_wsl.py / fit_log_wsl.py / diagnose_mel_wsl.py

`sweep_melbands_wsl.py` is worth keeping in mind for future parameter questions: a log
is monotonic, so Spearman rank correlation against the reference isolates the
FILTERBANK question from the COMPRESSION question instead of confounding them.

## Setup

WSL 2 / Ubuntu 26.04 / Python 3.14. Essentia in a venv at `~/essentia-venv`
(`essentia-tensorflow`, cp314 wheel — nothing built from source). TensorFlow logs a
wall of CUDA warnings on import; **nothing is broken**, it probes for a GPU and falls
back to CPU. Installing CUDA would accelerate nothing: Essentia is used only for mel
algorithms, and the models run through onnxruntime at ~2 ms/patch.

---

# Superseded — the state BEFORE the fix, kept only for the reasoning

⚠️ Everything below describes the broken state and was written while it was still
broken. **It is history, not status.** The questions it leaves open — "two ways
forward, undecided" — were answered by the fix above: the recipe was read out of
Essentia's source, in numpy, with no WSL runtime dependency. Do not act on this
section; read it only to avoid re-deriving what was already ruled out.

## What is set up and working

WSL 2 / **Ubuntu 26.04 LTS / Python 3.14**, 6 cores, 15 GB. Essentia installed in a
venv at `~/essentia-venv` (`essentia-tensorflow 2.1b6.dev1438` — the cp314 manylinux
wheel matched, nothing was built from source). Both needed algorithms are present:
`TensorflowInputMusiCNN`, `TensorflowInputVGGish`.

Reachable from Windows without a terminal:

    wsl -d Ubuntu -- bash /mnt/c/.../ai-bridge-listener/tools/run_in_wsl.sh validate "<file>"

TensorFlow logs a wall of CUDA warnings on import. **Nothing is broken** — it probes
for an optional GPU and falls back to CPU. `run_in_wsl.sh` silences it. Installing
CUDA would accelerate nothing here: Essentia is used only for mel algorithms (CPU
DSP), and the models run through onnxruntime on Windows at ~2 ms/patch.

## The result

    musicnn:  correlation 0.88-0.95   median ratio 26 ... 2025 (NOT constant)
    yamnet:   correlation 0.74-0.98   mean diff -2.06, ratio 1.5

Frame counts and band counts match exactly. The **values** do not.

Correlation of 0.88-0.98 is why this was invisible for so long: enough structure
survives that the output looked plausible rather than broken. **It is a live
hypothesis that this is why specific-label accuracy sat at 37%.**

## What is PROVEN, and what only looked proven

Verified identical to Essentia's primitives (`tools/bisect_mel_wsl.py`):

| stage | verdict |
|---|---|
| our spectrum vs `Spectrum` | identical (relmax 0.0000) |
| our EffNet filterbank vs `MelBands(unit_tri, magnitude)` | identical |
| our YAMNet filterbank vs `MelBands(unit_max, magnitude)` | identical |

⚠️ **Read that table carefully — it is weaker than it looks, and I misread it once.**
It compares our code against *a reconstruction of Essentia's chain that I wrote*. Two
implementations agreeing does not prove either matches what `TensorflowInputMusiCNN`
does internally. The numbers say it does not:

    mel range          [3.5e-07, 0.3148]      <- the shared mel
    essentia out range [5.2e-08, 4.4930]      <- TensorflowInputMusiCNN

Our log gives max 3.50 where Essentia reaches 4.49; at the bottom Essentia gives ~0
where ours gives 0.0015. **No elementwise function maps one to the other**
(`tools/fit_log_wsl.py` tried nine, best rel err 0.22). So `TensorflowInput*` is NOT
`MelBands` + a log — it does its own framing, padding or normalisation.

Also useful, and a red herring worth not re-chasing: Essentia's `Windowing` applies a
**zero-phase circular shift** by default. The windowed frames differ from ours while
the magnitude spectra are identical. That is expected, not a bug.

## Two ways forward — undecided, Kim's call

1. **Keep reverse-engineering `TensorflowInput*`** until it replicates in numpy.
   Preserves the "no WSL, 20 MB, runs anywhere" property that makes this project
   attractive. Depth unknown — the difference has been mislocated twice already.
   The next concrete step is reading Essentia's source for the composite algorithm
   rather than fitting curves to its output.
2. **Index using Essentia itself in WSL.** Correct by construction. The bridge is
   unaffected — it still reads SQLite on Windows — but building an index would need
   WSL. Converts an open-ended debugging problem into a known-good reference, and
   would finally settle whether 37% is the models' ceiling or our bug.

## Tools (all run inside WSL)

    tools/run_in_wsl.sh        wrapper; silences the CUDA noise
    tools/validate_mel_wsl.py  ours vs Essentia, flags bias and scale factors
    tools/bisect_mel_wsl.py    stage by stage against Essentia's primitives
    tools/fit_log_wsl.py       which compression maps the mel to Essentia's output
    tools/diagnose_mel_wsl.py  inverts Essentia's output through candidate logs

## Do not forget

Everything in `README.md` about accuracy (82% family / 37% specific) was measured on
the **current, mismatched** spectrograms. Those numbers are honest about what the
system does today, but they are **not** a measurement of the models' ability. If the
mel is fixed, re-run `python -m listener.evaluate` before quoting them again.

> **Answered, and the instrument was wrong too.** The mel was fixed and the library
> re-scanned, so the numbers are no longer measured on bad spectrograms. But
> `evaluate.py` turned out to be a breakage detector rather than a quality meter — it
> scored the corrected and the broken spectrogram *identically, to the unit*, across 16
> categories. Use **`rank_eval.py`** (MRR, hit@1, hit@3) for anything that claims to
> measure quality. Current: **hit@1 57.2%** after demoting AudioSet's interior nodes.
