# Mel validation against Essentia — RESUME POINT (2026-08-15)

**Result: MISMATCH. Our spectrograms are not the ones the models were trained on.**

This is the most consequential finding the project has produced, and it is only
half-explored. Read this before touching the mel code.

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
