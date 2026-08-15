"""WHERE does the chain diverge? Bisect it stage by stage against Essentia's primitives.

Ratio-fitting the final output failed because several differences compound. So instead
rebuild the chain with Essentia's OWN primitives — Windowing, Spectrum, MelBands,
UnaryOperator — and compare ours to theirs after each stage. The first stage that
disagrees is the bug; everything after it is downstream noise.

Run inside WSL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from listener import decode  # noqa: E402

import essentia.standard as es  # noqa: E402


def agree(a: np.ndarray, b: np.ndarray) -> str:
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], float), np.asarray(b[:n], float)
    if a.shape != b.shape:
        return f"SHAPE {a.shape} vs {b.shape}"
    denom = max(float(np.abs(b).max()), 1e-12)
    rel = float(np.abs(a - b).max()) / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(b) > 1e-9, a / b, np.nan)
    med = float(np.nanmedian(r))
    verdict = "SAME" if rel < 0.01 else f"DIFFER (x{med:.4g})"
    return f"{verdict:<22} relmax {rel:9.4f}  median ratio {med:12.6g}"


def main(path: str, which: str = "musicnn") -> int:
    cfg = decode.EFFNET if which == "musicnn" else decode.YAMNET
    n_mels, fmin = cfg.n_mels, cfg.fmin
    fmax = cfg.fmax if cfg.fmax is not None else 8000.0

    audio = np.asarray(es.MonoLoader(filename=path, sampleRate=16000,
                                     resampleQuality=4)(), dtype=np.float32)
    print(f"=== {which}: frame {cfg.frame_size} hop {cfg.hop_size} "
          f"bands {n_mels} range {fmin}-{fmax} ===")

    frames = list(es.FrameGenerator(audio, frameSize=cfg.frame_size,
                                    hopSize=cfg.hop_size, startFromZero=True))[:32]
    ours_bank, ours_window = decode._bank_and_window(cfg)

    # --- stage 1: the window itself -----------------------------------------
    for norm in (True, False):
        w = es.Windowing(type="hann", normalized=norm, size=cfg.frame_size)
        their_win = np.array([w(f) for f in frames])
        our_win = np.array([np.asarray(f) * ours_window for f in frames])
        print(f"  1 window(normalized={str(norm):<5})      {agree(our_win, their_win)}")

    # --- stage 2: spectrum ---------------------------------------------------
    w = es.Windowing(type="hann", normalized=False, size=cfg.frame_size)
    spec = es.Spectrum(size=cfg.frame_size)
    their_spec = np.array([spec(w(f)) for f in frames])
    our_spec = np.array([np.abs(np.fft.rfft(np.asarray(f) * ours_window))
                         for f in frames])
    print(f"  2 spectrum (unnormalized win) {agree(our_spec, their_spec)}")

    # --- stage 3: mel bands, across Essentia's normalisation options ---------
    for normalize in ("unit_tri", "unit_sum", "unit_max"):
        for mtype in ("magnitude", "power"):
            try:
                mb = es.MelBands(numberBands=n_mels, sampleRate=16000,
                                 lowFrequencyBound=fmin, highFrequencyBound=fmax,
                                 inputSize=cfg.frame_size // 2 + 1,
                                 normalize=normalize, type=mtype,
                                 warpingFormula="slaneyMel", weighting="linear")
                their_mel = np.array([mb(s) for s in their_spec])
            except Exception as exc:                              # noqa: BLE001
                print(f"  3 mel {normalize}/{mtype:<9} unavailable: {str(exc)[:40]}")
                continue
            our_mel = our_spec @ ours_bank.T
            print(f"  3 mel {normalize:<9}/{mtype:<9} {agree(our_mel, their_mel)}")
    return 0


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "musicnn")
