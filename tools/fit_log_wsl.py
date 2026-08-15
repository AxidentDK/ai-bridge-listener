"""The spectrum and filterbank are proven identical to Essentia's. Only the log differs.

So the problem reduces to one function: given the shared mel M, which compression f
gives Essentia's output E? Test candidates directly instead of inferring from ratios.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from listener import decode  # noqa: E402

import essentia.standard as es  # noqa: E402

CANDIDATES = {
    "log10(1 + 10000*M)   [ours]": lambda m: np.log10(1.0 + 10000.0 * m),
    "log10(1 + M)": lambda m: np.log10(1.0 + m),
    "ln(1 + M)": lambda m: np.log(1.0 + m),
    "ln(M + 1e-3)": lambda m: np.log(m + 1e-3),
    "ln(M + 1e-2)": lambda m: np.log(m + 1e-2),
    "log10(M + 1e-8)": lambda m: np.log10(m + 1e-8),
    "10*log10(M + 1e-10)": lambda m: 10.0 * np.log10(m + 1e-10),
    "log10(1 + 1000*M)": lambda m: np.log10(1.0 + 1000.0 * m),
    "log10(1 + 100000*M)": lambda m: np.log10(1.0 + 100000.0 * m),
}


def main(path: str) -> int:
    audio = np.asarray(es.MonoLoader(filename=path, sampleRate=16000,
                                     resampleQuality=4)(), dtype=np.float32)
    for which, cfg, algo, norm in (
            ("musicnn", decode.EFFNET, "TensorflowInputMusiCNN", "unit_tri"),
            ("yamnet", decode.YAMNET, "TensorflowInputVGGish", "unit_max")):
        print(f"\n=== {which} ===")
        frames = list(es.FrameGenerator(audio, frameSize=cfg.frame_size,
                                        hopSize=cfg.hop_size, startFromZero=True))[:64]
        w = es.Windowing(type="hann", normalized=False, size=cfg.frame_size)
        spec = es.Spectrum(size=cfg.frame_size)
        their_spec = np.array([spec(w(f)) for f in frames])
        mb = es.MelBands(numberBands=cfg.n_mels, sampleRate=16000,
                         lowFrequencyBound=cfg.fmin,
                         highFrequencyBound=cfg.fmax or 8000.0,
                         inputSize=cfg.frame_size // 2 + 1, normalize=norm,
                         type="magnitude", warpingFormula="slaneyMel",
                         weighting="linear")
        mel = np.array([mb(s) for s in their_spec], dtype=np.float64)

        alg = getattr(es, algo)()
        target = np.array([alg(f) for f in frames], dtype=np.float64)
        n = min(len(mel), len(target))
        mel, target = mel[:n], target[:n]

        print(f"  mel   range [{mel.min():.6g}, {mel.max():.6g}]")
        print(f"  essentia out range [{target.min():.6g}, {target.max():.6g}]")
        best = None
        for label, fn in CANDIDATES.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                got = fn(mel)
            if not np.isfinite(got).all():
                print(f"     {label:<28} non-finite")
                continue
            relmax = float(np.abs(got - target).max()) / max(float(np.abs(target).max()), 1e-12)
            mark = ""
            if relmax < 0.001:
                mark = "   <=== EXACT MATCH"
            elif relmax < 0.02:
                mark = "   <-- very close"
            print(f"     {label:<28} rel max err {relmax:10.6f}{mark}")
            if best is None or relmax < best[1]:
                best = (label, relmax)
        print(f"  best: {best[0]}  (rel err {best[1]:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
