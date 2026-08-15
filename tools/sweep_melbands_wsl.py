"""Find the filterbank parameters by SWEEP, using a monotonicity test.

Reading the source tells you what upstream intends; this tells you what the installed
build actually does, which is what produced the reference output.

THE TRICK. A log is monotonic, so if the MelBands parameters are correct then the map
from mel value to TensorflowInput* output is monotonic too, and Spearman rank
correlation is exactly 1.0 — whatever the log turns out to be. If the parameters are
wrong, the ordering differs and rho drops. So rho isolates the FILTERBANK question
completely from the COMPRESSION question, instead of confounding them the way a
direct value comparison does.

Run inside WSL.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from listener import decode  # noqa: E402

import essentia.standard as es  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a.ravel())).astype(np.float64)
    rb = np.argsort(np.argsort(b.ravel())).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def main(path: str) -> int:
    audio = np.asarray(es.MonoLoader(filename=path, sampleRate=16000,
                                     resampleQuality=4)(), dtype=np.float32)

    for which, cfg, algo in (("musicnn", decode.EFFNET, "TensorflowInputMusiCNN"),
                             ("yamnet", decode.YAMNET, "TensorflowInputVGGish")):
        print(f"\n{'=' * 74}\n{which}: frame {cfg.frame_size} hop {cfg.hop_size}\n{'=' * 74}")
        frames = list(es.FrameGenerator(audio, frameSize=cfg.frame_size,
                                        hopSize=cfg.hop_size, startFromZero=True))[:24]
        alg = getattr(es, algo)()
        target = np.array([alg(f) for f in frames], dtype=np.float64)
        n_bands = target.shape[1]
        print(f"  output bands: {n_bands}   range [{target.min():.6g}, {target.max():.6g}]")

        w = es.Windowing(type="hann", normalized=False, size=cfg.frame_size)
        spec = es.Spectrum(size=cfg.frame_size)
        spectra = np.array([spec(w(f)) for f in frames])

        results = []
        for norm, mtype, warp, weight, lo, hi in itertools.product(
                ("unit_sum", "unit_tri", "unit_max"),
                ("magnitude", "power"),
                ("slaneyMel", "htkMel"),
                ("linear", "warping"),
                (0.0, 125.0), (8000.0, 7500.0)):
            try:
                mb = es.MelBands(numberBands=n_bands, sampleRate=16000,
                                 lowFrequencyBound=lo, highFrequencyBound=hi,
                                 inputSize=cfg.frame_size // 2 + 1, normalize=norm,
                                 type=mtype, warpingFormula=warp, weighting=weight)
                mel = np.array([mb(s) for s in spectra], dtype=np.float64)
            except Exception:                                      # noqa: BLE001, S112
                continue
            rho = spearman(mel, target)
            results.append((rho, norm, mtype, warp, weight, lo, hi))

        results.sort(reverse=True)
        print(f"  {'rho':>10}  normalize   type       warping     weighting  lo      hi")
        for rho, norm, mtype, warp, weight, lo, hi in results[:6]:
            flag = "   <== MONOTONIC: filterbank matches" if rho > 0.99999 else ""
            print(f"  {rho:10.6f}  {norm:<10} {mtype:<10} {warp:<11} {weight:<9} "
                  f"{lo:<7.0f} {hi:.0f}{flag}")
        if results and results[0][0] <= 0.99999:
            print("  no combination is monotonic — the difference is NOT in these "
                  "parameters (framing or an extra step upstream)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
