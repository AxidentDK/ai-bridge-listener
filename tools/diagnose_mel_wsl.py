"""WHICH step differs? Test hypotheses instead of guessing at the recipe.

The validator says our spectrograms do not match Essentia's. A non-constant ratio
rules out a simple gain, so the difference is inside a nonlinearity. The candidates,
in the order they would bite:

  1. POWER vs MAGNITUDE spectrum — |X| against |X|^2. Squaring before a log is a
     factor of 2 afterwards, and before the log's shift it is not even that.
  2. WINDOW NORMALISATION — Essentia's Windowing normalises by default, scaling every
     frame by 1/sum(window).
  3. LOG COMPRESSION — log10(1 + 10000x) vs log(x + eps), and which base.
  4. FILTERBANK NORMALISATION — unit-area triangles against unit-height.

Run inside WSL. It inverts Essentia's output back through each candidate log and
compares against our pre-log mel, which localises the disagreement to one step
instead of leaving a single number for the whole chain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from listener import decode  # noqa: E402

import essentia.standard as es  # noqa: E402


def our_premel(audio: np.ndarray, cfg: decode.MelConfig) -> np.ndarray:
    """Our mel BEFORE any log compression — the raw filterbank output."""
    bank, window = decode._bank_and_window(cfg)
    n = 1 + max(0, (len(audio) - cfg.frame_size) // cfg.hop_size)
    frames = np.lib.stride_tricks.sliding_window_view(
        audio, cfg.frame_size)[::cfg.hop_size][:n]
    spec = np.abs(np.fft.rfft(frames * window, axis=-1)).astype(np.float32)
    return spec @ bank.T


def report(name: str, ours: np.ndarray, theirs: np.ndarray) -> None:
    print(f"  {name:<34} ours[{ours.min():9.4f},{ours.max():9.4f}] "
          f"mean {ours.mean():8.4f}   theirs[{theirs.min():9.4f},"
          f"{theirs.max():9.4f}] mean {theirs.mean():8.4f}")


def main(path: str) -> int:
    audio = es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
    audio = np.asarray(audio, dtype=np.float32)

    for algo, cfg in (("TensorflowInputMusiCNN", decode.EFFNET),
                      ("TensorflowInputVGGish", decode.YAMNET)):
        print(f"\n=== {algo} ===")
        frames_alg = getattr(es, algo)()
        theirs = np.array([frames_alg(f) for f in es.FrameGenerator(
            audio, frameSize=cfg.frame_size, hopSize=cfg.hop_size,
            startFromZero=True)], dtype=np.float32)
        ours_log = decode.melspectrogram(audio, cfg)
        ours_log = ours_log.reshape(-1, ours_log.shape[-1])
        n = min(len(ours_log), len(theirs))
        ours_log, theirs = ours_log[:n], theirs[:n]
        pre = our_premel(audio, cfg)[:n]

        print("  -- as computed --")
        report("ours (post-log) vs essentia", ours_log, theirs)

        print("  -- invert essentia through candidate logs, compare to our PRE-log --")
        cands = {
            "10**x - 1 / 10000  (musicnn)": (np.power(10.0, theirs) - 1.0) / 10000.0,
            "exp(x) - 0.001     (vggish)": np.exp(theirs) - 0.001,
            "exp(x) - 0.01": np.exp(theirs) - 0.01,
            "10**x": np.power(10.0, theirs),
        }
        for label, inv in cands.items():
            if not np.isfinite(inv).all():
                continue
            # melspectrogram() pads to fill a whole patch; our_premel() does not, so
            # the two can differ by a frame. Compare only the overlap.
            m = min(len(inv), len(pre))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(pre[:m]) > 1e-9, inv[:m] / pre[:m], np.nan)
            med = float(np.nanmedian(ratio))
            spread = float(np.nanstd(ratio[np.isfinite(ratio)]))
            flag = ""
            if 0.9 < med < 1.1 and spread < 0.5:
                flag = "  <== our pre-log matches this"
            elif 1.9 < med < 2.1:
                flag = "  <== factor 2: POWER vs MAGNITUDE"
            print(f"     {label:<30} median ratio {med:12.4f}  sd {spread:10.4f}{flag}")

        # Is the difference a pure square? Compare our magnitude against its square.
        with np.errstate(divide="ignore", invalid="ignore"):
            sq = np.where(np.abs(pre) > 1e-9, (pre ** 2) / np.maximum(pre, 1e-9), np.nan)
        print(f"     (our pre-log mean {pre.mean():.6f}, "
              f"as power {np.mean(pre ** 2):.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
