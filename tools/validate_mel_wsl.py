"""Compare our numpy mel-spectrogram against Essentia's own — the one open question.

RUN THIS INSIDE WSL (or any Linux), where Essentia installs:

    sudo apt update && sudo apt install -y python3-pip python3-venv
    python3 -m venv ~/essentia-venv && source ~/essentia-venv/bin/activate
    pip install essentia-tensorflow numpy soundfile soxr
    python3 /mnt/c/Users/<you>/source/repos/ai-bridge-listener/tools/validate_mel_wsl.py \
        /mnt/c/path/to/a.wav /mnt/c/path/to/b.wav

WHY IT MATTERS. The models were trained on Essentia's spectrograms. Ours follows the
documented recipe and behaves correctly on synthetic signals, but "behaves correctly"
is not "matches". A systematic offset would not crash — it would shift every embedding
into a region the classifier heads never saw, and they would answer confidently from
their priors. Every accuracy number we have rests on this being right.

WHAT COUNTS AS PASSING. Not bit-equality — different FFT libraries and float ordering
make that unreachable. What matters is whether the difference is small relative to the
signal, and whether it is NOISE or a systematic bias. A constant offset or a
consistent scale factor is the dangerous kind: it moves every frame the same way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Our implementation. Importable in WSL because it is pure numpy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from listener import decode  # noqa: E402

try:
    import essentia
    import essentia.standard as es
except ImportError:
    sys.exit("essentia not importable — see the header for the venv setup.")


def essentia_mel(path: str, algorithm: str) -> np.ndarray:
    """Essentia's own patch stack: [n_patches, frames, bands]."""
    loader = es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)
    audio = loader()
    frames = getattr(es, algorithm)()
    out = [frames(f) for f in es.FrameGenerator(
        audio, frameSize=decode.EFFNET.frame_size,
        hopSize=decode.EFFNET.hop_size, startFromZero=True)]
    return np.array(out, dtype=np.float32)


def compare(name: str, ours: np.ndarray, theirs: np.ndarray) -> None:
    n = min(len(ours), len(theirs))
    if n == 0:
        print(f"  {name}: nothing to compare (ours={ours.shape} theirs={theirs.shape})")
        return
    a, b = ours[:n].astype(np.float64), theirs[:n].astype(np.float64)
    diff = a - b
    scale = max(float(np.abs(b).max()), 1e-9)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    # A near-constant difference, or a near-constant ratio, is the dangerous kind:
    # it is a bias every frame shares rather than numerical noise.
    bias = float(diff.mean())
    spread = float(diff.std())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(b) > 1e-6, a / b, np.nan)
    print(f"  {name}")
    print(f"     shapes        ours {a.shape}  essentia {b.shape}")
    print(f"     correlation   {corr:.6f}")
    print(f"     max |diff|    {np.abs(diff).max():.4f}   ({100 * np.abs(diff).max() / scale:.2f}% of peak)")
    print(f"     mean diff     {bias:+.4f}   sd {spread:.4f}"
          f"   {'<-- SYSTEMATIC BIAS' if abs(bias) > 2 * spread and abs(bias) > 1e-3 else ''}")
    print(f"     median ratio  {np.nanmedian(ratio):.4f}"
          f"   {'<-- SCALE FACTOR' if abs(np.nanmedian(ratio) - 1) > 0.02 else ''}")
    verdict = ("MATCH" if corr > 0.999 and abs(bias) < 0.05
               else "CLOSE" if corr > 0.99
               else "MISMATCH — the tags rest on this")
    print(f"     verdict       {verdict}")


def main(argv: list[str]) -> int:
    if not argv:
        return print(__doc__) or 2
    print("essentia", essentia.__version__)
    algos = [a for a in ("TensorflowInputMusiCNN", "TensorflowInputVGGish")
             if hasattr(es, a)]
    print("available input algorithms:", algos or "NONE — check the install")

    for path in argv:
        print(f"\n=== {Path(path).name} ===")
        try:
            audio, sr = _load(path)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  could not read: {exc}")
            continue

        if "TensorflowInputMusiCNN" in algos:
            ours = decode.melspectrogram(audio, decode.EFFNET)
            ours_flat = ours.reshape(-1, ours.shape[-1]) if ours.size else ours
            compare("EffNet / MusiCNN input (96 bands)", ours_flat,
                    essentia_mel(path, "TensorflowInputMusiCNN"))
        if "TensorflowInputVGGish" in algos:
            ours = decode.melspectrogram(audio, decode.YAMNET)
            ours_flat = ours.reshape(-1, ours.shape[-1]) if ours.size else ours
            compare("YAMNet / VGGish input (64 bands)", ours_flat,
                    essentia_mel(path, "TensorflowInputVGGish"))
    return 0


def _load(path: str):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    return decode._resample(mono, sr), 16000


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
