"""Audio -> mel patches. The part that actually costs the wall-clock time.

Measured on this machine: the MODEL runs at ~2 ms per patch on CPU, so for any
library the model is not the bottleneck — decoding and mel computation are. That is
why this module is the one that gets parallelised, and the model does not.

Everything here is a pure function of a file path, which is what lets it run in a
process pool: no shared state, no database handle, nothing to lock.

⚠️ THE MEL PARAMETERS BELOW ARE NOT YET VALIDATED against Essentia's own
``TensorflowInputMusiCNN``. They follow the documented MusiCNN recipe, but the
failure mode if they are subtly wrong is the dangerous one: **nothing crashes and the
labels come out confident, plausible and meaningless.** Until a known-good reference
comparison is done, treat any tag this pipeline produces as unverified. The seam is
deliberate — ``melspectrogram`` is the only thing that has to change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf

TARGET_SR = 16000
MAX_SECONDS = 30.0          # a 10-minute stem tells us nothing a preview does not


@dataclass(frozen=True)
class MelConfig:
    """The two models want DIFFERENT spectrograms — same idea, different constants.

    Getting these swapped produces no error, just confident nonsense, so each model
    carries its own config rather than sharing a global.
    """
    name: str
    frame_size: int
    hop_size: int
    n_mels: int
    patch_frames: int
    log_shift: float
    log_scale: float
    fmin: float = 0.0
    fmax: float | None = None
    unit_area: bool = True
    """Divide each triangular filter by its bandwidth (Essentia's 'unit_tri').

    This is NOT cosmetic. Bandwidths run to hundreds of Hz, so area-normalising
    shrinks the mel values by orders of magnitude. Feed that to YAMNet, whose
    compression is ``log(mel + 0.001)``, and every band floors at log(0.001) — the
    model then reports **Silence** at 1.00 for perfectly loud audio. Google's YAMNet
    filterbank is not area-normalised; Essentia's is.
    """


#: Discogs-EffNet: [batch, 128, 96], the MusiCNN-style input.
EFFNET = MelConfig("effnet", frame_size=512, hop_size=256, n_mels=96,
                   patch_frames=128, log_shift=1.0, log_scale=10000.0)

#: YAMNet: [batch, 96, 64] — 25 ms window, 10 ms hop, 64 bands over 125-7500 Hz,
#: and log(mel + 0.001) rather than log10(1 + 10000*mel).
YAMNET = MelConfig("yamnet", frame_size=400, hop_size=160, n_mels=64,
                   patch_frames=96, log_shift=0.001, log_scale=1.0,
                   fmin=125.0, fmax=7500.0, unit_area=False)


@dataclass
class Decoded:
    path: str
    patches: dict[str, np.ndarray] | None   # config name -> [n, frames, mels]
    duration_sec: float | None
    sample_rate: int | None
    channels: int | None
    error: str | None = None


def _resample(x: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return x
    try:
        import soxr
        return soxr.resample(x, src_sr, TARGET_SR)
    except ImportError:
        # Linear fallback: worse than soxr and it WILL colour the mel bands, so it
        # is a last resort rather than a choice.
        n = int(round(len(x) * TARGET_SR / src_sr))
        return np.interp(np.linspace(0, len(x) - 1, n),
                         np.arange(len(x)), x).astype(np.float32)


def _hz_to_mel(f):     # Slaney
    f = np.asarray(f, dtype=np.float64)
    lin, log = f / 200.0 * 3.0, None
    min_log_hz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    log = min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep
    return np.where(f < min_log_hz, lin, log)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    min_log_hz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    return np.where(m < min_log_mel, 200.0 * m / 3.0,
                    min_log_hz * np.exp(logstep * (m - min_log_mel)))


def _mel_filterbank(cfg: MelConfig, sr=TARGET_SR):
    """Triangular filters, unit-area normalised (Essentia's 'unit_tri')."""
    fmax = cfg.fmax if cfg.fmax is not None else sr / 2
    fft_freqs = np.linspace(0, sr / 2, cfg.frame_size // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(cfg.fmin), _hz_to_mel(fmax),
                                   cfg.n_mels + 2))
    fb = np.zeros((cfg.n_mels, len(fft_freqs)), dtype=np.float64)
    for i in range(cfg.n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (fft_freqs - lo) / max(mid - lo, 1e-10)
        right = (hi - fft_freqs) / max(hi - mid, 1e-10)
        fb[i] = np.maximum(0.0, np.minimum(left, right))
        if cfg.unit_area and hi - lo > 0:
            fb[i] *= 2.0 / (hi - lo)
    return fb.astype(np.float32)


# Built once per process, not per file: a worker analyses thousands of files and the
# filterbank never changes.
_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _bank_and_window(cfg: MelConfig):
    if cfg.name not in _CACHE:
        _CACHE[cfg.name] = (_mel_filterbank(cfg),
                            np.hanning(cfg.frame_size).astype(np.float32))
    return _CACHE[cfg.name]


def melspectrogram(audio: np.ndarray, cfg: MelConfig = EFFNET) -> np.ndarray:
    """Mono 16 kHz float32 -> [n_patches, cfg.patch_frames, cfg.n_mels]."""
    bank, window = _bank_and_window(cfg)
    n_frames = 1 + max(0, (len(audio) - cfg.frame_size) // cfg.hop_size)
    if n_frames < cfg.patch_frames:
        need = cfg.frame_size + (cfg.patch_frames - 1) * cfg.hop_size
        # TILE the sample, do not zero-pad it. YAMNet's patch is 0.96 s; a 0.2 s perc
        # hit padded with zeros is 80% silence, and the model — which averages over
        # the patch — duly reports "Silence" at high confidence for a perfectly loud
        # sample. Repeating preserves the spectral character across the window.
        if len(audio) and need > len(audio):
            audio = np.tile(audio, int(np.ceil(need / len(audio))))[:need]
        else:
            audio = np.pad(audio, (0, max(0, need - len(audio))))
        n_frames = cfg.patch_frames

    # One strided view instead of a Python loop over frames — the difference between
    # a few ms and a few hundred ms per file.
    frames = np.lib.stride_tricks.sliding_window_view(
        audio, cfg.frame_size)[::cfg.hop_size][:n_frames]
    spec = np.abs(np.fft.rfft(frames * window, axis=-1)).astype(np.float32)
    mel = spec @ bank.T
    if cfg.log_scale == 1.0:
        mel = np.log(cfg.log_shift + mel)            # YAMNet: log(mel + 0.001)
    else:
        mel = np.log10(cfg.log_shift + cfg.log_scale * mel)   # MusiCNN-style
    mel = mel.astype(np.float32)

    n_patches = len(mel) // cfg.patch_frames
    if n_patches == 0:
        return np.zeros((0, cfg.patch_frames, cfg.n_mels), dtype=np.float32)
    return mel[:n_patches * cfg.patch_frames].reshape(
        n_patches, cfg.patch_frames, cfg.n_mels)


def process_file(path: str) -> Decoded:
    """Worker entry point. NEVER raises — a bad file is data, not a crash.

    One unreadable file among a million must not take down a pool worker and, with
    it, the rest of the batch.
    """
    try:
        with sf.SoundFile(path) as f:
            sr, channels = f.samplerate, f.channels
            want = int(MAX_SECONDS * sr)
            audio = f.read(frames=want, dtype="float32", always_2d=True)
            duration = len(f) / sr if sr else None
    except Exception as exc:                                   # noqa: BLE001
        return Decoded(path, None, None, None, None, f"{type(exc).__name__}: {exc}"[:300])

    if audio.size == 0:
        return Decoded(path, None, duration, sr, channels, "empty file")

    mono = audio.mean(axis=1).astype(np.float32)
    try:
        # Decode ONCE, then compute both spectrograms from the same signal. Decoding
        # is the expensive step; a second mel is nearly free.
        mono = _resample(mono, sr)
        patches = {cfg.name: melspectrogram(mono, cfg) for cfg in (EFFNET, YAMNET)}
    except Exception as exc:                                   # noqa: BLE001
        return Decoded(path, None, duration, sr, channels,
                       f"{type(exc).__name__}: {exc}"[:300])
    return Decoded(path, patches, duration, sr, channels, None)
