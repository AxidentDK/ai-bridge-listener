"""Audio -> mel patches. The part that actually costs the wall-clock time.

Measured on this machine: the MODEL runs at ~2 ms per patch on CPU, so for any
library the model is not the bottleneck — decoding and mel computation are. That is
why this module is the one that gets parallelised, and the model does not.

Everything here is a pure function of a file path, which is what lets it run in a
process pool: no shared state, no database handle, nothing to lock.

✅ THE MEL PARAMETERS BELOW ARE VALIDATED against Essentia's own
``TensorflowInputMusiCNN`` and ``TensorflowInputVGGish``. They were NOT, and this
paragraph used to say so — correctly, at the time. Five constants were wrong and every
one of them produced correct shapes and plausible values, which is the failure mode
this module has to be paranoid about: nothing crashes, and the labels come out
confident and meaningless. They were found by reading Essentia's C++ source rather
than fitting curves to its output, and the fix is confirmed downstream: tagging the
same 640 files through our mel and through Essentia's now agrees in all 16 categories.
See ``docs/MEL_VALIDATION.md`` for the five bugs and the red herrings.

The warning is kept here in its corrected form deliberately. A stale caution is worse
than none — it tells every future reader to distrust output that has since been proven
correct — and this project has now removed the same obsolete warning from two other
places. If ``melspectrogram`` changes again, re-run ``tools/validate_mel_wsl.py``
before trusting a single tag, and bump ``MEL_VERSION`` below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf

from . import features

TARGET_SR = 16000
MAX_SECONDS = 30.0          # a 10-minute stem tells us nothing a preview does not

#: Bump this whenever the spectrogram changes. It is part of the analyzer version
#: (see run.Tagger.version), because resumability compares versions to decide what to
#: skip — and a mel change alters every stored tag while leaving the model set
#: identical. Getting this wrong once already caused a "re-scan" that skipped all
#: 22,100 files and reported success.
#: v2 = verified against Essentia (power spectrum for MusiCNN; 512-pt FFT, htk mel,
#:      mel-domain slopes and ln(x+0.01) for VGGish).
MEL_VERSION = "mel2"


@dataclass(frozen=True)
class MelConfig:
    """One model's spectrogram recipe, transcribed from Essentia's source.

    These are not tunable knobs — they are the exact analysis parameters the models
    were trained with, hardcoded in Essentia's `TensorflowInputMusiCNN` and
    `TensorflowInputVGGish`. Every field below was wrong at least once, and not one
    of those errors produced an exception: the shapes stayed correct and only the
    values moved, so the models answered confidently from the wrong input. Change
    nothing here without re-running `tools/validate_mel_wsl.py`.
    """
    name: str
    frame_size: int          # samples handed to the window
    fft_size: int            # FFT length; frames are zero-padded up to this
    hop_size: int
    n_mels: int
    patch_frames: int
    fmin: float
    fmax: float
    warping: str             # "slaney" | "htk"   — mel scale for the band EDGES
    weighting: str           # "linear" | "warping" — domain for the triangle SLOPES
    power: bool              # accumulate |X|^2 (power) rather than |X| (magnitude)
    normalize: str           # "unit_tri" (÷ triangle area) | "unit_max" (peak 1, no ÷)
    log_kind: str            # "log10_scaled": log10(scale*x + shift); "ln": ln(x + shift)
    log_shift: float
    log_scale: float = 1.0


#: Discogs-EffNet, via TensorflowInputMusiCNN. Frame 512, no zero-padding.
#: Slaney-spaced edges with slopes measured in linear Hz, POWER accumulation,
#: unit_tri normalisation, then log10(10000*x + 1).
EFFNET = MelConfig("effnet", frame_size=512, fft_size=512, hop_size=256,
                   n_mels=96, patch_frames=128, fmin=0.0, fmax=8000.0,
                   warping="slaney", weighting="linear", power=True,
                   normalize="unit_tri", log_kind="log10_scaled",
                   log_shift=1.0, log_scale=10000.0)

#: YAMNet, via TensorflowInputVGGish. A 400-sample frame ZERO-PADDED to a 512-point
#: FFT (so 257 bins, not 201), htk mel edges with slopes in the mel domain,
#: MAGNITUDE accumulation, unnormalised triangles (peak 1), then ln(x + 0.01).
YAMNET = MelConfig("yamnet", frame_size=400, fft_size=512, hop_size=160,
                   n_mels=64, patch_frames=96, fmin=125.0, fmax=7500.0,
                   warping="htk", weighting="warping", power=False,
                   normalize="unit_max", log_kind="ln", log_shift=0.01)


@dataclass
class Decoded:
    path: str
    patches: dict[str, np.ndarray] | None   # config name -> [n, frames, mels]
    duration_sec: float | None
    sample_rate: int | None
    channels: int | None
    error: str | None = None
    #: Measured facts from ``features.analyze`` — pitch or key, tempo, width, envelope,
    #: loudness. None if measurement failed; that is not fatal to the tags.
    properties: dict | None = None


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


# Two mel scales, and they are NOT interchangeable: MusiCNN uses Slaney, VGGish uses
# HTK. Both appear in Essentia's essentiamath.h; using one for both was a real bug.
_SLANEY_SLOPE = 3.0 / 200.0          # linSlope
_SLANEY_BREAK_HZ = 1000.0
_SLANEY_BREAK_MEL = _SLANEY_BREAK_HZ * _SLANEY_SLOPE          # 15.0
_SLANEY_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(f, formula: str):
    f = np.asarray(f, dtype=np.float64)
    if formula == "htk":                                       # hz2mel10
        return 2595.0 * np.log10(f / 700.0 + 1.0)
    log = _SLANEY_BREAK_MEL + np.log(np.maximum(f, 1e-10)
                                     / _SLANEY_BREAK_HZ) / _SLANEY_LOGSTEP
    return np.where(f < _SLANEY_BREAK_HZ, f * _SLANEY_SLOPE, log)


def _mel_to_hz(m, formula: str):
    m = np.asarray(m, dtype=np.float64)
    if formula == "htk":                                       # mel102hz
        return 700.0 * (np.power(10.0, m / 2595.0) - 1.0)
    return np.where(m < _SLANEY_BREAK_MEL, m / _SLANEY_SLOPE,
                    _SLANEY_BREAK_HZ * np.exp(_SLANEY_LOGSTEP
                                              * (m - _SLANEY_BREAK_MEL)))


def _mel_filterbank(cfg: MelConfig, sr=TARGET_SR):
    """Triangular filters, transcribed from Essentia's TriangularBands::createFilters.

    Two subtleties that a naive librosa-style filterbank gets wrong:

    * The band EDGES are mel-spaced, but the triangle SLOPES are measured in whichever
      domain ``weighting`` names — linear Hz for MusiCNN, mel for VGGish. Mixing the
      two is the classic mismatch and it changes every coefficient.
    * ``unit_max`` means NO normalisation at all (raw triangles peaking at 1.0), not
      "scaled to peak 1". Only unit_tri/unit_sum divide, and unit_tri divides by the
      THEORETICAL triangle area (fstep1+fstep2)/2 rather than the summed weights.
    """
    n_bins = cfg.fft_size // 2 + 1
    freq_scale = (sr / 2.0) / (n_bins - 1)                     # Hz per FFT bin

    # Edges by REPEATED float32 ACCUMULATION, matching calculateFilterFrequencies:
    #     melFreq = low; for i: edge[i] = inv(melFreq); melFreq += increment;
    # Essentia is float32 throughout (typedef float Real), and accumulated rounding
    # drifts from the closed form np.linspace would give. It is a small difference
    # that lands on the band EDGES, where it moves which FFT bins fall inside a
    # triangle — so it is not as cosmetic as it looks.
    low_mel = np.float32(_hz_to_mel(cfg.fmin, cfg.warping))
    high_mel = np.float32(_hz_to_mel(cfg.fmax, cfg.warping))
    increment = np.float32((high_mel - low_mel) / np.float32(cfg.n_mels + 1))
    mel_freq = low_mel
    edge_list = []
    for _ in range(cfg.n_mels + 2):
        edge_list.append(float(_mel_to_hz(np.float32(mel_freq), cfg.warping)))
        mel_freq = np.float32(mel_freq + increment)
    edges = np.array(edge_list, dtype=np.float64)

    def w(x):
        return x if cfg.weighting == "linear" else _hz_to_mel(x, cfg.warping)

    fb = np.zeros((cfg.n_mels, n_bins), dtype=np.float64)
    for i in range(cfg.n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        fstep1 = float(w(mid) - w(lo))
        fstep2 = float(w(hi) - w(mid))
        if fstep1 <= 0 or fstep2 <= 0:
            continue
        # Inclusive ceil..floor over bins, exactly as Essentia iterates them.
        jbegin = int(np.ceil(lo / freq_scale))
        jend = int(np.floor(hi / freq_scale))
        for j in range(max(jbegin, 0), min(jend, n_bins - 1) + 1):
            binfreq = j * freq_scale
            if binfreq < mid:
                fb[i, j] = (float(w(binfreq)) - float(w(lo))) / fstep1
            else:
                fb[i, j] = (float(w(hi)) - float(w(binfreq))) / fstep2
        if cfg.normalize == "unit_tri":
            fb[i] /= (fstep1 + fstep2) / 2.0
        elif cfg.normalize == "unit_sum":
            total = fb[i].sum()
            if total > 0:
                fb[i] /= total
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
    # rfft's `n` zero-pads on the RIGHT, which is what VGGish wants: a 400-sample
    # frame into a 512-point FFT, giving 257 bins rather than 201. Getting this wrong
    # changes the frequency resolution itself, so no filterbank fix could compensate.
    spec = np.abs(np.fft.rfft(frames * window, n=cfg.fft_size, axis=-1))
    # Essentia squares PER BIN before the filterbank sum, so this is sum(w * |X|^2),
    # not sum(w * |X|) squared. Those differ.
    mel = (spec ** 2 if cfg.power else spec) @ bank.T
    if cfg.log_kind == "ln":
        mel = np.log(mel + cfg.log_shift)                       # VGGish: ln(x + 0.01)
    else:
        mel = np.log10(cfg.log_scale * mel + cfg.log_shift)     # MusiCNN: log10(1e4·x+1)
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

    # Measured features come off the SAME decode — `audio` is still stereo at its own
    # rate, which is the only moment stereo width can be measured, and `mono` is the
    # 16 kHz signal the mel already needed. A failure here costs the properties, never
    # the tags: measurement is the optional half.
    try:
        properties = features.analyze(audio, sr, mono, duration)
    except Exception as exc:                                   # noqa: BLE001
        properties = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return Decoded(path, patches, duration, sr, channels, None, properties)
