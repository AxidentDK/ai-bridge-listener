"""What the listener measures that the BRIDGE does not: a fundamental.

Everything both programs share now lives in ``shared_dsp`` — onsets, tempo, key,
chroma, stereo width, envelope, loudness — and is imported here rather than copied.
That module is the single source of truth; ``tests/test_shared_dsp_sync.py`` fails the
build if the bridge's copy diverges by a single byte.

What is left in this file is the half that is genuinely ours: YIN pitch detection and
the routing policy that decides when to ask for it. The bridge has no pitch estimator,
so there is nothing to share and nothing to drift.

WHY THE SPLIT EXISTS. The same measurement bug appeared in both programs three times in
one night. The worst was a tempo fix made here at 01:00 and reproduced verbatim in the
bridge at 03:00 — with the paragraph explaining the bug copied across intact. Reading a
warning is not executing one, so the check is now a SHA-256 that runs in both test
suites rather than a comment asking people to be careful.
"""
from __future__ import annotations

import numpy as np

from .shared_dsp import (  # noqa: F401  (re-exported: callers and tests import these)
    ANALYSIS_SR,
    KIND_LOOP,
    KIND_ONE_SHOT,
    LOOP_MIN_ONSETS,
    LOOP_MIN_SPAN,
    ONE_SHOT_MAX_ONSETS,
    ONE_SHOT_SECONDS,
    Prepared,
    chroma_of,
    classify,
    envelope,
    key_from_chroma,
    loudness_lufs,
    measure,
    onset_strength,
    onset_times,
    prepare,
    spectral_flatness,
    stereo,
    tempo,
)
from . import shared_dsp

#: Bump when anything that gets STORED changes — including anything in ``shared_dsp``,
#: since this string is what makes a re-scan actually re-analyse.
#: feat2 = pitch estimates must have spectral support (subsonic subharmonics were
#:         being reported with 0.7-0.8 confidence); bar count stored with tempo.
#: feat3 = tempo scored on three multiplied terms (autocorrelation, a narrowed
#:         perceptual prior, events-per-beat): 57% -> 69% exact.
#: feat4 = four bugs found by reviewing this file, incl. a prior width chosen on the
#:         WORST tempo band rather than the mean.
#: feat5 = chroma resolved low enough to see a bass note, and a tonal-contrast floor
#:         so a flat histogram gets no key at all.
#: feat6 = the shared DSP core. The maths is bit-identical to feat5 — proven on 120
#:         real files — but PRE-PROCESSING is now owned by `shared_dsp.prepare`, so
#:         the mono signal these measurements see comes from its resampler rather than
#:         soxr. That moves a handful of values (2 of 42 tempos by 0.1 BPM), which is
#:         exactly why it needs a version of its own.
#: feat7 = the chroma band starts at A0 instead of A1. A square or saturated sub has no
#:         even harmonics, so with the old floor its fundamental was discarded and the
#:         loudest survivor was the THIRD harmonic — every such bassline named a perfect
#:         fifth above its true root. 2% of real bass loops change key; the range from
#:         C2 up is untouched. Found with Gemini, who named the mechanism from the
#:         constant alone.
FEATURE_VERSION = "feat7"

# --- pitch: the one measurement the bridge has no counterpart for --------------------

_YIN_WIN = 2048              # 128 ms at 16 kHz — five periods of a 40 Hz 808
_YIN_THRESHOLD = 0.15
_YIN_MIN_HZ, _YIN_MAX_HZ = 25.0, 4000.0
#: An 808's transient is broadband noise and its second harmonic is often LOUDER than
#: its fundamental, so YIN latches an octave up. Two guards: start after the transient,
#: and low-pass anything bass-dominant before estimating.
_PITCH_SKIP_AFTER_PEAK_S = 0.05
_BASS_LOWPASS_HZ = 300.0
_BASS_DOMINANT_FRACTION = 0.5
#: Above this spectral flatness the file is noise-like and gets no pitch at all. The
#: bridge's describe layer already calls >0.35 "noisy"; the same line is drawn here.
_PITCH_MAX_FLATNESS = 0.35
#: Below this YIN confidence the estimate is refused. Set from observed material, and
#: the honest note is that it was set on a HANDFUL of files: a snare that YIN put at
#: 28.7 Hz scored 0.30, while estimates that survive inspection — an 808 at 52 Hz, a
#: kick at 47 Hz, a snare at 99 Hz — scored 0.41 to 0.99. The confidence is STORED
#: rather than only applied, so a caller can be stricter without a re-scan.
_YIN_MIN_CONFIDENCE = 0.4
#: A claimed fundamental must carry at least this share of the strongest partial's
#: magnitude, or it is not in the sound. Deliberately low — a missing-fundamental
#: timbre is real and common — but a subsonic subharmonic scores essentially zero.
_PITCH_SUPPORT = 0.05
#: FFT length for that support check. Not the analysis window — a zero-padded one, to
#: get bins fine enough to confirm a bass note. See ``_agree_with_spectrum``.
_SUPPORT_FFT = 16384

_NOTE_NAMES = shared_dsp.NOTE_NAMES


def _lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """Brick-wall low-pass in the frequency domain, on one short analysis window."""
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(len(x), 1.0 / sr) > cutoff] = 0.0
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def pitch(mono: np.ndarray, peak_idx: int, sr: int = ANALYSIS_SR) -> dict:
    """YIN fundamental for a one-shot, as Hz, MIDI note and a confidence.

    The difference function is computed through an FFT autocorrelation rather than the
    textbook double loop — same result, and the naive form would cost more than the
    neural network does.
    """
    start = peak_idx + int(_PITCH_SKIP_AFTER_PEAK_S * sr)
    if start + _YIN_WIN > len(mono):
        start = max(0, len(mono) - _YIN_WIN)
    x = mono[start:start + _YIN_WIN].astype(np.float64)
    if len(x) < _YIN_WIN // 2 or not np.any(x):
        return {}
    x = np.pad(x, (0, _YIN_WIN - len(x)))

    # Bass-dominant material gets low-passed first, or YIN locks to the louder second
    # harmonic and reports the octave above.
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    total = float((spec ** 2).sum()) + 1e-20
    if float((spec[freqs < _BASS_LOWPASS_HZ] ** 2).sum()) / total > _BASS_DOMINANT_FRACTION:
        x = _lowpass(x, sr, _BASS_LOWPASS_HZ)

    w = len(x)
    power = float(np.dot(x, x))
    if power <= 1e-12:
        return {}
    fft = np.fft.rfft(x, n=2 * w)
    acf = np.fft.irfft(fft * np.conj(fft))[:w]
    cumulative = np.concatenate(([0.0], np.cumsum(x ** 2)))
    tail = cumulative[w] - cumulative[:w]                     # energy of x[tau:]
    head = cumulative[w - np.arange(w)]                       # energy of x[:w-tau]
    diff = np.maximum(head + tail - 2.0 * acf, 0.0)

    tau_min = max(2, int(sr / _YIN_MAX_HZ))
    tau_max = min(w - 1, int(sr / _YIN_MIN_HZ))
    if tau_max <= tau_min:
        return {}
    # Cumulative mean normalised difference: what makes YIN robust to amplitude.
    cmnd = np.ones(w)
    running = np.cumsum(diff[1:]) / np.arange(1, w)
    cmnd[1:] = diff[1:] / np.maximum(running, 1e-12)

    window = cmnd[tau_min:tau_max]
    below = np.nonzero(window < _YIN_THRESHOLD)[0]
    if below.size:
        # First dip under the threshold, then walk to that dip's local minimum —
        # taking the GLOBAL minimum instead is the classic octave error.
        tau = tau_min + int(below[0])
        while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
    else:
        tau = tau_min + int(np.argmin(window))
    confidence = float(1.0 - min(1.0, cmnd[tau]))
    if confidence < _YIN_MIN_CONFIDENCE:
        return {}

    # Parabolic interpolation around the dip: without it the estimate quantises to
    # integer lag, which at 16 kHz is ~10 cents low down and much worse up high.
    if 0 < tau < w - 1:
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        # tau + (a - c) / (2 * (a - 2b + c)). The denominator is POSITIVE at a
        # minimum; writing it as (2b - a - c) flips the sign and shifts the estimate
        # away from the true dip — which is how a 440 Hz tone read back as 448.9.
        denom = 2.0 * (a - 2.0 * b + c)
        if abs(denom) > 1e-12:
            tau = tau + (a - c) / denom
    hz = float(sr) / float(tau)
    if not (_YIN_MIN_HZ <= hz <= _YIN_MAX_HZ):
        return {}

    # DOES THE SPECTRUM AGREE? YIN works on periodicity alone, so on complex or
    # polyphonic material it happily locks to a SUBHARMONIC — a period that divides
    # every real partial while carrying no energy of its own. Measured on real files:
    # a stab whose strongest partial is E2 at 165 Hz was reported at 27.4 Hz, and a
    # chord whose partials are C#3/G#3/F#2 at 30.7 Hz. Both scored 0.72-0.84
    # confidence, so the confidence floor cannot catch this; only looking at the
    # spectrum can. Neither answer had any energy within a semitone of itself.
    hz, confidence = _agree_with_spectrum(x, sr, hz, confidence)
    if hz is None:
        return {}
    return {"pitch_hz": round(hz, 2),
            "pitch_midi": int(round(69 + 12 * np.log2(hz / 440.0))),
            "pitch_confidence": round(confidence, 3)}


def _agree_with_spectrum(x: np.ndarray, sr: int, hz: float,
                         confidence: float) -> tuple[float | None, float]:
    """Keep a YIN estimate only if the spectrum has energy there; else use the peak.

    Returns (hz, confidence), or (None, _) when nothing musical is present at all.
    The fallback is the strongest spectral peak, which for the cases above is the
    named note itself — and its confidence is cut, because an estimate that had to be
    overruled is a weaker claim than one the two methods agreed on.
    """
    # ZERO-PADDED, and it matters down low. A 2048-point FFT at 16 kHz has 7.81 Hz
    # bins, so for four of the 36 notes between C0 and B2 — 34.65, 36.71, 43.65 and
    # 58.27 Hz among them — NO bin falls within a semitone of the note. The support
    # check would then reject a perfectly good YIN estimate and fall back to the
    # nearest strong bin, which is exactly the 808 and sub-bass material the two
    # guards above exist to get right. Padding to 16384 gives 0.98 Hz bins and costs
    # microseconds.
    padded = np.zeros(_SUPPORT_FFT, dtype=np.float64)
    window = x * np.hanning(len(x))
    padded[:min(len(window), _SUPPORT_FFT)] = window[:_SUPPORT_FFT]
    spec = np.abs(np.fft.rfft(padded))
    freqs = np.fft.rfftfreq(_SUPPORT_FFT, 1.0 / sr)
    usable = (freqs >= _YIN_MIN_HZ) & (freqs <= _YIN_MAX_HZ)
    if not usable.any() or not spec[usable].any():
        return None, confidence
    spec_u, freqs_u = spec[usable], freqs[usable]
    peak = float(spec_u.max())

    # Within a semitone of the claimed fundamental.
    near = np.abs(np.log2(np.maximum(freqs_u, 1e-9) / hz)) < (1.0 / 12.0)
    if near.any() and float(spec_u[near].max()) >= _PITCH_SUPPORT * peak:
        return hz, confidence
    strongest = float(freqs_u[int(np.argmax(spec_u))])
    if not (_YIN_MIN_HZ <= strongest <= _YIN_MAX_HZ):
        return None, confidence
    return strongest, confidence * 0.6

def analyze(audio: np.ndarray, sr: int, mono: np.ndarray | None = None,
            duration: float | None = None) -> dict:
    """Everything measurable about one file, routed by what the file IS.

    ``audio`` is the decoded signal at its own rate, still stereo — width has to be
    measured before any downmix. ``mono`` is accepted and IGNORED for measurement: the
    listener has a 16 kHz mono to hand from the mel pass, but taking it would mean this
    program picks its own resampling while the bridge picks another, which is precisely
    the drift ``shared_dsp.prepare`` exists to prevent. It costs one extra resample per
    file and buys the guarantee that both programs measure the same signal.
    """
    prepared = prepare(audio, sr, duration)
    props = measure(prepared)
    peak_idx = props.pop("peak_index", 0)
    props.pop("flatness", None)

    if props.get("kind") == KIND_ONE_SHOT:
        # A single fundamental, gated twice. Meaningless for a chord bed, and
        # meaningless for noise: a snare has none, and YIN will return one anyway — a
        # real snare in this library came back at 25 Hz.
        if spectral_flatness(prepared.mono) <= _PITCH_MAX_FLATNESS:
            props.update(pitch(prepared.mono, peak_idx))
    return props
