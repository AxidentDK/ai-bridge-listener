"""Measured facts about a file — the half that does NOT need a model.

The heads in ``run.py`` answer perceptual questions (what does this sound like?).
This module answers mechanical ones (how loud, how wide, how long does it ring, what
note is it, is it one hit or a bar of them?). Those are cheap, exact, and — as it
turned out — completely missing: the ``properties`` table was in the schema from the
start and had **zero rows** across 22,100 files.

ROUTING IS THE POINT, not an optimisation. Onset density decides what gets computed:
a one-shot gets a fundamental, a loop gets a key and a tempo. Computing both for
everything is not merely wasteful, it fills the table with confident junk — a BPM
derived from one transient, a "fundamental" of an atonal breakbeat — and the bridge
downstream has no way to tell that from a real measurement. Gemini made this argument
during review and it was the right call; the original plan computed everything for
everything.

Two algorithms here are PORTED rather than invented, from the bridge's
``audio_analysis.py`` and ``describe.py``, so that both programs answer "F minor" and
"-14 LUFS" the same way. Where the port deviates it is marked and the reason given.

Everything is a pure function of arrays, so it runs inside the existing decode worker
pool with no shared state — and, importantly, off the SAME decode. Reading the file
again to measure it would have doubled the cost of the expensive step.
"""
from __future__ import annotations

import numpy as np

#: Bump when anything here changes what gets STORED. It joins the analyzer version
#: string for the same reason ``MEL_VERSION`` does — see the warning in ``decode``.
#: feat2 = pitch estimates must have spectral support (subsonic subharmonics were
#:         being reported with 0.7-0.8 confidence); bar count stored with tempo.
#: feat3 = tempo scored on three multiplied terms (autocorrelation, a NARROWED
#:         perceptual prior, events-per-beat likelihood): 57% -> 69% exact, with
#:         half-time errors down from 92 to 26. Confidence is now prominence times
#:         metrical margin, which stopped it pointing the wrong way.
#: feat4 = four bugs Gemini found by reviewing this file: zero-padded FFT for the
#:         pitch support check (bass notes had no bin within a semitone), decay
#:         bounded at the next onset, metrical rivals looked up outside the tempo
#:         search window, and a prior width chosen on the WORST tempo band.
FEATURE_VERSION = "feat4"

ANALYSIS_SR = 16000          # the mel pass already produced this; reuse it

# --- the router ------------------------------------------------------------------

#: Spectral flux, not high-frequency content. HFC over-triggers on exactly the
#: material a sample library is full of — shakers, vinyl crackle, noise sweeps — and
#: would report them as dense loops.
_FLUX_FFT, _FLUX_HOP = 1024, 256                      # 16 ms hop at 16 kHz
_MIN_ONSET_GAP_S = 0.05                               # 50 ms: two hits, not one flam
_MEDIAN_WIN = 9                                       # ~145 ms moving threshold
_FLUX_DELTA = 0.6                                     # above the local median, in SDs
#: A peak must also reach this fraction of the file's largest rise. Stops the adaptive
#: threshold from tracking down into the noise when there is nothing to find.
_FLUX_MIN_FRACTION = 0.15

#: Onset counts that decide the route. A single hit, or a hit with a flam or a short
#: pre-delay, stays a one-shot; a bar of anything has at least four.
ONE_SHOT_MAX_ONSETS = 2
LOOP_MIN_ONSETS = 4
#: A loop's onsets must cover this much of its length. Without it, ripple in a decay
#: counts as a rhythm — see the 808 case in ``classify``.
LOOP_MIN_SPAN = 0.5
#: Exactly three onsets is genuinely ambiguous, so it falls back to the duration rule
#: that used to decide everything. Not a fudge — a triplet fill and a three-hit loop
#: are the same measurement, and length is the only thing left to separate them.
ONE_SHOT_SECONDS = 2.048

KIND_ONE_SHOT = "one_shot"
KIND_LOOP = "loop"

# --- stereo ----------------------------------------------------------------------

#: Width is measured ABOVE this, never broadband. Mono bass under a wide top end is
#: the standard production shape — bass is summed to mono to keep club systems
#: phase-aligned — so a broadband number averages that to "narrow" and hides the
#: thing being asked about.
_WIDTH_MIN_HZ = 250.0
_WIDTH_FFT, _WIDTH_HOP = 2048, 1024
_WIDTH_MAX_FRAMES = 256      # bounded work: a 30 s file costs what a 3 s file costs

# --- pitch -----------------------------------------------------------------------

_YIN_WIN = 2048              # 128 ms at 16 kHz — five periods of a 40 Hz 808
_YIN_THRESHOLD = 0.15
_YIN_MIN_HZ, _YIN_MAX_HZ = 25.0, 4000.0
#: An 808's transient is broadband noise and its second harmonic is often LOUDER than
#: its fundamental, so YIN latches an octave up. Two guards, both from Gemini: start
#: after the transient, and low-pass anything bass-dominant before estimating.
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
#: rather than only applied, so a caller can be stricter than this without a re-scan.
_YIN_MIN_CONFIDENCE = 0.4
#: A claimed fundamental must carry at least this share of the strongest partial's
#: magnitude, or it is not in the sound. Deliberately low — a missing-fundamental
#: timbre is real and common — but a subsonic subharmonic scores essentially zero.
_PITCH_SUPPORT = 0.05
#: FFT length for that support check. Not the analysis window — a zero-padded one, to
#: get bins fine enough to confirm a bass note. See ``_agree_with_spectrum``.
_SUPPORT_FFT = 16384

# --- key (ported from the bridge's describe.py) ----------------------------------

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
_CHROMA_MIN_HZ, _CHROMA_MAX_HZ = 55.0, 5000.0


def _frames(x: np.ndarray, n_fft: int, hop: int, max_frames: int | None = None):
    """Windowed frames, optionally decimated to bound the work on long files."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    count = 1 + (len(x) - n_fft) // hop
    if count < 1:
        return np.zeros((0, n_fft), dtype=np.float32)
    starts = np.arange(count) * hop
    if max_frames and count > max_frames:
        starts = starts[np.linspace(0, count - 1, max_frames).astype(int)]
    idx = np.arange(n_fft)[None, :] + starts[:, None]
    return (x[idx] * np.hanning(n_fft)[None, :]).astype(np.float32)


def _median_filter(x: np.ndarray, win: int) -> np.ndarray:
    """Moving median via a strided view. Pure numpy — no scipy in this project.

    Padded with ZEROS, not edge values. Edge padding replicates the first sample
    across half the window, so a large opening value becomes its own local median and
    thresholds itself out — which silently swallowed the onset at t=0, the one most
    one-shots have. Outside the signal there is no flux, and zero says exactly that.
    """
    if len(x) < win:
        return np.full_like(x, float(np.median(x)) if len(x) else 0.0)
    pad = win // 2
    padded = np.pad(x, (pad, pad), mode="constant", constant_values=0.0)
    strided = np.lib.stride_tricks.sliding_window_view(padded, win)
    return np.median(strided, axis=-1)[:len(x)]


def onset_strength(mono: np.ndarray) -> np.ndarray:
    """The spectral-flux curve itself. Tempo wants the whole curve, not just the peaks."""
    padded = np.pad(mono, (_FLUX_FFT // 2, 0))
    frames = _frames(padded, _FLUX_FFT, _FLUX_HOP)
    if len(frames) < 2:
        return np.zeros(0)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    spec = np.vstack([np.zeros((1, spec.shape[1]), dtype=spec.dtype), spec])
    return np.maximum(0.0, np.diff(spec, axis=0)).sum(axis=1)


def onset_times(mono: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Onset positions in seconds, by spectral flux against a moving median."""
    # The frames behind this are CENTRED, and silence is prepended, in
    # ``onset_strength``. Two bugs die there. A Hann window is zero at its edges, so a
    # transient at sample 0 — where most one-shots put theirs — was multiplied away and
    # the file reported no onset at all. And timing a frame by its START rather than
    # its centre reported every onset half a window early, which showed up as a 35 ms
    # bias in exactly the intervals tempo is derived from.
    flux = onset_strength(mono)
    if not flux.size or not np.any(flux):
        return np.zeros(0)
    # An adaptive threshold alone is not enough: with no real onsets it collapses to
    # the noise floor and starts reporting it. A decaying sine — nothing whatsoever
    # happening after the first instant — produced five evenly spaced "onsets" that
    # way, and was routed as a loop. So a peak must ALSO be significant against the
    # largest rise in the file.
    threshold = np.maximum(_median_filter(flux, _MEDIAN_WIN) + _FLUX_DELTA * flux.std(),
                           _FLUX_MIN_FRACTION * float(flux.max()))
    gap = max(1, int(_MIN_ONSET_GAP_S * sr / _FLUX_HOP))
    hits, last = [], -10 ** 6
    for i in np.nonzero(flux > threshold)[0]:
        # A peak, not merely a crossing: a rising slope alone fires several times
        # across one transient's attack.
        if i - last >= gap and (i == 0 or flux[i] >= flux[i - 1]):
            # Sub-frame position by parabolic interpolation of the flux peak. The hop
            # is 16 ms, and rounding to it put a 120 BPM loop at 117.2 — small per
            # onset, but tempo comes from the interval between them, so it lands
            # squarely in the digits a producer reads as the answer.
            offset = 0.0
            if 0 < i < len(flux) - 1:
                a, b, c = flux[i - 1], flux[i], flux[i + 1]
                denom = 2.0 * (a - 2.0 * b + c)
                if abs(denom) > 1e-12:
                    offset = float(np.clip((a - c) / denom, -0.5, 0.5))
            # Frame i now begins at sample i*hop (the silent frame absorbed the +1),
            # and a transient sits at the START of the frame that first sees it.
            hits.append(max(0.0, (i + offset) * _FLUX_HOP / float(sr)))
            last = i
    return np.asarray(hits)


def classify(onsets: np.ndarray, duration: float | None) -> str:
    """One hit or a bar of them. The decision every other measurement hangs off.

    Count alone is not enough, and a real file showed why: an 808 bass drum whose
    decay ripples produced four "onsets" — all inside the first 340 ms of a 2.3 s
    file — and was routed as a loop, earning it a tempo of 96.9 BPM and a key of B
    major, both meaningless. A loop's onsets SPAN it; a one-shot's artefacts cluster
    at the front. So the span is required as well as the count.
    """
    count = len(onsets)
    if count <= ONE_SHOT_MAX_ONSETS:
        return KIND_ONE_SHOT
    spans = True
    if duration and count >= 2:
        spans = (float(onsets[-1] - onsets[0]) / duration) >= LOOP_MIN_SPAN
    if count >= LOOP_MIN_ONSETS and spans:
        return KIND_LOOP
    if duration is None:
        return KIND_ONE_SHOT
    return KIND_ONE_SHOT if duration < ONE_SHOT_SECONDS or not spans else KIND_LOOP


def stereo(audio: np.ndarray, sr: int) -> tuple[float | None, float | None]:
    """(width above 250 Hz, broadband correlation). Both None if the file is mono.

    Width is reported as S / (M + S): 0 is a mono signal, 0.5 is equal mid and side
    energy. The unbounded S/M ratio a producer might name goes to infinity on
    out-of-phase material, which makes it useless to rank or threshold on.

    Correlation is kept alongside because it answers a DIFFERENT question — near -1
    means the channels fight and the mono sum partially cancels, which is a defect
    worth naming and which a width number alone never reveals.
    """
    if audio.ndim < 2 or audio.shape[1] < 2:
        return None, None
    left, right = audio[:, 0].astype(np.float32), audio[:, 1].astype(np.float32)
    correlation = None
    if float(left.std()) > 1e-9 and float(right.std()) > 1e-9:
        correlation = round(float(np.corrcoef(left, right)[0, 1]), 3)

    mid, side = (left + right) * 0.5, (left - right) * 0.5
    freqs = np.fft.rfftfreq(_WIDTH_FFT, 1.0 / sr)
    band = freqs >= _WIDTH_MIN_HZ
    if not band.any():
        return None, correlation
    m_frames = _frames(mid, _WIDTH_FFT, _WIDTH_HOP, _WIDTH_MAX_FRAMES)
    s_frames = _frames(side, _WIDTH_FFT, _WIDTH_HOP, _WIDTH_MAX_FRAMES)
    if not len(m_frames):
        return None, correlation
    m_energy = float((np.abs(np.fft.rfft(m_frames, axis=1))[:, band] ** 2).sum())
    s_energy = float((np.abs(np.fft.rfft(s_frames, axis=1))[:, band] ** 2).sum())
    total = m_energy + s_energy
    if total <= 1e-20:
        return None, correlation
    return round(s_energy / total, 4), correlation


def envelope(mono: np.ndarray, onsets: np.ndarray, kind: str,
             sr: int = ANALYSIS_SR) -> tuple[float, float, int]:
    """(attack_ms, decay_ms, peak_index) — both timed from the FIRST hit.

    ``attack_ms`` is time from the start of the file to the peak of the first hit, and
    it is the tag that serves DECISION-MAKING rather than search: a reverse cymbal or a
    chopped vocal can carry 40 ms of run-up, so a sample sequenced flat on the grid
    lands late. Knowing the offset lets the bridge move the note instead.

    It is scoped to the first hit rather than the whole file because the global peak of
    a loop is wherever its loudest hit happens to fall — real measurements of 1.0 s,
    1.2 s and 1.5 s "attack" on drum loops, which describe nothing about the attack.

    ``decay_ms`` is a T60 estimated from the first 20 dB of decay and multiplied by
    three — the standard T20 extrapolation. Measuring a full 60 dB directly would find
    the noise floor of most produced samples rather than the tail. It is returned as 0
    for anything with a second hit: what a loop decays to is the next hit, not silence.
    """
    env = np.abs(mono)
    win = max(1, int(0.005 * sr))
    env = np.convolve(env, np.ones(win, dtype=np.float32) / win, mode="same")
    if not env.size or not np.any(env):
        return 0.0, 0.0, 0
    # Look for the peak only within the first hit — up to the second onset, if there
    # is one.
    limit = len(env)
    if len(onsets) > 1:
        limit = max(1, min(limit, int(onsets[1] * sr)))
    peak_idx = int(np.argmax(env[:limit]))
    peak = float(env[peak_idx])
    attack_ms = round(peak_idx * 1000.0 / sr, 1)
    if kind == KIND_LOOP:
        # What a loop decays to is its next hit, not silence. Gating this on the ONSET
        # COUNT instead cost every two-onset one-shot its decay — a kick with an
        # audible click is still a kick that rings.
        return attack_ms, 0.0, peak_idx

    # The tail STOPS at the next onset. A one-shot may legitimately have two onsets —
    # a flam, a kick with an audible click, a double hit — and measuring decay through
    # the second one measures the second hit, not the first's ring-out. A double-hit
    # kick whose first hit decays in ~150 ms was reporting 386 ms, because the
    # envelope never fell 20 dB before the second impact lifted it again.
    tail = env[peak_idx:limit] if limit > peak_idx else env[peak_idx:]
    below = np.nonzero(tail <= peak * (10.0 ** (-20.0 / 20.0)))[0]
    if below.size:
        decay_ms = round(float(below[0]) * 3.0 * 1000.0 / sr, 1)
    else:
        # Never fell 20 dB inside the file: it is still ringing at the end, so the
        # honest answer is "at least this long", not an extrapolation off a slope
        # that was never observed.
        decay_ms = round(len(tail) * 1000.0 / sr, 1)
    return attack_ms, decay_ms, peak_idx


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


def spectral_flatness(mono: np.ndarray) -> float:
    """Geometric over arithmetic mean of the spectrum: ~1 is noise, near 0 is a tone.

    The cheap answer to "does this even HAVE a pitch". A hi-hat and a snare do not, and
    an estimator asked anyway will still return a number.
    """
    frames = _frames(mono, _WIDTH_FFT, _WIDTH_HOP, _WIDTH_MAX_FRAMES)
    if not len(frames):
        return 1.0
    spec = np.abs(np.fft.rfft(frames, axis=1))
    geo = np.exp(np.mean(np.log(spec + 1e-12), axis=1))
    ari = np.mean(spec, axis=1) + 1e-12
    return float(np.mean(geo / ari))


def key_from_chroma(chroma: list[float]) -> dict:
    """Krumhansl-Schmuckler, ported so both programs mean the same thing by 'F minor'."""
    if not any(chroma):
        return {}
    scored = []
    for root in range(12):
        rotated = list(chroma[root:]) + list(chroma[:root])
        for profile, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            scored.append((_correlate(rotated, profile), root, mode))
    scored.sort(reverse=True)
    score, root, mode = scored[0]
    return {"key": _NOTE_NAMES[root], "scale": mode,
            # Strength is the MARGIN over the runner-up, not the raw correlation.
            # C major and A minor share every note and both correlate highly; the gap
            # between them is what says whether the answer means anything.
            "key_strength": round(float(max(0.0, score - scored[1][0])), 3)}


def _correlate(hist, profile) -> float:
    n = len(profile)
    mh, mp = sum(hist) / n, sum(profile) / n
    num = sum((hist[i] - mh) * (profile[i] - mp) for i in range(n))
    den = (sum((hist[i] - mh) ** 2 for i in range(n)) ** 0.5
           * sum((profile[i] - mp) ** 2 for i in range(n)) ** 0.5)
    return num / den if den else 0.0


def chroma_of(mono: np.ndarray, sr: int = ANALYSIS_SR) -> list[float]:
    frames = _frames(mono, _WIDTH_FFT, _WIDTH_HOP, _WIDTH_MAX_FRAMES)
    if not len(frames):
        return [0.0] * 12
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(_WIDTH_FFT, 1.0 / sr)
    usable = (freqs > _CHROMA_MIN_HZ) & (freqs < _CHROMA_MAX_HZ)
    if not usable.any():
        return [0.0] * 12
    pcs = np.round(12 * np.log2(freqs[usable] / 440.0) + 69).astype(int) % 12
    weights = spec[:, usable].sum(axis=0)
    return [float(weights[pcs == pc].sum()) for pc in range(12)]


#: Tempo search range, and the prior it is weighted toward. Autocorrelation is
#: genuinely ambiguous between a pulse and its double or half, so SOMETHING has to
#: break the tie; a log-normal prior around 120 BPM is the standard choice.
_BPM_MIN, _BPM_MAX, _BPM_PRIOR = 60.0, 190.0, 120.0
#: Standard deviation of that prior, in OCTAVES. It was 1.0, which is the same as
#: having no prior at all: one sigma of a whole octave tells the search that a tempo
#: and its double are near enough equally fine, which is precisely the tie it exists
#: to break. Half-time was the single largest error class — 92 of 236 wrong tempos.
#:
#: ⚠️ THIS IS A BLUNT INSTRUMENT AND THE WIDTH IS A COMPROMISE, not an optimum. A
#: prior centred on 120 BPM inevitably favours material near 120: swept per tempo
#: band, 0.30 scores 53/83/47 (slow/mid/fast) and 0.80 scores 54/66/53 — tightening it
#: buys mid-tempo accuracy by taking it from drum-and-bass and trap, and loosening it
#: does the reverse. No value is good everywhere, because the prior is *assuming* an
#: answer rather than measuring one.
#:
#: 0.45 is chosen on the WORST BAND rather than the mean: it scores 54/81/50 for the
#: same 69% overall as 0.30, so it dominates it. Judging on the average is what let a
#: 36% collapse on fast material hide behind a headline that barely moved — Gemini
#: predicted that collapse from these multipliers alone, before it was measured.
#: The real fix is for the events-per-beat term to carry the octave decision so the
#: prior can be flattened; it is not strong enough to do that alone yet.
_BPM_PRIOR_WIDTH = 0.45

#: EVENTS PER BEAT — the term that actually separates a tempo from its half, since
#: duration cannot: a file that is 4 bars at 100 BPM is also exactly 8 bars at 200.
#:
#: Measured over 546 loops whose filenames state their tempo: at the TRUE tempo the
#: median is 2.10 events per beat (p25 1.50, p75 2.67); read at half-time it doubles
#: to 4.19. Sweeping the anchor, 1.5 picked the true tempo over its half on 88% of
#: them, against 78% at 2.0 and 49% at 3.0.
#:
#: ⚠️ THIS ANCHOR BELONGS TO THE DETECTOR, NOT TO MUSIC. A 16th-note groove has four
#: musical events per beat; `onset_times` counts only what clears `_FLUX_MIN_FRACTION`
#: of the largest rise, so it counts STRUCTURAL impacts — kicks, main snares, stabs —
#: and drops ghost notes, quiet hats and delay tails. Change that threshold or the
#: onset function and these numbers must be re-measured, or they will quietly mislead.
#:
#: The target SLOPES with tempo rather than being fixed. Fast music leans on half-time
#: phrasing (the snare on 3, not on 2 and 4), so it carries FEWER structural events per
#: beat than slow music, not more — visible in the same data: files named 140+ BPM
#: average 4.25 onsets/s (1.6 per beat) against 3.20 (2.3 per beat) for those named
#: under 100.
_EPB_AT_SLOW, _EPB_AT_FAST = 2.2, 1.5      # anchors at 85 and 170 BPM
_EPB_SLOW_BPM, _EPB_FAST_BPM = 85.0, 170.0
_EPB_WIDTH = 0.9                            # sigma in octaves; deliberately generous


#: Bar counts a sample-library loop is actually cut to. Powers of two, because that is
#: how loops are sold; odd lengths exist but guessing them costs more than it wins.
_BAR_CANDIDATES = (1, 2, 4, 8, 16, 32)
#: How far the autocorrelation estimate may sit from a whole-bar tempo and still be
#: snapped to it — a factor of 1.5 in either direction. Wide on purpose: the errors
#: this corrects are metrical (3/4, 4/3, 2/3 of the beat), not small.
_SNAP_MAX_LOG2 = 0.6


def _epb_target(bpm):
    """Expected structural events per beat at a given tempo — a sloped line."""
    frac = ((np.log2(np.asarray(bpm, dtype=float)) - np.log2(_EPB_SLOW_BPM))
            / (np.log2(_EPB_FAST_BPM) - np.log2(_EPB_SLOW_BPM)))
    return _EPB_AT_SLOW + frac * (_EPB_AT_FAST - _EPB_AT_SLOW)


def _density_likelihood(bpms: np.ndarray, n_onsets: int,
                        duration: float | None) -> np.ndarray:
    """How plausible each candidate tempo's implied subdivision density is."""
    if not duration or duration <= 0 or n_onsets < 2:
        return np.ones(len(bpms))
    beats = duration * np.asarray(bpms, dtype=float) / 60.0
    epb = n_onsets / np.maximum(beats, 1e-9)
    return np.exp(-0.5 * (np.log2(epb / _epb_target(bpms)) / _EPB_WIDTH) ** 2)


def _metrical_margin(acf: np.ndarray, lag: int) -> float:
    """1 - (strongest metrical rival / winner), over the half, double, third and 3/2.

    The confidence this replaces measured how PERIODIC a loop is, which is not the
    same question and turned out to be anti-correlated with being right: 53% accurate
    above 0.9 confidence against 65% below it. A perfectly quantised loop has a
    perfect half-time alias, so "very rhythmic" was being scored as "very certain"
    exactly where the tempo was most ambiguous.
    """
    if not 0 < lag < len(acf):
        return 0.0
    winner = float(acf[lag])
    if winner <= 1e-12:
        return 0.0
    rivals = []
    for factor in (0.5, 2.0, 1.0 / 3.0, 3.0, 1.5, 2.0 / 3.0):
        rival = int(round(lag * factor))
        if 0 < rival < len(acf) and abs(rival - lag) > 1:
            rivals.append(float(acf[rival]))
    if not rivals:
        return 1.0
    return float(np.clip(1.0 - max(rivals) / winner, 0.0, 1.0))


def _snap_to_whole_bars(bpm: float, duration: float | None) -> tuple[float, int | None]:
    """Pull a tempo estimate onto the nearest tempo that makes the file whole bars.

    Sample-library loops are cut to a whole number of bars, so their LENGTH already
    encodes the tempo: at B BPM in 4/4, N bars last 240*N/B seconds. That is an
    independent measurement, and a far more precise one than autocorrelation.

    It settled a disagreement rather than assuming one: six loops whose filenames
    stated a BPM were being reported at exactly 4/3 of it, and checking their
    durations showed every one to be 4.000 bars at the STATED tempo. The filenames
    were right; the estimator was hearing a dotted-eighth layer as the beat.

    Returns the tempo and the BAR COUNT it implies, which is stored. That matters
    because a tempo and its double are NOT interchangeable — the same pulse written
    at 85 and at 170 differs in note values, so the grid, the swing and every
    quantise decision differ with it. The bar count is the evidence for which of the
    two a file actually is, and storing it makes the claim checkable instead of
    asserted.
    """
    if not duration or duration <= 0:
        return bpm, None
    pairs = [(240.0 * n / duration, n) for n in _BAR_CANDIDATES]
    pairs = [(c, n) for c, n in pairs if _BPM_MIN <= c <= _BPM_MAX]
    if not pairs:
        return bpm, None
    best, bars = min(pairs, key=lambda p: abs(np.log2(p[0] / bpm)))
    if abs(np.log2(best / bpm)) <= _SNAP_MAX_LOG2:
        return best, bars
    return bpm, None


def tempo(onsets: np.ndarray, flux: np.ndarray | None = None,
          sr: int = ANALYSIS_SR, duration: float | None = None) -> dict:
    """BPM by autocorrelation of the onset-strength curve.

    This replaced a median inter-onset interval, which measures the COMMONEST GAP
    rather than the beat — on a busy break that is a sixteenth, and the answer comes
    back at four times the tempo or some unrelated fraction of it. Filenames carrying
    their own BPM made the failure impossible to miss: a file named `...-170bpm` was
    reported at 120, and one named `120 BPM` at 156.

    Autocorrelation asks the right question — at what lag does the whole envelope best
    resemble itself — so a break with sixteenths still correlates strongest at its
    beat. The inter-onset median stays as a fallback for material too short to
    autocorrelate.
    """
    frames_per_sec = sr / float(_FLUX_HOP)
    if flux is not None and len(flux) >= 8:
        x = flux - flux.mean()
        n = 1 << int(np.ceil(np.log2(len(x) * 2)))
        spectrum = np.fft.rfft(x, n=n)
        acf = np.fft.irfft(spectrum * np.conj(spectrum))[:len(x)]
        # UNBIASED: divide by the number of overlapping terms at each lag. Raw
        # autocorrelation has fewer products to sum as the lag grows, so it slopes
        # downward on its own and systematically prefers short lags — fast tempos.
        # The signature was unmistakable once filenames were used as ground truth:
        # six loops came back at exactly 4/3 of their stated BPM, which is the beat
        # period shortened by one notch, not a musical relationship.
        acf = acf / np.maximum(np.arange(len(x), 0, -1), 1)
        if acf[0] > 0:
            acf = acf / acf[0]
            lo = max(2, int(round(frames_per_sec * 60.0 / _BPM_MAX)))
            hi = min(len(acf) - 2, int(round(frames_per_sec * 60.0 / _BPM_MIN)))
            if hi > lo:
                lags = np.arange(lo, hi + 1)
                bpms = 60.0 * frames_per_sec / lags
                prior = np.exp(-0.5 * (np.log2(bpms / _BPM_PRIOR)
                                       / _BPM_PRIOR_WIDTH) ** 2)
                # Score each candidate by its HARMONICS, not by its own peak alone. A
                # real beat repeats at 2x, 3x and 4x its period, so support accumulates
                # there; a spurious peak has none. Without this, a dotted-eighth layer
                # — the 3-3-2 pattern all over electronic music — correlates strongly
                # at 3/4 of the beat, and a whole pack of loops came back at exactly
                # 4/3 of the tempo their own filenames stated.
                harmonic = np.zeros(len(lags))
                for k, weight in ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25)):
                    idx = lags * k
                    valid = idx < len(acf)
                    harmonic[valid] += weight * acf[idx[valid]]
                # Three multiplied terms, not one deciding: how periodic the envelope
                # is, whether the tempo is one a human would tap, and whether the
                # implied subdivision density is plausible. A SOFT weight on purpose —
                # as a hard gate, one missed kick would punt a file into the wrong
                # octave over an otherwise flawless autocorrelation peak.
                scored = harmonic * prior * _density_likelihood(bpms, len(onsets),
                                                                duration)
                best = int(np.argmax(scored))
                lag = lags[best]
                # Sub-frame refinement, same reason as everywhere else here: the lag
                # quantises to 16 ms and tempo is derived from it.
                if 0 < lag < len(acf) - 1:
                    a, b, c = acf[lag - 1], acf[lag], acf[lag + 1]
                    denom = 2.0 * (a - 2.0 * b + c)
                    if abs(denom) > 1e-12:
                        lag = lag + float(np.clip((a - c) / denom, -0.5, 0.5))
                bpm = 60.0 * frames_per_sec / float(lag)
                bpm, bars = _snap_to_whole_bars(bpm, duration)
                if _BPM_MIN <= bpm <= _BPM_MAX:
                    # Confidence needs BOTH a strong pulse and an absence of metrical
                    # ambiguity, so the two are multiplied. Prominence alone answered
                    # "how rhythmic is this", which is why it pointed the wrong way.
                    band = acf[lo:hi + 1]
                    peak = float(band[best])
                    spread = float(band.std()) or 1e-9
                    clarity = (peak - float(band.mean())) / spread
                    prominence = min(1.0, max(0.0, clarity / 3.0))
                    # Measured on the SCORED candidates, not the raw autocorrelation.
                    # The raw version was the honest-looking choice — evidence only,
                    # no prior double-counted — and it produced a confidence that was
                    # merely flat instead of inverted: a clean loop has a genuinely
                    # strong half-time alias in the ACF, so almost everything scored
                    # near zero. What a caller needs to know is how sure the system is
                    # GIVEN everything it knows, and if the prior legitimately rules a
                    # rival out then the answer really is more certain.
                    # Score a WIDER lag range than the tempo search itself, purely so
                    # the rivals can be looked up. The search is bounded to 60-190 BPM,
                    # but a rival at half or double the winner routinely falls outside
                    # that: at 100 BPM the winner sits at lag 38 and its half-time
                    # rival at 76, past the window's end of 62. Those rivals were being
                    # read as zero, making the margin a perfect 1.0 and the confidence
                    # highest exactly where the ambiguity was worst — for every tempo
                    # below about 120 BPM, which is most of a sample library.
                    wide_lo = max(2, lo // 3)
                    wide_hi = min(len(acf) - 2, hi * 3)
                    wide_lags = np.arange(wide_lo, wide_hi + 1)
                    wide_bpms = 60.0 * frames_per_sec / wide_lags
                    wide_harmonic = np.zeros(len(wide_lags))
                    for k, weight in ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25)):
                        idx = wide_lags * k
                        valid = idx < len(acf)
                        wide_harmonic[valid] += weight * acf[idx[valid]]
                    posterior = np.zeros(len(acf))
                    posterior[wide_lags] = wide_harmonic * _density_likelihood(
                        wide_bpms, len(onsets), duration)
                    confidence = prominence * _metrical_margin(posterior,
                                                               int(lags[best]))
                    if bars:
                        # Two independent lines of evidence agreeing — the envelope's
                        # own periodicity and the file's length — is worth more than
                        # either alone.
                        confidence = min(1.0, confidence + 0.1)
                    return {"bpm": round(bpm, 1),
                            "bpm_confidence": round(confidence, 2),
                            "bars": bars}

    if len(onsets) < 4:
        return {}
    iois = np.diff(onsets)
    iois = iois[iois > 0.04]
    if iois.size < 3:
        return {}
    median = float(np.median(iois))
    if median <= 0:
        return {}
    spread = float(np.median(np.abs(iois - median)) / median)
    bpm = 60.0 / median
    while bpm < _BPM_MIN:
        bpm *= 2
    while bpm > _BPM_MAX:
        bpm /= 2
    return {"bpm": round(bpm, 1),
            "bpm_confidence": round(max(0.0, 1.0 - spread * 2.0), 2)}


def loudness_lufs(audio: np.ndarray, sr: int) -> float | None:
    """Integrated loudness, BS.1770-4 gating.

    DEVIATION FROM THE BRIDGE'S PORT, stated because it matters: K-weighting is
    applied as a zero-phase filter in the frequency domain rather than as the
    specified cascade of two biquads. The bridge's biquad runs a Python loop per
    sample, which is fine for one clip on demand and would cost more than everything
    else here combined across 22,100 files. Block mean-square energy is unaffected by
    the phase difference, which is all the gating uses.
    """
    if audio.ndim < 2:
        audio = audio[:, None]
    block, hop = int(0.4 * sr), int(0.1 * sr)
    if hop < 1 or len(audio) < block:
        return None                                   # shorter than one 400 ms block
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    gain = _k_weight_response(freqs, sr)
    z = None
    for ch in range(audio.shape[1]):
        filtered = np.fft.irfft(np.fft.rfft(audio[:, ch]) * gain, n=len(audio))
        squared = np.concatenate(([0.0], np.cumsum(filtered ** 2)))
        starts = np.arange(0, len(audio) - block + 1, hop)
        mean_sq = (squared[starts + block] - squared[starts]) / block
        z = mean_sq if z is None else z + mean_sq
    if z is None or not len(z):
        return None
    lk = -0.691 + 10 * np.log10(z + 1e-12)
    gated = z[lk > -70.0]                             # absolute gate
    if not len(gated):
        return None
    relative = -0.691 + 10 * np.log10(gated.mean() + 1e-12) - 10.0
    final = gated[(-0.691 + 10 * np.log10(gated + 1e-12)) > relative]
    if not len(final):
        return None
    return round(float(-0.691 + 10 * np.log10(final.mean() + 1e-12)), 1)


def _biquad_response(freqs: np.ndarray, sr: int, b, a) -> np.ndarray:
    """|H| of a digital biquad on a frequency grid, from the same coefficients the
    time-domain filter would run."""
    z = np.exp(-2j * np.pi * freqs / sr)
    return np.abs((b[0] + b[1] * z + b[2] * z ** 2)
                  / (1.0 + a[1] * z + a[2] * z ** 2))


def _k_weight_response(freqs: np.ndarray, sr: int) -> np.ndarray:
    """Magnitude response of the BS.1770-4 K-weighting pair.

    The DIGITAL coefficients are built exactly as the bridge's time-domain port builds
    them, then evaluated on the grid. An earlier version sampled hand-derived ANALOGUE
    transfer functions instead, which looked right and was wrong by 3-4 LU — and wrong
    by a different amount at every frequency, so no single check would have caught it.
    Deriving the response from the filter that the standard actually specifies removes
    the opportunity.
    """
    # High-shelf pre-filter: +4 dB above ~1.68 kHz.
    f0, gain_db, q = 1681.9744509555319, 3.99984385397, 0.7071752369554193
    k = np.tan(np.pi * f0 / sr)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = _biquad_response(
        freqs, sr,
        ((vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0,
         (vh - vb * k / q + k * k) / a0),
        (1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0))
    # RLB high-pass at ~38 Hz.
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    highpass = _biquad_response(
        freqs, sr, (1 / a0, -2 / a0, 1 / a0),
        (1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0))
    return shelf * highpass


def analyze(audio: np.ndarray, sr: int, mono: np.ndarray,
            duration: float | None) -> dict:
    """Everything measurable about one file, routed by what the file IS.

    ``audio`` is the decoded signal at its own rate, still stereo — width must be
    measured before the downmix. ``mono`` is the 16 kHz signal the mel pass already
    produced, reused rather than recomputed.
    """
    onsets = onset_times(mono)
    kind = classify(onsets, duration)
    attack_ms, decay_ms, peak_idx = envelope(mono, onsets, kind)
    width, correlation = stereo(audio, sr)

    props: dict = {"kind": kind, "onsets": int(len(onsets)),
                   "attack_ms": attack_ms,
                   "decay_ms": decay_ms or None,
                   "stereo_width": width, "stereo_correlation": correlation,
                   "loudness_lufs": loudness_lufs(audio, sr)}

    if kind == KIND_ONE_SHOT:
        # A single fundamental. Meaningless for a chord bed, which is why it is gated.
        # Gated a second time on NOISINESS: a snare has no fundamental, and YIN will
        # nevertheless return one — a real snare in this library came back at 25 Hz.
        # Gemini proposed using the tonal/atonal head as the gatekeeper; flatness is
        # the same judgement made here, from the spectrum already to hand, without
        # making measurement wait on a model.
        if spectral_flatness(mono) <= _PITCH_MAX_FLATNESS:
            props.update(pitch(mono, peak_idx))
    else:
        # A loop gets the question asked the other way round: not "what note", but
        # "what key" — which is what a producer needs before dropping it into a set.
        props.update(tempo(onsets, onset_strength(mono), duration=duration))
        props.update(key_from_chroma(chroma_of(mono)))
    return props
