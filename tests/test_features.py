"""Tests for listener/features.py — the MEASURED half.

Synthetic signals, because the whole point of measurement is that the right answer is
known in advance. Two bugs found during development are pinned here explicitly, since
both produced plausible output and neither raised:

  * YIN's parabolic interpolation had its sign inverted, so a 440 Hz tone read 448.9.
  * K-weighting was sampled from hand-derived ANALOGUE transfer functions and was
    wrong by 3-4 LU, differently at every frequency.

No pytest, matching the bridge: run the file.
"""
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from listener import features as F  # noqa: E402

SR = F.ANALYSIS_SR


def _tone(hz, seconds=0.5, decay=3.0, sr=SR, harmonic=0.0):
    t = np.arange(int(seconds * sr)) / sr
    x = np.sin(2 * np.pi * hz * t) + harmonic * np.sin(2 * np.pi * 2 * hz * t)
    return (x * np.exp(-t * decay)).astype(np.float32)


def _hit(seed=0, seconds=0.15, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    rs = np.random.RandomState(seed)
    return (rs.randn(len(t)) * np.exp(-t * 40)
            + np.sin(2 * np.pi * 60 * t) * np.exp(-t * 12)).astype(np.float32)


def _cents(got, want):
    return abs(1200 * np.log2(got / want))


# --- pitch -----------------------------------------------------------------------

def test_pitch_is_accurate_across_the_range():
    """Within a few cents, not merely the right MIDI note. Rounding to a note hides
    exactly the interpolation bug this exists to catch: 448.9 for 440 still rounds
    to A4."""
    for hz in (41.2, 55.0, 110.0, 220.0, 440.0, 880.0):
        got = F.pitch(_tone(hz), 0).get("pitch_hz")
        assert got is not None, f"no pitch for {hz} Hz"
        assert _cents(got, hz) < 10, f"{hz} Hz -> {got} ({_cents(got, hz):.1f} cents)"


def test_pitch_finds_the_fundamental_of_an_808_not_its_octave():
    """An 808's second harmonic is often LOUDER than its fundamental, and its attack
    is broadband noise. Both guards — skip past the transient, low-pass the bass —
    exist for this case, and without them YIN answers the octave up."""
    t = np.arange(int(0.8 * SR)) / SR
    f0 = 41.2
    x = (0.4 * np.sin(2 * np.pi * f0 * t) + 0.9 * np.sin(2 * np.pi * 2 * f0 * t))
    x = (x * np.exp(-t * 4)).astype(np.float32)
    x[:int(0.005 * SR)] += np.random.RandomState(0).randn(int(0.005 * SR)) * 0.8
    got = F.pitch(x, 0).get("pitch_hz")
    assert got is not None and _cents(got, f0) < 20, f"expected ~{f0}, got {got}"


def test_pitch_is_refused_when_the_spectrum_has_nothing_there():
    """YIN measures PERIODICITY, so on complex material it locks to a subharmonic —
    a period that divides every real partial while carrying no energy of its own.
    Real files: a stab whose strongest partial is E2 (165 Hz) was reported at 27.4 Hz,
    and a chord of C#3/G#3/F#2 at 30.7 Hz. Both scored 0.72-0.84 confidence, so the
    confidence floor cannot catch it — only the spectrum can.

    Built here as a signal with partials at 3x, 4x and 5x a fundamental that is not
    present, which is the shape that provokes it.
    """
    sr = SR
    t = np.arange(int(0.6 * sr)) / sr
    base = 55.0
    x = sum(np.sin(2 * np.pi * base * k * t) for k in (3, 4, 5))
    x = (x * np.exp(-t * 2)).astype(np.float32)
    got = F.pitch(x, 0)
    if got.get("pitch_hz") is not None:
        # Whatever is reported must actually EXIST in the sound.
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
        near = np.abs(freqs - got["pitch_hz"]) < 6
        assert near.any() and spec[near].max() >= 0.05 * spec.max(), (
            f"reported {got['pitch_hz']} Hz, which carries no energy")


def test_noise_gets_no_pitch():
    """A hi-hat has no fundamental. An estimator asked anyway still returns a number,
    so the flatness gate has to refuse the question."""
    noise = (np.random.RandomState(3).randn(int(0.3 * SR)) * 0.3).astype(np.float32)
    assert F.spectral_flatness(noise) > F._PITCH_MAX_FLATNESS
    props = F.analyze(noise[:, None], SR, noise, 0.3)
    assert "pitch_hz" not in props, props


# --- the router ------------------------------------------------------------------

def test_single_hit_is_a_one_shot_and_a_bar_of_hits_is_a_loop():
    one = np.zeros(int(1.0 * SR), dtype=np.float32)
    hit = _hit()
    one[:len(hit)] += hit
    assert F.classify(F.onset_times(one), 1.0) == F.KIND_ONE_SHOT

    loop = np.zeros(int(2.0 * SR), dtype=np.float32)
    for i in range(8):
        start = int(i * 0.25 * SR)
        h = _hit(seed=i)
        loop[start:start + len(h)] += h
    assert F.classify(F.onset_times(loop), 2.0) == F.KIND_LOOP


def test_a_rippling_decay_is_not_a_rhythm():
    """The case that forced the span rule: a real 808 produced four onsets, all inside
    the first 340 ms of a 2.3 s file, and was routed as a loop — earning it a tempo
    and a key, both meaningless. Count alone cannot tell those apart; span can."""
    clustered = np.array([0.098, 0.202, 0.261, 0.339])
    assert F.classify(clustered, 2.315) == F.KIND_ONE_SHOT
    spread = np.array([0.1, 0.6, 1.1, 1.6, 2.1])
    assert F.classify(spread, 2.315) == F.KIND_LOOP


def test_events_per_beat_target_slopes_with_tempo():
    """Fast music leans on half-time phrasing — the snare on 3, not on 2 and 4 — so it
    carries FEWER structural events per beat than slow music, not more. Measured in the
    library: files named 140+ BPM average 1.6 events per beat against 2.3 for those
    named under 100. A flat target compromises between them and breaks fewer ties."""
    assert F._epb_target(85.0) > F._epb_target(170.0)
    assert 2.0 < F._epb_target(85.0) < 2.5
    assert 1.3 < F._epb_target(170.0) < 1.8


def test_density_prefers_the_tempo_implying_a_sane_subdivision():
    """The term that actually separates a tempo from its half, since duration cannot:
    4 bars at 100 BPM is also exactly 8 bars at 200."""
    # 8 seconds, 32 structural onsets -> 2 per beat at 120, 4 per beat at 60.
    likely = F._density_likelihood(np.array([120.0, 60.0]), 32, 8.0)
    assert likely[0] > likely[1], likely


def test_confidence_falls_when_a_metrical_rival_is_as_strong():
    """The old confidence measured how RHYTHMIC a file is and was anti-correlated with
    being right: a perfectly quantised loop has a perfect half-time alias, so "very
    rhythmic" scored as "very certain" exactly where the tempo was most ambiguous."""
    acf = np.zeros(200)
    acf[50] = 1.0
    acf[100] = 0.98                      # half-time rival, nearly as strong
    assert F._metrical_margin(acf, 50) < 0.1
    acf[100] = 0.10                      # rival now weak
    assert F._metrical_margin(acf, 50) > 0.8


def test_tempo_survives_frame_quantisation():
    """The flux hop is 16 ms and tempo comes from the interval BETWEEN onsets, so
    rounding to the hop put a 120 BPM loop at 117.2 — squarely in the digits a
    producer reads as the answer."""
    loop = np.zeros(int(4.0 * SR), dtype=np.float32)
    for i in range(16):
        start = int(i * 0.25 * SR)
        h = _hit(seed=i)
        loop[start:start + len(h)] += h
    got = F.tempo(F.onset_times(loop))
    assert got, "no tempo"
    assert abs(got["bpm"] - 120.0) < 2.0, got


# --- stereo ----------------------------------------------------------------------

def test_width_is_measured_above_the_bass():
    """Mono bass under a wide top end is the standard production shape. Measured
    broadband it averages to 'narrow', which is the opposite of the truth and the
    reason the band limit exists."""
    n = SR
    rs = np.random.RandomState(5)
    t = np.arange(n) / SR
    bass = (np.sin(2 * np.pi * 60 * t) * 0.5).astype(np.float32)
    highs = (rs.randn(n) * 0.05).astype(np.float32)
    stereo = np.stack([bass + highs, bass - highs], axis=1)
    width, correlation = F.stereo(stereo, SR)
    assert width > 0.9, width
    assert correlation > 0.9, correlation      # broadband would call this mono-ish


def test_identical_channels_are_not_wide_and_inverted_ones_are_flagged():
    n = SR
    m = (np.random.RandomState(6).randn(n) * 0.1).astype(np.float32)
    width, correlation = F.stereo(np.stack([m, m], axis=1), SR)
    assert width < 1e-6 and correlation > 0.99

    width, correlation = F.stereo(np.stack([m, -m], axis=1), SR)
    assert correlation < -0.99, "an out-of-phase pair must still be nameable"


def test_mono_files_have_no_width_at_all():
    m = (np.random.RandomState(7).randn(SR) * 0.1).astype(np.float32)
    assert F.stereo(m[:, None], SR) == (None, None)


# --- loudness --------------------------------------------------------------------

def test_lufs_matches_the_standards_own_calibration():
    """BS.1770's anchor: a 1 kHz sine at -20 dBFS in both channels is -20 LUFS. The
    K-weighting is ~0 dB at 1 kHz, so this pins the gain, the gating and the channel
    sum in one number."""
    t = np.arange(48000 * 3) / 48000.0
    sine = 0.1 * np.sin(2 * np.pi * 1000 * t)          # -20 dBFS RMS per channel
    got = F.loudness_lufs(np.stack([sine, sine], axis=1), 48000)
    assert abs(got - (-20.0)) < 0.3, got


def test_lufs_weights_low_frequencies_down():
    """K-weighting is not flat: 60 Hz must read quieter than 1 kHz at equal level.
    A wrong response still produces a plausible number, so the SHAPE is what to
    assert."""
    t = np.arange(48000 * 3) / 48000.0
    low = 0.1 * np.sin(2 * np.pi * 60 * t)
    mid = 0.1 * np.sin(2 * np.pi * 1000 * t)
    quiet = F.loudness_lufs(np.stack([low, low], axis=1), 48000)
    loud = F.loudness_lufs(np.stack([mid, mid], axis=1), 48000)
    assert quiet < loud - 3.0, (quiet, loud)


def test_too_short_for_a_block_is_none_not_a_guess():
    short = np.zeros((1000, 2), dtype=np.float32)
    assert F.loudness_lufs(short, 48000) is None


# --- key -------------------------------------------------------------------------

def test_key_separates_relative_major_from_relative_minor():
    """C major and A minor share every note; the profiles have to use EMPHASIS. If
    this passes by luck it will fail on the other one."""
    def progression(chords):
        out = np.zeros(int(len(chords) * 0.6 * SR), dtype=np.float32)
        for i, chord in enumerate(chords):
            start = int(i * 0.6 * SR)
            for midi in chord:
                x = _tone(440 * 2 ** ((midi - 69) / 12), 0.6, decay=1.5, harmonic=0.5)
                out[start:start + len(x)] += 0.3 * x[:len(out) - start]
        return out

    major = F.key_from_chroma(F.chroma_of(progression(
        [[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]])))
    assert (major["key"], major["scale"]) == ("C", "major"), major
    minor = F.key_from_chroma(F.chroma_of(progression(
        [[57, 60, 64], [62, 65, 69], [64, 68, 71], [57, 60, 64]])))
    assert (minor["key"], minor["scale"]) == ("A", "minor"), minor


# --- routing decides what gets measured ------------------------------------------

def test_a_loop_gets_a_tempo_and_no_fundamental():
    loop = np.zeros(int(4.0 * SR), dtype=np.float32)
    for i in range(16):
        start = int(i * 0.25 * SR)
        h = _hit(seed=i)
        loop[start:start + len(h)] += h
    props = F.analyze(loop[:, None], SR, loop, 4.0)
    assert props["kind"] == F.KIND_LOOP
    assert props.get("bpm") is not None
    assert "pitch_hz" not in props, "a single fundamental is meaningless for a bar"
    assert props["decay_ms"] is None, "what a loop decays to is its next hit"


def test_a_one_shot_gets_a_fundamental_and_no_tempo():
    props = F.analyze(_tone(110.0)[:, None], SR, _tone(110.0), 0.5)
    assert props["kind"] == F.KIND_ONE_SHOT
    assert props.get("pitch_hz") is not None
    assert props.get("bpm") is None, "a BPM from one transient is not a measurement"
    assert props["decay_ms"] > 0


def test_attack_is_timed_from_the_first_hit_not_the_loudest():
    """A loop's global peak is wherever its loudest hit falls — which produced
    'attack' times of 1.0-1.5 s on real drum loops, describing nothing."""
    loop = np.zeros(int(3.0 * SR), dtype=np.float32)
    for i in range(12):
        start = int(i * 0.25 * SR)
        h = _hit(seed=i) * (3.0 if i == 7 else 1.0)     # loudest hit late
        loop[start:start + len(h)] += h
    props = F.analyze(loop[:, None], SR, loop, 3.0)
    assert props["attack_ms"] < 100, props["attack_ms"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)
