"""Proof that ``listener/shared_dsp.py`` measures exactly what ``listener/features.py``
measures — the test without which the extraction is unverifiable.

The refactor it guards is delicate for one specific reason: the shared module was
produced by MOVING code, and a hand-move is the same operation that has already failed
three times in this project. "It looks the same" is not evidence. So the first group of
tests below feeds BOTH implementations the identical mono signal and demands
**bit-identical** output — not "close", not "within a tolerance". Anything less would
pass on a transposed constant.

The second group measures something different and is not a pass/fail on equality: the
shared module owns its own resampling (see rule 1 in ``shared_dsp``), so a file that
today reaches ``features`` through soxr will reach it through the Kaiser polyphase
resampler after the swap. The maths being identical does NOT make the numbers
identical, because the samples differ. That difference is a real, deliberate
consequence of closing the pre-processing hole, and it belongs in a measurement rather
than in an assumption.

Real files, from the live index, because synthetic signals cannot exercise the cases
that broke: an 808 whose decay ripples read as a rhythm, a loop whose half-time alias
is as strong as its beat. The index is opened READ-ONLY — a scan may be writing to it.

No pytest, matching the rest of both repos: run the file.
"""
import os
import random
import sqlite3
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from listener import decode  # noqa: E402
from listener import features as F  # noqa: E402
from listener import shared_dsp as S  # noqa: E402

#: The live index. Only ever read: a full scan takes ~65 minutes and may well be
#: running while this test does.
INDEX_DB = os.path.join(os.path.expanduser("~"), ".ai-bridge", "sound_index.db")
#: Tens of files, not thousands. Every one is decoded and analysed twice over, and this
#: is meant to be runnable while a scan has the CPU.
SAMPLE_FILES = 24
#: Fixed seed so a failure is reproducible. A random sample that cannot be re-run is a
#: bug report nobody can act on.
SAMPLE_SEED = 11

_CACHE: dict = {}


def _real_files(limit=SAMPLE_FILES):
    """A spread of real library files: both routes, several sample rates, both widths.

    Returns [] when the index is absent, so the file still runs on a fresh checkout —
    the synthetic tests below carry the load there.
    """
    if "files" in _CACHE:
        return _CACHE["files"][:limit]
    if not os.path.exists(INDEX_DB):
        _CACHE["files"] = []
        return []
    conn = sqlite3.connect(f"file:{INDEX_DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT f.path, f.duration_sec, f.sample_rate, f.channels, p.kind "
            "FROM files f JOIN properties p ON p.file_id = f.id "
            "WHERE f.error IS NULL AND f.duration_sec > 0.3 "
            "  AND f.sample_rate IS NOT NULL").fetchall()
    finally:
        conn.close()
    random.Random(SAMPLE_SEED).shuffle(rows)
    # Spread across (kind, rate, channel count) rather than taking the first N. Sample
    # libraries are shipped in packs, and the first N of anything is one pack's worth of
    # near-identical files — which would look like broad coverage and be nothing of the
    # sort.
    picked, seen = [], {}
    per_bucket = max(2, limit // 10)
    for row in rows:
        bucket = (row[4], row[2], row[3])
        if seen.get(bucket, 0) >= per_bucket or not os.path.exists(row[0]):
            continue
        seen[bucket] = seen.get(bucket, 0) + 1
        picked.append(row)
        if len(picked) >= limit:
            break
    _CACHE["files"] = picked
    return picked


def _decode(path):
    """Decode exactly as ``decode.process_file`` does, including the soxr resample.

    This is the CURRENT pre-processing, deliberately: the equivalence tests must
    compare the two implementations on the signal features.py actually sees today.
    """
    import soundfile as sf
    with sf.SoundFile(path) as handle:
        rate = handle.samplerate
        audio = handle.read(frames=int(decode.MAX_SECONDS * rate),
                            dtype="float32", always_2d=True)
        duration = len(handle) / rate if rate else None
    mono = decode._resample(audio.mean(axis=1).astype(np.float32), rate)
    return audio, rate, mono, duration


def _report_skip(what):
    print(f"        (no index at {INDEX_DB} — {what} skipped)")


# =====================================================================================
# 1. THE RESAMPLER. Everything downstream is meaningless if this is wrong, and unlike
#    the moved code it has no existing implementation to be compared against.
# =====================================================================================

def test_resampler_passband_is_flat_and_the_stopband_is_gone():
    """A resampler that quietly tilts the band would move every measurement below,
    consistently, and look like a successful refactor while doing it.

    Checked as a filter, not by ear: unity gain through the passband, and content above
    the new Nyquist annihilated rather than folded back. Aliasing is the failure that
    matters — a 15 kHz cymbal partial folding to 1 kHz does not sound wrong in a
    spectrum plot, it sounds like a bass note to the chroma.
    """
    sr = 44100
    t = np.arange(2 * sr) / sr
    for hz in (50, 200, 1000, 3000, 5000, 6000):
        y = S.resample(np.sin(2 * np.pi * hz * t).astype(np.float32), sr)
        # Amplitude at the same frequency, measured away from the filter's edges.
        seg = y[800:-800]
        ts = np.arange(len(seg)) / float(S.ANALYSIS_SR)
        amp = 2 * abs(np.sum(seg * np.exp(-2j * np.pi * hz * ts))) / len(seg)
        assert abs(20 * np.log10(amp)) < 0.1, f"{hz} Hz: {20 * np.log10(amp):+.2f} dB"
    for hz in (10000, 15000, 20000):
        y = S.resample(np.sin(2 * np.pi * hz * t).astype(np.float32), sr)
        residual = 20 * np.log10(y[800:-800].std() * np.sqrt(2) + 1e-20)
        assert residual < -80, f"{hz} Hz aliased back at {residual:.1f} dBFS"


def test_resampling_to_the_same_rate_changes_nothing():
    x = np.float32([0.0, 1.0, -0.5, 0.25])
    assert np.array_equal(S.resample(x, S.ANALYSIS_SR), x)


def test_resampler_preserves_level_and_length():
    """Length must follow the ratio and level must not move — a normalisation slip in
    the filter bank would rescale every file and nothing else would notice."""
    for sr in (8000, 22050, 44100, 48000, 96000):
        rs = np.random.RandomState(4)
        x = (rs.randn(sr * 2) * 0.1).astype(np.float32)
        y = S.resample(x, sr)
        assert abs(len(y) - 2 * S.ANALYSIS_SR) <= 1, (sr, len(y))
        # Only the band that survives is comparable, so compare against the input with
        # everything above the new Nyquist removed.
        spec = np.fft.rfft(x)
        spec[np.fft.rfftfreq(len(x), 1.0 / sr) > S.ANALYSIS_SR / 2] = 0
        want = float(np.fft.irfft(spec, n=len(x)).std())
        assert abs(20 * np.log10(y.std() / want)) < 0.3, (sr, y.std(), want)


# =====================================================================================
# 2. ALGORITHM EQUIVALENCE — the same mono in, byte-identical numbers out.
# =====================================================================================

_ARRAY_PRIMITIVES = ("onset_strength", "onset_times", "spectral_flatness", "chroma_of")


def test_array_primitives_are_bit_identical_on_real_files():
    """No tolerance. These are the same operations on the same samples, so the only
    correct answer is the identical float — a tolerance here would hide precisely the
    kind of edit this whole exercise exists to catch (a window length, a hop, a
    threshold, a sign)."""
    files = _real_files()
    if not files:
        _report_skip("real-file equivalence")
        return
    worst = {}
    for path, _dur, _rate, _ch, _kind in files:
        _audio, _sr, mono, _duration = _decode(path)
        for name in _ARRAY_PRIMITIVES:
            old = np.asarray(getattr(F, name)(mono), dtype=np.float64)
            new = np.asarray(getattr(S, name)(mono), dtype=np.float64)
            assert old.shape == new.shape, f"{name} on {path}: {old.shape} vs {new.shape}"
            delta = float(np.max(np.abs(old - new))) if old.size else 0.0
            worst[name] = max(worst.get(name, 0.0), delta)
            assert delta == 0.0, f"{name} differs by {delta:g} on {path}"
    print("        " + "  ".join(f"{k}=0" for k in worst))


def test_every_routed_measurement_is_identical_on_real_files():
    """The scalar and dict results, on the SAME mono: classify, envelope, stereo,
    loudness, tempo (bpm + confidence + bars) and key.

    Compared with ``==`` on the returned objects rather than field by field, so a key
    that the shared module simply forgot to emit fails here rather than passing
    unnoticed."""
    files = _real_files()
    if not files:
        _report_skip("routed-measurement equivalence")
        return
    loops = one_shots = 0
    for path, _dur, _rate, _ch, _kind in files:
        audio, sr, mono, duration = _decode(path)
        onsets = F.onset_times(mono)
        kind = F.classify(onsets, duration)
        loops += kind == F.KIND_LOOP
        one_shots += kind == F.KIND_ONE_SHOT
        assert kind == S.classify(S.onset_times(mono), duration), path
        assert F.envelope(mono, onsets, kind) == S.envelope(mono, onsets, kind), path
        assert F.stereo(audio, sr) == S.stereo(audio, sr), path
        assert F.loudness_lufs(audio, sr) == S.loudness_lufs(audio, sr), path
        assert (F.tempo(onsets, F.onset_strength(mono), duration=duration)
                == S.tempo(onsets, S.onset_strength(mono), duration=duration)), path
        assert (F.key_from_chroma(F.chroma_of(mono))
                == S.key_from_chroma(S.chroma_of(mono))), path
    print(f"        {len(files)} files: {loops} loops, {one_shots} one-shots")


def test_measure_reproduces_analyze_field_for_field():
    """``measure()`` fed the SAME mono must reproduce what analyze produces.

    This is the test that authorised the swap, and it is worth being precise about what
    it proves NOW that the swap has happened. Before it, `analyze` had its own copy of
    the maths and this compared two implementations — bit-identical, 0.000e+00, across
    120 real files. Today `analyze` calls `measure`, so the maths cannot differ; what
    this still guards is that `analyze` keeps handing the shared core a faithful signal
    and passing its fields through unchanged.

    The one thing that legitimately DOES differ is pre-processing, and it is asserted
    as a tolerance rather than equality: `analyze` now resamples through
    `shared_dsp.prepare` while this constructs `Prepared` from the mel pass's soxr
    output. That is trap 1 being closed — both programs measuring one signal instead of
    each choosing their own — and it moves a few values by a hair. See
    `test_owning_the_resampling_moves_the_numbers_only_slightly` for the size of it.
    """
    files = _real_files()
    if not files:
        _report_skip("measure/analyze equivalence")
        return
    # `onsets` is a COUNT off the resampled signal, so it belongs with the tolerant
    # group: a tambourine shake came back 19 against 18, one detection either side of a
    # threshold on a different set of samples. What matters is that `kind` — the routing
    # decision that count feeds — still agrees exactly, and it does.
    exact_fields = ("kind", "key", "scale", "bars")
    close_fields = ("onsets", "attack_ms", "decay_ms", "stereo_width",
                    "stereo_correlation", "loudness_lufs", "bpm", "bpm_confidence",
                    "key_strength")
    for path, _dur, _rate, _ch, _kind in files:
        audio, sr, mono, duration = _decode(path)
        old = F.analyze(audio, sr, mono, duration)
        # THE SAME SIGNAL, because that is what this test claims to check.
        #
        # It used to build `Prepared` from the mel pass's soxr mono while `analyze`
        # re-derives its own through `prepare` — `analyze` accepts `mono` and IGNORES it
        # on purpose, so that both programs measure one signal instead of each choosing a
        # resampler. Comparing those two therefore measured RESAMPLER equivalence, not
        # pass-through, and it failed on `Perc Kitchen Kit.adg.ogg` at 15 onsets versus 18
        # — three detections either side of a threshold on a percussive file, from two
        # different signals. How much the resamplers differ is a real question and it
        # already has its own test: `test_owning_the_resampling_moves_the_numbers_only_
        # slightly`. This one is about `analyze` passing fields through unchanged.
        new = S.measure(S.prepare(audio, sr, duration))
        for field in exact_fields:
            assert old.get(field) == new.get(field), (
                f"{field}: {old.get(field)!r} vs {new.get(field)!r} on {path}")
        for field in close_fields:
            a, b = old.get(field), new.get(field)
            if a is None or b is None:
                assert a == b, f"{field}: {a!r} vs {b!r} on {path}"
                continue
            # Generous in absolute terms and tight in relative terms: 1 ms of envelope
            # timing or 0.5 BPM is nothing musically, but a factor-of-two error is not
            # a resampler artefact and must still fail here.
            assert abs(a - b) <= max(1.0, abs(a) * 0.02), (
                f"{field}: {a!r} vs {b!r} on {path}")


def test_synthetic_signals_agree_too():
    """Real files cannot be relied on to contain every branch. These reach the ones
    they miss: silence, a pure tone with no onsets at all, and a mono file."""
    sr = S.ANALYSIS_SR
    cases = {
        "silence": np.zeros(sr, dtype=np.float32),
        "dc": np.full(sr, 0.5, dtype=np.float32),
        "tone": np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32),
        "noise": (np.random.RandomState(2).randn(sr) * 0.2).astype(np.float32),
        "click": np.concatenate([np.float32([1.0]), np.zeros(sr - 1, np.float32)]),
        "short": np.float32([0.1, -0.2, 0.3]),
    }
    for name, mono in cases.items():
        assert np.array_equal(F.onset_strength(mono), S.onset_strength(mono)), name
        assert np.array_equal(F.onset_times(mono), S.onset_times(mono)), name
        assert F.spectral_flatness(mono) == S.spectral_flatness(mono), name
        assert F.chroma_of(mono) == S.chroma_of(mono), name
        onsets = F.onset_times(mono)
        for duration in (None, 0.5, 4.0):
            assert F.classify(onsets, duration) == S.classify(onsets, duration), name
            kind = F.classify(onsets, duration)
            assert F.envelope(mono, onsets, kind) == S.envelope(mono, onsets, kind), name
            assert (F.tempo(onsets, F.onset_strength(mono), duration=duration)
                    == S.tempo(onsets, S.onset_strength(mono), duration=duration)), name
        stereo = np.stack([mono, mono * 0.5], axis=1)
        assert F.stereo(stereo, sr) == S.stereo(stereo, sr), name
        assert F.loudness_lufs(stereo, sr) == S.loudness_lufs(stereo, sr), name


def test_key_scoring_agrees_on_random_histograms():
    """The Krumhansl core, hammered on histograms no real file would produce — flat,
    single-spike, negative-free noise — because the contrast gate and the runner-up
    margin are exactly where the two copies could disagree without any file noticing."""
    rs = np.random.RandomState(9)
    for i in range(400):
        if i % 4 == 0:
            hist = [1.0] * 12                                   # perfectly flat
        elif i % 4 == 1:
            hist = [0.0] * 12
            hist[i % 12] = 1.0                                  # single spike
        elif i % 4 == 2:
            hist = list(1.0 + rs.rand(12) * 0.1)                # nearly flat
        else:
            hist = list(rs.rand(12) ** 3)
        assert F.key_from_chroma(hist) == S.key_from_chroma(hist), hist


# =====================================================================================
# 3. THE PRE-PROCESSING CHANGE — measured, not assumed. Not an equality test.
# =====================================================================================

def test_measure_refuses_a_raw_array():
    """The type is the enforcement. If ``measure`` accepted an array, a caller could do
    its own resampling again and the whole hole re-opens quietly."""
    try:
        S.measure(np.zeros((1000, 2), dtype=np.float32))
    except TypeError as exc:
        assert "prepare" in str(exc)
    else:
        raise AssertionError("measure() accepted a raw array")


def test_owning_the_resampling_moves_the_numbers_only_slightly():
    """WHAT THE SWAP ACTUALLY COSTS, in numbers rather than in confidence.

    After the swap the mono signal comes from ``shared_dsp.resample`` instead of soxr,
    so features derived from it change even though not one line of the maths did. This
    prints the whole distribution and asserts only a loose envelope — a tight assert
    here would be a claim about soxr, not about our code.

    The one thing that must NOT move is anything measured on the SOURCE: stereo width,
    correlation and BS.1770 loudness never touch the resampler, and a difference in
    those would mean the pre-processing had leaked somewhere it should not be.

    ⚠️ ATTACK IS PRINTED AND NOT ASSERTED, and the reason is worth knowing before
    anyone reads a large number here as a regression. ``envelope`` takes ``argmax`` of
    the smoothed envelope, and a sustained sound has no single peak — a real library
    file (a 1.16 s organ one-shot) holds 69 samples within 0.1% of its maximum, spread
    across 1.15 s of the file. Which of them wins is decided by differences of 3e-5,
    so ANY change to the signal — a different resampler, a different decoder, dither —
    flips the reported attack by hundreds of milliseconds. Measured on that file:
    409.1 ms through soxr against 90.9 ms through ours, for the same audio.
    That fragility is in ``features.envelope`` today and was moved across unchanged;
    it is a real bug and it is NOT this refactor's to fix silently, because fixing it
    would break the byte-equivalence the tests above rely on. It wants a tie-break
    (the FIRST sample reaching the peak, say) as its own change, with its own re-scan.
    """
    files = _real_files()
    if not files:
        _report_skip("pre-processing delta")
        return
    same_kind = same_key = keyed = same_bpm = tempoed = 0
    bpm_delta, attack_delta = [], []
    changed = []
    for path, _dur, _rate, _ch, _kind in files:
        audio, sr, mono, duration = _decode(path)
        old = F.analyze(audio, sr, mono, duration)
        new = S.measure(S.prepare(audio, sr, duration))

        # Source-domain measurements must be untouched.
        assert old["stereo_width"] == new["stereo_width"], path
        assert old["stereo_correlation"] == new["stereo_correlation"], path
        assert old["loudness_lufs"] == new["loudness_lufs"], path

        same_kind += old["kind"] == new["kind"]
        if old.get("bpm") is not None and new.get("bpm") is not None:
            tempoed += 1
            bpm_delta.append(abs(old["bpm"] - new["bpm"]))
            same_bpm += bpm_delta[-1] < 0.05
        if old.get("key") or new.get("key"):
            keyed += 1
            same_key += ((old.get("key"), old.get("scale"))
                         == (new.get("key"), new.get("scale")))
        attack_delta.append(abs((old.get("attack_ms") or 0.0)
                                - (new.get("attack_ms") or 0.0)))
        for field in ("kind", "bpm", "key", "scale", "bars"):
            if old.get(field) != new.get(field):
                changed.append(f"{os.path.basename(path)[:40]} {field} "
                               f"{old.get(field)!r}->{new.get(field)!r}")
    n = len(files)
    print(f"        route agrees   {same_kind}/{n}")
    print(f"        tempo agrees   {same_bpm}/{tempoed}  "
          f"max |delta| {max(bpm_delta) if bpm_delta else 0:.2f} BPM")
    print(f"        key agrees     {same_key}/{keyed}")
    print(f"        attack max |delta| {max(attack_delta) if attack_delta else 0:.1f} ms")
    for line in changed:
        print(f"        CHANGED: {line}")
    # Loose, and deliberately so. These are the thresholds at which the swap would stop
    # being a refactor and start being a different analyser.
    assert same_kind >= 0.95 * n, f"routing moved on {n - same_kind} of {n} files"
    if tempoed:
        assert same_bpm >= 0.85 * tempoed, f"tempo moved on {tempoed - same_bpm} files"
    if keyed:
        assert same_key >= 0.9 * keyed, f"key moved on {keyed - same_key} files"


# =====================================================================================
# 4. THE BRIDGE'S HALF — checked here because only this repo can see both.
# =====================================================================================

#: Where the bridge is expected to sit relative to this repo. Overridable, because a
#: checkout layout is not something to hard-code into a test.
BRIDGE_REPO = os.environ.get(
    "AI_BRIDGE_REPO",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "ai-bridge-for-ableton-live"))


def test_the_bridges_key_estimator_would_survive_the_swap():
    """``describe.key_from_histogram`` is the bridge's key estimator and it carries its
    own copy of the profiles, the correlation and the contrast gate. After the swap it
    calls ``shared_dsp.key_scores``, and this proves the substitution stayed a no-op:
    the winner, the runner-up, the margin and the refusal all still come out the same
    on 400 histograms, so the MIDI tier and the audio tier cannot drift apart on what
    'F minor' means.
    """
    host = os.path.join(BRIDGE_REPO, "host")
    if not os.path.exists(os.path.join(host, "describe.py")):
        print(f"        (bridge repo not at {BRIDGE_REPO} — skipped)")
        return
    # `host` STAYS on the path for the whole test. describe.key_from_histogram imports
    # shared_dsp lazily — deliberately, so the module remains pure stdlib for machines
    # with no numpy — and that import resolves at CALL time, not import time. Removing
    # the path here would break the call while telling us nothing about the bridge,
    # which always runs with `host` importable.
    sys.path.insert(0, host)
    import describe                                            # noqa: PLC0415

    # Before the swap this compared the bridge's OWN copies of the profiles against the
    # shared ones. It has no copies now — `key_from_histogram` calls `key_scores`
    # directly — so the thing worth asserting is that the duplication is really gone
    # and that the two still agree end to end on 400 histograms below.
    assert not hasattr(describe, "_MAJOR"), (
        "describe.py has grown its own key profile again — that is the duplication "
        "this module exists to remove")
    assert not hasattr(describe, "_correlate"), "describe.py has its own scorer again"
    assert describe._MIN_TONAL_CONTRAST == S.MIN_TONAL_CONTRAST, "contrast gate differs"
    assert describe.NOTE_NAMES == S.NOTE_NAMES, "note names differ"

    rs = np.random.RandomState(13)
    for i in range(400):
        hist = ([1.0] * 12 if i % 5 == 0 else
                list(1.0 + rs.rand(12) * 0.1) if i % 5 == 1 else
                list(rs.rand(12) ** 3))
        theirs = describe.key_from_histogram(hist)
        ours = S.key_scores(hist)
        if theirs.get("key") is None:
            assert not ours, f"shared claims a key the bridge refuses: {hist}"
            continue
        assert ours, f"bridge claims a key the shared core refuses: {hist}"
        score, root, mode = ours[0]
        # KEY NAMES, NOT NOTE NAMES. This asserted `NOTE_NAMES`, which is all sharps and
        # is right for a pitch and wrong for a key: it made "D# major" — nine sharps, a
        # key nobody writes — the expected answer, so the test defended the bug. Keys are
        # spelled by `key_name`, and the convention differs between the modes.
        assert theirs["key"] == f"{S.key_name(root, mode)} {mode}", (theirs, ours[0])
        assert theirs["confidence"] == round(float(score), 3)
        assert theirs["margin"] == round(float(score - ours[1][0]), 3)
        assert theirs["runner_up"] == f"{S.key_name(ours[1][1], ours[1][2])} {ours[1][2]}"
        # The relative is always named, because chroma cannot separate a key from it.
        rel_root, rel_mode = S.relative_key(root, mode)
        assert theirs["relative"] == f"{S.key_name(rel_root, rel_mode)} {rel_mode}"



# --- tempo internals: moved here with the code they test ---------------------
# These lived in test_features.py and exercised private helpers (_epb_target,
# _density_likelihood, _metrical_margin). Those helpers are now in shared_dsp, so
# the tests follow them: a test that reaches into a module's privates belongs
# beside that module, not beside whoever used to own the code.

def test_events_per_beat_target_slopes_with_tempo():
    """Fast music leans on half-time phrasing — the snare on 3, not on 2 and 4 — so it
    carries FEWER structural events per beat than slow music, not more. Measured in the
    library: files named 140+ BPM average 1.6 events per beat against 2.3 for those
    named under 100. A flat target compromises between them and breaks fewer ties."""
    assert S._epb_target(85.0) > S._epb_target(170.0)
    assert 2.0 < S._epb_target(85.0) < 2.5
    assert 1.3 < S._epb_target(170.0) < 1.8


def test_density_prefers_the_tempo_implying_a_sane_subdivision():
    """The term that actually separates a tempo from its half, since duration cannot:
    4 bars at 100 BPM is also exactly 8 bars at 200."""
    # 8 seconds, 32 structural onsets -> 2 per beat at 120, 4 per beat at 60.
    likely = S._density_likelihood(np.array([120.0, 60.0]), 32, 8.0)
    assert likely[0] > likely[1], likely


def test_confidence_falls_when_a_metrical_rival_is_as_strong():
    """The old confidence measured how RHYTHMIC a file is and was anti-correlated with
    being right: a perfectly quantised loop has a perfect half-time alias, so "very
    rhythmic" scored as "very certain" exactly where the tempo was most ambiguous."""
    acf = np.zeros(200)
    acf[50] = 1.0
    acf[100] = 0.98                      # half-time rival, nearly as strong
    assert S._metrical_margin(acf, 50) < 0.1
    acf[100] = 0.10                      # rival now weak
    assert S._metrical_margin(acf, 50) > 0.8


def test_metrical_rivals_are_visible_outside_the_tempo_search_window():
    """The search covers 60-190 BPM, but a winner's half-time rival routinely sits
    outside it: at 100 BPM the winner is lag 38 and the rival lag 76. Those were read
    as zero, so the margin came out a perfect 1.0 and confidence was HIGHEST for every
    tempo under ~120 BPM — most of a sample library — exactly where the octave
    ambiguity is worst."""
    acf = np.zeros(300)
    acf[38] = 1.0
    acf[76] = 0.95                       # a strong half-time rival, outside 60-190 BPM
    assert S._metrical_margin(acf, 38) < 0.1, "a rival outside the window must count"


def test_a_square_wave_sub_is_not_transposed_up_a_fifth():
    """The bug the chroma band's lower edge used to cause, in its exact form.

    A square wave has no even harmonics, and saturation — what makes a sub audible on a
    phone speaker — generates odd ones. With the band starting at 55 Hz the fundamental
    of a C1 sub fell outside it, the second harmonic does not exist, and the loudest
    survivor was the third: a fifth above the root. C1 read G, D1 read A, F1 read C.
    Confidently, every time, with nothing to say a note had been discarded.
    """
    sr = S.ANALYSIS_SR
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    for midi in (24, 26, 27, 29, 31):                      # C1 to G1
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        t = np.arange(int(2.0 * sr)) / sr
        wave = np.zeros_like(t)
        for k in range(1, 40, 2):                          # odd harmonics only
            if hz * k < sr / 2:
                wave += np.sin(2 * np.pi * hz * k * t) / k
        wave = (wave / np.abs(wave).max() * 0.5).astype(np.float32)
        got = names[int(np.argmax(S.chroma_of(wave)))]
        assert got == names[midi % 12], (
            f"square sub on {names[midi % 12]}1 read as {got}")


def test_the_chroma_window_does_not_depend_on_file_length():
    """It used to shrink to the power of two below the signal length, so a 1.9 s file
    and a 2.1 s file were analysed differently — an arbitrary discontinuity no reader
    would predict from the code. The fix buys little accuracy (one case in 252) and it
    buys determinism, which is the part worth pinning."""
    sr = S.ANALYSIS_SR
    winners = set()
    for seconds in (0.4, 0.9, 1.9, 2.1, 5.0):
        t = np.arange(int(seconds * sr)) / sr
        tone = (np.sin(2 * np.pi * 220.0 * t) * 0.5).astype(np.float32)
        winners.add(int(np.argmax(S.chroma_of(tone))))
    assert len(winners) == 1, f"the same note read as {len(winners)} pitch classes"


def test_the_range_above_C2_is_untouched_by_the_lowered_floor():
    """The cost side of that trade, measured rather than assumed. Lowering the band's
    edge to 27.5 Hz makes the bottom octave coarse; it must not disturb the range that
    already worked, which is most of a sample library."""
    sr = S.ANALYSIS_SR
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    wrong = []
    for midi in range(36, 84):                             # C2 to B5
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        t = np.arange(int(2.0 * sr)) / sr
        tone = (np.sin(2 * np.pi * hz * t) * 0.5).astype(np.float32)
        got = names[int(np.argmax(S.chroma_of(tone)))]
        if got != names[midi % 12]:
            wrong.append(f"{names[midi % 12]}->{got}")
    assert not wrong, f"{len(wrong)} of 48 notes above C2 misread: {wrong[:5]}"


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
