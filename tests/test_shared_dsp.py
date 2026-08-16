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
    """``measure()`` is what both programs will call, so it — not just its parts — has
    to match. Fed the same mono, every field ``features.analyze`` produces must come
    back identical. Pitch is excluded because it is not shared; see the module note."""
    files = _real_files()
    if not files:
        _report_skip("measure/analyze equivalence")
        return
    shared_fields = ("kind", "onsets", "attack_ms", "decay_ms", "stereo_width",
                     "stereo_correlation", "loudness_lufs", "bpm", "bpm_confidence",
                     "bars", "key", "scale", "key_strength")
    for path, _dur, _rate, _ch, _kind in files:
        audio, sr, mono, duration = _decode(path)
        old = F.analyze(audio, sr, mono, duration)
        # Constructed directly rather than through prepare(), because this test is
        # about the MATHS: it must see the same samples features.py sees today. The
        # pre-processing change is measured separately, below.
        new = S.measure(S.Prepared(mono, audio, sr, duration))
        for field in shared_fields:
            assert old.get(field) == new.get(field), (
                f"{field}: {old.get(field)!r} vs {new.get(field)!r} on {path}")


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
    calls ``shared_dsp.key_scores`` instead. This proves that substitution is a no-op
    BEFORE it is made — the winner, the runner-up, the margin and the refusal all have
    to come out the same, or the bridge silently starts naming different keys.
    """
    host = os.path.join(BRIDGE_REPO, "host")
    if not os.path.exists(os.path.join(host, "describe.py")):
        print(f"        (bridge repo not at {BRIDGE_REPO} — skipped)")
        return
    sys.path.insert(0, host)
    try:
        import describe                                        # noqa: PLC0415
    finally:
        sys.path.remove(host)

    assert describe._MAJOR == S._MAJOR, "major profile differs"
    assert describe._MINOR == S._MINOR, "minor profile differs"
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
        assert theirs["key"] == f"{S.NOTE_NAMES[root]} {mode}", (theirs, ours[0])
        assert theirs["confidence"] == round(float(score), 3)
        assert theirs["margin"] == round(float(score - ours[1][0]), 3)
        assert theirs["runner_up"] == f"{S.NOTE_NAMES[ours[1][1]]} {ours[1][2]}"


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
