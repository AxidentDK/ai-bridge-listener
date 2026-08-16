"""Tests for listener/drums.py — the classifier that supplies the words AudioSet lacks.

The labelling rules carry the judgement here, so they get the most attention: the
training set is only as good as its refusal to learn from contradictions.

No pytest: run the file.
"""
import os
import sqlite3
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from listener import drums  # noqa: E402
from listener.db import Store  # noqa: E402


def test_filename_beats_folder_when_they_disagree():
    """The known failure of folder labels: a file in Cymbals/ called `Hat 01.wav` is a
    hi-hat. The filename is consistently the more specific of the two in this library,
    so it wins."""
    assert drums.label_from_path(r"D:\Kit\Cymbals\Hat 01.wav") == "hat"
    assert drums.label_from_path(r"D:\Kit\Percussion\Kick Deep.wav") == "kick"


def test_agreement_is_the_strongest_case():
    assert drums.label_from_path(r"D:\Kit\Kicks\Kick 08.wav") == "kick"
    assert drums.label_from_path(r"D:\Kit\Snares\Snare Tight.wav") == "snare"


def test_a_folder_alone_is_not_enough():
    """A file sitting in Kicks/ whose own name says nothing may be anything — an
    ambience, a loop, a mislaid vocal. Folder-only evidence is not trained on."""
    assert drums.label_from_path(r"D:\Kit\Kicks\Untitled 3.wav") is None


def test_an_ambiguous_filename_is_dropped_not_guessed():
    """Two specific and different claims in one name is a contradiction, and the answer
    is to drop the file rather than pick one — better 4,000 clean examples than 10,000
    noisy ones."""
    assert drums.label_from_path(r"D:\Kit\Loops\Kick Snare Combo.wav") is None


def test_the_classes_include_what_audioset_cannot_say():
    """The entire reason this module exists: AudioSet has no `kick` class and no `tom`
    class, so no amount of better audio analysis produces those words."""
    assert "kick" in drums.CLASSES
    assert "tom" in drums.CLASSES


def test_it_learns_a_separable_problem_and_reports_honestly():
    """Two well-separated synthetic clusters must be learnable, and the reported
    accuracy must reflect held-out data rather than what it was fitted on."""
    rng = np.random.RandomState(0)
    n, d = 200, 32
    kick = np.hstack([rng.randn(n, d // 2) + 4.0, rng.randn(n, d // 2)])
    hat = np.hstack([rng.randn(n, d // 2), rng.randn(n, d // 2) + 4.0])
    X = np.vstack([kick, hat]).astype(np.float32)
    y = np.array([drums.CLASSES.index("kick")] * n + [drums.CLASSES.index("hat")] * n)
    report = drums.evaluate(X, y)
    assert report["overall"] > 0.9, report
    assert report["n_test"] > 0 and report["n_train"] > report["n_test"]


def test_probabilities_are_probabilities():
    rng = np.random.RandomState(1)
    X = rng.randn(40, 16).astype(np.float32)
    y = rng.randint(0, len(drums.CLASSES), size=40)
    probs = drums.predict(drums.fit(X, y, epochs=20), X)
    assert probs.shape == (40, len(drums.CLASSES))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0).all()


def test_applying_to_an_index_writes_only_confident_verdicts_and_no_loops():
    """Tags are replaced wholesale, long files are left alone (a 4-bar loop in Kicks/
    is a loop that features kicks, not a kick sample), and nothing is written below the
    confidence floor."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index.db")
        store = Store(path, "test-1")
        rng = np.random.RandomState(2)
        for i in range(6):
            store.record(f"{i}.wav", size=i, mtime=float(i),
                         duration=0.4 if i < 4 else 30.0,
                         embedding=rng.randn(8).astype(np.float32))
        store.commit()
        store.close()

        X = rng.randn(60, 8).astype(np.float32)
        y = rng.randint(0, len(drums.CLASSES), size=60)
        model = drums.fit(X, y, epochs=50)

        result = drums.apply_to_index(path, model, min_confidence=0.0)
        assert result["skipped_too_long"] == 2, result
        assert result["tagged"] == 4, result

        conn = sqlite3.connect(path)
        rows = conn.execute(
            "SELECT namespace, label, confidence, model FROM tags").fetchall()
        assert len(rows) == 4
        assert {r[0] for r in rows} == {drums.NAMESPACE}
        assert all(r[1] in drums.CLASSES for r in rows)
        assert all(0.0 <= r[2] <= 1.0 for r in rows)
        assert {r[3] for r in rows} == {drums.MODEL_NAME}

        # An impossible floor must write nothing at all, and must first clear the old.
        again = drums.apply_to_index(path, model, min_confidence=1.01)
        assert again["tagged"] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM tags WHERE namespace=?",
            (drums.NAMESPACE,)).fetchone()[0] == 0
        conn.close()


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
