"""Name a drum sound — the vocabulary AudioSet structurally cannot supply.

AudioSet has no `kick` class and no `tom` class. Not "weak at them": the labels do not
exist, so no amount of better audio analysis can produce them, and a producer searching
a library for a kick is asking a question the tagger has no word for. This module adds
the ten words that matter, by training a small classifier on the embeddings the scan
already stored.

WHY THIS IS CHEAP. The 1280-d Discogs-EffNet embedding of every file is in the database
already — that was the point of storing it. Training is a matrix multiply over ~4,000
vectors, and labelling the whole library afterwards is another one. Neither decodes a
single audio file, so this runs in seconds against a 30,000-file library rather than the
65 minutes a re-scan costs.

WHERE THE LABELS COME FROM. Nobody hand-labelled anything: the library labels itself.
A file in `Kicks/` called `Kick 08.wav` is a kick, and there are thousands of them.
That is weak supervision, and its known failure is the contradiction — a file in
`Cymbals/` called `Hat 01.wav` is a hi-hat, not a cymbal. Folder and filename are both
read, the FILENAME WINS when the two disagree (it is consistently the more specific of
the two in this library), and genuine contradictions are DROPPED rather than trained on.
Better 4,000 clean examples than 10,000 noisy ones.

Measured on a stratified 20% held-out split, 3,999 labelled files:

    overall 81%      kick 94%   hat 90%   tom 87%   snare 82%
                     crash 79%  clap 71%  perc 62%  ride 60%  rim 59%  shaker 31%

The confusions are the ones a person would make on a single hit — shaker with hi-hat,
rim with snare, tom with kick — and the thin classes are thin because the library holds
only 65 shakers to learn from, not because the model is confused about what a shaker is.

Confidence is honest here, which is not something to assume: at >= 0.9 it is 91%
correct, against 81% overall. So a caller can trade recall for precision and get what
they asked for.

numpy only. The listener's entire promise is a ~20 MB runtime, and a linear model on a
good embedding does not need more than that.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np

MODEL_PATH = Path.home() / ".ai-bridge" / "models" / "drum-classifier.npz"
MODEL_NAME = "drum-linear-1"
NAMESPACE = "drum"

#: What a producer reaches for, in their words. `perc` is the deliberate catch-all for
#: hand percussion — conga, bongo, tabla, cowbell — which is a real search term even
#: though it names a family rather than an instrument.
CLASS_PATTERNS = {
    "kick":   r"\b(kick|bd|bassdrum|bass ?drum)\b",
    "snare":  r"\b(snare|sd|snr)\b",
    "clap":   r"\b(clap|clp|handclap)\b",
    "hat":    r"\b(hi.?hat|hihat|hat|hh|chh|ohh)\b",
    "ride":   r"\b(ride|rd)\b",
    "crash":  r"\b(crash|cr(a)?sh)\b",
    "tom":    r"\b(tom|floor ?tom|rack ?tom)\b",
    "rim":    r"\b(rim|rimshot|sidestick|side ?stick)\b",
    "shaker": r"\b(shaker|shk|maraca)\b",
    "perc":   r"\b(perc|conga|bongo|tabla|djembe|cowbell|tamb(ourine)?|woodblock)\b",
}
CLASSES = sorted(CLASS_PATTERNS)
_COMPILED = {name: re.compile(rx, re.I) for name, rx in CLASS_PATTERNS.items()}

#: Above this a file is a loop that FEATURES kicks, not a kick sample. A drum one-shot
#: is short; three seconds is generous for one and short for a bar of anything.
MAX_ONE_SHOT_SECONDS = 3.0

#: THE GATE, and the reason it exists. This is a CLOSED-SET classifier: it was trained
#: only on files whose names say a drum word, so it has no way to answer "not a drum" —
#: softmax must pick one of ten. Applied to a whole library it therefore labels
#: everything, confidently. Measured on the first run: 13,697 files tagged, including
#: `Rave Synth Chord Em.wav` as a tom at 0.98 and a rack preset as a kick at 1.00.
#:
#: The fix is a division of labour rather than a better classifier. AudioSet CAN say
#: "this is percussion" — that is a class it has — and cannot say WHICH drum, which is
#: exactly the hole this module fills. So AudioSet gates, and this names. Together with
#: the one-shot test that is two independent signals, and it took 13,697 down to 7,409.
#:
#: An EXPLICIT list, never a LIKE pattern: `%rum%` matches "inst-rum-ent", which put
#: "Musical instrument", "Brass instrument" and "Plucked string instrument" into a list
#: of percussion on the first attempt.
PERCUSSIVE_EVENTS = (
    "Percussion", "Drum kit", "Drum machine", "Drum", "Snare drum", "Bass drum",
    "Cymbal", "Hi-hat", "Rimshot", "Wood block", "Tabla", "Tambourine", "Maraca",
    "Timpani", "Gong", "Clapping", "Drum roll", "Electronic drum", "Cowbell",
    "Bongo", "Conga", "Steelpan", "Crash cymbal", "Ride cymbal", "Hand clap",
    "Finger snapping", "Castanets", "Snare", "Tom-tom", "Sizzle cymbal",
)
#: Do not record a verdict weaker than this. At 0.5 the classifier is 83% right; below
#: it the label is a guess wearing a number.
MIN_CONFIDENCE = 0.5


def label_from_path(path: str) -> str | None:
    """The drum class this file's own name claims, or None.

    Filename beats folder when they disagree, and a genuine contradiction — both
    specific and different — returns None so the caller can drop it.
    """
    parts = path.replace("/", "\\").split("\\")
    name_hits = {c for c, rx in _COMPILED.items() if rx.search(parts[-1])}
    folder_hits = {c for c, rx in _COMPILED.items() if rx.search(" ".join(parts[:-1]))}
    agreed = name_hits & folder_hits
    if len(agreed) == 1:
        return next(iter(agreed))
    if len(name_hits) == 1:
        return next(iter(name_hits))
    # Folder alone is weaker evidence and is not used for training: a file sitting in
    # `Drums/Kicks/` whose name says nothing may still be anything.
    return None


def _embedding(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16)[:dim].astype(np.float32)


def training_set(db_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X, y, paths) for every file whose name states a class unambiguously."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    vectors, labels, paths = [], [], []
    for path, duration, blob, dim in conn.execute(
            "SELECT f.path, f.duration_sec, e.vector, e.dim FROM files f "
            "JOIN embeddings e ON e.file_id = f.id WHERE f.error IS NULL"):
        if duration and duration > MAX_ONE_SHOT_SECONDS:
            continue
        label = label_from_path(path)
        if label is None:
            continue
        vectors.append(_embedding(blob, dim))
        labels.append(CLASSES.index(label))
        paths.append(path)
    conn.close()
    if not vectors:
        return np.zeros((0, 0)), np.zeros(0, dtype=int), []
    return np.stack(vectors), np.array(labels), paths


def fit(X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.5,
        l2: float = 1e-3, seed: int = 0) -> dict:
    """Multinomial logistic regression by gradient descent.

    L2 is not optional: 1280 features against ~3,000 training rows will otherwise fit
    the noise perfectly and generalise to nothing.
    """
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-6            # embeddings are not zero-centred
    Xs = (X - mean) / scale
    rng = np.random.RandomState(seed)
    n, d = Xs.shape
    W = rng.randn(d, len(CLASSES)).astype(np.float32) * 0.01
    b = np.zeros(len(CLASSES), dtype=np.float32)
    Y = np.zeros((n, len(CLASSES)), dtype=np.float32)
    Y[np.arange(n), y] = 1.0
    for _ in range(epochs):
        p = _softmax(Xs @ W + b)
        g = (p - Y) / n
        W -= lr * (Xs.T @ g + l2 * W)
        b -= lr * g.sum(axis=0)
    return {"W": W, "b": b, "mean": mean, "scale": scale,
            "classes": np.array(CLASSES)}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """Class probabilities, one row per input embedding."""
    Xs = (X - model["mean"]) / model["scale"]
    return _softmax(Xs @ model["W"] + model["b"])


def evaluate(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Stratified 80/20 split, so the thin classes appear on both sides."""
    rng = np.random.RandomState(seed)
    train, test = [], []
    for c in range(len(CLASSES)):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * 0.8)
        train += list(idx[:cut])
        test += list(idx[cut:])
    train, test = np.array(train), np.array(test)
    model = fit(X[train], y[train], seed=seed)
    probs = predict(model, X[test])
    pred = probs.argmax(axis=1)
    per_class = {}
    for c, name in enumerate(CLASSES):
        sel = y[test] == c
        if sel.any():
            per_class[name] = (float((pred[sel] == c).mean()), int(sel.sum()))
    confident = probs.max(axis=1) >= 0.9
    return {
        "overall": float((pred == y[test]).mean()),
        "per_class": per_class,
        "n_train": len(train), "n_test": len(test),
        "at_high_confidence": (float((pred[confident] == y[test][confident]).mean())
                               if confident.any() else None),
    }


def save(model: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **model)


def load(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def apply_to_index(db_path: Path, model: dict,
                   min_confidence: float = MIN_CONFIDENCE) -> dict:
    """Write a `drum` tag for every file the classifier is confident about.

    No audio is decoded: this reads stored embeddings and writes tags, so relabelling
    the whole library after retraining costs seconds rather than a re-scan.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    # Ask the gate FIRST, in SQL, so the classifier is never even shown a synth pad.
    # See PERCUSSIVE_EVENTS for why this is not optional.
    placeholders = ",".join("?" for _ in PERCUSSIVE_EVENTS)
    rows = conn.execute(
        f"""SELECT f.id, f.duration_sec, e.vector, e.dim FROM files f            -- noqa: S608
            JOIN embeddings e ON e.file_id = f.id
            LEFT JOIN properties p ON p.file_id = f.id
            WHERE f.error IS NULL
              AND (p.kind IS NULL OR p.kind = 'one_shot')
              AND EXISTS (SELECT 1 FROM tags a WHERE a.file_id = f.id
                          AND a.namespace = 'audio_event'
                          AND a.label IN ({placeholders}))""",
        PERCUSSIVE_EVENTS).fetchall()
    total = conn.execute(
        "SELECT COUNT(*) FROM files WHERE error IS NULL").fetchone()[0]
    conn.execute("DELETE FROM tags WHERE namespace = ?", (NAMESPACE,))
    written = skipped_long = 0
    batch = []
    for file_id, duration, blob, dim in rows:
        if duration and duration > MAX_ONE_SHOT_SECONDS:
            skipped_long += 1
            continue
        probs = predict(model, _embedding(blob, dim)[None, :])[0]
        best = int(probs.argmax())
        if probs[best] < min_confidence:
            continue
        batch.append((file_id, NAMESPACE, CLASSES[best], float(probs[best]), MODEL_NAME))
        written += 1
    conn.executemany(
        "INSERT OR REPLACE INTO tags(file_id, namespace, label, confidence, model) "
        "VALUES(?,?,?,?,?)", batch)
    conn.commit()
    conn.close()
    return {"tagged": written, "skipped_too_long": skipped_long,
            "passed_gate": len(rows), "in_index": total}
