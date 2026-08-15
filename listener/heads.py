"""Classification heads: load them, and apply the RIGHT activation to each.

Every head is the same two-layer network on top of a 1280-d embedding:

    h = relu(x @ W1 + b1)        # 1280 -> 512
    z = h @ W2 + b2              # 512  -> n

but what happens to ``z`` is **not** the same, and this is the trap:

    Softmax   20 heads   single-label — the classes are mutually exclusive
    Sigmoid    6 heads   multi-label  — each class is an independent yes/no
    Linear     2 heads   regression   — one number, not a class at all

Applying sigmoid to a softmax head does not raise. It returns values in [0, 1] that
look exactly like probabilities and are wrong — so the activation is read from each
head's own metadata and never assumed.

TensorFlow is not involved: the weights come straight out of the frozen graph with a
protobuf wire-format walk (see ``read_weights``). TF has no Python 3.14 wheel, and
tf2onnx cannot even import without it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODELS = Path.home() / ".ai-bridge" / "models"
SUFFIX = "-discogs-effnet-1"

# genre_discogs400 is NOT loaded as a head: the embedding model's own second output
# (`activations`, 400-d) already IS this classifier. Its .json is still read, for the
# class names.
BUILTIN_400 = "genre_discogs400"


# ---------------------------------------------------------------- protobuf reading

def _varint(buf, i):
    val = shift = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _fields(buf):
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fnum, wtype = key >> 3, key & 7
        if wtype == 0:
            val, i = _varint(buf, i)
            yield fnum, val
        elif wtype == 2:
            ln, i = _varint(buf, i)
            yield fnum, buf[i:i + ln]
            i += ln
        elif wtype == 5:
            yield fnum, buf[i:i + 4]
            i += 4
        elif wtype == 1:
            yield fnum, buf[i:i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wtype}")


def _parse_tensor(buf):
    shape, content = [], None
    for fnum, payload in _fields(buf):
        if fnum == 2:
            for d_num, d_payload in _fields(payload):
                if d_num == 2:
                    for s_num, size in _fields(d_payload):
                        if s_num == 1:
                            shape.append(size)
        elif fnum == 4:
            content = payload
    if content is None or not shape:
        return None
    arr = np.frombuffer(content, dtype="<f4")
    return arr.reshape(shape) if arr.size == int(np.prod(shape)) else None


def read_weights(pb_path: Path) -> dict[str, np.ndarray]:
    """{const_node_name: ndarray} from a frozen GraphDef, without TensorFlow."""
    out = {}
    for fnum, node in _fields(pb_path.read_bytes()):
        if fnum != 1:
            continue
        name = op = None
        tensors = []
        for f, payload in _fields(node):
            if f == 1:
                name = payload.decode("utf-8", "replace")
            elif f == 2:
                op = payload.decode("utf-8", "replace")
            elif f == 5:
                for a, apayload in _fields(payload):
                    if a == 2:
                        for v, vpayload in _fields(apayload):
                            if v == 8:
                                t = _parse_tensor(vpayload)
                                if t is not None:
                                    tensors.append(t)
        if op == "Const" and tensors and name:
            out[name] = tensors[0]
    return out


# ---------------------------------------------------------------------- the heads

@dataclass
class Head:
    name: str
    classes: list[str]
    activation: str                 # Softmax | Sigmoid | Linear
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    @property
    def is_regression(self) -> bool:
        return self.activation == "Linear"

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        h = np.maximum(embedding @ self.W1 + self.b1, 0.0)
        z = h @ self.W2 + self.b2
        if self.activation == "Softmax":
            e = np.exp(z - z.max())
            return e / e.sum()
        if self.activation == "Sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        return z                                     # Linear: a value, not a score


def _activation_from(meta: dict) -> str:
    outs = meta.get("schema", {}).get("outputs", [])
    for o in outs:
        if o.get("output_purpose") == "predictions":
            return o.get("op") or "Sigmoid"
    return (outs[0].get("op") if outs else None) or "Sigmoid"


def load_head(name: str, models: Path = MODELS) -> Head:
    meta = json.loads((models / f"{name}{SUFFIX}.json").read_text(encoding="utf-8"))
    w = read_weights(models / f"{name}{SUFFIX}.pb")
    missing = [k for k in ("dense/kernel", "dense/bias", "dense_1/kernel", "dense_1/bias")
               if k not in w]
    if missing:
        raise KeyError(f"{name}: missing {missing}")
    return Head(name=name, classes=meta.get("classes", []),
                activation=_activation_from(meta),
                W1=w["dense/kernel"], b1=w["dense/bias"],
                W2=w["dense_1/kernel"], b2=w["dense_1/bias"])


def load_all(models: Path = MODELS, only: list[str] | None = None,
             report: bool = True) -> dict[str, Head]:
    """Every head that loads. A head that will not load is skipped, not fatal.

    Skipping is right — one unreadable head must not stop a 30,000-file scan — but it
    used to be SILENT, and that is the dangerous half. A classification head that
    failed to load would simply produce no tags in its namespace, and the only symptom
    is a producer searching for a style and being told their library does not contain
    it. Failures are now named on stderr.

    Two heads in the current model set legitimately fail (`approachability_regression`
    and `engagement_regression`: no `dense_1/bias`, being regressions rather than
    classifiers) and cost nothing, since ``tags_for`` returns nothing for a regression
    anyway. That is exactly why the message matters — without it there is no way to
    tell those apart from a head that should have worked.
    """
    heads: dict[str, Head] = {}
    skipped: list[str] = []
    for pb in sorted(models.glob(f"*{SUFFIX}.pb")):
        name = pb.name[:-len(f"{SUFFIX}.pb")]
        if name == BUILTIN_400 or (only and name not in only):
            continue
        try:
            heads[name] = load_head(name, models)
        except Exception as exc:                                 # noqa: BLE001
            skipped.append(f"{name} ({type(exc).__name__}: {exc})")
    if skipped and report:
        print(f"  {len(heads)} heads loaded; {len(skipped)} skipped:", file=sys.stderr)
        for line in skipped:
            print(f"    - {line}", file=sys.stderr)
    return heads


def builtin_400_classes(models: Path = MODELS) -> list[str]:
    """Class names for the embedding model's own 400-d `activations` output."""
    try:
        meta = json.loads((models / f"{BUILTIN_400}{SUFFIX}.json").read_text(encoding="utf-8"))
        return meta.get("classes", [])
    except OSError:
        return []


def tags_for(head: Head, embedding: np.ndarray, *, top_k: int = 5,
             floor: float = 0.15) -> list[tuple[str, float]]:
    """(label, score) pairs worth recording. Regression heads return [] — their
    single value belongs in `properties`, not in a tag table.

    ``embedding`` may be one vector or a stack of per-patch vectors. Given a stack,
    the head runs on EACH patch and the PREDICTIONS are averaged — which is not the
    same as averaging the embeddings first, because a relu-then-softmax head is
    non-linear and does not commute with a mean. Averaging inputs describes a
    half-drone, half-techno file as the point between them, which may be neither.
    """
    if head.is_regression:
        return []
    if embedding.ndim > 1:
        probs = np.mean([head.predict(vec) for vec in embedding], axis=0)
    else:
        probs = head.predict(embedding)
    if head.activation == "Softmax":
        # Mutually exclusive: only the winner is a claim about the sound.
        i = int(np.argmax(probs))
        label = head.classes[i] if i < len(head.classes) else str(i)
        return [(label, float(probs[i]))]
    order = np.argsort(probs)[::-1][:top_k]
    return [(head.classes[i] if i < len(head.classes) else str(i), float(probs[i]))
            for i in order if probs[i] >= floor]
