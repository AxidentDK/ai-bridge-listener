"""A/B: does a CORRECT spectrogram actually produce better tags?

The mel is proven not to match Essentia's. The tempting next move is to fix it — but
that assumes correct spectrograms improve accuracy, and we have no evidence for that.
If 37% is the models' ceiling on one-shot material, matching Essentia perfectly
changes nothing and the work is wasted.

So measure first. Tag the same files twice, changing ONE thing:

    A  mel from our numpy code      -> same ONNX -> same heads -> tags
    B  mel from Essentia itself     -> same ONNX -> same heads -> tags

Same audio array feeds both (decoded once), same models, same head weights, same
thresholds. The only variable is the spectrogram. Then run listener.evaluate over
each database and compare.

Run inside WSL:
    bash tools/run_in_wsl.sh ab <filelist.txt> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from listener import decode, heads as heads_mod  # noqa: E402
from listener.db import Store  # noqa: E402

import essentia.standard as es  # noqa: E402
import onnxruntime as ort  # noqa: E402

MODELS = Path("/mnt/c/Users/Kim/.ai-bridge/models")
EVENT_TOP_K, EVENT_FLOOR = 15, 0.02
BATCH = 8

#: Which Essentia algorithm corresponds to each of our configs.
ESSENTIA_ALGO = {"effnet": "TensorflowInputMusiCNN", "yamnet": "TensorflowInputVGGish"}


def load_audio(path: str) -> np.ndarray | None:
    """Decode ONCE. Both paths get the identical array, so the comparison isolates
    the spectrogram rather than also measuring two different decoders."""
    try:
        data, sr = sf.read(path, frames=int(30 * 48000), dtype="float32", always_2d=True)
    except Exception:                                              # noqa: BLE001
        return None
    if data.size == 0:
        return None
    return decode._resample(data.mean(axis=1).astype(np.float32), sr)


def pad_or_tile(audio: np.ndarray, cfg: decode.MelConfig) -> np.ndarray:
    """Same short-file handling for both paths — otherwise B would be penalised on
    one-shots for a reason that has nothing to do with the spectrogram."""
    need = cfg.frame_size + (cfg.patch_frames - 1) * cfg.hop_size
    if len(audio) >= need or not len(audio):
        return audio
    return np.tile(audio, int(np.ceil(need / len(audio))))[:need]


def essentia_patches(audio: np.ndarray, cfg: decode.MelConfig) -> np.ndarray:
    """Essentia's own mel, stacked into the patches the ONNX model expects."""
    alg = getattr(es, ESSENTIA_ALGO[cfg.name])()
    frames = [alg(f) for f in es.FrameGenerator(
        pad_or_tile(audio, cfg), frameSize=cfg.frame_size, hopSize=cfg.hop_size,
        startFromZero=True)]
    if not frames:
        return np.zeros((0, cfg.patch_frames, cfg.n_mels), dtype=np.float32)
    mel = np.asarray(frames, dtype=np.float32)
    n = len(mel) // cfg.patch_frames
    if n == 0:
        return np.zeros((0, cfg.patch_frames, cfg.n_mels), dtype=np.float32)
    return mel[:n * cfg.patch_frames].reshape(n, cfg.patch_frames, cfg.n_mels)


def our_patches(audio: np.ndarray, cfg: decode.MelConfig) -> np.ndarray:
    return decode.melspectrogram(pad_or_tile(audio, cfg), cfg)


class Models:
    """Loaded once and shared by both arms, so they cannot differ."""

    def __init__(self) -> None:
        self.effnet = ort.InferenceSession(
            str(MODELS / "discogs-effnet-bsdynamic-1.onnx"),
            providers=["CPUExecutionProvider"])
        self.effnet_in = self.effnet.get_inputs()[0].name
        self.yamnet = ort.InferenceSession(
            str(MODELS / "audioset-yamnet-1.onnx"), providers=["CPUExecutionProvider"])
        self.yamnet_in = self.yamnet.get_inputs()[0].name
        self.heads = heads_mod.load_all(models=MODELS)
        self.style = heads_mod.builtin_400_classes(MODELS)
        import json
        self.events = json.loads(
            (MODELS / "audioset-yamnet-1.json").read_text(encoding="utf-8"))["classes"]
        self.version = f"ab-{len(self.heads)}heads"

    def tag(self, eff: np.ndarray, yam: np.ndarray) -> list[tuple[str, str, float, str]]:
        out: list[tuple[str, str, float, str]] = []
        if len(eff):
            embs, acts = [], []
            for i in range(0, len(eff), BATCH):
                r = self.effnet.run(None, {self.effnet_in: eff[i:i + BATCH]})
                acts.append(r[0]); embs.append(r[1])
            emb = np.concatenate(embs).mean(axis=0)
            style = np.concatenate(acts).mean(axis=0)
            for name, head in self.heads.items():
                for label, score in heads_mod.tags_for(head, emb):
                    out.append((name, label, score, self.version))
            if self.style and len(style) == len(self.style):
                for i in np.argsort(style)[::-1][:3]:
                    if style[i] >= 0.10:
                        out.append(("style_discogs400", self.style[i],
                                    float(style[i]), self.version))
        if len(yam):
            acts = []
            for i in range(0, len(yam), BATCH):
                acts.append(self.yamnet.run(None, {self.yamnet_in: yam[i:i + BATCH]})[1])
            scores = np.concatenate(acts).mean(axis=0)
            if len(scores) == len(self.events):
                for i in np.argsort(scores)[::-1][:EVENT_TOP_K]:
                    if scores[i] >= EVENT_FLOOR:
                        out.append(("audio_event", self.events[i], float(scores[i]),
                                    self.version))
        return out


def main(list_path: str, out_dir: str) -> int:
    paths = [ln.strip() for ln in Path(list_path).read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    models = Models()
    print(f"{len(paths)} files | {len(models.heads)} heads | "
          f"{len(models.events)} event classes")

    stores = {"A_ours": Store(out / "ab_ours.db", "A-our-mel"),
              "B_essentia": Store(out / "ab_essentia.db", "B-essentia-mel")}
    done = failed = 0
    for path in paths:
        audio = load_audio(path)
        if audio is None or not len(audio):
            failed += 1
            continue
        try:
            arms = {
                "A_ours": (our_patches(audio, decode.EFFNET),
                           our_patches(audio, decode.YAMNET)),
                "B_essentia": (essentia_patches(audio, decode.EFFNET),
                               essentia_patches(audio, decode.YAMNET)),
            }
        except Exception as exc:                                   # noqa: BLE001
            print(f"  mel failed on {Path(path).name}: {exc}")
            failed += 1
            continue
        # Windows-style path so listener.evaluate's rules see the same names.
        win = path.replace("/mnt/c/", "C:/").replace("/mnt/d/", "D:/")
        for arm, (eff, yam) in arms.items():
            stores[arm].record(win, size=0, mtime=0.0,
                               duration=len(audio) / decode.TARGET_SR,
                               tags=models.tag(eff, yam))
        done += 1
        if done % 50 == 0:
            for s in stores.values():
                s.commit()
            print(f"  {done}/{len(paths)}")
    for s in stores.values():
        s.close()
    print(f"\ndone {done}, failed {failed}")
    print(f"A (ours):     {out / 'ab_ours.db'}")
    print(f"B (essentia): {out / 'ab_essentia.db'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
