"""Orchestration: resumable, parallel, interruptible.

Shape of the run, and why:

* **Decoding runs in a process pool.** It is the measured bottleneck and it is pure —
  a path in, an array out — so it parallelises with no shared state.
* **The model runs in the main process**, batched. It is ~2 ms per patch, so it needs
  no help, and keeping one session avoids loading a 17 MB model per worker.
* **Writes are single-threaded.** SQLite takes one writer, and results arrive as
  workers finish anyway.
* **Progress is committed as it goes.** Ctrl-C is a supported way to stop: whatever
  finished is durable, and the next run picks up from there.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import decode, scan
from .db import Store

MODELS = Path.home() / ".ai-bridge" / "models"
BATCH_PATCHES = 8            # measured sweet spot on CPU; larger is WORSE (cache)
COMMIT_EVERY = 50            # files, not seconds: bounded loss on a hard kill

#: How many AudioSet events to keep per file, and the confidence floor. These are part
#: of the analyzer VERSION (see Tagger.version) because changing them changes what is
#: stored — and anything that changes stored output must force a re-analysis.
EVENT_TOP_K = 15
EVENT_FLOOR = 0.02

_stop = False


def _handle_sigint(_sig, _frm):
    global _stop
    if _stop:
        print("\nsecond interrupt — exiting now", file=sys.stderr)
        raise SystemExit(130)
    _stop = True
    print("\ninterrupt: finishing in-flight files, then stopping "
          "(everything done so far is saved)", file=sys.stderr)


class Tagger:
    """Embedding model + every compatible classification head. Loaded once per run.

    All 28 heads together are ~1.3 MB of matrix multiply per file — irrelevant next to
    decoding, so there is no reason to be selective at analysis time. Be selective when
    *searching*, not when recording.
    """

    def __init__(self, only=None):
        import onnxruntime as ort
        from . import heads as heads_mod            # noqa: PLC0415

        onnx = MODELS / "discogs-effnet-bsdynamic-1.onnx"
        self.session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.heads = heads_mod.load_all(only=only)
        self.style_classes = heads_mod.builtin_400_classes()
        self._tags_for = heads_mod.tags_for

        # YAMNet — AudioSet events. The music heads have no vocabulary for a snare;
        # AudioSet has Bass drum, Snare drum, Hi-hat, Cymbal, Drum machine. For a
        # library of one-shots this is the model that can actually describe them.
        self.yamnet = self.yamnet_classes = None
        ypath = MODELS / "audioset-yamnet-1.onnx"
        if ypath.exists():
            self.yamnet = ort.InferenceSession(str(ypath),
                                               providers=["CPUExecutionProvider"])
            self.yamnet_input = self.yamnet.get_inputs()[0].name
            meta = MODELS / "audioset-yamnet-1.json"
            if meta.exists():
                self.yamnet_classes = json.loads(
                    meta.read_text(encoding="utf-8")).get("classes", [])
        # The version must cover EVERYTHING that changes what gets stored, not just
        # which models are loaded. It previously named only the model set, so raising
        # the event top_k left the version identical — and the resumability check
        # dutifully skipped all 22,100 files and reported success without re-analysing
        # anything. Resumability worked; the version was an incomplete description of
        # the analysis.
        self.version = (f"{decode.MEL_VERSION}+discogs-effnet-1+{len(self.heads)}heads-1"
                        + (f"+yamnet-1(k{EVENT_TOP_K},f{EVENT_FLOOR})"
                           if self.yamnet else ""))

    def embed(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[n, 128, 96] -> (embedding[1280], style_activations[400]).

        Averaged over patches: these heads describe a FILE, not an instant. The 400-d
        activations come free from the same forward pass — that output IS the
        Discogs400 style classifier, so its separate head is never loaded.
        """
        embs, acts = [], []
        for i in range(0, len(patches), BATCH_PATCHES):
            res = self.session.run(None, {self.input_name: patches[i:i + BATCH_PATCHES]})
            acts.append(res[0])                      # activations [n, 400]
            embs.append(res[1])                      # embeddings  [n, 1280]
        return (np.concatenate(embs, axis=0).mean(axis=0),
                np.concatenate(acts, axis=0).mean(axis=0))

    def tag_events(self, patches: np.ndarray, top_k: int = EVENT_TOP_K,
                   floor: float = EVENT_FLOOR):
        """AudioSet events from YAMNet — what the sound IS, not what genre it is.

        top_k was 6, and that was throwing away right answers. AudioSet's 521 classes
        are hierarchical, so generic parents (`Music`, `Musical instrument`, `Sound
        effect`) reliably outrank the specific child you actually want: a string
        sample was tagged `Bowed string instrument` *below* the cutoff, so search
        never saw it. Keeping 15 at a lower floor costs a few MB across the library
        and recovers labels the model had already found.
        """
        if self.yamnet is None or patches is None or not len(patches):
            return []
        acts = []
        for i in range(0, len(patches), BATCH_PATCHES):
            acts.append(self.yamnet.run(None,
                                        {self.yamnet_input: patches[i:i + BATCH_PATCHES]})[1])
        scores = np.concatenate(acts, axis=0).mean(axis=0)
        if not self.yamnet_classes or len(scores) != len(self.yamnet_classes):
            return []
        order = np.argsort(scores)[::-1][:top_k]
        return [("audio_event", self.yamnet_classes[i], float(scores[i]), self.version)
                for i in order if scores[i] >= floor]

    def tag(self, patches: dict, style_top_k: int = 3, style_floor: float = 0.10):
        """patches: {'effnet': [...], 'yamnet': [...]} from one decode."""
        out: list[tuple[str, str, float, str]] = []
        eff = patches.get("effnet") if isinstance(patches, dict) else patches
        if eff is not None and len(eff):
            emb, style = self.embed(eff)
            for name, head in self.heads.items():
                for label, score in self._tags_for(head, emb):
                    out.append((name, label, score, self.version))
            if self.style_classes and len(style) == len(self.style_classes):
                order = np.argsort(style)[::-1][:style_top_k]
                out += [("style_discogs400", self.style_classes[i], float(style[i]),
                         self.version) for i in order if style[i] >= style_floor]
        if isinstance(patches, dict):
            out += self.tag_events(patches.get("yamnet"))
        return out


def run(db_path: Path, candidates: list[scan.Candidate], workers: int | None = None,
        tag: bool = True) -> dict:
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    tagger = Tagger() if tag else None
    analyzer = tagger.version if tagger else "decode-only"

    store = Store(db_path, analyzer)
    todo, skipped = scan.plan(candidates, store.already_done(), analyzer)
    print(f"{len(candidates):,} candidates | {skipped:,} already done | "
          f"{len(todo):,} to analyse | {workers} decode workers")
    if not todo:
        store.close()
        return {"analysed": 0, "skipped": skipped, "failed": 0, "seconds": 0.0}

    signal.signal(signal.SIGINT, _handle_sigint)
    started = time.perf_counter()
    done = failed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(decode.process_file, c.path): c for c in todo}
        for fut in as_completed(futures):
            cand = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:                            # noqa: BLE001
                res = decode.Decoded(cand.path, None, None, None, None,
                                     f"worker died: {exc}"[:300])
            tags = None
            error = res.error
            # res.patches is now {config_name: array}; "did we get anything usable"
            # means at least one config produced patches, not that the dict exists.
            usable = bool(res.patches) and any(
                v is not None and len(v) for v in res.patches.values())
            if error is None and usable:
                if tagger:
                    try:
                        tags = tagger.tag(res.patches)
                    except Exception as exc:                    # noqa: BLE001
                        error = f"tagging failed: {exc}"[:300]
            elif error is None:
                error = "no audio decoded"

            store.record(cand.path, size=cand.size, mtime=cand.mtime,
                         duration=res.duration_sec, sample_rate=res.sample_rate,
                         channels=res.channels, error=error, tags=tags)
            done += 1
            failed += bool(error)
            if done % COMMIT_EVERY == 0:
                store.commit()
                rate = done / (time.perf_counter() - started)
                print(f"  {done:,}/{len(todo):,}  {rate:.1f} files/s  {failed} failed")
            if _stop:
                for f in futures:
                    f.cancel()
                break

    elapsed = time.perf_counter() - started
    counts = store.counts()
    store.close()
    return {"analysed": done, "skipped": skipped, "failed": failed,
            "seconds": round(elapsed, 1),
            "files_per_sec": round(done / elapsed, 2) if elapsed else 0.0,
            "interrupted": _stop, **counts}
