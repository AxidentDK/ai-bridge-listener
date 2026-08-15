# AI Bridge Listener

> ⚠️ **Work in progress, and honestly measured.** It runs over tens of thousands of
> files and produces useful results, but see **[How good is it?](#how-good-is-it)** —
> the numbers are middling in places and the reasons are documented rather than
> glossed over.

The optional **heavy half** of [AI Bridge for Ableton
Live](https://github.com/AxidentDK/ai-bridge-for-ableton-live). It listens through a
sample library once and writes what it heard into SQLite. The bridge reads that file
and nothing else — it never imports this program, never loads a model.

Without it, the bridge searches by filename and Live's own acoustic similarity. With
it, you can search by **what things sound like**.

## What it needs

`numpy`, `onnxruntime`, `soundfile`, `soxr`. That's it — **about 20 MB**.

**No TensorFlow, no Essentia, no CUDA, no WSL, no GPU.** It runs on a six-year-old
laptop CPU at ~26 files/second, which is roughly 14 minutes for 22,000 files. Plain
`onnxruntime` contains no GPU code at all, so there is nothing to configure.

That is possible because of two things. The expensive part — the Discogs-EffNet
embedding extractor — is published as ONNX. And every classification head is two dense
layers, a ReLU and a sigmoid or softmax, so its four weight arrays are read straight
out of the frozen TensorFlow graph with a protobuf wire-format walk (`heads.py`). No
TensorFlow needed to run a model TensorFlow produced.

## Models

Fetch them yourself from [Essentia's model
repository](https://essentia.upf.edu/models.html) into `~/.ai-bridge/models`:

- `discogs-effnet-bsdynamic-1.onnx` — the embedding extractor (17 MB)
- `audioset-yamnet-1.onnx` — AudioSet audio events, 521 classes (14 MB)
- any `*-discogs-effnet-1.pb` + `.json` classification heads you want

> **Licence:** the MTG models are **CC BY-NC-SA 4.0** — NonCommercial and ShareAlike,
> with a proprietary licence available from MTG on request. They are **not**
> redistributed here and must not be committed. This code is Apache-2.0; the models
> are not, so a commercial user needs to sort that out with MTG directly.

## Use

```sh
python -m listener --folder /path/to/samples --folder /more/samples
python -m listener                      # or take the file list from Live's own index
python -m listener.evaluate             # how good are the tags?
```

Resumable and interruptible: progress commits every 50 files, Ctrl-C is a supported
way to stop, and re-running skips anything whose size, mtime and analyzer version are
unchanged.

## How good is it?

Measured against filenames as weak ground truth over 6,734 files. Sample libraries are
named by content, so a file in `Snares/` called `Snare 08.wav` is a snare — noisy
evidence individually, meaningful in bulk.

| | |
|---|---|
| Landed in the right family | **82%** |
| Named the specific thing | **37%** |

Strong on piano and guitar (81% specific), percussion families, and environmental
sound. Weak where the vocabulary runs out: **AudioSet has no `kick` class and no `tom`
class**, so those can never be named specifically no matter how good the audio
analysis is. And `Clapping` in AudioSet means *audience applause* — a produced clap
one-shot is a sharp transient, which the model reasonably hears as closer to a
gunshot.

**This measures agreement with human naming, not DSP correctness.** A systematically
wrong spectrogram that still separated snares from pads would score well here.

## Known limits

- ✅ **The spectrograms are verified against Essentia's own** (2026-08-15). They did
  not match at first — five wrong constants, none of which raised an error — and the
  fix is confirmed downstream rather than by correlation alone: tagging 640 files
  through our mel and through Essentia's now produces **identical results in all 16
  categories**. See [`docs/MEL_VALIDATION.md`](docs/MEL_VALIDATION.md) for the bugs,
  the traps, and the tooling. Validation needs Linux (Essentia has no Windows build);
  ordinary use does not.
- Heads trained on full music tracks are unreliable on one-shots. The bridge suppresses
  those verdicts below ~2 s rather than reporting confident nonsense.
- Preset preview audio describes a *performance* as well as a sound.

Apache-2.0.
