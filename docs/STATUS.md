# Where this stands, and what is open

**Purpose of this file:** the other docs record *findings* — what was learned, what was
fixed, what not to re-chase. None of them recorded **how a session ended**, so the next
session had to reconstruct it from git log and a database. That is the gap this file
exists to close. **Update it at the end of a session, even when everything is green** —
"nothing is blocked" is itself the thing worth writing down.

Last updated: **2026-08-15**.

---

## State

Both repos clean, everything committed and pushed. Nothing is blocked.

| | |
|---|---|
| `ai-bridge-for-ableton-live` | 60 tools, **103 tests passing** |
| `ai-bridge-listener` | index built: **22,100 files / 2.88 M tags**, analyzer `mel2+discogs-effnet-1+25heads-1+yamnet-1(k15,f0.02)` |
| spectrograms | **verified against Essentia's own** — see `MEL_VALIDATION.md` |
| identification quality | **hit@1 57.2%**, hit@3 ~85% (after demoting AudioSet's interior nodes) |

The whole library has already been re-scanned on the corrected mel. `live_sidecar_status`
and `live_find_sound` work end to end through MCP.

## The one open thread

**A Gemini review of the sidecar architecture.** The brief was written but never sent:
it needs Kim signed in to the **built-in** browser (it was signed out, and therefore
pinned to Flash-Lite). Deliberately deferred to a fresh session, being a
reasoning-heavy conversation.

Two decisions that go with it, already reasoned through and still standing:

* **Start a NEW Gemini chat**, not the existing MidiGen one — different architecture and
  vocabulary, and Gemini's cross-chat memory bleeding sidecar facts into MidiGen
  discussions is a known failure mode.
* **Do not use the external Chrome.** Kim was explicit. The built-in browser only.

## Known, deliberately unfixed

* **22 files fail to decode** and are recorded with an `error`. Every one is a macOS
  AppleDouble stub (`._Name.wav`) inside Akai MPC content — metadata forks, not audio.
  They never surface in results (`f.error IS NULL` filters them), so this is cosmetic;
  the scanner could skip `._*` outright.
* **AudioSet has no `kick` and no `tom` class**, so those can never be named
  specifically however good the analysis gets. Not a bug, a vocabulary ceiling.
* **`Clapping` in AudioSet means audience applause.** A produced clap one-shot is a
  sharp transient, which the model reasonably hears as nearer a gunshot.

## Changed on 2026-08-15 (later session)

`live_find_sound`'s `query` **now searches what a file was heard as, not only what it is
called.** It was a bare `f.path LIKE`, which made the tool's most-reachable parameter a
filename grep wearing a listening result's clothes — `query="cymbal"` returned files
tagged `Steam` and `Tools`, ranked `relevance: 0.0`, with nothing saying the audio had
never been consulted.

Now: matched **word by word** across both tag labels and the path, scored with the tag's
own confidence plus `_NAME_WEIGHT` for a name hit, and every result carries
**`matched_by`** (`heard` / `name` / `heard+name`) so the two can never be confused
again. The filename half is kept deliberately — in a library named by content the name
is real evidence, and without the sidecar it is the only seed
`live_similar_sounds` can start from.

Measured on the real index: `query="cymbal"` went from MPC demo files tagged `Steam` at
0.0, to actual crash cymbals tagged `Cymbal` at 1.14. `query="vinyl crackle"` now puts
the file tagged `Crackle` on top — a phrase match never found it, because no single
label reads "vinyl crackle".

⚠️ **A running MCP server keeps the old code.** The change is only visible after the
bridge's host process is restarted; a live `live_find_sound` call will otherwise still
return `relevance: 0.0` and no `matched_by`, which looks exactly like the bug.

**Scores are comparable only WITHIN one query.** They are averaged over the words, so
adding a word that matches little drags the average down — `pad ambient` scores below
`ambient` on the same file. Ranking only ever compares files inside one query, where the
ordering is the wanted one.
