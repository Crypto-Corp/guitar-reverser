# Guitar Reverser

Turn a guitar **video or song** into a playable, beginner-friendly tab. You get the chords, string/fret for every note, fretting fingers, picking fingers (p/i/m/a) or strum directions. It all plays back in a **3D fretboard player** synced to the original video.

```
video.mp4 ─┬─ audio ── Demucs (stems) ── Basic Pitch (notes) ── beats/key/chords (HMM)
           │                                                   │
           └─ picture ── OWLv2 (guitar, neck, headstock) ── MediaPipe hands ── fret-position prior
                                                               │
                         Viterbi fingering optimiser  ◄────────┘
                         (playability + chord shapes + video + ringing notes)
                                       │
         song.json · tab.txt · MIDI ── web player (three.js 3D fretboard, tab, chords, stems)
```

## Why it's more than another audio-to-tab tool

| Problem | What existing tools do | What this does |
|---|---|---|
| Same note can be played in up to 5 places | guess from audio | **reads the fretting hand in the video** (MediaPipe + fret-spacing geometry) and uses it as a soft prior |
| Voice, other instruments | transcribe everything | **Demucs 6-stem separation**, then drops notes that copy the sung melody (vocal-bleed filter) |
| Ghost notes / harmonics | kept | octave-ghost filter, plus a **robust decoder** that can drop a quiet note when keeping it would force an implausible hand jump |
| Fingering | note by note | **global Viterbi** over hand states: stretch, hand shifts (worse when fast), notes cut while ringing, **open-chord shapes** (the classic 16-chord beginner chart is built in) |
| Picking hand | ignored | **p/i/m/a** for fingerstyle, **↓/↑** strums from string onset order, wrist motion in the video, or the "down on the beat" rule |
| Beginners | one dense tab | three levels: **Chords** (shapes + strum pattern), **Easy tab** (bass + melody on chord shapes), **Full tab** |
| Tuning off A440 | split notes | estimates the offset and cancels it by resampling before transcription |

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
cd web && npm install
```

Models download on first use: Demucs `htdemucs_6s`, OWLv2 base (Hugging Face) and the MediaPipe hand landmarker (into `models/`).

## Use

```bash
.venv/bin/reverser analyze ~/Downloads/song.mp4 --title "My song"
```

```bash
npm --prefix web run dev
```

Then open http://localhost:5173.

Options: `--no-video` (audio only), `--capo N` (force capo, default auto), `--force` (ignore caches). Audio files (`.mp3`, `.wav`, …) work too. You just don't get the video prior.

Outputs go to `web/public/songs/<slug>/`: `song.json` (everything the player needs), `tab.txt` (ASCII tab), `transcription.mid`, a 720p web video and two stems (guitar / everything else). Intermediate files are cached in `work/<slug>/`, so re-runs take seconds.

## Player

- **3D fretboard**: notes fall onto the exact string and fret, coloured by fretting finger (1 index, 2 middle, 3 ring, 4 pinky, open). Ringing strings glow and vibrate, the next chord shape falls in, and strum arrows sweep across. Views: player (as you see your guitar), teacher (mirror), top. You can drag to orbit.
- **Tab lane**: scrolls under a fixed playhead, with chord names, bar numbers, sustain bars and p/i/m/a or ↓/↑ under each note. Click or drag to seek, Ctrl+wheel to zoom.
- **Practice**: speed 40–100 % (pitch preserved), loop the current section, count-in, metronome on the detected beats. Separate volume for the **guitar** and **voice & rest** stems, so you can mute the guitar and play along. **Tab synth** plays the transcription itself, so you can hear what was transcribed.
- **Side panel**: video with the hand-tracking overlay (fretting hand yellow, picking hand magenta, estimated fret), now → next chord diagrams with a countdown, the most common picking/strum pattern, and every chord of the song (click to jump).
- Keys: `space` play · `←/→` ±5 s · `L` loop · `1/2/3` modes · `-/+` speed.

## Accuracy: what to expect

- Works best on **one guitar, clearly audible**, like lessons, covers or open mics. Dense band mixes with distorted guitars are much harder.
- On the bundled test video (one acoustic guitar plus a singer), the chords come out as a coherent C-major F → Am → C → Em / G progression. 98 % of notes land on frets 0–4, which matches the hand position the camera sees, and only 6 notes were dropped by the decoder.
- The video prior only exists while the guitar is on screen (green marks on the timeline). Elsewhere fingering comes from audio and playability alone.
- Rhythm uses librosa's beat tracker. Syncopated ballads can make the downbeat ambiguous.
- Not measured yet against human tabs. The next step is a benchmark on GuitarSet (see Roadmap).

## Project layout

```
reverser/         Python analysis package
  audio.py        stems → tuning → Basic Pitch → cleanup → beats → chords (HMM) → key
  vision.py       OWLv2 guitar parts → MediaPipe hands → fret position → picking-hand motion
  fingering.py    events → Viterbi string/fret/finger → strum vs pluck, p/i/m/a, ↓/↑
  arrange.py      quantisation, bars, patterns, sections, easy tab, chord-sheet strums
  theory.py       tuning, chord vocabulary, open/barre voicings, capo suggestion, key profiles
  pipeline.py     end-to-end orchestration + caching; cli.py = `reverser` command
web/              Vite + TypeScript + three.js player
tests/            fast unit tests (`.venv/bin/python -m pytest tests`)
```

## Roadmap

- Benchmark on **GuitarSet** (string/fret ground truth, CC BY 4.0): note F1, tab accuracy, with and without each stage.
- Swap Basic Pitch for a guitar-specific transcriber (e.g. a model fine-tuned on GuitarSet/SynthTab) when one with open weights is available.
- Read fingertips against detected strings and frets for per-note video evidence, not just hand position.
- Technique detection (hammer-on, pull-off, slide, bend, palm mute).
- Guitar Pro / MusicXML export and a mic-based "wait for me" practice mode.

## Licences of the bricks

Demucs (MIT), Basic Pitch (Apache-2.0), librosa (ISC), MediaPipe (Apache-2.0), OWLv2 / transformers (Apache-2.0), three.js (MIT). Analysed songs stay local: `web/public/songs/` and `work/` are git-ignored, because they contain the original audio and video.
