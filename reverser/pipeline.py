"""End-to-end pipeline: video/audio file -> song.json + media for the web player."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import pickle
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

from . import arrange, audio, export, fingering, theory

log = logging.getLogger("reverser")
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:60] or "song"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _has_video(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_name,disposition",
         "-of", "json", str(path)], capture_output=True, text=True).stdout
    streams = json.loads(out or "{}").get("streams", [])
    return any(s.get("codec_name") not in ("png", "mjpeg") for s in streams)


@contextlib.contextmanager
def _quiet():
    """Basic Pitch / CoreML print debug lines on stdout; keep the CLI readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


class Step:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        print(f"→ {self.name}…", flush=True)
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            print(f"  ✓ {self.name} ({time.time() - self.t:.1f}s)", flush=True)


def run(src: Path, out_root: Path, work_root: Path, *, title: str | None = None, slug: str | None = None,
        use_video: bool = True, capo: int | None = None, demucs_model: str = "htdemucs_6s",
        force: bool = False) -> Path:
    src = Path(src).expanduser().resolve()
    slug = slug or slugify(title or src.stem)
    work = work_root / slug
    out = out_root / slug
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    is_video = _has_video(src)

    # ------------------------------------------------------------- media
    with Step("Extracting audio"):
        mix = work / "mix.wav"
        if force or not mix.exists():
            _run(["ffmpeg", "-y", "-i", str(src), "-map", "0:a:0", "-ac", "2", "-ar", "44100", str(mix)])
    if is_video:
        with Step("Transcoding video for the web player"):
            vid = out / "video.mp4"
            if force or not vid.exists():
                _run(["ffmpeg", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0", "-vf",
                      "scale=-2:'min(720,ih)'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                      "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(vid)])
    with Step(f"Separating stems (Demucs {demucs_model})"):
        stem_dir = work / "demucs" / demucs_model / "mix"
        if force or not (stem_dir / "vocals.wav").exists():
            subprocess.run([sys.executable, "-m", "demucs", "-n", demucs_model, "-d", _torch_device(), "--float32",
                            "-o", str(work / "demucs"), str(mix)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        guitar_wav = audio.build_guitar_signal(stem_dir, work / "guitar_sum.wav")
        _encode_stems(stem_dir, guitar_wav, mix, out, force)

    # -------------------------------------------------------------- audio
    cache = work / "analysis.pkl"
    if cache.exists() and not force:
        A = pickle.load(open(cache, "rb"))
        print("→ Audio analysis (cached)")
    else:
        A = {}
        with Step("Estimating tuning"):
            y, sr = audio.load_mono(guitar_wav)
            A["offset"] = audio.estimate_tuning(y, sr)
            yt, A["scale"] = audio.retune(y, sr, A["offset"])
            import soundfile as sf

            sf.write(work / "guitar_tuned.wav", yt, sr)
            v, vsr = audio.load_mono(stem_dir / "vocals.wav")
            vt, _ = audio.retune(v, vsr, A["offset"])
            sf.write(work / "vocals_tuned.wav", vt, vsr)
        with Step("Transcribing guitar notes (Basic Pitch)"), _quiet():
            A["raw"], A["post"], A["times"] = audio.transcribe(work / "guitar_tuned.wav", A["scale"])
            A["vocal_notes"], _, _ = audio.transcribe(work / "vocals_tuned.wav", A["scale"], onset=0.5, frame=0.3)
        A["et"], A["en"] = audio.guitar_energy(y, sr)
        A["span"] = audio.find_music_span(A["et"], A["en"], A["times"], A["post"])
        with Step("Tracking beats"):
            A["grid"] = audio.track_beats(y, sr, *A["span"])
        pickle.dump(A, open(cache, "wb"))

    notes, clean_stats = audio.clean_notes(A["raw"], A["vocal_notes"])
    s0, s1 = A["span"]
    notes = [n for n in notes if s0 - 1.0 <= n.start <= s1 + 1.0]
    with Step("Recognising chords"):
        chords, (tonic, mode, key_r), grid = audio.analyse_harmony(
            A["post"], A["times"], A["grid"], A["et"], A["en"], A["span"], notes)

    # -------------------------------------------------------------- video
    prior_fn, vel_t, vel_v, overlay, vision_stats = None, None, None, [], {}
    if is_video and use_video:
        from . import vision

        vcache = work / "vision.pkl"
        if vcache.exists() and not force:
            shots, frames = pickle.load(open(vcache, "rb"))
            print("→ Video analysis (cached)")
        else:
            model_path = _hand_model()
            with Step("Finding the guitar in the video (OWLv2)"):
                shots = vision.detect_shots(src, step=1.0)
            with Step("Tracking hands (MediaPipe)"):
                frames = vision.track_hands(src, shots, model_path, fps=10.0)
            pickle.dump((shots, frames), open(vcache, "wb"))
        vision.analyse_geometry(frames, shots)
        vision.smooth_positions(frames)
        prior_fn = vision.position_prior(frames)
        vel_t, vel_v = vision.picking_velocity(frames)
        overlay = vision.export_overlay(frames)
        vis = [s for s in shots if s.guitar_visible]
        vision_stats = {
            "guitarVisibleSeconds": len(vis),
            "handFrames": sum(1 for f in frames if f.hands),
            "positionFrames": sum(1 for f in frames if f.fret_pos is not None),
            "guitarTimeline": [round(s.t, 1) for s in vis],
        }

    # ------------------------------------------------------------ fusion
    with Step("Optimising fingering (Viterbi)"):
        events, capo, capo_scores, dropped = fingering.fit_tab(notes, chords, capo=capo, prior_fn=prior_fn)
        clean_stats["dropped_by_fingering"] = dropped
        clean_stats["final"] = sum(len(e.notes) for e in events)
        arrange.quantise(events, grid)
        bf, tf = arrange.beat_function(grid), arrange.time_function(grid)
        fingering.assign_picking(events, bf, vel_t, vel_v)

    with Step("Arranging (bars, patterns, sections, easy tab)"):
        first_bar = min(e.bar for e in events) if events else 0
        last_bar = max(e.bar for e in events) if events else 0
        n_bars = last_bar + 1
        rows = arrange.bar_chords(chords, grid, max(n_bars, 0), tf)
        patterns = arrange.bar_patterns(events, grid)
        main_pattern = arrange.common_pattern([p for p in patterns if first_bar <= p["bar"] <= last_bar])
        sections = arrange.find_sections(rows, max(0, first_bar), last_bar)
        easy = arrange.easy_events(events, chords, grid, capo)
        cap_suggest, cap_rank = theory.suggest_capo(
            [(c.root, c.quality, c.end - c.start) for c in chords if c.root is not None])
        sheet = arrange.chord_sheet_events(chords, grid, capo, tf, bf,
                                           main_pattern["cells"] if main_pattern["style"] == "strum" else [])

    # ------------------------------------------------------------ export
    with Step("Exporting"):
        chord_out = []
        for c in chords:
            d = c.to_dict()
            if c.root is not None:
                vs = fingering._chord_voicings(c, capo)
                v = next((x for x in vs if x.open_shape), vs[0] if vs else None)
                d["voicing"] = v.to_dict() if v else None
                d["shape"] = theory.chord_name((c.root - capo) % 12, c.quality) if capo else c.name
                d["simple"] = theory.chord_name(c.root, theory.SIMPLIFY.get(c.quality, c.quality))
            chord_out.append(d)
        song = {
            "version": 1,
            "meta": {
                "title": title or src.stem,
                "slug": slug,
                "source": src.name,
                "duration": round(float(A["times"][-1]), 2),
                "tuning": theory.STANDARD_TUNING,
                "tuningName": "Standard (E A D G B E)",
                "tuningOffsetCents": round(A["offset"] * 100, 1),
                "capo": capo,
                "capoScores": [{"capo": c, "cost": round(s, 3)} for s, c in capo_scores],
                "easyCapo": cap_suggest,
                "key": f"{theory.ROOT_NAMES[tonic]} {mode}",
                "keyConfidence": round(key_r, 3),
                "tempo": round(grid.tempo, 1),
                "meter": grid.meter,
                "musicStart": round(s0, 2),
                "musicEnd": round(s1, 2),
                "style": main_pattern["style"],
                "hasVideo": is_video,
                "noteStats": clean_stats,
                "vision": vision_stats,
                "pipeline": ["demucs:" + demucs_model, "basic-pitch", "librosa-beats", "chord-hmm",
                             "owlv2" if overlay else None, "mediapipe-hands" if overlay else None, "viterbi-fingering"],
            },
            "media": {
                "video": "video.mp4" if is_video else None,
                "mix": "mix.m4a",
                "guitar": "guitar.m4a",
                "backing": "backing.m4a",
            },
            "beats": [round(float(b), 4) for b in grid.beats],
            "downbeatOffset": grid.downbeat_offset,
            "chords": chord_out,
            "barChords": rows,
            "sections": sections,
            "pattern": main_pattern,
            "barPatterns": patterns,
            "events": [e.to_dict() for e in events],
            "easyEvents": [e.to_dict() for e in easy],
            "chordSheet": sheet,
            "hands": overlay,
        }
        (out / "song.json").write_text(json.dumps(song, separators=(",", ":")))
        (out / "tab.txt").write_text(
            f"{song['meta']['title']}\nKey {song['meta']['key']} · {song['meta']['tempo']} BPM · capo {capo}\n\n"
            + export.ascii_tab(events, grid.meter, rows, max(0, first_bar), last_bar, grid.downbeat_offset))
        export.write_midi(events, out / "transcription.mid", grid.tempo)
        _update_index(out_root, song)
    print(f"\nDone → {out}")
    print(f"  key {song['meta']['key']}, {song['meta']['tempo']} BPM, {grid.meter}/4, capo {capo}, "
          f"{len(events)} events, {len([c for c in chords if c.root is not None])} chord changes")
    return out


def _torch_device() -> str:
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return "cpu"


def _hand_model() -> Path:
    p = MODELS / "hand_landmarker.task"
    if not p.exists():
        MODELS.mkdir(exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(HAND_MODEL_URL, p)
    return p


def _encode_stems(stem_dir: Path, guitar_wav: Path, mix: Path, out: Path, force: bool) -> None:
    import soundfile as sf

    def enc(wav, name):
        dst = out / name
        if force or not dst.exists():
            _run(["ffmpeg", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "160k", str(dst)])

    enc(mix, "mix.m4a")
    enc(guitar_wav, "guitar.m4a")
    backing = stem_dir.parent / "backing.wav"
    if force or not backing.exists():
        acc = None
        for name in ("vocals", "drums", "piano"):
            p = stem_dir / f"{name}.wav"
            if p.exists():
                x, sr = sf.read(str(p), always_2d=True)
                acc = x if acc is None else acc[: len(x)] + x[: len(acc)]
        sf.write(str(backing), acc.astype(np.float32), sr)
    enc(backing, "backing.m4a")


def _update_index(out_root: Path, song: dict) -> None:
    idx_path = out_root / "index.json"
    try:
        idx = json.loads(idx_path.read_text())
    except Exception:
        idx = {"songs": []}
    idx["songs"] = [s for s in idx["songs"] if s["slug"] != song["meta"]["slug"]]
    m = song["meta"]
    idx["songs"].append({"slug": m["slug"], "title": m["title"], "key": m["key"], "tempo": m["tempo"],
                         "capo": m["capo"], "duration": m["duration"]})
    idx_path.write_text(json.dumps(idx, indent=2))
