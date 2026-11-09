"""GuitarSet benchmark: notes, tab (string/fret), chords, beats, tempo and key.

GuitarSet (Xi et al., ISMIR 2018, CC BY 4.0) has 360 excerpts of solo acoustic guitar
(6 players x 5 styles x 3 progressions x 2 tempi x comp/solo), recorded with a hexaphonic
pickup, so every note has its true string and fret.

Stage 1 (audio models: tuning, Basic Pitch, beats) is cached per track in <root>/cache.
Stage 2 (the music logic this project adds) is pure Python and runs in parallel, so
ablations re-run in a minute.

Metrics
- notes: onset F1 (50 ms, exact semitone, offsets ignored), mir_eval.
- tab: a note counts only if onset, pitch AND string are right (fret follows from pitch+string).
  "string accuracy" = share of correctly transcribed notes that also got the right string.
- oracle tab: the fingering stage alone, fed the true notes (isolates it from transcription).
- chords: mir_eval root / majmin / sevenths, duration-weighted, against the *performed* chords.
- beats / downbeats: mir_eval F-measure (70 ms). Tempo: within 4 %, or an octave off.
- key: mir_eval weighted score (fifths/relative/parallel get partial credit).
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio, fingering, theory
from .audio import ChordSeg, Note

NAME_RE = re.compile(r"^(\d\d)_([A-Za-z]+)(\d)-(\d+)-([A-G][b#]?)_(comp|solo)$")
HARTE_Q = {"": "maj", "m": "min", "7": "7", "maj7": "maj7", "m7": "min7", "m7b5": "hdim7"}
STYLE_NAMES = {"BN": "Bossa nova", "Funk": "Funk", "Jazz": "Jazz", "Rock": "Rock", "SS": "Singer-songwriter"}


# ------------------------------------------------------------------ reference


@dataclass
class Reference:
    name: str
    player: str
    style: str
    kind: str  # comp | solo
    tempo: float
    key: str  # "Eb major"
    notes: list[tuple[float, float, int, int]]  # start, end, midi, string (0 = low E)
    chord_intervals: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    chord_labels: list[str] = field(default_factory=list)
    beats: np.ndarray = field(default_factory=lambda: np.zeros(0))
    downbeats: np.ndarray = field(default_factory=lambda: np.zeros(0))


def load_reference(jams_path: Path) -> Reference:
    name = jams_path.stem
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"unexpected GuitarSet name {name}")
    player, style, _prog, tempo, _key, kind = m.groups()
    J = json.loads(jams_path.read_text())
    notes, chord_anns, beats, downbeats, key = [], [], [], [], None
    for a in J["annotations"]:
        ns = a["namespace"]
        data = a["data"]
        if ns == "note_midi":
            s = int(a["annotation_metadata"]["data_source"])
            for d in data:
                notes.append((float(d["time"]), float(d["time"] + d["duration"]), int(round(d["value"])), s))
        elif ns == "chord":
            chord_anns.append(data)
        elif ns == "beat_position":
            for d in data:
                beats.append(float(d["time"]))
                if int(d["value"]["position"]) == 1:
                    downbeats.append(float(d["time"]))
        elif ns == "key_mode" and data:
            key = str(data[0]["value"]).replace(":", " ")
    # GuitarSet stores two chord annotations: the lead sheet (instructed) first, then the
    # chords inferred from what was actually played. Score against what was played.
    chords = chord_anns[-1] if chord_anns else []
    iv = np.array([[d["time"], d["time"] + d["duration"]] for d in chords]) if chords else np.zeros((0, 2))
    labels = [str(d["value"]) for d in chords]
    notes.sort()
    return Reference(name, player, style, kind, float(tempo), key or "C major", notes, iv, labels,
                     np.array(sorted(beats)), np.array(sorted(downbeats)))


# -------------------------------------------------------------------- stage 1


def analyse_audio(wav: Path, cache: Path) -> dict:
    """Audio models, cached. Mirrors pipeline.run (no stem separation: GuitarSet is solo guitar)."""
    if cache.exists():
        A = pickle.loads(cache.read_bytes())
        if A.get("tv") == audio.TRANSCRIBE_VERSION:
            return A
    import contextlib
    import io

    import soundfile as sf

    y, sr = audio.load_mono(wav)
    offset = audio.estimate_tuning(y, sr)
    yt, scale = audio.retune(y, sr, offset)
    src = wav
    if scale != 1.0:
        src = cache.with_suffix(".tuned.wav")
        sf.write(str(src), yt, sr)
    with contextlib.redirect_stdout(io.StringIO()):
        raw, post, times = audio.transcribe(src, scale)
    if src != wav:
        src.unlink()
    et, en = audio.guitar_energy(y, sr)
    dur = len(y) / sr
    A = {
        "tv": audio.TRANSCRIBE_VERSION,
        "offset": offset,
        "raw": [(n.start, n.end, n.pitch, n.amp) for n in raw],
        "post": post.astype(np.float16),
        "times": times,
        "et": et,
        "en": en,
        "grid": audio.track_beats(y, sr, 0.0, dur),
        "duration": dur,
    }
    cache.write_bytes(pickle.dumps(A))
    return A


# -------------------------------------------------------------------- stage 2


def _intervals(items):
    if not items:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[s, max(e, s + 0.02)] for s, e, *_ in items], dtype=float)
    hz = 440.0 * 2.0 ** ((np.array([p for _, _, p, *_ in items], dtype=float) - 69) / 12)
    return iv, hz


def _match(ref, est):
    """mir_eval note matching on onset (50 ms) and pitch; offsets ignored."""
    import mir_eval

    if not ref or not est:
        return []
    ri, rp = _intervals(ref)
    ei, ep = _intervals(est)
    return mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)


def _harte(c: ChordSeg) -> str:
    return "N" if c.root is None else f"{theory.ROOT_NAMES[c.root]}:{HARTE_Q.get(c.quality, 'maj')}"


def _ref_chordsegs(ref: Reference) -> list[ChordSeg]:
    """Performed chords as ChordSegs (for the 'oracle chords' ablation)."""
    out = []
    inv = {v: k for k, v in HARTE_Q.items()}
    for (s, e), lab in zip(ref.chord_intervals, ref.chord_labels):
        if lab == "N" or ":" not in lab:
            out.append(ChordSeg(s, e, "N", None, "", 1.0))
            continue
        root, q = lab.split(":", 1)
        q = q.split("/")[0]
        q = inv.get(q, "m" if q.startswith("min") else ("7" if q.startswith("7") else ""))
        r = _pitch_class(root)
        out.append(ChordSeg(s, e, theory.chord_name(r, q), r, q, 1.0))
    return out


def _pitch_class(name: str) -> int:
    return ({"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[name[0]] + name.count("#") - name.count("b")) % 12


def _lowest_fret(pitch, tuning=theory.STANDARD_TUNING):
    """Classic baseline: every note on the string where it needs the lowest fret."""
    opts = [(pitch - t, s) for s, t in enumerate(tuning) if 0 <= pitch - t <= 19]
    return min(opts)[1] if opts else -1


def _oracle_tab(ref: Reference, chords: list[ChordSeg]) -> tuple[int, int]:
    """Fingering stage alone on the true notes. Returns (right string, total)."""
    notes, truth = [], {}
    for s, e, p, string in ref.notes:
        n = Note(s, e, p, 0.7)
        truth[id(n)] = string
        notes.append(n)
    events = fingering.group_events(notes)
    fingering.assign_fingering(events, chords, capo=0)
    right = sum(1 for ev in events for n in ev.notes if truth.get(id(n)) == n.string)
    return right, len(ref.notes)


def evaluate_track(args) -> dict:
    jams_path, cache_path = args
    import mir_eval

    ref = load_reference(Path(jams_path))
    A = pickle.loads(Path(cache_path).read_bytes())
    post = A["post"].astype(np.float32)
    r = {"name": ref.name, "player": ref.player, "style": ref.style, "kind": ref.kind,
         "duration": A["duration"], "n_ref": len(ref.notes)}

    # notes: raw Basic Pitch, then after the guitar-aware cleanup
    raw = [Note(*t) for t in A["raw"]]
    r["n_raw"], r["tp_raw"] = len(raw), len(_match(ref.notes, [(n.start, n.end, n.pitch) for n in raw]))
    notes, _ = audio.clean_notes([Note(*t) for t in A["raw"]], None)
    span = (0.0, A["duration"])

    chords, (tonic, mode, _), grid = audio.analyse_harmony(post, A["times"], A["grid"], A["et"], A["en"], span, notes)

    # end-to-end tab, exactly as the app does it (auto capo)
    events, capo, _, dropped = fingering.fit_tab(notes, chords)
    est = [(n.start, n.end, n.pitch, n.string) for ev in events for n in ev.notes]
    m = _match(ref.notes, est)
    r["capo"] = capo
    r["n_est"] = len(est)
    r["tp_notes"] = len(m)
    r["tp_tab"] = sum(1 for i, j in m if ref.notes[i][3] == est[j][3])
    r["tp_tab_lowest"] = sum(1 for i, j in m if ref.notes[i][3] == _lowest_fret(est[j][2]))
    r["dropped"] = dropped

    # fingering stage alone (true notes), with ablations
    r["oracle_full"], r["oracle_n"] = _oracle_tab(ref, chords)
    r["oracle_nochords"], _ = _oracle_tab(ref, [])
    r["oracle_refchords"], _ = _oracle_tab(ref, _ref_chordsegs(ref))
    r["oracle_lowest"] = sum(1 for _, _, p, s in ref.notes if _lowest_fret(p) == s)
    frets = [p - theory.STANDARD_TUNING[s] for _, _, p, s in ref.notes]
    r["ref_fret_hist"] = np.bincount(np.clip(frets, 0, 20), minlength=21).tolist()

    # chords
    if len(ref.chord_labels):
        est_iv = np.array([[c.start, c.end] for c in chords]) if chords else np.array([[0.0, A["duration"]]])
        est_lab = [_harte(c) for c in chords] if chords else ["N"]
        sc = mir_eval.chord.evaluate(ref.chord_intervals, ref.chord_labels, est_iv, est_lab)
        r["chord_dur"] = float(np.sum(np.diff(ref.chord_intervals, axis=1)))
        for k in ("root", "majmin", "sevenths"):
            r[f"chord_{k}"] = float(sc[k])

    # rhythm
    est_beats = np.asarray(grid.beats, float)
    r["beat_f"] = float(mir_eval.beat.f_measure(ref.beats, est_beats)) if len(ref.beats) and len(est_beats) else 0.0
    est_db = grid.bar_starts()
    r["downbeat_f"] = float(mir_eval.beat.f_measure(ref.downbeats, est_db)) if len(ref.downbeats) and len(est_db) else 0.0
    ratio = grid.tempo / ref.tempo
    r["tempo"] = grid.tempo
    r["tempo_acc1"] = abs(ratio - 1) < 0.04
    r["tempo_acc2"] = any(abs(ratio / k - 1) < 0.04 for k in (1, 2, 0.5, 3, 1 / 3))
    r["meter"] = grid.meter
    r["key_est"] = f"{theory.ROOT_NAMES[tonic]} {mode}"
    r["key_ref"] = ref.key
    r["key_score"] = float(mir_eval.key.weighted_score(ref.key, r["key_est"]))
    return r


# ---------------------------------------------------------------------- report


def _f1(tp, n_est, n_ref):
    p = tp / n_est if n_est else 0.0
    rc = tp / n_ref if n_ref else 0.0
    return p, rc, (2 * p * rc / (p + rc) if p + rc else 0.0)


def summarise(rows: list[dict]) -> dict:
    S = lambda k, rr=rows: sum(r[k] for r in rr)  # noqa: E731
    out = {"tracks": len(rows), "minutes": round(S("duration") / 60, 1)}
    out["notes_raw"] = _f1(S("tp_raw"), S("n_raw"), S("n_ref"))
    out["notes"] = _f1(S("tp_notes"), S("n_est"), S("n_ref"))
    out["tab"] = _f1(S("tp_tab"), S("n_est"), S("n_ref"))
    out["string_acc"] = S("tp_tab") / max(1, S("tp_notes"))
    out["string_acc_lowest"] = S("tp_tab_lowest") / max(1, S("tp_notes"))
    n = max(1, S("oracle_n"))
    out["oracle"] = {k: S(f"oracle_{k}") / n for k in ("lowest", "nochords", "full", "refchords")}
    with_ch = [r for r in rows if "chord_dur" in r]
    for kind in ("comp", "solo", "all"):
        rr = [r for r in with_ch if kind == "all" or r["kind"] == kind]
        w = sum(r["chord_dur"] for r in rr) or 1.0
        out[f"chords_{kind}"] = {k: sum(r[f"chord_{k}"] * r["chord_dur"] for r in rr) / w
                                 for k in ("root", "majmin", "sevenths")}
    out["beat_f"] = float(np.mean([r["beat_f"] for r in rows]))
    out["downbeat_f"] = float(np.mean([r["downbeat_f"] for r in rows]))
    out["tempo_acc1"] = float(np.mean([r["tempo_acc1"] for r in rows]))
    out["tempo_acc2"] = float(np.mean([r["tempo_acc2"] for r in rows]))
    out["key"] = float(np.mean([r["key_score"] for r in rows]))
    out["capo0"] = float(np.mean([r["capo"] == 0 for r in rows]))
    hist = np.sum([r["ref_fret_hist"] for r in rows], axis=0)
    out["ref_frets_0_4"] = float(hist[:5].sum() / max(1, hist.sum()))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["style"], r["kind"])].append(r)
    out["by_group"] = {
        f"{STYLE_NAMES.get(s, s)} {k}": {
            "tab_f1": _f1(S("tp_tab", g), S("n_est", g), S("n_ref", g))[2],
            "string_acc": S("tp_tab", g) / max(1, S("tp_notes", g)),
            "oracle": S("oracle_full", g) / max(1, S("oracle_n", g)),
            "oracle_lowest": S("oracle_lowest", g) / max(1, S("oracle_n", g)),
        }
        for (s, k), g in sorted(groups.items())
    }
    return out


def format_report(s: dict) -> str:
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    prf = lambda t: f"P {pct(t[0])} · R {pct(t[1])} · **F1 {pct(t[2])}**"  # noqa: E731
    L = [f"GuitarSet: {s['tracks']} excerpts, {s['minutes']} min of solo acoustic guitar (mono mic)\n",
         "| Stage | Result |", "|---|---|",
         f"| Notes, raw Basic Pitch | {prf(s['notes_raw'])} |",
         f"| Notes, after cleanup + fingering | {prf(s['notes'])} |",
         f"| **Tab** (onset + pitch + string) | {prf(s['tab'])} |",
         f"| String accuracy on correct notes | **{pct(s['string_acc'])}** (lowest-fret baseline {pct(s['string_acc_lowest'])}) |",
         f"| Chords, comp (root / maj-min / 7ths) | {pct(s['chords_comp']['root'])} / **{pct(s['chords_comp']['majmin'])}** / {pct(s['chords_comp']['sevenths'])} |",
         f"| Chords, solo (root / maj-min / 7ths) | {pct(s['chords_solo']['root'])} / {pct(s['chords_solo']['majmin'])} / {pct(s['chords_solo']['sevenths'])} |",
         f"| Beat F / downbeat F | {pct(s['beat_f'])} / {pct(s['downbeat_f'])} |",
         f"| Tempo within 4 % (octave errors allowed) | {pct(s['tempo_acc1'])} ({pct(s['tempo_acc2'])}) |",
         f"| Key (mir_eval weighted) | {pct(s['key'])} |",
         f"| Capo 0 chosen (truth: no capo) | {pct(s['capo0'])} |",
         "", "Fingering stage alone, fed the true notes (string accuracy):", "",
         "| Method | String accuracy |", "|---|---|",
         f"| Lowest fret (classic baseline) | {pct(s['oracle']['lowest'])} |",
         f"| Viterbi, no chord shapes | {pct(s['oracle']['nochords'])} |",
         f"| Viterbi + detected chords (shipped) | **{pct(s['oracle']['full'])}** |",
         f"| Viterbi + true chords (upper bound of the chord prior) | {pct(s['oracle']['refchords'])} |",
         "", f"{pct(s['ref_frets_0_4'])} % of the true notes are on frets 0–4.", "",
         "| Style | Tab F1 | String acc. | Oracle fingering | Lowest fret |", "|---|---|---|---|---|"]
    for g, v in s["by_group"].items():
        L.append(f"| {g} | {pct(v['tab_f1'])} | {pct(v['string_acc'])} | {pct(v['oracle'])} | {pct(v['oracle_lowest'])} |")
    return "\n".join(L)


def run(root: Path, limit: int | None = None, jobs: int | None = None, players: str | None = None) -> dict:
    root = Path(root)
    ann = sorted((root / "annotation").glob("*.jams"))
    if players:
        ann = [p for p in ann if p.stem[:2] in players.split(",")]
    if limit:
        # spread the subset over players/styles instead of taking the first files
        ann = ann[:: max(1, len(ann) // limit)][:limit]
    if not ann:
        raise SystemExit(f"no .jams files in {root / 'annotation'} (see README: Benchmark)")
    cache_dir = root / "cache"
    cache_dir.mkdir(exist_ok=True)
    t0 = time.time()
    def fresh(p):
        c = cache_dir / f"{p.stem}.pkl"
        return c.exists() and pickle.loads(c.read_bytes()).get("tv") == audio.TRANSCRIBE_VERSION

    todo = [p for p in ann if not fresh(p)]
    for i, p in enumerate(todo):
        wav = root / "audio" / f"{p.stem}_mic.wav"
        analyse_audio(wav, cache_dir / f"{p.stem}.pkl")
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            print(f"  audio models {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    work = [(str(p), str(cache_dir / f"{p.stem}.pkl")) for p in ann]
    with ProcessPoolExecutor(jobs) as ex:
        rows = list(ex.map(evaluate_track, work, chunksize=2))
    s = summarise(rows)
    s["seconds"] = round(time.time() - t0, 1)
    (root / "results.json").write_text(json.dumps({"summary": s, "tracks": rows}, indent=1, default=float))
    return s
