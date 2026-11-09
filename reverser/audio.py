"""Audio analysis: tuning, note transcription, beats, key and chords.

Open-source bricks used here:
- Spotify Basic Pitch (Apache-2.0) for polyphonic note transcription;
- librosa (ISC) for tuning estimation, onset envelope and beat tracking.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from . import theory

log = logging.getLogger(__name__)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@dataclass
class Note:
    start: float
    end: float
    pitch: int  # MIDI, already corrected for tuning offset
    amp: float
    # filled by later stages
    string: int = -1
    fret: int = -1
    finger: int = 0  # fretting finger 0=open, 1..4
    pluck: str = ""  # picking-hand finger p/i/m/a, or "" for strums
    event: int = -1

    def to_dict(self):
        d = asdict(self)
        d["start"] = round(self.start, 4)
        d["end"] = round(self.end, 4)
        d["amp"] = round(self.amp, 3)
        return d


# --------------------------------------------------------------------------- io


def load_mono(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    y, file_sr = sf.read(str(path), always_2d=True)
    y = y.mean(axis=1).astype(np.float32)
    if sr and sr != file_sr:
        import librosa

        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")
        file_sr = sr
    return y, file_sr


def build_guitar_signal(stem_dir: Path, out: Path) -> Path:
    """Guitar stem + bass stem (an acoustic guitar's low strings leak into 'bass')."""
    y, sr = load_mono(stem_dir / "guitar.wav")
    for extra, gain in (("bass.wav", 1.0), ("other.wav", 0.5)):
        p = stem_dir / extra
        if p.exists():
            e, _ = load_mono(p)
            y = y + gain * e[: len(y)]
    peak = float(np.max(np.abs(y))) or 1.0
    sf.write(str(out), (0.95 * y / peak).astype(np.float32), sr)
    return out


# ----------------------------------------------------------------------- tuning


def estimate_tuning(y: np.ndarray, sr: int) -> float:
    """Global tuning offset in semitones relative to A440 (range -0.5..0.5)."""
    import librosa

    return float(librosa.estimate_tuning(y=y, sr=sr, n_fft=4096, resolution=0.01, bins_per_octave=12))


def retune(y: np.ndarray, sr: int, offset_semitones: float) -> tuple[np.ndarray, float]:
    """Cancel a tuning offset by resampling (no phase-vocoder artefacts).

    Playing the audio at rate r shifts pitch by 12*log2(r) and divides times by r,
    so note times found in the returned signal must be multiplied by `time_scale`.
    """
    if abs(offset_semitones) < 0.08:
        return y, 1.0
    import librosa

    r = 2.0 ** (-offset_semitones / 12.0)
    y2 = librosa.resample(y, orig_sr=sr, target_sr=int(round(sr / r)), res_type="soxr_hq")
    return y2.astype(np.float32), r


# ------------------------------------------------------------------ transcription


def _pkg_resources_shim() -> None:
    """resampy (a Basic Pitch dependency) still imports pkg_resources, which setuptools>=81
    no longer ships. It only needs resource_filename: provide it via importlib."""
    import importlib.util
    import sys

    if "pkg_resources" in sys.modules or importlib.util.find_spec("pkg_resources") is not None:
        return
    import types
    from importlib import resources

    mod = types.ModuleType("pkg_resources")
    mod.resource_filename = lambda pkg, name: str(resources.files(pkg).joinpath(name))  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = mod


# Bump when transcription settings change: cached analyses with another version are redone.
TRANSCRIBE_VERSION = "basic-pitch:on0.55:fr0.42:min60:nomelodia"


def transcribe(audio_path: Path, time_scale: float = 1.0, *, onset=0.55, frame=0.42, min_ms=60.0, melodia=False):
    """Run Basic Pitch. Returns (notes, posteriorgram (T x 88), frame_times).

    Thresholds fitted on GuitarSet players 00-02 (note F1 on players 03-05: 72.4 -> 79.0 %;
    the old values were 0.45 / 0.28 with the Melodia trick).
    """
    _pkg_resources_shim()
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict
    from basic_pitch.note_creation import model_frames_to_time

    model = Model(ICASSP_2022_MODEL_PATH)
    output, _midi, events = predict(
        str(audio_path),
        model,
        onset_threshold=onset,
        frame_threshold=frame,
        minimum_note_length=min_ms,
        minimum_frequency=70.0,
        maximum_frequency=1400.0,
        multiple_pitch_bends=False,
        melodia_trick=melodia,
    )
    notes = [Note(s * time_scale, e * time_scale, int(p), float(a)) for s, e, p, a, *_ in events]
    notes.sort(key=lambda n: (n.start, n.pitch))
    post = np.asarray(output["note"], dtype=np.float32)
    times = np.asarray(model_frames_to_time(post.shape[0]), dtype=np.float64) * time_scale
    return notes, post, times


def clean_notes(
    notes: list[Note],
    vocal_notes: list[Note] | None = None,
    tuning: list[int] = theory.STANDARD_TUNING,
    max_fret: int = 19,
    merge_ratio: float = 0.6,
) -> tuple[list[Note], dict]:
    """Guitar-aware cleanup of raw transcription."""
    stats = {"raw": len(notes)}
    lo, hi = min(tuning), max(tuning) + max_fret
    notes = [n for n in notes if lo <= n.pitch <= hi and n.end - n.start > 0.03]
    stats["in_range"] = len(notes)

    # Merge same-pitch fragments split by a tiny gap (vibrato/decay artefacts). Only a much
    # weaker fragment is merged: a similar-strength one is a re-plucked note (GuitarSet: F1 63.7 -> 73.0).
    notes.sort(key=lambda n: (n.pitch, n.start))
    merged: list[Note] = []
    for n in notes:
        if merged and merged[-1].pitch == n.pitch and n.start - merged[-1].end < 0.03 and n.amp < merge_ratio * merged[-1].amp:
            merged[-1].end = max(merged[-1].end, n.end)
        else:
            merged.append(n)
    notes = sorted(merged, key=lambda n: (n.start, n.pitch))
    stats["merged"] = len(notes)

    # Weak harmonic ghosts: an octave/12th above a much louder simultaneous note.
    keep = []
    for n in notes:
        ghost = False
        for m in notes:
            if m is n or m.start - n.start > 0.06:
                if m.start - n.start > 0.06:
                    break
                continue
            if abs(m.start - n.start) < 0.05 and n.pitch - m.pitch in (12, 19, 24) and n.amp < 0.4 * m.amp:
                ghost = True
                break
        if not ghost:
            keep.append(n)
    notes = keep
    stats["no_ghosts"] = len(notes)

    # Vocal bleed: weak guitar-stem notes that copy the sung melody.
    if vocal_notes:
        vn = sorted(vocal_notes, key=lambda n: n.start)
        starts = np.array([v.start for v in vn])
        keep = []
        for n in notes:
            i0 = max(0, int(np.searchsorted(starts, n.start - 3.0)))
            dur = n.end - n.start
            bleed = False
            for v in vn[i0:]:
                if v.start > n.end:
                    break
                if v.pitch == n.pitch:
                    ov = min(v.end, n.end) - max(v.start, n.start)
                    if ov > 0.6 * dur and n.amp < 0.45:
                        bleed = True
                        break
            if not bleed:
                keep.append(n)
        notes = keep
    stats["no_vocal_bleed"] = len(notes)

    # Drop very quiet notes relative to the local loudness.
    if notes:
        amps = np.array([n.amp for n in notes])
        floor = max(0.12, float(np.percentile(amps, 10)) * 0.8)
        notes = [n for n in notes if n.amp >= floor]
    stats["final"] = len(notes)
    return notes, stats


# ------------------------------------------------------------------------ beats


@dataclass
class BeatGrid:
    beats: np.ndarray  # beat times (s)
    tempo: float
    meter: int  # beats per bar
    downbeat_offset: int  # index of the first downbeat in `beats`
    source: str = "librosa"  # "beat_this" when the neural tracker supplied beats + downbeats

    def bar_starts(self) -> np.ndarray:
        return self.beats[self.downbeat_offset :: self.meter]

    def to_dict(self):
        return {
            "beats": [round(float(b), 4) for b in self.beats],
            "tempo": round(self.tempo, 1),
            "meter": self.meter,
            "downbeatOffset": self.downbeat_offset,
        }


def _beat_this(y: np.ndarray, sr: int) -> BeatGrid | None:
    """Beats + downbeats from beat_this (Foscarin et al., ISMIR 2024) when it is installed.

    Optional dependency: `uv pip install --python .venv/bin/python beat-this`.
    Any failure (not installed, no weights, API change) falls back to librosa.
    """
    try:
        import torch
        from beat_this.inference import Audio2Beats
    except Exception:
        return None
    try:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            a2b = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
        except Exception:
            a2b = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        beats, downbeats = a2b(y.astype(np.float32), sr)
        beats, downbeats = np.asarray(beats, float), np.asarray(downbeats, float)
        if len(beats) < 8:
            return None
        tempo = 60.0 / float(np.median(np.diff(beats)))
        # downbeats -> meter and phase: index of each downbeat among the beats
        idx = [int(np.argmin(np.abs(beats - d))) for d in downbeats]
        gaps = np.diff(idx)
        meter = int(np.bincount(gaps[(gaps >= 2) & (gaps <= 7)]).argmax()) if len(gaps) and ((gaps >= 2) & (gaps <= 7)).any() else 4
        offset = int(np.bincount([i % meter for i in idx]).argmax()) if idx else 0
        return BeatGrid(beats=beats, tempo=tempo, meter=meter, downbeat_offset=offset, source="beat_this")
    except Exception as e:  # pragma: no cover - depends on an optional package
        log.warning("beat_this failed (%s); using librosa", e)
        return None


def track_beats(y: np.ndarray, sr: int, start: float = 0.0, end: float | None = None) -> BeatGrid:
    grid = _beat_this(y, sr)
    if grid is not None:
        keep = (grid.beats >= start - 0.5) & ((end is None) | (grid.beats <= (end or 0) + 0.5))
        first = int(np.argmax(keep)) if keep.any() else 0
        grid.beats = grid.beats[keep]
        grid.downbeat_offset = (grid.downbeat_offset - first) % grid.meter
        return grid
    import librosa

    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    tempo, frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, tightness=120, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    # Ballads are often tracked at double time; keep tempo in a natural 60-140 range.
    if tempo > 150 and len(beats) > 4:
        beats, tempo = beats[::2], tempo / 2
    beats = beats[(beats >= start - 0.5) & ((end is None) | (beats <= (end or 0) + 0.5))]
    return BeatGrid(beats=beats, tempo=tempo, meter=4, downbeat_offset=0)


def detect_click_track(y: np.ndarray, sr: int, span: tuple[float, float] | None = None) -> BeatGrid | None:
    """A metronome in the recording is a ground-truth beat grid.

    Looks for very regular, sharp transients (Demucs puts clicks in the drums stem).
    Accepted only if >= 16 clicks with inter-onset intervals within ~4 % of each other,
    covering most of the played span. Accented clicks (louder or brighter) give the downbeat.
    """
    import librosa

    if y is None or len(y) < sr * 5 or float(np.max(np.abs(y))) < 1e-4:
        return None
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if env.max() <= 0:
        return None
    peaks = librosa.util.peak_pick(env, pre_max=8, post_max=8, pre_avg=40, post_avg=40,
                                   delta=0.25 * float(env.max()), wait=int(0.2 * sr / hop))
    if len(peaks) < 16:
        return None
    t = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    ioi = np.diff(t)
    period = float(np.median(ioi))
    if not 0.25 <= period <= 1.6:
        return None
    # keep the longest run of clicks spaced by ~one period (allow a missed click = 2 periods)
    ok = np.abs(ioi / period - np.round(ioi / period)) < 0.06
    ok &= (np.round(ioi / period) >= 1) & (np.round(ioi / period) <= 2)
    runs, cur = [], [0]
    for i, good in enumerate(ok):
        if good:
            cur.append(i + 1)
        else:
            runs.append(cur)
            cur = [i + 1]
    runs.append(cur)
    run = max(runs, key=len)
    if len(run) < 16:
        return None
    tc = t[run]
    k = np.round((tc - tc[0]) / period)
    slope, icpt = np.polyfit(k, tc, 1)  # constant tempo fit
    resid = tc - (slope * k + icpt)
    if float(np.median(np.abs(resid))) > 0.02:
        return None
    s0, s1 = span if span else (0.0, len(y) / sr)
    if (tc[-1] - tc[0]) < 0.5 * max(1e-6, s1 - s0):
        return None
    if 60.0 / slope > 160:  # eighth-note clicks / hi-hat: beat = every other click
        beats_all = icpt + slope * np.arange(int(np.floor((min(s0, tc[0]) - icpt) / slope)),
                                             int(np.ceil((max(s1, tc[-1]) - icpt) / slope)) + 1, 2)
        beats_all = beats_all[(beats_all >= 0) & (beats_all <= len(y) / sr)]
        return BeatGrid(beats=beats_all, tempo=30.0 / slope, meter=4, downbeat_offset=0, source="metronome")
    n0 = int(np.floor((min(s0, tc[0]) - icpt) / slope))
    n1 = int(np.ceil((max(s1, tc[-1]) - icpt) / slope))
    beats = icpt + slope * np.arange(n0, n1 + 1)
    beats = beats[(beats >= 0) & (beats <= len(y) / sr)]
    # accents: brightness + level of each detected click
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    acc = np.array([env[peaks[i]] / env.max() + cent[min(len(cent) - 1, peaks[i])] / (cent.max() + 1e-9) for i in run])
    best = (0.0, 4, 0)
    for meter in (4, 3, 2, 6):
        idx = (k - k.min()).astype(int)
        for ph in range(meter):
            on = acc[idx % meter == ph]
            off = acc[idx % meter != ph]
            if len(on) < 3 or len(off) < 3:
                continue
            gap = float(on.mean() - off.mean()) / (float(acc.std()) + 1e-9)
            if gap > best[0] + (0.15 if meter in (2, 6) else 0.0):
                best = (gap, meter, ph)
    gap, meter, ph = best
    if gap < 0.8:  # no clear accent: keep 4/4 and let chords/bass decide the phase
        meter, ph, src = 4, 0, "metronome"
    else:
        src = "metronome+accent"
    first_click_beat = int(round((tc[0] - beats[0]) / slope))
    offset = (first_click_beat + ph) % meter
    return BeatGrid(beats=beats, tempo=60.0 / slope, meter=meter, downbeat_offset=offset, source=src)


def fit_meter_and_downbeat(grid: BeatGrid, chord_change_times: list[float], bass_onsets: list[float]) -> BeatGrid:
    """Chords change and bass notes land mostly on beat 1: pick meter/phase that agrees."""
    if len(grid.beats) < 8 or grid.source in ("beat_this", "metronome+accent"):
        return grid  # downbeats already known (neural tracker or accented metronome)
    b = grid.beats

    def nearest_beat(t):
        i = int(np.argmin(np.abs(b - t)))
        return i if abs(b[i] - t) < 0.18 else None

    change_idx = [i for i in (nearest_beat(t) for t in chord_change_times) if i is not None]
    bass_idx = [i for i in (nearest_beat(t) for t in bass_onsets) if i is not None]
    best_by_meter = {}
    for meter in (4, 3):
        for phase in range(meter):
            score = sum(2.0 for i in change_idx if (i - phase) % meter == 0)
            score += sum(0.5 for i in bass_idx if (i - phase) % meter == 0)
            score -= sum(0.6 for i in change_idx if (i - phase) % meter not in (0, meter // 2))
            norm = score / (len(change_idx) * 2.0 + len(bass_idx) * 0.5 + 1e-9)
            if meter not in best_by_meter or norm > best_by_meter[meter][0]:
                best_by_meter[meter] = (norm, phase)
    # 4/4 is far more common: only switch to 3/4 on clear evidence.
    meter = 3 if best_by_meter[3][0] > best_by_meter[4][0] + 0.12 else 4
    return BeatGrid(beats=b, tempo=grid.tempo, meter=meter, downbeat_offset=best_by_meter[meter][1])


def key_scores_from_chords(chords: list["ChordSeg"]) -> dict[tuple[int, str], float]:
    """Score every key by how much playing time its diatonic chords cover (tonic favoured)."""
    total = sum(c.end - c.start for c in chords if c.root is not None) or 1.0
    out = {}
    for mode in ("major", "minor"):
        for tonic in range(12):
            dia = theory.diatonic_chords(tonic, mode)
            tri = {(r, q if q in ("", "m") else ("m" if "m" in q else "")) for r, q in dia}
            s = 0.0
            for c in chords:
                if c.root is None:
                    continue
                d = c.end - c.start
                simple = (c.root, "m" if c.quality.startswith("m") and c.quality != "maj7" else "")
                s += d if ((c.root, c.quality) in dia or simple in tri) else -0.6 * d
                if c.root == tonic and simple[1] == ("" if mode == "major" else "m"):
                    s += 0.35 * d
                if c.root == (tonic + 7) % 12:
                    s += 0.1 * d
            if mode == "minor":
                s -= 0.05 * total  # relative-major ties go to major
            out[(tonic, mode)] = s / total
    return out


def estimate_key(notes: list[Note], draft_chords: list["ChordSeg"], span=None, w_notes: float = 2.0):
    """Key from the note pitch-class profile (Krumhansl) plus the draft chords.

    Chords alone confuse parallel keys when thirds are quiet (fingerstyle): C minor read as
    C major. On GuitarSet the combination lifts key accuracy 68 -> 75 % on unseen players.
    """
    hist = np.zeros(12)
    for n in notes:
        if span is None or span[0] - 1 <= n.start <= span[1] + 1:
            hist[n.pitch % 12] += (n.end - n.start) * n.amp
    # sqrt-compress: otherwise the many tonic/fifth notes drown the thirds and sixths that
    # tell major from minor (GuitarSet, unseen players: 72.8 -> 76.8 %)
    prof = theory.key_profile_scores(np.sqrt(hist)) if hist.any() else {k: 0.0 for k in key_scores_from_chords([])}
    csc = key_scores_from_chords(draft_chords)
    (tonic, mode) = max(csc, key=lambda k: csc[k] + w_notes * prof[k])
    return tonic, mode, csc[(tonic, mode)] + w_notes * prof[(tonic, mode)]


def key_from_chords(chords: list["ChordSeg"]) -> tuple[int, str, float]:
    sc = key_scores_from_chords(chords)
    (tonic, mode), s = max(sc.items(), key=lambda kv: kv[1])
    return tonic, mode, s


# ---------------------------------------------------------------------- chords


@dataclass
class ChordSeg:
    start: float
    end: float
    name: str  # 'N' for no chord
    root: int | None
    quality: str
    confidence: float

    def to_dict(self):
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "name": self.name,
            "root": self.root,
            "quality": self.quality,
            "confidence": round(self.confidence, 3),
        }


KEY_BONUS = (0.04, 0.05)  # score bonus for chords in the key, penalty for chords outside it


def _chord_templates():
    labels, tmpl = [], []
    weights = {0: 1.0, 3: 0.85, 4: 0.85, 7: 0.75, 10: 0.55, 11: 0.55}
    for root in range(12):
        for q in theory.RECOGNISED_QUALITIES:
            v = np.zeros(12)
            for iv in theory.QUALITIES[q]:
                v[(root + iv) % 12] = weights.get(iv, 0.6)
            labels.append((root, q))
            tmpl.append(v / np.linalg.norm(v))
    return labels, np.array(tmpl)


def chroma_from_posteriorgram(post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold a (T x 88) note posteriorgram into treble chroma and bass chroma."""
    T = post.shape[0]
    midi = np.arange(21, 109)
    pcs = midi % 12
    # Basic Pitch posteriors have a ~0.1 floor on every bin: remove it, then
    # square to sharpen so that summing octaves does not flatten the chroma.
    sharp = np.clip(post - 0.15, 0, None) ** 2
    chroma = np.zeros((T, 12), dtype=np.float32)
    for pc in range(12):
        chroma[:, pc] = sharp[:, pcs == pc].sum(axis=1)
    # Bass: activation-weighted, favouring the lowest sounding note per frame.
    bass = np.zeros((T, 12), dtype=np.float32)
    low = post[:, (midi >= 38) & (midi <= 57)]
    low_midi = midi[(midi >= 38) & (midi <= 57)]
    active = low > 0.25
    has = active.any(axis=1)
    first = np.argmax(active, axis=1)
    for t in np.where(has)[0]:
        bass[t, low_midi[first[t]] % 12] = low[t, first[t]]
    return chroma, bass


def recognise_chords(
    post: np.ndarray,
    times: np.ndarray,
    beats: np.ndarray,
    energy_times: np.ndarray,
    energy: np.ndarray,
    key: tuple[int, str] | None = None,
    subdivide: int = 1,
) -> list[ChordSeg]:
    """Beat-synchronous chord recognition with a Viterbi smoother."""
    labels, tmpl = _chord_templates()
    chroma, bass = chroma_from_posteriorgram(post)

    # Beat-synchronous frames (optionally split each beat).
    grid = [float(beats[0])]
    for a, b in zip(beats[:-1], beats[1:]):
        for k in range(1, subdivide + 1):
            grid.append(float(a + (b - a) * k / subdivide))
    grid = np.array(grid)
    S = len(grid) - 1
    C = np.zeros((S, 12))
    B = np.zeros((S, 12))
    E = np.zeros(S)
    for i in range(S):
        m = (times >= grid[i]) & (times < grid[i + 1])
        if m.any():
            C[i] = chroma[m].mean(axis=0)
            B[i] = bass[m].mean(axis=0)
        em = (energy_times >= grid[i]) & (energy_times < grid[i + 1])
        E[i] = energy[em].mean() if em.any() else 0.0

    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    sim = Cn @ tmpl.T  # S x K
    Bn = B / (B.sum(axis=1, keepdims=True) + 1e-9)
    roots = np.array([r for r, _ in labels])
    quals = [q for _, q in labels]
    bass_root = Bn[:, roots]  # bass energy on each label's root
    score = sim + 0.35 * bass_root
    seventh_pen = np.array([0.035 if q in ("7", "maj7", "m7") else (0.05 if q == "m7b5" else 0.0) for q in quals])
    score = score - seventh_pen
    if key is not None:
        dia = theory.diatonic_chords(*key)
        score = score + np.array([KEY_BONUS[0] if lab in dia else -KEY_BONUS[1] for lab in labels])

    # Add a "no chord" state driven by guitar energy.
    thr = max(1e-6, float(np.percentile(E[E > 0], 20)) * 0.5) if (E > 0).any() else 1e-6
    n_score = np.where(E < thr, 1.2, 0.0) + np.where(C.sum(axis=1) < 1e-3, 1.0, 0.0)
    score = np.concatenate([score, n_score[:, None]], axis=1)
    K = score.shape[1]
    kappa = 12.0
    emit = kappa * score
    stay = np.log(0.88 if subdivide == 1 else 0.94)
    move = np.log((1 - np.exp(stay)) / (K - 1))
    dp = emit[0].copy()
    back = np.zeros((S, K), dtype=np.int32)
    for i in range(1, S):
        best_prev = int(np.argmax(dp))
        cand_move = dp[best_prev] + move
        cand_stay = dp + stay
        choose_stay = cand_stay >= cand_move
        back[i] = np.where(choose_stay, np.arange(K), best_prev)
        dp = np.where(choose_stay, cand_stay, cand_move) + emit[i]
    path = np.zeros(S, dtype=np.int32)
    path[-1] = int(np.argmax(dp))
    for i in range(S - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    segs: list[ChordSeg] = []
    for i, k in enumerate(path):
        if k == K - 1:
            name, root, q = "N", None, ""
        else:
            root, q = labels[k]
            name = theory.chord_name(root, q)
        # confidence = margin between the chosen label and the runner-up
        row = np.sort(score[i])[::-1]
        conf = float(np.clip((score[i, k] - row[1]) * 5 + 0.5, 0, 1)) if score[i, k] >= row[0] else 0.3
        if segs and segs[-1].name == name:
            segs[-1].end = float(grid[i + 1])
            segs[-1].confidence = 0.5 * (segs[-1].confidence + conf)
        else:
            segs.append(ChordSeg(float(grid[i]), float(grid[i + 1]), name, root, q, conf))
    return segs


def analyse_harmony(post, times, grid: BeatGrid, et, en, span, notes: list[Note]):
    """Chords (two passes: no key prior -> key from chords -> key prior), key, meter/downbeat."""
    s0, s1 = span
    draft = recognise_chords(post, times, grid.beats, et, en, None)
    draft = [c for c in draft if c.end > s0 and c.start < s1]
    tonic, mode, key_r = estimate_key(notes, draft, span)
    chords = recognise_chords(post, times, grid.beats, et, en, (tonic, mode))
    chords = [c for c in chords if c.end > s0 - 1 and c.start < s1 + 1]
    bass_onsets = [n.start for n in notes if n.pitch < 52]
    grid = fit_meter_and_downbeat(grid, [c.start for c in chords[1:]], bass_onsets)
    return chords, (tonic, mode, key_r), grid


def guitar_energy(y: np.ndarray, sr: int, hop: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=hop)[0]
    return librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop), rms


def find_music_span(times: np.ndarray, energy: np.ndarray, post_times: np.ndarray, post: np.ndarray):
    """Rough start/end of the played music (skips talking intros/outros)."""
    act = post.max(axis=1)
    win = 2.0
    hop_t = post_times[1] - post_times[0]
    k = max(1, int(win / hop_t))
    smooth = np.convolve((act > 0.4).astype(float), np.ones(k) / k, mode="same")
    playing = smooth > 0.35
    if not playing.any():
        return float(post_times[0]), float(post_times[-1])
    idx = np.where(playing)[0]
    # longest run of playing frames, allowing gaps of up to 6 s
    runs, s, prev = [], idx[0], idx[0]
    gap = int(6.0 / hop_t)
    for i in idx[1:]:
        if i - prev > gap:
            runs.append((s, prev))
            s = i
        prev = i
    runs.append((s, prev))
    s, e = max(runs, key=lambda r: r[1] - r[0])
    return float(post_times[s]), float(post_times[e])
