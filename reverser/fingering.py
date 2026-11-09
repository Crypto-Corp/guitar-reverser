"""String/fret/finger assignment as a shortest path over playable hand states.

Each onset cluster ("event") gets candidate fingerings (one string per note).
A Viterbi pass minimises:
  unary  : hand height, stretch, high frets, mismatch with the current chord's
           standard shape, disagreement with the video's fret-position prior;
  pairwise: hand shifts (worse when fast) and cutting a note that still rings.
Picking-hand fingers (p/i/m/a) and strum directions are then derived.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from . import theory
from .audio import ChordSeg, Note

MAX_FRET = 17


@dataclass
class Event:
    t: float
    notes: list[Note]
    kind: str = "pluck"  # 'pluck' | 'strum'
    direction: str | None = None  # 'D' | 'U' for strums
    beat: float = -1.0  # fractional beat index
    qbeat: float = -1.0  # quantised beat index
    bar: int = -1
    hand_pos: int = 0
    chord: str = "N"

    def to_dict(self):
        return {
            "t": round(self.t, 4),
            "end": round(max(n.end for n in self.notes), 4),
            "kind": self.kind,
            "dir": self.direction,
            "beat": round(self.qbeat, 4),
            "bar": self.bar,
            "pos": self.hand_pos,
            "chord": self.chord,
            "notes": [
                {
                    "s": n.string,
                    "f": n.fret,
                    "p": n.pitch,
                    "fg": n.finger,
                    "pk": n.pluck,
                    "end": round(n.end, 4),
                    "amp": round(n.amp, 3),
                }
                for n in sorted(self.notes, key=lambda n: n.string)
            ],
        }


def group_events(notes: list[Note], window: float = 0.045) -> list[Event]:
    notes = sorted(notes, key=lambda n: n.start)
    events: list[Event] = []
    for n in notes:
        if events and n.start - events[-1].t <= window:
            ev = events[-1]
            if any(m.pitch == n.pitch for m in ev.notes):
                same = next(m for m in ev.notes if m.pitch == n.pitch)
                same.end = max(same.end, n.end)
                same.amp = max(same.amp, n.amp)
                continue
            ev.notes.append(n)
        else:
            events.append(Event(t=n.start, notes=[n]))
    for ev in events:
        if len(ev.notes) > 6:
            ev.notes = sorted(ev.notes, key=lambda n: -n.amp)[:6]
        ev.notes.sort(key=lambda n: n.pitch)
    return events


@dataclass
class _State:
    frets: tuple  # per kept note (aligned with `keep`)
    strings: tuple
    pos: int | None  # lowest fretted fret, None if all open
    cost: float
    keep: tuple = ()  # indices into ev.notes that are actually played


def _candidates(pitches, tuning, capo, limit=60):
    out = []
    n = len(pitches)

    def rec(i, used, strings, frets):
        if i == n:
            fretted = [f for f in frets if f > 0]
            span = (max(fretted) - min(fretted)) if fretted else 0
            if span > 5:
                return
            out.append((tuple(strings), tuple(frets)))
            return
        p = pitches[i]
        for s in range(6):
            if s in used:
                continue
            f = p - tuning[s] - capo
            if 0 <= f <= MAX_FRET:
                rec(i + 1, used | {s}, strings + [s], frets + [f])

    rec(0, frozenset(), [], [])
    return out


def _chord_voicings(seg: ChordSeg | None, capo: int):
    if seg is None or seg.root is None:
        return []
    root = (seg.root - capo) % 12
    vs = theory.voicings_for(root, seg.quality)
    simple = theory.SIMPLIFY.get(seg.quality, seg.quality)
    if simple != seg.quality:
        vs += theory.voicings_for(root, simple)
    return vs[:3]


# Cost weights of the fingering model. Units are arbitrary; only ratios matter.
# Fitted by coordinate search on GuitarSet players 00-02 (true notes) and checked on
# players 03-05: string accuracy 67.2 % -> 76.0 % on the unseen players (solos 45 -> 60 %).
# The fit removed the old "stay low on the neck" bias (pos, open = 0): real players don't.
COSTS: dict[str, float] = {
    "pos": 0.0,         # per fret of hand position (index finger): lower is easier
    "span": 0.05,        # per fret between lowest and highest fretted note
    "span_quad": 0.9,   # extra, quadratic, for stretches beyond 3 frets
    "high": 0.25,        # per note above fret 12
    "open": 0.0,        # bonus per open string (beginner friendly, rings)
    "shape_hit": 0.55,   # note matches the current chord's standard shape
    "shape_mute": 0.35,  # note on a string the shape mutes
    "shape_miss": 0.2,   # note elsewhere on a string the shape uses
    "shape_diff": 0.05,  # per difficulty point of that shape
    "prior": 0.4,        # disagreement with the video's fret-position estimate
    "shift": 0.5,       # per fret of hand shift (scaled up when fast)
    "jump": 0.4,         # extra for a shift larger than `jump_at`
    "jump_at": 4,
    "ring": 0.3,         # cutting a note that is still ringing on that string
    "anchor": 0.0,      # per fret between the index finger and the lowest fretted note
    "box": 2,            # frets the hand covers above the index without shifting
}
_POP = np.array([bin(i).count("1") for i in range(64)], dtype=np.float64)


def _anchors(frets) -> list[int | None]:
    """Possible index-finger positions for a fingering (None = only open strings)."""
    fretted = [f for f in frets if f > 0]
    if not fretted:
        return [None]
    lo, hi = min(fretted), max(fretted)
    box = int(COSTS["box"])
    if hi - lo >= box:
        return [lo]
    return list(range(max(1, hi - box), lo + 1))


def _unary(strings, frets, voicings, prior, n_notes, anchor=None):
    C = COSTS
    fretted = [f for f in frets if f > 0]
    lo = min(fretted) if fretted else None
    pos = anchor if anchor is not None else lo
    span = (max(fretted) - min(fretted)) if fretted else 0
    c = 0.0
    c += C["pos"] * (pos or 0)
    if lo is not None and pos is not None:
        c += C["anchor"] * (lo - pos)
    c += C["span"] * span + C["span_quad"] * max(0, span - 3) ** 2
    c += C["high"] * sum(1 for f in frets if f > 12)
    c -= C["open"] * sum(1 for f in frets if f == 0)
    if voicings:
        best = None
        for v in voicings:
            vc = 0.0
            for s, f in zip(strings, frets):
                if v.frets[s] == f:
                    vc -= C["shape_hit"]
                elif v.frets[s] == -1:
                    vc += C["shape_mute"]
                else:
                    vc += C["shape_miss"]
            vc += C["shape_diff"] * v.difficulty()
            best = vc if best is None else min(best, vc)
        c += best / max(1, n_notes) ** 0.5
    if prior is not None:
        p_pos, conf = prior
        if p_pos is not None and conf > 0:
            hp = pos if pos is not None else 1
            c += conf * C["prior"] * max(0.0, abs(hp - p_pos) - 1.5)
    return c, pos


def _states(cands, keep, voicings, prior, extra=0.0):
    out = []
    for strings, frets in cands:
        for a in _anchors(frets):
            c, pos = _unary(strings, frets, voicings, prior, len(frets), a)
            out.append(_State(frets, strings, pos, c + extra, keep))
    return out


def assign_fingering(
    events: list[Event],
    chords: list[ChordSeg],
    tuning: list[int] = theory.STANDARD_TUNING,
    capo: int = 0,
    prior_fn=None,
    beam: int = 160,
) -> float:
    """In-place Viterbi assignment over (fingering, hand position) states. Returns the path cost."""
    chord_starts = [c.start for c in chords]

    def chord_at(t):
        i = bisect.bisect_right(chord_starts, t) - 1
        return chords[i] if 0 <= i < len(chords) and chords[i].end > t else None

    layers: list[list[_State]] = []
    for ev in events:
        seg = chord_at(ev.t + 0.02)
        ev.chord = seg.name if seg else "N"
        voicings = _chord_voicings(seg, capo)
        prior = prior_fn(ev.t) if prior_fn else None
        if prior is not None and prior[0] is not None and capo:
            # the video measures absolute frets; fingering frets count from the capo
            prior = (max(0.0, prior[0] - capo), prior[1])
        ev.notes = [n for n in ev.notes if n.pitch - tuning[0] - capo >= 0]  # below the capo: unplayable
        pitches = [n.pitch for n in ev.notes]
        cands = _candidates(pitches, tuning, capo)
        while not cands and len(pitches) > 1:
            # unplayable cluster: drop the quietest note and retry
            drop = min(range(len(ev.notes)), key=lambda k: ev.notes[k].amp)
            ev.notes.pop(drop)
            pitches = [n.pitch for n in ev.notes]
            cands = _candidates(pitches, tuning, capo)
        full = tuple(range(len(ev.notes)))
        states = _states(cands, full, voicings, prior)
        # Robust decoding: the quietest note may be a ghost (harmonic, vocal bleed).
        # Offer states without it; dropping costs more the louder the note is.
        if ev.notes:
            k = min(range(len(ev.notes)), key=lambda i: ev.notes[i].amp)
            amp = ev.notes[k].amp
            drop_cost = 0.5 + 2.2 * amp
            if len(ev.notes) >= 2:
                sub = tuple(i for i in full if i != k)
                sub_c = _candidates([ev.notes[i].pitch for i in sub], tuning, capo)
                states += _states(sub_c, sub, voicings, prior, drop_cost)
            elif amp < 0.35:
                states.append(_State(tuple(), tuple(), None, drop_cost + 0.4, tuple()))
        states.sort(key=lambda s: s.cost)
        layers.append(states[:beam] or [_State(tuple(), tuple(), None, 5.0, tuple())])

    if not layers:
        assign_fingering.last_dropped = 0
        return 0.0

    # Vectorised Viterbi. Open-string-only states keep the previous hand position
    # (`eff`), so an open note does not reset the hand.
    C = COSTS
    nan = float("nan")

    def arrays(L):
        pos = np.array([nan if s.pos is None else s.pos for s in L], dtype=np.float64)
        cost = np.array([s.cost for s in L], dtype=np.float64)
        mask = np.array([sum(1 << st for st in s.strings) for s in L], dtype=np.int64)
        return pos, cost, mask

    pos0, cost0, _ = arrays(layers[0])
    dp, eff = cost0, pos0
    back: list[np.ndarray] = [np.full(len(layers[0]), -1)]
    for k in range(1, len(layers)):
        prev_ev, ev = events[k - 1], events[k]
        gap = max(0.0, ev.t - prev_ev.t)
        speed = 1.0 / (gap + 0.12)
        ringing = {i for i, n in enumerate(prev_ev.notes) if n.end > ev.t + 0.06}
        rmask = np.array([sum(1 << p.strings[slot] for slot, i in enumerate(p.keep) if i in ringing)
                          for p in layers[k - 1]], dtype=np.int64)
        pos, cost, mask = arrays(layers[k])
        d = np.abs(pos[None, :] - eff[:, None])
        d = np.where(np.isnan(d), 0.0, d)
        trans = C["shift"] * d * speed ** 0.5 + C["jump"] * (d > C["jump_at"])
        trans += C["ring"] * _POP[rmask[:, None] & mask[None, :]]
        tot = dp[:, None] + trans
        arg = np.argmin(tot, axis=0)
        dp = tot[arg, np.arange(len(arg))] + cost
        eff = np.where(np.isnan(pos), eff[arg], pos)
        back.append(arg)
    # backtrack
    j = int(np.argmin(dp))
    total = float(dp[j])
    dropped = 0
    for k in range(len(layers) - 1, -1, -1):
        s = layers[k][j]
        ev = events[k]
        kept = [ev.notes[i] for i in s.keep]
        dropped += len(ev.notes) - len(kept)
        for n, st, fr in zip(kept, s.strings, s.frets):
            n.string, n.fret = st, fr
        ev.notes = kept
        ev.hand_pos = s.pos or 0
        j = int(back[k][j])
    events[:] = [e for e in events if e.notes]
    _assign_fretting_fingers(events, chords, capo, chord_at)
    assign_fingering.last_dropped = dropped
    return total


def _assign_fretting_fingers(events, chords, capo, chord_at):
    anchor = 1
    for ev in events:
        seg = chord_at(ev.t + 0.02)
        vs = _chord_voicings(seg, capo)
        matched = None
        for v in vs:
            if all(v.frets[n.string] == n.fret for n in ev.notes):
                matched = v
                break
        if matched:
            for n in ev.notes:
                n.finger = matched.fingers[n.string] if n.fret > 0 else 0
            fretted = [n.fret for n in ev.notes if n.fret > 0]
            if fretted:
                anchor = max(1, min(fretted) - (min(matched.fingers[n.string] for n in ev.notes if n.fret > 0) - 1))
            continue
        fretted = [n.fret for n in ev.notes if n.fret > 0]
        if fretted:
            lo, hi = min(fretted), max(fretted)
            if ev.hand_pos > 0 and ev.hand_pos <= lo and hi <= ev.hand_pos + 3:
                anchor = ev.hand_pos  # index-finger position chosen by the Viterbi hand model
            elif not (anchor <= lo and hi <= anchor + 3):
                anchor = lo
        for n in ev.notes:
            n.finger = 0 if n.fret == 0 else int(np.clip(n.fret - anchor + 1, 1, 4))


def assign_picking(events: list[Event], beat_fn, vel_t=None, vel_v=None) -> None:
    """Strum vs pluck, strum direction, and p/i/m/a fingers."""
    for ev in events:
        strings = sorted(n.string for n in ev.notes)
        spread = max(n.start for n in ev.notes) - min(n.start for n in ev.notes)
        if len(ev.notes) >= 4 and strings[-1] - strings[0] >= 3:
            ev.kind = "strum"
            # direction from the order in which strings sound
            d = None
            if spread > 0.008:
                ss = np.array([n.string for n in ev.notes], float)
                ts = np.array([n.start for n in ev.notes], float)
                if ss.std() > 0 and ts.std() > 0:
                    r = float(np.corrcoef(ss, ts)[0, 1])
                    if abs(r) > 0.35:
                        d = "D" if r > 0 else "U"
            if d is None and vel_t is not None and len(vel_t):
                i = int(np.argmin(np.abs(vel_t - ev.t)))
                if abs(vel_t[i] - ev.t) < 0.12 and abs(vel_v[i]) > 80:
                    d = "D" if vel_v[i] > 0 else "U"
            if d is None:
                b = beat_fn(ev.t)
                frac = b - np.floor(b)
                d = "U" if 0.3 < frac < 0.7 else "D"
            ev.direction = d
            for n in ev.notes:
                n.pluck = ""
        else:
            ev.kind = "pluck"
            ordered = sorted(ev.notes, key=lambda n: n.string)
            fingers = []
            if ordered and ordered[0].string <= 2:
                fingers.append("p")
                rest = ordered[1:]
            else:
                rest = ordered
            pool = ["i", "m", "a"]
            if len(rest) <= 3:
                # treble strings G/B/e map naturally to i/m/a
                natural = {3: "i", 4: "m", 5: "a"}
                assigned = []
                used = set()
                for n in rest:
                    f = natural.get(n.string)
                    if f is None or f in used:
                        f = next((x for x in pool if x not in used), "p")
                    used.add(f)
                    assigned.append(f)
                # keep the finger order ascending with string order
                order = sorted(assigned, key=pool.index) if all(a in pool for a in assigned) else assigned
                fingers += order
            else:
                fingers += ["p"] * (len(rest) - 3) + pool
            for n, f in zip(ordered, fingers):
                n.pluck = f


def fit_tab(notes: list[Note], chords: list[ChordSeg], capo: int | None = None, prior_fn=None,
            max_capo: int = 5, tuning: list[int] = theory.STANDARD_TUNING):
    """Notes -> fingered events. Picks the capo (cheapest path) unless one is given.

    Returns (events, capo, capo_scores, dropped_note_count). `notes` are modified in place.
    """
    def cost_for(c):
        evs = group_events([Note(n.start, n.end, n.pitch, n.amp) for n in notes])
        return assign_fingering(evs, chords, tuning, capo=c, prior_fn=prior_fn) / max(1, len(evs))

    scores = []
    if capo is None:
        capo, scores = choose_capo(cost_for, max_capo=max_capo)
    events = group_events(notes)
    assign_fingering(events, chords, tuning, capo=capo, prior_fn=prior_fn)
    return events, capo, scores, getattr(assign_fingering, "last_dropped", 0)


def choose_capo(notes_events_fn, max_capo: int = 5) -> tuple[int, list]:
    """Try several capo positions; the cheapest fingering path wins."""
    scores = []
    for capo in range(0, max_capo + 1):
        cost = notes_events_fn(capo)
        scores.append((cost + 0.02 * capo, capo))
    scores.sort()
    return scores[0][1], scores
