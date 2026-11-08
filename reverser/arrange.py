"""Rhythm quantisation, bars, picking/strum patterns, sections and an easy version."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np

from . import theory
from .audio import BeatGrid, ChordSeg, Note
from .fingering import Event


def beat_function(grid: BeatGrid):
    """Continuous map time -> fractional beat index (extrapolated linearly)."""
    b = grid.beats
    idx = np.arange(len(b), dtype=float)
    period = 60.0 / max(grid.tempo, 1e-3)

    def f(t: float) -> float:
        if len(b) < 2:
            return t / period
        if t < b[0]:
            return (t - b[0]) / (b[1] - b[0])
        if t > b[-1]:
            return idx[-1] + (t - b[-1]) / (b[-1] - b[-2])
        return float(np.interp(t, b, idx))

    return f


def time_function(grid: BeatGrid):
    b = grid.beats
    idx = np.arange(len(b), dtype=float)

    def g(beat: float) -> float:
        if beat < 0:
            return float(b[0] + beat * (b[1] - b[0]))
        if beat > idx[-1]:
            return float(b[-1] + (beat - idx[-1]) * (b[-1] - b[-2]))
        return float(np.interp(beat, idx, b))

    return g


def quantise(events: list[Event], grid: BeatGrid) -> None:
    """Snap onsets to 16ths or 8th-note triplets, whichever fits each beat best."""
    bf = beat_function(grid)
    for ev in events:
        ev.beat = bf(ev.t)
    # choose per-beat subdivision by total snapping error
    by_beat: dict[int, list[Event]] = {}
    for ev in events:
        by_beat.setdefault(int(np.floor(ev.beat + 1 / 8)), []).append(ev)
    for _, evs in by_beat.items():
        def err(div):
            return sum(abs(e.beat * div - round(e.beat * div)) / div for e in evs)

        div = 3 if err(3) < 0.6 * err(4) and len(evs) >= 2 else 4
        for e in evs:
            e.qbeat = round(e.beat * div) / div
    for ev in events:
        ev.bar = int(np.floor((ev.qbeat - grid.downbeat_offset) / grid.meter))


def bar_patterns(events: list[Event], grid: BeatGrid) -> list[dict]:
    """Per-bar rhythm string on an 8th grid (strums) or PIMA sequence (fingerstyle)."""
    bars: dict[int, list[Event]] = {}
    for ev in events:
        bars.setdefault(ev.bar, []).append(ev)
    out = []
    slots = grid.meter * 2
    for bar, evs in sorted(bars.items()):
        cells = ["-"] * slots
        picks = []
        n_strum = sum(1 for e in evs if e.kind == "strum")
        for e in evs:
            pos = e.qbeat - grid.downbeat_offset - bar * grid.meter
            k = int(round(pos * 2))
            if 0 <= k < slots:
                if e.kind == "strum":
                    cells[k] = e.direction or "D"
                elif cells[k] == "-":
                    # finger + string number (tab convention 1 = high e), e.g. "p5" or "p4.m2"
                    ns = sorted(e.notes, key=lambda n: n.string)
                    toks = [f"{n.pluck}{6 - n.string}" for n in ns if n.pluck]
                    if len(toks) > 2:
                        toks = [toks[0], toks[-1]]
                    cells[k] = ".".join(toks) if toks else "-"
            picks.append("".join(n.pluck for n in sorted(e.notes, key=lambda n: n.string)) or (e.direction or ""))
        style = "strum" if n_strum >= max(1, len(evs) // 2) else "fingerstyle"
        out.append({"bar": bar, "style": style, "cells": cells, "picks": picks})
    return out


def common_pattern(patterns: list[dict]) -> dict:
    if not patterns:
        return {"style": "unknown", "cells": [], "count": 0}
    styles = Counter(p["style"] for p in patterns)
    style = styles.most_common(1)[0][0]
    pool = [p["cells"] for p in patterns if p["style"] == style and sum(c != "-" for c in p["cells"]) >= 2]
    if not pool:
        return {"style": style, "cells": [], "count": 0, "of": len(patterns)}

    def fingers(tok):
        return "".join(ch for ch in tok if ch.isalpha())

    def sim(a, b):
        # same token = 1, same fingers on other strings (chord changed) = 0.8,
        # both played otherwise = 0.4, rest vs note = 0
        tot = 0.0
        for x, y in zip(a, b):
            if x == y:
                tot += 1.0
            elif x != "-" and y != "-":
                tot += 0.8 if fingers(x) == fingers(y) else 0.4
        return tot / len(a)

    # medoid bar: the one most similar to all others (robust to transcription noise)
    scores = [sum(sim(a, b) for b in pool) for a in pool]
    medoid = pool[int(np.argmax(scores))]
    close = sum(1 for b in pool if sim(medoid, b) >= 0.75)
    return {"style": style, "cells": medoid, "count": close, "of": len(patterns)}


def bar_chords(chords: list[ChordSeg], grid: BeatGrid, n_bars: int, tf) -> list[list[str]]:
    """Chord name per beat for every bar (for the chord-sheet view)."""
    out = []
    for bar in range(n_bars):
        names = []
        for beat in range(grid.meter):
            t = tf(grid.downbeat_offset + bar * grid.meter + beat + 0.5)
            name = "N"
            for c in chords:
                if c.start <= t < c.end:
                    name = c.name
                    break
            names.append(name)
        out.append(names)
    return out


def find_sections(bar_chord_rows: list[list[str]], first_bar: int, last_bar: int) -> list[dict]:
    """Group bars into repeated chord loops (2 or 4 bars) and label them A, B, C..."""
    rows = bar_chord_rows
    L = 4  # phrases of 4 bars

    def simplify(name):
        p = theory.parse_chord(name)
        return "N" if p is None else theory.chord_name(p[0], theory.SIMPLIFY.get(p[1], p[1]))

    def phrase(b):
        # half-bar resolution, qualities simplified: robust to one-beat glitches
        out = []
        for r in rows[b : b + L]:
            half = max(1, len(r) // 2)
            out += [simplify(Counter(r[:half]).most_common(1)[0][0]), simplify(Counter(r[half:]).most_common(1)[0][0])]
        return out

    def sim(a, b):
        n = max(len(a), len(b))
        return sum(1 for x, y in zip(a, b) if x == y) / n if n else 0.0

    sections: list[dict] = []
    protos: list[tuple[str, list[str]]] = []
    bar = max(0, first_bar)
    while bar <= min(last_bar, len(rows) - 1):
        ph = phrase(bar)
        if all(c == "N" for c in ph):
            bar += L
            continue
        best = max(protos, key=lambda p: sim(p[1], ph), default=None)
        if best is not None and sim(best[1], ph) >= 0.6:
            lab = best[0]
        else:
            lab = chr(ord("A") + len(protos)) if len(protos) < 26 else "Z"
            protos.append((lab, ph))
        end = min(bar + L, len(rows))
        if sections and sections[-1]["label"] == lab and sections[-1]["endBar"] == bar:
            sections[-1]["endBar"] = end
            sections[-1]["repeats"] += 1
        else:
            sections.append({"label": lab, "startBar": bar, "endBar": end, "repeats": 1, "chords": rows[bar:end]})
        bar += L
    return sections


def easy_events(events: list[Event], chords: list[ChordSeg], grid: BeatGrid, capo: int) -> list[Event]:
    """Beginner tab: on-beat bass notes + one treble note, snapped to open chord shapes."""
    from .fingering import _chord_voicings

    starts = [c.start for c in chords]
    import bisect

    out = []
    for ev in events:
        frac = ev.qbeat - np.floor(ev.qbeat)
        if frac not in (0.0, 0.5):
            continue  # drop 16ths/triplet ornaments
        i = bisect.bisect_right(starts, ev.t + 0.02) - 1
        seg = chords[i] if 0 <= i < len(chords) else None
        vs = _chord_voicings(seg, capo)
        v = next((x for x in vs if x.open_shape), vs[0] if vs else None)
        new = copy.deepcopy(ev)
        notes = sorted(new.notes, key=lambda n: n.pitch)
        keep = [notes[0]] if frac == 0.0 else []
        if len(notes) > 1:
            keep.append(max(notes[1:], key=lambda n: n.amp))
        elif frac == 0.5:
            keep = notes[:1]
        if v is not None:
            pcs = theory.chord_pitch_classes((seg.root - capo) % 12, seg.quality) if seg and seg.root is not None else set()
            for n in keep:
                if v.frets[n.string] >= 0 and v.frets[n.string] != n.fret:
                    # snap onto the chord shape if the shape's note is a chord tone
                    p = theory.STANDARD_TUNING[n.string] + capo + v.frets[n.string]
                    if (p - capo) % 12 in pcs or not pcs:
                        n.fret = v.frets[n.string]
                        n.pitch = p
                        n.finger = v.fingers[n.string]
        new.notes = keep
        new.kind = "pluck" if ev.kind == "pluck" else "strum"
        out.append(new)
    return out


def chord_sheet_events(chords: list[ChordSeg], grid: BeatGrid, capo: int, tf, bf, pattern_cells: list[str]) -> list[dict]:
    """Chord-mode practice track: the full voicing strummed on the common pattern."""
    from .fingering import _chord_voicings

    events = []
    if not pattern_cells:
        pattern_cells = ["D", "-", "D", "U", "-", "U", "D", "U"][: grid.meter * 2]
    for c in chords:
        if c.root is None:
            continue
        vs = _chord_voicings(c, capo)
        v = next((x for x in vs if x.open_shape), vs[0] if vs else None)
        if v is None:
            continue
        b0, b1 = bf(c.start), bf(c.end)
        k = int(np.ceil(b0 * 2))
        while k / 2 < b1 - 0.01:
            bar_pos = (k - grid.downbeat_offset * 2) % (grid.meter * 2)
            cell = pattern_cells[bar_pos] if bar_pos < len(pattern_cells) else "-"
            if cell != "-":
                d = cell if cell in ("D", "U") else "D"
                if cell not in ("D", "U"):
                    d = "P"  # fingerstyle cell: pluck the chord
                events.append({"t": round(tf(k / 2), 4), "dir": d, "chord": c.name, "frets": v.frets})
            k += 1
    return events
