"""Exporters: ASCII tab and MIDI."""

from __future__ import annotations

from pathlib import Path

from .fingering import Event

STRING_LABELS = ["E", "A", "D", "G", "B", "e"]  # low -> high


def ascii_tab(events: list[Event], meter: int, chords_per_bar: list[list[str]], first_bar: int,
              last_bar: int, downbeat_offset: int = 0, cols_per_beat: int = 4, bars_per_line: int = 4) -> str:
    """Classic 6-line text tab, one 16th per column, with chord names on top."""
    by_bar: dict[int, list[Event]] = {}
    for ev in events:
        by_bar.setdefault(ev.bar, []).append(ev)
    lines_out = []
    bars = list(range(first_bar, last_bar + 1))
    for chunk_start in range(0, len(bars), bars_per_line):
        chunk = bars[chunk_start : chunk_start + bars_per_line]
        rows = {s: "" for s in range(6)}
        chord_row = ""
        for bar in chunk:
            width = meter * cols_per_beat
            cells = {s: ["-"] * width for s in range(6)}
            names = chords_per_bar[bar] if 0 <= bar < len(chords_per_bar) else []
            crow = [" "] * (width * 3)
            last = None
            for beat, name in enumerate(names):
                if name != last and name != "N":
                    pos = beat * cols_per_beat * 3
                    for k, ch in enumerate(name):
                        if pos + k < len(crow):
                            crow[pos + k] = ch
                last = name
            for ev in by_bar.get(bar, []):
                beat_in_bar = ev.qbeat - downbeat_offset - ev.bar * meter
                col = int(round(beat_in_bar * cols_per_beat))
                col = max(0, min(width - 1, col))
                for n in ev.notes:
                    cells[n.string][col] = str(n.fret)
            for s in range(6):
                txt = ""
                for c in cells[s]:
                    txt += c.ljust(3, "-")
                rows[s] += "|" + txt
            chord_row += " " + "".join(crow)
        lines_out.append("  " + chord_row.rstrip())
        for s in range(5, -1, -1):
            lines_out.append(f"{STRING_LABELS[s]} " + rows[s] + "|")
        lines_out.append("")
    return "\n".join(lines_out)


def write_midi(events: list[Event], path: Path, tempo: float) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=24, name="Nylon guitar")  # GM 25: acoustic nylon
    for ev in events:
        for n in ev.notes:
            vel = int(max(30, min(127, 40 + n.amp * 90)))
            inst.notes.append(pretty_midi.Note(velocity=vel, pitch=n.pitch, start=n.start, end=max(n.end, n.start + 0.05)))
    pm.instruments.append(inst)
    pm.write(str(path))
