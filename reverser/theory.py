"""Music theory primitives: tuning, chord vocabulary, guitar voicings.

Conventions used everywhere in the project:
- strings are indexed 0..5 from LOW E (6th string) to HIGH E (1st string);
- a fret of -1 means the string is muted, 0 means open;
- frets in voicings are relative to the capo when a capo is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Prefer flats/sharps the way guitar charts usually spell chord roots.
ROOT_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Chord qualities: intervals (semitones from root) and template weights.
QUALITIES: dict[str, list[int]] = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "m7b5": [0, 3, 6, 10],  # half-diminished (ii of a minor key, vii of a major key)
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}
# Qualities considered by the chord recogniser (sus chords are added only
# through voicing lookup; they are too easily confused in recognition).
RECOGNISED_QUALITIES = ["", "m", "7", "maj7", "m7", "m7b5"]

# How a beginner version simplifies each quality.
SIMPLIFY = {"": "", "m": "m", "7": "7", "maj7": "", "m7": "m", "m7b5": "m7b5", "sus2": "", "sus4": ""}


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def parse_chord(name: str) -> tuple[int, str] | None:
    """'F#m7' -> (6, 'm7'). Returns None for 'N' (no chord)."""
    if not name or name == "N":
        return None
    root = name[0].upper()
    rest = name[1:]
    if rest[:1] in ("#", "b"):
        root += rest[0]
        rest = rest[1:]
    flats = {"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb", "Cb": "B", "Fb": "E"}
    root = flats.get(root, root)
    if root not in ROOT_NAMES:
        raise ValueError(f"unknown chord root in {name!r}")
    if rest not in QUALITIES:
        raise ValueError(f"unknown chord quality in {name!r}")
    return ROOT_NAMES.index(root), rest


def chord_name(root: int, quality: str) -> str:
    return f"{ROOT_NAMES[root % 12]}{quality}"


def chord_pitch_classes(root: int, quality: str) -> set[int]:
    return {(root + i) % 12 for i in QUALITIES[quality]}


@dataclass
class Voicing:
    """One way to play a chord. frets/fingers are low-E -> high-E."""

    name: str
    frets: list[int]
    fingers: list[int]
    barre: int | None = None  # fret of the barre, if any
    open_shape: bool = False  # True for first-position open chords
    tags: list[str] = field(default_factory=list)

    @property
    def base_fret(self) -> int:
        fretted = [f for f in self.frets if f > 0]
        return min(fretted) if fretted else 0

    def pitches(self, tuning: list[int] = STANDARD_TUNING, capo: int = 0) -> list[int | None]:
        return [None if f < 0 else tuning[s] + capo + f for s, f in enumerate(self.frets)]

    def difficulty(self) -> float:
        """Rough beginner difficulty: open chords easy, barres hard."""
        fretted = [f for f in self.frets if f > 0]
        span = (max(fretted) - min(fretted)) if fretted else 0
        d = 1.0 + 0.4 * span + 0.15 * len(fretted)
        if self.barre is not None:
            d += 2.5
        if not self.open_shape:
            d += 1.0
        return d

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "frets": self.frets,
            "fingers": self.fingers,
            "barre": self.barre,
            "baseFret": self.base_fret,
            "open": self.open_shape,
        }


def _v(name, frets, fingers, barre=None):
    return Voicing(name, frets, fingers, barre, open_shape=barre is None)


# First-position chords. The first 16 are exactly the beginner chart shapes.
OPEN_VOICINGS: list[Voicing] = [
    _v("E", [0, 2, 2, 1, 0, 0], [0, 2, 3, 1, 0, 0]),
    _v("A", [-1, 0, 2, 2, 2, 0], [0, 0, 1, 2, 3, 0]),
    _v("D", [-1, -1, 0, 2, 3, 2], [0, 0, 0, 1, 3, 2]),
    _v("C", [-1, 3, 2, 0, 1, 0], [0, 3, 2, 0, 1, 0]),
    _v("G", [3, 2, 0, 0, 0, 3], [2, 1, 0, 0, 0, 3]),
    _v("Em", [0, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0]),
    _v("Am", [-1, 0, 2, 2, 1, 0], [0, 0, 2, 3, 1, 0]),
    _v("Dm", [-1, -1, 0, 2, 3, 1], [0, 0, 0, 2, 3, 1]),
    _v("E7", [0, 2, 0, 1, 0, 0], [0, 2, 0, 1, 0, 0]),
    _v("A7", [-1, 0, 2, 0, 2, 0], [0, 0, 1, 0, 2, 0]),
    _v("G7", [3, 2, 0, 0, 0, 1], [3, 2, 0, 0, 0, 1]),
    _v("Cadd9", [-1, 3, 2, 0, 3, 3], [0, 2, 1, 0, 3, 4]),
    _v("B7", [-1, 2, 1, 2, 0, 2], [0, 2, 1, 3, 0, 4]),
    _v("D7", [-1, -1, 0, 2, 1, 2], [0, 0, 0, 2, 1, 3]),
    _v("Am7", [-1, 0, 2, 0, 1, 0], [0, 0, 2, 0, 1, 0]),
    _v("F", [-1, -1, 3, 2, 1, 1], [0, 0, 3, 2, 1, 1], barre=None),
    # A few more common open shapes.
    _v("Cmaj7", [-1, 3, 2, 0, 0, 0], [0, 3, 2, 0, 0, 0]),
    _v("Fmaj7", [-1, -1, 3, 2, 1, 0], [0, 0, 3, 2, 1, 0]),
    _v("Dmaj7", [-1, -1, 0, 2, 2, 2], [0, 0, 0, 1, 1, 1]),
    _v("Amaj7", [-1, 0, 2, 1, 2, 0], [0, 0, 2, 1, 3, 0]),
    _v("Em7", [0, 2, 2, 0, 3, 0], [0, 1, 2, 0, 4, 0]),
    _v("Dm7", [-1, -1, 0, 2, 1, 1], [0, 0, 0, 2, 1, 1]),
    _v("C7", [-1, 3, 2, 3, 1, 0], [0, 3, 2, 4, 1, 0]),
    _v("Dsus2", [-1, -1, 0, 2, 3, 0], [0, 0, 0, 1, 3, 0]),
    _v("Dsus4", [-1, -1, 0, 2, 3, 3], [0, 0, 0, 1, 3, 4]),
    _v("Asus2", [-1, 0, 2, 2, 0, 0], [0, 0, 1, 2, 0, 0]),
    _v("Asus4", [-1, 0, 2, 2, 3, 0], [0, 0, 1, 2, 3, 0]),
    _v("Esus4", [0, 2, 2, 2, 0, 0], [0, 2, 3, 4, 0, 0]),
    _v("Gmaj7", [3, 2, 0, 0, 0, 2], [3, 2, 0, 0, 0, 1]),
    _v("Bm7b5", [-1, 2, 3, 2, 3, -1], [0, 1, 3, 2, 4, 0]),
    _v("Am7b5", [-1, 0, 1, 0, 1, -1], [0, 0, 1, 0, 2, 0]),
    _v("Dm7b5", [-1, -1, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]),
    _v("Em7b5", [0, 1, 0, 0, 3, 0], [0, 1, 0, 0, 4, 0]),
]
for v in OPEN_VOICINGS[:16]:
    v.tags.append("beginner-chart")
# Cadd9 is played as a G-family substitute; index it under C as well.
_ALIASES = {"Cadd9": "C"}

# Movable barre shapes, relative to the barre fret b (root on 6th or 5th string).
# (quality, root string, offsets, fingers)
_BARRE_SHAPES = [
    ("", 0, [0, 2, 2, 1, 0, 0], [1, 3, 4, 2, 1, 1]),
    ("m", 0, [0, 2, 2, 0, 0, 0], [1, 3, 4, 1, 1, 1]),
    ("7", 0, [0, 2, 0, 1, 0, 0], [1, 3, 1, 2, 1, 1]),
    ("m7", 0, [0, 2, 0, 0, 0, 0], [1, 3, 1, 1, 1, 1]),
    ("maj7", 0, [0, -99, 1, 1, 0, -99], [1, 0, 3, 4, 2, 0]),
    ("", 1, [-99, 0, 2, 2, 2, 0], [0, 1, 2, 3, 4, 1]),
    ("m", 1, [-99, 0, 2, 2, 1, 0], [0, 1, 3, 4, 2, 1]),
    ("7", 1, [-99, 0, 2, 0, 2, 0], [0, 1, 3, 1, 4, 1]),
    ("m7", 1, [-99, 0, 2, 0, 1, 0], [0, 1, 3, 1, 2, 1]),
    ("maj7", 1, [-99, 0, 2, 1, 2, 0], [0, 1, 3, 2, 4, 1]),
    ("m7b5", 1, [-99, 0, 1, 0, 1, -99], [0, 1, 3, 2, 4, 0]),
    ("m7b5", 0, [0, -99, 0, 0, -1, -99], [2, 0, 3, 4, 1, 0]),
]


def barre_voicings(root: int, quality: str, tuning: list[int] = STANDARD_TUNING) -> list[Voicing]:
    out = []
    for q, root_string, offsets, fingers in _BARRE_SHAPES:
        if q != quality:
            continue
        b = (root - tuning[root_string]) % 12
        if b == 0:
            b = 12  # the open version lives in OPEN_VOICINGS
        if b > 10:
            continue
        frets = [-1 if o == -99 else b + o for o in offsets]
        fing = [0 if o == -99 else f for o, f in zip(offsets, fingers)]
        uses_barre = q not in ("maj7", "m7b5") or (q == "maj7" and root_string == 1)
        out.append(
            Voicing(chord_name(root, quality), frets, fing, barre=b if uses_barre else None, open_shape=False)
        )
    return out


def voicings_for(root: int, quality: str, tuning: list[int] = STANDARD_TUNING) -> list[Voicing]:
    """All known voicings of a chord, easiest first."""
    target = chord_pitch_classes(root, quality)
    found = []
    if tuning == STANDARD_TUNING:
        for v in OPEN_VOICINGS:
            name = _ALIASES.get(v.name, v.name)
            parsed = parse_chord(name) if name in (v.name, "C") else None
            if v.name == "Cadd9":
                continue  # alias only used when explicitly requested
            if parsed == (root % 12, quality):
                found.append(v)
    found += barre_voicings(root, quality, tuning)
    # Sanity: every sounding note must belong to the chord (or be the add9 alias).
    good = []
    for v in found:
        pcs = {p % 12 for p in v.pitches(tuning) if p is not None}
        if pcs <= target | {(root + 2) % 12}:
            good.append(v)
    return sorted(good, key=lambda v: v.difficulty())


def best_voicing(root: int, quality: str, tuning: list[int] = STANDARD_TUNING) -> Voicing | None:
    vs = voicings_for(root, quality, tuning)
    return vs[0] if vs else None


def is_open_chord(root: int, quality: str) -> bool:
    v = best_voicing(root, quality)
    return v is not None and v.open_shape


def suggest_capo(chords: list[tuple[int, str, float]], max_capo: int = 7) -> tuple[int, list[dict]]:
    """Pick the capo that makes the song easiest with open shapes.

    chords: (root, quality, duration_seconds). Returns (capo, ranking).
    The shapes played are the chord transposed DOWN by the capo amount.
    """
    ranking = []
    for capo in range(0, max_capo + 1):
        cost = 0.0
        for root, quality, dur in chords:
            q = SIMPLIFY.get(quality, quality)
            v = best_voicing((root - capo) % 12, q) or best_voicing((root - capo) % 12, quality)
            cost += dur * (v.difficulty() if v else 10.0)
        cost += 0.15 * capo * sum(d for *_, d in chords)  # mild preference for no capo
        ranking.append({"capo": capo, "cost": round(cost, 2)})
    ranking.sort(key=lambda r: r["cost"])
    return ranking[0]["capo"], ranking


# Krumhansl-Kessler key profiles.
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def key_profile_scores(chroma_sum) -> dict[tuple[int, str], float]:
    """Krumhansl-Schmuckler correlation of a 12-d pitch-class total with all 24 keys."""
    import numpy as np

    c = np.asarray(chroma_sum, dtype=float)
    c = (c - c.mean()) / (c.std() + 1e-9)
    out = {}
    for mode, prof in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
        p = np.asarray(prof)
        p = (p - p.mean()) / p.std()
        for t in range(12):
            out[(t, mode)] = float(np.dot(c, np.roll(p, t)) / 12)
    return out


def estimate_key(chroma_sum) -> tuple[int, str, float]:
    """Return (tonic, 'major'|'minor', correlation) from a 12-d chroma total."""
    import numpy as np

    c = np.asarray(chroma_sum, dtype=float)
    c = (c - c.mean()) / (c.std() + 1e-9)
    best = (0, "major", -2.0)
    for mode, prof in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
        p = np.asarray(prof)
        p = (p - p.mean()) / p.std()
        for t in range(12):
            r = float(np.dot(c, np.roll(p, t)) / 12)
            if r > best[2]:
                best = (t, mode, r)
    return best


def diatonic_chords(tonic: int, mode: str) -> set[tuple[int, str]]:
    if mode == "major":
        degrees = [(0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (9, "m"), (7, "7"), (0, "maj7"), (5, "maj7"),
                   (2, "m7"), (9, "m7"), (4, "m7"), (11, "m7b5")]
    else:
        degrees = [(0, "m"), (3, ""), (5, "m"), (7, "m"), (7, ""), (7, "7"), (8, ""), (10, ""), (0, "m7"),
                   (5, "m7"), (3, "maj7"), (8, "maj7"), (2, "m7b5")]
    return {((tonic + d) % 12, q) for d, q in degrees}
