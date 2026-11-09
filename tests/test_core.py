"""Fast tests for the music logic (no audio/video models needed)."""

from reverser import theory
from reverser.audio import ChordSeg, Note
from reverser.fingering import assign_fingering, assign_picking, group_events


def test_beginner_chart_voicings_sound_their_chord():
    for v in theory.OPEN_VOICINGS:
        if v.name == "Cadd9":
            continue
        root, quality = theory.parse_chord(v.name)
        pcs = {p % 12 for p in v.pitches() if p is not None}
        assert pcs <= theory.chord_pitch_classes(root, quality), v.name


def test_barre_shapes_cover_every_root():
    for root in range(12):
        for q in ("", "m", "7", "m7"):
            assert theory.best_voicing(root, q) is not None, theory.chord_name(root, q)


def test_capo_suggestion_prefers_open_shapes():
    # Eb - Bb - Cm - Ab is all barre chords; capo 3 turns it into C - G - Am - F.
    song = [(3, "", 4.0), (10, "", 4.0), (0, "m", 4.0), (8, "", 4.0)]
    capo, _ = theory.suggest_capo(song)
    assert capo in (1, 3)


def test_key_estimation_from_c_major_profile():
    chroma = [5, 0, 3, 0, 4, 3, 0, 5, 0, 3, 0, 1]
    tonic, mode, _ = theory.estimate_key(chroma)
    assert (tonic, mode) == (0, "major")


def _chord_notes(frets, t=0.0, dur=1.0):
    return [Note(t + 0.005 * s, t + dur, theory.STANDARD_TUNING[s] + f, 0.8) for s, f in enumerate(frets) if f >= 0]


def test_strummed_c_chord_gets_the_open_c_shape():
    notes = _chord_notes([-1, 3, 2, 0, 1, 0])
    events = group_events(notes)
    chords = [ChordSeg(0.0, 2.0, "C", 0, "", 1.0)]
    assign_fingering(events, chords)
    got = {n.string: n.fret for n in events[0].notes}
    assert got == {1: 3, 2: 2, 3: 0, 4: 1, 5: 0}
    fingers = {n.string: n.finger for n in events[0].notes}
    assert fingers == {1: 3, 2: 2, 3: 0, 4: 1, 5: 0}


def test_arpeggio_is_fingerpicked_with_thumb_on_bass():
    # A-minor arpeggio: A string, then G, B, e one by one
    pitches = [45, 57, 60, 64]
    notes = [Note(0.4 * i, 0.4 * i + 0.35, p, 0.7) for i, p in enumerate(pitches)]
    events = group_events(notes)
    chords = [ChordSeg(0.0, 2.0, "Am", 9, "m", 1.0)]
    assign_fingering(events, chords)
    assign_picking(events, lambda t: t / 0.5)
    plucks = [events[i].notes[0].pluck for i in range(4)]
    assert plucks[0] == "p"
    assert all(e.kind == "pluck" for e in events)
    # stays in first position, on the Am shape
    assert all(n.fret <= 2 for e in events for n in e.notes)


def test_strum_direction_from_string_order():
    down = [Note(0.01 * s, 0.5, theory.STANDARD_TUNING[s] + f, 0.8) for s, f in enumerate([0, 2, 2, 1, 0, 0])]
    events = group_events(down, window=0.08)
    assign_fingering(events, [ChordSeg(0, 1, "E", 4, "", 1.0)])
    assign_picking(events, lambda t: 0.0)
    assert events[0].kind == "strum" and events[0].direction == "D"


def test_replucked_notes_are_not_merged_but_decay_fragments_are():
    from reverser.audio import clean_notes

    # same pitch re-strummed right after itself, similar strength: two real notes
    notes = [Note(0.0, 0.50, 60, 0.8), Note(0.51, 1.0, 60, 0.7)]
    kept, _ = clean_notes(notes)
    assert len(kept) == 2
    # a much weaker fragment glued to the end of a note is a decay artefact
    notes = [Note(0.0, 0.50, 60, 0.8), Note(0.51, 1.0, 60, 0.3)]
    kept, _ = clean_notes(notes)
    assert len(kept) == 1 and kept[0].end == 1.0
