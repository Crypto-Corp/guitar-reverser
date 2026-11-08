"""Video analysis: where is the guitar, where are the hands, which fret?

Open-source bricks:
- OWLv2 (Apache-2.0, via Hugging Face transformers) open-vocabulary detector,
  queried for the guitar, its neck, headstock and sound hole;
- MediaPipe Hand Landmarker (Apache-2.0) for 21-point hand skeletons.

The fretting-hand position along the neck is converted to a fret number with
the equal-temperament fret formula d(n) = L * (1 - 2^(-n/12)), using the nut
(headstock edge) and the sound hole as two reference points on the neck axis.
It is a *soft prior* for the fingering optimiser, never a hard decision.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
QUERIES = ["a guitar", "a guitar neck with frets", "a guitar headstock", "the sound hole of an acoustic guitar"]
LABELS = ["guitar", "neck", "headstock", "soundhole"]

# Acoustic guitar geometry (mm): scale length and nut -> sound-hole-centre distance.
# Steel-string (14-fret neck, ~645 mm) and classical (12-fret, ~650 mm) guitars
# both put the sound-hole centre roughly 400 mm from the nut.
SCALE_MM = 645.0
NUT_TO_SOUNDHOLE_MM = 405.0


@dataclass
class Shot:
    t: float
    boxes: dict  # label -> best (score, [x0, y0, x1, y1]) in pixels
    width: int
    height: int
    cands: dict = field(default_factory=dict)  # label -> up to 3 (score, box), best first

    def candidates(self, label: str, min_score: float = 0.1) -> list:
        lst = self.cands.get(label) or ([self.boxes[label]] if label in self.boxes else [])
        return [c for c in lst if c[0] >= min_score]

    @property
    def guitar_visible(self) -> bool:
        g = self.boxes.get("guitar")
        if not g or g[0] < 0.25:
            return False
        x0, y0, x1, y1 = g[1]
        return (x1 - x0) * (y1 - y0) > 0.04 * self.width * self.height


@dataclass
class HandFrame:
    t: float
    hands: list  # list of (21, 2) arrays in normalised image coords
    fret_pos: float | None = None
    fret_conf: float = 0.0
    fretting: int | None = None  # index into hands
    picking: int | None = None
    axis: list | None = None  # [nut_x, nut_y, hole_x, hole_y] normalised
    pick_xy: tuple | None = None
    extras: dict = field(default_factory=dict)


def _box_center(b):
    x0, y0, x1, y1 = b
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=float)


def _segment_point(box, toward):
    """Point where the ray from the box centre toward `toward` leaves the box."""
    c = _box_center(box)
    d = np.asarray(toward, float) - c
    if np.allclose(d, 0):
        return c
    x0, y0, x1, y1 = box
    ts = []
    for dim, lo, hi in ((0, x0, x1), (1, y0, y1)):
        if abs(d[dim]) > 1e-9:
            for edge in (lo, hi):
                t = (edge - c[dim]) / d[dim]
                if t > 0:
                    ts.append(t)
    t = min(ts) if ts else 0.0
    return c + d * min(t, 1.0)


class VideoReader:
    """Sequential frame access by timestamp (fast: decode forward only)."""

    def __init__(self, path: Path, max_width: int = 960):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.idx = -1
        self.frame = None
        self.max_width = max_width

    @property
    def duration(self):
        return self.n / self.fps

    def at(self, t: float):
        target = int(round(t * self.fps))
        if target < self.idx:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, target)
            self.idx = target - 1
        while self.idx < target:
            ok = self.cap.grab()
            if not ok:
                return None
            self.idx += 1
        ok, frame = self.cap.retrieve()
        if not ok:
            return None
        h, w = frame.shape[:2]
        if w > self.max_width:
            s = self.max_width / w
            frame = self.cv2.resize(frame, (self.max_width, int(h * s)), interpolation=self.cv2.INTER_AREA)
        return frame

    def close(self):
        self.cap.release()


def detect_shots(video: Path, step: float = 1.0, device: str | None = None, progress=None) -> list[Shot]:
    """Run OWLv2 every `step` seconds to find the guitar and its parts."""
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    proc = Owlv2Processor.from_pretrained(OWL_MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).eval().to(device)
    reader = VideoReader(video)
    shots = []
    t = 0.0
    while t < reader.duration - 0.05:
        frame = reader.at(t)
        if frame is None:
            break
        h, w = frame.shape[:2]
        img = Image.fromarray(frame[:, :, ::-1])
        inputs = proc(text=[QUERIES], images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        size = max(w, h)  # OWLv2 pads to a square
        res = proc.post_process_grounded_object_detection(
            out, threshold=0.08, target_sizes=torch.tensor([[size, size]])
        )[0]
        cands: dict[str, list] = {}
        for s, lab, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            box = [float(max(0, b[0])), float(max(0, b[1])), float(min(w, b[2])), float(min(h, b[3]))]
            cands.setdefault(LABELS[lab], []).append((float(s), box))
        for k in cands:
            cands[k] = sorted(cands[k], key=lambda c: -c[0])[:3]
        boxes = {k: v[0] for k, v in cands.items()}
        shots.append(Shot(t, boxes, w, h, cands))
        if progress:
            progress(t / reader.duration)
        t += step
    reader.close()
    return shots


def _interp_shot(shots: list[Shot], t: float) -> Shot | None:
    if not shots:
        return None
    i = min(range(len(shots)), key=lambda k: abs(shots[k].t - t))
    return shots[i]


def track_hands(video: Path, shots: list[Shot], model_path: Path, fps: float = 10.0, progress=None) -> list[HandFrame]:
    """MediaPipe hands at `fps`, only where the guitar is visible."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=4,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)
    reader = VideoReader(video)
    frames: list[HandFrame] = []
    visible = [s.t for s in shots if s.guitar_visible]
    if not visible:
        return frames
    step = 1.0 / fps
    t = 0.0
    while t < reader.duration - 0.05:
        shot = _interp_shot(shots, t)
        if shot is None or not shot.guitar_visible:
            t += step
            continue
        frame = reader.at(t)
        if frame is None:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
        hands = [np.array([[p.x, p.y] for p in lm], dtype=np.float32) for lm in res.hand_landmarks]
        frames.append(HandFrame(t=t, hands=hands))
        if progress:
            progress(t / reader.duration)
        t += step
    reader.close()
    landmarker.close()
    return frames


def _fret_from_distance(d_mm: float) -> float:
    ratio = 1.0 - d_mm / SCALE_MM
    if ratio <= 0.05:
        return 24.0
    return -12.0 * math.log2(ratio)


def analyse_geometry(frames: list[HandFrame], shots: list[Shot]) -> None:
    """Assign fretting/picking hands and estimate the fret position per frame."""
    for fr in frames:
        shot = _interp_shot(shots, fr.t)
        if shot is None or not fr.hands:
            continue
        W, H = shot.width, shot.height
        hands_px = [h * np.array([W, H]) for h in fr.hands]
        guitar = shot.boxes.get("guitar")
        if guitar is None:
            continue
        gb = guitar[1]

        def inside(pt, box, pad=0.0):
            x0, y0, x1, y1 = box
            pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
            return x0 - pw <= pt[0] <= x1 + pw and y0 - ph <= pt[1] <= y1 + ph

        # Body reference: a sound hole inside the guitar box, else the guitar box centre.
        holes = [c for c in shot.candidates("soundhole", 0.1) if inside(_box_center(c[1]), gb, 0.05)]
        hole = holes[0] if holes else None
        body = _box_center(hole[1]) if hole else _box_center(gb)
        necks = [c for c in shot.candidates("neck", 0.08) if inside(_box_center(c[1]), gb, 0.25)]
        neck = necks[0] if necks else None

        # Fretting hand: the hand whose knuckles are furthest from the body while
        # lying on the guitar/neck; picking hand: the one closest to the body.
        def knuckles(h):
            return h[[5, 9, 13, 17]].mean(axis=0)

        on_guitar = [i for i, h in enumerate(hands_px)
                     if inside(knuckles(h), gb, 0.15) or (neck and inside(knuckles(h), neck[1], 0.15))]
        if not on_guitar:
            continue
        fret_i = max(on_guitar, key=lambda i: np.linalg.norm(knuckles(hands_px[i]) - body))
        hand_vec = knuckles(hands_px[fret_i]) - body
        if np.linalg.norm(hand_vec) < 0.08 * max(W, H):
            fret_i = None
        pick_cands = [i for i in on_guitar if i != fret_i]
        pick_i = min(pick_cands, key=lambda i: np.linalg.norm(hands_px[i][0] - body)) if pick_cands else None
        fr.fretting, fr.picking = fret_i, pick_i
        if fret_i is None:
            continue
        u = hand_vec / np.linalg.norm(hand_vec)  # the neck runs from the body through the fretting hand
        hand_along = float(np.linalg.norm(hand_vec))

        # Nut: a headstock candidate that sits on the neck line beyond the hand,
        # else the far edge of the neck box (if not cut by the frame border).
        nut, nut_conf, head_c = None, 0.0, None
        for s, hb in shot.candidates("headstock", 0.1):
            hc = _box_center(hb)
            v = hc - body
            along = float(np.dot(v, u))
            perp = float(abs(v[0] * u[1] - v[1] * u[0]))
            if along > hand_along * 0.9 and perp < 0.25 * along:
                nut = _segment_point(hb, body)
                nut_conf, head_c = min(1.0, s * 3) * (1 - perp / (0.25 * along)) ** 0.5, hc
                break
        if nut is None and neck is not None:
            far = _segment_point(neck[1], body + u * 4 * max(W, H))
            x0, y0, x1, y1 = neck[1]
            at_border = min(x0, y0) < 4 or x1 > W - 4 or y1 > H - 4
            if not at_border and np.dot(far - body, u) > hand_along:
                nut, nut_conf, head_c = far, 0.35 * min(1.0, neck[0] * 3), far
        tip = head_c if head_c is not None else body + u * hand_along * 1.3
        fr.axis = [float(x) for x in (body[0] / W, body[1] / H, tip[0] / W, tip[1] / H)]

        if nut is not None:
            hole_to_nut = np.dot(nut - body, u)
            if hole_to_nut > 30:
                # index fingertip & index MCP average: where the hand sits on the neck
                p = hands_px[fret_i][[5, 6, 8]].mean(axis=0)
                d_from_nut = (hole_to_nut - np.dot(p - body, u)) / hole_to_nut * NUT_TO_SOUNDHOLE_MM
                if -40 < d_from_nut < 420:
                    fr.fret_pos = float(np.clip(_fret_from_distance(max(0.0, d_from_nut)), 0, 15))
                    fr.fret_conf = float(nut_conf * (0.6 + 0.4 * min(1.0, (hole[0] if hole else 0.1) * 3)))
        if pick_i is not None:
            perp_down = np.array([-u[1], u[0]])
            if perp_down[1] < 0:
                perp_down = -perp_down  # point toward the floor = toward the high-E string
            hp = hands_px[pick_i]
            tip = hp[[0, 5, 8, 9]].mean(axis=0)
            fr.pick_xy = (float(np.dot(tip, perp_down)), float(np.dot(tip, u)))


def smooth_positions(frames: list[HandFrame], window: float = 0.6) -> None:
    """Median-filter fret position over time to remove single-frame glitches."""
    ts = np.array([f.t for f in frames])
    vals = np.array([f.fret_pos if f.fret_pos is not None else np.nan for f in frames])
    for i, f in enumerate(frames):
        if f.fret_pos is None:
            continue
        m = (np.abs(ts - f.t) <= window / 2) & ~np.isnan(vals)
        if m.sum() >= 2:
            f.fret_pos = float(np.median(vals[m]))


def picking_velocity(frames: list[HandFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Velocity of the picking hand across the strings (+ = toward high E = down-strum)."""
    pts = [(f.t, f.pick_xy[0]) for f in frames if f.pick_xy is not None]
    if len(pts) < 3:
        return np.array([]), np.array([])
    t = np.array([p[0] for p in pts])
    x = np.array([p[1] for p in pts])
    v = np.zeros_like(x)
    for i in range(1, len(x) - 1):
        dt = t[i + 1] - t[i - 1]
        if 0 < dt < 0.35:
            v[i] = (x[i + 1] - x[i - 1]) / dt
    return t, v


def position_prior(frames: list[HandFrame]):
    """Return a function t -> (fret_pos, confidence) or (None, 0)."""
    pts = [(f.t, f.fret_pos, f.fret_conf) for f in frames if f.fret_pos is not None]
    if not pts:
        return lambda t: (None, 0.0)
    ts = np.array([p[0] for p in pts])

    def prior(t: float):
        i = int(np.argmin(np.abs(ts - t)))
        if abs(ts[i] - t) > 0.25:
            return None, 0.0
        return pts[i][1], pts[i][2]

    return prior


def export_overlay(frames: list[HandFrame]) -> list[dict]:
    """Compact per-frame hand skeletons for drawing over the video in the player."""
    out = []
    for f in frames:
        if not f.hands:
            continue
        out.append(
            {
                "t": round(f.t, 2),
                "h": [[round(float(v), 4) for v in h.flatten()] for h in f.hands],
                "f": f.fretting,
                "p": f.picking,
                "pos": None if f.fret_pos is None else round(f.fret_pos, 1),
                "ax": None if f.axis is None else [round(v, 4) for v in f.axis],
            }
        )
    return out
