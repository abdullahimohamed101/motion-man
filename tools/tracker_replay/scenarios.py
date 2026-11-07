"""Deterministic detection scenarios for tracker replay (plan native-g2 step 2, NA§8 test list).

Each scenario yields per-frame detections ``[x1, y1, x2, y2, score, cls]`` on a 1280x720 canvas. Values are
quantised to 1/16 px and scores to 1/256 so they are exact in float32: the Python reference and the native
tracker read identical numbers. Classes use the COCO ids the app sees (0 person, 2 car).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

PERSON, CAR = 0, 2


def q(v: float) -> float:
    return round(v * 16) / 16


def s(v: float) -> float:
    return round(v * 256) / 256


def box(cx: float, cy: float, w: float, h: float, score: float, cls: int) -> list[float]:
    return [q(cx - w / 2), q(cy - h / 2), q(cx + w / 2), q(cy + h / 2), s(score), float(cls)]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    build: Callable[[], list[list[list[float]]]]
    instances: int = 1

    def frames(self) -> list[list[list[float]]]:
        return self.build()


def _walker(
    frames: int,
    start: tuple[float, float],
    step: tuple[float, float],
    size=(60, 160),
    score=0.9,
    cls=PERSON,
    visible: Callable[[int], bool] = lambda f: True,
    score_at: Callable[[int], float] | None = None,
):
    """One object moving linearly; returns per-frame lists (empty where not visible)."""
    out = []
    for f in range(frames):
        if not visible(f):
            out.append([])
            continue
        cx, cy = start[0] + step[0] * f, start[1] + step[1] * f
        out.append([box(cx, cy, size[0], size[1], score_at(f) if score_at else score, cls)])
    return out


def _merge(*tracks: list[list[list[float]]]) -> list[list[list[float]]]:
    return [[d for t in tracks for d in t[f]] for f in range(len(tracks[0]))]


def entry_exit():
    return _walker(30, (100, 360), (12, 0), visible=lambda f: 3 <= f <= 20)


def reappear_within_buffer():
    # Hidden for 10 frames (< track_buffer 30): the lost track is found again with the same ID.
    return _walker(40, (100, 360), (10, 0), visible=lambda f: not 12 <= f < 22)


def reappear_after_buffer():
    # Hidden for 40 frames (> track_buffer): the old track is removed and a new ID starts.
    return _walker(70, (100, 360), (4, 0), visible=lambda f: not 10 <= f < 50)


def crossing():
    a = _walker(40, (200, 360), (15, 0))
    b = _walker(40, (800, 380), (-15, 0))
    return _merge(a, b)


def overlap():
    # Two people walking side by side with boxes overlapping by about half their width.
    a = _walker(30, (300, 360), (8, 1), size=(60, 160))
    b = _walker(30, (330, 362), (8, 1), size=(60, 160), score=0.8)
    return _merge(a, b)


def low_conf_recovery():
    # Score dips into the second-stage band (0.1 < s < 0.25) for 6 frames: kept by the low-score association.
    return _walker(30, (100, 360), (10, 0), score_at=lambda f: 0.15 if 10 <= f < 16 else 0.85)


def low_conf_below_floor():
    # Score below track_low_thresh for 6 frames: the track is lost, then recovered when the score returns.
    return _walker(30, (100, 360), (10, 0), score_at=lambda f: 0.05 if 10 <= f < 16 else 0.85)


def empty_frames():
    a = _walker(30, (100, 300), (10, 0), visible=lambda f: f % 7 != 3)
    b = _walker(30, (900, 400), (-6, 2), size=(120, 80), cls=CAR, visible=lambda f: f % 7 != 3)
    return _merge(a, b)


def class_conflict():
    # A car track, then a person detection at the same place; the agnostic reference re-labels the track.
    car = _walker(20, (400, 400), (5, 0), size=(120, 80), cls=CAR, visible=lambda f: f < 10)
    person = _walker(20, (400, 400), (5, 0), size=(120, 80), cls=PERSON, visible=lambda f: f >= 10)
    return _merge(car, person)


def assignment_ties():
    # Two identical detections at the same place in every frame: assignment has an exact tie each time.
    a = _walker(15, (500, 360), (6, 0))
    b = _walker(15, (500, 360), (6, 0))
    return _merge(a, b)


def new_track_threshold():
    # 0.20 is a low-band score: never starts a track. 0.30 starts one each frame but, with fuse_score, its
    # unconfirmed match cost 1 - IoU*score stays above 0.7, so it is never confirmed and burns a new ID every
    # frame. A flickering detection starts unconfirmed tracks that die. A steady person from frame 6 shows the
    # IDs those consumed (later starts are unconfirmed until matched on the next frame).
    weak = _walker(16, (200, 200), (5, 0), score=0.20)
    late = _walker(16, (800, 500), (-5, 0), score=0.30, visible=lambda f: f >= 4)
    flicker = _walker(16, (600, 150), (0, 0), score=0.9, visible=lambda f: f in (5, 9))
    steady = _walker(16, (1000, 300), (-4, 0), score=0.9, visible=lambda f: f >= 6)
    return _merge(weak, late, flicker, steady)


def degenerate_boxes():
    a = _walker(10, (300, 300), (5, 0))
    zero = [[[q(600), q(300), q(600), q(400), s(0.9), float(PERSON)]] for _ in range(10)]  # zero width: dropped
    return _merge(a, zero)


def two_instances():
    return crossing()


def crowd():
    """Eight walkers with noise, misses, score changes and false positives: a seeded stress case."""
    rng = np.random.default_rng(20261107)
    frames: list[list[list[float]]] = [[] for _ in range(80)]
    for k in range(8):
        cx, cy = rng.uniform(100, 1180), rng.uniform(150, 600)
        vx, vy = rng.uniform(-9, 9), rng.uniform(-3, 3)
        w, h = rng.uniform(40, 90), rng.uniform(110, 200)
        cls = CAR if k % 4 == 3 else PERSON
        for f in range(80):
            x, y = cx + vx * f + rng.normal(0, 1.5), cy + vy * f + rng.normal(0, 1.5)
            if not (w / 2 < x < 1280 - w / 2 and h / 2 < y < 720 - h / 2) or rng.random() < 0.08:
                continue
            score = float(np.clip(rng.normal(0.7, 0.2), 0.02, 0.99))
            frames[f].append(box(x, y, w * rng.uniform(0.95, 1.05), h * rng.uniform(0.95, 1.05), score, cls))
    for f in range(80):
        if rng.random() < 0.15:
            frames[f].append(box(rng.uniform(50, 1230), rng.uniform(50, 670), 50, 50, rng.uniform(0.1, 0.4), PERSON))
    return frames


ALL = [
    Scenario("entry_exit", "one person enters, walks and leaves", entry_exit),
    Scenario("reappear_within_buffer", "hidden 10 frames, found again with the same ID", reappear_within_buffer),
    Scenario("reappear_after_buffer", "hidden 40 frames, past the buffer: a new ID", reappear_after_buffer),
    Scenario("crossing", "two people cross paths", crossing),
    Scenario("overlap", "two people walk side by side, boxes half overlapping", overlap),
    Scenario("low_conf_recovery", "score dips into the low band; second-stage association keeps it",
             low_conf_recovery),
    Scenario("low_conf_below_floor", "score drops below track_low_thresh, then returns", low_conf_below_floor),
    Scenario("empty_frames", "every 7th frame has no detections", empty_frames),
    Scenario("class_conflict", "a person detection lands on a car track (agnostic reference)", class_conflict),
    Scenario("assignment_ties", "two identical detections each frame: exact assignment ties", assignment_ties),
    Scenario("new_track_threshold", "low-band, late and flickering starts; unconfirmed tracks",
             new_track_threshold),
    Scenario("degenerate_boxes", "a zero-width detection is dropped before tracking", degenerate_boxes),
    Scenario("two_instances", "two trackers fed the same input give the same IDs", two_instances, instances=2),
    Scenario("crowd", "eight seeded walkers with noise, misses and false positives", crowd),
]  # fmt: skip
