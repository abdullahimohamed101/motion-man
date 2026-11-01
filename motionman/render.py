"""Draw annotations on a copy of the frame (spec §6, §9 track overlays)."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import cv2
import numpy as np

from motionman.types import Track

# BGR. Lime, cyan, amber, violet; colour repeats, so the ID label carries identity.
PALETTE = ((53, 230, 163), (238, 211, 34), (36, 191, 251), (250, 139, 167))
NO_ID_COLOR = (170, 161, 161)  # neutral grey for objects the tracker has not assigned an ID
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Trails:
    """Recent box centres per track ID, bounded in length and age (INV-4).

    An ID is dropped once it has gone ``max_age`` consecutive frames unobserved, so a briefly
    occluded object resumes its trail but a departed one does not linger in memory.
    """

    def __init__(self, max_points: int = 30, max_age: int = 60) -> None:
        self.max_points = max_points
        self.max_age = max_age
        self._points: dict[int, deque[tuple[int, int]]] = {}
        self._last_seen: dict[int, int] = {}

    def update(self, frame_index: int, tracks: Sequence[Track]) -> None:
        for t in tracks:
            if t.track_id is None:  # no ID, no trail (spec §6)
                continue
            x1, y1, x2, y2 = t.bbox_xyxy
            pts = self._points.setdefault(t.track_id, deque(maxlen=self.max_points))
            pts.append((round((x1 + x2) / 2), round((y1 + y2) / 2)))
            self._last_seen[t.track_id] = frame_index
        for tid in [tid for tid, seen in self._last_seen.items() if frame_index - seen >= self.max_age]:
            del self._points[tid], self._last_seen[tid]

    def points(self, track_id: int) -> list[tuple[int, int]]:
        return list(self._points.get(track_id, ()))

    def __len__(self) -> int:
        return len(self._points)


def color_for(track_id: int | None) -> tuple[int, int, int]:
    return NO_ID_COLOR if track_id is None else PALETTE[track_id % len(PALETTE)]


def annotate(image: np.ndarray, tracks: Sequence[Track], trails: Trails | None = None) -> np.ndarray:
    """Draw trails, boxes and labels on a copy. Trails are drawn only for objects visible this frame."""
    out = image.copy()
    scale = max(0.4, min(out.shape[:2]) / 1000)  # keep labels readable at any resolution
    thickness = max(2, round(scale * 3))
    if trails is not None:
        for t in tracks:
            pts = trails.points(t.track_id) if t.track_id is not None else []
            if len(pts) >= 2:
                cv2.polylines(out, [np.array(pts, np.int32)], False, color_for(t.track_id), thickness, cv2.LINE_AA)
    for t in tracks:
        color = color_for(t.track_id)
        x1, y1, x2, y2 = (round(v) for v in t.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        prefix = "#?" if t.track_id is None else f"#{t.track_id}"
        _label(out, f"{prefix} {t.class_name} {t.confidence:.2f}", (x1, y1), color, scale, thickness)
    return out


def _label(
    img: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float, thickness: int
) -> None:
    (w, h), base = cv2.getTextSize(text, _FONT, scale, thickness)
    x, y = origin
    top = y - h - base - 4 if y - h - base - 4 >= 0 else y  # flip below the box edge at the top of the frame
    cv2.rectangle(img, (x, top), (x + w + 4, top + h + base + 4), color, cv2.FILLED)
    cv2.putText(img, text, (x + 2, top + h + 2), _FONT, scale, (5, 17, 11), thickness, cv2.LINE_AA)
