"""Draw annotations on a copy of the frame (spec §6, §9 track overlays)."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from motionman.types import Track

# BGR. Lime, cyan, amber, violet; colour repeats, so the ID label carries identity.
PALETTE = ((53, 230, 163), (238, 211, 34), (36, 191, 251), (250, 139, 167))
NO_ID_COLOR = (170, 161, 161)  # neutral grey for objects the tracker has not assigned an ID
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(track_id: int | None) -> tuple[int, int, int]:
    return NO_ID_COLOR if track_id is None else PALETTE[track_id % len(PALETTE)]


def annotate(image: np.ndarray, tracks: Sequence[Track]) -> np.ndarray:
    out = image.copy()
    scale = max(0.4, min(out.shape[:2]) / 1000)  # keep labels readable at any resolution
    thickness = max(2, round(scale * 3))
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
