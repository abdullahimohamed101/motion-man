"""Draw annotations on a copy of the frame (spec §6, §9 track overlays)."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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


# --- Inspector stills (plan P1 D2): drawn from saved records, with view options; never used for the MP4 -----

SPOTLIGHT_DIM = 0.35  # brightness kept outside the spotlighted object
MAX_TRAIL_LENGTH = 60


@dataclass(frozen=True)
class ViewOptions:
    """Presentation only: changing any of these re-renders, never re-runs the model (UI2 settings contract)."""

    labels: bool = True
    trail_length: int = 30  # frames of observed history behind each visible object; 0 hides trails
    trail_opacity: float = 1.0
    spotlight: int | None = None  # a track ID to emphasise; others are dimmed and unlabelled
    classes: tuple[str, ...] | None = None  # only these classes; None shows every class

    def shows(self, track: Track) -> bool:
        return self.classes is None or track.class_name in self.classes


def annotate_still(
    image: np.ndarray,
    tracks: Sequence[Track],
    trails: Mapping[int, Sequence[tuple[int, int]]],
    options: ViewOptions = ViewOptions(),  # noqa: B008 - frozen, immutable default
) -> np.ndarray:
    """Draw one inspected source frame. ``trails`` maps track ID → observed centres, oldest first."""
    out = image.copy()
    shown = [t for t in tracks if options.shows(t)]
    scale = max(0.4, min(out.shape[:2]) / 1000)
    thickness = max(2, round(scale * 3))
    focus = options.spotlight
    if focus is not None:
        lit = out.copy()
        out = (out * SPOTLIGHT_DIM).astype(np.uint8)
        for t in shown:
            if t.track_id == focus:
                x1, y1, x2, y2 = (round(v) for v in t.bbox_xyxy)
                out[y1:y2, x1:x2] = lit[y1:y2, x1:x2]
    if options.trail_length > 0 and options.trail_opacity > 0:
        layer = out.copy()
        for t in shown:
            if t.track_id is None or (focus is not None and t.track_id != focus):
                continue
            pts = list(trails.get(t.track_id, ()))[-options.trail_length :]
            if len(pts) >= 2:
                cv2.polylines(layer, [np.array(pts, np.int32)], False, color_for(t.track_id), thickness, cv2.LINE_AA)
        out = cv2.addWeighted(layer, options.trail_opacity, out, 1 - options.trail_opacity, 0)
    for t in shown:
        x1, y1, x2, y2 = (round(v) for v in t.bbox_xyxy)
        if focus is not None and t.track_id != focus:
            cv2.rectangle(out, (x1, y1), (x2, y2), NO_ID_COLOR, 1)
            continue
        color = color_for(t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if options.labels:
            prefix = "#?" if t.track_id is None else f"#{t.track_id}"
            _label(out, f"{prefix} {t.class_name} {t.confidence:.2f}", (x1, y1), color, scale, thickness)
    return out


REGION_COLOR = (224, 214, 208)  # mist: saved regions
PENDING_COLOR = (34, 242, 228)  # acid lime, as a signal stroke: the region being edited


def draw_regions(image: np.ndarray, regions: Sequence, pending=None) -> np.ndarray:
    """Draw region outlines on a copy of a source frame (plan P3 D12). Lines get an arrowhead at B and a "+" on
    their positive side (left when facing A→B on screen); ``pending`` is drawn in the signal colour."""
    img = image.copy()
    h, w = img.shape[:2]
    scale = max(1, round(w / 960))
    for region, color in [*((r, REGION_COLOR) for r in regions), *(((pending, PENDING_COLOR),) if pending else ())]:
        c = [round(v) for v in region.to_px(w, h)]
        if region.kind == "line":
            ax, ay, bx, by = c
            length = max(1.0, ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
            tip = min(0.3, 24 * scale / length)  # arrowhead about 24 px, whatever the line's length
            cv2.arrowedLine(img, (ax, ay), (bx, by), color, 2 * scale, cv2.LINE_AA, tipLength=tip)
            nx, ny = (by - ay) / length, -(bx - ax) / length  # left normal on screen (image y points down)
            mx, my = (ax + bx) / 2, (ay + by) / 2
            plus = (round(mx + nx * 18 * scale), round(my + ny * 18 * scale))
            cv2.putText(img, "+", (plus[0] - 6 * scale, plus[1] + 6 * scale), _FONT, 0.7 * scale, color,
                        2 * scale, cv2.LINE_AA)  # fmt: skip
            anchor = (min(ax, bx), min(ay, by))
        else:
            x1, y1, x2, y2 = c
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 * scale, cv2.LINE_AA)
            anchor = (x1, y1)
        _label(img, region.name, anchor, color, 0.5 * scale, scale)
    return img


ARROW_HORIZON_S = 0.5  # an arrow shows where the centre would be after this long at its current velocity
MIN_ARROW_PX_S = 10.0  # slower than this is drawn as no arrow: box jitter on parked objects is not motion


def draw_motion(image: np.ndarray, motions: Mapping) -> np.ndarray:
    """Velocity arrows on a copy of an inspected still, labelled in px/s (image space, no real-world units)."""
    img = image.copy()
    scale = max(1, round(img.shape[1] / 960))
    for m in motions.values():
        start = (round(m.x), round(m.y))
        end = (round(m.x + m.vx * ARROW_HORIZON_S), round(m.y + m.vy * ARROW_HORIZON_S))
        if m.speed < MIN_ARROW_PX_S or start == end:
            continue
        cv2.arrowedLine(img, start, end, PENDING_COLOR, 2 * scale, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(img, f"{m.speed:.0f} px/s", (end[0] + 4 * scale, end[1]), _FONT, 0.45 * scale, PENDING_COLOR,
                    scale, cv2.LINE_AA)  # fmt: skip
    return img


HEAT_BLUR_CELLS = 2.0  # display smoothing (Gaussian sigma, in grid cells); the saved cells are never smoothed
HEAT_SCALE_PERCENTILE = 95  # the colour scale saturates here, so a few static hotspots can't hide everything else


def heatmap_overlay(image: np.ndarray, heat: dict, opacity: float) -> np.ndarray:
    """Blend an analysis heatmap over a copy of a frame. Display only: cells are lightly smoothed, the colour scale
    is a square root that saturates at the 95th percentile of the occupied area, and transparency follows
    intensity, so quiet areas fade out instead of darkening the frame. The saved object-µs are never changed; an
    all-zero map returns the frame unchanged."""
    img = image.copy()
    cols, rows = heat["grid"]
    if not heat["cells"] or opacity <= 0:
        return img
    grid = np.zeros((rows, cols), np.float64)
    for c, r, v in heat["cells"]:
        grid[r, c] = v
    smooth = cv2.GaussianBlur(grid, (0, 0), HEAT_BLUR_CELLS)
    # Scale to a high percentile of the occupied area, not its maximum: a parked car or a traffic light can sit in
    # one cell for the whole clip and would otherwise wash out every path people actually walked.
    occupied = smooth[smooth > smooth.max() * 1e-3]
    top = float(np.percentile(occupied, HEAT_SCALE_PERCENTILE)) or float(smooth.max())
    level = np.sqrt(np.clip(smooth / top, 0, 1))
    h, w = img.shape[:2]
    big = cv2.resize(level, (w, h), interpolation=cv2.INTER_LINEAR).clip(0, 1)
    colours = cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float64)
    alpha = (opacity * big)[..., None]
    return (img * (1 - alpha) + colours * alpha).round().astype(np.uint8)
