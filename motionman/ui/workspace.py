"""Pure helpers for the result workspace (plan P1 step 6). No Streamlit, so the rules are unit-tested.

Per-run context (D9): each opened run keeps its own inspected frame, selected object and view options. Values
are validated against the run they're restored into, so an ID from one run never selects something in another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from motionman.inspection import TrackSummary
from motionman.render import MAX_TRAIL_LENGTH, ViewOptions


@dataclass
class RunContext:
    frame: int = 0
    track: int | None = None
    view: dict = field(default_factory=lambda: _view_to_dict(ViewOptions()))
    spotlight: bool = True  # spotlight the selected object; the ID itself lives in ``track``


def restore(saved: RunContext | None, frame_count: int, tracks: dict[int, TrackSummary]) -> RunContext:
    """A context for this run: the saved one, with anything that doesn't exist in this run dropped."""
    ctx = saved or RunContext()
    frame = ctx.frame if 0 <= ctx.frame < frame_count else 0
    track = ctx.track if ctx.track in tracks else None
    view = _view_to_dict(view_options(ctx.view))
    return RunContext(frame, track, view, ctx.spotlight)


def view_options(view: dict, *, spotlight: int | None = None) -> ViewOptions:
    """ViewOptions from stored widget values, clamped to valid ranges."""
    length = min(max(int(view.get("trail_length", 30)), 0), MAX_TRAIL_LENGTH)
    opacity = min(max(float(view.get("trail_opacity", 1.0)), 0.0), 1.0)
    classes = view.get("classes")
    return ViewOptions(
        labels=bool(view.get("labels", True)),
        trail_length=length,
        trail_opacity=opacity,
        spotlight=spotlight,
        classes=tuple(classes) if classes else None,
    )


def _view_to_dict(v: ViewOptions) -> dict:
    d = asdict(v)
    d.pop("spotlight")
    d["classes"] = list(v.classes) if v.classes else None
    return d


def format_t(t_us: int) -> str:
    return f"{t_us / 1_000_000:.2f} s"


def frame_caption(index: int, t_us: int, count: int) -> str:
    return f"Source frame {index:,} of {count - 1:,} · {format_t(t_us)}"


def track_order(tracks: dict[int, TrackSummary]) -> list[int]:
    """Longest-observed first, so the selector opens on the objects worth looking at."""
    return sorted(tracks, key=lambda t: (-tracks[t].observations, t))


def track_label(summary: TrackSummary) -> str:
    return f"#{summary.track_id} {summary.class_name} · {summary.observations:,} frames"


def intervals_text(summary: TrackSummary, fps: float, limit: int = 4) -> str:
    """Observed intervals as source times; the span alone would suggest continuous presence."""
    shown = [f"{a / fps:.1f}–{b / fps:.1f} s" for a, b in summary.intervals[:limit]]
    more = len(summary.intervals) - limit
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


def contact_frames(summary: TrackSummary, n: int = 6) -> list[int]:
    """Up to ``n`` frames where the object was actually observed, spread across its observations.
    Samples never fall in a gap: a contact sheet must not fabricate visibility (MD§11)."""
    observed = [f for a, b in summary.intervals for f in range(a, b + 1)]
    if len(observed) <= n:
        return observed
    step = (len(observed) - 1) / (n - 1)
    return [observed[round(i * step)] for i in range(n)]


def crop_box(bbox: list[float], width: int, height: int, pad: float = 0.25) -> tuple[int, int, int, int]:
    """A padded, in-bounds crop around a box, for contact sheets."""
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (max(0, int(x1 - px)), max(0, int(y1 - py)), min(width, int(x2 + px) + 1), min(height, int(y2 + py) + 1))
