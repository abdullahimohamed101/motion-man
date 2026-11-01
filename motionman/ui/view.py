"""What the page shows for a given JobState. Pure functions, no Streamlit, so the rules are unit-tested.

Spec §9: values come from this run's report only; "Pending" before completion, never fake zeroes;
an unavailable latency is "N/A"; state is never shown by colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from motionman.ui.runner import JobState, SourceInfo

PENDING = "Pending"
NOT_AVAILABLE = "N/A"
NO_RESULT = "—"  # error state: there are no results to show, and a partial run's numbers are not results


@dataclass(frozen=True)
class Badge:
    label: str  # always shown: the text carries the state, colour only reinforces it
    color: str
    icon: str


BADGES = {
    "idle": Badge("Idle", "gray", ":material/radio_button_unchecked:"),
    "processing": Badge("Processing", "gray", ":material/progress_activity:"),
    "complete": Badge("Complete", "green", ":material/check_circle:"),
    "error": Badge("Error", "red", ":material/error:"),
}


@dataclass(frozen=True)
class Metric:
    label: str
    value: str
    help: str | None = None


def metrics(state: JobState) -> list[Metric]:
    labels = ("Processing FPS", "Detect+track p95", "Frames processed", "Processing duration")
    report = state.report if state.status == "complete" else None
    if report is None:
        filler = NO_RESULT if state.status == "error" else PENDING
        return [Metric(label, filler) for label in labels]
    t = report["timing"]
    latency = t["detect_track_ms"]
    p95 = NOT_AVAILABLE if latency["p95"] is None else f"{latency['p95']:.1f} ms"
    return [
        Metric(labels[0], _num(t["end_to_end_fps"], "{:.2f}"), "Processed frames ÷ processing wall time, end to end."),
        Metric(labels[1], p95, latency.get("reason") or "model.track incl. ByteTrack, first 10 frames excluded."),
        Metric(labels[2], f"{report['processed_frames']:,}"),
        Metric(labels[3], _num(t["processing_wall_s"], "{:.2f} s"), "Model load not included."),
    ]


def progress(state: JobState, now: float | None = None) -> tuple[float | None, str]:
    """(fraction or None when no usable total, status line). Percentages only with a known total."""
    elapsed = f"{state.live_elapsed_s(now):.1f} s elapsed"
    if state.processed == 0:
        return (0.0 if state.total else None), f"Loading model and starting… · {elapsed}"
    if state.total:
        fraction = min(state.processed / state.total, 1.0)
        return fraction, f"{state.processed:,} / {state.total:,} frames · {fraction:.0%} · {elapsed}"
    return None, f"{state.processed:,} frames · {elapsed}"


def device_label(state: JobState, selected: str) -> str:
    if state.device_used:
        return f"Device: {state.device_used.upper()}"
    return f"Device: {selected} (selected)"


def source_lines(info: SourceInfo) -> list[str]:
    duration = f"{info.duration_s:.1f} s" if info.duration_s else "unknown duration"
    return [info.name, f"{info.width}×{info.height} · {duration}", f"{info.fps:.3f} fps playback"]


def run_details(state: JobState) -> list[tuple[str, str]]:
    r = state.report
    if not r:
        return []
    m, env = r["model"], r["environment"]
    pkgs = ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v)
    return [
        ("Status", r["status"]),
        (
            "Source",
            f"{state.source_name} ({r['source']['width']}×{r['source']['height']}, {r['source']['fps']:.3f} fps)",
        ),
        ("Model", f"{m['weights']} · imgsz {m['imgsz']} · {m['precision'].upper()} · {m['tracker']}"),
        ("Thresholds", f"confidence {m['conf']} · IoU {m['iou']}"),
        ("Device", f"requested {r['device']['requested']} · used {r['device']['used']}"),
        ("Machine", f"{env['cpu']} · {env['platform']} · Python {env['python']}"),
        ("Packages", pkgs),
        ("Model load", _num(r["timing"]["model_load_s"], "{:.2f} s")),
        ("Run folder", str(state.run_dir)),
    ]


def _num(value: float | None, fmt: str) -> str:
    return NOT_AVAILABLE if value is None else fmt.format(value)
