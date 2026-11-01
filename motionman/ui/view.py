"""What the page shows for a given JobState. Pure functions, no Streamlit, so the rules are unit-tested.

Spec §9: values come from this run's report only; "Pending" before completion, never fake zeroes;
an unavailable latency is "N/A"; state is never shown by colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from motionman.ui.runner import JobState, SourceInfo

PENDING = "Pending"
NOT_AVAILABLE = "N/A"
NO_RESULT = "—"  # error state: there are no results to show, and a partial run's numbers are not results
NOT_RECORDED = "Not recorded in this run"  # runs made before stage timings existed (plan P2 D12)
STAGE_LABELS = {"decode_ms": "Decode", "detect_track_ms": "Detect + track", "render_ms": "Render", "write_ms": "Write"}


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


METRIC_LABELS = ("Processing FPS", "Detect+track p95", "Frames processed", "Processing duration")


def metrics(state: JobState) -> list[Metric]:
    report = state.report if state.status == "complete" else None
    if report is None:
        filler = NO_RESULT if state.status == "error" else PENDING
        return [Metric(label, filler) for label in METRIC_LABELS]
    return report_metrics(report)


def report_metrics(report: dict) -> list[Metric]:
    """The four headline numbers of one run's report.json; missing values read N/A, never 0."""
    labels = METRIC_LABELS
    t = report["timing"]
    latency = t["detect_track_ms"]
    p95 = NOT_AVAILABLE if latency["p95"] is None else f"{latency['p95']:.1f} ms"
    return [
        Metric(labels[0], _num(t["end_to_end_fps"], "{:.2f}"), "Processed frames ÷ processing wall time, end to end."),
        Metric(labels[1], p95, latency.get("reason") or "model.track incl. ByteTrack, first 10 frames excluded."),
        Metric(labels[2], f"{report['processed_frames']:,}"),
        Metric(labels[3], _num(t["processing_wall_s"], "{:.2f} s"), "Model load not included."),
    ]


@dataclass(frozen=True)
class StageRow:
    label: str
    p50: str
    p95: str
    share: float | None  # fraction of processing wall time over every frame, warmup included
    scope: str | None = None


def stage_breakdown(report: dict) -> list[StageRow] | None:
    """Per-stage p50/p95 and share of wall time, plus the unattributed remainder; None for runs before P2."""
    t = report.get("timing") or {}
    stages, wall = t.get("stages"), t.get("processing_wall_s")
    if not stages:
        return None
    scopes = t.get("stage_scopes") or {}
    rows = [
        StageRow(
            label,
            _num(stages[key]["p50"], "{:.2f} ms"),
            _num(stages[key]["p95"], "{:.2f} ms"),
            stages[key]["total_s"] / wall if wall else None,
            scopes.get(key),
        )  # fmt: skip
        for key, label in STAGE_LABELS.items()
    ]
    u = t.get("unattributed")
    rows.append(StageRow("Unattributed", NO_RESULT, NO_RESULT, u["share_of_wall"] if u else None,
                         u["scope"] if u else None))  # fmt: skip
    return rows


def phase_metrics(report: dict) -> list[Metric]:
    """Setup, processing, finalize and the whole job; runs before P2 say so instead of showing zeros."""
    t = report.get("timing") or {}
    recorded = "job_wall_s" in t

    def metric(label: str, key: str, scope: str | None) -> Metric:
        if recorded or key == "processing_wall_s":
            return Metric(label, _num(t.get(key), "{:.2f} s"), scope)
        return Metric(label, NO_RESULT, NOT_RECORDED)  # a dash fits the card; the help says why

    return [
        metric("Setup", "setup_s", t.get("setup_scope")),
        metric("Processing", "processing_wall_s", t.get("processing_wall_scope")),
        metric("Finalize", "finalize_s", t.get("finalize_scope")),
        metric("Job total", "job_wall_s", "Setup + processing + finalize."),
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
    if not state.report:
        return []
    return report_details(state.report, state.source_name, state.run_dir)


def report_details(r: dict, source_name: str | None, run_dir) -> list[tuple[str, str]]:
    """Settings, versions and hardware of one run, for the collapsed Run details."""
    m, env = r["model"], r["environment"]
    pkgs = ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v)
    return [
        ("Status", r["status"]),
        (
            "Source",
            f"{source_name} ({r['source']['width']}×{r['source']['height']}, {r['source']['fps']:.3f} fps)",
        ),
        ("Model", f"{m['weights']} · imgsz {m['imgsz']} · {m['precision'].upper()} · {m['tracker']}"),
        ("Thresholds", f"confidence {m['conf']} · IoU {m['iou']}"),
        ("Device", f"requested {r['device']['requested']} · used {r['device']['used']}"),
        ("Machine", f"{env['cpu']} · {env['platform']} · Python {env['python']}"),
        ("Packages", pkgs),
        ("Model load", _num(r["timing"]["model_load_s"], "{:.2f} s")),
        ("Run folder", str(run_dir)),
    ]


def when_label(iso: str | None, now: datetime | None = None) -> str | None:
    """A human time for a run's start, in local time: "Today, 7:45 PM", "Yesterday, 7:45 PM", "Mon, 7:45 PM"
    inside the last week, "1 Nov, 7:45 PM" this year, else "1 Nov 2025". None when the time is unknown."""
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = now or datetime.now().astimezone()
    if t.tzinfo is not None:
        t = t.astimezone(now.tzinfo)
    clock = t.strftime("%I:%M %p").lstrip("0")
    days = (now.date() - t.date()).days
    if days == 0:
        day = "Today"
    elif days == 1:
        day = "Yesterday"
    elif 1 < days < 7:
        day = t.strftime("%a")
    elif t.year == now.year:
        day = f"{t.day} {t.strftime('%b')}"
    else:
        return f"{t.day} {t.strftime('%b %Y')}"
    return f"{day}, {clock}"


def duration_label(frames: int, fps: float) -> str:
    """A clip length from its frame count: "56 s" under two minutes, else "4 min 12 s"."""
    seconds = frames / fps if fps else 0.0
    if seconds < 120:
        return f"{seconds:.0f} s"
    whole = timedelta(seconds=round(seconds))
    minutes, secs = divmod(int(whole.total_seconds()), 60)
    return f"{minutes} min {secs} s"


def run_row(display_name: str | None, started_at: str | None, frames: int | None, legacy: bool,
            now: datetime | None = None) -> str:  # fmt: skip
    """One Library row: clip name, human time and size; legacy runs say so instead of claiming a frame count."""
    tail = "Legacy (unverified)" if legacy else (f"{frames:,} frames" if frames else None)
    return " · ".join(x for x in (display_name or "Unknown source", when_label(started_at, now), tail) if x)


def _num(value: float | None, fmt: str) -> str:
    return NOT_AVAILABLE if value is None else fmt.format(value)
