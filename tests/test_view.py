from pathlib import Path

import pytest

from motionman.ui import view
from motionman.ui.runner import JobState, SourceInfo


def report(p95=12.34, reason=None, fps=31.27, wall=43.36, frames=1356):
    return {
        "status": "success",
        "processed_frames": frames,
        "source": {"width": 1920, "height": 1080, "fps": 23.976},
        "model": {
            "weights": "yolo11n.pt",
            "imgsz": 640,
            "precision": "fp32",
            "tracker": "bytetrack.yaml",
            "conf": 0.25,
            "iou": 0.7,
        },
        "device": {"requested": "auto", "used": "cpu"},
        "environment": {
            "cpu": "Apple M5 Pro",
            "platform": "macOS",
            "python": "3.11.17",
            "packages": {"torch": "2.15.0", "lap": None},
        },
        "timing": {
            "end_to_end_fps": fps,
            "processing_wall_s": wall,
            "model_load_s": 3.9,
            "detect_track_ms": {"p50": 10.0, "p95": p95, "reason": reason},
        },
    }


@pytest.mark.parametrize("status", ["idle", "processing"])
def test_metrics_pending_before_completion(status):
    values = [m.value for m in view.metrics(JobState(status=status))]
    assert values == ["Pending"] * 4  # never fake zeroes


def test_metrics_after_error_show_no_results_even_with_a_partial_report():
    values = [m.value for m in view.metrics(JobState(status="error", report=report()))]
    assert values == ["—"] * 4  # a failed run's partial numbers are not presented as results


def test_metrics_come_from_the_report():
    values = [m.value for m in view.metrics(JobState(status="complete", report=report()))]
    assert values == ["31.27", "12.3 ms", "1,356", "43.36 s"]


def test_unavailable_latency_is_na_with_reason_not_zero():
    reason = "10 frames processed; percentiles need more than 10 (warmup excluded)"
    m = view.metrics(JobState(status="complete", report=report(p95=None, reason=reason)))[1]
    assert m.value == "N/A" and m.help == reason


def test_progress_with_known_total_shows_percentage():
    s = JobState(status="processing", processed=339, total=1356, elapsed_s=12.0, started_at=100.0)
    fraction, line = view.progress(s, now=112.5)
    assert fraction == pytest.approx(0.25)
    assert line == "339 / 1,356 frames · 25% · 12.5 s elapsed"


def test_progress_without_total_shows_count_only():
    fraction, line = view.progress(JobState(status="processing", processed=40, total=None, elapsed_s=2.0))
    assert fraction is None and "%" not in line and line.startswith("40 frames")


def test_progress_during_model_load():
    s = JobState(status="processing", processed=0, total=None, started_at=10.0)
    fraction, line = view.progress(s, now=13.0)
    assert fraction is None and line == "Loading model and starting… · 3.0 s elapsed"


def test_device_label_prefers_the_device_actually_used():
    assert view.device_label(JobState(device_used="cpu"), "Auto") == "Device: CPU"
    assert view.device_label(JobState(), "Auto") == "Device: Auto (selected)"


def test_every_state_has_a_text_label():
    assert {b.label for b in view.BADGES.values()} == {"Idle", "Processing", "Complete", "Error"}
    assert view.BADGES["complete"].color == "green" and view.BADGES["error"].color == "red"


def test_source_lines_label_playback_fps_separately():
    info = SourceInfo("crosswalk.mp4", 1920, 1080, 23.976, 1356, 56.56)
    assert view.source_lines(info) == ["crosswalk.mp4", "1920×1080 · 56.6 s", "23.976 fps playback"]


def test_run_details_only_for_a_report():
    assert view.run_details(JobState()) == []
    details = dict(view.run_details(JobState(status="complete", report=report(), run_dir=Path("runs/ui/x"))))
    assert details["Device"] == "requested auto · used cpu"
    assert details["Packages"] == "torch 2.15.0"  # missing packages are left out
