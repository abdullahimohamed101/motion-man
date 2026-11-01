from datetime import UTC
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


def p2_report():
    stage = {"p50": 2.0, "p95": 3.0, "samples": 10, "warmup_excluded": 10, "total_s": 0.5}
    return {
        "timing": {
            "processing_wall_s": 10.0, "setup_s": 3.0, "finalize_s": 0.8, "job_wall_s": 13.8,
            "stages": {"decode_ms": stage, "detect_track_ms": {**stage, "total_s": 8.9},
                       "render_ms": {**stage, "p50": None, "p95": None, "reason": "too few"}, "write_ms": stage},
            "stage_scopes": {"decode_ms": "inside next()"},
            "unattributed": {"total_s": 0.1, "share_of_wall": 0.01, "scope": "loop work"},
        }
    }  # fmt: skip


def test_stage_breakdown_shows_shares_and_na_never_zero():
    rows = view.stage_breakdown(p2_report())
    assert [r.label for r in rows] == ["Decode", "Detect + track", "Render", "Write", "Unattributed"]
    assert rows[1].share == pytest.approx(0.89) and rows[0].p50 == "2.00 ms" and rows[0].scope == "inside next()"
    assert rows[2].p50 == view.NOT_AVAILABLE and rows[4].share == 0.01


def test_phase_metrics_for_new_and_old_runs():
    assert [m.value for m in view.phase_metrics(p2_report())] == ["3.00 s", "10.00 s", "0.80 s", "13.80 s"]
    old = {"timing": {"processing_wall_s": 30.1, "end_to_end_fps": 45.0}}  # a report from before P2
    assert view.stage_breakdown(old) is None
    phases = view.phase_metrics(old)
    assert [m.value for m in phases] == [view.NO_RESULT, "30.10 s", view.NO_RESULT, view.NO_RESULT]
    assert phases[0].help == phases[3].help == view.NOT_RECORDED  # never a zero


def test_when_label_reads_like_a_person_wrote_it():
    from datetime import datetime

    now = datetime(2026, 11, 1, 19, 50, tzinfo=UTC)
    assert view.when_label("2026-11-01T19:45:56.123Z", now) == "Today, 7:45 PM"
    assert view.when_label("2026-10-31T09:05:00+00:00", now) == "Yesterday, 9:05 AM"
    assert view.when_label("2026-10-27T23:59:00Z", now) == "Tue, 11:59 PM"
    assert view.when_label("2026-03-04T08:00:00Z", now) == "4 Mar, 8:00 AM"
    assert view.when_label("2025-12-25T08:00:00Z", now) == "25 Dec 2025"
    assert view.when_label(None, now) is None and view.when_label("not a date", now) is None


def test_duration_and_library_rows_never_show_ids():
    from datetime import datetime

    now = datetime(2026, 11, 1, 19, 50, tzinfo=UTC)
    assert view.duration_label(1356, 23.976) == "57 s"
    assert view.duration_label(6000, 23.976) == "4 min 10 s"
    assert view.duration_label(10, 0) == "0 s"
    row = view.run_row("crosswalk.mp4", "2026-11-01T19:45:56Z", 1356, False, now)
    assert row == "crosswalk.mp4 · Today, 7:45 PM · 1,356 frames"
    assert view.run_row("demo.mp4", None, 240, True, now) == "demo.mp4 · Legacy (unverified)"
    assert view.run_row(None, None, None, False, now) == "Unknown source"


# --- Events & zones (plan P3 step 6) ---

EVENTS = [
    {"event_id": "e_1", "type": "LINE_CROSSING", "rule_id": "door", "track_id": 3, "class_name": "person",
     "direction": "NEGATIVE_TO_POSITIVE", "confirming_frame": 30, "confirming_t_us": 1_000_000, "value": None},
    {"event_id": "e_2", "type": "DWELL_THRESHOLD", "rule_id": "pen", "track_id": 4, "class_name": "car",
     "direction": None, "confirming_frame": 90, "confirming_t_us": 3_000_000, "value": 2_500_000},
    {"event_id": "e_3", "type": "OCCUPANCY_THRESHOLD", "rule_id": "pen", "track_id": None, "class_name": None,
     "direction": None, "confirming_frame": 120, "confirming_t_us": 4_000_000, "value": 3},
]  # fmt: skip


def test_event_rows_speak_plain_language():
    rows = [view.event_row(e, {"door": "Door", "pen": "Pen"}) for e in EVENTS]
    assert rows[0] == {"Time (s)": 1.0, "Event": "Line crossing", "Region": "Door", "Track": "#3",
                       "Class": "person", "Detail": "towards +", "Frame": 30}  # fmt: skip
    assert rows[1]["Detail"] == "2.5 s inside" and rows[2]["Detail"] == "3 inside at once"
    assert rows[2]["Track"] == "—" and rows[2]["Class"] == "—"


def test_event_filters_combine_and_default_to_everything():
    keep = lambda f: [e["event_id"] for e in EVENTS if f.keeps(e)]  # noqa: E731
    assert keep(view.EventFilter()) == ["e_1", "e_2", "e_3"]
    assert keep(view.EventFilter(rules=("pen",))) == ["e_2", "e_3"]
    assert keep(view.EventFilter(rules=("pen",), types=("DWELL_THRESHOLD",))) == ["e_2"]
    assert keep(view.EventFilter(classes=("person",))) == ["e_1"]
    assert keep(view.EventFilter(track=4)) == ["e_2"]
    assert keep(view.EventFilter(window_s=(1.0, 3.0))) == ["e_1", "e_2"]


def test_region_cards_show_totals_and_never_divide_by_zero():
    line = {"kind": "line", "name": "Door", "crossings_total": 3,
            "crossings": {"NEGATIVE_TO_POSITIVE": 2, "POSITIVE_TO_NEGATIVE": 1}}  # fmt: skip
    assert view.region_card(line) == ("Door", "3 crossings", "2 towards + · 1 away from +")
    zone = {"kind": "rect", "name": "Pen", "entries": 4, "exits": 2, "peak_occupancy": 3, "occupancy_us": 500_000,
            "covered_us": 1_000_000, "dwell_median_us": 1_500_000}  # fmt: skip
    assert view.region_card(zone) == ("Pen", "4 entered", "2 left · peak 3 at once · mean 0.50 · median stay 1.5 s")
    empty = {**zone, "entries": 0, "exits": 0, "peak_occupancy": 0, "occupancy_us": 0, "covered_us": 0,
             "dwell_median_us": None}  # fmt: skip
    assert view.region_card(empty)[2] == "0 left · peak 0 at once · mean — · no stays"
