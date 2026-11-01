import json
from pathlib import Path

import pytest

from motionman.metrics import Metrics, build_report, latency_summary, write_report
from motionman.types import RunConfig, VideoInfo


def test_latency_excludes_warmup_frames():
    samples = [1000.0] * 10 + [10.0, 20.0, 30.0, 40.0, 50.0]  # slow warmup must not leak in
    s = latency_summary(samples, warmup=10)
    assert s["samples"] == 5 and s["warmup_excluded"] == 10
    assert s["p50"] == 30.0
    assert s["p95"] == pytest.approx(48.0)  # linear interpolation between 40 and 50


@pytest.mark.parametrize("n", [0, 1, 10])
def test_latency_is_null_with_reason_for_ten_or_fewer_frames(n):
    s = latency_summary([5.0] * n, warmup=10)
    assert s["p50"] is None and s["p95"] is None and s["samples"] == 0
    assert str(n) in s["reason"]


def test_latency_with_eleven_frames_uses_one_sample():
    s = latency_summary([99.0] * 10 + [7.0], warmup=10)
    assert s["p50"] == 7.0 and s["p95"] == 7.0 and s["samples"] == 1


def test_end_to_end_fps_is_frames_over_wall_time():
    m = Metrics()
    m.processed_frames, m.processing_wall_s = 90, 4.5
    assert m.end_to_end_fps == 20.0


def test_end_to_end_fps_is_none_without_frames_or_time():
    m = Metrics()
    assert m.end_to_end_fps is None
    m.processed_frames = 10
    assert m.end_to_end_fps is None  # never started


def test_stop_is_idempotent():
    m = Metrics()
    m.start()
    m.stop()
    first = m.processing_wall_s
    m.stop()
    assert m.processing_wall_s == first is not None


def _report(m: Metrics, status="success", error=None):
    cfg = RunConfig(Path("in.mp4"), Path("out"), max_frames=None)
    info = VideoInfo(Path("in.mp4"), 640, 480, 30.0, 90)
    return build_report(status=status, error=error, config=cfg, info=info, metrics=m, device_used="cpu")


def test_report_records_settings_and_timing():
    m = Metrics()
    m.processed_frames, m.processing_wall_s, m.model_load_s = 15, 1.5, 0.25
    m.detect_track_ms = [100.0] * 10 + [20.0] * 5
    r = _report(m)
    assert r["status"] == "success" and r["error"] is None
    assert r["processed_frames"] == 15
    assert r["source"] == {
        "path": "in.mp4",
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "frame_count_reported_by_container": 90,
    }
    assert r["model"] == {
        "weights": "yolo11n.pt",
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "tracker": "bytetrack.yaml",
        "precision": "fp32",
    }
    assert r["timing"]["end_to_end_fps"] == 10.0
    assert r["timing"]["model_load_s"] == 0.25
    assert r["timing"]["detect_track_ms"]["p50"] == 20.0
    assert "not inference-only" in r["timing"]["detect_track_scope"]
    assert r["environment"]["packages"]["numpy"]
    json.dumps(r)


def test_partial_report_keeps_status_and_null_timing():
    r = _report(Metrics(), status="interrupted", error="interrupted by user")
    assert r["status"] == "interrupted" and r["error"] == "interrupted by user"
    assert r["timing"]["end_to_end_fps"] is None and r["timing"]["processing_wall_s"] is None


def test_write_report_leaves_no_temp_file(tmp_path):
    path = tmp_path / "report.json"
    write_report(path, {"status": "success"})
    assert json.loads(path.read_text()) == {"status": "success"}
    assert list(tmp_path.iterdir()) == [path]
