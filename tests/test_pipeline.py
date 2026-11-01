"""End-to-end pipeline on a synthetic clip with a scripted fake tracker: no weights, network or GPU (INV-8)."""

import json
from pathlib import Path

import pytest

from motionman import pipeline, video_io
from motionman.tracking import DeviceError
from motionman.types import RunConfig, Track
from motionman.video_io import VideoError, count_frames, open_reader

from .conftest import FPS, HEIGHT, WIDTH, write_clip


class FakeTracker:
    """Frame 0..n: one tracked object; frame 3 empty; frame 5 adds an object without an ID."""

    instances: list["FakeTracker"] = []

    def __init__(self, config: RunConfig, fail_at: int | None = None, exc: BaseException | None = None):
        self.config = config
        self.calls = 0
        self.fail_at, self.exc = fail_at, exc
        FakeTracker.instances.append(self)

    def update(self, image):
        i = self.calls
        if i == self.fail_at:
            raise self.exc
        self.calls += 1
        if i == 3:
            return []
        tracks = [Track(1, 0, "person", 0.9, (float(i), 5.0, float(i) + 10, 20.0))]
        if i == 5:
            tracks.append(Track(None, 2, "car", 0.4, (30.0, 30.0, 40.0, 40.0)))
        return tracks


def run(config, **tracker_kwargs):
    return pipeline.run(
        config,
        make_tracker=lambda cfg: FakeTracker(cfg, **tracker_kwargs),
        resolve=lambda device: "cpu",
    )


def read_jsonl(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / pipeline.TRACKS).read_text().splitlines()]


def read_report(out: Path) -> dict:
    return json.loads((out / pipeline.REPORT).read_text())


def test_success_writes_three_agreeing_artifacts(clip, tmp_path):
    out = tmp_path / "run"
    report = run(RunConfig(clip, out))
    assert sorted(p.name for p in out.iterdir()) == ["annotated.mp4", "report.json", "tracks.jsonl"]
    records = read_jsonl(out)
    assert len(records) == 20 == report["processed_frames"] == report["output_frames_decoded"]
    assert count_frames(out / pipeline.ANNOTATED) == 20
    assert [r["frame_index"] for r in records] == list(range(20))
    assert records[4]["timestamp_s"] == pytest.approx(4 / FPS, abs=1e-6)
    assert read_report(out) == report and report["status"] == "success" and report["error"] is None


def test_empty_frames_and_missing_ids_are_recorded_not_invented(clip, tmp_path):
    out = tmp_path / "run"
    run(RunConfig(clip, out))
    records = read_jsonl(out)
    assert records[3]["objects"] == []  # empty frame still gets its record (INV-3)
    assert [o["track_id"] for o in records[5]["objects"]] == [1, None]  # null ID kept (INV-2)


def test_output_keeps_source_dimensions_and_fps(clip, tmp_path):
    out = tmp_path / "run"
    run(RunConfig(clip, out))
    cap, info = open_reader(out / pipeline.ANNOTATED)
    cap.release()
    assert (info.width, info.height, info.fps) == (WIDTH, HEIGHT, FPS)


def test_max_frames(clip, tmp_path):
    out = tmp_path / "run"
    report = run(RunConfig(clip, out, max_frames=7))
    assert report["processed_frames"] == 7 == len(read_jsonl(out)) == count_frames(out / pipeline.ANNOTATED)


def test_short_run_has_null_latency_with_reason(clip, tmp_path):
    report = run(RunConfig(clip, tmp_path / "run", max_frames=10))
    lat = report["timing"]["detect_track_ms"]
    assert lat["p50"] is None and lat["p95"] is None and "10" in lat["reason"]


def test_timing_fields_are_measured(clip, tmp_path):
    t = run(RunConfig(clip, tmp_path / "run"))["timing"]
    assert t["processing_wall_s"] > 0 and t["model_load_s"] >= 0
    # Wall time is rounded to 1 ms and fps to 0.01, and a synthetic run takes only a few ms, so check
    # that fps lies within the bounds the rounding allows rather than a fixed percentage.
    wall = t["processing_wall_s"]
    assert 20 / (wall + 0.0005) - 0.01 <= t["end_to_end_fps"] <= 20 / max(wall - 0.0005, 1e-9) + 0.01
    assert t["detect_track_ms"]["samples"] == 10  # 20 frames minus 10 warmup


def test_fresh_tracker_per_run_gets_resolved_device(clip, tmp_path):
    FakeTracker.instances.clear()
    run(RunConfig(clip, tmp_path / "a", device="auto"))
    run(RunConfig(clip, tmp_path / "b", device="auto"))
    assert len(FakeTracker.instances) == 2  # never shared across videos (INV-1)
    assert all(t.config.device == "cpu" for t in FakeTracker.instances)
    assert read_report(tmp_path / "a")["device"] == {"requested": "auto", "used": "cpu"}


def test_non_empty_output_dir_is_refused_and_untouched(clip, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "keep.txt").write_text("earlier result")
    with pytest.raises(VideoError, match="new or empty"):
        run(RunConfig(clip, out))
    assert [p.name for p in out.iterdir()] == ["keep.txt"]  # no report written into someone else's dir (INV-5)


def test_bad_source_creates_nothing(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(VideoError, match="source not found"):
        run(RunConfig(tmp_path / "nope.mp4", out))
    assert not out.exists()


def test_device_error_creates_nothing(clip, tmp_path):
    def no_cuda(device):
        raise DeviceError("CUDA is not available")

    out = tmp_path / "run"
    with pytest.raises(DeviceError):
        pipeline.run(RunConfig(clip, out, device="cuda"), make_tracker=FakeTracker, resolve=no_cuda)
    assert not out.exists()


def test_tracker_failure_mid_run_closes_files_and_reports_error(clip, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="boom"):
        run(RunConfig(clip, out), fail_at=8, exc=RuntimeError("boom"))
    report = read_report(out)
    assert report["status"] == "error" and "RuntimeError: boom" in report["error"]
    assert report["processed_frames"] == 8 == len(read_jsonl(out))
    assert count_frames(out / pipeline.ANNOTATED) == 8  # writer was released: partial video is playable
    assert report["output_frames_decoded"] is None


def test_interrupt_mid_run_reports_interrupted(clip, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run(RunConfig(clip, out), fail_at=5, exc=KeyboardInterrupt())
    report = read_report(out)
    assert report["status"] == "interrupted"
    assert report["processed_frames"] == 5 == len(read_jsonl(out)) == count_frames(out / pipeline.ANNOTATED)


def test_no_decodable_frames_is_an_error(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(video_io, "read_frames", lambda cap, fps, max_frames=None: iter(()))
    out = tmp_path / "run"
    with pytest.raises(VideoError, match="no decodable frames"):
        run(RunConfig(clip, out))
    assert read_report(out)["status"] == "error"


def test_output_frame_count_mismatch_fails_the_run(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(video_io, "count_frames", lambda path: 19)
    out = tmp_path / "run"
    with pytest.raises(VideoError, match="output check failed: wrote 20 frames but annotated.mp4 decodes to 19"):
        run(RunConfig(clip, out))
    assert read_report(out)["status"] == "error"


def test_writer_failure_reports_error_and_releases_reader(clip, tmp_path, monkeypatch):
    released = []
    real_open_reader = video_io.open_reader

    class RecordingCapture:  # cv2.VideoCapture attributes are read-only, so wrap instead of patching
        def __init__(self, cap):
            self._cap = cap

        def read(self):
            return self._cap.read()

        def release(self):
            released.append(True)
            self._cap.release()

    def tracking_reader(path):
        cap, info = real_open_reader(path)
        return RecordingCapture(cap), info

    def broken_writer(path, info):
        raise VideoError(f"cannot open video writer: {path}")

    monkeypatch.setattr(video_io, "open_reader", tracking_reader)
    monkeypatch.setattr(video_io, "open_writer", broken_writer)
    out = tmp_path / "run"
    with pytest.raises(VideoError, match="cannot open video writer"):
        run(RunConfig(clip, out))
    assert released == [True]
    assert read_report(out)["status"] == "error"
    assert not (out / pipeline.TRACKS).exists()


def test_non_standard_even_size_runs(tmp_path):
    # Real encoders round odd sizes, so odd metadata is covered by FakeCapture in test_video_io.
    out = tmp_path / "run"
    report = run(RunConfig(write_clip(tmp_path / "even.mp4", 3, width=66, height=50), out))
    assert report["source"]["width"] == 66 and report["status"] == "success"
