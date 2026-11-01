"""Constant-frame-rate verification and resource limits (spec A§6, plan P0 step 3)."""

import shutil
import subprocess

import numpy as np
import pytest

from motionman import pipeline, video_io
from motionman.types import Limits, RunConfig
from motionman.video_io import VideoError, open_reader, read_frames, timestamp_drift_frames

from .conftest import write_clip
from .test_pipeline import FakeTracker


class TimedCapture:
    """Fake cv2.VideoCapture with explicit per-frame timestamps (ms) and metadata."""

    def __init__(self, times_ms, fps=30.0, width=64, height=48, count=None):
        self.times, self.i = list(times_ms), -1
        cv2 = video_io.cv2
        self.props = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: len(self.times) if count is None else count,
        }

    def isOpened(self):  # noqa: N802 - mirrors the OpenCV API
        return True

    def grab(self):
        if self.i + 1 >= len(self.times):
            return False
        self.i += 1
        return True

    def read(self):
        ok = self.grab()
        return ok, (np.zeros((48, 64, 3), np.uint8) if ok else None)

    def get(self, prop):
        if prop == video_io.cv2.CAP_PROP_POS_MSEC:
            return self.times[self.i]
        return self.props[prop]

    def release(self):
        pass


def install(monkeypatch, tmp_path, make):
    """Every VideoCapture(path) gets a fresh fake (open_reader opens twice: preflight, then the run)."""
    monkeypatch.setattr(video_io.cv2, "VideoCapture", lambda path: make())
    path = tmp_path / "x.mp4"
    path.write_bytes(b"x")
    return path


def cfr(n, fps=30.0):
    return [i * 1000.0 / fps for i in range(n)]


def test_drift_is_measured_in_frames():
    assert timestamp_drift_frames(10, 10 * 1000 / 30, 30.0) == pytest.approx(0.0)
    assert timestamp_drift_frames(10, 10 * 1000 / 30 + 16.7, 30.0) == pytest.approx(0.5, abs=0.01)


def test_constant_rate_passes_and_reader_starts_at_frame_zero(monkeypatch, tmp_path):
    path = install(monkeypatch, tmp_path, lambda: TimedCapture(cfr(300)))
    cap, info = open_reader(path)
    frames = list(read_frames(cap, info.fps))
    assert len(frames) == 300 and frames[0].index == 0  # preflight consumed a different capture


def test_jitter_within_half_a_frame_is_accepted(monkeypatch, tmp_path):
    jittered = [t + (10.0 if i % 2 else -10.0) for i, t in enumerate(cfr(100))]  # ±0.3 frame at 30 fps
    path = install(monkeypatch, tmp_path, lambda: TimedCapture(jittered))
    cap, info = open_reader(path)
    assert len(list(read_frames(cap, info.fps))) == 100


def test_variable_rate_is_rejected_in_preflight_before_any_run(monkeypatch, tmp_path):
    vfr = cfr(30) + [1000.0 + i * 66.7 for i in range(1, 60)]  # 30 fps, then 15 fps
    path = install(monkeypatch, tmp_path, lambda: TimedCapture(vfr))
    with pytest.raises(VideoError, match="unsupported timebase: frame 30 "):
        open_reader(path)


def test_drift_after_the_preflight_window_fails_the_run(monkeypatch, tmp_path):
    late = cfr(260) + [260 * 1000 / 30 + 200.0]  # constant for 260 frames, then a 6-frame jump
    path = install(monkeypatch, tmp_path, lambda: TimedCapture(late))
    cap, info = open_reader(path)  # first 240 frames look constant
    with pytest.raises(VideoError, match="frame 260"):
        list(read_frames(cap, info.fps))


def test_missing_timestamps_cannot_pass_as_constant_rate(monkeypatch, tmp_path):
    path = install(monkeypatch, tmp_path, lambda: TimedCapture([0.0] * 50))
    with pytest.raises(VideoError, match="unsupported timebase: frame 1"):
        open_reader(path)


@pytest.mark.parametrize(
    ("limits", "capture", "message"),
    [
        (Limits(max_long_side=32), {"width": 64, "height": 48}, "64x48 exceeds the 32x1080 limit"),
        (Limits(max_short_side=40), {"width": 64, "height": 48}, "exceeds the 1920x40 limit"),
        (Limits(max_frames=100), {"count": 101}, "101 frames exceeds the 100-frame limit"),
        (Limits(max_duration_s=1.0), {"count": 60}, "video is 0.0 min long; the limit is 0.0166667 min"),
    ],
)
def test_limits_are_enforced_before_inference(monkeypatch, tmp_path, limits, capture, message):
    path = install(monkeypatch, tmp_path, lambda: TimedCapture(cfr(60), **capture))
    with pytest.raises(VideoError, match=message):
        open_reader(path, limits)


def test_file_size_limit(tmp_path):
    path = write_clip(tmp_path / "c.mp4", 5)
    with pytest.raises(VideoError, match="the limit is 0 MiB"):
        open_reader(path, Limits(max_file_bytes=10))


def test_runtime_frame_limit_is_an_error_not_truncation(monkeypatch, tmp_path):
    path = install(monkeypatch, tmp_path, lambda: TimedCapture(cfr(20), count=0))  # container gives no count
    cap, info = open_reader(path, Limits(max_frames=10))
    with pytest.raises(VideoError, match="exceeds the 10-frame limit"):
        list(read_frames(cap, info.fps, frame_limit=10))


def test_requested_max_frames_is_not_a_limit_breach(monkeypatch, tmp_path):
    path = install(monkeypatch, tmp_path, lambda: TimedCapture(cfr(20)))
    cap, info = open_reader(path)
    assert len(list(read_frames(cap, info.fps, max_frames=10, frame_limit=10))) == 10


def test_over_limit_source_creates_no_output(tmp_path):
    out = tmp_path / "run"
    clip = write_clip(tmp_path / "c.mp4", 20)
    config = RunConfig(clip, out, limits=Limits(max_frames=5))
    with pytest.raises(VideoError, match="20 frames exceeds the 5-frame limit"):
        pipeline.run(config, make_tracker=FakeTracker, resolve=lambda d: "cpu")
    assert not out.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed; real VFR clip not generated")
def test_real_variable_rate_clip_is_rejected(tmp_path):
    path = tmp_path / "vfr.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", "-t", "3", "-vf",
         "setpts='if(lt(N,30),N/30/TB,(30+(N-30)*2)/30/TB)'", "-fps_mode", "vfr", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=60,
    )  # fmt: skip
    with pytest.raises(VideoError, match="unsupported timebase"):
        open_reader(path)
