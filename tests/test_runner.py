import threading

import pytest

from motionman import pipeline
from motionman.pipeline import OutputCheckError
from motionman.tracking import DeviceError, ModelError
from motionman.ui import runner as runner_module
from motionman.ui.runner import Runner, RunSettings, describe_error, probe, probe_upload
from motionman.video_io import VideoError

from .test_pipeline import FakeTracker


@pytest.fixture(autouse=True)
def job_lock_released():
    """A test that leaves the process-wide job lock held would make every later test fail; say so here."""
    yield
    assert not runner_module._JOB_LOCK.locked(), "a run was left holding the job lock"


class GatedTracker(FakeTracker):
    """Blocks at frame ``gate_at`` until released, so tests can observe a run in progress."""

    def __init__(self, config, gate_at=5, fail_at=None, exc=None):
        super().__init__(config, fail_at=fail_at, exc=exc)
        self.gate_at = gate_at
        self.reached = threading.Event()
        self.release = threading.Event()

    def update(self, image):
        if self.calls == self.gate_at:
            self.reached.set()
            assert self.release.wait(10), "test never released the gate"
        return super().update(image)


class GateBox:
    """Tracker factory whose tracker blocks at ``gate_at``. Trackers are built on the worker thread, so
    tests wait on events rather than reading a list that may not be filled yet."""

    def __init__(self, gate_at, skip_first=False):
        self.gate_at, self.skip_first = gate_at, skip_first
        self._reached, self._release = threading.Event(), threading.Event()
        self._built = 0

    def factory(self, cfg):
        self._built += 1
        if self.skip_first and self._built == 1:
            return FakeTracker(cfg)
        t = GatedTracker(cfg, gate_at=self.gate_at)
        t.reached, t.release = self._reached, self._release
        return t

    def reached(self, timeout):
        return self._reached.wait(timeout)

    def open(self):
        self._release.set()


def make_runner(tmp_path, tracker=FakeTracker, resolve=lambda device: "cpu"):
    return Runner(runs_root=tmp_path / "runs" / "ui", make_tracker=tracker, resolve=resolve)


def run_to_end(runner, **start):
    assert runner.start(RunSettings(), **start)
    runner.join(30)
    return runner.snapshot()


def test_initial_state_is_idle(tmp_path):
    s = make_runner(tmp_path).snapshot()
    assert s.status == "idle" and s.report is None and not s.busy


def test_demo_source_run_completes_with_three_artifacts(clip, tmp_path):
    runner = make_runner(tmp_path)
    s = run_to_end(runner, source=clip)
    assert s.status == "complete" and s.message is None
    assert s.report["status"] == "success" and s.report["processed_frames"] == 20 == s.processed
    assert sorted(s.artifacts) == ["annotated.mp4", "report.json", "tracks.jsonl"]
    assert s.out_dir == s.run_dir / "out" and s.run_dir.parent == tmp_path / "runs" / "ui"
    assert s.device_requested == "cpu" and s.device_used == "cpu" and s.source_name == "clip.mp4"


def test_upload_is_saved_into_its_own_run_dir(clip, tmp_path):
    s = run_to_end(make_runner(tmp_path), upload=("My Clip.MP4", clip.read_bytes()))
    assert s.status == "complete" and s.source_name == "My Clip.MP4"
    assert (s.run_dir / "source.mp4").read_bytes() == clip.read_bytes()


def test_every_run_gets_a_new_directory(clip, tmp_path):
    runner = make_runner(tmp_path)
    first = run_to_end(runner, source=clip).run_dir
    second = run_to_end(runner, source=clip).run_dir
    assert first != second and first.exists() and second.exists()


def test_auto_records_the_device_actually_used(clip, tmp_path):
    runner = make_runner(tmp_path)
    assert runner.start(RunSettings(device="auto"), source=clip)
    runner.join(30)
    s = runner.snapshot()
    assert (s.device_requested, s.device_used) == ("auto", "cpu")
    assert s.report["device"] == {"requested": "auto", "used": "cpu"}


def test_progress_is_visible_while_processing_and_runs_are_exclusive(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PROGRESS_INTERVAL_S", 0.0)
    gate = GateBox(gate_at=5)
    runner = make_runner(tmp_path, tracker=gate.factory)
    other = make_runner(tmp_path / "other")
    assert runner.start(RunSettings(), source=clip)
    try:
        assert gate.reached(10)
        s = runner.snapshot()
        assert s.busy and s.status == "processing" and s.processed == 5 and s.total == 20
        assert s.report is None  # nothing from a finished run is shown while processing
        assert not runner.start(RunSettings(), source=clip)  # duplicate submission refused
        assert not other.start(RunSettings(), source=clip)  # one job per process, across sessions
    finally:
        gate.open()
        runner.join(30)
    assert runner.snapshot().status == "complete"
    assert other.start(RunSettings(), source=clip)  # the lock is free again
    other.join(30)


def test_starting_a_new_run_clears_the_previous_result(clip, tmp_path):
    gate = GateBox(gate_at=0, skip_first=True)  # the second run blocks so we can look at its state
    runner = make_runner(tmp_path, tracker=gate.factory)
    assert run_to_end(runner, source=clip).report is not None
    assert runner.start(RunSettings(), source=clip)
    try:
        assert gate.reached(10)
        s = runner.snapshot()
        assert s.status == "processing" and s.report is None and s.artifacts == {} and s.processed == 0
    finally:
        gate.open()
        runner.join(30)


def test_invalid_upload_gives_message_and_next_step(tmp_path):
    runner = make_runner(tmp_path)
    s = run_to_end(runner, upload=("notes.mp4", b"not a video"))
    assert s.status == "error" and "cannot open video" in s.message
    assert s.next_step.startswith("Choose another video")
    assert s.report is None and s.artifacts == {}
    assert runner.start(RunSettings(), upload=("again.mp4", b"x"))  # lock released after an error
    runner.join(30)


def test_device_error_suggests_cpu(clip, tmp_path):
    def no_cuda(device):
        raise DeviceError("--device cuda requested but CUDA is not available on this machine")

    s = run_to_end(make_runner(tmp_path, resolve=no_cuda), source=clip)
    assert s.status == "error" and "CUDA is not available" in s.message
    assert s.next_step == "Choose CPU or Auto and run again."


def test_mid_run_failure_keeps_partial_artifacts_and_report(clip, tmp_path):
    def failing(cfg):
        return FakeTracker(cfg, fail_at=5, exc=RuntimeError("boom"))

    s = run_to_end(make_runner(tmp_path, tracker=failing), source=clip)
    assert s.status == "error" and s.message == "Unexpected error: RuntimeError: boom"
    assert s.report["status"] == "error" and s.report["processed_frames"] == 5  # this run's partial report
    assert set(s.artifacts) == {"annotated.mp4", "report.json", "tracks.jsonl"}


@pytest.mark.parametrize(
    ("exc", "next_step"),
    [
        (DeviceError("x"), "Choose CPU or Auto"),
        (ModelError("x"), "Check that yolo11n.pt"),
        (OutputCheckError("x"), "Run again."),
        (VideoError("cannot open video writer: out.mp4"), "Check free disk space"),
        (VideoError("odd dimensions 639x479"), "Choose another video"),
        (RuntimeError("x"), "See the terminal"),
    ],
)
def test_describe_error_always_gives_a_next_step(exc, next_step):
    message, step = describe_error(exc)
    assert message and step.startswith(next_step)


def test_start_needs_exactly_one_source(clip, tmp_path):
    runner = make_runner(tmp_path)
    with pytest.raises(ValueError):
        runner.start(RunSettings())
    with pytest.raises(ValueError):
        runner.start(RunSettings(), source=clip, upload=("a.mp4", b""))


def test_live_elapsed_moves_before_first_progress_report(clip, tmp_path):
    gate = GateBox(gate_at=0)
    runner = make_runner(tmp_path, tracker=gate.factory)
    assert runner.start(RunSettings(), source=clip)
    try:
        assert gate.reached(10)
        s = runner.snapshot()
        assert s.processed == 0 and s.elapsed_s == 0.0
        assert s.live_elapsed_s(now=s.started_at + 3.0) == pytest.approx(3.0)
    finally:
        gate.open()
        runner.join(30)
    done = runner.snapshot()
    assert done.live_elapsed_s(now=done.started_at + 999) == done.elapsed_s  # frozen once finished


def test_probe_reports_playback_metadata(clip):
    info = probe(clip)
    assert (info.name, info.width, info.height, info.fps, info.frame_count) == ("clip.mp4", 64, 48, 30.0, 20)
    assert info.duration_s == pytest.approx(20 / 30)


def test_probe_upload_uses_original_name_and_leaves_no_file(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module.tempfile, "tempdir", str(tmp_path))
    info = probe_upload("Street.MP4", clip.read_bytes())
    assert info.name == "Street.MP4" and info.frame_count == 20
    assert not any(p.name.startswith("motionman-probe-") for p in tmp_path.iterdir())


def test_probe_upload_rejects_non_video():
    with pytest.raises(VideoError, match="cannot open video"):
        probe_upload("notes.mp4", b"not a video")
