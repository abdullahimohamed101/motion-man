"""Page-level tests with Streamlit's AppTest: real script, fake runner, synthetic clip, no browser or model."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from motionman.ui import runner as runner_module
from motionman.ui.runner import JobState, RunSettings

APP = str(Path(__file__).resolve().parents[1] / "motionman" / "ui" / "app.py")


class FakeRunner:
    def __init__(self):
        self.state = JobState()
        self.starts: list[tuple[RunSettings, dict]] = []
        self.accept = True

    def snapshot(self):
        return self.state

    def start(self, settings, **source):
        self.starts.append((settings, source))
        return self.accept


@pytest.fixture
def fake(monkeypatch, clip):
    fake = FakeRunner()
    monkeypatch.setattr(runner_module, "Runner", lambda: fake)
    monkeypatch.setenv("MOTIONMAN_DEMO_CLIP", str(clip))
    st.cache_resource.clear()  # get_runner() is cached per process
    yield fake
    st.cache_resource.clear()


def render(fake=None, state=None):
    if state is not None:
        fake.state = state
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def badge(at):
    return next(m.value for m in at.markdown if "-badge[" in m.value)


def run_button(at):
    return next(b for b in at.button if b.label == "Run tracking")


def report(p95=30.4, reason=None):
    return {
        "status": "success",
        "processed_frames": 1356,
        "source": {"width": 1920, "height": 1080, "fps": 23.976},
        "model": {"weights": "yolo11n.pt", "imgsz": 640, "precision": "fp32", "tracker": "bytetrack.yaml"}
        | {"conf": 0.25, "iou": 0.7},
        "device": {"requested": "cpu", "used": "cpu"},
        "environment": {"cpu": "Apple M5 Pro", "platform": "macOS", "python": "3.11.17", "packages": {"torch": "2"}},
        "timing": {
            "end_to_end_fps": 32.14,
            "processing_wall_s": 42.19,
            "model_load_s": 4.1,
            "detect_track_ms": {"p50": 27.1, "p95": p95, "reason": reason},
        },
    }


def complete_state(tmp_path, **report_kwargs):
    files = {}
    for name in ("annotated.mp4", "tracks.jsonl", "report.json"):
        files[name] = tmp_path / name
        files[name].write_bytes(b"x")
    return JobState(
        status="complete",
        source_name="crosswalk.mp4",
        device_requested="cpu",
        device_used="cpu",
        processed=1356,
        report=report(**report_kwargs),
        artifacts=files,
        run_dir=tmp_path,
    )


def test_idle_page_shows_pending_and_source_metadata(fake):
    at = render()
    assert [m.value for m in at.metric] == ["Pending"] * 4
    assert "Idle" in badge(at) and "gray-badge" in badge(at)
    assert not run_button(at).disabled
    captions = [c.value for c in at.caption]
    assert "Device: CPU (selected)" in captions
    assert any("30.000 fps playback" in c and "64×48" in c for c in captions)  # playback FPS labelled apart
    assert all(d.proto.disabled for d in at.get("download_button"))  # nothing to download yet


def test_missing_demo_clip_disables_run_with_a_hint(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("MOTIONMAN_DEMO_CLIP", str(tmp_path / "nope.mp4"))
    at = render()
    assert run_button(at).disabled
    assert any("Demo clip not found" in c.value for c in at.caption)


def test_run_tracking_starts_the_runner_with_the_chosen_settings(fake, clip):
    at = render()
    at.segmented_control(key="device").set_value("CUDA")
    at.slider(key="conf").set_value(0.4)
    run_button(at).click()
    at.run()
    assert fake.starts == [(RunSettings(device="cuda", conf=0.4), {"source": clip})]


def test_refused_start_says_a_run_is_in_progress(fake):
    fake.accept = False
    at = render()
    run_button(at).click()
    at.run()
    assert any("already in progress" in w.value for w in at.warning)


def test_processing_disables_run_and_shows_progress(fake):
    state = JobState(status="processing", source_name="clip.mp4", device_used="cpu", processed=339, total=1356)
    at = render(fake, state)
    assert run_button(at).disabled  # no duplicate submissions
    assert "Processing" in badge(at)
    assert [m.value for m in at.metric] == ["Pending"] * 4  # no numbers until the run completes
    assert "Device: CPU" in [c.value for c in at.caption]


def test_complete_page_shows_this_runs_report(fake, tmp_path):
    at = render(fake, complete_state(tmp_path))
    assert "Complete" in badge(at) and "green-badge" in badge(at)
    assert [m.value for m in at.metric] == ["32.14", "30.4 ms", "1,356", "42.19 s"]
    assert any(m.value == "#### Metrics · crosswalk.mp4" for m in at.markdown)
    assert not any(d.proto.disabled for d in at.get("download_button"))
    assert not run_button(at).disabled


def test_unavailable_latency_shows_na_not_zero(fake, tmp_path):
    at = render(fake, complete_state(tmp_path, p95=None, reason="10 frames processed"))
    assert at.metric[1].value == "N/A"


def test_error_page_gives_next_step_and_drops_old_numbers(fake):
    state = JobState(
        status="error",
        source_name="clip.mp4",
        message="--device cuda requested but CUDA is not available on this machine",
        next_step="Choose CPU or Auto and run again.",
        report=report(),  # even with a partial report, its numbers are not shown as results
    )
    at = render(fake, state)
    assert "Error" in badge(at) and "Complete" not in badge(at)
    assert at.error[0].value.startswith("--device cuda requested")
    assert any(m.value == "**Next step:** Choose CPU or Auto and run again." for m in at.markdown)
    assert [m.value for m in at.metric] == ["—"] * 4
    assert not run_button(at).disabled  # the user can fix the setting and retry


def test_complete_then_failed_run_shows_no_stale_results(fake, tmp_path):
    at = render(fake, complete_state(tmp_path))
    assert "Complete" in badge(at)
    fake.state = JobState(status="error", message="boom", next_step="Try again.")  # the next run fails
    at.run()
    assert not at.exception
    assert "Error" in badge(at) and not any("Complete" in m.value for m in at.markdown)
    assert [m.value for m in at.metric] == ["—"] * 4
    assert all(d.proto.disabled for d in at.get("download_button"))  # no downloads from the earlier run
