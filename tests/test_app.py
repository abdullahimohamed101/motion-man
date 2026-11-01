"""The workbench page with Streamlit's AppTest (plan P1 step 6): real published runs on disk, a fake runner,
and model entry points that raise if anything tries to run inference."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from motionman import pipeline, presets, store, tracking
from motionman.presets import Preset, TrackingSettings
from motionman.render import ViewOptions
from motionman.types import RunConfig, Track
from motionman.ui import runner as runner_module
from motionman.ui.runner import JobState, RunSettings

from .test_pipeline import FakeTracker

APP = str(Path(__file__).resolve().parents[1] / "motionman" / "ui" / "app.py")


class OtherTracker(FakeTracker):
    """Run B: a single object with ID 2, never ID 1."""

    def update(self, image):
        super().update(image)
        return [Track(2, 2, "car", 0.8, (30.0, 20.0, 50.0, 40.0))]


class FakeRunner:
    def __init__(self, root):
        self.root = root
        self.state = JobState()
        self.starts: list[tuple[RunSettings, dict]] = []
        self.allow_start = False

    def snapshot(self):
        return self.state

    def start(self, settings, **source):
        if not self.allow_start:
            raise AssertionError("inspection must never start a run")
        self.starts.append((settings, source))
        return True


def no_model(*args, **kwargs):
    raise AssertionError("inspection must never load or call the model")


@pytest.fixture
def world(monkeypatch, clip, tmp_path):
    root = tmp_path / "data"
    a = pipeline.run(RunConfig(clip, None, data_root=root), make_tracker=FakeTracker, resolve=lambda d: "cpu")
    b = pipeline.run(RunConfig(clip, None, data_root=root), make_tracker=OtherTracker, resolve=lambda d: "cpu")
    fake = FakeRunner(store.data_root(root))
    monkeypatch.setattr(runner_module, "Runner", lambda: fake)
    monkeypatch.setattr(pipeline, "default_tracker", no_model)  # P1-2: any inference attempt fails the test
    monkeypatch.setattr(tracking, "Tracker", no_model)
    monkeypatch.setenv("MOTIONMAN_DATA_ROOT", str(root))
    monkeypatch.setenv("MOTIONMAN_DEMO_CLIP", str(clip))
    monkeypatch.setenv("MOTIONMAN_LEGACY_RUNS", str(tmp_path / "legacy"))
    st.cache_resource.clear()
    yield {"root": store.data_root(root), "a": a["run_id"], "b": b["run_id"], "fake": fake, "tmp": tmp_path}
    st.cache_resource.clear()


def render(at=None):
    at = (at or AppTest.from_file(APP, default_timeout=30)).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def open_run(at, run_id):
    at.selectbox(key="run_picker").set_value(run_id)
    return render(at)


def badge(at):
    return next(m.value for m in at.markdown if "-badge[" in m.value)


def captions(at):
    return [c.value for c in at.caption]


def test_fresh_page_is_the_guided_setup(world):
    at = render()
    assert any(m.value.startswith("### Choose a video to track objects") for m in at.markdown)
    assert "Ready" in badge(at)
    assert [b.label for b in at.button] == ["Run tracking"] and not at.button[0].disabled
    assert at.expander[0].label == "Advanced settings"  # thresholds and device are tucked away
    assert any("64×48" in c for c in captions(at))  # the demo was preflighted


def test_run_tracking_passes_the_advanced_settings(world, clip):
    world["fake"].allow_start = True
    at = render()
    at.segmented_control(key="device").set_value("Auto")
    at.slider(key="conf").set_value(0.4)
    at.button[0].click()
    render(at)
    assert world["fake"].starts == [(RunSettings(device="auto", conf=0.4), {"source": clip})]


def test_a_tracking_preset_fills_the_next_runs_settings(world):
    presets.save(world["root"], Preset("night", tracking=TrackingSettings(device="cuda", conf=0.6)))
    at = render()
    presets.save(world["root"], Preset("look", view=ViewOptions(labels=False)))
    assert at.selectbox(key="tracking_preset").options == ["None", "night"]  # view-only presets aren't listed here
    at.selectbox(key="tracking_preset").set_value("night")
    render(at)
    assert at.segmented_control(key="device").value == "CUDA" and at.slider(key="conf").value == 0.6


def test_opening_a_run_shows_inspect_and_performance(world):
    at = open_run(render(), world["a"])
    assert [t.label for t in at.tabs] == ["Inspect", "Performance"]
    assert "Complete" in badge(at) and any(b.label == "New video" for b in at.button)
    assert at.slider(key=f"frame-{world['a']}").value == 0
    assert [m.value for m in at.metric][2] == "20"  # Performance: frames processed, from this run's report


def test_inspecting_never_runs_the_model(world):
    at = open_run(render(), world["a"])
    at.slider(key=f"frame-{world['a']}").set_value(7)
    at = render(at)
    at.selectbox(key=f"track-{world['a']}").set_value(1)
    at = render(at)
    at.toggle(key=f"labels-{world['a']}").set_value(False)
    at = render(at)
    at.slider(key=f"trail-{world['a']}").set_value(5)
    at = render(at)
    at.segmented_control(key=f"playback-{world['a']}").set_value("Original")
    at = render(at)
    open_run(at, world["b"])
    # Every interaction above re-rendered without a single call to the runner or the model.
    assert world["fake"].starts == []


def test_object_panel_reports_this_frame_and_intervals(world):
    at = open_run(render(), world["a"])
    at.selectbox(key=f"track-{world['a']}").set_value(1)
    at.slider(key=f"frame-{world['a']}").set_value(3)  # track 1 is missing at frame 3
    at = render(at)
    assert any("Not observed in this frame" in c for c in captions(at))
    assert any("Observed 19 frames: 0.0–0.1 s, 0.1–0.6 s" in c for c in captions(at))
    at.slider(key=f"frame-{world['a']}").set_value(4)
    at = render(at)
    assert any("Observed in this frame" in c and "Not" not in c for c in captions(at))


def test_first_and_last_seen_move_the_inspected_frame(world):
    at = open_run(render(), world["a"])
    at.selectbox(key=f"track-{world['a']}").set_value(1)
    at = render(at)
    next(b for b in at.button if b.label == "Last seen").click()
    at = render(at)
    assert at.slider(key=f"frame-{world['a']}").value == 19


def test_each_run_keeps_its_own_context_and_ids_never_carry_over(world):
    a, b = world["a"], world["b"]
    at = open_run(render(), a)
    at.slider(key=f"frame-{a}").set_value(7)
    at.selectbox(key=f"track-{a}").set_value(1)
    at = render(at)
    at = open_run(at, b)
    assert at.slider(key=f"frame-{b}").value == 0
    assert at.selectbox(key=f"track-{b}").value is None  # ID 1 doesn't exist in run B
    at = open_run(at, a)
    assert at.slider(key=f"frame-{a}").value == 7 and at.selectbox(key=f"track-{a}").value == 1


def test_new_video_returns_to_setup_and_stays_there(world):
    at = open_run(render(), world["a"])
    next(b for b in at.button if b.label == "New video").click()
    at = render(at)
    at = render(at)  # a second rerun must not reopen the previous run
    assert any(m.value.startswith("### Choose a video") for m in at.markdown)


def test_evidence_png_export_writes_a_file(world):
    at = open_run(render(), world["a"])
    at.slider(key=f"frame-{world['a']}").set_value(4)
    at = render(at)
    next(b for b in at.button if b.label == "Create evidence PNG").click()
    render(at)
    pngs = list((world["root"].path / "exports").glob("*/evidence.png"))
    assert len(pngs) == 1 and (pngs[0].parent / "export.json").is_file()


def test_a_finished_run_opens_its_result(world):
    world["fake"].state = JobState(status="complete", report={"run_id": world["b"]}, run_dir=None)
    at = render()
    assert [t.label for t in at.tabs] == ["Inspect", "Performance"]
    assert at.selectbox(key="run_picker").value == world["b"]


def test_runner_error_shows_next_step_on_setup(world):
    world["fake"].state = JobState(
        status="error", message="CUDA is not available", next_step="Choose CPU or Auto and run again."
    )
    at = render()
    assert "Error" in badge(at) and at.error[0].value == "CUDA is not available"
    assert any(m.value == "**Next step:** Choose CPU or Auto and run again." for m in at.markdown)


def test_a_damaged_run_is_refused_with_a_next_step(world):
    run_dir = world["root"].runs / world["a"]
    with (run_dir / "tracks.jsonl").open("ab") as f:
        f.write(b"x")
    at = open_run(render(), world["a"])
    assert at.error[0].value.startswith("This run can't be opened")


def test_legacy_runs_are_hidden_until_asked_for(world, clip):
    pipeline.run(RunConfig(clip, world["tmp"] / "legacy" / "old"), make_tracker=FakeTracker, resolve=lambda d: "cpu")
    at = render()
    assert "legacy:old" not in at.selectbox(key="run_picker").options
    at.toggle(key="show_legacy").set_value(True)
    at = render(at)
    assert any("Legacy (unverified)" in o for o in at.selectbox(key="run_picker").options)


def test_a_broken_legacy_run_shows_an_error_not_a_crash(world, clip):
    pipeline.run(RunConfig(clip, world["tmp"] / "legacy" / "old"), make_tracker=FakeTracker, resolve=lambda d: "cpu")
    tracks = world["tmp"] / "legacy" / "old" / "tracks.jsonl"
    tracks.write_text(tracks.read_text()[:-40])  # truncated mid-record
    at = render()
    at.toggle(key="show_legacy").set_value(True)
    at = open_run(render(at), "legacy:old")
    assert at.error[0].value.startswith("This run can't be opened")


def test_a_file_changed_after_opening_is_verified_again(world):
    at = open_run(render(), world["a"])
    with (world["root"].runs / world["a"] / "tracks.jsonl").open("ab") as f:
        f.write(b"x")
    at.slider(key=f"frame-{world['a']}").set_value(2)
    at = render(at)
    assert at.error[0].value.startswith("This run can't be opened")


def test_a_view_preset_applies_to_this_runs_view_only(world):
    presets.save(world["root"], Preset("quiet", view=ViewOptions(labels=False, trail_length=5)))
    presets.save(world["root"], Preset("gpu", tracking=TrackingSettings(device="cuda")))
    at = open_run(render(), world["a"])
    assert at.selectbox(key=f"view_preset-{world['a']}").options == ["None", "quiet"]
    at.selectbox(key=f"view_preset-{world['a']}").set_value("quiet")
    at = render(at)
    assert at.toggle(key=f"labels-{world['a']}").value is False and at.slider(key=f"trail-{world['a']}").value == 5
    assert world["fake"].starts == []


def test_saving_over_an_existing_view_preset_needs_replace(world):
    presets.save(world["root"], Preset("mine", view=ViewOptions(trail_length=7)))
    at = open_run(render(), world["a"])
    at.text_input(key="preset_name").set_value("mine")
    next(b for b in at.button if b.label == "Save view").click()
    at = render(at)
    assert "already exists" in at.error[0].value
    assert presets.load(world["root"], "mine").view.trail_length == 7


def test_performance_shows_where_the_time_went(world):
    at = open_run(render(), world["a"])
    labels = [m.label for m in at.metric]
    assert labels[4:] == ["Setup", "Processing", "Finalize", "Job total"]
    assert all(m.value.endswith(" s") for m in at.metric[4:])
    stages = at.dataframe[0].value
    assert list(stages["Stage"]) == ["Decode", "Detect + track", "Render", "Write", "Unattributed"]
    assert any("warmup" in c and "background threads" in c for c in captions(at))
