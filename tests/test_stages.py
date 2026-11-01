"""Stage timings, run-level durations and the warmup policy (plan P2 step 1, D1–D4)."""

import json
import time

import pytest

from motionman import metrics, pipeline, store, video_io
from motionman.types import RunConfig

from .test_pipeline import FakeTracker, run

STAGES = ("decode_ms", "detect_track_ms", "render_ms", "write_ms")


class SlowTracker(FakeTracker):
    """Each update takes about 20 ms, so detect_track should clearly dominate the stage breakdown."""

    def update(self, image):
        time.sleep(0.02)
        return super().update(image)


def timing_rows(out):
    return [json.loads(line) for line in (out / pipeline.TIMINGS).read_text().splitlines()]


def test_every_processed_frame_has_one_row_of_stage_times(clip, tmp_path):
    out = tmp_path / "run"
    report = run(RunConfig(clip, out))
    rows = timing_rows(out)
    assert [r["source_frame_index"] for r in rows] == list(range(20)) and report["processed_frames"] == 20
    assert all(set(r) == {"source_frame_index", *STAGES} and all(r[s] >= 0 for s in STAGES) for r in rows)


def test_stage_totals_fit_inside_the_wall_time_and_the_rest_is_shown(clip, tmp_path):
    t = run(RunConfig(clip, tmp_path / "run"))["timing"]
    attributed = sum(t["stages"][s]["total_s"] for s in STAGES)
    assert attributed <= t["processing_wall_s"] + 0.002  # each total is rounded to 1 ms
    assert t["unattributed"]["total_s"] == pytest.approx(t["processing_wall_s"] - attributed, abs=0.005)
    assert set(t["stage_scopes"]) == set(STAGES)


def test_setup_processing_and_finalize_make_up_the_job(clip, tmp_path):
    t = run(RunConfig(clip, tmp_path / "run"))["timing"]
    assert t["setup_s"] >= t["model_load_s"] >= 0 and t["finalize_s"] > 0
    parts = t["setup_s"] + t["processing_wall_s"] + t["finalize_s"]
    assert t["job_wall_s"] == pytest.approx(parts, abs=0.005)
    assert "not included" in t["finalize_scope"]


def test_finalize_covers_the_output_check_and_setup_covers_model_load(clip, tmp_path, monkeypatch):
    real_count = video_io.count_frames

    def slow_count(path):
        time.sleep(0.2)
        return real_count(path)

    def slow_tracker(cfg):
        time.sleep(0.2)
        return FakeTracker(cfg)

    monkeypatch.setattr(video_io, "count_frames", slow_count)
    r = pipeline.run(RunConfig(clip, tmp_path / "run"), make_tracker=slow_tracker, resolve=lambda d: "cpu")
    t = r["timing"]
    assert t["finalize_s"] >= 0.2 and t["setup_s"] >= 0.2 and t["processing_wall_s"] < 0.2


def test_the_slow_stage_is_the_one_that_dominates(clip, tmp_path):
    t = pipeline.run(RunConfig(clip, tmp_path / "run"), make_tracker=SlowTracker, resolve=lambda d: "cpu")["timing"]
    totals = {s: t["stages"][s]["total_s"] for s in STAGES}
    assert max(totals, key=totals.get) == "detect_track_ms" and totals["detect_track_ms"] >= 0.38
    assert t["stages"]["detect_track_ms"]["p50"] >= 19


def test_warmup_policy_and_short_runs_are_explicit(clip, tmp_path):
    t = run(RunConfig(clip, tmp_path / "a"))["timing"]
    assert t["warmup"] == {"policy": "exclude_first_inferred", "frames": 10, "samples": 10}
    assert all(t["stages"][s]["samples"] == 10 and t["stages"][s]["warmup_excluded"] == 10 for s in STAGES)
    short = run(RunConfig(clip, tmp_path / "b", max_frames=8))["timing"]
    assert short["warmup"]["samples"] == 0
    for s in STAGES:
        assert short["stages"][s]["p50"] is None and "warmup excluded" in short["stages"][s]["reason"]
        assert short["stages"][s]["total_s"] >= 0  # still counted toward throughput


def test_a_failed_run_still_reports_what_it_measured(clip, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run(RunConfig(clip, out), fail_at=6, exc=RuntimeError("boom"))
    report = json.loads((out / pipeline.REPORT).read_text())
    assert report["status"] == "error" and len(timing_rows(out)) == 6
    assert report["timing"]["job_wall_s"] > 0 and report["timing"]["finalize_s"] is not None


def test_timings_are_a_hashed_artifact_of_a_published_run(clip, tmp_path):
    root = tmp_path / "data"
    report = pipeline.run(RunConfig(clip, None, data_root=root), make_tracker=FakeTracker, resolve=lambda d: "cpu")
    run_dir = store.data_root(root).runs / report["run_id"]
    manifest = json.loads((run_dir / store.MANIFEST).read_text())
    entry = next(a for a in manifest["artifacts"] if a["path"] == pipeline.TIMINGS)
    assert entry["sha256"] == store.sha256_file(run_dir / pipeline.TIMINGS)


def test_metrics_finish_is_idempotent_and_needs_a_job_start():
    m = metrics.Metrics()
    m.finish()
    assert m.job_wall_s is None  # never invented without a start
    m.begin_job()
    m.start()
    m.stop()
    m.finish()
    first = m.job_wall_s
    m.finish()
    assert m.job_wall_s == first and m.finalize_s is not None and m.setup_s is not None
