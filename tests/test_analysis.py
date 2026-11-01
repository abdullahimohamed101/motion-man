"""Analysis runner: deterministic, reusable, never infers, never touches the run (plan P3 step 3, D1, D3, D8).

The fake tracker moves one object right by 1 px per frame (centre x = frame + 5, y = 12.5) in a 64×48 clip at
30 fps, skips frame 3, and adds an unassigned car on frame 5. A vertical line at x = 12 px is crossed once.
"""

import dataclasses
import time

import pytest

from motionman import analysis, analytics, pipeline, regions, store
from motionman.types import RunConfig

from .test_pipeline import FakeTracker


class SlowTracker(FakeTracker):
    def update(self, image):
        time.sleep(0.01)  # the same detections, produced much more slowly
        return super().update(image)


def publish(clip, root, tracker=FakeTracker):
    report = pipeline.run(RunConfig(clip, None, data_root=root.path), make_tracker=tracker, resolve=lambda d: "cpu")
    return analysis.find_run(root, report["run_id"])


def region_set(run, *items, name="Test"):
    body = {"schema_version": "1.0", "input_id": run.manifest["input_id"], "name": name, "regions": list(items)}
    return regions.parse(body)


LINE = {"id": "door", "name": "Door", "kind": "line", "points": [[0.1875, 0.0], [0.1875, 1.0]], "classes": None}
ZONE = {"id": "pen", "name": "Pen", "kind": "rect", "box": [0.21875, 0.0, 0.46875, 0.625], "classes": None,
        "dwell_threshold_s": None, "occupancy_threshold": None}  # fmt: skip


@pytest.fixture
def root(tmp_path):
    return store.data_root(tmp_path / "data")


@pytest.fixture
def run(clip, root):
    return publish(clip, root)


def files(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def test_an_analysis_publishes_its_files_and_finds_the_crossing(run, root):
    result = analysis.analyze(run, region_set(run, LINE, ZONE), root)
    assert result.folder == root.path / "analyses" / run.run_id / result.analysis_id and not result.reused
    assert set(files(result.folder)) == {"analysis_spec.json", "events.jsonl", "summary.json", "manifest.json"}
    events = list(analysis.read_events(result))
    assert [(e["type"], e["rule_id"], e["track_id"], e["direction"]) for e in events] == [
        ("LINE_CROSSING", "door", 1, "NEGATIVE_TO_POSITIVE"), ("ZONE_ENTRY", "pen", 1, None)]  # fmt: skip
    assert all(e["event_id"].startswith("e_") and e["analysis_id"] == result.analysis_id for e in events)
    s = result.summary
    assert s["coverage"]["frames"] == 20 and s["unassigned_detections"] == 1 and s["events"]["total"] == 2
    door, pen = s["regions"]
    assert door["crossings_total"] == 1 and (pen["entries"], pen["peak_occupancy"]) == (1, 1)
    assert pen["visits"] == 1 and pen["dwell_total_us"] > 0
    assert result.spec["records_sha256"] == run.records_sha256 and result.spec["params"]["deadband_px"] == 3.0


def test_identical_config_gives_identical_ids_and_bytes(run, root):
    rs = region_set(run, LINE, ZONE)
    first = analysis.analyze(run, rs, root)
    before = files(first.folder)
    again = analysis.analyze(run, rs, root)
    assert again.reused and again.analysis_id == first.analysis_id
    for p in first.folder.iterdir():
        p.unlink()
    first.folder.rmdir()
    rebuilt = analysis.analyze(run, rs, root)
    assert not rebuilt.reused and files(rebuilt.folder) == before  # rebuilt from scratch: byte-identical


def strip(result):
    events = [
        {k: v for k, v in e.items() if k not in ("event_id", "analysis_id")} for e in analysis.read_events(result)
    ]
    summary = {k: v for k, v in result.summary.items() if k not in ("analysis_id", "run_id")}
    return events, summary


def test_processing_speed_does_not_change_the_result(clip, root):
    fast = publish(clip, root)
    slow = publish(clip, root, tracker=SlowTracker)
    a = analysis.analyze(fast, region_set(fast, LINE, ZONE), root)
    b = analysis.analyze(slow, region_set(slow, LINE, ZONE), root)
    assert a.analysis_id != b.analysis_id  # different runs, so different analyses ...
    assert strip(a) == strip(b) and strip(a)[0]  # ... with identical source-time results


def test_analysis_never_infers_and_leaves_the_run_untouched(run, root, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("analysis must never run the model")

    monkeypatch.setattr(pipeline, "default_tracker", refuse)
    before = {p.name: store.sha256_file(p) for p in run.summary.path.iterdir()}
    analysis.analyze(run, region_set(run, LINE, ZONE), root)
    assert {p.name: store.sha256_file(p) for p in run.summary.path.iterdir()} == before


def test_regions_change_makes_a_new_analysis_and_keeps_the_old_one(run, root):
    old = analysis.analyze(run, region_set(run, LINE), root)
    moved = {**LINE, "points": [[0.25, 0.0], [0.25, 1.0]]}
    new = analysis.analyze(run, region_set(run, moved), root)
    assert new.analysis_id != old.analysis_id
    assert analysis.load(root, run.run_id, old.analysis_id).summary == old.summary
    assert analysis.list_analyses(root, run.run_id) == sorted([old.analysis_id, new.analysis_id],
        key=lambda a: analysis.folder_for(root, run.run_id, a).stat().st_mtime)  # fmt: skip


def test_a_damaged_analysis_is_rebuilt_not_trusted(run, root):
    rs = region_set(run, LINE)
    result = analysis.analyze(run, rs, root)
    good = (result.folder / "events.jsonl").read_bytes()
    (result.folder / "events.jsonl").write_text("tampered\n")
    with pytest.raises(analysis.AnalysisError, match="missing or damaged"):
        analysis.load(root, run.run_id, result.analysis_id)
    assert analysis.list_analyses(root, run.run_id) == []
    rebuilt = analysis.analyze(run, rs, root)
    assert not rebuilt.reused and (rebuilt.folder / "events.jsonl").read_bytes() == good
    assert [p.name for p in rebuilt.folder.parent.iterdir()] == [result.analysis_id]  # nothing left aside


def test_the_window_limits_events_and_is_part_of_the_identity(run, root):
    rs = region_set(run, LINE)
    early = analysis.analyze(run, rs, root, window_us=(0, 200_000))  # frames 0–5: before the crossing confirms
    assert early.summary["events"]["total"] == 0 and early.summary["coverage"]["frames"] == 6
    assert early.spec["window_us"] == [0, 200_000] and early.analysis_id != analysis.analyze(run, rs, root).analysis_id
    with pytest.raises(analysis.AnalysisError, match="no frames"):
        analysis.analyze(run, rs, root, window_us=(10_000_000, None))
    with pytest.raises(analysis.AnalysisError, match="window"):
        analysis.analyze(run, rs, root, window_us=(500, 100))


def test_params_are_part_of_the_identity(run, root):
    rs = region_set(run, LINE)
    loose = analysis.analyze(run, rs, root, params=analytics.Params(confirm_us=0, confirm_obs=1))
    assert loose.analysis_id != analysis.analyze(run, rs, root).analysis_id
    assert loose.spec["params"] == {"deadband_px": 3.0, "confirm_us": 0, "confirm_obs": 1, "max_gap_us": 500_000}


def test_regions_for_another_video_or_a_legacy_run_are_refused(run, root):
    other = regions.parse({"schema_version": "1.0", "input_id": "sha256:" + "0" * 64, "name": "x", "regions": [LINE]})
    with pytest.raises(analysis.AnalysisError, match="different video"):
        analysis.analyze(run, other, root)
    legacy = dataclasses.replace(run, manifest=None, records_sha256=None)
    with pytest.raises(analysis.AnalysisError, match="legacy run"):
        analysis.analyze(legacy, region_set(run, LINE), root)
    with pytest.raises(analysis.AnalysisError, match="no published run"):
        analysis.find_run(root, "r_nope")


def test_a_failed_build_publishes_nothing_and_leaves_no_temp_folder(run, root, monkeypatch):
    real = analytics.Engine.feed

    def explode(self, frame, t_us, objects):
        if frame == 10:
            raise RuntimeError("disk full")
        return real(self, frame, t_us, objects)

    monkeypatch.setattr(analytics.Engine, "feed", explode)
    with pytest.raises(RuntimeError, match="disk full"):
        analysis.analyze(run, region_set(run, LINE), root)
    assert list((root.path / "analyses" / run.run_id).iterdir()) == []
