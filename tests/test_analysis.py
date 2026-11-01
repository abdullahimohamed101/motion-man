"""Analysis runner: deterministic, reusable, never infers, never touches the run (plan P3 step 3, D1, D3, D8).

The fake tracker moves one object right by 1 px per frame (centre x = frame + 5, y = 12.5) in a 64×48 clip at
30 fps, skips frame 3, and adds an unassigned car on frame 5. A vertical line at x = 12 px is crossed once.
"""

import dataclasses
import json
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
    assert set(files(result.folder)) == {"analysis_spec.json", "events.jsonl", "summary.json", "heatmap.json",
                                         "manifest.json"}  # fmt: skip
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
    summary["bookmarks"] = [{k: v for k, v in b.items() if k != "event_id"} for b in summary["bookmarks"]]
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


# --- step 4: heatmap, bookmarks, export, CLI ---

WHOLE = {**ZONE, "id": "all", "name": "All", "box": [0.0, 0.0, 1.0, 1.0]}


def test_the_heatmap_holds_exactly_the_observed_dwell(run, root):
    result = analysis.analyze(run, region_set(run, WHOLE), root)
    heat = analysis.read_heatmap(result)
    (zone,) = result.summary["regions"]
    assert heat["grid"] == [256, 144] and heat["units"] == "object_us" and "image-space" in heat["label"]
    assert heat["total_us"] == zone["dwell_total_us"] == sum(v for _, _, v in heat["cells"]) > 0
    # Centre x = frame + 5 in a 64 px frame: 4 columns per pixel; frame 3 is missing, so no interval starts there.
    assert {c for c, _, _ in heat["cells"]} == {(f + 5) * 4 for f in range(19) if f not in (2, 3)}
    assert {r for _, r, _ in heat["cells"]} == {int(12.5 * 144 / 48)}


def test_an_empty_window_gives_a_defined_empty_heatmap(run, root):
    result = analysis.analyze(
        run, region_set(run, WHOLE), root, window_us=(100_000, 133_333)
    )  # frame 3 only (the end is exclusive)
    heat = analysis.read_heatmap(result)
    assert heat["total_us"] == 0 and heat["cells"] == [] and result.summary["coverage"]["frames"] == 1


def test_bookmarks_name_their_criterion(run, root):
    result = analysis.analyze(run, region_set(run, LINE, ZONE), root)
    marks = {(b["criterion"], b["rule_id"]): b for b in result.summary["bookmarks"]}
    assert set(marks) == {("first crossing", "door"), ("first entry", "pen"), ("peak occupancy", "pen")}
    events = {e["type"]: e for e in analysis.read_events(result)}
    assert marks[("first crossing", "door")]["event_id"] == events["LINE_CROSSING"]["event_id"]
    assert marks[("peak occupancy", "pen")]["value"] == 1 and marks[("peak occupancy", "pen")]["event_id"] is None
    frames = [b["frame"] for b in result.summary["bookmarks"]]
    assert frames == sorted(frames)


def test_exports_keep_ids_and_units(run, root):
    import csv
    import io
    import json

    dwell_zone = {**ZONE, "dwell_threshold_s": 0.1}
    result = analysis.analyze(run, region_set(run, LINE, dwell_zone), root)
    events = list(analysis.read_events(result))
    rows = list(csv.DictReader(io.StringIO(analysis.export_csv(result))))
    assert tuple(rows[0]) == analysis.EVENT_COLUMNS and len(rows) == len(events)
    assert [r["event_id"] for r in rows] == [e["event_id"] for e in events]
    by_type = {r["type"]: r for r in rows}
    assert by_type["LINE_CROSSING"]["rule_name"] == "Door" and by_type["LINE_CROSSING"]["value_unit"] == ""
    assert by_type["DWELL_THRESHOLD"]["value_unit"] == "us" and int(by_type["DWELL_THRESHOLD"]["value"]) >= 100_000
    crossing = next(e for e in events if e["type"] == "LINE_CROSSING")
    assert by_type["LINE_CROSSING"]["confirming_t_us"] == str(crossing["confirming_t_us"])
    doc = json.loads(analysis.export_json(result))
    assert doc["events"] == events and doc["units"]["time"] == "us" and doc["summary"] == result.summary


def write_regions(tmp_path, run, *items):
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(region_set(run, *items).to_dict()))
    return path


def test_cli_analyze_by_file_and_by_id(run, root, tmp_path, capsys):
    from motionman import cli

    path = write_regions(tmp_path, run, LINE, ZONE)
    out_csv, out_json = tmp_path / "events.csv", tmp_path / "analysis.json"
    args = ["analyze", "--run", run.run_id, "--data-root", str(root.path)]
    assert cli.main([*args, "--regions", str(path), "--csv", str(out_csv), "--json", str(out_json)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "line door: 1 crossings (1 negative→positive" in out and "zone pen: 1 entries" in out
    assert out_csv.read_text().startswith("event_id,type") and json.loads(out_json.read_text())["events"]
    rs_id = region_set(run, LINE, ZONE).region_set_id  # the file was saved to the region store as well
    assert cli.main([*args, "--regions", rs_id]) == cli.EXIT_OK
    assert "(already up to date)" in capsys.readouterr().out
    assert cli.main([*args, "--regions", rs_id, "--window", "0,0.2"]) == cli.EXIT_OK
    assert ", 0 events" in capsys.readouterr().out


def test_cli_analyze_refuses_bad_input_with_exit_2(run, root, tmp_path, capsys):
    from motionman import cli

    good = write_regions(tmp_path, run, LINE)
    args = ["analyze", "--data-root", str(root.path)]
    assert cli.main([*args, "--run", "r_nope", "--regions", str(good)]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert cli.main([*args, "--run", run.run_id, "--regions", str(bad)]) == 2
    assert cli.main([*args, "--run", run.run_id, "--regions", "rs_0000000000000000"]) == 2
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema_version": "1.0", "input_id": "sha256:" + "0" * 64, "name": "x",
                                 "regions": [LINE]}))  # fmt: skip
    assert cli.main([*args, "--run", run.run_id, "--regions", str(other)]) == 2
    taken = tmp_path / "taken.csv"
    taken.write_text("keep me")
    assert cli.main([*args, "--run", run.run_id, "--regions", str(good), "--csv", str(taken)]) == 2
    assert taken.read_text() == "keep me"
    assert "different video" in capsys.readouterr().err
