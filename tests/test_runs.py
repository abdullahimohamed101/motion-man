"""Run shelf: discovery and verified opening (plan P1 step 1, D4/D5)."""

import json

import pytest

from motionman import pipeline, runs, store
from motionman.types import RunConfig

from .test_pipeline import FakeTracker


def canonical_run(clip, root, **kw):
    return pipeline.run(RunConfig(clip, None, data_root=root, **kw), make_tracker=FakeTracker, resolve=lambda d: "cpu")


@pytest.fixture
def root(tmp_path):
    return store.data_root(tmp_path / "data")


def test_published_runs_are_listed_newest_first_with_display_names(clip, root):
    first = canonical_run(clip, root.path)
    second = canonical_run(clip, root.path, max_frames=5)
    shelf = runs.list_runs(root)
    assert [r.run_id for r in shelf] == [second["run_id"], first["run_id"]]  # by start time, not by ID
    assert all(r.kind == "canonical" and r.display_name == "clip.mp4" for r in shelf)
    assert {r.frames for r in shelf} == {20, 5}


def test_failed_attempts_in_staging_are_never_listed(clip, root):
    with pytest.raises(RuntimeError):
        pipeline.run(
            RunConfig(clip, None, data_root=root.path),
            make_tracker=lambda c: FakeTracker(c, fail_at=3, exc=RuntimeError("boom")),
            resolve=lambda d: "cpu",
        )
    assert runs.list_runs(root) == [] and len(list(root.staging.iterdir())) == 1


def test_a_folder_in_runs_without_a_succeeded_manifest_is_not_listed(root):
    (root.runs / "r_fake").mkdir(parents=True)
    (root.runs / "r_fake" / store.MANIFEST).write_text(json.dumps({"run_id": "r_fake", "status": "FAILED"}))
    (root.runs / "r_other").mkdir()
    assert runs.list_runs(root) == []


def test_opening_verifies_and_exposes_what_the_inspector_needs(clip, root):
    report = canonical_run(clip, root.path)
    (summary,) = runs.list_runs(root)
    run = runs.open_run(summary, root)
    assert run.run_id == report["run_id"] and run.source_verified
    assert run.source_path.read_bytes() == clip.read_bytes()
    assert (run.fps, run.width, run.height) == (30.0, 64, 48)
    assert run.records_sha256 == store.sha256_file(run.records_path)


@pytest.mark.parametrize("artifact", ["tracks.jsonl", "annotated.mp4", "report.json"])
def test_a_tampered_artifact_refuses_the_run(clip, root, artifact):
    canonical_run(clip, root.path)
    (summary,) = runs.list_runs(root)
    with (summary.path / artifact).open("ab") as f:
        f.write(b"x")
    with pytest.raises(runs.RunError, match=f"{artifact} does not match its manifest"):
        runs.open_run(summary, root)


def test_a_changed_or_missing_source_refuses_the_run(clip, root):
    canonical_run(clip, root.path)
    (summary,) = runs.list_runs(root)
    (source,) = list(root.inputs.glob("*/source"))
    source.write_bytes(b"different")
    with pytest.raises(runs.RunError, match="source video is missing or changed"):
        runs.open_run(summary, root)
    source.unlink()
    with pytest.raises(runs.RunError, match="source video is missing or changed"):
        runs.open_run(summary, root)


def test_runspec_and_schema_mismatches_refuse_the_run(clip, root):
    canonical_run(clip, root.path)
    (summary,) = runs.list_runs(root)
    manifest = json.loads((summary.path / store.MANIFEST).read_text())
    (summary.path / store.MANIFEST).write_text(json.dumps({**manifest, "schema_version": "9"}))
    with pytest.raises(runs.RunError, match="unsupported manifest schema"):
        runs.open_run(summary, root)
    (summary.path / store.MANIFEST).write_text(json.dumps({**manifest, "config_sha256": "sha256:" + "0" * 64}))
    with pytest.raises(runs.RunError, match="runspec.json does not match"):
        runs.open_run(summary, root)


def test_a_run_written_before_backend_metadata_still_opens(clip, root):
    """Runs published with runspec and report 1.0 (no backend block) keep opening (ADR-006, plan N2-2)."""
    canonical_run(clip, root.path)
    (summary,) = runs.list_runs(root)
    folder = summary.path
    spec = json.loads((folder / store.RUNSPEC).read_text())
    del spec["settings"]["backend"]
    spec.update(schema_version="1.0", config_sha256=store.sha256_json(spec["settings"]))
    report = json.loads((folder / pipeline.REPORT).read_text())
    for key in ("schema_version", "backend"):
        del report[key]
    report["config_sha256"] = spec["config_sha256"]
    (folder / store.RUNSPEC).write_text(json.dumps(spec))
    (folder / pipeline.REPORT).write_text(json.dumps(report))
    manifest = json.loads((folder / store.MANIFEST).read_text())
    for a in manifest["artifacts"]:
        a.update(bytes=(folder / a["path"]).stat().st_size, sha256=store.sha256_file(folder / a["path"]))
    manifest["config_sha256"] = spec["config_sha256"]
    (folder / store.MANIFEST).write_text(json.dumps(manifest))
    run = runs.open_run(runs.list_runs(root)[0], root)
    assert store.runspec_backend(run.runspec) == {"name": "ultralytics", "precision": "fp32"}
    assert "schema_version" not in run.report


def test_a_new_run_records_the_legacy_backend(clip, root):
    report = canonical_run(clip, root.path)
    run = runs.open_run(runs.list_runs(root)[0], root)
    assert report["schema_version"] == "1.1" and report["backend"]["name"] == "ultralytics"
    assert run.runspec["schema_version"] == "1.1" and run.runspec["settings"]["backend"] == report["backend"]


def test_legacy_outputs_are_listed_read_only_and_unverified(clip, root, tmp_path):
    legacy_dir = tmp_path / "runs"
    pipeline.run(RunConfig(clip, legacy_dir / "a"), make_tracker=FakeTracker, resolve=lambda d: "cpu")
    mvp = legacy_dir / "b"  # an MVP-era folder: report + records, no manifest, source path gone
    mvp.mkdir()
    (mvp / "tracks.jsonl").write_text('{"frame_index": 0, "timestamp_s": 0.0, "objects": []}\n')
    (mvp / "report.json").write_text(json.dumps(
        {"status": "success", "processed_frames": 1, "created_at": "2026-10-31T21:00:00+00:00",
         "source": {"path": "clips/gone.mp4", "width": 64, "height": 48, "fps": 30.0}}
    ))  # fmt: skip
    (legacy_dir / "c").mkdir()  # an empty folder is ignored
    shelf = runs.list_runs(root, legacy_dir)
    assert [r.run_id for r in shelf] == ["legacy:a", "legacy:b"]
    assert all(r.kind == "legacy" and r.label.endswith("Legacy (unverified)") for r in shelf)
    a, b = (runs.open_run(r, root) for r in shelf)
    assert a.source_path == clip and not a.source_verified  # present, but MVP-style runs carry no proof
    assert b.source_path is None and b.manifest is None and (b.fps, b.width) == (30.0, 64)


def test_labels_read_well():
    s = runs.RunSummary(
        "r_20261101T062222Z_2ad4d6", "canonical", None, "crosswalk.mp4", "2026-11-01T06:22:22+00:00", 1356
    )
    assert s.label == "crosswalk.mp4 · 2026-11-01 06:22 · 2ad4d6"
    old = runs.RunSummary("legacy:demo", "legacy", None, "demo.mp4", None, 20)
    assert old.label == "demo.mp4 · demo · Legacy (unverified)"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: {**m, "artifacts": [a for a in m["artifacts"] if a["path"] != "tracks.jsonl"]},
         "doesn't cover tracks.jsonl"),
        (lambda m: {**m, "records_sha256": None}, "doesn't cover the records hash"),
        (lambda m: {**m, "artifacts": [*m["artifacts"], {"path": "../../escape", "sha256": "x"}]},
         "outside the run folder"),
        (lambda m: {**m, "artifacts": [{"sha256": "x"}, *m["artifacts"]]}, "outside the run folder"),
        (lambda m: {**m, "artifacts": "nope"}, "artifacts are malformed"),
    ],
)  # fmt: skip
def test_a_manifest_that_does_not_cover_the_run_is_refused(clip, root, change, message):
    canonical_run(clip, root.path)
    (summary,) = runs.list_runs(root)
    manifest = json.loads((summary.path / store.MANIFEST).read_text())
    (summary.path / store.MANIFEST).write_text(json.dumps(change(manifest)))
    with pytest.raises(runs.RunError, match=message):
        runs.open_run(summary, root)


def test_a_legacy_report_without_source_timing_is_refused_not_a_crash(root, tmp_path):
    folder = tmp_path / "runs" / "odd"
    folder.mkdir(parents=True)
    (folder / "tracks.jsonl").write_text("")
    (folder / "report.json").write_text(json.dumps({"status": "success", "source": {"path": "x.mp4"}}))
    (summary,) = runs.list_runs(root, tmp_path / "runs")
    with pytest.raises(runs.RunError, match="no usable source timing"):
        runs.open_run(summary, root)
