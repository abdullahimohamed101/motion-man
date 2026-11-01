"""Canonical runs: data root, staging, validation, manifest and atomic publication (spec A§2–A§5, plan P0 step 4)."""

import json

import pytest

from motionman import pipeline, records, store
from motionman.types import Limits, RunConfig

from .conftest import write_clip
from .test_pipeline import FakeTracker


def canonical(clip, root, **kw):
    return RunConfig(clip, None, data_root=root, **kw)


def run(config, tracker=FakeTracker):
    return pipeline.run(config, make_tracker=tracker, resolve=lambda d: "cpu")


def manifest(folder):
    return json.loads((folder / store.MANIFEST).read_text())


def test_success_is_published_with_a_verifiable_manifest(clip, tmp_path):
    root = tmp_path / "data"
    report = run(canonical(clip, root))
    run_dir = root / "runs" / report["run_id"]
    assert report["run_dir"] == str(run_dir) and report["mode"] == "canonical"
    assert list((root / "staging").iterdir()) == []  # nothing left in staging
    m = manifest(run_dir)
    assert m["status"] == "SUCCEEDED" and m["run_id"] == report["run_id"] and m["error"] is None
    assert m["validation"] == {"source_complete": "PASS", "records": "PASS", "video_decode": "PASS"}
    assert m["counts"] == {
        "source_frames_reported": 20, "decoded_frames": 20, "inferred_frames": 20, "records": 20, "written_frames": 20
    }  # fmt: skip
    assert {a["path"] for a in m["artifacts"]} == {"annotated.mp4", "tracks.jsonl", "report.json", "runspec.json"}
    for a in m["artifacts"]:  # every hash and size matches the bytes on disk
        assert (
            store.sha256_file(run_dir / a["path"]) == a["sha256"] and (run_dir / a["path"]).stat().st_size == a["bytes"]
        )
    assert m["records_sha256"] == store.sha256_file(run_dir / "tracks.jsonl")
    assert m["input_id"] == store.sha256_file(clip) == report["input_id"]
    spec = json.loads((run_dir / store.RUNSPEC).read_text())
    assert spec["config_sha256"] == m["config_sha256"] == report["config_sha256"]
    assert {r["run_id"] for r in records.read_records(run_dir / "tracks.jsonl")} == {report["run_id"]}


def test_input_is_imported_once_and_runs_get_distinct_ids(clip, tmp_path):
    root = tmp_path / "data"
    a, b = run(canonical(clip, root)), run(canonical(clip, root))
    assert a["run_id"] != b["run_id"] and a["input_id"] == b["input_id"]
    inputs = list((root / "inputs").iterdir())
    assert len(inputs) == 1 and inputs[0].name == a["input_id"].removeprefix("sha256:")
    assert (inputs[0] / "source").read_bytes() == clip.read_bytes()
    assert json.loads((inputs[0] / "meta.json").read_text())["display_name"] == "clip.mp4"
    assert sorted(p.name for p in (root / "runs").iterdir()) == sorted([a["run_id"], b["run_id"]])


def test_legacy_mode_writes_the_same_manifest_in_place(clip, tmp_path):
    out = tmp_path / "out"
    report = run(RunConfig(clip, out))
    assert report["mode"] == "legacy" and report["run_dir"] == str(out)
    assert manifest(out)["status"] == "SUCCEEDED" and manifest(out)["input_id"] == store.sha256_file(clip)
    assert not (tmp_path / "data").exists()  # legacy mode never touches a data root


def test_mid_run_failure_is_not_published_and_keeps_diagnostics(clip, tmp_path):
    root = tmp_path / "data"
    with pytest.raises(RuntimeError, match="boom"):
        run(canonical(clip, root), tracker=lambda c: FakeTracker(c, fail_at=6, exc=RuntimeError("boom")))
    assert list((root / "runs").iterdir()) == []
    (staged,) = list((root / "staging").iterdir())
    m = manifest(staged)
    assert m["status"] == "FAILED" and "boom" in m["error"] and m["records_sha256"] is None
    assert (m["counts"]["decoded_frames"], m["counts"]["inferred_frames"], m["counts"]["records"]) == (7, 6, 6)


def test_interrupt_is_recorded_and_not_published(clip, tmp_path):
    root = tmp_path / "data"
    with pytest.raises(KeyboardInterrupt):
        run(canonical(clip, root), tracker=lambda c: FakeTracker(c, fail_at=4, exc=KeyboardInterrupt()))
    assert list((root / "runs").iterdir()) == []
    (staged,) = list((root / "staging").iterdir())
    assert manifest(staged)["status"] == "INTERRUPTED"


def test_failure_before_any_output_leaves_no_staging_folder(clip, tmp_path):
    def no_device(device):
        raise pipeline.video_io.VideoError("CUDA is not available")

    root = tmp_path / "data"
    with pytest.raises(pipeline.video_io.VideoError):
        pipeline.run(canonical(clip, root), make_tracker=FakeTracker, resolve=no_device)
    assert not (root / "staging").exists() or list((root / "staging").iterdir()) == []


def test_lease_refuses_a_second_concurrent_run(clip, tmp_path):
    root = store.data_root(tmp_path / "data")
    with store.execution_lease(root):
        with pytest.raises(store.LeaseError, match="another MotionMan run"):
            run(canonical(clip, root.path))
    assert run(canonical(clip, root.path))["status"] == "success"  # released afterwards


def test_source_changed_during_a_legacy_run_fails(tmp_path):
    clip = write_clip(tmp_path / "clip.mp4", 12)

    class Tamper(FakeTracker):
        def update(self, image):
            if self.calls == 5:
                with clip.open("ab") as f:
                    f.write(b"tampered")
            return super().update(image)

    out = tmp_path / "out"
    with pytest.raises(store.StoreError, match="changed during the run"):
        run(RunConfig(clip, out), tracker=Tamper)
    assert manifest(out)["status"] == "FAILED"


def test_invalid_records_fail_validation_and_are_not_published(clip, tmp_path, monkeypatch):
    def broken(path, expected, *args):
        raise store.StoreError("tracks.jsonl: 19 records, expected 20")

    monkeypatch.setattr(store, "validate_records_file", broken)
    root = tmp_path / "data"
    with pytest.raises(pipeline.OutputCheckError, match="record check failed"):
        run(canonical(clip, root))
    (staged,) = list((root / "staging").iterdir())
    assert manifest(staged)["validation"]["records"] == "FAIL" and list((root / "runs").iterdir()) == []


def test_output_budget_breach_fails_the_run(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_check_free_space", lambda folder, budget: None)
    root = tmp_path / "data"
    with pytest.raises(pipeline.OutputCheckError, match="budget"):
        run(canonical(clip, root, limits=Limits(output_budget_bytes=1000)))
    (staged,) = list((root / "staging").iterdir())
    assert manifest(staged)["status"] == "FAILED" and list((root / "runs").iterdir()) == []


def test_not_enough_free_disk_fails_before_writing(clip, tmp_path):
    root = tmp_path / "data"
    with pytest.raises(store.StoreError, match="GiB free"):
        run(canonical(clip, root, limits=Limits(output_budget_bytes=10**18)))
    assert list((root / "staging").iterdir()) == []


def test_publish_never_overwrites(tmp_path):
    root = store.data_root(tmp_path / "data").ensure()
    (root.staging / "r_a").mkdir()
    (root.runs / "r_a").mkdir()
    with pytest.raises(store.StoreError, match="already published"):
        store.publish(root.staging / "r_a", root, "r_a")


def test_data_root_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("MOTIONMAN_DATA_ROOT", raising=False)
    assert str(store.data_root().path) == "data"
    monkeypatch.setenv("MOTIONMAN_DATA_ROOT", str(tmp_path))
    assert store.data_root().path == tmp_path
    assert store.data_root(tmp_path / "x").path == tmp_path / "x"


def test_managed_path_comes_from_the_hash_not_the_filename(tmp_path):
    src = tmp_path / "evil name.MOV"
    src.write_bytes(b"video bytes")
    input_id, managed = store.import_input(src, store.data_root(tmp_path / "data").ensure())
    assert managed.name == "source" and managed.parent.name == input_id.removeprefix("sha256:")
