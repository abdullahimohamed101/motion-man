import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from motionman import store
from motionman.types import RunConfig, VideoInfo

INFO = VideoInfo(Path("in.mp4"), 1920, 1080, 24000 / 1001, 1356)


def runspec(config=None, **kw):
    kw.setdefault("input_id", "sha256:" + "a" * 64)
    kw.setdefault("device_used", "cpu")
    return store.resolve_runspec(config or RunConfig(Path("in.mp4"), Path("out")), INFO, **kw)


def test_sha256_file_matches_known_vector(tmp_path):
    p = tmp_path / "abc.bin"
    p.write_bytes(b"abc")
    assert store.sha256_file(p) == "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_canonical_json_is_order_independent_and_rejects_nan():
    assert store.canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    assert store.sha256_json({"x": 1, "y": 2}) == store.sha256_json({"y": 2, "x": 1})
    with pytest.raises(ValueError):
        store.canonical_json({"v": float("nan")})


def test_run_ids_have_the_documented_shape_and_never_repeat():
    fixed = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)
    rid = store.new_run_id(fixed)
    assert re.fullmatch(r"r_20261101T120000Z_[0-9a-f]{6}", rid)
    assert len({store.new_run_id(fixed) for _ in range(50)}) == 50  # a benchmark repeat is a new attempt


def test_tracker_config_is_read_from_the_installed_package():
    cfg = store.tracker_config()
    assert cfg["tracker_type"] == "bytetrack" and "track_buffer" in cfg


def test_runspec_resolves_everything_and_hashes_settings():
    spec = runspec(class_names={0: "person", 2: "car"}, codec="avc1", limits={"max_frames": 36000})
    s = spec["settings"]
    assert spec["schema_version"] == "1.1" and spec["config_sha256"] == store.sha256_json(s)
    assert s["backend"] == {"name": "ultralytics", "precision": "fp32"}  # the legacy default (ADR-006)
    assert s["input"]["input_id"].startswith("sha256:") and s["input"]["frame_range"] == [0, None]
    assert s["tracking"]["config"]["tracker_type"] == "bytetrack"
    assert s["tracking"]["config_sha256"] == store.sha256_json(s["tracking"]["config"])
    assert s["inference"] == {
        "device_requested": "cpu",
        "device_used": "cpu",
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "class_filter": None,
        "precision": "fp32",
    }
    assert s["model"]["class_map_sha256"] == store.sha256_json({"0": "person", "2": "car"})
    assert s["rendering"]["codec"] == "avc1" and s["limits"] == {"max_frames": 36000}


def test_same_settings_same_hash_and_any_setting_change_changes_it():
    base = runspec()["config_sha256"]
    assert runspec()["config_sha256"] == base
    assert runspec(RunConfig(Path("in.mp4"), Path("out"), conf=0.4))["config_sha256"] != base
    assert runspec(device_used="cuda")["config_sha256"] != base
    assert runspec(input_id="sha256:" + "b" * 64)["config_sha256"] != base


NATIVE = {"name": "tensorrt", "precision": "fp32", "profile": "rtdetr_r18_640", "engine_sha256": "sha256:" + "e" * 64}


def test_a_native_backend_sets_precision_and_tracking_and_is_hashed():
    tracking = {"type": "bytetrack", "implementation": "motionman-native", "config": {"class_policy": "aware"}}
    spec = runspec(backend={**NATIVE, "precision": "fp16"}, tracking=tracking)
    s = spec["settings"]
    assert s["backend"]["name"] == "tensorrt" and s["inference"]["precision"] == "fp16"
    assert s["tracking"]["implementation"] == "motionman-native"
    assert s["tracking"]["config_sha256"] == store.sha256_json({"class_policy": "aware"})
    assert runspec(backend=NATIVE)["config_sha256"] != runspec()["config_sha256"]
    other_engine = {**NATIVE, "engine_sha256": "sha256:" + "f" * 64}
    assert runspec(backend=other_engine)["config_sha256"] != runspec(backend=NATIVE)["config_sha256"]


def test_a_backend_without_name_or_precision_is_refused():
    with pytest.raises(ValueError, match="name and a precision"):
        runspec(backend={"name": "tensorrt"})


def test_runspec_backend_reads_both_versions():
    assert store.runspec_backend(runspec(backend=NATIVE)) == NATIVE
    old = runspec()
    del old["settings"]["backend"]  # a 1.0 runspec has no block: it was the legacy backend
    assert store.runspec_backend({**old, "schema_version": "1.0"}) == {"name": "ultralytics", "precision": "fp32"}
    assert store.runspec_backend(None) == {"name": "ultralytics", "precision": "fp32"}


def test_code_revision_is_recorded_but_not_hashed(monkeypatch):
    base = runspec()
    monkeypatch.setattr(store, "code_revision", lambda repo=store.REPO_ROOT: {"commit": "other", "dirty": False})
    moved = runspec()
    assert moved["reproducibility"]["code"]["commit"] == "other"
    assert moved["config_sha256"] == base["config_sha256"]


def test_missing_weights_are_null_with_a_reason_not_invented(tmp_path):
    s = runspec(weights_path=tmp_path / "nope.pt")["settings"]["model"]
    assert s["weights_sha256"] is None and s["weights_reason"] == "weights file not present"


def test_present_weights_and_requirements_are_hashed(tmp_path):
    w = tmp_path / "w.pt"
    w.write_bytes(b"weights")
    spec = runspec(weights_path=w)
    assert spec["settings"]["model"]["weights_sha256"] == store.sha256_file(w)
    assert spec["reproducibility"]["requirements"]["sha256"].startswith("sha256:")


def test_code_revision_reports_this_checkout():
    rev = store.code_revision()
    assert re.fullmatch(r"[0-9a-f]{40}", rev["commit"]) and isinstance(rev["dirty"], bool)


def test_code_revision_outside_git_is_unknown(tmp_path):
    assert store.code_revision(tmp_path)["commit"] is None
