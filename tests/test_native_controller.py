"""The native controller publishes only verified runs (plan native-g2 step 6, N2-6, ADR-006). A fake engine
(tests/fake_native.py) stands in for motionman-native, so no GPU is needed (INV-8)."""

import json
import os
import re
import signal
import sys
import threading
from pathlib import Path

import pytest

from motionman import cli, metrics, native, pipeline, runs, store, ui
from motionman.types import RunConfig

from .conftest import write_clip

FAKE = Path(__file__).with_name("fake_native.py")
ENGINE_SHA = "sha256:" + "ab" * 32


@pytest.fixture
def model_dir(tmp_path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"classes": {"names": ["person", "bicycle", "car"]}}))
    (d / "engine_manifest.fp32.json").write_text(json.dumps({
        "engine": {"file": "model.fp32.engine", "sha256": ENGINE_SHA},
        "identity": {"profile": "rtdetr_r18_640", "precision": "fp32", "onnx_sha256": "sha256:" + "cd" * 32,
                     "gpu": "Fake GPU 9000", "tensorrt": "11.2.1.2"},
    }))  # fmt: skip
    return d


@pytest.fixture
def setup(tmp_path, model_dir, monkeypatch):
    clip = write_clip(tmp_path / "clip.mp4", 12)
    root = tmp_path / "data"

    def go(mode="success", **kw):
        monkeypatch.setenv("FAKE_NATIVE_MODE", mode)
        cfg = native.NativeConfig(binary=FAKE, command_prefix=(sys.executable,), model_dir=model_dir)
        return native.run(RunConfig(clip, None, device="cuda", model="rtdetr_r18_640", data_root=root, **kw), cfg)

    return go, root


def staged(root: Path) -> dict:
    (folder,) = (root / "staging").iterdir()
    return json.loads((folder / store.MANIFEST).read_text())


def test_success_is_verified_and_published_with_the_backend(setup):
    go, root = setup
    report = go()
    run_dir = root / "runs" / report["run_id"]
    m = json.loads((run_dir / store.MANIFEST).read_text())
    assert m["status"] == "SUCCEEDED" and list((root / "staging").iterdir()) == []
    assert m["validation"] == {"native_receipt": "PASS", "source_complete": "PASS", "records": "PASS",
                               "video_decode": "PASS", "timings": "PASS", "timestamps": "PASS"}  # fmt: skip
    assert "native_receipt.json" in {a["path"] for a in m["artifacts"]}
    saved = json.loads((run_dir / "report.json").read_text())
    assert metrics.report_schema(saved) == "1.1"
    assert saved["backend"]["name"] == "tensorrt" and saved["backend"]["engine_sha256"] == ENGINE_SHA
    assert saved["environment"]["gpu"] == "Fake GPU 9000" and saved["device"]["used"] == "cuda:0"
    t = saved["timing"]
    assert t["stages"]["detect_track_ms"]["total_s"] == pytest.approx(12 * 1.5 / 1000)  # preprocess + infer + track
    assert set(t["native_stages"]) == {"preprocess_ms", "infer_ms", "track_ms"}
    assert t["detect_track_scope"].startswith("motionman-native") and t["model_load_s"] == 0.5
    spec = json.loads((run_dir / store.RUNSPEC).read_text())
    assert spec["settings"]["tracking"]["class_policy"] == "aware" and spec["settings"]["rendering"]["codec"] == "mp4v"
    assert spec["class_names"] == {"0": "person", "1": "bicycle", "2": "car"}
    data = store.data_root(root)
    (summary,) = runs.list_runs(data)
    opened = runs.open_run(summary, data)  # existing readers accept a native run
    assert opened is not None


def test_a_tail_without_decoder_timestamps_publishes_as_unverified(setup):
    go, root = setup
    report = go("drained_tail")
    m = json.loads((root / "runs" / report["run_id"] / store.MANIFEST).read_text())
    assert m["status"] == "SUCCEEDED" and m["validation"]["timestamps"] == "UNVERIFIED"


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("crash", "without writing native_receipt.json"),
        ("missing_records", "record check failed"),
        ("short_records", "record check failed"),
        ("short_video", "decodes to 11"),
        ("bad_receipt", "run_id 'r_someone_else'"),
        ("status_error", "motionman-native error (exit 0): CUDA error"),
        ("refused", "motionman-native refused (exit 3)"),
    ],
)
def test_failed_or_inconsistent_runs_are_never_published(setup, mode, message):
    go, root = setup
    with pytest.raises(native.NativeError if mode not in ("missing_records", "short_records", "short_video")
                       else pipeline.OutputCheckError, match=re.escape(message)):  # fmt: skip
        go(mode)
    assert list((root / "runs").iterdir()) == []
    m = staged(root)
    assert m["status"] == "FAILED" and message in m["error"]
    assert m["records_sha256"] is None


def test_refusal_is_a_setup_error_for_the_cli(setup):
    go, _ = setup
    with pytest.raises(native.NativeError) as e:
        go("refused")
    assert e.value.exit_code == 2


def test_engine_that_stops_itself_is_interrupted_not_published(setup):
    go, root = setup
    with pytest.raises(KeyboardInterrupt):
        go("interrupted")
    assert list((root / "runs").iterdir()) == []
    m = staged(root)
    assert m["status"] == "INTERRUPTED" and "after frame 6" in m["error"]
    assert m["counts"]["records"] == 6


def test_ctrl_c_stops_the_engine_after_a_whole_frame(setup):
    go, root = setup
    timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            go("slow")
    finally:
        timer.cancel()
    assert list((root / "runs").iterdir()) == []
    m = staged(root)
    assert m["status"] == "INTERRUPTED"
    (folder,) = (root / "staging").iterdir()
    receipt = json.loads((folder / "native_receipt.json").read_text())
    n = receipt["frames"]["processed"]
    assert receipt["status"] == "interrupted" and 0 < n < 12
    assert len((folder / "tracks.jsonl").read_text().splitlines()) == n == m["counts"]["records"]


def test_missing_engine_or_model_fails_before_any_output(tmp_path, model_dir):
    clip = write_clip(tmp_path / "clip.mp4", 3)
    root = tmp_path / "data"
    config = RunConfig(clip, None, device="cuda", model="rtdetr_r18_640", data_root=root)
    with pytest.raises(native.NativeSetupError, match="motionman-native not found"):
        native.run(config, native.NativeConfig(binary=tmp_path / "nope", model_dir=model_dir))
    with pytest.raises(native.NativeSetupError, match="native model file missing"):
        native.run(config, native.NativeConfig(binary=FAKE, command_prefix=(sys.executable,),
                                               model_dir=tmp_path / "empty"))  # fmt: skip
    assert not (root / "staging").exists() or list((root / "staging").iterdir()) == []


def test_cli_backend_flags():
    config, backend, policy = cli.parse(["--source", "x.mp4", "--backend", "native"])
    assert (backend, policy, config.device, config.model) == ("native", "aware", "cuda", "rtdetr_r18_640")
    legacy = cli.parse_args(["--source", "x.mp4"])
    assert (legacy.device, legacy.model) == ("cpu", "yolo11n.pt")  # legacy defaults unchanged
    for bad in (["--backend", "native", "--device", "cpu"], ["--backend", "native", "--telemetry"],
                ["--class-policy", "agnostic"]):  # fmt: skip
        with pytest.raises(SystemExit):
            cli.parse(["--source", "x.mp4", *bad])


def test_native_report_opens_in_the_ui_views(setup):
    from motionman.ui import view

    go, root = setup
    report = go()
    saved = json.loads((root / "runs" / report["run_id"] / "report.json").read_text())
    assert [m.label for m in view.report_metrics(saved)]  # headline numbers render
    rows = view.stage_breakdown(saved)
    assert rows and rows[1].label == "Detect + track"
    assert ui is not None
