"""Opt-in telemetry: bounded RSS samples, honest GPU availability, never failing the run (plan P2 D9)."""

import json
import sys
import time
import types

from motionman import cli, pipeline, store, telemetry
from motionman.types import RunConfig

from .test_pipeline import FakeTracker

NO_GPU = (None, {"status": "Unavailable", "reason": "test"})


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_samples_are_written_and_summarised(tmp_path):
    clock, values = Clock(), iter([100, 300, 200] * 3)
    t = telemetry.Telemetry(tmp_path / "t.jsonl", rss=lambda: next(values) * 1024 * 1024, gpu=NO_GPU, clock=clock,
                            interval_s=3600)  # fmt: skip
    t.start()  # one sample at t = 0
    clock.t += 1.5
    t.sample()
    summary = t.stop()  # plus a final one
    assert [r["t_s"] for r in rows(tmp_path / "t.jsonl")] == [0.0, 1.5, 1.5]
    assert summary["samples"] == 3 and summary["rss_mb"] == {"peak": 300.0, "median": 200.0}
    assert summary["gpu"]["status"] == "Unavailable" and summary["error"] is None and not summary["truncated"]
    assert t.stop() == summary  # idempotent


def test_the_thread_samples_about_once_per_interval(tmp_path):
    t = telemetry.Telemetry(tmp_path / "t.jsonl", rss=lambda: 1, gpu=NO_GPU, interval_s=0.01)
    t.start()
    time.sleep(0.1)
    assert t.stop()["samples"] >= 4


def test_samples_are_bounded(tmp_path):
    t = telemetry.Telemetry(tmp_path / "t.jsonl", rss=lambda: 1, gpu=NO_GPU, interval_s=3600, max_samples=3)
    t.start()
    for _ in range(5):
        t.sample()
    summary = t.stop()
    assert summary["samples"] == 3 and summary["truncated"] and len(rows(tmp_path / "t.jsonl")) == 3


def test_a_failing_sampler_stops_quietly_and_says_so(tmp_path):
    def broken():
        raise PermissionError("no access")

    t = telemetry.Telemetry(tmp_path / "t.jsonl", rss=broken, gpu=NO_GPU, interval_s=3600)
    t.start()
    summary = t.stop()
    assert summary["samples"] == 0 and summary["rss_mb"] is None and "PermissionError" in summary["error"]


def test_gpu_counters_are_unavailable_without_cuda_and_nvml(monkeypatch):
    assert telemetry.gpu_probe("cpu") == (None, {"status": "Unavailable", "reason": "device cpu has no GPU counters"})
    broken = types.ModuleType("pynvml")

    def no_driver():
        raise OSError("driver not loaded")

    broken.nvmlInit = no_driver
    monkeypatch.setitem(sys.modules, "pynvml", broken)
    sampler, info = telemetry.gpu_probe("cuda:0")
    assert sampler is None and info == {"status": "Unavailable", "reason": "NVML not usable here (OSError)"}


def test_a_run_with_telemetry_publishes_the_samples(clip, tmp_path):
    root = tmp_path / "data"
    cfg = RunConfig(clip, None, data_root=root, telemetry=True)
    report = pipeline.run(cfg, make_tracker=FakeTracker, resolve=lambda d: "cpu")
    run_dir = store.data_root(root).runs / report["run_id"]
    samples = rows(run_dir / pipeline.TELEMETRY)
    assert len(samples) >= 2 and all(r["rss_bytes"] > 0 for r in samples)
    assert report["telemetry"]["enabled"] and report["telemetry"]["gpu"]["status"] == "Unavailable"
    manifest = json.loads((run_dir / store.MANIFEST).read_text())
    assert pipeline.TELEMETRY in {a["path"] for a in manifest["artifacts"]}


def test_telemetry_is_off_by_default_and_never_changes_the_config_hash(clip, tmp_path):
    plain = pipeline.run(RunConfig(clip, tmp_path / "a"), make_tracker=FakeTracker, resolve=lambda d: "cpu")
    sampled = pipeline.run(RunConfig(clip, tmp_path / "b", telemetry=True), make_tracker=FakeTracker,
                           resolve=lambda d: "cpu")  # fmt: skip
    assert plain["telemetry"] == {"enabled": False} and not (tmp_path / "a" / pipeline.TELEMETRY).exists()
    assert plain["config_sha256"] == sampled["config_sha256"]


def test_cli_flag_turns_it_on():
    assert cli.parse_args(["--source", "x.mp4", "--telemetry"]).telemetry is True
    assert cli.parse_args(["--source", "x.mp4"]).telemetry is False
