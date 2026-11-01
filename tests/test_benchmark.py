"""Benchmark orchestration: fresh trials, median and range, failed trials visible (plan P2 step 4, D5)."""

import functools
import json
import statistics
import sys
import types

import pytest

from motionman import benchmark, cli, store

from .test_pipeline import FakeTracker

AC = {"source": "ac", "percent": 80}


def bench(clip, tmp_path, **kw):
    kw = {"make_tracker": FakeTracker, "resolve": lambda d: "cpu", "power": lambda: AC, **kw}
    return benchmark.run_benchmark(clip, data_root=tmp_path / "data", **kw)


def test_every_trial_is_a_fresh_published_run(clip, tmp_path):
    FakeTracker.instances.clear()
    path, doc = bench(clip, tmp_path, repeats=3, label="test")
    trials = doc["trials"]
    assert len(FakeTracker.instances) == 3  # one tracker per trial: inference never reused
    assert len({t["run_id"] for t in trials}) == 3 and len({t["records_sha256"] for t in trials}) == 3
    root = store.data_root(tmp_path / "data")
    assert all((root.runs / t["run_id"] / store.MANIFEST).is_file() for t in trials)
    assert json.loads(path.read_text()) == doc and doc["status"] == "complete" and doc["label"] == "test"
    assert doc["reuse"].startswith("none") and doc["source"]["input_id"].startswith("sha256:")


def test_summary_is_the_median_trial_and_the_range(clip, tmp_path):
    _, doc = bench(clip, tmp_path, repeats=3)
    fps = [t["end_to_end_fps"] for t in doc["trials"]]
    s = doc["summary"]
    assert s["fps"] == {"median": round(statistics.median(fps), 2), "min": min(fps), "max": max(fps), "trials": 3}
    assert s["config_consistent"] and s["config_sha256"] == doc["trials"][0]["config_sha256"]
    assert set(s["stage_p50_median_ms"]) == {"decode_ms", "detect_track_ms", "render_ms", "write_ms"}
    assert s["power_sources"] == ["ac"] and all(t["power"] == AC for t in doc["trials"])


def test_a_failed_trial_stays_visible_and_is_left_out_of_the_median(clip, tmp_path):
    calls = []

    def flaky(cfg):
        calls.append(cfg)
        if len(calls) == 2:
            raise RuntimeError("out of memory")
        return FakeTracker(cfg)

    _, doc = bench(clip, tmp_path, repeats=3, make_tracker=flaky)
    assert [t["status"] for t in doc["trials"]] == ["succeeded", "failed", "succeeded"]
    assert doc["trials"][1]["error"] == "RuntimeError: out of memory"
    assert doc["summary"]["failed"] == 1 and doc["summary"]["fps"]["trials"] == 2


def test_when_nothing_succeeds_the_median_is_null_with_a_reason(clip, tmp_path):
    def broken(cfg):
        raise RuntimeError("no model")

    path, doc = bench(clip, tmp_path, repeats=2, make_tracker=broken)
    assert doc["summary"]["fps"] == {"median": None, "reason": "no succeeded trial"} and path.is_file()


def test_ctrl_c_keeps_the_finished_trials(clip, tmp_path):
    calls = []

    def interrupt_second(cfg):
        calls.append(cfg)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return FakeTracker(cfg)

    with pytest.raises(KeyboardInterrupt):
        bench(clip, tmp_path, repeats=3, make_tracker=interrupt_second)
    (path,) = (tmp_path / "data" / "benchmarks").glob("*/benchmark.json")
    doc = json.loads(path.read_text())
    assert doc["status"] == "interrupted" and len(doc["trials"]) == 1 and doc["summary"]["succeeded"] == 1


def test_power_state_is_reported_or_unknown_never_guessed(monkeypatch):
    fake = types.ModuleType("psutil")
    fake.sensors_battery = lambda: types.SimpleNamespace(power_plugged=False, percent=41.6)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert benchmark.power_state() == {"source": "battery", "percent": 42}
    fake.sensors_battery = lambda: None
    assert benchmark.power_state() == {"source": "unknown", "reason": "no battery reported by this host"}


def test_cli_benchmark_prints_the_median_and_exits_by_outcome(clip, tmp_path, monkeypatch, capsys):
    real = benchmark.run_benchmark
    monkeypatch.setattr(benchmark, "run_benchmark", functools.partial(
        real, make_tracker=FakeTracker, resolve=lambda d: "cpu", power=lambda: AC))  # fmt: skip
    code = cli.main(["benchmark", "--source", str(clip), "--repeats", "2", "--data-root", str(tmp_path / "d")])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK and "trial 1:" in out and "trial 2:" in out and "median" in out and "ac" in out
    with pytest.raises(SystemExit) as e:
        cli.main(["benchmark", "--source", str(tmp_path / "missing.mp4")])
    assert e.value.code == 2
