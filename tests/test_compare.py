"""Benchmark comparison: intended changes vs confounders, speed-only wording (plan P2 step 5, D6)."""

import json

import pytest

from motionman import benchmark, cli, compare, store

from .test_pipeline import FakeTracker


@pytest.fixture
def make(clip, tmp_path, monkeypatch):
    root = tmp_path / "data"
    code = {"commit": "abc123", "dirty": False}  # the real checkout may be dirty while tests run
    monkeypatch.setattr(store, "code_revision", lambda *a, **k: dict(code))

    def bench(repeats=3, power="ac", percent=None, dirty=False, **kw):
        code["dirty"] = dirty
        state = {"source": power} if percent is None else {"source": power, "percent": percent}
        path, doc = benchmark.run_benchmark(clip, repeats=repeats, data_root=root, make_tracker=FakeTracker,
                                            resolve=lambda d: "cpu", power=lambda: state, **kw)  # fmt: skip
        return doc["benchmark_id"]

    bench.root = root
    return bench


def by_name(doc):
    return {f["name"]: f for f in doc["factors"]}


def test_identical_settings_are_a_controlled_comparison(make):
    a, b = make(), make()
    doc = compare.compare(a, b, data_root=make.root)
    assert doc["confounders"] == [] and doc["notes"] == [] and doc["controlled"]
    assert not any(f["changed"] for f in doc["factors"])
    assert doc["verdict"].endswith("Controlled: nothing else differs.")


def test_an_intended_change_is_not_a_confounder(make):
    a, b = make(imgsz=640), make(imgsz=320)
    doc = compare.compare(a, b, ["imgsz"], data_root=make.root)
    assert by_name(doc)["imgsz"] == {"name": "imgsz", "a": 640, "b": 320, "changed": True, "intended": True}
    assert doc["confounders"] == [] and "intended change(s): imgsz" in doc["verdict"]


def test_undeclared_differences_are_confounders(make):
    a, b = make(imgsz=640, power="ac"), make(imgsz=320, power="battery")
    doc = compare.compare(a, b, ["imgsz"], data_root=make.root)
    assert doc["confounders"] == ["power"] and not doc["controlled"]
    assert "Not controlled: also differs in power" in doc["verdict"]
    undeclared = compare.compare(a, b, data_root=make.root)
    assert undeclared["confounders"] == ["imgsz", "power"] and "no change declared" in undeclared["verdict"]


def test_weak_evidence_is_noted(make):
    a, b = make(repeats=2), make(repeats=3)
    doc = compare.compare(a, b, ["device"], data_root=make.root)
    assert doc["unchanged_intended"] == ["device"]
    assert any("fewer than 3" in n for n in doc["notes"]) and not doc["controlled"]


def test_dirty_code_and_a_wide_battery_gap_are_noted(make):
    doc = compare.compare(make(dirty=True), make(dirty=True), data_root=make.root)
    assert sum("uncommitted changes" in n for n in doc["notes"]) == 2 and not doc["controlled"]
    a, b = make(power="battery", percent=90), make(power="battery", percent=15)
    gap = compare.compare(a, b, data_root=make.root)
    assert any("battery level differs widely (A 90%, B 15%)" in n for n in gap["notes"]) and not gap["controlled"]
    close = compare.compare(a, make(power="battery", percent=80), data_root=make.root)
    assert close["notes"] == [] and close["controlled"]


def test_speed_is_the_only_claim(make):
    doc = compare.compare(make(), make(), data_root=make.root)
    assert "accura" not in doc["verdict"].lower() and "not accuracy" in doc["scope"]
    s = doc["speed"]
    expected = (doc["b"]["fps"]["median"] - doc["a"]["fps"]["median"]) / doc["a"]["fps"]["median"] * 100
    assert s["median_fps_change_pct"] == round(expected, 1) and isinstance(s["ranges_overlap"], bool)
    assert set(s["stage_p50_delta_ms"]) == {"decode_ms", "detect_track_ms", "render_ms", "write_ms"}


def test_bad_inputs_are_clear_errors(make, tmp_path):
    a = make()
    with pytest.raises(compare.CompareError, match="unknown factor"):
        compare.compare(a, a, ["colour"], data_root=make.root)
    with pytest.raises(compare.CompareError, match="can't read benchmark"):
        compare.compare(a, "b_nope", data_root=make.root)

    def broken(cfg):
        raise RuntimeError("no model")

    _, failed = benchmark.run_benchmark(tmp_path / "missing.mp4", repeats=1, data_root=make.root,
                                        make_tracker=broken, power=lambda: {"source": "ac"})  # fmt: skip
    with pytest.raises(compare.CompareError, match="no succeeded trial"):
        compare.compare(a, failed["benchmark_id"], data_root=make.root)


def test_every_listed_factor_is_extracted(make):
    doc = compare.compare(make(), make(), data_root=make.root)
    assert [f["name"] for f in doc["factors"]] == list(compare.FACTORS)


def test_cli_compare_prints_confounders_writes_a_file_and_exits_0(make, capsys):
    a, b = make(power="ac"), make(power="battery")
    code = cli.main(["compare", a, b, "--data-root", str(make.root)])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK and "CONFOUNDER power" in out and "Not controlled" in out
    (path,) = (make.root / "comparisons").glob("*/comparison.json")
    assert json.loads(path.read_text())["confounders"] == ["power"]
    assert cli.main(["compare", a, "b_missing", "--data-root", str(make.root)]) == 2
