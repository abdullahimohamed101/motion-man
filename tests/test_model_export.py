"""G1 model manifest checks (spec C§5, NG-6). Pure comparison only: no torch, GPU or download (INV-8)."""

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("model_export_common", ROOT / "tools" / "model_export" / "common.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


def _consistent_manifest(profile):
    exp = profile["export"]
    return {
        "profile": profile["profile"],
        "upstream": {"commit": profile["upstream"]["commit"], "config": profile["upstream"]["config"]},
        "checkpoint": {"sha256": profile["checkpoint"]["sha256"]},
        "onnx": {
            "opset": exp["opset"],
            "tensors": {
                k: {n: {"dtype": s["dtype"], "shape": s["shape"]} for n, s in exp[k].items()}
                for k in ("inputs", "outputs")
            },
        },
        "classes": {"names": [f"c{i}" for i in range(80)]},
    }


def test_profile_pins_upstream_checkpoint_and_tensor_contract():
    p = common.load_profile("rtdetr_r18_640")
    assert len(p["upstream"]["commit"]) == 40
    assert p["checkpoint"]["sha256"].startswith("sha256:") and len(p["checkpoint"]["sha256"]) == 71
    assert list(p["export"]["inputs"]) == ["images", "orig_target_sizes"]
    assert list(p["export"]["outputs"]) == ["labels", "boxes", "scores"]
    assert p["reference"]["tf32"] is False


def test_consistent_manifest_passes():
    p = common.load_profile("rtdetr_r18_640")
    assert common.compare_manifest(p, _consistent_manifest(p)) == []


def test_every_identity_change_is_reported():
    p = common.load_profile("rtdetr_r18_640")
    base = _consistent_manifest(p)
    edits = {
        "checkpoint sha256": lambda m: m["checkpoint"].update(sha256="sha256:" + "0" * 64),
        "upstream commit": lambda m: m["upstream"].update(commit="f" * 40),
        "opset": lambda m: m["onnx"].update(opset=17),
        "boxes dtype": lambda m: m["onnx"]["tensors"]["outputs"]["boxes"].update(dtype="float16"),
        "outputs names": lambda m: m["onnx"]["tensors"]["outputs"].pop("scores"),
        "class vocabulary": lambda m: m["classes"]["names"].pop(),
    }
    for what, edit in edits.items():
        m = copy.deepcopy(base)
        edit(m)
        problems = common.compare_manifest(p, m)
        assert any(what in x for x in problems), (what, problems)


def test_golden_image_hashes_cover_every_golden_frame():
    golden = json.loads(common.GOLDEN.read_text())
    frames = json.loads((ROOT / "configs" / "reference" / "inputs.json").read_text())["inputs"]["crosswalk"]
    names = set(golden["images"])
    assert {f"crosswalk_{i:05d}" for i in frames["golden_frames"]} <= names
    assert {"synthetic_portrait", "synthetic_landscape", "synthetic_pattern"} <= names
    assert all(v["sha256"].startswith("sha256:") for v in golden["images"].values())
