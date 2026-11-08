"""G1 model manifest checks (spec C§5, NG-6). Pure comparison only: no torch, GPU or download (INV-8)."""

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"model_export_{name}", ROOT / "tools" / "model_export" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


common = _load("common")
parity = _load("parity")
build_engine = _load("build_engine")


def test_engine_identity_refuses_any_difference():
    expected = {"onnx_sha256": "sha256:aa", "tensorrt": "11.2.1.2", "gpu": "RTX", "precision": "fp32"}
    assert build_engine.compare_identity(expected, {"identity": dict(expected)}) == []
    for key, other in [
        ("onnx_sha256", "sha256:bb"),
        ("tensorrt", "11.4.0.106"),
        ("gpu", "RTX 4090"),
        ("precision", "fp16"),
    ]:
        problems = build_engine.compare_identity(expected, {"identity": {**expected, key: other}})
        assert len(problems) == 1 and problems[0].startswith(key), problems
    assert build_engine.compare_identity(expected, {}) != []  # no identity at all is refused


def test_engine_shape_spec_matches_trtexec_syntax():
    spec = build_engine.shape_spec({"images": [1, 3, 640, 640], "orig_target_sizes": [1, 2]})
    assert spec == "images:1x3x640x640,orig_target_sizes:1x2"
    assert build_engine.PRECISION_FLAGS["fp32"] == ["--noTF32"]


def _det(label, score, box):
    return {"label": label, "class_name": f"c{label}", "score": score, "box_xyxy": box}


def _doc(dets):
    return {"images": {"img": {"non_finite": 0, "detections": dets}}}


def _cmp(ref, cand):
    return parity.compare(_doc(ref), _doc(cand), 0.25, 0.99, 0.005, 0.02, 1024)


def test_parity_matches_by_class_and_geometry_not_position():
    ref = [_det(0, 0.9, [0, 0, 100, 100]), _det(2, 0.8, [200, 200, 300, 260]), _det(0, 0.1, [5, 5, 9, 9])]
    cand = [_det(2, 0.801, [200, 200, 300, 260.1]), _det(0, 0.899, [0, 0, 100, 100])]  # reordered, tiny drift
    res = _cmp(ref, cand)
    assert (res["ref_matched"], res["ref_detections"]) == (2, 2)  # the 0.1 detection is below the threshold
    assert res["cand_match_rate"] == 1.0 and res["max_score_error"] < 0.0011


def test_parity_rejects_wrong_class_moved_box_and_extra_boxes():
    ref = [_det(0, 0.9, [0, 0, 100, 100])]
    assert _cmp(ref, [_det(1, 0.9, [0, 0, 100, 100])])["ref_matched"] == 0  # same box, other class
    assert _cmp(ref, [_det(0, 0.9, [0, 0, 100, 102])])["ref_matched"] == 0  # IoU 0.98 < 0.99
    extra = _cmp(ref, [_det(0, 0.9, [0, 0, 100, 100]), _det(0, 0.7, [400, 400, 450, 450])])
    assert extra["ref_match_rate"] == 1.0 and extra["cand_match_rate"] == 0.5


def test_parity_notes_near_threshold_and_tiny_misses():
    res = _cmp([_det(0, 0.26, [0, 0, 20, 20])], [])
    assert res["ref_match_rate"] == 0.0
    assert "near-threshold" in res["notes"][0] and "tiny" in res["notes"][0]


def test_profile_freezes_parity_gates_from_spec():
    par = common.load_profile("rtdetr_r18_640")["parity"]
    assert par["gates"]["fp32"] == {"min_iou": 0.99, "max_score_error": 0.005, "min_match_rate": 0.99}
    assert par["gates"]["mixed"] == {"min_iou": 0.95, "max_score_error": 0.02, "min_match_rate": 0.99}
    assert par["preprocess_gates"]["resize_mae"] == 1 / 255 and par["preprocess_gates"]["resize_p99"] == 3 / 255


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
