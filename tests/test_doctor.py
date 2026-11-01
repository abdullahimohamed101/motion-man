"""Doctor: read-only checks with next steps, and labelled input-quality hints (plan P2 step 6, D7, D8)."""

import json
import tempfile
import types

import cv2
import numpy as np
import pytest

from motionman import cli, doctor

from .conftest import write_clip

DEVICE_OK = doctor.Check("device", doctor.PASS, "--device cpu → cpu (fake)")


def fake_loader(cfg):
    return types.SimpleNamespace(class_names={0: "person", 1: "car"})


def run(tmp_path, **kw):
    weights = tmp_path.parent / f"{tmp_path.name}-fake.pt"  # outside tmp_path, which some tests watch
    if "model" not in kw and not weights.exists():
        weights.write_bytes(b"not real weights")  # the fake loader never reads it
    kw = {"probe_devices": lambda d: DEVICE_OK, "model_loader": fake_loader, "data_root": tmp_path / "data",
          "model": str(weights), **kw}  # fmt: skip
    return doctor.run_doctor(**kw)


def by_name(result):
    return {c["name"]: c for c in result["checks"]}


def write_frames(path, frames, fps=24.0):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def test_a_healthy_environment_passes_every_check(tmp_path):
    result = run(tmp_path)
    checks = by_name(result)
    assert list(checks) == ["python", "packages", "weights", "device", "model load", "encode/decode", "storage",
                            "capacity", "timer"]  # fmt: skip
    assert checks["model load"]["status"] == doctor.PASS and "2 classes" in checks["model load"]["detail"]
    assert checks["encode/decode"]["status"] in (doctor.PASS, doctor.WARN)  # WARN = mp4v fallback without H.264
    assert checks["timer"]["status"] == doctor.PASS and result["ok"] and result["input_quality"] is None


def test_the_doctor_changes_nothing_and_removes_its_own_probe(tmp_path, monkeypatch):
    made = []
    real = tempfile.mkdtemp

    def tracked(**kw):
        made.append(real(dir=tmp_path, **kw))
        return made[-1]

    monkeypatch.setattr(tempfile, "mkdtemp", tracked)
    existing = tmp_path / "data"
    (existing / "runs").mkdir(parents=True)
    (existing / "runs" / "keep.txt").write_text("x")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if "motionman-doctor-" not in str(p))
    run(tmp_path)
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert len(made) == 1 and not any("motionman-doctor-" in str(p) for p in after)
    assert after == before  # nothing created, changed or removed outside its own probe folder


def test_a_missing_data_root_is_not_created(tmp_path):
    result = run(tmp_path, data_root=tmp_path / "new" / "data")
    assert by_name(result)["storage"]["status"] == doctor.PASS and not (tmp_path / "new").exists()


def test_an_unavailable_device_fails_with_a_next_step_and_skips_the_model(tmp_path):
    no_cuda = doctor.Check("device", doctor.FAIL, "CUDA is not available", "Use --device cpu or auto.")
    result = run(tmp_path, device="cuda", probe_devices=lambda d: no_cuda)
    checks = by_name(result)
    assert not result["ok"] and checks["device"]["next_step"] == "Use --device cpu or auto."
    assert checks["model load"]["status"] == doctor.UNAVAILABLE and "device" in checks["model load"]["next_step"]


def test_model_failures_and_skips_are_reported(tmp_path):
    def broken(cfg):
        raise RuntimeError("bad weights")

    failed = by_name(run(tmp_path, model_loader=broken))["model load"]
    assert failed["status"] == doctor.FAIL and "bad weights" in failed["detail"] and failed["next_step"]
    skipped = by_name(run(tmp_path, load_model=False))["model load"]
    assert skipped["status"] == doctor.UNAVAILABLE and "--no-model" in skipped["detail"]


def test_package_pins_are_compared(tmp_path, monkeypatch):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==0.0.1  # wrong on purpose\n")
    drift = by_name(run(tmp_path, requirements=req))["packages"]
    assert drift["status"] == doctor.WARN and "numpy" in drift["detail"] and "pinned 0.0.1" in drift["detail"]
    monkeypatch.setattr(doctor, "REQUIRED", ("numpy", "not-a-real-package"))
    missing = by_name(run(tmp_path, requirements=req))["packages"]
    assert missing["status"] == doctor.FAIL and "not-a-real-package" in missing["detail"]


def test_missing_weights_warn_and_are_never_loaded_or_downloaded(tmp_path):
    def must_not_load(cfg):
        raise AssertionError("loading missing weights would download them")

    checks = by_name(run(tmp_path, model=str(tmp_path / "nope.pt"), model_loader=must_not_load))
    assert checks["weights"]["status"] == doctor.WARN and checks["weights"]["next_step"]
    assert checks["model load"]["status"] == doctor.UNAVAILABLE and "download" in checks["model load"]["detail"]


def test_a_bad_source_fails_like_a_run_would(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    result = run(tmp_path, source=bad)
    assert by_name(result)["source"]["status"] == doctor.FAIL and not result["ok"] and result["input_quality"] is None


def test_a_good_source_passes_and_is_measured(tmp_path):
    result = run(tmp_path, source=write_clip(tmp_path / "clip.mp4", 40))
    assert by_name(result)["source"]["status"] == doctor.PASS and "64×48" in by_name(result)["source"]["detail"]
    q = result["input_quality"]
    assert q["samples"] > 0 and {"median_luma", "median_laplacian_variance", "repeated_share"} <= set(q)


def noise(level, seed):
    return np.random.default_rng(seed).integers(0, level, (96, 128, 3), dtype=np.uint8)


def hints(path):
    return {h["hint"] for h in doctor.input_hints(path)["hints"]}


def test_hints_name_dark_blurry_and_repeated_input(tmp_path):
    dark = write_frames(tmp_path / "dark.mp4", [noise(30, i) for i in range(40)])
    assert "possibly dark" in hints(dark)
    smooth = np.tile(np.linspace(80, 200, 128, dtype=np.uint8), (96, 1))
    blurry = write_frames(tmp_path / "blurry.mp4", [np.dstack([np.roll(smooth, i, 1)] * 3) for i in range(40)])
    assert hints(blurry) == {"possibly blurry"}
    still = noise(255, 7)
    repeated = write_frames(tmp_path / "repeated.mp4", [still] * 40)
    assert "possible repeated frames" in hints(repeated) and "possibly dark" not in hints(repeated)
    lively = write_frames(tmp_path / "lively.mp4", [noise(255, i) for i in range(40)])
    assert hints(lively) == set()


def test_cli_doctor_prints_next_steps_and_exits_by_outcome(tmp_path, monkeypatch, capsys, clip):
    no_cuda = doctor.Check("device", doctor.FAIL, "CUDA is not available", "Use --device cpu or auto.")
    monkeypatch.setattr(doctor, "_probe_devices", lambda d: no_cuda if d == "cuda" else DEVICE_OK)
    root = ["--data-root", str(tmp_path / "data"), "--no-model"]
    assert cli.main(["doctor", "--source", str(clip), *root]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "PASS        source" in out and "input quality" in out and out.rstrip().endswith("ok")
    assert cli.main(["doctor", "--device", "cuda", *root]) == cli.EXIT_FAILURE
    assert "next: Use --device cpu or auto." in capsys.readouterr().out
    assert cli.main(["doctor", "--json", *root]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["ok"] is True


@pytest.mark.parametrize("name", ["python", "timer"])
def test_host_checks_never_fail_on_a_supported_host(tmp_path, name):
    assert by_name(run(tmp_path))[name]["status"] != doctor.FAIL
