"""Read-only environment checks and input-quality hints (plan P2 D7, D8; MD§12 "Telemetry and doctor", MD§17).

The doctor diagnoses; it never installs packages, changes device settings or touches existing data. The only
files it writes are its own encode probe, in a fresh temporary folder that it removes when done.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np

from motionman import store, video_io
from motionman.types import DEFAULT_LIMITS, Limits, MotionManError, RunConfig, VideoInfo

PASS, WARN, FAIL, UNAVAILABLE = "PASS", "WARN", "FAIL", "UNAVAILABLE"
REQUIRED = ("ultralytics", "torch", "opencv-python", "numpy", "lap", "psutil")
TESTED_PYTHON = (3, 11)
PROBE_FRAMES = 24
# Input-quality hints (D8): documented heuristics, worded as possibilities, never as faults.
HINT_SAMPLES = 60
DARK_LUMA = 40.0  # median mean luma (0–255) of the sampled frames
BLUR_LAPLACIAN = 50.0  # median variance of the Laplacian, measured at 640 px wide
REPEAT_SHARE = 0.20  # share of sampled frames identical to the frame before them


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    next_step: str | None = None
    duration_ms: float | None = None


def run_doctor(
    *,
    source: Path | None = None,
    device: str = "cpu",
    model: str = "yolo11n.pt",
    data_root: Path | None = None,
    load_model: bool = True,
    limits: Limits = DEFAULT_LIMITS,
    probe_devices: Callable[[str], Check] | None = None,
    model_loader: Callable[[RunConfig], object] | None = None,
    requirements: Path | None = store.REPO_ROOT / "requirements.txt",
) -> dict:
    """Run every check and return ``{"checks": [...], "hints": [...], "ok": bool}``. ``ok`` is False only on FAIL."""
    checks = [_python(), _packages(requirements), _weights(model)]
    device_check = (probe_devices or _probe_devices)(device)
    checks.append(device_check)
    if not load_model:
        checks.append(Check("model load", UNAVAILABLE, "skipped (--no-model)"))
    elif device_check.status == FAIL:
        checks.append(Check("model load", UNAVAILABLE, "skipped: the requested device is not available",
                            "Fix the device check first, or run with --device cpu."))  # fmt: skip
    else:
        checks.append(_timed(lambda: _model(model, device, model_loader)))
    scratch = Path(tempfile.mkdtemp(prefix="motionman-doctor-"))
    try:
        checks.append(_timed(lambda: _encode_decode(scratch)))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)  # only the folder this call created
    quality = None
    if source is not None:
        source_check = _timed(lambda: _source(source, limits))
        checks.append(source_check)
        if source_check.status != FAIL:
            quality = input_hints(source)
    root = store.data_root(data_root).path
    checks += [_storage(root), _capacity(root, limits), _timer()]
    return {
        "checks": [asdict(c) for c in checks],
        "input_quality": quality,
        "ok": all(c.status != FAIL for c in checks),
        "read_only": "no packages, settings or existing files changed; the encode probe folder was removed",
    }


def _timed(fn: Callable[[], Check]) -> Check:
    t = time.perf_counter()
    c = fn()
    return Check(c.name, c.status, c.detail, c.next_step, round((time.perf_counter() - t) * 1000, 1))


def _python() -> Check:
    v = platform.python_version()
    if tuple(int(x) for x in v.split(".")[:2]) == TESTED_PYTHON:
        return Check("python", PASS, f"Python {v}")
    return Check("python", WARN, f"Python {v}; MotionMan is tested on 3.11",
                 "Create the venv with `uv venv -p 3.11`.")  # fmt: skip


def _pins(requirements: Path | None) -> dict[str, str]:
    if requirements is None or not requirements.is_file():
        return {}
    pins = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def _packages(requirements: Path | None) -> Check:
    pins, missing, drift = _pins(requirements), [], []
    for name in REQUIRED:
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if name in pins and have != pins[name]:
            drift.append(f"{name} {have} (pinned {pins[name]})")
    if missing:
        return Check("packages", FAIL, f"missing: {', '.join(missing)}",
                     "Install the pinned set: `uv pip install -r requirements-dev.txt`.")  # fmt: skip
    if drift:
        return Check("packages", WARN, "differs from requirements.txt: " + ", ".join(drift),
                     "Reinstall the pins so results compare with the documented benchmarks.")  # fmt: skip
    note = "match requirements.txt" if pins else "installed (requirements.txt not found, versions not compared)"
    return Check("packages", PASS, f"{len(REQUIRED)} required packages {note}")


def _weights(model: str) -> Check:
    path = Path(model)
    if path.is_file():
        return Check(
            "weights", PASS, f"{model} present ({path.stat().st_size / 1e6:.1f} MB, {store.sha256_file(path)[:19]}…)"
        )
    return Check("weights", WARN, f"{model} not found locally; Ultralytics would download it on first use",
                 f"Download {model} once with network access, or pass --model with a local file.")  # fmt: skip


def _probe_devices(requested: str) -> Check:
    """Resolve the device like a run would; also say whether CUDA and MPS exist (MotionMan has no MPS option)."""
    from motionman.tracking import DeviceError, resolve_device

    try:
        import torch

        extra = f"CUDA {'available' if torch.cuda.is_available() else 'not available'}; " + (
            "MPS available but not used (no MotionMan option)" if torch.backends.mps.is_available() else "no MPS"
        )
    except Exception as e:
        return Check("device", FAIL, f"torch failed to import ({type(e).__name__}: {e})",
                     "Reinstall torch from requirements.txt.")  # fmt: skip
    try:
        used = resolve_device(requested)
    except DeviceError as e:
        return Check("device", FAIL, f"{e} ({extra})", "Use --device cpu or auto on this machine.")
    return Check("device", PASS, f"--device {requested} → {used} ({extra})")


def _model(model: str, device: str, loader: Callable[[RunConfig], object] | None) -> Check:
    from motionman import pipeline

    load = loader or pipeline.default_tracker
    used = "cpu" if device == "auto" else device
    try:
        tracker = load(RunConfig(Path("."), None, device=used, model=model))
    except Exception as e:
        return Check("model load", FAIL, f"{type(e).__name__}: {e}",
                     "Check the weights file and the device; the run would fail the same way.")  # fmt: skip
    names = getattr(tracker, "class_names", None)
    return Check(
        "model load", PASS, f"{model} loaded on {used} with warmup" + (f", {len(names)} classes" if names else "")
    )


def _encode_decode(scratch: Path) -> Check:
    """Write a tiny H.264 probe with the run's writer, decode it back, count frames. Lives only in ``scratch``."""
    info = VideoInfo(scratch / "probe.mp4", 64, 48, 24.0, PROBE_FRAMES)
    try:
        writer, codec = video_io.open_writer(info.path, info)
        for i in range(PROBE_FRAMES):
            frame = np.full((48, 64, 3), i * 8, np.uint8)
            writer.write(frame)
        writer.release()
        decoded = video_io.count_frames(info.path)
    except (MotionManError, cv2.error, OSError) as e:
        return Check("encode/decode", FAIL, f"{type(e).__name__}: {e}",
                     "Reinstall opencv-python from requirements.txt; runs need a working video writer.")  # fmt: skip
    if decoded != PROBE_FRAMES:
        return Check("encode/decode", FAIL, f"wrote {PROBE_FRAMES} frames with {codec}, decoded {decoded}",
                     "The video codec is unreliable here; runs would fail their output check.")  # fmt: skip
    if codec != "avc1":
        return Check("encode/decode", WARN, f"H.264 unavailable; fell back to {codec} (may not play in browsers)",
                     "Use an opencv-python build with H.264 for browser playback.")  # fmt: skip
    return Check("encode/decode", PASS, f"{PROBE_FRAMES}-frame {codec} probe written, decoded and removed")


def _source(source: Path, limits: Limits) -> Check:
    try:
        cap, info = video_io.open_reader(source, limits)  # the same preflight a run does
        cap.release()
    except MotionManError as e:
        return Check("source", FAIL, str(e), "Fix the input as the message says; a run would refuse it too.")
    count = f"{info.frame_count:,} frames" if info.frame_count else "frame count unknown"
    return Check("source", PASS, f"{source.name}: {info.width}×{info.height}, {info.fps:.3f} fps, {count}")


def _existing(path: Path) -> Path:
    return path if path.exists() else next((p for p in path.parents if p.exists()), Path("."))


def _storage(root: Path) -> Check:
    near = _existing(root)
    if not os.access(near, os.W_OK):
        return Check("storage", FAIL, f"{near} is not writable", "Choose a writable --data-root.")
    where = str(root) if near == root else f"{root} (not created yet; {near} is writable)"
    return Check("storage", PASS, f"data root {where}")


def _capacity(root: Path, limits: Limits) -> Check:
    free, budget = shutil.disk_usage(_existing(root)).free, limits.output_budget_bytes
    detail = f"{free / 1024**3:.1f} GiB free; a run may write up to {budget / 1024**3:.1f} GiB"
    if free < budget:
        return Check("capacity", FAIL, detail, "Free disk space or use a --data-root on a larger volume.")
    return Check("capacity", PASS, detail)


def _timer() -> Check:
    info = time.get_clock_info("perf_counter")
    detail = f"perf_counter resolution {info.resolution * 1e9:.0f} ns, monotonic {info.monotonic}"
    if info.monotonic and info.resolution <= 1e-6:
        return Check("timer", PASS, detail)
    return Check("timer", WARN, detail, "Stage times below 1 ms will be imprecise on this host.")


def input_hints(source: Path, samples: int = HINT_SAMPLES) -> dict:
    """Measurements and labelled hints from up to ``samples`` evenly spaced frames, each compared with the frame
    before it. Decoded sequentially: seeking is slow on sparse-keyframe clips."""
    cap = cv2.VideoCapture(str(source))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, total // samples) if total else 1
        lumas, sharpness, repeats, n, prev, i = [], [], 0, 0, None, 0
        while n < samples:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0 and i > 0 and prev is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (640, max(1, round(640 * gray.shape[0] / gray.shape[1]))))
                lumas.append(float(gray.mean()))
                sharpness.append(float(cv2.Laplacian(small, cv2.CV_64F).var()))
                repeats += bool(np.array_equal(frame, prev))
                n += 1
            prev, i = frame, i + 1
    finally:
        cap.release()
    if not n:
        return {"samples": 0, "hints": [], "reason": "no frames could be sampled"}
    luma, sharp, share = float(np.median(lumas)), float(np.median(sharpness)), repeats / n
    hints = []
    if luma < DARK_LUMA:
        hints.append({"hint": "possibly dark", "value": round(luma, 1), "threshold": DARK_LUMA,
                      "meaning": "low median brightness; detections may be fewer or less confident"})  # fmt: skip
    if sharp < BLUR_LAPLACIAN:
        hints.append({"hint": "possibly blurry", "value": round(sharp, 1), "threshold": BLUR_LAPLACIAN,
                      "meaning": "little fine detail; could be blur, defocus, or simply a plain scene"})  # fmt: skip
    if share > REPEAT_SHARE:
        hints.append({"hint": "possible repeated frames", "value": round(share, 2), "threshold": REPEAT_SHARE,
                      "meaning": "many frames identical to the previous one; a static scene can do this"})  # fmt: skip
    return {
        "samples": n,
        "median_luma": round(luma, 1),
        "median_laplacian_variance": round(sharp, 1),
        "repeated_share": round(share, 2),
        "hints": hints,
        "note": "hints are possibilities from simple heuristics, not faults",
    }
