"""Timing and counts for one run, and the report.json writer (spec §8)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from time import perf_counter

import numpy as np

import motionman
from motionman.types import RunConfig, VideoInfo

WARMUP_FRAMES = 10
PACKAGES = ("ultralytics", "torch", "torchvision", "opencv-python", "numpy", "lap")
DETECT_TRACK_SCOPE = (
    "Tracker.update: model.track (YOLO detection + ByteTrack association) plus conversion to records; "
    "not inference-only"
)


class Metrics:
    def __init__(self) -> None:
        self.processed_frames = 0
        self.model_load_s: float | None = None
        self.processing_wall_s: float | None = None
        self.detect_track_ms: list[float] = []  # one sample per processed frame; fine for short demo clips
        self._start: float | None = None

    def start(self) -> None:
        self._start = perf_counter()

    def stop(self) -> None:
        """Idempotent: the first call after start() fixes the wall time."""
        if self._start is not None and self.processing_wall_s is None:
            self.processing_wall_s = perf_counter() - self._start

    @property
    def end_to_end_fps(self) -> float | None:
        if not self.processing_wall_s or not self.processed_frames:
            return None
        return self.processed_frames / self.processing_wall_s


def latency_summary(samples_ms: list[float], warmup: int = WARMUP_FRAMES) -> dict:
    """p50/p95 excluding the first ``warmup`` samples; null with a reason when too few remain."""
    measured = samples_ms[warmup:]
    if not measured:
        return {
            "p50": None,
            "p95": None,
            "samples": 0,
            "warmup_excluded": min(warmup, len(samples_ms)),
            "reason": f"{len(samples_ms)} frames processed; percentiles need more than {warmup} (warmup excluded)",
        }
    return {
        "p50": round(float(np.percentile(measured, 50)), 3),
        "p95": round(float(np.percentile(measured, 95)), 3),
        "samples": len(measured),
        "warmup_excluded": warmup,
    }


def build_report(
    *,
    status: str,
    error: str | None,
    config: RunConfig,
    info: VideoInfo,
    metrics: Metrics,
    device_used: str | None,
    output_frames_decoded: int | None = None,
) -> dict:
    fps = metrics.end_to_end_fps
    return {
        "status": status,  # success | error | interrupted; anything but success is a partial run
        "error": error,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "motionman_version": motionman.__version__,
        "source": {
            "path": str(info.path),
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frame_count_reported_by_container": info.frame_count,
        },
        "processed_frames": metrics.processed_frames,
        "output_frames_decoded": output_frames_decoded,
        "max_frames": config.max_frames,
        "model": {
            "weights": config.model,
            "imgsz": config.imgsz,
            "conf": config.conf,
            "iou": config.iou,
            "tracker": "bytetrack.yaml",
            "precision": "fp32",
        },
        "device": {"requested": config.device, "used": device_used},
        "timing": {
            "model_load_s": _round(metrics.model_load_s, 3),
            "processing_wall_s": _round(metrics.processing_wall_s, 3),
            "processing_wall_scope": "perf_counter from before the first decode to after the writer is released",
            "end_to_end_fps": _round(fps, 2),
            "detect_track_ms": latency_summary(metrics.detect_track_ms),
            "detect_track_scope": DETECT_TRACK_SCOPE,
        },
        "environment": environment(device_used),
    }


def environment(device_used: str | None) -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "gpu": None,
        "packages": {name: _version(name) for name in PACKAGES},
    }
    if device_used and device_used.startswith("cuda"):
        import torch

        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def write_report(path: Path, report: dict) -> None:
    """Write via a temp file and rename, so a crash never leaves half a report."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"
