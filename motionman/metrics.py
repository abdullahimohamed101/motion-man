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
from motionman.types import LEGACY_BACKEND, RunConfig, VideoInfo

WARMUP_FRAMES = 10
REPORT_SCHEMA = "1.1"  # 1.1 adds schema_version and backend (ADR-006); a report without a version is 1.0
PACKAGES = ("ultralytics", "torch", "torchvision", "opencv-python", "numpy", "lap")
STAGES = ("decode_ms", "detect_track_ms", "render_ms", "write_ms")
STAGE_SCOPES = {
    "decode_ms": "time inside next() on the frame reader, including the source timestamp check",
    "render_ms": "trail update plus drawing boxes, labels and trails",
    "write_ms": "writing the annotated frame to the video encoder plus the JSONL record line",
}
SETUP_SCOPE = (
    "from run start to just before the first decode: input checks, import and hashing, model load, writer open"
)
FINALIZE_SCOPE = (
    "from writer release to the report write: output decode check, records validation, hashing; "
    "the final atomic rename into runs/ happens afterwards and is not included"
)
WARMUP_POLICY = "exclude_first_inferred"
DETECT_TRACK_SCOPE = (
    "Tracker.update: model.track (YOLO detection + ByteTrack association) plus conversion to records; "
    "not inference-only"
)


class Metrics:
    def __init__(self) -> None:
        self.processed_frames = 0  # frames fully handled: tracked, drawn, written to video and JSONL
        self.jsonl_records = 0
        self.decoded_frames = 0  # frames read from the source
        self.inferred_frames = 0  # frames the tracker processed
        self.model_load_s: float | None = None
        self.processing_wall_s: float | None = None
        # One float per processed frame and stage: bounded by the input frame limit (INV-4 is about frames).
        self.decode_ms: list[float] = []
        self.detect_track_ms: list[float] = []
        self.render_ms: list[float] = []
        self.write_ms: list[float] = []
        self.setup_s: float | None = None
        self.finalize_s: float | None = None
        self.job_wall_s: float | None = None
        self._job: float | None = None
        self._start: float | None = None
        self._stopped: float | None = None
        self.telemetry: dict = {"enabled": False}

    def begin_job(self) -> None:
        self._job = perf_counter()

    def start(self) -> None:
        self._start = perf_counter()
        if self._job is not None:
            self.setup_s = self._start - self._job

    def add_frame(self, decode: float, detect_track: float, render: float, write: float) -> dict:
        """Record one processed frame's stage times (ms); returns the rounded row for timings.jsonl."""
        row = dict(zip(STAGES, (decode, detect_track, render, write), strict=True))
        for name, value in row.items():
            getattr(self, name).append(value)
        return {k: round(v, 3) for k, v in row.items()}

    def stop(self) -> None:
        """Idempotent: the first call after start() fixes the wall time."""
        if self._start is not None and self.processing_wall_s is None:
            self._stopped = perf_counter()
            self.processing_wall_s = self._stopped - self._start

    def finish(self) -> None:
        """Idempotent: fixes finalize (from stop) and the whole job's wall time."""
        if self._job is None or self.job_wall_s is not None:
            return
        now = perf_counter()
        if self._stopped is not None:
            self.finalize_s = now - self._stopped
        self.job_wall_s = now - self._job

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


def stage_summary(samples_ms: list[float], warmup: int = WARMUP_FRAMES) -> dict:
    """Steady-state percentiles (warmup excluded) plus the total over every frame, for share of wall time."""
    return {**latency_summary(samples_ms, warmup), "total_s": round(sum(samples_ms) / 1000, 3)}


def timing_section(metrics: Metrics) -> dict:
    wall = metrics.processing_wall_s
    stages = {name: stage_summary(getattr(metrics, name)) for name in STAGES}
    unattributed = None
    if wall and metrics.processed_frames:
        attributed = sum(sum(getattr(metrics, name)) for name in STAGES) / 1000
        unattributed = {
            "total_s": round(wall - attributed, 3),
            "share_of_wall": round((wall - attributed) / wall, 4),
            "scope": "processing wall outside the four stages: record building, timing rows, progress updates, "
            "the final next() that ends the loop, writer release and closing the JSONL and timings files",
        }
    return {
        "model_load_s": _round(metrics.model_load_s, 3),
        "setup_s": _round(metrics.setup_s, 3),
        "setup_scope": SETUP_SCOPE,
        "processing_wall_s": _round(wall, 3),
        "processing_wall_scope": "perf_counter from before the first decode to after the writer is released",
        "finalize_s": _round(metrics.finalize_s, 3),
        "finalize_scope": FINALIZE_SCOPE,
        "job_wall_s": _round(metrics.job_wall_s, 3),
        "end_to_end_fps": _round(metrics.end_to_end_fps, 2),
        "detect_track_ms": latency_summary(metrics.detect_track_ms),
        "detect_track_scope": DETECT_TRACK_SCOPE,
        "stages": stages,
        "stage_scopes": {"detect_track_ms": DETECT_TRACK_SCOPE, **STAGE_SCOPES},
        "unattributed": unattributed,
        "warmup": {
            "policy": WARMUP_POLICY,
            "frames": WARMUP_FRAMES,
            "samples": max(0, metrics.processed_frames - WARMUP_FRAMES),
        },  # fmt: skip
        "instrumentation": "wall clock (perf_counter) per stage; no device synchronization",
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
    backend: dict | None = None,
    tracker: str = "bytetrack.yaml",
) -> dict:
    backend = dict(backend or LEGACY_BACKEND)
    return {
        "schema_version": REPORT_SCHEMA,
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
        "jsonl_records": metrics.jsonl_records,
        "output_frames_decoded": output_frames_decoded,
        "max_frames": config.max_frames,
        "model": {
            "weights": config.model,
            "imgsz": config.imgsz,
            "conf": config.conf,
            "iou": config.iou,
            "tracker": tracker,
            "precision": backend["precision"],
        },
        "backend": backend,
        "device": {"requested": config.device, "used": device_used},
        "timing": timing_section(metrics),
        "telemetry": metrics.telemetry,
        "environment": environment(device_used),
    }


def report_schema(report: dict) -> str:
    """A report's schema version; reports written before 1.1 carry none and are 1.0."""
    return str(report.get("schema_version") or "1.0")


def report_backend(report: dict) -> dict:
    """The backend a report records; a 1.0 report was produced by the legacy backend at its recorded precision."""
    if report.get("backend"):
        return dict(report["backend"])
    return {**LEGACY_BACKEND, "precision": (report.get("model") or {}).get("precision") or LEGACY_BACKEND["precision"]}


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
