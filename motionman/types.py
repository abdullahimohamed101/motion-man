"""Plain records shared by every module (spec §6). No OpenCV or Ultralytics imports here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MotionManError(Exception):
    """An expected failure with a message the user can act on (bad input, device, writer...)."""

    exit_code = 2  # invalid argument, input, device or model; subclasses override for tool failures


@dataclass(frozen=True)
class Track:
    track_id: int | None  # None when the tracker has not assigned an ID; never invented (INV-2)
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # original-frame pixel coordinates


@dataclass(frozen=True)
class Frame:
    index: int  # zero-based
    timestamp_s: float  # index / source_fps
    image: np.ndarray  # BGR, original dimensions


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int | None  # container's estimate; may be missing or wrong


@dataclass(frozen=True)
class Limits:
    """Guardrails, not measured capacity (spec A§6). Recorded in each run; exceeding one is a visible failure."""

    max_file_bytes: int = 512 * 1024**2
    max_duration_s: float = 600.0
    max_long_side: int = 1920
    max_short_side: int = 1080
    max_frames: int = 36_000
    output_budget_bytes: int = 2 * 1024**3


DEFAULT_LIMITS = Limits()


@dataclass(frozen=True)
class RunConfig:
    source: Path
    output_dir: Path | None  # legacy mode writes here; None = canonical run under the data root (D2)
    device: str = "cpu"
    model: str = "yolo11n.pt"
    imgsz: int = 640
    max_frames: int | None = None
    conf: float = 0.25  # D5: explicit, recorded thresholds
    iou: float = 0.7
    limits: Limits = DEFAULT_LIMITS
    data_root: Path | None = None  # canonical mode only; None = MOTIONMAN_DATA_ROOT or ./data
