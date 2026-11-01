"""Plain records shared by every module (spec §6). No OpenCV or Ultralytics imports here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
class RunConfig:
    source: Path
    output_dir: Path
    device: str = "cpu"
    model: str = "yolo11n.pt"
    imgsz: int = 640
    max_frames: int | None = None
    conf: float = 0.25  # D5: explicit, recorded thresholds
    iou: float = 0.7
