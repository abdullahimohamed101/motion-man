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

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy],
        }


def frame_record(index: int, timestamp_s: float, tracks: list[Track]) -> dict:
    """One JSONL line (spec §6): written for every processed frame, even with no objects."""
    return {"frame_index": index, "timestamp_s": round(timestamp_s, 6), "objects": [t.to_dict() for t in tracks]}


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
