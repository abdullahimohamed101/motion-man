"""The only module that touches Ultralytics. One instance owns tracker state for one run (INV-1)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from motionman.types import Track


class Tracker:
    def __init__(self, model: str, device: str, imgsz: int, conf: float, iou: float) -> None:
        # Imported here so offline tests never load torch (INV-8).
        from ultralytics import YOLO

        self._model = YOLO(model)
        self._cuda_sync = None
        if device.startswith("cuda"):
            import torch

            self._cuda_sync = torch.cuda.synchronize
        self._kwargs = {
            "persist": True,
            "tracker": "bytetrack.yaml",
            "device": device,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "half": False,  # FP32, as recorded in the report
            "verbose": False,
        }

    def update(self, image: np.ndarray) -> list[Track]:
        result = self._model.track(image, **self._kwargs)[0]
        tracks = tracks_from_boxes(result.boxes, result.names)
        if self._cuda_sync is not None:
            self._cuda_sync()  # GPU work is asynchronous; finish it so the caller's timing is real (spec §8)
        return tracks


def tracks_from_boxes(boxes: Any, names: Mapping[int, str]) -> list[Track]:
    """Convert an Ultralytics ``Boxes`` (or any object with the same fields) to plain records.

    ``boxes.id`` is None when the tracker assigned no IDs this frame; those objects keep a None ID.
    """
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.tolist()
    cls = boxes.cls.tolist()
    conf = boxes.conf.tolist()
    ids = [None] * len(xyxy) if boxes.id is None else [int(i) for i in boxes.id.tolist()]
    return [
        Track(
            track_id=tid,
            class_id=int(c),
            class_name=names.get(int(c), str(int(c))),
            confidence=float(p),
            bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
        )
        for b, c, p, tid in zip(xyxy, cls, conf, ids, strict=True)
    ]
