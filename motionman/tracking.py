"""The only module that touches Ultralytics. One instance owns tracker state for one run (INV-1)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from motionman.types import MotionManError, Track

log = logging.getLogger(__name__)


class DeviceError(MotionManError):
    pass


class ModelError(MotionManError):
    pass


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def resolve_device(requested: str, cuda_available: Callable[[], bool] = _cuda_available) -> str:
    """cpu stays cpu without importing torch; auto falls back to cpu (logged); explicit cuda must exist."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_available():
            raise DeviceError("--device cuda requested but CUDA is not available on this machine; use --device cpu")
        return "cuda"
    if requested == "auto":
        if cuda_available():
            log.info("device auto → cuda")
            return "cuda"
        log.info("device auto → cpu (CUDA not available)")
        return "cpu"
    raise DeviceError(f"unknown device {requested!r}; expected cpu, cuda or auto")


class Tracker:
    def __init__(self, model: str, device: str, imgsz: int, conf: float, iou: float) -> None:
        # Imported here so offline tests never load torch (INV-8).
        from ultralytics import YOLO

        try:
            self._model = YOLO(model)
        except Exception as e:  # missing file, failed download, corrupt weights
            raise ModelError(f"cannot load model {model!r}: {e}") from e
        self.class_names = dict(self._model.names)  # recorded as a class-map hash in the RunSpec
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
            "quantize": 32,  # FP32, as recorded in the report (replaces the deprecated `half`)
            "verbose": False,
        }
        # YOLO() is lazy: the first call imports the predictor stack and loads weights (~2.5 s on CPU).
        # One predict on a blank frame moves that into model load time. Predict mode never runs the
        # tracker callbacks, so tracker state and IDs are unchanged.
        predict_kwargs = {k: v for k, v in self._kwargs.items() if k not in ("persist", "tracker")}
        self._model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), **predict_kwargs)
        if self._cuda_sync is not None:
            self._cuda_sync()

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
