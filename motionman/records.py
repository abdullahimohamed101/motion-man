"""Versioned frame records (spec A§4): the v1.0 writer, validation, and a reader for pre-v1.0 output.

One record per decoded source frame. Boxes are in source pixels as [x1, y1, x2, y2] with finite values,
positive extent and in-bounds coordinates. Pre-v1.0 lines (``frame_index`` / ``timestamp_s``) are adapted
on read, never rewritten, and carry ``legacy: True`` with unknown fields left as None (not invented).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from pathlib import Path

from motionman.types import MotionManError, Track

SCHEMA_VERSION = "1.0"
OVERRUN_TOLERANCE_PX = 0.5  # small numerical overrun past the image edge is clipped; anything larger is rejected
FRAME_STATUSES = {"INFERRED"}  # NOT_INFERRED arrives with sampling experiments (P5)


class RecordError(MotionManError):
    """A frame record is malformed or uses an unsupported schema version."""


def source_t_us(frame_index: int, fps: float) -> int:
    """Source time for validated constant-frame-rate input (A§4). Never derived from wall clock."""
    return round(frame_index * 1_000_000 / fps)


def frame_record(run_id: str, frame_index: int, fps: float, width: int, height: int, tracks: Sequence[Track]) -> dict:
    """Build and validate one v1.0 record. Raises RecordError for a box outside the documented policy."""
    objects = [
        {
            "observation_id": f"{frame_index}:{n}",
            "track_id": t.track_id,
            "class_id": t.class_id,
            "class_name": t.class_name,
            "confidence": round(t.confidence, 4),
            "bbox_xyxy": [round(v, 2) for v in _checked_box(t.bbox_xyxy, width, height)],
            "mask_ref": None,
        }
        for n, t in enumerate(tracks)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source_frame_index": frame_index,
        "source_t_us": source_t_us(frame_index, fps),
        "sample_index": frame_index,  # every-frame mode: one inference per decoded frame
        "frame_status": "INFERRED",
        "source_width": width,
        "source_height": height,
        "objects": objects,
    }


def validate(record: dict) -> None:
    """Check a v1.0 record's shape and values; raise RecordError with the first problem found."""
    if record.get("schema_version") != SCHEMA_VERSION:
        raise RecordError(f"unsupported schema_version {record.get('schema_version')!r}; expected {SCHEMA_VERSION}")
    for key, kind in (("source_frame_index", int), ("source_t_us", int), ("source_width", int), ("source_height", int)):
        if not isinstance(record.get(key), kind) or isinstance(record.get(key), bool) or record[key] < 0:
            raise RecordError(f"{key} must be a non-negative integer, got {record.get(key)!r}")
    if record.get("frame_status") not in FRAME_STATUSES:
        raise RecordError(f"unknown frame_status {record.get('frame_status')!r}")
    if not isinstance(record.get("objects"), list):
        raise RecordError("objects must be a list")
    seen = set()
    for obj in record["objects"]:
        oid = obj.get("observation_id")
        if not isinstance(oid, str) or oid in seen:
            raise RecordError(f"observation_id must be a unique string within the frame, got {oid!r}")
        seen.add(oid)
        tid = obj.get("track_id")
        if tid is not None and (not isinstance(tid, int) or isinstance(tid, bool)):
            raise RecordError(f"track_id must be an integer or null, got {tid!r}")
        _checked_box(obj.get("bbox_xyxy"), record["source_width"], record["source_height"], clip=False)


def read_records(path: Path) -> Iterator[dict]:
    """Yield v1.0-shaped records from a JSONL file, adapting pre-v1.0 lines. Unknown versions fail clearly."""
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise RecordError(f"{path}:{line_no}: not valid JSON ({e.msg}); the file may be truncated") from e
            if "schema_version" in raw:
                try:
                    validate(raw)
                except RecordError as e:
                    raise RecordError(f"{path}:{line_no}: {e}") from e
                yield raw
            elif "frame_index" in raw and "timestamp_s" in raw:
                yield _from_legacy(raw)
            else:
                raise RecordError(f"{path}:{line_no}: unrecognised record format")


def _from_legacy(raw: dict) -> dict:
    """Adapt an MVP line. Source dimensions and run_id were not recorded, so they stay None."""
    index = raw["frame_index"]
    return {
        "schema_version": SCHEMA_VERSION,
        "legacy": True,
        "run_id": None,
        "source_frame_index": index,
        "source_t_us": round(raw["timestamp_s"] * 1_000_000),  # MVP stored seconds rounded to 6 decimals
        "sample_index": index,
        "frame_status": "INFERRED",
        "source_width": None,
        "source_height": None,
        "objects": [
            {**obj, "observation_id": f"{index}:{n}", "mask_ref": None} for n, obj in enumerate(raw["objects"])
        ],
    }


def _checked_box(box, width: int, height: int, clip: bool = True) -> tuple[float, float, float, float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise RecordError(f"bbox_xyxy must have 4 values, got {box!r}")
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        raise RecordError(f"bbox_xyxy values must be numbers, got {box!r}") from None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise RecordError(f"bbox_xyxy must be finite, got {box!r}")
    tol = OVERRUN_TOLERANCE_PX
    if x1 < -tol or y1 < -tol or x2 > width + tol or y2 > height + tol:
        raise RecordError(f"bbox_xyxy {box!r} lies outside the {width}x{height} source image")
    if clip:
        x1, y1, x2, y2 = max(x1, 0.0), max(y1, 0.0), min(x2, float(width)), min(y2, float(height))
    if x2 <= x1 or y2 <= y1:
        raise RecordError(f"bbox_xyxy {box!r} has no positive extent")
    return x1, y1, x2, y2
