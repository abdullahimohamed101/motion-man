"""Open and validate the source and output videos; yield frames (spec §6, §8)."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from pathlib import Path

import cv2

from motionman.types import DEFAULT_LIMITS, Frame, Limits, MotionManError, VideoInfo

log = logging.getLogger(__name__)

# avc1 (H.264) plays in browsers and QuickTime; mp4v is the fallback when the build lacks H.264 (D4).
_CODECS = ("avc1", "mp4v")
# A positive FPS is not proof of constant frame rate (spec A§6): frame timestamps must sit on i / fps.
TIMESTAMP_TOLERANCE_FRAMES = 0.5
CFR_PREFLIGHT_FRAMES = 240


class VideoError(MotionManError):
    """A video could not be read or written; the message says why."""


def open_reader(path: Path, limits: Limits = DEFAULT_LIMITS) -> tuple[cv2.VideoCapture, VideoInfo]:
    """Open and validate a source before any inference: metadata, limits, then a constant-frame-rate check
    on the first frames. The returned capture is positioned at frame 0."""
    if not path.is_file():
        raise VideoError(f"source not found: {path}")
    if path.stat().st_size > limits.max_file_bytes:
        raise VideoError(
            f"{path}: file is {_mib(path.stat().st_size)}; the limit is {_mib(limits.max_file_bytes)}. "
            "Choose a smaller clip or raise the limit."
        )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    problem = None
    if not math.isfinite(fps) or fps <= 0:
        problem = f"invalid FPS ({fps}); only constant-frame-rate video is supported"
    elif width <= 0 or height <= 0:
        problem = f"invalid dimensions {width}x{height}"
    elif width % 2 or height % 2:
        problem = f"odd dimensions {width}x{height}; width and height must be even"
    elif max(width, height) > limits.max_long_side or min(width, height) > limits.max_short_side:
        problem = f"{width}x{height} exceeds the {limits.max_long_side}x{limits.max_short_side} limit"
    elif count > limits.max_frames:
        problem = f"{count} frames exceeds the {limits.max_frames}-frame limit"
    elif count > 0 and count / fps > limits.max_duration_s:
        problem = f"video is {count / fps / 60:.1f} min long; the limit is {limits.max_duration_s / 60:g} min"
    if problem:
        cap.release()
        raise VideoError(f"{path}: {problem}")
    try:
        _check_timebase(cap, fps, CFR_PREFLIGHT_FRAMES)
    finally:
        cap.release()
    cap = cv2.VideoCapture(str(path))  # reopen instead of seeking back: seeks are not guaranteed frame-exact
    if not cap.isOpened():
        raise VideoError(f"cannot reopen video: {path}")
    return cap, VideoInfo(path, width, height, fps, count if count > 0 else None)


def read_frames(
    cap: cv2.VideoCapture, fps: float, max_frames: int | None = None, frame_limit: int | None = None
) -> Iterator[Frame]:
    """Yield frames in order, checking every timestamp against constant FPS. ``max_frames`` is the user's
    requested stop; ``frame_limit`` is a guardrail whose breach is an error, never a silent truncation."""
    index = 0
    while max_frames is None or index < max_frames:
        ok, image = cap.read()
        if not ok:
            return
        if frame_limit is not None and index >= frame_limit:
            raise VideoError(f"video exceeds the {frame_limit}-frame limit; nothing past it was processed")
        _check_frame_time(index, cap.get(cv2.CAP_PROP_POS_MSEC), fps)
        yield Frame(index, index / fps, image)
        index += 1


def timestamp_drift_frames(index: int, pos_ms: float, fps: float) -> float:
    """How far a frame's container timestamp is from ``index / fps``, in frames."""
    return abs(pos_ms - index * 1000.0 / fps) * fps / 1000.0


def _check_frame_time(index: int, pos_ms: float, fps: float) -> None:
    if timestamp_drift_frames(index, pos_ms, fps) > TIMESTAMP_TOLERANCE_FRAMES:
        raise VideoError(
            f"This video uses an unsupported timebase: frame {index} is at {pos_ms:.1f} ms, expected "
            f"{index * 1000.0 / fps:.1f} ms at a constant {fps:.3f} fps. Choose a constant-frame-rate clip."
        )


def _check_timebase(cap: cv2.VideoCapture, fps: float, frames: int) -> None:
    decoded = 0
    for index in range(frames):
        if not cap.grab():
            break
        _check_frame_time(index, cap.get(cv2.CAP_PROP_POS_MSEC), fps)
        decoded += 1
    if decoded == 0:
        raise VideoError("no decodable frames")


def _mib(n: int) -> str:
    return f"{n / 1024**2:.0f} MiB"


def open_writer(path: Path, info: VideoInfo) -> cv2.VideoWriter:
    """Writer with the source's dimensions and playback FPS, whatever the processing speed."""
    for codec in _CODECS:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), info.fps, (info.width, info.height))
        if writer.isOpened():
            if codec != _CODECS[0]:
                log.warning("H.264 writer unavailable; using %s (may not play in browsers)", codec)
            return writer
        writer.release()
    raise VideoError(f"cannot open video writer: {path}")


def count_frames(path: Path) -> int:
    """Decode every frame and count them; the container's frame count is not trusted."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"cannot reopen output video: {path}")
        n = 0
        while cap.grab():
            n += 1
        return n
    finally:
        cap.release()
