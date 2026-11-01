"""Open and validate the source and output videos; yield frames (spec §6, §8)."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from pathlib import Path

import cv2

from motionman.types import Frame, VideoInfo

log = logging.getLogger(__name__)

# avc1 (H.264) plays in browsers and QuickTime; mp4v is the fallback when the build lacks H.264 (D4).
_CODECS = ("avc1", "mp4v")


class VideoError(Exception):
    """A video could not be read or written; the message says why."""


def open_reader(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    if not path.is_file():
        raise VideoError(f"source not found: {path}")
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
    if problem:
        cap.release()
        raise VideoError(f"{path}: {problem}")
    return cap, VideoInfo(path, width, height, fps, count if count > 0 else None)


def read_frames(cap: cv2.VideoCapture, fps: float, max_frames: int | None = None) -> Iterator[Frame]:
    index = 0
    while max_frames is None or index < max_frames:
        ok, image = cap.read()
        if not ok:
            return
        yield Frame(index, index / fps, image)
        index += 1


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
