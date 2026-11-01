"""run(config): the synchronous decode → track → annotate → write loop (spec §4)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np

from motionman import render, video_io
from motionman.types import RunConfig, Track

log = logging.getLogger(__name__)

ANNOTATED = "annotated.mp4"


class FrameTracker(Protocol):
    def update(self, image: np.ndarray) -> list[Track]: ...


def default_tracker(config: RunConfig) -> FrameTracker:
    from motionman.tracking import Tracker

    return Tracker(config.model, config.device, config.imgsz, config.conf, config.iou)


def prepare_output_dir(config: RunConfig) -> None:
    """New or empty only: never overwrite earlier results (INV-5)."""
    out = config.output_dir
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise video_io.VideoError(f"output directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)


def run(config: RunConfig, make_tracker: Callable[[RunConfig], FrameTracker] = default_tracker) -> int:
    """Process the source and return the number of frames written."""
    cap, info = video_io.open_reader(config.source)
    writer = None
    processed = 0
    try:
        prepare_output_dir(config)
        tracker = make_tracker(config)  # fresh tracker per run; never shared across videos (INV-1)
        writer = video_io.open_writer(config.output_dir / ANNOTATED, info)
        trails = render.Trails()
        for frame in video_io.read_frames(cap, info.fps, config.max_frames):
            tracks = tracker.update(frame.image)
            trails.update(frame.index, tracks)
            writer.write(render.annotate(frame.image, tracks, trails))
            processed += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if processed == 0:
        raise video_io.VideoError(f"no decodable frames in {config.source}")
    log.info("processed %d frames → %s", processed, config.output_dir / ANNOTATED)
    return processed
