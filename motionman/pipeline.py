"""run(config): the synchronous decode → track → annotate → write loop (spec §4)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from time import perf_counter
from typing import Protocol

import numpy as np

from motionman import metrics, render, video_io
from motionman.tracking import resolve_device
from motionman.types import RunConfig, Track, frame_record

ANNOTATED = "annotated.mp4"
TRACKS = "tracks.jsonl"
REPORT = "report.json"


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


def run(
    config: RunConfig,
    make_tracker: Callable[[RunConfig], FrameTracker] = default_tracker,
    resolve: Callable[[str], str] = resolve_device,
) -> dict:
    """Process the source, write the three artifacts, and return the report.

    Input and device problems fail before the output directory is created. Once it exists,
    report.json is written on every exit, with status ``error`` or ``interrupted`` for partial
    runs (INV-6). A run only succeeds if the reopened output decodes to ``processed_frames``.
    """
    cap, info = video_io.open_reader(config.source)
    m = metrics.Metrics()
    status, error = "error", None
    device_used: str | None = None
    decoded: int | None = None
    out_ready = False
    writer = jsonl = None
    try:
        device_used = resolve(config.device)
        prepare_output_dir(config)
        out_ready = True
        t = perf_counter()
        # Fresh tracker per run; never shared across videos (INV-1).
        tracker = make_tracker(dataclasses.replace(config, device=device_used))
        m.model_load_s = perf_counter() - t
        writer = video_io.open_writer(config.output_dir / ANNOTATED, info)
        jsonl = (config.output_dir / TRACKS).open("w", encoding="utf-8")
        trails = render.Trails()
        m.start()
        for frame in video_io.read_frames(cap, info.fps, config.max_frames):
            t = perf_counter()
            tracks = tracker.update(frame.image)
            m.detect_track_ms.append((perf_counter() - t) * 1000)
            trails.update(frame.index, tracks)
            writer.write(render.annotate(frame.image, tracks, trails))
            jsonl.write(json.dumps(frame_record(frame.index, frame.timestamp_s, tracks)) + "\n")  # streamed (INV-3)
            m.processed_frames += 1
        writer.release()
        writer = None
        m.stop()
        if m.processed_frames == 0:
            raise video_io.VideoError(f"no decodable frames in {config.source}")
        decoded = video_io.count_frames(config.output_dir / ANNOTATED)  # outside the timed region
        if decoded != m.processed_frames:
            raise video_io.VideoError(
                f"output check failed: wrote {m.processed_frames} frames but {ANNOTATED} decodes to {decoded}"
            )
        status = "success"
    except KeyboardInterrupt:
        status, error = "interrupted", "interrupted by user"
        raise
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if jsonl is not None:
            jsonl.close()
        m.stop()
        report = metrics.build_report(
            status=status,
            error=error,
            config=config,
            info=info,
            metrics=m,
            device_used=device_used,
            output_frames_decoded=decoded,
        )
        if out_ready:
            metrics.write_report(config.output_dir / REPORT, report)
    return report
