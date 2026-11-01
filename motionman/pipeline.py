"""run(config): the synchronous decode → track → annotate → write loop (spec §4)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from time import perf_counter
from typing import Protocol

import numpy as np

from motionman import metrics, render, video_io
from motionman.types import RunConfig, Track, frame_record

log = logging.getLogger(__name__)

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


def run(config: RunConfig, make_tracker: Callable[[RunConfig], FrameTracker] = default_tracker) -> dict:
    """Process the source, write the three artifacts, and return the report.

    report.json is written on every exit once the output directory is ours, with status
    ``error`` or ``interrupted`` for partial runs (INV-6).
    """
    cap, info = video_io.open_reader(config.source)
    m = metrics.Metrics()
    status, error = "error", None
    out_ready = False
    writer = jsonl = None
    try:
        prepare_output_dir(config)
        out_ready = True
        t = perf_counter()
        tracker = make_tracker(config)  # fresh tracker per run; never shared across videos (INV-1)
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
            status=status, error=error, config=config, info=info, metrics=m, device_used=config.device
        )
        if out_ready:
            metrics.write_report(config.output_dir / REPORT, report)
    log.info(
        "%d frames in %.2fs (%.2f fps end to end) → %s",
        m.processed_frames,
        m.processing_wall_s,
        m.end_to_end_fps,
        config.output_dir,
    )
    return report
