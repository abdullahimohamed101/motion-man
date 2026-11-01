"""run(config): the synchronous decode → track → annotate → write loop (spec §4)."""

from __future__ import annotations

import dataclasses
import json
import logging
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
PROGRESS_INTERVAL_S = 0.25  # keep UI redraws modest (spec §9)

log = logging.getLogger(__name__)

# on_progress(processed_frames, total_frames_or_None, elapsed_s). total is None when no usable count is known.
ProgressCallback = Callable[[int, int | None, float], None]


class FrameTracker(Protocol):
    def update(self, image: np.ndarray) -> list[Track]: ...


def default_tracker(config: RunConfig) -> FrameTracker:
    from motionman.tracking import Tracker

    return Tracker(config.model, config.device, config.imgsz, config.conf, config.iou)


class OutputCheckError(video_io.VideoError):
    """The written video does not decode to the frames we processed: a tool failure, not bad input."""

    exit_code = 1


def check_output_dir(config: RunConfig) -> None:
    """New or empty only: never overwrite earlier results (INV-5). Creates nothing."""
    out = config.output_dir
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise video_io.VideoError(f"output directory must be new or empty: {out}")


def _known_total(container_count: int | None, max_frames: int | None) -> int | None:
    """Frames we expect to process, if knowable. The container count is an estimate; None means unknown."""
    if container_count is None:
        return max_frames
    return container_count if max_frames is None else min(container_count, max_frames)


def _count_partial(path) -> int | None:
    """Decoded frames of a partial video, for error/interrupted reports; None if it cannot be read."""
    try:
        return video_io.count_frames(path) if path.exists() else None
    except video_io.VideoError:
        return None


def run(
    config: RunConfig,
    make_tracker: Callable[[RunConfig], FrameTracker] = default_tracker,
    resolve: Callable[[str], str] = resolve_device,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Process the source, write the three artifacts, and return the report.

    Input, device, output-dir and model problems fail before the output directory is created.
    Once it exists, report.json is written on every exit, with status ``error`` or ``interrupted``
    for partial runs (INV-6). A run only succeeds if the reopened output decodes to
    ``processed_frames``.

    ``on_progress`` is called at most every ``PROGRESS_INTERVAL_S`` and once after the last frame.
    The CLI passes none.
    """
    cap, info = video_io.open_reader(config.source, config.limits)
    m = metrics.Metrics()
    status, error = "error", None
    device_used: str | None = None
    decoded: int | None = None
    out_ready = False
    writer = jsonl = None
    try:
        device_used = resolve(config.device)
        check_output_dir(config)
        t = perf_counter()
        # Fresh tracker per run; never shared across videos (INV-1). Includes model warmup.
        tracker = make_tracker(dataclasses.replace(config, device=device_used))
        m.model_load_s = perf_counter() - t
        config.output_dir.mkdir(parents=True, exist_ok=True)
        out_ready = True
        writer = video_io.open_writer(config.output_dir / ANNOTATED, info)
        jsonl = (config.output_dir / TRACKS).open("w", encoding="utf-8")
        trails = render.Trails()
        total = _known_total(info.frame_count, config.max_frames)
        m.start()
        started = last_report = perf_counter()
        reported = -1
        for frame in video_io.read_frames(cap, info.fps, config.max_frames, config.limits.max_frames):
            t = perf_counter()
            tracks = tracker.update(frame.image)
            m.detect_track_ms.append((perf_counter() - t) * 1000)
            trails.update(frame.index, tracks)
            writer.write(render.annotate(frame.image, tracks, trails))
            jsonl.write(json.dumps(frame_record(frame.index, frame.timestamp_s, tracks)) + "\n")  # streamed (INV-3)
            m.jsonl_records += 1
            m.processed_frames += 1
            if on_progress is not None and perf_counter() - last_report >= PROGRESS_INTERVAL_S:
                last_report = perf_counter()
                reported = m.processed_frames
                on_progress(reported, total, last_report - started)
        if on_progress is not None and reported != m.processed_frames:
            on_progress(m.processed_frames, total, perf_counter() - started)
        writer.release()
        writer = None
        m.stop()
        if m.processed_frames == 0:
            raise video_io.VideoError(f"no decodable frames in {config.source}")
        decoded = video_io.count_frames(config.output_dir / ANNOTATED)  # outside the timed region
        if decoded != m.processed_frames:
            raise OutputCheckError(
                f"output check failed: wrote {m.processed_frames} frames but {ANNOTATED} decodes to {decoded}"
            )
        status = "success"
    except KeyboardInterrupt:
        status, error = "interrupted", "interrupted by user"
        raise
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    except BaseException as e:  # SystemExit, or a host like Streamlit stopping its script thread
        error = f"{type(e).__module__}.{type(e).__name__}: {e}".rstrip(": ")
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if jsonl is not None:
            jsonl.close()
        m.stop()
        if status != "success" and out_ready and decoded is None:
            # Ctrl-C can land between the video and JSONL writes, so record what each artifact really holds.
            decoded = _count_partial(config.output_dir / ANNOTATED)
        report: dict = {"status": status, "error": error}
        try:
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
        except Exception:  # never let a report failure mask the run's own error
            log.exception("could not write %s", config.output_dir / REPORT)
            if status == "success":
                raise
    return report
