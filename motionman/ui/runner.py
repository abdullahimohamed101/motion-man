"""Background runs for the UI (plan ui.md D2).

Streamlit stops the script thread whenever a widget changes, so inference must not run there. A
Runner starts ``pipeline.run`` on a worker thread and exposes a thread-safe snapshot that the page
polls. One job runs at a time per process (INV-1); each run gets a fresh tracker and a new directory
under ``runs/ui/`` (INV-5). No Streamlit imports here, so this module is tested without a browser.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, perf_counter
from typing import Literal

from motionman import pipeline
from motionman.pipeline import OutputCheckError
from motionman.tracking import DeviceError, ModelError, resolve_device
from motionman.types import MotionManError, RunConfig
from motionman.video_io import VideoError

log = logging.getLogger(__name__)

Status = Literal["idle", "processing", "complete", "error"]

# One job per process, across browser sessions: tracking is CPU-heavy and never runs in parallel here.
_JOB_LOCK = threading.Lock()


@dataclass(frozen=True)
class RunSettings:
    device: str = "cpu"  # cpu | cuda | auto
    conf: float = 0.25


@dataclass(frozen=True)
class JobState:
    status: Status = "idle"
    source_name: str | None = None
    device_requested: str | None = None
    device_used: str | None = None  # known once resolved; the header shows this, not the request
    processed: int = 0
    total: int | None = None  # None: no usable total, so the page shows a count, not a percentage
    elapsed_s: float = 0.0  # last value reported by the pipeline
    started_at: float | None = None  # time.monotonic() at start, so the page can show live elapsed time
    run_dir: Path | None = None
    out_dir: Path | None = None
    report: dict | None = None  # this run's report.json, or None (never a previous run's)
    message: str | None = None  # short explanation for the error state
    next_step: str | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)  # files that exist, for downloads

    @property
    def busy(self) -> bool:
        return self.status == "processing"

    def live_elapsed_s(self, now: float | None = None) -> float:
        """Elapsed time that keeps moving during model load, before the first progress report."""
        if self.status == "processing" and self.started_at is not None:
            return max(self.elapsed_s, (monotonic() if now is None else now) - self.started_at)
        return self.elapsed_s


class Runner:
    def __init__(
        self,
        runs_root: Path = Path("runs/ui"),
        make_tracker: Callable[[RunConfig], pipeline.FrameTracker] = pipeline.default_tracker,
        resolve: Callable[[str], str] = resolve_device,
    ) -> None:
        self.runs_root = runs_root
        self._make_tracker = make_tracker
        self._resolve = resolve
        self._lock = threading.Lock()
        self._state = JobState()
        self._thread: threading.Thread | None = None

    def snapshot(self) -> JobState:
        with self._lock:
            return self._state

    def start(
        self, settings: RunSettings, *, source: Path | None = None, upload: tuple[str, bytes] | None = None
    ) -> bool:
        """Start a run from a file on disk or an uploaded (name, bytes). False if a run is in progress."""
        if (source is None) == (upload is None):
            raise ValueError("pass exactly one of source or upload")
        if not _JOB_LOCK.acquire(blocking=False):
            return False
        try:
            run_dir = self._new_run_dir()
            if upload is not None:
                name, data = upload
                source = run_dir / f"source{Path(name).suffix.lower() or '.mp4'}"
                source.write_bytes(data)
                source_name = name
            else:
                source_name = source.name
            config = RunConfig(source=source, output_dir=run_dir / "out", device=settings.device, conf=settings.conf)
            # A new run replaces everything: no stale Complete badge or old measurements (spec §9).
            self._set(
                JobState(
                    status="processing",
                    source_name=source_name,
                    device_requested=settings.device,
                    run_dir=run_dir,
                    out_dir=config.output_dir,
                    started_at=monotonic(),
                )
            )
            self._thread = threading.Thread(target=self._work, args=(config,), name="motionman-run", daemon=True)
            self._thread.start()
        except BaseException:
            _JOB_LOCK.release()
            raise
        return True

    def join(self, timeout: float | None = None) -> None:
        """Wait for the current run (tests and shutdown)."""
        if self._thread is not None:
            self._thread.join(timeout)

    def _work(self, config: RunConfig) -> None:
        started = perf_counter()
        try:
            report = pipeline.run(
                config, make_tracker=self._make_tracker, resolve=self._resolve_and_record, on_progress=self._progress
            )
        except Exception as e:  # the worker must always end in a terminal state
            if not isinstance(e, MotionManError):
                log.exception("run failed")
            message, next_step = describe_error(e)
            self._update(
                status="error",
                message=message,
                next_step=next_step,
                elapsed_s=perf_counter() - started,
                report=_read_report(config.output_dir),  # partial report, if the run got that far
                artifacts=_existing_artifacts(config.output_dir),
            )
        else:
            self._update(
                status="complete",
                report=report,
                processed=report["processed_frames"],
                elapsed_s=perf_counter() - started,
                artifacts=_existing_artifacts(config.output_dir),
            )
        finally:
            _JOB_LOCK.release()

    def _resolve_and_record(self, requested: str) -> str:
        used = self._resolve(requested)
        self._update(device_used=used)
        return used

    def _progress(self, processed: int, total: int | None, elapsed_s: float) -> None:
        self._update(processed=processed, total=total, elapsed_s=elapsed_s)

    def _new_run_dir(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        for n in range(1000):
            run_dir = self.runs_root / (stamp if n == 0 else f"{stamp}-{n}")
            try:
                run_dir.mkdir(parents=True, exist_ok=False)  # never reuse a directory (INV-5)
                return run_dir
            except FileExistsError:
                continue
        raise RuntimeError(f"no free run directory under {self.runs_root}")

    def _set(self, state: JobState) -> None:
        with self._lock:
            self._state = state

    def _update(self, **changes) -> None:
        with self._lock:
            self._state = dataclasses.replace(self._state, **changes)


def describe_error(e: BaseException) -> tuple[str, str]:
    """A short explanation and a next step for the page (spec §9: errors give a next step)."""
    text = str(e)
    if isinstance(e, DeviceError):
        return text, "Choose CPU or Auto and run again."
    if isinstance(e, ModelError):
        return text, "Check that yolo11n.pt is in the project folder or that the network is reachable, then retry."
    if isinstance(e, OutputCheckError):
        return text, "Run again. If it repeats, keep report.json from this run and report the problem."
    if isinstance(e, VideoError):
        if "writer" in text or "output directory" in text:
            return text, "Check free disk space and write access to the runs/ui folder, then run again."
        return text, "Choose another video: a constant-frame-rate MP4 with even width and height."
    return (
        f"Unexpected error: {type(e).__name__}: {text}",
        "See the terminal running the UI for details, then try again.",
    )


def _existing_artifacts(out_dir: Path) -> dict[str, Path]:
    names = (pipeline.ANNOTATED, pipeline.TRACKS, pipeline.REPORT)
    return {name: out_dir / name for name in names if (out_dir / name).is_file()}


def _read_report(out_dir: Path) -> dict | None:
    try:
        return json.loads((out_dir / pipeline.REPORT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
