"""Background runs for the UI (plan ui.md D2; plan P0 step 5).

Streamlit stops the script thread whenever a widget changes, so inference must not run there. A Runner starts
``pipeline.run`` on a worker thread and exposes a thread-safe snapshot that the page polls. Runs are canonical
(spec A§5): imported by hash, staged, validated and published under the data root, guarded by the same
cross-process lease the CLI uses. An in-process lock additionally refuses a second start immediately, so a
double click gets "a run is in progress" instead of a lease error. No Streamlit imports here.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, perf_counter
from typing import Literal

from motionman import pipeline, store, video_io
from motionman.pipeline import OutputCheckError
from motionman.tracking import DeviceError, ModelError, resolve_device
from motionman.types import MotionManError, RunConfig
from motionman.video_io import VideoError

log = logging.getLogger(__name__)

Status = Literal["idle", "processing", "complete", "error"]

# One job per process, across browser sessions; the data-root lease extends this to other processes (CLI).
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
    run_dir: Path | None = None  # the published run folder; only set once a run succeeds
    report: dict | None = None  # this run's report.json, or None (never a previous run's, never a partial one)
    message: str | None = None  # short explanation for the error state
    next_step: str | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)  # files of the published run, for downloads

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
        data_root: Path | None = None,
        make_tracker: Callable[[RunConfig], pipeline.FrameTracker] = pipeline.default_tracker,
        resolve: Callable[[str], str] = resolve_device,
    ) -> None:
        self.root = store.data_root(data_root)
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
        upload_dir: Path | None = None
        try:
            if upload is not None:
                # Same filesystem as the data root; the pipeline imports it by hash, then this copy is deleted.
                name, data = upload
                (self.root.path / ".uploads").mkdir(parents=True, exist_ok=True)
                upload_dir = Path(tempfile.mkdtemp(prefix="upload-", dir=self.root.path / ".uploads"))
                source = upload_dir / f"upload{store.safe_suffix(name)}"
                source.write_bytes(data)
                source_name = name
            else:
                source_name = source.name
            config = RunConfig(
                source=source,
                output_dir=None,
                device=settings.device,
                conf=settings.conf,
                data_root=self.root.path,
                source_name=source_name,
            )
            # A new run replaces everything: no stale Complete badge or old measurements (spec §9).
            self._set(
                JobState(
                    status="processing",
                    source_name=source_name,
                    device_requested=settings.device,
                    started_at=monotonic(),
                )
            )
            self._thread = threading.Thread(
                target=self._work, args=(config, upload_dir), name="motionman-run", daemon=True
            )
            self._thread.start()
        except Exception as e:  # e.g. the upload can't be written or the thread can't start
            log.exception("could not start the run")
            _remove(upload_dir)
            message, next_step = describe_error(e)
            # A terminal state, so the page doesn't stay on "processing" with Run disabled.
            self._set(JobState(status="error", message=message, next_step=next_step))
            _JOB_LOCK.release()
        except BaseException:
            _remove(upload_dir)
            self._set(JobState())
            _JOB_LOCK.release()
            raise
        return True

    def join(self, timeout: float | None = None) -> None:
        """Wait for the current run (tests and shutdown)."""
        if self._thread is not None:
            self._thread.join(timeout)

    def _work(self, config: RunConfig, upload_dir: Path | None) -> None:
        started = perf_counter()
        try:
            report = pipeline.run(
                config, make_tracker=self._make_tracker, resolve=self._resolve_and_record, on_progress=self._progress
            )
            run_dir = Path(report["run_dir"])
            self._update(
                status="complete",
                report=report,
                processed=report["processed_frames"],
                elapsed_s=perf_counter() - started,
                run_dir=run_dir,
                artifacts=_existing_artifacts(run_dir),
            )
        except BaseException as e:  # the worker must always end in a terminal state
            self._fail(e, started)
            if not isinstance(e, Exception):
                raise
        finally:
            _remove(upload_dir)
            if self.snapshot().status == "processing":  # the error handling itself failed
                self._update(
                    status="error",
                    message="The run stopped unexpectedly.",
                    next_step="See the terminal running the UI for details, then try again.",
                )
            _JOB_LOCK.release()

    def _fail(self, e: BaseException, started: float) -> None:
        if not isinstance(e, MotionManError):
            log.error("run failed", exc_info=e)
        message, next_step = describe_error(e)
        # A failed attempt is never published; its partial files stay in staging for diagnosis (UI2 copy:
        # "This attempt did not finish. Partial files are not a completed result.").
        self._update(status="error", message=message, next_step=next_step, elapsed_s=perf_counter() - started)

    def _resolve_and_record(self, requested: str) -> str:
        used = self._resolve(requested)
        self._update(device_used=used)
        return used

    def _progress(self, processed: int, total: int | None, elapsed_s: float) -> None:
        self._update(processed=processed, total=total, elapsed_s=elapsed_s)

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
        return text.split("; use --device")[0], "Choose CPU or Auto and run again."  # drop the CLI-only hint
    if isinstance(e, ModelError):
        return text, "Check that yolo11n.pt is in the project folder or that the network is reachable, then retry."
    if isinstance(e, OutputCheckError):
        return (
            f"This attempt did not finish. Partial files are not a completed result. ({text})",
            "Run again. If it repeats, keep the attempt's folder under data/staging and report the problem.",
        )
    if isinstance(e, store.LeaseError):
        return (
            "Another run is using the data folder, possibly from the command line.",
            "Wait for it to finish, then run again.",
        )
    if isinstance(e, store.SourceChangedError):
        return text, "Choose the video again; it changed while it was being processed."
    if isinstance(e, store.StoreError):
        return text, "Check free disk space and write access to the data folder, then run again."
    if isinstance(e, VideoError):
        if "writer" in text:
            return text, "Check free disk space and write access to the data folder, then run again."
        return text, "Choose another video: a constant-frame-rate MP4 with even width and height."
    return (
        f"Unexpected error: {type(e).__name__}: {text}",
        "See the terminal running the UI for details, then try again.",
    )


def _existing_artifacts(run_dir: Path) -> dict[str, Path]:
    names = (pipeline.ANNOTATED, pipeline.TRACKS, pipeline.REPORT)
    return {name: run_dir / name for name in names if (run_dir / name).is_file()}


def _remove(folder: Path | None) -> None:
    if folder is not None:
        shutil.rmtree(folder, ignore_errors=True)


@dataclass(frozen=True)
class SourceInfo:
    name: str
    width: int
    height: int
    fps: float  # source playback rate, shown apart from processing FPS (spec §9)
    frame_count: int | None
    duration_s: float | None


def probe(path: Path, name: str | None = None) -> SourceInfo:
    """Read and validate a video's metadata for the sidebar; raises VideoError like a real run would."""
    cap, info = video_io.open_reader(path)
    cap.release()
    duration = info.frame_count / info.fps if info.frame_count else None
    return SourceInfo(name or path.name, info.width, info.height, info.fps, info.frame_count, duration)


def probe_upload(name: str, data: bytes) -> SourceInfo:
    """Probe uploaded bytes through a temporary file, deleted afterwards (runs copy the upload themselves)."""
    with tempfile.TemporaryDirectory(prefix="motionman-probe-") as tmp:
        path = Path(tmp) / f"upload{store.safe_suffix(name)}"
        path.write_bytes(data)
        try:
            return probe(path, name)
        except video_io.VideoError as e:  # name the user's file, not our temporary copy
            raise video_io.VideoError(str(e).replace(str(path), name)) from e
