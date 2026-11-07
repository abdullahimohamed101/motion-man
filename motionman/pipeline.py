"""run(config): the synchronous decode → track → annotate → write loop (spec §4), with identified, validated and
published outputs (spec A§2–A§5).

Two modes (plan P0 D2). Canonical (``output_dir is None``): the input is imported by hash into the data root,
the run writes into ``staging/<run_id>/`` under an exclusive lease, and only a validated run is renamed into
``runs/<run_id>/``. Legacy (``output_dir`` given): the MVP behaviour, writing in place, plus the same manifest.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from motionman import metrics, records, render, store, telemetry, video_io
from motionman.tracking import resolve_device
from motionman.types import RunConfig, Track

ANNOTATED = "annotated.mp4"
TRACKS = "tracks.jsonl"
REPORT = "report.json"
TIMINGS = "timings.jsonl"  # one row of stage times per processed frame (plan P2 D3)
TELEMETRY = "telemetry.jsonl"  # opt-in, about one resource sample per second (plan P2 D9)
NATIVE_RECEIPT = "native_receipt.json"  # what motionman-native says it did (ADR-006); checked, never trusted alone
PROGRESS_INTERVAL_S = 0.25  # keep UI redraws modest (spec §9)
MANIFEST_STATUS = {"success": "SUCCEEDED", "error": "FAILED", "interrupted": "INTERRUPTED"}

log = logging.getLogger(__name__)

# on_progress(processed_frames, total_frames_or_None, elapsed_s). total is None when no usable count is known.
ProgressCallback = Callable[[int, int | None, float], None]


class FrameTracker(Protocol):
    def update(self, image: np.ndarray) -> list[Track]: ...


def default_tracker(config: RunConfig) -> FrameTracker:
    from motionman.tracking import Tracker

    return Tracker(config.model, config.device, config.imgsz, config.conf, config.iou)


class OutputCheckError(video_io.VideoError):
    """The written output fails validation: a tool failure, not bad input."""

    exit_code = 1


def check_output_dir(out: Path) -> None:
    """New or empty only: never overwrite earlier results (INV-5). Creates nothing."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise video_io.VideoError(f"output directory must be new or empty: {out}")


def _known_total(container_count: int | None, max_frames: int | None) -> int | None:
    """Frames we expect to process, if knowable. The container count is an estimate; None means unknown."""
    if container_count is None:
        return max_frames
    return container_count if max_frames is None else min(container_count, max_frames)


def _count_partial(path: Path) -> int | None:
    """Decoded frames of a partial video, for error/interrupted reports; None if it cannot be read."""
    try:
        return video_io.count_frames(path) if path.exists() else None
    except video_io.VideoError:
        return None


def _check_free_space(folder: Path, budget: int) -> None:
    probe = folder if folder.exists() else next((p for p in folder.parents if p.exists()), Path("."))
    free = shutil.disk_usage(probe).free
    if free < budget:
        raise store.StoreError(
            f"only {free / 1024**3:.1f} GiB free near {folder}; a run may write up to {budget / 1024**3:.1f} GiB"
        )


def run(
    config: RunConfig,
    make_tracker: Callable[[RunConfig], FrameTracker] = default_tracker,
    resolve: Callable[[str], str] = resolve_device,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Process the source, write and validate the artifacts, and return the report (with ``run_dir``).

    Input, limit, device, output-dir, disk and model problems fail before any output folder exists. Once it
    exists, report.json and manifest.json are written on every exit; only a run whose records, video and
    hashes validate is SUCCEEDED, and only that is published (INV-6). ``on_progress`` is called at most every
    ``PROGRESS_INTERVAL_S`` and once after the last frame; the CLI passes none.
    """
    canonical = config.output_dir is None
    root = store.data_root(config.data_root) if canonical else None
    lease = store.execution_lease(root) if root is not None else contextlib.nullcontext()
    with lease:
        return _run(config, root, make_tracker, resolve, on_progress)


def _run(
    config: RunConfig,
    root: store.DataRoot | None,
    make_tracker: Callable[[RunConfig], FrameTracker],
    resolve: Callable[[str], str],
    on_progress: ProgressCallback | None,
) -> dict:
    cap, info = video_io.open_reader(config.source, config.limits)  # validate the user's file first
    started_at = datetime.now(UTC)
    run_id = store.new_run_id(started_at)
    m = metrics.Metrics()
    m.begin_job()
    status, error = "error", None
    device_used: str | None = None
    decoded: int | None = None
    input_id = runspec = records_sha = None
    out: Path | None = None
    validation = {"source_complete": "NOT_RUN", "records": "NOT_RUN", "video_decode": "NOT_RUN"}
    writer = jsonl = timings = sampler = None
    try:
        device_used = resolve(config.device)
        if root is not None:
            root.ensure()
            cap.release()
            input_id, source = store.import_input(config.source, root, config.source_name)
            cap, info = video_io.open_reader(source, config.limits)  # process the managed copy
            target = root.staging / run_id
            final_dir = root.runs / run_id
        else:
            check_output_dir(config.output_dir)
            source = config.source
            input_id = store.sha256_file(source)
            target = final_dir = config.output_dir
        _check_free_space(target.parent, config.limits.output_budget_bytes)
        t = perf_counter()
        # Fresh tracker per run; never shared across videos (INV-1). Includes model warmup.
        tracker = make_tracker(dataclasses.replace(config, device=device_used))
        m.model_load_s = perf_counter() - t
        target.mkdir(parents=True, exist_ok=root is None)
        out = target
        writer, codec = video_io.open_writer(out / ANNOTATED, info)
        runspec = store.resolve_runspec(
            config,
            info,
            input_id=input_id,
            device_used=device_used,
            weights_path=Path(config.model),
            class_names=getattr(tracker, "class_names", None),
            codec=codec,
            limits=dataclasses.asdict(config.limits),
        )
        store.write_json_atomic(out / store.RUNSPEC, runspec)
        jsonl = (out / TRACKS).open("w", encoding="utf-8")
        timings = (out / TIMINGS).open("w", encoding="utf-8")
        if config.telemetry:
            sampler = telemetry.Telemetry(out / TELEMETRY, device_used=device_used)
            sampler.start()
        trails = render.Trails()
        total = _known_total(info.frame_count, config.max_frames)
        m.start()
        started = last_report = perf_counter()
        reported = -1
        frames = video_io.read_frames(
            cap, info.fps, config.max_frames, config.limits.max_frames, config.limits.max_duration_s
        )
        # Stage boundaries (plan P2 D1): decode = inside next(); detect_track = tracker.update;
        # render = trails + drawing; write = the video frame and the JSONL line. The rest is unattributed.
        while True:
            t0 = perf_counter()
            frame = next(frames, None)
            t1 = perf_counter()
            if frame is None:
                break
            m.decoded_frames += 1
            tracks = tracker.update(frame.image)
            t2 = perf_counter()
            m.inferred_frames += 1
            record = records.frame_record(run_id, frame.index, info.fps, info.width, info.height, tracks)
            trails.update(frame.index, tracks)
            annotated = render.annotate(frame.image, tracks, trails)
            t3 = perf_counter()
            writer.write(annotated)
            jsonl.write(json.dumps(record) + "\n")  # streamed (INV-3)
            t4 = perf_counter()
            m.jsonl_records += 1
            m.processed_frames += 1
            row = m.add_frame((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000)
            timings.write(json.dumps({"source_frame_index": frame.index, **row}) + "\n")
            if on_progress is not None and perf_counter() - last_report >= PROGRESS_INTERVAL_S:
                last_report = perf_counter()
                reported = m.processed_frames
                on_progress(reported, total, last_report - started)
        if on_progress is not None and reported != m.processed_frames:
            on_progress(m.processed_frames, total, perf_counter() - started)
        writer.release()
        writer = None
        jsonl.close()
        jsonl = None
        timings.close()
        timings = None
        m.stop()
        if sampler is not None:
            m.telemetry = sampler.stop()
        # Validate before claiming success (A§5): counts, records, decoded video, unchanged source, budget.
        if m.processed_frames == 0:
            raise video_io.VideoError(f"no decodable frames in {config.source}")
        # A decoder that stops early looks like end of file; a known container count tells them apart (A§4).
        expected = _known_total(info.frame_count, config.max_frames)
        if info.frame_count is None:
            validation["source_complete"] = "UNVERIFIED"  # container gives no count; not claimed complete
        elif m.decoded_frames < expected:
            validation["source_complete"] = "FAIL"
            raise video_io.VideoError(
                f"decoding stopped at frame {m.decoded_frames} of the {expected} the container reports; "
                "the file may be truncated or damaged"
            )
        else:
            validation["source_complete"] = "PASS"
        decoded = video_io.count_frames(out / ANNOTATED)  # outside the timed region
        validation["video_decode"] = "PASS" if decoded == m.processed_frames else "FAIL"
        if decoded != m.processed_frames:
            raise OutputCheckError(
                f"output check failed: wrote {m.processed_frames} frames but {ANNOTATED} decodes to {decoded}"
            )
        try:
            records_sha = store.validate_records_file(out / TRACKS, m.jsonl_records, run_id, info.width, info.height)
        except (store.StoreError, records.RecordError) as e:
            validation["records"] = "FAIL"
            raise OutputCheckError(f"record check failed: {e}") from e
        validation["records"] = "PASS"
        if store.sha256_file(source) != input_id:
            raise store.SourceChangedError(f"{source} changed during the run; its results cannot be trusted")
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
        if timings is not None:
            timings.close()
        m.stop()
        if sampler is not None:
            m.telemetry = sampler.stop()  # idempotent; a failed run still reports what was sampled
        report: dict = {"status": status, "error": error}
        if out is not None:
            if status != "success" and decoded is None:
                # Ctrl-C can land between the video and JSONL writes, so record what each artifact really holds.
                decoded = _count_partial(out / ANNOTATED)
            report, status, error = _finish(
                out, root, run_id, status, error, config, info, m, device_used, decoded,
                input_id, runspec, records_sha, validation, started_at, final_dir,
            )  # fmt: skip
        elif root is not None and status != "success":
            log.warning("run %s failed before writing any output", run_id)
    return report


def _finish(out, root, run_id, status, error, config, info, m, device_used, decoded,
            input_id, runspec, records_sha, validation, started_at, final_dir, **report_args):  # fmt: skip
    """Write report and manifest; publish a SUCCEEDED canonical run. Never masks the run's own error.
    ``report_args`` (backend, tracker, gpu) describe a native run (motionman/native.py)."""
    m.finish()  # finalize ends here: the report records it, so the rename below is excluded (plan P2 D2)
    report: dict = {"status": status, "error": error}
    budget_breach = False
    try:
        report = metrics.build_report(
            status=status,
            error=error,
            config=config,
            info=info,
            metrics=m,
            device_used=device_used,
            output_frames_decoded=decoded,
            **report_args,
        )
        report.update(
            run_id=run_id,
            run_dir=str(final_dir if status == "success" else out),  # only a published run lives in runs/
            mode="canonical" if root is not None else "legacy",
            input_id=input_id,
            config_sha256=runspec["config_sha256"] if runspec else None,
        )
        metrics.write_report(out / REPORT, report)
        names = [
            n for n in (ANNOTATED, TRACKS, TIMINGS, TELEMETRY, NATIVE_RECEIPT, REPORT, store.RUNSPEC)
            if (out / n).is_file()
        ]  # fmt: skip
        artifacts = store.artifact_entries(out, names)
        if status == "success" and sum(a["bytes"] for a in artifacts) > config.limits.output_budget_bytes:
            status, error = "error", "OutputBudgetError: outputs exceed the configured budget"
            budget_breach = True
            report.update(status=status, error=error, run_dir=str(out))
            metrics.write_report(out / REPORT, report)
            artifacts = store.artifact_entries(out, names)
        manifest = store.build_manifest(
            run_id=run_id,
            status=MANIFEST_STATUS[status],
            error=error,
            input_id=input_id,
            config_sha256=report["config_sha256"],
            records_sha256=records_sha if status == "success" else None,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            counts={"source_frames_reported": info.frame_count, "decoded_frames": m.decoded_frames,
                    "inferred_frames": m.inferred_frames, "records": m.jsonl_records,
                    "written_frames": decoded},
            artifacts=artifacts,
            validation=validation,
        )  # fmt: skip
        store.write_json_atomic(out / store.MANIFEST, manifest)
        if status != "success" and root is not None:
            log.warning("run %s did not succeed; diagnostics kept in %s (not published)", run_id, out)
    except Exception:
        log.exception("could not finish run %s in %s", run_id, out)
        if status == "success":
            raise
    if budget_breach:  # the run itself raised nothing, so the caller must hear about this failure here
        raise OutputCheckError(error)
    if status == "success" and root is not None:
        try:
            store.publish(out, root, run_id)
        except Exception as e:
            # Never leave a SUCCEEDED manifest behind in staging for an attempt that did not publish.
            store.write_json_atomic(
                out / store.MANIFEST, {**manifest, "status": "FAILED", "error": f"PublishError: {e}"}
            )
            raise
    return report, status, error
