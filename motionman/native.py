"""``--backend native``: one video through ``motionman-native`` on the NVIDIA PC (plan native-g2 step 6, ADR-006).

Python keeps everything that decides whether a run is trusted. It validates and imports the input, resolves and
writes ``runspec.json``, launches the native engine on those exact bytes and relays its status lines. After the
engine exits it checks the receipt against the runspec, then validates every output itself (records, the decoded
video, timing rows, the source hash) before it writes ``report.json`` and ``manifest.json`` and publishes through
the same code as the Python pipeline. The engine's exit code alone is never success; an interrupted, failed or
inconsistent run is never published (INV-5, INV-6).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from motionman import metrics, pipeline, store, video_io
from motionman.types import MotionManError, RunConfig

DEFAULT_PROFILE = "rtdetr_r18_640"
DEFAULT_BINARY = store.REPO_ROOT / "build" / "linux-cuda-release" / "native" / "motionman-native"
BINARY_ENV = "MOTIONMAN_NATIVE_BIN"
CODEC = "mp4v"  # Ubuntu's OpenCV 4.6 has no H.264 writer (plan native-g2 step 5)
DEVICE = "cuda:0"
NATIVE_STAGES = ("preprocess_ms", "infer_ms", "track_ms")  # together they are the report's detect_track_ms
TIMING_KEYS = ["source_frame_index", "decode_ms", *NATIVE_STAGES, "render_ms", "write_ms"]
EXIT_REFUSED, EXIT_INTERRUPTED = 3, 130
STOP_TIMEOUT_S = 30.0  # after Ctrl-C, how long the engine gets to finish its frame and write the receipt

# What a native report's timing numbers cover (G2 is serial: one frame at a time, so stages add up).
SCOPES = {
    "detect_track_ms": "motionman-native: C++ preprocessing + TensorRT inference (batch 1, synchronous) + admission, "
    "ByteTrack and record objects; the three are also reported as native_stages",
    "decode_ms": "motionman-native: OpenCV CPU decode plus the timestamp, limit and frame size checks",
    "render_ms": "motionman-native: record line, trail update and drawing boxes, labels and trails",
    "write_ms": "motionman-native: writing the annotated frame to the encoder plus the JSONL record line",
    "setup": "Python input checks, import and hashing, runspec, then in motionman-native: start, source hashing, "
    "engine identity check and load",
    "processing_wall": "motionman-native steady clock from before the first decode to after the video writer and "
    "record files are closed",
    "unattributed": "native loop time outside the stages: timing rows, progress lines and the final read that "
    "ends the loop, closing the writer and files",
    "instrumentation": "steady_clock per stage in motionman-native; its stream is synchronised once per frame after "
    "the output copy; no global device synchronization",
}

log = logging.getLogger(__name__)


class NativeError(MotionManError):
    """The engine failed, refused the run, or its outputs do not check out: nothing is published."""

    exit_code = 1


class NativeSetupError(MotionManError):
    """The engine, profile or model files are missing or do not fit the request (before any output exists)."""


@dataclass(frozen=True)
class NativeConfig:
    profile: str = DEFAULT_PROFILE
    precision: str = "fp32"
    class_policy: str = "aware"  # agnostic only for parity with the Python reference (ADR-006)
    binary: Path | None = None  # default: $MOTIONMAN_NATIVE_BIN, else the repo's linux-cuda-release build
    model_dir: Path | None = None  # default: ~/.cache/motionman/models/<profile>
    command_prefix: tuple[str, ...] = ()  # e.g. (sys.executable,) to run a fake engine script in tests

    def resolved_binary(self) -> Path:
        return self.binary or Path(os.environ.get(BINARY_ENV) or DEFAULT_BINARY)

    def resolved_model_dir(self) -> Path:
        return self.model_dir or Path.home() / ".cache" / "motionman" / "models" / self.profile

    @property
    def profile_json(self) -> Path:
        return store.REPO_ROOT / "configs" / "native" / f"{self.profile}.json"


def backend_block(engine_manifest: dict) -> dict:
    """The runspec and report ``backend`` block for a TensorRT engine (ADR-006), from its engine manifest."""
    identity = engine_manifest["identity"]
    return {
        "name": "tensorrt",
        "precision": identity["precision"],
        "profile": identity["profile"],
        "onnx_sha256": identity["onnx_sha256"],
        "engine_sha256": engine_manifest["engine"]["sha256"],
        "gpu": identity["gpu"],
        "tensorrt": identity["tensorrt"],
    }


def native_runspec(config: RunConfig, info, *, input_id: str, engine_manifest: dict, model_manifest: dict,
                   class_policy: str, engine_path: Path | None = None, codec: str = CODEC) -> dict:  # fmt: skip
    """The runspec a native run gets (ADR-006): the Python resolution plus the backend, tracker class policy and
    the model's class list."""
    return store.resolve_runspec(
        config,
        info,
        input_id=input_id,
        device_used=DEVICE,
        weights_path=engine_path,
        class_names=dict(enumerate(model_manifest["classes"]["names"])),
        codec=codec,
        limits=dataclasses.asdict(config.limits),
        backend=backend_block(engine_manifest),
        tracking={"type": "bytetrack", "config": store.tracker_config(), "class_policy": class_policy},
    )


def _load_model(native: NativeConfig) -> tuple[Path, dict, dict, dict]:
    """(model dir, profile, model manifest, engine manifest); missing pieces fail before any output exists."""
    if native.class_policy not in ("aware", "agnostic"):
        raise NativeSetupError(f"class policy must be aware or agnostic, not {native.class_policy!r}")
    binary = native.resolved_binary()
    if not binary.is_file() or (not native.command_prefix and not os.access(binary, os.X_OK)):
        raise NativeSetupError(f"motionman-native not found at {binary}; build it (native/README) or set {BINARY_ENV}")
    model_dir = native.resolved_model_dir()
    paths = [native.profile_json, model_dir / "manifest.json", model_dir / f"engine_manifest.{native.precision}.json"]
    for p in paths:
        if not p.is_file():
            raise NativeSetupError(f"native model file missing: {p}")
    profile, model, engine = (json.loads(p.read_text(encoding="utf-8")) for p in paths)
    return model_dir, profile, model, engine


def run(config: RunConfig, native: NativeConfig | None = None,
        on_progress: pipeline.ProgressCallback | None = None) -> dict:  # fmt: skip
    """Run one video natively; same contract as ``pipeline.run`` (report with ``run_dir``; only a verified run is
    SUCCEEDED and only that is published)."""
    native = native or NativeConfig()
    canonical = config.output_dir is None
    root = store.data_root(config.data_root) if canonical else None
    lease = store.execution_lease(root) if root is not None else contextlib.nullcontext()
    with lease:
        return _run(config, root, native, on_progress)


def _run(config: RunConfig, root: store.DataRoot | None, native: NativeConfig,
         on_progress: pipeline.ProgressCallback | None) -> dict:  # fmt: skip
    model_dir, profile, model_manifest, engine_manifest = _load_model(native)
    if config.imgsz != profile["export"]["input_size"]:
        raise NativeSetupError(
            f"profile {native.profile} takes {profile['export']['input_size']} px, not {config.imgsz}"
        )
    if config.telemetry:
        raise NativeSetupError("--telemetry samples the Python process; it is not available with the native backend")
    cap, info = video_io.open_reader(config.source, config.limits)  # validate the user's file first
    cap.release()
    started_at = datetime.now(UTC)
    run_id = store.new_run_id(started_at)
    m = metrics.Metrics()
    m.begin_job()
    job_started = perf_counter()
    m.scopes = dict(SCOPES)
    backend = backend_block(engine_manifest)
    status, error = "error", None
    decoded: int | None = None
    input_id = runspec = records_sha = receipt = None
    out: Path | None = None
    validation = {"native_receipt": "NOT_RUN", "source_complete": "NOT_RUN", "records": "NOT_RUN",
                  "video_decode": "NOT_RUN", "timings": "NOT_RUN"}  # fmt: skip
    try:
        if root is not None:
            root.ensure()
            input_id, source = store.import_input(config.source, root, config.source_name)
            cap, info = video_io.open_reader(source, config.limits)  # the managed copy is what native reads
            cap.release()
            target = root.staging / run_id
            final_dir = root.runs / run_id
        else:
            pipeline.check_output_dir(config.output_dir)
            source = config.source
            input_id = store.sha256_file(source)
            target = final_dir = config.output_dir
        pipeline._check_free_space(target.parent, config.limits.output_budget_bytes)
        target.mkdir(parents=True, exist_ok=root is None)
        out = target
        runspec = native_runspec(
            config, info, input_id=input_id, engine_manifest=engine_manifest, model_manifest=model_manifest,
            class_policy=native.class_policy, engine_path=model_dir / engine_manifest["engine"]["file"],
        )  # fmt: skip
        store.write_json_atomic(out / store.RUNSPEC, runspec)
        cmd = [*native.command_prefix, str(native.resolved_binary()), "run", "--runspec", str(out / store.RUNSPEC),
               "--run-id", run_id, "--source", str(source), "--out", str(out),
               "--profile", str(native.profile_json), "--model-dir", str(model_dir)]  # fmt: skip
        total = pipeline._known_total(info.frame_count, config.max_frames)
        launched = perf_counter()
        code = _launch(cmd, total, on_progress)
        stopped = perf_counter()

        receipt = _read_receipt(out)
        _fill_metrics(m, receipt, out, launched - job_started, stopped)
        if receipt is None:
            raise NativeError(f"motionman-native exited {code} without writing {pipeline.NATIVE_RECEIPT}")
        if receipt.get("status") == "interrupted" or code == EXIT_INTERRUPTED:
            raise KeyboardInterrupt  # stopped by a signal: reported like a Ctrl-C of the Python pipeline
        if code != 0 or receipt.get("status") != "success":
            err = NativeError(f"motionman-native {receipt.get('status')} (exit {code}): {receipt.get('error')}")
            if code == EXIT_REFUSED:
                err.exit_code = 2  # the engine, model or runspec does not fit: a setup problem, not a crash
            raise err
        validation["native_receipt"] = "FAIL"
        _check_receipt(receipt, run_id, out, input_id, backend)
        validation["native_receipt"] = "PASS"
        processed = m.processed_frames
        if processed == 0:
            raise video_io.VideoError(f"no decodable frames in {config.source}")
        expected = pipeline._known_total(info.frame_count, config.max_frames)
        if info.frame_count is None:
            validation["source_complete"] = "UNVERIFIED"
        elif m.decoded_frames < expected:
            validation["source_complete"] = "FAIL"
            raise video_io.VideoError(
                f"decoding stopped at frame {m.decoded_frames} of the {expected} the container reports; "
                "the file may be truncated or damaged"
            )
        else:
            validation["source_complete"] = "PASS"
        decoded = video_io.count_frames(out / pipeline.ANNOTATED)
        validation["video_decode"] = "PASS" if decoded == processed else "FAIL"
        if decoded != processed:
            raise pipeline.OutputCheckError(
                f"output check failed: native wrote {processed} frames but {pipeline.ANNOTATED} decodes to {decoded}"
            )
        try:
            records_sha = store.validate_records_file(out / pipeline.TRACKS, processed, run_id, info.width,
                                                      info.height)  # fmt: skip
        except Exception as e:
            validation["records"] = "FAIL"
            raise pipeline.OutputCheckError(f"record check failed: {e}") from e
        validation["records"] = "PASS"
        validation["timings"] = "PASS" if len(m.decode_ms) == processed else "FAIL"
        if len(m.decode_ms) != processed:
            raise pipeline.OutputCheckError(f"timing check failed: {len(m.decode_ms)} rows for {processed} frames")
        if store.sha256_file(source) != input_id:
            raise store.SourceChangedError(f"{source} changed during the run; its results cannot be trusted")
        if on_progress is not None:
            on_progress(processed, total, m.processing_wall_s or 0.0)
        status = "success"
    except KeyboardInterrupt:
        status, error = "interrupted", "interrupted" + (f": {receipt['error']}" if receipt and receipt.get("error")
                                                         else " by user")  # fmt: skip
        raise
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    except BaseException as e:
        error = f"{type(e).__module__}.{type(e).__name__}: {e}".rstrip(": ")
        raise
    finally:
        report: dict = {"status": status, "error": error}
        if out is not None:
            if receipt is None:  # e.g. Ctrl-C reached Python first; the engine has since stopped
                try:
                    receipt = _read_receipt(out)
                    _fill_metrics(m, receipt, out, None, perf_counter())
                except NativeError as e:  # never mask the run's own error
                    log.warning("could not read the partial native outputs: %s", e)
            if status != "success" and decoded is None:
                decoded = pipeline._count_partial(out / pipeline.ANNOTATED)
            gpu = ((receipt or {}).get("engine") or {}).get("gpu") or backend["gpu"]
            tracker = f"native bytetrack ({native.class_policy}) with bytetrack.yaml thresholds"
            report, status, error = pipeline._finish(
                out, root, run_id, status, error, config, info, m, DEVICE, decoded,
                input_id, runspec, records_sha, validation, started_at, final_dir,
                backend=backend, tracker=tracker, gpu=gpu,
            )  # fmt: skip
        elif root is not None and status != "success":
            log.warning("run %s failed before writing any output", run_id)
    return report


def _launch(cmd: list[str], total: int | None, on_progress: pipeline.ProgressCallback | None) -> int:
    """Start the engine, relay its progress lines, and return its exit code. Its stderr (human logs) passes
    through. If Python is interrupted, the engine is asked to stop after its current frame and is waited for, so
    its receipt says what it finished."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    started = last = perf_counter()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                log.info("motionman-native: %s", line.rstrip())
                continue
            if event.get("event") == "progress" and on_progress is not None:
                now = perf_counter()
                if now - last >= pipeline.PROGRESS_INTERVAL_S:
                    last = now
                    on_progress(int(event["processed"]), total, now - started)
        return proc.wait()
    except BaseException:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                log.warning("motionman-native did not stop within %.0f s; killing it", STOP_TIMEOUT_S)
                proc.kill()
                proc.wait()
        raise
    finally:
        if proc.stdout is not None:
            proc.stdout.close()


def _read_receipt(out: Path) -> dict | None:
    path = out / pipeline.NATIVE_RECEIPT
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise NativeError(f"{pipeline.NATIVE_RECEIPT} is not valid JSON: {e}") from e
    return doc if isinstance(doc, dict) else None


def _fill_metrics(m: metrics.Metrics, receipt: dict | None, out: Path, before_launch_s: float | None,
                  stopped: float) -> None:  # fmt: skip
    """Counts and phase times from the receipt, stage times from timings.jsonl. Best effort for partial runs: what
    cannot be read stays unset, and the checks that follow decide whether the run is trusted."""
    frames = (receipt or {}).get("frames") or {}
    timing = (receipt or {}).get("timing") or {}
    m.decoded_frames = int(frames.get("decoded") or 0)
    m.inferred_frames = m.processed_frames = int(frames.get("processed") or 0)
    m.jsonl_records = int(frames.get("records") or 0)
    m.model_load_s = timing.get("model_load_s")
    if "processing_wall_s" in timing and before_launch_s is not None:
        native_setup = (timing.get("wall_s") or 0.0) - timing["processing_wall_s"]
        m.record_external(setup_s=before_launch_s + native_setup, processing_wall_s=timing["processing_wall_s"],
                          stopped_at=stopped)  # fmt: skip
    path = out / pipeline.TIMINGS
    if not path.is_file():
        return
    native = {k: [] for k in NATIVE_STAGES}
    m.decode_ms, m.detect_track_ms, m.render_ms, m.write_ms = [], [], [], []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            try:
                row = json.loads(line)
            except ValueError:
                break  # a torn last line of a stopped run
            if list(row) != TIMING_KEYS or row["source_frame_index"] != n:
                raise NativeError(f"{pipeline.TIMINGS}: row {n} does not have the native timing layout")
            m.decode_ms.append(row["decode_ms"])
            m.detect_track_ms.append(sum(row[k] for k in NATIVE_STAGES))
            m.render_ms.append(row["render_ms"])
            m.write_ms.append(row["write_ms"])
            for k in NATIVE_STAGES:
                native[k].append(row[k])
    m.native_stages = native


def _check_receipt(receipt: dict, run_id: str, out: Path, input_id: str, backend: dict) -> None:
    """A success receipt must describe this run: these bytes of the runspec and source, this engine, and counts
    that agree with each other."""
    engine = receipt.get("engine") or {}
    frames = receipt.get("frames") or {}
    problems = []
    if receipt.get("run_id") != run_id:
        problems.append(f"run_id {receipt.get('run_id')!r} is not {run_id!r}")
    if receipt.get("runspec_sha256") != store.sha256_file(out / store.RUNSPEC):
        problems.append("runspec_sha256 differs from the runspec Python wrote")
    if receipt.get("source_sha256") != input_id:
        problems.append("source_sha256 differs from the imported input")
    for key in ("engine_sha256", "precision", "profile"):
        if engine.get(key) != backend[key]:
            problems.append(f"engine {key} {engine.get(key)!r} is not {backend[key]!r}")
    counts = [frames.get(k) for k in ("processed", "records", "timing_rows", "video_written")]
    if len(set(counts)) != 1 or not isinstance(counts[0], int) or (frames.get("decoded") or 0) < counts[0]:
        problems.append(f"frame counts disagree: {frames}")
    if problems:
        raise NativeError(f"{pipeline.NATIVE_RECEIPT} does not match this run: " + "; ".join(problems))
