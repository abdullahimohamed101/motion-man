"""Repeated, comparable benchmark runs (plan P2 D5, MD§12 "Comparison protocol").

Each trial is a fresh canonical ``pipeline.run``: a new tracker, a new run_id and new records. Nothing from an
earlier trial or run is reused. The result reports the median trial throughput and the range, never a pooled
number, and failed trials stay listed. Power state is recorded per trial because it moves CPU numbers the most.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from motionman import metrics, pipeline, store
from motionman.tracking import resolve_device
from motionman.types import MotionManError, RunConfig

SCHEMA_VERSION = "1.0"
STOP_CODES = (2, 75)  # invalid input/device/model, or another run holds the lease: stop the series, keep the code
REUSE_POLICY = "none: every trial is a fresh pipeline.run with a new tracker, run_id and records"
BENCHMARK = "benchmark.json"


def power_state() -> dict:
    """AC, battery or unknown. Unknown when the host has no battery API; never guessed."""
    try:
        import psutil

        battery = psutil.sensors_battery()
    except Exception as e:
        return {"source": "unknown", "reason": f"battery API failed ({type(e).__name__})"}
    if battery is None:
        return {"source": "unknown", "reason": "no battery reported by this host"}
    return {"source": "ac" if battery.power_plugged else "battery", "percent": round(battery.percent)}


def new_benchmark_id(now: datetime | None = None) -> str:
    return "b" + store.new_run_id(now)[1:]


def run_benchmark(
    source: Path,
    *,
    repeats: int = 3,
    device: str = "cpu",
    imgsz: int = 640,
    max_frames: int | None = None,
    telemetry: bool = False,
    label: str | None = None,
    data_root: Path | None = None,
    make_tracker: Callable = pipeline.default_tracker,
    resolve: Callable[[str], str] = resolve_device,
    power: Callable[[], dict] = power_state,
    on_trial: Callable[[dict], None] | None = None,
) -> tuple[Path, dict]:
    """Run ``repeats`` trials and write ``<data root>/benchmarks/<id>/benchmark.json``. Returns (path, doc).

    A trial that raises is recorded as failed and the next one still runs, unless the error is invalid input or a
    busy data root (exit 2 or 75), which stops the series with that code recorded. Ctrl-C stops the series and
    the cut-off trial is listed as interrupted; the file is still written either way.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    root = store.data_root(data_root).ensure()
    bid = new_benchmark_id()
    folder = root.path / "benchmarks" / bid
    folder.mkdir(parents=True)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": bid,
        "label": label,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "running",
        "source": {"path_name": Path(source).name, "input_id": None},
        "settings": {
            "repeats": repeats,
            "device": device,
            "imgsz": imgsz,
            "max_frames": max_frames,
            "telemetry": telemetry,
        },  # fmt: skip
        "reuse": REUSE_POLICY,
        "trials": [],
    }
    try:
        for i in range(1, repeats + 1):
            trial = {"trial": i, "power": power()}
            config = RunConfig(source, None, device=device, imgsz=imgsz, max_frames=max_frames,
                               data_root=root.path, telemetry=telemetry)  # fmt: skip
            stop = False
            try:
                report = pipeline.run(config, make_tracker=make_tracker, resolve=resolve)
                trial.update(_trial_result(report, root))
            except KeyboardInterrupt:
                doc["trials"].append({**trial, "status": "interrupted"})
                raise
            except Exception as e:  # a failed trial stays visible; the series goes on
                trial.update(status="failed", error=f"{type(e).__name__}: {e}")
                code = getattr(e, "exit_code", None) if isinstance(e, MotionManError) else None
                if code in STOP_CODES:  # bad input or a busy data root: every later trial would fail the same way
                    trial["exit_code"], stop = code, True
            doc["trials"].append(trial)
            if on_trial is not None:
                on_trial(trial)
            if stop:
                break
        doc["status"] = "stopped" if stop else "complete"
    except KeyboardInterrupt:
        doc["status"] = "interrupted"
        raise
    finally:
        doc.update(_summary(doc["trials"]))
        ok = [t for t in doc["trials"] if t["status"] == "succeeded"]
        if ok:
            doc["source"]["input_id"] = ok[0]["input_id"]
            doc["environment"] = metrics.environment(ok[0]["device_used"])
        store.write_json_atomic(folder / BENCHMARK, doc)
    return folder / BENCHMARK, doc


def _trial_result(report: dict, root: store.DataRoot) -> dict:
    t = report["timing"]
    manifest = json.loads((root.runs / report["run_id"] / store.MANIFEST).read_text(encoding="utf-8"))
    tel = report.get("telemetry") or {}
    return {
        "status": "succeeded",
        "run_id": report["run_id"],
        "input_id": report["input_id"],
        "config_sha256": report["config_sha256"],
        "records_sha256": manifest["records_sha256"],
        "device_used": report["device"]["used"],
        "processed_frames": report["processed_frames"],
        "end_to_end_fps": t["end_to_end_fps"],
        "processing_wall_s": t["processing_wall_s"],
        "setup_s": t["setup_s"],
        "job_wall_s": t["job_wall_s"],
        "stages": {k: {"p50": v["p50"], "p95": v["p95"], "total_s": v["total_s"]} for k, v in t["stages"].items()},
        "rss_peak_mb": (tel.get("rss_mb") or {}).get("peak"),
    }


def _summary(trials: list[dict]) -> dict:
    ok = [t for t in trials if t["status"] == "succeeded"]
    configs = sorted({t["config_sha256"] for t in ok})
    summary: dict = {
        "succeeded": len(ok),
        "failed": sum(t["status"] == "failed" for t in trials),
        "interrupted": sum(t["status"] == "interrupted" for t in trials),
        "config_sha256": configs[0] if len(configs) == 1 else None,
        "config_consistent": len(configs) <= 1,
    }
    fps = [t["end_to_end_fps"] for t in ok if t["end_to_end_fps"] is not None]
    if fps:
        summary["fps"] = {"median": round(statistics.median(fps), 2), "min": min(fps), "max": max(fps),
                          "trials": len(fps)}  # fmt: skip
    else:
        summary["fps"] = {"median": None, "reason": "no succeeded trial"}
    stage_p50 = {}
    for name in metrics.STAGES:
        values = [t["stages"][name]["p50"] for t in ok if t["stages"][name]["p50"] is not None]
        stage_p50[name] = round(statistics.median(values), 3) if values else None
    summary["stage_p50_median_ms"] = stage_p50
    summary["power_sources"] = sorted({t["power"]["source"] for t in trials})
    return {"summary": summary}
