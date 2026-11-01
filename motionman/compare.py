"""Compare two benchmarks and say what else differed (plan P2 D6, MD§12 "Comparison protocol").

The caller names the factors changed on purpose (``intended``). Every other factor that differs is a
confounder, and the verdict says the comparison is not controlled. Only speed is compared: a change in box
counts or confidence is never called an accuracy improvement.
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from motionman import benchmark, store
from motionman.types import MotionManError

SCHEMA_VERSION = "1.0"
BATTERY_GAP = 20  # percentage points between the two sides' median battery level worth a note
SPEED_ONLY = "Speed only. Changes in boxes, IDs or confidence are not accuracy and are not compared here."


class CompareError(MotionManError):
    exit_code = 2


def _factors(doc: dict, runspec: dict, report: dict) -> dict:
    """Every factor the protocol checks, as plain JSON values, from a benchmark and its first trial's run."""
    s, env = runspec["settings"], doc.get("environment") or report.get("environment") or {}
    repro = runspec.get("reproducibility") or {}
    return {
        "input": s["input"]["input_id"],
        "frame_policy": s["input"]["frame_range"],
        "sampling": s["sampling"],
        "output_policy": {k: s["rendering"].get(k) for k in ("enabled", "codec", "trail_points")},
        "warmup": {k: (report["timing"].get("warmup") or {}).get(k) for k in ("policy", "frames")},
        "task": s["model"]["task"],
        "model": {"weights": s["model"]["weights"], "sha256": s["model"].get("weights_sha256")},
        "class_map": s["model"].get("class_map_sha256"),
        "tracker": s["tracking"]["config_sha256"],
        "imgsz": s["inference"]["imgsz"],
        "conf": s["inference"]["conf"],
        "iou": s["inference"]["iou"],
        "class_filter": s["inference"]["class_filter"],
        "precision": s["inference"]["precision"],
        "device": s["inference"]["device_used"],
        "telemetry": doc["settings"].get("telemetry", False),
        "code": repro.get("code"),
        "software": {
            "python": repro.get("python"),
            "requirements": (repro.get("requirements") or {}).get("sha256"),
            "packages": env.get("packages"),
        },  # fmt: skip
        "hardware": {"cpu": env.get("cpu"), "gpu": env.get("gpu"), "machine": env.get("machine")},
        "platform": repro.get("platform") or env.get("platform"),
        "power": doc["summary"].get("power_sources"),
    }


FACTORS = (  # the order _factors returns them in; tests check the two agree
    "input", "frame_policy", "sampling", "output_policy", "warmup", "task", "model", "class_map", "tracker", "imgsz",
    "conf", "iou", "class_filter", "precision", "device", "telemetry", "code", "software", "hardware", "platform",
    "power",
)  # fmt: skip


def load(ref: str, data_root: Path | None = None) -> tuple[Path, dict]:
    """A benchmark by file path or by ID under ``<data root>/benchmarks``."""
    path = Path(ref)
    if not path.is_file():
        path = store.data_root(data_root).path / "benchmarks" / ref / benchmark.BENCHMARK
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CompareError(f"can't read benchmark {ref!r}: {e}") from None
    if not isinstance(doc, dict) or doc.get("schema_version") != benchmark.SCHEMA_VERSION or "trials" not in doc:
        raise CompareError(f"{ref!r} is not a benchmark file")
    return path, doc


def _profile(path: Path, doc: dict) -> dict:
    ok = [t for t in doc["trials"] if t.get("status") == "succeeded"]
    if not ok:
        raise CompareError(f"benchmark {doc['benchmark_id']} has no succeeded trial to compare")
    run_dir = path.parents[2] / "runs" / ok[0]["run_id"]  # <data root>/benchmarks/<id>/benchmark.json
    try:
        runspec = json.loads((run_dir / store.RUNSPEC).read_text(encoding="utf-8"))
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CompareError(f"benchmark {doc['benchmark_id']}: its run {ok[0]['run_id']} is unreadable ({e})") from None
    return _factors(doc, runspec, report)


def compare(a_ref: str, b_ref: str, intended: list[str] | None = None, data_root: Path | None = None) -> dict:
    intended = list(dict.fromkeys(intended or []))
    unknown = sorted(set(intended) - set(FACTORS))
    if unknown:
        raise CompareError(f"unknown factor(s) {', '.join(unknown)}; choose from {', '.join(FACTORS)}")
    (pa, a), (pb, b) = load(a_ref, data_root), load(b_ref, data_root)
    fa, fb = _profile(pa, a), _profile(pb, b)
    factors = [{"name": n, "a": fa[n], "b": fb[n], "changed": fa[n] != fb[n], "intended": n in intended}
               for n in FACTORS]  # fmt: skip
    confounders = [f["name"] for f in factors if f["changed"] and not f["intended"]]
    unchanged_intended = [f["name"] for f in factors if f["intended"] and not f["changed"]]
    notes = []
    for label, d, f in (("A", a, fa), ("B", b, fb)):
        s = d["summary"]
        if s["failed"]:
            notes.append(f"{label} has {s['failed']} failed trial(s), left out of its median")
        if not s.get("config_consistent", True):
            notes.append(f"{label}'s trials did not all use the same settings")
        if len(s.get("power_sources") or []) > 1:
            notes.append(f"{label} mixes power sources across its trials")
        if d["summary"]["fps"].get("trials", 0) < 3:
            notes.append(f"{label} has fewer than 3 succeeded trials (the protocol asks for at least 3)")
        if (f["code"] or {}).get("dirty"):
            notes.append(f"{label} ran with uncommitted changes; two dirty runs on one commit may differ in code")
    pa_pct, pb_pct = _battery_percent(a), _battery_percent(b)
    if pa_pct is not None and pb_pct is not None and abs(pa_pct - pb_pct) >= BATTERY_GAP:
        notes.append(f"battery level differs widely (A {pa_pct}%, B {pb_pct}%); it can move CPU speed")
    speed = _speed(a, b)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "a": _side(a),
        "b": _side(b),
        "intended": intended,
        "factors": factors,
        "confounders": confounders,
        "unchanged_intended": unchanged_intended,
        "notes": notes,
        "speed": speed,
        "controlled": not confounders and not notes,
        "verdict": _verdict(speed, intended, confounders, notes),
        "scope": SPEED_ONLY,
    }


def write(doc: dict, data_root: Path | None = None) -> Path:
    cid = "c" + store.new_run_id()[1:]
    path = store.data_root(data_root).path / "comparisons" / cid / "comparison.json"
    path.parent.mkdir(parents=True)
    store.write_json_atomic(path, {"comparison_id": cid, **doc})
    return path


def _battery_percent(doc: dict) -> float | None:
    """Median battery percent over a benchmark's trials on battery; None when not on battery or not reported."""
    pcts = [t["power"]["percent"] for t in doc["trials"]
            if (t.get("power") or {}).get("source") == "battery" and t["power"].get("percent") is not None]  # fmt: skip
    return statistics.median(pcts) if pcts else None


def _side(doc: dict) -> dict:
    return {"benchmark_id": doc["benchmark_id"], "label": doc.get("label"), "fps": doc["summary"]["fps"],
            "stage_p50_median_ms": doc["summary"].get("stage_p50_median_ms")}  # fmt: skip


def _speed(a: dict, b: dict) -> dict:
    fa, fb = a["summary"]["fps"], b["summary"]["fps"]
    if fa.get("median") is None or fb.get("median") is None:
        return {"median_fps_change_pct": None, "reason": "a side has no median"}
    sa, sb = a["summary"].get("stage_p50_median_ms") or {}, b["summary"].get("stage_p50_median_ms") or {}
    deltas = {k: round(sb[k] - sa[k], 3) if sa.get(k) is not None and sb.get(k) is not None else None for k in sa}
    return {
        "median_fps_change_pct": round((fb["median"] - fa["median"]) / fa["median"] * 100, 1),
        "ranges_overlap": fa["min"] <= fb["max"] and fb["min"] <= fa["max"],
        "stage_p50_delta_ms": deltas,
    }


def _verdict(speed: dict, intended: list[str], confounders: list[str], notes: list[str]) -> str:
    if speed.get("median_fps_change_pct") is None:
        return "No speed comparison: a side has no succeeded trial."
    change = f"B's median FPS is {speed['median_fps_change_pct']:+.1f}% versus A"
    if speed["ranges_overlap"]:
        change += " (the trial ranges overlap, so the difference may be noise)"
    cause = f" with intended change(s): {', '.join(intended)}" if intended else " with no change declared"
    if confounders:
        return f"{change}{cause}. Not controlled: also differs in {', '.join(confounders)}."
    if notes:
        return f"{change}{cause}. No confounders, but see notes."
    return f"{change}{cause}. Controlled: nothing else differs."
