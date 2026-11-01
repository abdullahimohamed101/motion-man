"""Run a region set's rules over one saved run and publish the result (spec B§2, B§5; plan P3 D1, D3, D8).

One streaming pass over the run's verified records feeds ``analytics.Engine``. Nothing here runs a model or
touches the run's own files. The result lives in ``<data root>/analyses/<run_id>/<analysis_id>/``: it is built
in a temporary folder and renamed in only when complete, an identical analysis reuses the published folder, and
a damaged one is rebuilt. ``analysis_id`` hashes the AnalysisSpec, which names every input and setting, so the
same run, regions and settings always give the same id and the same bytes.
"""

from __future__ import annotations

import csv
import io
import json
import os
import secrets
import shutil
import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from motionman import analytics, records, regions, runs, store
from motionman.types import MotionManError

SCHEMA_VERSION = "1.0"
IMPL_VERSION = 1  # bump when a rule, an output field or its meaning changes
REFERENCE_POINT = "bbox_center"
HEATMAP_GRID = (256, 144)
SPEC, EVENTS, SUMMARY, MANIFEST = "analysis_spec.json", "events.jsonl", "summary.json", "manifest.json"
HEATMAP = "heatmap.json"
EVENT_COLUMNS = ("event_id", "type", "rule_id", "rule_name", "track_id", "class_name", "direction", "confirming_frame",
                 "confirming_t_us", "evidence_start_us", "evidence_end_us", "value", "value_unit")  # fmt: skip
VALUE_UNITS = {analytics.DWELL_THRESHOLD: "us", analytics.OCCUPANCY_THRESHOLD: "objects"}


class AnalysisError(MotionManError):
    """The analysis can't run on this input; the message says why. Exit code 2 (bad input)."""


@dataclass(frozen=True)
class Result:
    analysis_id: str
    folder: Path
    spec: dict
    summary: dict
    reused: bool  # True when an identical, intact analysis was already published


def find_run(root: store.DataRoot, run_id: str) -> runs.OpenRun:
    """Open a published run by ID, re-verifying every artifact hash."""
    summary = next((s for s in runs.list_runs(root) if s.run_id == run_id), None)
    if summary is None:
        raise AnalysisError(f"no published run {run_id!r} under {root.runs}")
    try:
        return runs.open_run(summary, root)
    except runs.RunError as e:
        raise AnalysisError(f"can't analyze a run that fails verification: {e}") from None


def build_spec(
    run: runs.OpenRun,
    region_set: regions.RegionSet,
    params: analytics.Params | None = None,
    window_us: tuple[int, int | None] = (0, None),
) -> dict:
    """The full, canonical description of an analysis. Refuses runs whose source time isn't guaranteed."""
    if run.manifest is None or run.records_sha256 is None:
        raise AnalysisError(f"{run.run_id} is a legacy run without verified source time; rerun the video to analyze it")
    input_id = run.manifest.get("input_id")
    if region_set.input_id != input_id:
        raise AnalysisError("these regions were drawn for a different video than this run's source")
    start, end = window_us
    if start < 0 or (end is not None and end <= start):
        raise AnalysisError(f"the analysis window must have 0 ≤ start < end, got {start}–{end} µs")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "analysis",
        "run_id": run.run_id,
        "input_id": input_id,
        "records_sha256": run.records_sha256,
        "frame_size": [run.width, run.height],
        "fps": run.fps,
        "region_set": region_set.to_dict(),
        "region_set_sha256": region_set.sha256,
        "reference_point": REFERENCE_POINT,
        "window_us": [start, end],
        "params": (params or analytics.DEFAULT_PARAMS).to_dict(),
        "heatmap_grid": list(HEATMAP_GRID),
        "implementation_version": IMPL_VERSION,
    }


def analysis_id(spec: dict) -> str:
    return "a_" + store.sha256_json(spec).removeprefix("sha256:")[:16]


def folder_for(root: store.DataRoot, run_id: str, aid: str) -> Path:
    return root.path / "analyses" / run_id / aid


def analyze(
    run: runs.OpenRun,
    region_set: regions.RegionSet,
    root: store.DataRoot,
    params: analytics.Params | None = None,
    window_us: tuple[int, int | None] = (0, None),
) -> Result:
    """Analyze ``run`` with ``region_set``; reuse an identical intact result, else build and publish one."""
    spec = build_spec(run, region_set, params, window_us)
    aid = analysis_id(spec)
    target = folder_for(root, run.run_id, aid)
    if target.exists() and _intact(target, aid):
        return Result(aid, target, spec, _read_json(target / SUMMARY), reused=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp-{aid}-{secrets.token_hex(3)}"
    tmp.mkdir()
    try:
        summary = _build(run, region_set, spec, aid, tmp)
        _publish(tmp, target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # a failed or interrupted build leaves nothing behind (INV-6)
    return Result(aid, target, spec, summary, reused=False)


def load(root: store.DataRoot, run_id: str, aid: str) -> Result:
    """A published analysis, re-verified against its manifest."""
    target = folder_for(root, run_id, aid)
    if not _intact(target, aid):
        raise AnalysisError(f"analysis {aid} for {run_id} is missing or damaged; analyze again to rebuild it")
    return Result(aid, target, _read_json(target / SPEC), _read_json(target / SUMMARY), reused=True)


def list_analyses(root: store.DataRoot, run_id: str) -> list[str]:
    """IDs of intact published analyses for one run, oldest first."""
    base = root.path / "analyses" / run_id
    if not base.is_dir():
        return []
    found = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("a_") and _intact(p, p.name)]
    return [p.name for p in sorted(found, key=lambda p: (p.stat().st_mtime, p.name))]


def read_events(result: Result) -> Iterator[dict]:
    with (result.folder / EVENTS).open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_heatmap(result: Result) -> dict:
    return _read_json(result.folder / HEATMAP)


def export_csv(result: Result) -> str:
    """Events as CSV: one row per event, IDs kept, times in µs, each value with its unit."""
    names = {r["id"]: r["name"] for r in result.spec["region_set"]["regions"]}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(EVENT_COLUMNS)
    for e in read_events(result):
        start, stop = e["evidence_interval_us"]
        w.writerow([e["event_id"], e["type"], e["rule_id"], names.get(e["rule_id"], ""), _cell(e["track_id"]),
                    _cell(e["class_name"]), _cell(e["direction"]), e["confirming_frame"], e["confirming_t_us"],
                    start, stop, _cell(e["value"]), VALUE_UNITS.get(e["type"], "")])  # fmt: skip
    return buf.getvalue()


def export_json(result: Result) -> str:
    """The analysis as one JSON document: what was asked (spec), the totals (summary) and every event."""
    doc = {"analysis_id": result.analysis_id, "units": {"time": "us", "dwell": "us", "position": "source px"},
           "spec": result.spec, "summary": result.summary, "events": list(read_events(result))}  # fmt: skip
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def _cell(v) -> str:
    return "" if v is None else str(v)


def _build(run: runs.OpenRun, region_set: regions.RegionSet, spec: dict, aid: str, out: Path) -> dict:
    params = analytics.Params(**spec["params"])
    engine = analytics.Engine(region_set.regions, run.width, run.height, params)
    heat = analytics.Heatmap(run.width, run.height, tuple(spec["heatmap_grid"]), params.max_gap_us)
    start, end = spec["window_us"]
    frames = events = unassigned = 0
    first = last = None
    ordinals: dict[str, int] = {}
    counts: dict[str, int] = {}
    firsts: dict[tuple[str, str], dict] = {}  # first event of each (rule, type), for bookmarks
    with (out / EVENTS).open("w", encoding="utf-8") as f:
        for rec in records.read_records(run.records_path):
            t = rec["source_t_us"]
            if t < start or (end is not None and t >= end):
                continue
            objs = []
            for o in rec["objects"]:
                x1, y1, x2, y2 = o["bbox_xyxy"]
                objs.append(analytics.Obs(o["track_id"], o["class_name"], (x1 + x2) / 2, (y1 + y2) / 2))
                unassigned += o["track_id"] is None
            heat.feed(rec["source_frame_index"], t, objs)
            for e in engine.feed(rec["source_frame_index"], t, objs):
                n = ordinals[e.rule_id] = ordinals.get(e.rule_id, 0) + 1
                eid = "e_" + store.sha256_json([aid, e.rule_id, e.track_id, n]).removeprefix("sha256:")[:16]
                f.write(store.canonical_json({"event_id": eid, "analysis_id": aid, **e.to_dict()}) + "\n")
                counts[e.type] = counts.get(e.type, 0) + 1
                events += 1
                firsts.setdefault((e.rule_id, e.type), {"event_id": eid, "frame": e.confirming_frame,
                                                        "t_us": e.confirming_t_us, "track_id": e.track_id})  # fmt: skip
            frames += 1
            first = rec["source_frame_index"] if first is None else first
            last = rec["source_frame_index"]
    if not frames:
        raise AnalysisError("the analysis window contains no frames of this run")
    close = records.source_t_us(last + 1, run.fps)  # the last frame lasts until the next would have started
    stop = close if end is None else min(close, end)
    engine.finish(stop)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": aid,
        "run_id": run.run_id,
        "coverage": {"frames": frames, "first_frame": first, "last_frame": last,
                     "start_us": records.source_t_us(first, run.fps), "end_us": stop},
        "events": {"total": events, "by_type": dict(sorted(counts.items()))},
        "unassigned_detections": unassigned,
        "regions": [_region_summary(r, engine.stats[r.id], run.fps) for r in region_set.regions],
    }  # fmt: skip
    summary["bookmarks"] = _bookmarks(region_set, summary["regions"], firsts)
    _write_json(out / SPEC, spec)
    _write_json(out / SUMMARY, summary)
    _write_json(out / HEATMAP, {"schema_version": SCHEMA_VERSION, "analysis_id": aid, **heat.to_dict()})
    artifacts = [{"path": name, "bytes": (out / name).stat().st_size, "sha256": store.sha256_file(out / name)}
                 for name in (SPEC, EVENTS, SUMMARY, HEATMAP)]  # fmt: skip
    _write_json(out / MANIFEST, {
        "schema_version": SCHEMA_VERSION, "kind": "analysis", "analysis_id": aid, "status": "SUCCEEDED",
        "run_id": run.run_id, "records_sha256": run.records_sha256, "region_set_sha256": region_set.sha256,
        "implementation_version": IMPL_VERSION, "artifacts": artifacts,
    })  # fmt: skip
    return summary


BOOKMARKS = (("line", analytics.LINE_CROSSING, "first crossing"), ("rect", analytics.ZONE_ENTRY, "first entry"))


def _bookmarks(region_set: regions.RegionSet, summaries: list[dict], firsts: dict) -> list[dict]:
    """Frames worth a look, each naming why it was picked; never a judgment of importance (MD§11)."""
    out = []
    for region, s in zip(region_set.regions, summaries, strict=True):
        for kind, etype, criterion in BOOKMARKS:
            if region.kind == kind and (region.id, etype) in firsts:
                out.append({"criterion": criterion, "rule_id": region.id, **firsts[(region.id, etype)]})
        if region.kind == "rect" and s["peak_frame"] is not None and s["peak_occupancy"] > 0:
            out.append({"criterion": "peak occupancy", "rule_id": region.id, "event_id": None,
                        "frame": s["peak_frame"], "t_us": s["peak_t_us"], "track_id": None,
                        "value": s["peak_occupancy"]})  # fmt: skip
    return sorted(out, key=lambda b: (b["frame"], b["rule_id"], b["criterion"]))


def _region_summary(region: regions.Region, s: analytics.RegionStats, fps: float) -> dict:
    base = {"id": region.id, "name": region.name, "kind": region.kind}
    if region.kind == "line":
        return {**base, "crossings": dict(s.crossings), "crossings_total": sum(s.crossings.values())}
    visits = sorted(s.visits_us)
    return {
        **base,
        "entries": s.entries,
        "exits": s.exits,
        "peak_occupancy": s.peak_occupancy,
        "peak_frame": s.peak_frame,
        "peak_t_us": records.source_t_us(s.peak_frame, fps) if s.peak_frame is not None else None,
        "occupancy_us": s.occupancy_us,  # mean occupancy = occupancy_us / covered_us (kept as integers)
        "covered_us": s.covered_us,
        "peak_unassigned_inside": s.peak_unassigned,
        "visits": len(visits),
        "dwell_total_us": sum(visits),
        "dwell_median_us": statistics.median_low(visits) if visits else None,
        "dwell_max_us": visits[-1] if visits else None,
    }


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _intact(folder: Path, aid: str) -> bool:
    try:
        manifest = _read_json(folder / MANIFEST)
        if manifest.get("analysis_id") != aid or manifest.get("status") != "SUCCEEDED":
            return False
        for a in manifest["artifacts"]:
            path = folder / a["path"]
            if Path(a["path"]).name != a["path"] or store.sha256_file(path) != a["sha256"]:
                return False
        return analysis_id(_read_json(folder / SPEC)) == aid
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _publish(tmp: Path, target: Path) -> None:
    """Rename the finished folder into place; a damaged earlier copy is moved aside first, then removed."""
    if target.exists():
        aside = target.parent / f".damaged-{target.name}-{secrets.token_hex(3)}"
        os.replace(target, aside)
        os.replace(tmp, target)
        shutil.rmtree(aside, ignore_errors=True)
    else:
        os.replace(tmp, target)
