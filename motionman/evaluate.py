"""Fixed-threshold detection evaluation against a reviewed revision (spec B§8, plan P4 D6–D8).

``score`` is pure: records and a revision in, counts out. ``evaluate`` checks compatibility, re-hashes the run's
records while reading them, and publishes ``evaluations/<run_id>/<evaluation_id>/`` the way P3 publishes analyses
(temp folder, atomic rename, an intact identical result is reused). This is per-class precision and recall at
one confidence and one IoU threshold on the reviewed frames only; it is never called mAP and says nothing about
frames, classes or footage that weren't reviewed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motionman import datasets, records, runs, store
from motionman.types import MotionManError

SCHEMA_VERSION = "1.0"
EVALUATOR_VERSION = "detection-v1"
METRIC_NAME = "fixed-threshold detection precision/recall"
DEFAULT_IOU = 0.5
MAX_EXAMPLES = 12
RESULT, MANIFEST = "evaluation.json", "manifest.json"


class EvaluationError(MotionManError):
    """The evaluation can't run on this input; the message says why. Exit code 2."""


@dataclass(frozen=True)
class Result:
    evaluation_id: str
    folder: Path
    doc: dict
    reused: bool


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ann_order(a: dict) -> tuple:
    m = re.fullmatch(r"a(\d+)", a["annotation_id"])
    return (int(m.group(1)) if m else 1 << 62, a["annotation_id"])  # a2 before a10


def _obs_order(o: dict) -> int:
    return int(str(o.get("observation_id", "0:0")).split(":")[-1])


def _inside(box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def score(by_frame: dict[int, dict], revision: dict, conf: float, iou_threshold: float) -> dict:
    """Match predictions to reviewed boxes per frame and class (plan D7). ``by_frame`` maps frame → record."""
    per_class: dict[str, dict] = {}
    ious: list[float] = []
    fps_ex: list[dict] = []
    fns_ex: list[dict] = []
    counts = {"reviewed": 0, "excluded": 0, "scored": 0, "not_evaluated": [], "ignored_predictions": 0,
              "unscored_predictions": 0, "unscored_annotations": 0}  # fmt: skip
    anns_by_frame: dict[int, list[dict]] = {}
    for a in revision["annotations"]:
        anns_by_frame.setdefault(a["frame"], []).append(a)
    for f in revision["frames"]:
        frame = f["frame"]
        if f["status"] == "excluded":
            counts["excluded"] += 1
            continue
        counts["reviewed"] += 1
        rec = by_frame.get(frame)
        if rec is None:
            counts["not_evaluated"].append(frame)  # outside this run's processed range: not a miss
            continue
        counts["scored"] += 1
        complete = set(f["classes_complete"])
        gts_all = anns_by_frame.get(frame, [])
        counts["unscored_annotations"] += sum(a["class_name"] not in complete for a in gts_all)
        kept = [o for o in rec["objects"] if o["confidence"] >= conf]
        counts["unscored_predictions"] += sum(o["class_name"] not in complete for o in kept)
        for cls in sorted(complete):
            c = per_class.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0, "frames": 0})
            c["frames"] += 1
            gts = sorted((a for a in gts_all if a["class_name"] == cls), key=_ann_order)
            preds = []
            for o in kept:
                if o["class_name"] != cls:
                    continue
                x1, y1, x2, y2 = o["bbox_xyxy"]
                if any(_inside(box, (x1 + x2) / 2, (y1 + y2) / 2) for box in f["ignored"]):
                    counts["ignored_predictions"] += 1  # neither TP nor FP (B§8)
                    continue
                preds.append(o)
            preds.sort(key=lambda o: (-o["confidence"], _obs_order(o)))
            matched: set[str] = set()
            for o in preds:
                best, best_iou = None, iou_threshold
                for g in gts:  # in annotation order, so the first of equal IoUs wins (lower annotation_id)
                    if g["annotation_id"] in matched:
                        continue
                    v = iou(o["bbox_xyxy"], g["bbox_xyxy"])
                    if v >= best_iou and (best is None or v > best_iou):
                        best, best_iou = g, v
                if best is not None:
                    matched.add(best["annotation_id"])
                    c["tp"] += 1
                    ious.append(best_iou)
                else:
                    c["fp"] += 1
                    near = max((iou(o["bbox_xyxy"], g["bbox_xyxy"]) for g in gts), default=0.0)
                    fps_ex.append({"frame": frame, "class_name": cls, "observation_id": o["observation_id"],
                                   "confidence": o["confidence"], "bbox_xyxy": o["bbox_xyxy"],
                                   "best_iou": round(near, 4)})  # fmt: skip
            for g in gts:
                if g["annotation_id"] not in matched:
                    c["fn"] += 1
                    fns_ex.append({"frame": frame, "class_name": cls, "annotation_id": g["annotation_id"],
                                   "bbox_xyxy": g["bbox_xyxy"]})  # fmt: skip
    classes = {cls: {**c, **_rates(c)} for cls, c in sorted(per_class.items())}
    total = {k: sum(c[k] for c in per_class.values()) for k in ("tp", "fp", "fn")}
    return {
        "frames": counts,
        "classes": classes,
        "total": {**total, **_rates(total)},
        "matched_iou": _quantiles(ious),
        "examples": {"false_positives": sorted(fps_ex, key=lambda e: (-e["confidence"], e["frame"]))[:MAX_EXAMPLES],
                     "false_negatives": sorted(fns_ex, key=lambda e: (e["frame"], _ann_order(e)))[:MAX_EXAMPLES]},
    }  # fmt: skip


def _rates(c: dict) -> dict:
    """Precision and recall, each null with a reason when its denominator is zero (never a made-up 0 or 1)."""
    out: dict = {}
    if c["tp"] + c["fp"]:
        out["precision"] = round(c["tp"] / (c["tp"] + c["fp"]), 4)
    else:
        out.update(precision=None, precision_reason="no predictions to score")
    if c["tp"] + c["fn"]:
        out["recall"] = round(c["tp"] / (c["tp"] + c["fn"]), 4)
    else:
        out.update(recall=None, recall_reason="no reviewed boxes of this class")
    return out


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p10": None, "p50": None, "p90": None, "reason": "no matched boxes"}
    p10, p50, p90 = (round(float(v), 4) for v in np.percentile(values, [10, 50, 90]))
    return {"count": len(values), "p10": p10, "p50": p50, "p90": p90}


def evaluate(run: runs.OpenRun, ref: str, root: store.DataRoot, conf: float | None = None,
             iou_threshold: float = DEFAULT_IOU) -> Result:  # fmt: skip
    """Score ``run`` against frozen revision ``ref`` and publish the result (or reuse an identical one)."""
    revision = datasets.load_revision(root, ref)
    run_conf = float(run.runspec["settings"]["inference"]["conf"]) if run.runspec else None
    _check(run, revision, ref, run_conf, conf, iou_threshold)
    conf = run_conf if conf is None else conf
    rev_sha = datasets.revision_sha256(root, ref)
    thresholds = {"confidence": conf, "iou": iou_threshold}
    spec = {"run_id": run.run_id, "records_sha256": run.records_sha256, "dataset": ref, "revision_sha256": rev_sha,
            "thresholds": thresholds, "evaluator_version": EVALUATOR_VERSION}  # fmt: skip
    eid = "ev_" + store.sha256_json(spec).removeprefix("sha256:")[:16]
    target = root.path / "evaluations" / run.run_id / eid
    if _intact(target, eid):
        return Result(eid, target, _read(target / RESULT), reused=True)
    wanted = {f["frame"] for f in revision["frames"]}
    by_frame = _records(run, wanted)
    doc = {"schema_version": SCHEMA_VERSION, "kind": "evaluation", "evaluation_id": eid, "metric": METRIC_NAME,
           "evaluator_version": EVALUATOR_VERSION, "run_id": run.run_id, "records_sha256": run.records_sha256,
           "dataset": {"ref": ref, "sha256": rev_sha, "split": revision["split"], "input_id": revision["input_id"]},
           "thresholds": {"confidence": conf, "iou": iou_threshold}, "scope": (
               "Only the reviewed frames and, on each, only the classes marked complete. Not mAP; says nothing about "
               "other frames, classes, thresholds or footage."),
           **score(by_frame, revision, conf, iou_threshold)}  # fmt: skip
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp-{eid}-{secrets.token_hex(3)}"
    tmp.mkdir()
    try:
        _write(tmp / RESULT, doc)
        _write(tmp / MANIFEST, {"schema_version": SCHEMA_VERSION, "kind": "evaluation", "evaluation_id": eid,
                                "status": "SUCCEEDED", "artifacts": [{"path": RESULT,
                                "sha256": store.sha256_file(tmp / RESULT)}]})  # fmt: skip
        try:
            if target.exists():  # damaged earlier copy: move aside, publish, remove
                aside = target.parent / f".damaged-{eid}-{secrets.token_hex(3)}"
                os.replace(target, aside)
                os.replace(tmp, target)
                shutil.rmtree(aside, ignore_errors=True)
            else:
                os.replace(tmp, target)
        except OSError:
            if _intact(target, eid):  # an identical evaluation was published meanwhile
                return Result(eid, target, _read(target / RESULT), reused=True)
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return Result(eid, target, doc, reused=False)


def _check(run, revision: dict, ref: str, run_conf: float | None, conf: float | None, iou_threshold: float) -> None:
    if run.manifest is None or run.records_sha256 is None or run_conf is None:
        raise EvaluationError(f"{run.run_id} is a legacy run; evaluate a verified run")
    if revision["split"] != "eval":
        raise EvaluationError(f"{ref} is a tuning revision; only frozen eval revisions are scored (plan D6)")
    if revision["input_id"] != run.manifest["input_id"]:
        raise EvaluationError(f"{ref} was reviewed on a different video than this run's source")
    if revision["frame_size"] != [run.width, run.height]:
        raise EvaluationError(f"{ref} is {revision['frame_size']} px but this run is {run.width}×{run.height}")
    if conf is not None and not run_conf <= conf <= 1:
        raise EvaluationError(f"confidence must be between the run's own threshold ({run_conf}) and 1; a lower "
                              "one can't add predictions the run never kept")  # fmt: skip
    if not 0 < iou_threshold <= 1:
        raise EvaluationError("IoU threshold must be in (0, 1]")


def _records(run, wanted: set[int]) -> dict[int, dict]:
    """The wanted frames' records, with the whole file re-hashed: it must still be the one the run verified."""
    digest, out = hashlib.sha256(), {}
    with run.records_path.open("rb") as f:
        for n, line in enumerate(f, start=1):
            digest.update(line)
            if line.strip():
                rec = records.parse_line(line, f"{run.records_path.name}:{n}")
                if rec["source_frame_index"] in wanted:
                    out[rec["source_frame_index"]] = rec
    if f"sha256:{digest.hexdigest()}" != run.records_sha256:
        raise EvaluationError(f"{run.records_path.name} changed after the run was verified; reopen the run")
    return out


def load(root: store.DataRoot, run_id: str, eid: str) -> Result:
    target = root.path / "evaluations" / run_id / eid
    if not _intact(target, eid):
        raise EvaluationError(f"evaluation {eid} for {run_id} is missing or damaged; evaluate again")
    return Result(eid, target, _read(target / RESULT), reused=True)


def _intact(folder: Path, eid: str) -> bool:
    try:
        manifest = _read(folder / MANIFEST)
        return (manifest.get("evaluation_id") == eid and manifest.get("status") == "SUCCEEDED"
                and [a["path"] for a in manifest["artifacts"]] == [RESULT]
                and store.sha256_file(folder / RESULT) == manifest["artifacts"][0]["sha256"]
                and _read(folder / RESULT).get("evaluation_id") == eid)  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
        return False


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
