"""Where two runs of the same video disagree, box by box (spec B§8, plan P4 D9).

Per common frame, boxes are matched one-to-one by position only: every pair at or above the IoU threshold, best
IoU first (ties: lower observation index in A, then in B), class ignored. Matched pairs either agree on the class
or are a class disagreement; the rest are unmatched in A or in B. Track IDs are never compared (they are local to
each run, MD fixture 7) and no run is declared the winner: differences are review candidates, not errors.
Results are published like P3/P4 results under ``disagreements/<disagreement_id>/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

from motionman import records, runs, store
from motionman.evaluate import _quantiles, iou
from motionman.types import MotionManError

SCHEMA_VERSION = "1.0"
VERSION = "spatial-v1"
DEFAULT_IOU = 0.5
TOP_FRAMES = 20
SUMMARY, ITEMS, MANIFEST = "disagreement.json", "items.jsonl", "manifest.json"
KINDS = ("matched_same_class", "class_disagreement", "unmatched_a", "unmatched_b")


class DisagreeError(MotionManError):
    """The two runs can't be compared; the message says why. Exit code 2."""


@dataclass(frozen=True)
class Result:
    disagreement_id: str
    folder: Path
    doc: dict
    reused: bool


def match_frame(a_objs: list[dict], b_objs: list[dict], threshold: float) -> list[tuple]:
    """``[(kind, a_obj | None, b_obj | None, iou | None), …]`` for one frame, in a stable order."""
    pairs = [(iou(a["bbox_xyxy"], b["bbox_xyxy"]), i, j) for i, a in enumerate(a_objs) for j, b in enumerate(b_objs)]
    pairs = sorted((p for p in pairs if p[0] >= threshold), key=lambda p: (-p[0], p[1], p[2]))
    used_a, used_b, out = set(), set(), []
    for v, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        kind = "matched_same_class" if a_objs[i]["class_name"] == b_objs[j]["class_name"] else "class_disagreement"
        out.append((kind, a_objs[i], b_objs[j], v))
    out += [("unmatched_a", a, None, None) for i, a in enumerate(a_objs) if i not in used_a]
    out += [("unmatched_b", None, b, None) for j, b in enumerate(b_objs) if j not in used_b]
    return out


def compare(a: runs.OpenRun, b: runs.OpenRun, root: store.DataRoot, iou_threshold: float = DEFAULT_IOU) -> Result:
    """Compare two verified runs of one video and publish the result (or reuse an identical one)."""
    for r in (a, b):
        if r.manifest is None or r.records_sha256 is None:
            raise DisagreeError(f"{r.run_id} is a legacy run; compare verified runs")
    if a.manifest["input_id"] != b.manifest["input_id"]:
        raise DisagreeError("the two runs are of different videos; boxes can only be compared on the same frames")
    if (a.width, a.height) != (b.width, b.height):
        raise DisagreeError("the two runs have different frame sizes")
    if not 0 < iou_threshold <= 1:
        raise DisagreeError("IoU threshold must be in (0, 1]")
    spec = {"a": [a.run_id, a.records_sha256], "b": [b.run_id, b.records_sha256], "iou": iou_threshold,
            "version": VERSION}  # fmt: skip
    did = "dg_" + store.sha256_json(spec).removeprefix("sha256:")[:16]
    target = root.path / "disagreements" / did
    if _intact(target, did):
        return Result(did, target, _read(target / SUMMARY), reused=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp-{did}-{secrets.token_hex(3)}"
    tmp.mkdir()
    try:
        doc = _build(a, b, iou_threshold, did, tmp)
        try:
            if target.exists():
                aside = target.parent / f".damaged-{did}-{secrets.token_hex(3)}"
                os.replace(target, aside)
                os.replace(tmp, target)
                shutil.rmtree(aside, ignore_errors=True)
            else:
                os.replace(tmp, target)
        except OSError:
            if _intact(target, did):
                return Result(did, target, _read(target / SUMMARY), reused=True)
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return Result(did, target, doc, reused=False)


def _build(a, b, threshold: float, did: str, out: Path) -> dict:
    counts = dict.fromkeys(KINDS, 0)
    by_class: dict[str, dict[str, int]] = {}
    ious: list[float] = []
    per_frame: dict[int, int] = {}
    frames = 0
    ha, hb = hashlib.sha256(), hashlib.sha256()
    with a.records_path.open("rb") as fa, b.records_path.open("rb") as fb, (out / ITEMS).open("w") as items:
        while True:
            la, lb = fa.readline(), fb.readline()
            ha.update(la)
            hb.update(lb)
            if not la.strip() or not lb.strip():
                break  # one run ended: only frames both processed are compared
            ra, rb = records.parse_line(la, a.records_path.name), records.parse_line(lb, b.records_path.name)
            if ra["source_frame_index"] != rb["source_frame_index"]:
                raise DisagreeError(f"frame order differs at {ra['source_frame_index']}/{rb['source_frame_index']}")
            frame, frames = ra["source_frame_index"], frames + 1
            for kind, oa, ob, v in match_frame(ra["objects"], rb["objects"], threshold):
                counts[kind] += 1
                cls = (oa or ob)["class_name"]
                by_class.setdefault(cls, dict.fromkeys(KINDS, 0))[kind] += 1
                if v is not None:
                    ious.append(v)
                if kind != "matched_same_class":
                    per_frame[frame] = per_frame.get(frame, 0) + 1
                    item = {"frame": frame, "kind": kind, "a": _brief(oa), "b": _brief(ob),
                            "iou": round(v, 4) if v is not None else None}  # fmt: skip
                    items.write(store.canonical_json(item) + "\n")
        for f, h in ((fa, ha), (fb, hb)):  # hash the rest, so both files are checked in full
            for line in f:
                h.update(line)
    for r, h in ((a, ha), (b, hb)):
        if f"sha256:{h.hexdigest()}" != r.records_sha256:
            raise DisagreeError(f"{r.run_id}'s records changed after the run was verified; reopen the run")
    top = sorted(per_frame.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_FRAMES]
    doc = {"schema_version": SCHEMA_VERSION, "kind": "disagreement", "disagreement_id": did, "version": VERSION,
           "runs": {"a": _side(a), "b": _side(b)}, "iou_threshold": threshold, "frames_compared": frames,
           "counts": counts, "by_class": dict(sorted(by_class.items())), "matched_iou": _quantiles(ious),
           "frames_with_differences": len(per_frame),
           "top_frames": [{"frame": f, "differences": n} for f, n in top],
           "note": "Boxes matched by position only; track IDs are local to each run and never compared. Differences "
                   "are review candidates; neither run is assumed right."}  # fmt: skip
    (out / SUMMARY).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = [{"path": n, "sha256": store.sha256_file(out / n)} for n in (SUMMARY, ITEMS)]
    manifest = {"schema_version": SCHEMA_VERSION, "disagreement_id": did, "status": "SUCCEEDED", "artifacts": artifacts}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return doc


def read_items(result: Result):
    with (result.folder / ITEMS).open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load(root: store.DataRoot, did: str) -> Result:
    target = root.path / "disagreements" / did
    if not _intact(target, did):
        raise DisagreeError(f"comparison {did} is missing or damaged; compare again")
    return Result(did, target, _read(target / SUMMARY), reused=True)


def _brief(o: dict | None) -> dict | None:
    if o is None:
        return None
    return {"observation_id": o["observation_id"], "class_name": o["class_name"], "confidence": o["confidence"],
            "bbox_xyxy": o["bbox_xyxy"]}  # fmt: skip


def _side(r) -> dict:
    inf = (r.runspec or {}).get("settings", {}).get("inference", {})
    return {"run_id": r.run_id, "records_sha256": r.records_sha256, "imgsz": inf.get("imgsz"), "conf": inf.get("conf")}


def _intact(folder: Path, did: str) -> bool:
    try:
        manifest = _read(folder / MANIFEST)
        listed = {x["path"]: x["sha256"] for x in manifest["artifacts"]}
        return (manifest.get("disagreement_id") == did and manifest.get("status") == "SUCCEEDED"
                and set(listed) == {SUMMARY, ITEMS}
                and all(store.sha256_file(folder / n) == h for n, h in listed.items())
                and _read(folder / SUMMARY).get("disagreement_id") == did)  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
