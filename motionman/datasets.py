"""Reviewed datasets: an editable draft and immutable, hash-indexed revisions (spec B§7, plan P4 D1–D6).

Ground truth exists only where a person reviewed a frame (INV-7). A draft starts from a run and a list of frames
to review; nothing is labelled until the reviewer acts. "Accept" copies a prediction into an annotation; the run's
records are only ever read. Commit validates the draft and writes ``revisions/r<N>.json`` once, never overwriting,
with its hash in ``revisions/index.json``. A revision is either ``eval`` (frozen for scoring) or ``tuning``.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from motionman import store
from motionman.types import MotionManError

SCHEMA_VERSION = "1.0"
DATASET_ID = re.compile(r"[a-z0-9-]{1,40}")
REF = re.compile(r"([a-z0-9-]{1,40})@r([1-9][0-9]*)")
SPLITS = ("eval", "tuning")
ORIGINS = ("accepted", "added", "corrected", "imported")
STATUSES = ("reviewed", "excluded")
DEFAULT_STEP = 30  # D4: every 30th frame (45 of the crosswalk clip's 1,356)
DRAFT, INDEX = "draft.json", "index.json"


class DatasetError(MotionManError):
    """A dataset, draft or revision is invalid or missing; the message says what and where."""


def even_frames(count: int, step: int = DEFAULT_STEP) -> list[int]:
    """Evenly spaced frames to review: 0, step, 2·step, … below ``count``."""
    if count < 1 or step < 1:
        raise DatasetError("a frame selection needs at least one frame and a positive step")
    return list(range(0, count, step))


def folder(root: store.DataRoot, dataset_id: str) -> Path:
    if not isinstance(dataset_id, str) or not DATASET_ID.fullmatch(dataset_id):
        raise DatasetError(f"dataset id must be 1–40 characters of a–z, 0–9 or '-', got {dataset_id!r}")
    return root.path / "datasets" / dataset_id


def list_datasets(root: store.DataRoot) -> list[str]:
    base = root.path / "datasets"
    return (
        sorted(p.name for p in base.iterdir() if p.is_dir() and DATASET_ID.fullmatch(p.name)) if base.is_dir() else []
    )


# --- drafts -----------------------------------------------------------------------------------------------------


def new_draft(root: store.DataRoot, dataset_id: str, run, frames: list[int]) -> dict:
    """Start a dataset from a run (its video, frame size and the frames to review). Refuses an existing id."""
    base = folder(root, dataset_id)
    if base.exists():
        raise DatasetError(f"a dataset named {dataset_id!r} already exists")
    if run.manifest is None:
        raise DatasetError("a dataset needs a verified run (not a legacy run) for its video identity")
    count = (run.manifest.get("counts") or {}).get("decoded_frames") or 0
    selected = sorted(set(frames))
    if not selected or selected[0] < 0 or selected[-1] >= count:
        raise DatasetError(f"frames to review must be within this run's 0..{count - 1}")
    draft = {"schema_version": SCHEMA_VERSION, "kind": "draft", "dataset_id": dataset_id,
             "input_id": run.manifest["input_id"], "frame_size": [run.width, run.height],
             "source_run_id": run.run_id, "selected": selected, "frames": [], "annotations": [],
             "next_annotation": 1}  # fmt: skip
    save_draft(root, draft)
    return draft


def load_draft(root: store.DataRoot, dataset_id: str) -> dict:
    path = folder(root, dataset_id) / DRAFT
    if not path.is_file():
        raise DatasetError(f"no draft for dataset {dataset_id!r}")
    return validate(_read(path), kind="draft")


def save_draft(root: store.DataRoot, draft: dict) -> None:
    validate(draft, kind="draft")  # never write something load_draft would refuse
    base = folder(root, draft["dataset_id"])
    base.mkdir(parents=True, exist_ok=True)
    store.write_json_atomic(base / DRAFT, draft)


def frame_entry(draft: dict, frame: int) -> dict | None:
    return next((f for f in draft["frames"] if f["frame"] == frame), None)


def set_frame(draft: dict, frame: int, status: str, classes_complete: list[str], ignored: list | None = None) -> None:
    """Record the reviewer's verdict on one frame. Excluding a frame removes its annotations."""
    entry = {"frame": frame, "status": status, "classes_complete": sorted(set(classes_complete)),
             "ignored": [list(b) for b in (ignored or [])]}  # fmt: skip
    draft["frames"] = sorted([f for f in draft["frames"] if f["frame"] != frame] + [entry], key=lambda f: f["frame"])
    if status == "excluded":
        draft["annotations"] = [a for a in draft["annotations"] if a["frame"] != frame]
    if frame not in draft["selected"]:
        draft["selected"] = sorted([*draft["selected"], frame])


def add_annotation(draft: dict, frame: int, class_name: str, bbox, origin: str, note: str = "",
                   gt_identity: int | None = None) -> str:  # fmt: skip
    """Add one ground-truth box on a frame already marked reviewed; returns its annotation id."""
    entry = frame_entry(draft, frame)
    if entry is None or entry["status"] != "reviewed":
        raise DatasetError(f"frame {frame} must be marked reviewed before it gets annotations")
    aid = f"a{draft['next_annotation']}"
    draft["annotations"].append({"annotation_id": aid, "frame": frame, "class_name": class_name,
                                 "bbox_xyxy": [round(float(v), 2) for v in bbox], "origin": origin, "note": note,
                                 "gt_identity": gt_identity})  # fmt: skip
    draft["next_annotation"] += 1
    _check_annotation(draft["annotations"][-1], draft["frame_size"], {f["frame"]: f for f in draft["frames"]})
    return aid


def accept_prediction(draft: dict, frame: int, obj: dict) -> str:
    """Copy one predicted object (from the run's records, which are only read) as an ``accepted`` annotation."""
    return add_annotation(draft, frame, obj["class_name"], obj["bbox_xyxy"], "accepted",
                          note=f"from {draft['source_run_id']} observation {obj['observation_id']}")  # fmt: skip


def remove_annotation(draft: dict, annotation_id: str) -> None:
    before = len(draft["annotations"])
    draft["annotations"] = [a for a in draft["annotations"] if a["annotation_id"] != annotation_id]
    if len(draft["annotations"]) == before:
        raise DatasetError(f"no annotation {annotation_id!r} in this draft")


# --- revisions --------------------------------------------------------------------------------------------------


def commit(root: store.DataRoot, dataset_id: str, split: str, now: datetime | None = None) -> tuple[int, Path]:
    """Freeze the draft's reviewed and excluded frames as the next revision. Never overwrites a revision."""
    if split not in SPLITS:
        raise DatasetError(f"split must be one of {', '.join(SPLITS)}")
    draft = load_draft(root, dataset_id)
    if not any(f["status"] == "reviewed" for f in draft["frames"]):
        raise DatasetError("nothing to commit: no frame has been reviewed yet")
    index = _index(root, dataset_id)
    base = folder(root, dataset_id) / "revisions"
    # Skip any revision file already on disk, so a commit interrupted before the index update can't block the next
    # one; such an orphan is never in the index, so load_revision refuses it.
    on_disk = [int(m.group(1)) for p in base.glob("r*.json") if (m := re.fullmatch(r"r(\d+)\.json", p.name))]
    number = max([r["revision"] for r in index["revisions"]] + on_disk + [0]) + 1
    revision = {"schema_version": SCHEMA_VERSION, "kind": "revision", "dataset_id": dataset_id, "revision": number,
                "parent_revision": number - 1 or None, "split": split, "input_id": draft["input_id"],
                "frame_size": draft["frame_size"], "source_run_id": draft["source_run_id"],
                "frames": draft["frames"], "annotations": draft["annotations"],
                "committed_at_utc": (now or datetime.now(UTC)).isoformat(timespec="seconds")}  # fmt: skip
    validate(revision, kind="revision")
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"r{number}.json"
    body = (json.dumps(revision, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)  # exclusive: a revision is written once
    except FileExistsError:
        raise DatasetError(f"revision r{number} of {dataset_id} already exists; reload and commit again") from None
    with os.fdopen(fd, "wb") as f:
        f.write(body)
    reviewed = sum(f["status"] == "reviewed" for f in revision["frames"])
    index["revisions"].append({"revision": number, "split": split, "sha256": store.sha256_file(path),
                               "frames_reviewed": reviewed, "annotations": len(revision["annotations"])})  # fmt: skip
    store.write_json_atomic(base / INDEX, index)
    return number, path


def parse_ref(ref: str) -> tuple[str, int]:
    m = REF.fullmatch(ref or "")
    if not m:
        raise DatasetError(f"a dataset revision is written <dataset>@r<N>, e.g. crosswalk-eval@r1; got {ref!r}")
    return m.group(1), int(m.group(2))


def load_revision(root: store.DataRoot, ref: str) -> dict:
    """A committed revision, verified against the hash recorded when it was committed."""
    dataset_id, number = parse_ref(ref)
    entry = next((r for r in _index(root, dataset_id)["revisions"] if r["revision"] == number), None)
    path = folder(root, dataset_id) / "revisions" / f"r{number}.json"
    if entry is None or not path.is_file():
        raise DatasetError(f"no revision {ref}")
    if store.sha256_file(path) != entry["sha256"]:
        raise DatasetError(f"revision {ref} does not match the hash recorded at commit; it was changed")
    return validate(_read(path), kind="revision")


def revision_sha256(root: store.DataRoot, ref: str) -> str:
    dataset_id, number = parse_ref(ref)
    return next(r["sha256"] for r in _index(root, dataset_id)["revisions"] if r["revision"] == number)


def list_revisions(root: store.DataRoot, dataset_id: str) -> list[dict]:
    return list(_index(root, dataset_id)["revisions"])


def import_file(root: store.DataRoot, path: Path, dataset_id: str | None = None) -> dict:
    """Import a MotionMan dataset JSON (a draft or a revision) as a new draft. Its labels become ``imported``,
    and they count as reviewed ground truth only once a person commits the draft."""
    body = _read(Path(path))
    kind = body.get("kind") if isinstance(body, dict) else None
    if kind not in ("draft", "revision"):
        raise DatasetError(f"{Path(path).name} is not a MotionMan dataset (kind must be draft or revision)")
    validate(body, kind=kind)
    new_id = dataset_id or body["dataset_id"]
    if folder(root, new_id).exists():
        raise DatasetError(f"a dataset named {new_id!r} already exists; import with another id")
    annotations = [{**a, "origin": "imported"} for a in body["annotations"]]
    draft = {"schema_version": SCHEMA_VERSION, "kind": "draft", "dataset_id": new_id, "input_id": body["input_id"],
             "frame_size": body["frame_size"], "source_run_id": body.get("source_run_id"),
             "selected": sorted({f["frame"] for f in body["frames"]} | set(body.get("selected", []))),
             "frames": body["frames"], "annotations": annotations,
             "next_annotation": 1 + max((int(a["annotation_id"][1:]) for a in annotations), default=0)}  # fmt: skip
    save_draft(root, draft)
    return draft


# --- validation -------------------------------------------------------------------------------------------------


def validate(doc: object, kind: str) -> dict:
    """Check a draft or revision completely; raise DatasetError naming the first problem."""
    if not isinstance(doc, dict) or doc.get("kind") != kind:
        raise DatasetError(f"expected a dataset {kind}")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError(f"unsupported schema_version {doc.get('schema_version')!r}")
    folder(store.DataRoot(Path(".")), doc.get("dataset_id"))  # id check only
    if not isinstance(doc.get("input_id"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", doc["input_id"]):
        raise DatasetError("input_id must be the video's 'sha256:<64 hex>' identity")
    size = doc.get("frame_size")
    if not (isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) and v > 0 for v in size)):
        raise DatasetError("frame_size must be [width, height] in pixels")
    frames = doc.get("frames")
    if not isinstance(frames, list):
        raise DatasetError("frames must be a list")
    by_frame = {}
    for f in frames:
        _check_frame(f, size)
        if f["frame"] in by_frame:
            raise DatasetError(f"frame {f['frame']} is listed twice")
        by_frame[f["frame"]] = f
    annotations = doc.get("annotations")
    if not isinstance(annotations, list):
        raise DatasetError("annotations must be a list")
    ids = set()
    for a in annotations:
        _check_annotation(a, size, by_frame)
        if a["annotation_id"] in ids:
            raise DatasetError(f"annotation id {a['annotation_id']} is used twice")
        ids.add(a["annotation_id"])
    if kind == "revision":
        if doc.get("split") not in SPLITS:
            raise DatasetError(f"split must be one of {', '.join(SPLITS)}")
        if not isinstance(doc.get("revision"), int) or doc["revision"] < 1:
            raise DatasetError("revision must be a positive whole number")
    return doc


def _check_frame(f: object, size: list[int]) -> None:
    if not isinstance(f, dict) or not isinstance(f.get("frame"), int) or isinstance(f.get("frame"), bool):
        raise DatasetError("each frame needs a whole-number frame index")
    where = f"frame {f['frame']}"
    if f["frame"] < 0 or f.get("status") not in STATUSES:
        raise DatasetError(f"{where}: status must be reviewed or excluded")
    classes = f.get("classes_complete")
    if not isinstance(classes, list) or not all(isinstance(c, str) and c.strip() for c in classes):
        raise DatasetError(f"{where}: classes_complete must be a list of class names")
    if f["status"] == "reviewed" and not classes:
        raise DatasetError(f"{where}: a reviewed frame must name at least one class it fully covers")
    for box in f.get("ignored", []):
        _check_box(box, size, f"{where} ignored area")


def _check_annotation(a: object, size: list[int], frames: dict) -> None:
    if not isinstance(a, dict) or not isinstance(a.get("annotation_id"), str):
        raise DatasetError("each annotation needs an annotation_id")
    where = f"annotation {a['annotation_id']}"
    entry = frames.get(a.get("frame"))
    if entry is None or entry["status"] != "reviewed":
        raise DatasetError(f"{where}: annotations are allowed only on reviewed frames")
    if not isinstance(a.get("class_name"), str) or not a["class_name"].strip():
        raise DatasetError(f"{where}: class_name is required")
    if a.get("origin") not in ORIGINS:
        raise DatasetError(f"{where}: origin must be one of {', '.join(ORIGINS)}")
    if a.get("gt_identity") is not None and (
        not isinstance(a["gt_identity"], int) or isinstance(a["gt_identity"], bool)
    ):
        raise DatasetError(f"{where}: gt_identity must be a whole number or null")
    _check_box(a.get("bbox_xyxy"), size, where)


def _check_box(box: object, size: list[int], where: str) -> None:
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int | float) and not isinstance(v, bool)
                                                              and math.isfinite(v) for v in box)):  # fmt: skip
        raise DatasetError(f"{where}: a box is [x1, y1, x2, y2] in source pixels")
    x1, y1, x2, y2 = box
    w, h = size
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise DatasetError(f"{where}: box must have positive size and lie within the {w}×{h} frame")


def _index(root: store.DataRoot, dataset_id: str) -> dict:
    path = folder(root, dataset_id) / "revisions" / INDEX
    return _read(path) if path.is_file() else {"schema_version": SCHEMA_VERSION, "revisions": []}


def _read(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DatasetError(f"can't read {Path(path).name}: {e}") from None
