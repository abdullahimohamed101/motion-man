"""Region sets: the lines and rectangles an analysis asks about (spec B§3, plan P3 D2).

Coordinates are normalized to [0, 1] of the source frame and bound to one ``input_id``, so a region set means
the same place in every run of the same video. A set is content-addressed: identical content always has the same
``region_set_id`` and is stored once under ``<data root>/regions/<input_id>/``. Saving a set never runs anything.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from motionman import store
from motionman.types import MotionManError

SCHEMA_VERSION = "1.0"
REGION_ID = re.compile(r"[a-z0-9-]{1,32}")
INPUT_ID = re.compile(r"sha256:[0-9a-f]{64}")
KINDS = ("line", "rect")
MAX_REGIONS = 32
MAX_NAME = 60
COORD_DECIMALS = 6  # 1e-6 of a 1920 px frame is 0.002 px: rounding keeps ids stable without moving anything
MAX_DWELL_S = 3600
MAX_OCCUPANCY = 1000
_FIELDS = {"line": {"id", "name", "kind", "points", "classes"},
           "rect": {"id", "name", "kind", "box", "classes", "dwell_threshold_s", "occupancy_threshold"}}  # fmt: skip


class RegionError(MotionManError):
    """A region set is invalid; the message names the region and field."""


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    kind: str  # "line" or "rect"
    coords: tuple[float, ...]  # line: (ax, ay, bx, by) in direction A→B; rect: (x1, y1, x2, y2) with x1<x2, y1<y2
    classes: tuple[str, ...] | None = None  # None = every class
    dwell_threshold_s: float | None = None  # rect only
    occupancy_threshold: int | None = None  # rect only

    def to_px(self, width: int, height: int) -> tuple[float, ...]:
        """The same coordinates in source pixels."""
        return tuple(v * (width if i % 2 == 0 else height) for i, v in enumerate(self.coords))

    def to_dict(self) -> dict:
        body: dict = {"id": self.id, "name": self.name, "kind": self.kind, "classes": _list(self.classes)}
        if self.kind == "line":
            ax, ay, bx, by = self.coords
            body["points"] = [[ax, ay], [bx, by]]
        else:
            body["box"] = list(self.coords)
            body["dwell_threshold_s"] = self.dwell_threshold_s
            body["occupancy_threshold"] = self.occupancy_threshold
        return body


@dataclass(frozen=True)
class RegionSet:
    input_id: str
    name: str
    regions: tuple[Region, ...]

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "input_id": self.input_id, "name": self.name,
                "regions": [r.to_dict() for r in self.regions]}  # fmt: skip

    @property
    def sha256(self) -> str:
        return store.sha256_json(self.to_dict())

    @property
    def region_set_id(self) -> str:
        return "rs_" + self.sha256.removeprefix("sha256:")[:16]


def parse(body: object) -> RegionSet:
    """Validate a region-set document. Every refusal names what is wrong; nothing is guessed or clamped."""
    if not isinstance(body, dict):
        raise RegionError("a region set must be a JSON object")
    extra = set(body) - {"schema_version", "input_id", "name", "regions"}
    if extra:
        raise RegionError(f"unknown field(s): {', '.join(sorted(extra))}")
    if body.get("schema_version") != SCHEMA_VERSION:
        raise RegionError(f"unsupported schema_version {body.get('schema_version')!r}; expected {SCHEMA_VERSION!r}")
    input_id = body.get("input_id")
    if not isinstance(input_id, str) or not INPUT_ID.fullmatch(input_id):
        raise RegionError("input_id must be the video's 'sha256:<64 hex>' identity")
    name = _name(body.get("name"), "the region set")
    items = body.get("regions")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_REGIONS:
        raise RegionError(f"regions must be a list of 1 to {MAX_REGIONS} regions")
    regions = tuple(_region(item, i) for i, item in enumerate(items))
    ids = [r.id for r in regions]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise RegionError(f"duplicate region id(s): {', '.join(dupes)}")
    return RegionSet(input_id, name, regions)


def folder(root: store.DataRoot, input_id: str) -> Path:
    return root.path / "regions" / input_id.removeprefix("sha256:")


def save(root: store.DataRoot, region_set: RegionSet) -> Path:
    """Store the set under its content id; saving identical content again returns the same file."""
    parse(region_set.to_dict())  # never write something load() would refuse
    path = folder(root, region_set.input_id) / f"{region_set.region_set_id}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        store.write_json_atomic(path, region_set.to_dict())
    return path


def load_file(path: Path) -> RegionSet:
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RegionError(f"can't read region set {Path(path).name}: {e}") from None
    return parse(body)


def load(root: store.DataRoot, input_id: str, region_set_id: str) -> RegionSet:
    if not re.fullmatch(r"rs_[0-9a-f]{16}", region_set_id):
        raise RegionError(f"{region_set_id!r} is not a region set id")
    path = folder(root, input_id) / f"{region_set_id}.json"
    if not path.is_file():
        raise RegionError(f"no region set {region_set_id} for this video")
    region_set = load_file(path)
    if region_set.region_set_id != region_set_id or region_set.input_id != input_id:
        raise RegionError(f"region set {region_set_id} does not match its content; it was edited by hand")
    return region_set


def list_sets(root: store.DataRoot, input_id: str) -> list[RegionSet]:
    """Valid saved sets for one video, by name; unreadable files are skipped, never half-loaded."""
    out = []
    for path in sorted(folder(root, input_id).glob("rs_*.json")):
        try:
            region_set = load_file(path)
        except RegionError:
            continue
        if region_set.input_id == input_id and path.stem == region_set.region_set_id:
            out.append(region_set)
    return sorted(out, key=lambda s: (s.name.lower(), s.region_set_id))


def _region(item: object, i: int) -> Region:
    if not isinstance(item, dict):
        raise RegionError(f"region {i + 1} must be an object")
    rid = item.get("id")
    if not isinstance(rid, str) or not REGION_ID.fullmatch(rid):
        raise RegionError(f"region {i + 1}: id must be 1–32 characters of a–z, 0–9 or '-'")
    kind = item.get("kind")
    if kind not in KINDS:
        raise RegionError(f"region {rid}: kind must be 'line' or 'rect'")
    extra = set(item) - _FIELDS[kind]
    if extra:
        raise RegionError(f"region {rid}: unknown field(s) for a {kind}: {', '.join(sorted(extra))}")
    name = _name(item.get("name"), f"region {rid}")
    classes = _classes(item.get("classes"), rid)
    if kind == "line":
        pts = item.get("points")
        if not (isinstance(pts, list) and len(pts) == 2 and all(isinstance(p, list) and len(p) == 2 for p in pts)):
            raise RegionError(f"region {rid}: points must be [[x, y], [x, y]]")
        coords = tuple(_coord(v, rid) for p in pts for v in p)
        if coords[:2] == coords[2:]:
            raise RegionError(f"region {rid}: a line needs two different points (zero length)")
        return Region(rid, name, kind, coords, classes)
    box = item.get("box")
    if not (isinstance(box, list) and len(box) == 4):
        raise RegionError(f"region {rid}: box must be [x1, y1, x2, y2]")
    coords = tuple(_coord(v, rid) for v in box)
    if not (coords[0] < coords[2] and coords[1] < coords[3]):
        raise RegionError(f"region {rid}: box needs x1 < x2 and y1 < y2 (positive area)")
    dwell = item.get("dwell_threshold_s")
    if dwell is not None and (not _number(dwell) or not 0 < dwell <= MAX_DWELL_S):
        raise RegionError(f"region {rid}: dwell_threshold_s must be null or a number in (0, {MAX_DWELL_S}]")
    occ = item.get("occupancy_threshold")
    if occ is not None and (isinstance(occ, bool) or not isinstance(occ, int) or not 1 <= occ <= MAX_OCCUPANCY):
        raise RegionError(f"region {rid}: occupancy_threshold must be null or a whole number 1–{MAX_OCCUPANCY}")
    return Region(rid, name, kind, coords, classes, float(dwell) if dwell is not None else None, occ)


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_NAME:
        raise RegionError(f"{what}: name must be 1–{MAX_NAME} characters")
    return value.strip()


def _classes(value: object, rid: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(c, str) and c.strip() for c in value):
        raise RegionError(f"region {rid}: classes must be null (every class) or a list of class names")
    if len(set(value)) != len(value):
        raise RegionError(f"region {rid}: classes are listed twice")
    return tuple(value)


def _number(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _coord(v: object, rid: str) -> float:
    if not _number(v) or not 0 <= v <= 1:
        raise RegionError(f"region {rid}: coordinates must be numbers from 0 to 1 (fractions of the frame)")
    return round(float(v), COORD_DECIMALS)


def _list(values: tuple[str, ...] | None) -> list[str] | None:
    return list(values) if values is not None else None
