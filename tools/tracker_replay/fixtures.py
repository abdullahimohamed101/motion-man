"""Tracker replay fixtures: read, write and compare (plan native-g2 steps 2, 3, 7). Pure Python and numpy only,
so pytest can use it without loading Ultralytics or torch (INV-8).

A fixture frame holds the input ``dets`` rows ``[x1, y1, x2, y2, score, cls]``, the tracker's output ``tracks``
rows ``[x1, y1, x2, y2, id, score, cls, idx]`` and ``unconfirmed`` (whether unconfirmed tracks were pending).
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "motionman.tracker_replay/1"
BOX_TOL_PX = 1e-3  # boxes come from a float64 Kalman state cast to float32; associations must match exactly


def load(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA}")
    return doc


def dump(doc: dict, path: Path) -> None:
    """Header fields indented, then one frame per line, so a diff shows which frames changed."""
    head = {k: v for k, v in doc.items() if k != "frames"}
    text = json.dumps(head, indent=1, sort_keys=True)[:-2]
    frames = ",\n".join(json.dumps(f, sort_keys=True, separators=(",", ":")) for f in doc["frames"])
    path.write_text(f'{text},\n "frames": [\n{frames}\n ]\n}}\n', encoding="utf-8")


def canonical_ids(frames: list[dict]) -> list[list[int]]:
    """Track IDs renamed 1, 2, 3… in order of first appearance (C§8: compare associations, not counter values)."""
    names: dict[int, int] = {}
    out = []
    for f in frames:
        row_ids = []
        for row in f["tracks"]:
            row_ids.append(names.setdefault(int(row[4]), len(names) + 1))
        out.append(row_ids)
    return out


def compare(expected: list[dict], actual: list[dict], box_tol: float = BOX_TOL_PX, rename: bool = True) -> list[str]:
    """Problems between two replays, frame by frame; empty means they agree.

    Rows are matched by detection index (``idx``), which both trackers report. Associations, classes, scores and
    the unconfirmed flag must be equal; boxes may differ by ``box_tol`` px. With ``rename``, IDs are compared
    after canonical renaming; without it, raw IDs must be equal.
    """
    problems: list[str] = []
    if len(expected) != len(actual):
        return [f"{len(actual)} frames, expected {len(expected)}"]
    ids_e = canonical_ids(expected) if rename else [[int(r[4]) for r in f["tracks"]] for f in expected]
    ids_a = canonical_ids(actual) if rename else [[int(r[4]) for r in f["tracks"]] for f in actual]
    for n, (fe, fa) in enumerate(zip(expected, actual, strict=True)):
        if fe["unconfirmed"] != fa["unconfirmed"]:
            problems.append(f"frame {n}: unconfirmed {fa['unconfirmed']}, expected {fe['unconfirmed']}")
        by_idx_e = {int(r[7]): (r, i) for r, i in zip(fe["tracks"], ids_e[n], strict=True)}
        by_idx_a = {int(r[7]): (r, i) for r, i in zip(fa["tracks"], ids_a[n], strict=True)}
        if by_idx_e.keys() != by_idx_a.keys():
            problems.append(f"frame {n}: output detections {sorted(by_idx_a)}, expected {sorted(by_idx_e)}")
            continue
        for idx, (re_, ie) in by_idx_e.items():
            ra, ia = by_idx_a[idx]
            if ia != ie:
                problems.append(f"frame {n} det {idx}: id {ia}, expected {ie}")
            if ra[5] != re_[5] or ra[6] != re_[6]:
                problems.append(f"frame {n} det {idx}: score/cls {ra[5:7]}, expected {re_[5:7]}")
            worst = max(abs(a - e) for a, e in zip(ra[:4], re_[:4], strict=True))
            if worst > box_tol:
                problems.append(f"frame {n} det {idx}: box off by {worst:.6f} px")
    return problems
