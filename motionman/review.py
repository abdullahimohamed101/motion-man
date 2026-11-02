"""Review queue: frames worth a person's look (spec B§7, plan P4 D5).

Simple heuristics over saved records, plus differences between two runs. They are *candidates, not errors*: a
low-confidence box may be right, a track gap may be a real occlusion, and two runs may both be wrong. The queue
only suggests frames; nothing here labels anything or runs a model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from motionman.evaluate import iou

LOW_CONFIDENCE = 0.40  # below this, a box is worth a second look
ABRUPT_IOU = 0.5  # a track's box overlapping its previous-frame box less than this jumped suddenly
CAP = 200  # candidates shown at once; the totals still count everything
KIND_ORDER = ("disagreement", "abrupt_box_change", "track_gap", "low_confidence")
KIND_LABELS = {"disagreement": "Runs disagree", "abrupt_box_change": "Box jumped", "track_gap": "Track reappeared",
               "low_confidence": "Low confidence"}  # fmt: skip
NOTE = "Candidates, not errors: each is a reason to look at a frame, not a finding."


@dataclass(frozen=True)
class Candidate:
    frame: int
    kind: str
    detail: str
    observation_id: str | None = None
    track_id: int | None = None

    def sort_key(self) -> tuple:
        return (KIND_ORDER.index(self.kind), self.frame, self.observation_id or "")


def candidates(records: Iterable[dict], disagreement_items: Iterable[dict] = (), cap: int = CAP,
               ) -> tuple[list[Candidate], dict[str, int]]:  # fmt: skip
    """``(shown, totals)``: up to ``cap`` candidates, kinds taking turns (each in frame order), and the count of
    every kind before the cap."""
    found: list[Candidate] = []
    last: dict[int, tuple[int, list[float]]] = {}  # track → (frame last seen, its box)
    for rec in records:
        frame = rec["source_frame_index"]
        for o in rec["objects"]:
            oid, tid = o["observation_id"], o["track_id"]
            if o["confidence"] < LOW_CONFIDENCE:
                found.append(Candidate(frame, "low_confidence", f"{o['class_name']} at {o['confidence']:.2f}", oid,
                                       tid))  # fmt: skip
            if tid is None:
                continue
            prev = last.get(tid)
            if prev is not None:
                gap = frame - prev[0] - 1
                if gap > 0:
                    found.append(Candidate(frame, "track_gap", f"#{tid} {o['class_name']} back after {gap} frames",
                                           oid, tid))  # fmt: skip
                else:
                    overlap = iou(prev[1], o["bbox_xyxy"])
                    if overlap < ABRUPT_IOU:
                        found.append(Candidate(frame, "abrupt_box_change",
                                               f"#{tid} {o['class_name']} box overlap {overlap:.2f} with the frame "
                                               "before", oid, tid))  # fmt: skip
            last[tid] = (frame, o["bbox_xyxy"])
    for item in disagreement_items:
        a, b = item.get("a"), item.get("b")
        what = {"class_disagreement": f"{a and a['class_name']} vs {b and b['class_name']}",
                "unmatched_a": f"{a and a['class_name']} only in run A",
                "unmatched_b": f"{b and b['class_name']} only in run B"}.get(item["kind"], item["kind"])  # fmt: skip
        found.append(Candidate(item["frame"], "disagreement", what, (a or b or {}).get("observation_id")))
    totals = {k: sum(c.kind == k for c in found) for k in KIND_ORDER}
    # Take turns between kinds (each in frame order): on the crosswalk run, run differences alone number 8,703 and a
    # strict kind-first order filled all 200 places with them, hiding every other reason to look.
    queues = [sorted((c for c in found if c.kind == k), key=Candidate.sort_key) for k in KIND_ORDER]
    shown: list[Candidate] = []
    for i in range(max((len(q) for q in queues), default=0)):
        shown += [q[i] for q in queues if i < len(q)]
        if len(shown) >= cap:
            break
    return shown[:cap], totals


def frames_of(shown: list[Candidate]) -> list[int]:
    """The distinct frames in ``shown``, in queue order (for "Add to review")."""
    seen: dict[int, None] = {}
    for c in shown:
        seen.setdefault(c.frame, None)
    return list(seen)
