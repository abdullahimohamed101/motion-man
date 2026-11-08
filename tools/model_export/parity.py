"""Detection parity between two runs of the same model on the same golden inputs (spec C§6; NG-8, later NG-10).

Detections are matched by class and geometry, never by array position: for each image, reference detections at or
above the frozen score threshold are matched greedily (highest score first) to unused candidate detections of the
same class with the highest IoU. Both directions are checked, so a candidate cannot pass by adding boxes. Gates
and the threshold come from the profile's frozen `parity` section; near-threshold and tiny-object misses are listed,
not dropped.

    python tools/model_export/parity.py rtdetr_r18_640 --ref r0 --cand onnx --gate fp32
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cache_dir, load_profile  # noqa: E402


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(ref: list[dict], cand: list[dict], threshold: float, min_iou: float) -> tuple[list, list]:
    """Pairs (ref, cand, iou) for ref detections ≥ threshold, and the unmatched ref detections."""
    used: set[int] = set()
    pairs, missed = [], []
    for r in sorted((r for r in ref if r["score"] >= threshold), key=lambda r: -r["score"]):
        best, best_iou = None, 0.0
        for j, c in enumerate(cand):
            if j in used or c["label"] != r["label"]:
                continue
            v = iou(r["box_xyxy"], c["box_xyxy"])
            if v > best_iou:
                best, best_iou = j, v
        if best is not None and best_iou >= min_iou:
            used.add(best)
            pairs.append((r, cand[best], best_iou))
        else:
            missed.append((r, best_iou))
    return pairs, missed


def compare(
    ref_doc: dict, cand_doc: dict, threshold: float, min_iou: float, max_dscore: float, near: float, tiny_area: float
) -> dict:
    total = matched = cand_total = cand_matched = non_finite = 0
    worst_dscore = 0.0
    notes = []
    for name, ref_img in ref_doc["images"].items():
        cand_img = cand_doc["images"][name]
        non_finite += ref_img.get("non_finite", 0) + cand_img.get("non_finite", 0)
        for direction, a, b in (("ref", ref_img, cand_img), ("cand", cand_img, ref_img)):
            pairs, missed = match(a["detections"], b["detections"], threshold, min_iou)
            n = len(pairs) + len(missed)
            if direction == "ref":
                total += n
                matched += len(pairs)
                for r, c, _ in pairs:
                    worst_dscore = max(worst_dscore, abs(r["score"] - c["score"]))
            else:
                cand_total += n
                cand_matched += len(pairs)
            for r, best in missed:
                x1, y1, x2, y2 = r["box_xyxy"]
                tags = [
                    t
                    for t, hit in (
                        ("near-threshold", r["score"] < threshold + near),
                        ("tiny", (x2 - x1) * (y2 - y1) < tiny_area),
                    )
                    if hit
                ]
                notes.append(
                    f"{name}: {direction} {r['class_name']} score {r['score']:.4f} best IoU {best:.4f} "
                    f"unmatched {' '.join(tags)}".rstrip()
                )
    return {
        "ref_detections": total,
        "ref_matched": matched,
        "ref_match_rate": matched / total if total else 1.0,
        "cand_detections": cand_total,
        "cand_matched": cand_matched,
        "cand_match_rate": cand_matched / cand_total if cand_total else 1.0,
        "max_score_error": worst_dscore,
        "non_finite": non_finite,
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--ref", default="r0")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--gate", required=True, help="gate name in the profile's parity.gates, e.g. fp32")
    args = ap.parse_args()
    profile = load_profile(args.profile)
    par = profile["parity"]
    gate = par["gates"][args.gate]
    d = cache_dir(profile)
    ref_doc = json.loads((d / f"{args.ref}.json").read_text())
    cand_doc = json.loads((d / f"{args.cand}.json").read_text())
    res = compare(
        ref_doc,
        cand_doc,
        par["score_threshold"],
        gate["min_iou"],
        gate["max_score_error"],
        par["near_threshold"],
        par["tiny_area_px"],
    )
    ok = (
        res["ref_match_rate"] >= gate["min_match_rate"]
        and res["cand_match_rate"] >= gate["min_match_rate"]
        and res["max_score_error"] <= gate["max_score_error"]
        and res["non_finite"] == 0
    )
    for n in res["notes"]:
        print("NOTE", n)
    print(
        f"{args.cand} vs {args.ref} ({args.gate}: IoU ≥ {gate['min_iou']}, |Δscore| ≤ {gate['max_score_error']}, "
        f"rate ≥ {gate['min_match_rate']}, threshold {par['score_threshold']}): "
        f"ref {res['ref_matched']}/{res['ref_detections']} ({res['ref_match_rate']:.4f}), "
        f"cand {res['cand_matched']}/{res['cand_detections']} ({res['cand_match_rate']:.4f}), "
        f"max |Δscore| {res['max_score_error']:.2e}, non-finite {res['non_finite']} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
