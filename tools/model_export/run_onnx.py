"""ONNX Runtime FP32 detections on the golden tensors (plan native-g0-g1 step 7, NG-8). Feeds the exact input tensors
R0 used (`golden_tensors/<image>.npy`), so any difference is the runtime, not preprocessing. Writes `onnx.json` in
the same shape as `r0.json` for `parity.py`.

    ~/.cache/motionman/venv/bin/python tools/model_export/run_onnx.py rtdetr_r18_640
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cache_dir, load_profile, repo_commit, sha256_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort

    profile = load_profile(args.profile)
    d = cache_dir(profile)
    manifest = json.loads((d / "manifest.json").read_text())
    onnx_path = d / manifest["onnx"]["file"]
    if sha256_file(onnx_path) != manifest["onnx"]["sha256"]:
        raise SystemExit("model.onnx does not match manifest.json")
    r0 = json.loads((d / "r0.json").read_text())
    names = manifest["classes"]["names"]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    results = {}
    for name, ref in r0["images"].items():
        x = np.load(d / "golden_tensors" / f"{name}.npy")
        sizes = np.array([[ref["width"], ref["height"]]], dtype=np.int64)
        labels, boxes, scores = sess.run(["labels", "boxes", "scores"], {"images": x, "orig_target_sizes": sizes})
        labels, boxes, scores = labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()
        results[name] = {
            "width": ref["width"],
            "height": ref["height"],
            "non_finite": int(sum(1 for b, s in zip(boxes, scores, strict=True) if not np.isfinite([*b, s]).all())),
            "detections": [
                {"label": lab, "class_name": names[lab], "score": s, "box_xyxy": b}
                for lab, b, s in zip(labels, boxes, scores, strict=True)
            ],
        }
    doc = {
        "schema_version": "1.0",
        "name": "onnx",
        "profile": profile["profile"],
        "onnx_sha256": manifest["onnx"]["sha256"],
        "runtime": f"onnxruntime {ort.__version__} CPUExecutionProvider",
        "precision": "fp32",
        "inputs": "R0 golden tensors",
        "code": repo_commit(),
        "images": results,
    }
    (d / "onnx.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {d / 'onnx.json'} ({len(results)} images)")


if __name__ == "__main__":
    main()
