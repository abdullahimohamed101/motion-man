"""Golden parity images for G1 (plan native-g0-g1 D7): the crosswalk golden frames plus three synthetic images.

Runs on the Mac, where the clip lives. Images are written as lossless PNG outside git; their hashes are written to
`configs/reference/golden_images.json` so the PC can prove it uses the same pixels.

    .venv/bin/python tools/model_export/golden_images.py --source data/inputs/<hex>/source.mp4 --out <dir>
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "configs" / "reference" / "inputs.json"
OUT_JSON = ROOT / "configs" / "reference" / "golden_images.json"


def synthetic() -> dict[str, np.ndarray]:
    """Deterministic images that stress resize and channel order: portrait, landscape and a fine pattern (BGR)."""
    y, x = np.mgrid[0:1280, 0:720]
    portrait = np.stack([(x * 255 // 719), (y * 255 // 1279), ((x + y) % 256)], axis=-1).astype(np.uint8)
    cv2.rectangle(portrait, (100, 300), (600, 900), (0, 0, 255), -1)  # pure red block: catches BGR/RGB swaps
    landscape = np.full((720, 1280, 3), 40, np.uint8)
    cv2.circle(landscape, (400, 360), 200, (0, 255, 0), -1)
    cv2.rectangle(landscape, (800, 150), (1150, 600), (255, 0, 0), -1)
    pattern = (((np.indices((1000, 1000)).sum(axis=0) // 1) % 2) * 255).astype(np.uint8)  # 1-px checkerboard
    checker = cv2.merge([pattern] * 3)
    return {"synthetic_portrait": portrait, "synthetic_landscape": landscape, "synthetic_pattern": checker}


def frames(source: Path, indices: list[int]) -> dict[str, np.ndarray]:
    cap = cv2.VideoCapture(str(source))
    out = {}
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if not ok:
                raise SystemExit(f"could not read frame {i} from {source}")
            out[f"crosswalk_{i:05d}"] = img
    finally:
        cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ref = json.loads(INPUTS.read_text())["inputs"]["crosswalk"]
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if f"sha256:{digest}" != ref["input_id"]:
        raise SystemExit("source does not match configs/reference/inputs.json crosswalk input_id")
    args.out.mkdir(parents=True, exist_ok=True)
    images = {}
    for name, img in {**frames(args.source, ref["golden_frames"]), **synthetic()}.items():
        path = args.out / f"{name}.png"
        if not cv2.imwrite(str(path), img):
            raise SystemExit(f"could not write {path}")
        h, w = img.shape[:2]
        images[name] = {
            "file": path.name,
            "width": w,
            "height": h,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    doc = {
        "schema_version": "1.0",
        "description": "G1 parity images (D7). PNGs stay out of git; these hashes identify them. Crosswalk frames "
        "are decoded once on the Mac with OpenCV seeking, so the PNG, not the video, is the input.",
        "source_input_id": ref["input_id"],
        "images": images,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{len(images)} images -> {args.out}; hashes -> {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
