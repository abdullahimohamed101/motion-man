"""R0: the pinned PyTorch FP32 reference detections on the golden images (plan native-g0-g1 step 6, spec C§6).

Preprocessing is the upstream validation transform (read from the pinned config and refused if it changed):
PIL RGB, torchvision Resize to 640×640, float32 / 255, NCHW. TF32 is off, so CUDA matmuls and convolutions are
true FP32. Writes `r0.json` (all 300 detections per image, full float precision) and the input tensors as
`golden_tensors/<image>.npy` for the C++ preprocessing check in step 7.

    ~/.cache/motionman/venv/bin/python tools/model_export/reference.py rtdetr_r18_640 --images ~/.cache/motionman/golden
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GOLDEN, build_model, cache_dir, load_profile, repo_commit, sha256_file  # noqa: E402

EXPECTED_OPS = [{"type": "Resize", "size": [640, 640]}, {"type": "ConvertPILImage", "dtype": "float32", "scale": True}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--images", type=Path, required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from torchvision.transforms import v2 as T

    profile = load_profile(args.profile)
    ref = profile["reference"]
    torch.backends.cuda.matmul.allow_tf32 = ref["tf32"]
    torch.backends.cudnn.allow_tf32 = ref["tf32"]
    torch.set_float32_matmul_precision("highest")
    device = torch.device(ref["device"])

    model, names, cfg = build_model(profile)
    ops = cfg.yaml_cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
    if ops != EXPECTED_OPS:
        raise SystemExit(f"upstream validation transform changed: {ops}")
    from src.data.transforms._transforms import ConvertPILImage  # pinned upstream, imported by build_model's path

    size = profile["export"]["input_size"]
    transform = T.Compose([T.Resize([size, size]), ConvertPILImage(dtype="float32", scale=True)])
    model = model.to(device)

    golden = json.loads(GOLDEN.read_text())["images"]
    tensors_dir = cache_dir(profile) / "golden_tensors"
    tensors_dir.mkdir(exist_ok=True)
    results = {}
    for name, meta in golden.items():
        path = args.images / meta["file"]
        if sha256_file(path) != meta["sha256"]:
            raise SystemExit(f"{path} does not match configs/reference/golden_images.json")
        img = Image.open(path).convert("RGB")
        w, h = img.size
        x = transform(img).unsqueeze(0).contiguous()
        arr = x.numpy()
        np.save(tensors_dir / f"{name}.npy", arr)
        with torch.no_grad():
            labels, boxes, scores = model(x.to(device), torch.tensor([[w, h]], device=device))
        labels, boxes, scores = labels[0].cpu().tolist(), boxes[0].cpu().tolist(), scores[0].cpu().tolist()
        bad = sum(1 for b, s in zip(boxes, scores, strict=True) if not all(map(np.isfinite, [*b, s])))
        results[name] = {
            "image_sha256": meta["sha256"],
            "width": w,
            "height": h,
            "input_tensor": {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": "sha256:" + hashlib.sha256(arr.tobytes()).hexdigest(),
            },
            "non_finite": bad,
            "detections": [
                {"label": lab, "class_name": names[lab], "score": s, "box_xyxy": b}
                for lab, b, s in zip(labels, boxes, scores, strict=True)
            ],
        }
        top = [d for d in results[name]["detections"] if d["score"] >= 0.5]
        print(f"{name}: {w}x{h}, {len(top)} detections >= 0.5, non-finite {bad}")

    doc = {
        "schema_version": "1.0",
        "name": "R0",
        "profile": profile["profile"],
        "checkpoint_sha256": profile["checkpoint"]["sha256"],
        "upstream_commit": profile["upstream"]["commit"],
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "precision": ref["precision"],
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "preprocessing": {
            "source": "upstream val_dataloader ops",
            "ops": ops,
            "image_decode": "PIL convert('RGB')",
            "layout": "NCHW",
        },
        "code": repo_commit(),
        "images": results,
    }
    (cache_dir(profile) / "r0.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {cache_dir(profile) / 'r0.json'}")


if __name__ == "__main__":
    main()
