"""Export the pinned RT-DETR checkpoint to ONNX and write manifest.json (plan native-g0-g1 step 6, spec C§5).

Uses the upstream deploy model and deploy postprocessor with the upstream exporter's arguments (opset, names,
dynamic batch); no simplifier, no extra NMS or scaling. Runs on the PC in the export venv:

    ~/.cache/motionman/venv/bin/python tools/model_export/export.py rtdetr_r18_640
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_model, cache_dir, load_profile, repo_commit, sha256_file  # noqa: E402

ONNX_DTYPES = {1: "float32", 6: "int32", 7: "int64", 10: "float16"}


def tensor_specs(values) -> dict:
    out = {}
    for v in values:
        t = v.type.tensor_type
        dims = [d.dim_param or d.dim_value for d in t.shape.dim]
        out[v.name] = {"dtype": ONNX_DTYPES.get(t.elem_type, str(t.elem_type)), "declared_shape": dims}
    return out


def observed_shapes(onnx_path: Path, size: int) -> dict:
    """Run the ONNX graph on CPU at batch 1 and 2: a dimension that follows the batch is "N", the rest are what the
    graph really produces. The exporter leaves output dims symbolic, so the declared shapes alone prove nothing."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    runs = []
    for n in (1, 2):
        feeds = {
            "images": np.random.default_rng(0).random((n, 3, size, size), dtype=np.float32),
            "orig_target_sizes": np.full((n, 2), size, dtype=np.int64),
        }
        outs = sess.run(None, feeds)
        runs.append({o.name: (list(feeds[o.name].shape)) for o in sess.get_inputs()})
        runs[-1].update({o.name: list(v.shape) for o, v in zip(sess.get_outputs(), outs, strict=True)})
    return {
        name: ["N" if a == 1 and b == 2 else a for a, b in zip(runs[0][name], runs[1][name], strict=True)]
        for name in runs[0]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    args = ap.parse_args()

    import onnx
    import torch

    profile = load_profile(args.profile)
    exp = profile["export"]
    model, names, _ = build_model(profile)
    out_dir = cache_dir(profile)
    onnx_path = out_dir / "model.onnx"

    size = exp["input_size"]
    data = torch.rand(1, 3, size, size)
    sizes = torch.tensor([[size, size]])
    with torch.no_grad():
        model(data, sizes)
        torch.onnx.export(
            model,
            (data, sizes),
            str(onnx_path),
            input_names=list(exp["inputs"]),
            output_names=list(exp["outputs"]),
            dynamic_axes={"images": {0: "N"}, "orig_target_sizes": {0: "N"}},
            opset_version=exp["opset"],
            do_constant_folding=True,
            dynamo=False,
        )
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    tensors = {"inputs": tensor_specs(m.graph.input), "outputs": tensor_specs(m.graph.output)}
    observed = observed_shapes(onnx_path, size)
    for group in tensors.values():
        for name, spec in group.items():
            spec["shape"] = observed[name]

    manifest = {
        "schema_version": "1.0",
        "profile": profile["profile"],
        "profile_sha256": sha256_file(
            Path(__file__).resolve().parents[2] / "configs" / "native" / f"{args.profile}.json"
        ),
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "upstream": {
            k: profile["upstream"][k]
            for k in ("repository", "commit", "subdir", "config", "config_overrides", "licence")
        },
        "checkpoint": {
            "sha256": sha256_file(cache_dir(profile) / "checkpoint.pth"),
            "url": profile["checkpoint"]["url"],
            "state": profile["checkpoint"]["state"],
        },
        "exporter": {
            "tool": "tools/model_export/export.py",
            "tool_sha256": sha256_file(Path(__file__)),
            **repo_commit(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "torch_onnx_exporter": "torchscript (dynamo=False)",
        },
        "onnx": {
            "file": onnx_path.name,
            "sha256": sha256_file(onnx_path),
            "bytes": onnx_path.stat().st_size,
            "opset": next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
            "tensors": tensors,
            "shape_note": "shape = observed with onnxruntime CPU at batch 1 and 2; declared_shape = as exported",
            "ops": sorted({n.op_type for n in m.graph.node}),
        },
        "classes": {"vocabulary": profile["classes"]["vocabulary"], "names": names},
        "precision_policy": "fp32 export; TensorRT precision is chosen at engine build (step 8)",
        "preprocessing_revision": None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported {onnx_path} ({manifest['onnx']['bytes']} bytes, {manifest['onnx']['sha256']})")
    print(json.dumps(manifest["onnx"]["tensors"]))


if __name__ == "__main__":
    main()
