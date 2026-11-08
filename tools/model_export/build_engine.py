"""Build the TensorRT engine for a profile with trtexec and write engine_manifest.json (plan step 8, D6, spec C§5).

The manifest pins everything the engine depends on: ONNX and checkpoint hashes, exporter and preprocessing
revisions, GPU, driver, TensorRT and CUDA versions, the shape profile, precision and every build flag. The build log
is kept next to the engine. `--check` compares a manifest with the files and the current host and refuses (exit 1)
on any difference: an engine is only valid for the exact inputs and stack it was built from.

    python tools/model_export/build_engine.py rtdetr_r18_640 --precision fp32
    python tools/model_export/build_engine.py rtdetr_r18_640 --precision fp32 --check [--manifest <path>]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cache_dir, load_profile, repo_commit, sha256_file  # noqa: E402

PRECISION_FLAGS = {"fp32": ["--noTF32"]}  # TensorRT 11 is strongly typed: an FP32 ONNX builds FP32; no TF32
TRT_HEADER = Path("/usr/include/x86_64-linux-gnu/NvInferVersion.h")
CUDA_VERSION_JSON = Path("/usr/local/cuda/version.json")


def host_identity() -> dict:
    """GPU, driver, TensorRT and CUDA as this host reports them now."""
    gpu = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[0]
    )
    name, cc, driver = (s.strip() for s in gpu.split(","))
    header = TRT_HEADER.read_text()
    trt = ".".join(
        re.search(rf"#define TRT_{p}_ENTERPRISE (\d+)", header).group(1) for p in ("MAJOR", "MINOR", "PATCH", "BUILD")
    )
    cuda = json.loads(CUDA_VERSION_JSON.read_text())["cuda"]["version"]
    return {"gpu": name, "compute_capability": cc, "driver": driver, "tensorrt": trt, "cuda_toolkit": cuda}


def expected_identity(profile: dict, precision: str, host: dict) -> dict:
    """What an engine for this profile, precision and host must have been built from."""
    d = cache_dir(profile)
    model = json.loads((d / "manifest.json").read_text())
    shapes = {
        name: [1 if dim == "N" else dim for dim in spec["shape"]]
        for name, spec in model["onnx"]["tensors"]["inputs"].items()
    }
    return {
        "profile": profile["profile"],
        "precision": precision,
        "onnx_sha256": model["onnx"]["sha256"],
        "checkpoint_sha256": model["checkpoint"]["sha256"],
        "upstream_commit": model["upstream"]["commit"],
        "exporter_tool_sha256": model["exporter"]["tool_sha256"],
        "preprocessing_revision": profile["preprocessing"]["revision"],
        "shape_profile": {"min": shapes, "opt": shapes, "max": shapes},
        "build_flags": PRECISION_FLAGS[precision],
        "plugins": [],
        **host,
    }


def compare_identity(expected: dict, manifest: dict) -> list[str]:
    """Every field where an engine manifest differs from what this host and these inputs require."""
    got = manifest.get("identity", {})
    return [f"{k}: engine built with {got.get(k)!r}, now {v!r}" for k, v in expected.items() if got.get(k) != v]


def shape_spec(shapes: dict) -> str:
    return ",".join(f"{name}:{'x'.join(str(d) for d in dims)}" for name, dims in shapes.items())


def build(profile: dict, precision: str) -> int:
    d = cache_dir(profile)
    host = host_identity()
    ident = expected_identity(profile, precision, host)
    if sha256_file(d / "model.onnx") != ident["onnx_sha256"]:
        raise SystemExit("model.onnx does not match manifest.json; re-export first")
    engine = d / f"model.{precision}.engine"
    log = d / f"model.{precision}.build.log"
    layers = d / f"model.{precision}.layers.json"
    sp = ident["shape_profile"]
    cmd = [
        "trtexec",
        f"--onnx={d / 'model.onnx'}",
        f"--saveEngine={engine}",
        f"--minShapes={shape_spec(sp['min'])}",
        f"--optShapes={shape_spec(sp['opt'])}",
        f"--maxShapes={shape_spec(sp['max'])}",
        *ident["build_flags"],
        "--skipInference",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layers}",
    ]
    t0 = time.monotonic()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    seconds = time.monotonic() - t0
    if rc != 0 or not engine.exists():
        print(f"trtexec failed (exit {rc}); see {log}")
        return 1
    text = log.read_text()
    if f"TensorRT version: {'.'.join(host['tensorrt'].split('.')[:3])}" not in text:
        print(f"warning: build log does not state TensorRT {host['tensorrt']}")
    manifest = {
        "schema_version": "1.0",
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "identity": ident,
        "engine": {"file": engine.name, "sha256": sha256_file(engine), "bytes": engine.stat().st_size},
        "build": {
            "tool": "trtexec",
            "command": [str(c) for c in cmd],
            "seconds": round(seconds, 1),
            "log": log.name,
            "log_sha256": sha256_file(log),
            "layers": layers.name,
            "builder": "tools/model_export/build_engine.py",
            "builder_sha256": sha256_file(Path(__file__)),
            **repo_commit(),
        },
    }
    out = d / f"engine_manifest.{precision}.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"built {engine.name}: {manifest['engine']['bytes']} bytes in {seconds:.1f} s, {manifest['engine']['sha256']}"
    )
    host_keys = ("gpu", "compute_capability", "driver", "tensorrt", "cuda_toolkit")
    print(f"identity: {json.dumps({k: ident[k] for k in host_keys})}")
    return 0


def check(profile: dict, precision: str, manifest_path: Path | None) -> int:
    d = cache_dir(profile)
    path = manifest_path or d / f"engine_manifest.{precision}.json"
    manifest = json.loads(path.read_text())
    problems = compare_identity(expected_identity(profile, precision, host_identity()), manifest)
    engine = d / manifest["engine"]["file"]
    if not engine.exists() or sha256_file(engine) != manifest["engine"]["sha256"]:
        problems.append(f"engine file {engine} does not match the manifest hash")
    for p in problems:
        print("REFUSED", p)
    print("engine ok" if not problems else f"engine refused: {len(problems)} mismatches")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--precision", choices=sorted(PRECISION_FLAGS), default="fp32")
    ap.add_argument("--check", action="store_true", help="verify an existing engine instead of building")
    ap.add_argument("--manifest", type=Path, help="engine manifest to check (default: the cached one)")
    args = ap.parse_args()
    profile = load_profile(args.profile)
    return check(profile, args.precision, args.manifest) if args.check else build(profile, args.precision)


if __name__ == "__main__":
    sys.exit(main())
