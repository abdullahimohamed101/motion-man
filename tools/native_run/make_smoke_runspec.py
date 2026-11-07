"""Runspec for the GPU smoke test `native_run_smoke` (plan native-g2 step 5).

Built by Python's own ``store.resolve_runspec`` on the crosswalk clip, as the step 6 controller will, with a
``tensorrt`` backend block taken from the PC's model and engine manifests, ``class_policy`` aware, codec mp4v and a
60-frame range. The smoke test runs ``motionman-native run`` on these exact bytes.

Usage (on the Mac, with the PC's manifests copied locally):
    python tools/native_run/make_smoke_runspec.py --source clips/crosswalk.mp4 \
        --model-manifest manifest.json --engine-manifest engine_manifest.fp32.json --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "native_run" / "crosswalk_60.runspec.json"
sys.path.insert(0, str(ROOT))

from motionman import store, video_io  # noqa: E402
from motionman.types import RunConfig  # noqa: E402

FRAMES = 60
CODEC = "mp4v"  # Ubuntu's OpenCV 4.6 has no H.264 writer (plan Risks)


def build(source: Path, model_manifest: dict, engine_manifest: dict) -> dict:
    config = RunConfig(source=source, output_dir=None, device="cuda:0", model="rtdetr_r18_640", max_frames=FRAMES)
    cap, info = video_io.open_reader(source, config.limits)
    cap.release()
    identity = engine_manifest["identity"]
    backend = {
        "name": "tensorrt",
        "precision": identity["precision"],
        "profile": identity["profile"],
        "onnx_sha256": identity["onnx_sha256"],
        "engine_sha256": engine_manifest["engine"]["sha256"],
        "gpu": identity["gpu"],
        "tensorrt": identity["tensorrt"],
    }
    tracking = {"type": "bytetrack", "config": store.tracker_config(), "class_policy": "aware"}
    names = dict(enumerate(model_manifest["classes"]["names"]))
    return store.resolve_runspec(
        config,
        info,
        input_id=store.sha256_file(source),
        device_used="cuda:0",
        class_names=names,
        codec=CODEC,
        limits={"max_frames": config.limits.max_frames, "max_duration_s": config.limits.max_duration_s},
        backend=backend,
        tracking=tracking,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--model-manifest", type=Path, required=True)
    ap.add_argument("--engine-manifest", type=Path, required=True)
    ap.add_argument("--write", action="store_true", help=f"write {FIXTURE.relative_to(ROOT)}")
    args = ap.parse_args()
    runspec = build(
        args.source,
        json.loads(args.model_manifest.read_text(encoding="utf-8")),
        json.loads(args.engine_manifest.read_text(encoding="utf-8")),
    )
    text = json.dumps(runspec, indent=2, sort_keys=True) + "\n"
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(text, encoding="utf-8")
    print(f"{FIXTURE.relative_to(ROOT)}: config_sha256 {runspec['config_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
