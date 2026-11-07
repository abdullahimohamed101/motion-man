"""Runspec for the GPU smoke test `native_run_smoke` (plan native-g2 step 5).

Built by the controller's own ``native.native_runspec`` (``store.resolve_runspec`` plus the ``tensorrt`` backend
block, ``class_policy`` aware and codec avc1) on the crosswalk clip with the PC's model and engine manifests and a
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

from motionman import native, store, video_io  # noqa: E402
from motionman.types import RunConfig  # noqa: E402

FRAMES = 60


def build(source: Path, model_manifest: dict, engine_manifest: dict) -> dict:
    config = RunConfig(source=source, output_dir=None, device="cuda", model="rtdetr_r18_640", max_frames=FRAMES)
    cap, info = video_io.open_reader(source, config.limits)
    cap.release()
    return native.native_runspec(
        config, info, input_id=store.sha256_file(source), engine_manifest=engine_manifest,
        model_manifest=model_manifest, class_policy="aware",
    )  # fmt: skip


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
