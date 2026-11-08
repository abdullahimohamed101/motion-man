"""Real replay parity (plan native-g2 step 7, N2-8, spec C§8).

Reads the admitted detections that ``motionman-native run --dump-detections`` wrote for a real clip, replays them
through the Python reference (Ultralytics ``BYTETracker`` with the installed ``bytetrack.yaml``, as ``reference.py``
does) in ``agnostic`` mode, and writes the result as a replay fixture. With ``--native``, it then runs the C++
checker on that fixture: native ByteTrack must give the same outputs, the same raw IDs and boxes within 1e-3 px.

The fixture is derived from footage and is large, so it stays out of git. Imports Ultralytics, so never in pytest.

Usage (on the PC, with the app env):
  python tools/tracker_replay/compare.py --dump dets.jsonl --out replay.json \
      --native build/linux-cuda-release/native/tests/test_tracker_replay
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures  # noqa: E402  (sibling modules; tools/ is not a package)
import reference  # noqa: E402


def load_dump(path: Path) -> list[list[list[float]]]:
    frames = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            row = json.loads(line)
            if row["frame"] != n:
                raise ValueError(f"{path}: line {n + 1} is frame {row['frame']}, expected {n}")
            frames.append(row["dets"])
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True, help="detections from motionman-native --dump-detections")
    ap.add_argument("--out", type=Path, required=True, help="replay fixture to write (not committed)")
    ap.add_argument("--name", default="crosswalk", help="scenario name in the fixture")
    ap.add_argument("--native", type=Path, default=None, help="test_tracker_replay binary to check the fixture")
    args = ap.parse_args()
    frames = load_dump(args.dump)
    config = reference.tracker_config()
    result = reference.replay(frames, config)
    doc = {
        "schema": fixtures.SCHEMA,
        "scenario": args.name,
        "description": f"admitted detections of {args.dump.name}, dumped by motionman-native",
        "class_policy": "agnostic",
        "instances": 1,
        "reference": reference.reference_identity(),
        "config": config,
        "frames": [{"dets": d, **r} for d, r in zip(frames, result, strict=True)],
    }
    fixtures.dump(doc, args.out)
    ids = {int(row[4]) for f in result for row in f["tracks"]}
    dets = sum(len(d) for d in frames)
    outputs = sum(len(f["tracks"]) for f in result)
    print(f"reference {doc['reference']}: {len(frames)} frames, {dets} detections, {outputs} track outputs, "
          f"{len(ids)} IDs (max {max(ids, default=0)}) → {args.out}")  # fmt: skip
    if args.native is None:
        return 0
    return subprocess.run([str(args.native), str(args.out)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
