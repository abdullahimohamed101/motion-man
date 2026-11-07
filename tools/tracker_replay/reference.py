"""Python reference tracker for replay parity (plan native-g2 step 2, spec C§8, ADR-006).

Drives Ultralytics' own ``BYTETracker`` (the tracker the legacy backend uses, with the installed
``bytetrack.yaml``) on cached detections, bypassing inference. Detections enter as float32, as they do from
``Boxes.cpu().numpy()`` in ``model.track``; the output rows are the tracker's own ``[x1, y1, x2, y2, id, score,
cls, idx]`` float32 array. No GPU, network or weights (INV-8).

This imports Ultralytics, so it never runs inside pytest (INV-8); pytest checks the committed fixtures instead.

Usage:
  python tools/tracker_replay/reference.py --check   # rerun the reference; exit 1 if any fixture differs
  python tools/tracker_replay/reference.py --write   # regenerate tests/fixtures/tracker/*.json
"""

from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "tracker"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(ROOT))  # motionman, when run as a script

import fixtures  # noqa: E402  (sibling modules; tools/ is not a package)
import scenarios  # noqa: E402


class Detections:
    """The slice of Ultralytics ``Boxes`` that ``BYTETracker.update`` reads: conf, cls, xywh, xyxy, masks, len."""

    def __init__(self, rows: np.ndarray) -> None:
        self.data = rows.reshape(-1, 6).astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, mask) -> Detections:
        return Detections(self.data[mask])

    @property
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4]

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.data[:, 0], self.data[:, 1], self.data[:, 2], self.data[:, 3]
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 5]


def tracker_config() -> dict:
    from motionman import store

    return store.tracker_config()


def new_tracker(config: dict):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    return BYTETracker(IterableSimpleNamespace(**config))


def replay(frames: list[list[list[float]]], config: dict, instances: int = 1) -> list[dict]:
    """Feed each frame's ``[x1, y1, x2, y2, score, cls]`` rows to ``instances`` independent trackers in turn.

    Returns one entry per frame: the first tracker's output rows and whether it holds unconfirmed tracks (the
    legacy backend records nothing for such a frame when the output is empty). Every other instance must
    produce identical output, since IDs are per tracker; a difference raises.
    """
    trackers = [new_tracker(config) for _ in range(instances)]
    out = []
    for n, dets in enumerate(frames):
        results = []
        for t in trackers:
            rows = t.update(Detections(np.asarray(dets, dtype=np.float32)))
            results.append(
                {
                    "tracks": [[float(v) for v in row] for row in np.asarray(rows, dtype=np.float32).reshape(-1, 8)],
                    "unconfirmed": any(not s.is_activated for s in t.tracked_stracks),
                }
            )
        if any(r != results[0] for r in results[1:]):
            raise AssertionError(f"frame {n}: tracker instances disagree on identical input")
        out.append(results[0])
    return out


def reference_identity() -> dict:
    return {
        "ultralytics": metadata.version("ultralytics"),
        "lap": metadata.version("lap"),
        "numpy": metadata.version("numpy"),
        "tracker": "bytetrack.yaml",
    }


def build_fixture(scenario: scenarios.Scenario, config: dict) -> dict:
    frames = scenario.frames()
    result = replay(frames, config, scenario.instances)
    return {
        "schema": fixtures.SCHEMA,
        "scenario": scenario.name,
        "description": scenario.description,
        "class_policy": "agnostic",
        "instances": scenario.instances,
        "reference": reference_identity(),
        "config": config,
        "frames": [{"dets": dets, **r} for dets, r in zip(frames, result, strict=True)],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare a fresh replay with the committed fixtures")
    mode.add_argument("--write", action="store_true", help="write tests/fixtures/tracker/<scenario>.json")
    args = ap.parse_args()
    config = tracker_config()
    failed = 0
    for scenario in scenarios.ALL:
        fixture = build_fixture(scenario, config)
        ids = {int(row[4]) for f in fixture["frames"] for row in f["tracks"]}
        line = f"{scenario.name}: {len(fixture['frames'])} frames, {len(ids)} track ids"
        path = FIXTURES / f"{scenario.name}.json"
        if args.write:
            FIXTURES.mkdir(parents=True, exist_ok=True)
            fixtures.dump(fixture, path)
        elif args.check:
            committed = fixtures.load(path)
            problems = (
                []
                if committed["frames"] == fixture["frames"]
                else (fixtures.compare(committed["frames"], fixture["frames"], rename=False) or ["inputs differ"])
            )
            if committed["config"] != config:
                problems.append("installed bytetrack.yaml differs from the fixture's config")
            line += " ok" if not problems else " DIFFERS: " + "; ".join(problems[:3])
            failed += bool(problems)
        print(line)
    if args.check:
        print(f"{len(scenarios.ALL) - failed}/{len(scenarios.ALL)} fixtures reproduced by {reference_identity()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
