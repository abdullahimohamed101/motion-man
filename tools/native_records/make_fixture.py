"""Shared fixture for the native record, timing and trail writers (plan native-g2 step 4, N2-5).

Python's own code produces the expected output: ``records.frame_record`` + ``json.dumps`` for tracks.jsonl lines,
the legacy ``{"source_frame_index", **round(ms, 3)}`` shape for timings.jsonl lines, and ``render.Trails`` for
trail points. The C++ writers must reproduce these bytes exactly (CTest ``native_records``); pytest rebuilds the
fixture and checks it still matches, so it can never drift from Python. Inputs use float32 values where the
native engine produces float32 (confidences, boxes).

Usage: python tools/native_records/make_fixture.py --write
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "native_records.json"
SCHEMA = "motionman.native_records/1"
NAMES = {0: "person", 2: "car", 9: "traffic light"}
sys.path.insert(0, str(ROOT))

from motionman import records, render  # noqa: E402
from motionman.types import Track  # noqa: E402


def f32(v: float) -> float:
    return float(np.float32(v))


def _record_case(frame_index, fps, width, height, objs, run_id="r_20261107T120000Z_abc123"):
    tracks = [Track(tid, cls, NAMES[cls], conf, tuple(box)) for tid, cls, conf, box in objs]
    case = {
        "run_id": run_id,
        "frame_index": frame_index,
        "fps": fps,
        "width": width,
        "height": height,
        # Standard JSON has no NaN: a non-finite coordinate is stored as null and read back as NaN.
        "objects": [
            [tid, cls, NAMES[cls], conf, *(v if math.isfinite(v) else None for v in box)]
            for tid, cls, conf, box in objs
        ],  # fmt: skip
    }
    try:
        case["line"] = json.dumps(records.frame_record(run_id, frame_index, fps, width, height, tracks))
    except records.RecordError:
        case["line"] = None  # the native writer must refuse this frame's object the same way
    return case


def record_cases() -> list[dict]:
    w, h = 1280, 720
    hand = [
        (0, 30.0, []),  # empty frame
        (1, 30.0, [(1, 0, f32(0.8984375), [100.0, 200.0, 160.0, 360.0])]),
        (2, 30000 / 1001, [(None, 2, f32(0.31), [0.125, 0.375, 2.675, 4.625])]),  # exact ties, null id
        (3, 30000 / 1001, [(7, 0, f32(0.99995), [f32(1279.995), f32(719.995), f32(1280.4), f32(720.2)])]),
        (4, 25.0, [(8, 9, f32(0.00004), [-0.4, -0.0, 10.0, 10.0])]),  # small overrun clipped, -0.0, tiny conf
        (5, 24000 / 1001, [(9, 0, 0.5, [-0.6, 0.0, 10.0, 10.0])]),  # overrun > 0.5 px: refused
        (6, 30.0, [(10, 0, 0.5, [50.0, 50.0, 50.0, 60.0])]),  # zero width: refused
        (7, 30.0, [(11, 0, 0.5, [math.nan, 0.0, 10.0, 10.0])]),  # non-finite: refused
        (10_000_000, 30000 / 1001, [(12, 2, f32(0.7), [f32(10.005), f32(20.015), f32(30.025), f32(40.035)])]),
        (11, 60.0, [(1, 0, f32(0.25), [0.0, 0.0, 1280.0, 720.0]), (2, 2, f32(0.33333), [1.0, 1.0, 2.0, 2.0])]),
    ]
    cases = [_record_case(i, fps, w, h, objs) for i, fps, objs in hand]
    rng = np.random.default_rng(20261109)
    for _ in range(120):
        objs = []
        for _ in range(int(rng.integers(0, 5))):
            x1, y1 = rng.uniform(-0.5, w - 2), rng.uniform(-0.5, h - 2)
            x2, y2 = rng.uniform(x1 + 0.5, w + 0.5), rng.uniform(y1 + 0.5, h + 0.5)
            tid = None if rng.random() < 0.2 else int(rng.integers(1, 10_000))
            cls = int(rng.choice([0, 2, 9]))
            objs.append((tid, cls, f32(rng.uniform(0.1, 1.0)), [f32(x1), f32(y1), f32(x2), f32(y2)]))
        fps = float(rng.choice([25.0, 30.0, 30000 / 1001, 60000 / 1001, 24000 / 1001]))
        cases.append(_record_case(int(rng.integers(0, 2_000_000)), fps, w, h, objs))
    return cases


def timing_cases() -> list[dict]:
    rng = np.random.default_rng(20261110)
    names = ["decode_ms", "preprocess_ms", "infer_ms", "track_ms", "render_ms", "write_ms"]
    fixed = [0.0005, 0.0015, 1.2345, 2.0, 1234.5675, 0.0004999, 99.9995]
    cases = []
    # Inputs use only uniform draws and exact scaling: lognormal/normal call libm, which differs in the last bit
    # between macOS and Linux and made the fixture platform-dependent.
    scales = [0.01, 1.0, 100.0]
    for n in range(60):
        values = fixed if n == 0 else [rng.uniform(0, 1) * float(rng.choice(scales)) for _ in range(6)] + [0.0]
        stages = dict(zip(names, values[: len(names)], strict=True))
        cases.append(
            {
                "frame_index": n,
                "stages": [[k, v] for k, v in stages.items()],
                "line": json.dumps({"source_frame_index": n, **{k: round(v, 3) for k, v in stages.items()}}),
            }
        )
    return cases


CHECKPOINTS = {29, 30, 31, 69, 70, 98, 99, 108, 109, 110, 128, 129, 130}  # caps, return after 30, expiry at 60


def trail_cases() -> list[dict]:
    """Frames of (id, box) through render.Trails; the expected points per ID after each frame."""
    rng = np.random.default_rng(20261111)
    trails = render.Trails()
    frames = []
    for f in range(140):
        objs = []
        for tid in (1, 2, 3):
            hidden = (tid == 2 and 40 <= f < 70) or (tid == 3 and f >= 50)  # 2 returns after 30, 3 leaves for good
            if not hidden:
                cx, cy = 100 + 3 * f + 50 * tid + rng.uniform(-3, 3), 300 + rng.uniform(-3, 3)
                objs.append([tid, f32(cx - 20), f32(cy - 40), f32(cx + 20), f32(cy + 40)])
        if f % 10 == 0:
            objs.append([None, 10.0, 10.0, 20.0, 20.0])  # no ID, no trail
        tracks = [Track(o[0], 0, "person", 0.9, tuple(o[1:])) for o in objs]
        trails.update(f, tracks)
        frame = {"objects": objs}
        if f % 10 == 9 or f in CHECKPOINTS:  # full trail state at checkpoints keeps the fixture small
            frame["points"] = {str(t): [list(p) for p in trails.points(t)] for t in (1, 2, 3) if trails.points(t)}
        frames.append(frame)
    return frames


def build() -> dict:
    return {
        "schema": SCHEMA,
        "class_names": {str(k): v for k, v in NAMES.items()},
        "records": record_cases(),
        "timings": timing_cases(),
        "trails": trail_cases(),
    }


def dump(doc: dict) -> str:
    head = {k: v for k, v in doc.items() if k not in ("records", "timings", "trails")}
    parts = [json.dumps(head, sort_keys=True)[:-1]]
    for key in ("records", "timings", "trails"):
        items = ",\n".join(json.dumps(c, sort_keys=True, allow_nan=False) for c in doc[key])
        parts.append(f'"{key}": [\n{items}\n]')
    return ", ".join(parts[:1]) + ",\n" + ",\n".join(parts[1:]) + "}\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help=f"write {FIXTURE.relative_to(ROOT)}")
    args = ap.parse_args()
    text = dump(build())
    if args.write:
        FIXTURE.write_text(text, encoding="utf-8")
    same = FIXTURE.is_file() and FIXTURE.read_text(encoding="utf-8") == text
    print(f"{FIXTURE.relative_to(ROOT)}: {'up to date' if same else 'DIFFERS (run with --write)'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
