"""Command-line entry point (spec §5). Full validation and exit codes land in plan step 5."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from motionman import pipeline
from motionman.types import RunConfig
from motionman.video_io import VideoError


def parse_args(argv: list[str] | None = None) -> RunConfig:
    p = argparse.ArgumentParser(prog="motionman", description="Detect, track and annotate objects in a video.")
    p.add_argument("--source", type=Path, required=True, help="input MP4 (constant frame rate)")
    p.add_argument("--output-dir", type=Path, required=True, help="new or empty directory for results")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=None)
    a = p.parse_args(argv)
    return RunConfig(a.source, a.output_dir, a.device, a.model, a.imgsz, a.max_frames)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = parse_args(argv)
    try:
        pipeline.run(config)
    except VideoError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
