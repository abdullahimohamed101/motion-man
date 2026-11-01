"""Command-line entry point (spec §5).

Exit codes: 0 success · 1 unexpected failure · 2 invalid arguments, input, device or model · 130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from motionman import pipeline
from motionman.types import MotionManError, RunConfig

EXIT_OK, EXIT_FAILURE, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _imgsz(text: str) -> int:
    value = _positive_int(text)
    if value % 32 or not 32 <= value <= 4096:
        # Ultralytics would silently round it, and the report would then record the wrong size.
        raise argparse.ArgumentTypeError(f"must be a multiple of 32 between 32 and 4096, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motionman",
        description="Detect, track and annotate objects in one recorded video.",
        epilog="Writes annotated.mp4, tracks.jsonl and report.json to --output-dir.",
    )
    p.add_argument("--source", type=Path, required=True, help="input video (constant frame rate, even dimensions)")
    p.add_argument("--output-dir", type=Path, required=True, help="new or empty directory for the results")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="default: cpu")
    p.add_argument("--model", default="yolo11n.pt", help="Ultralytics weights (default: yolo11n.pt)")
    p.add_argument("--imgsz", type=_imgsz, default=640, help="inference size in pixels (default: 640)")
    p.add_argument("--max-frames", type=_positive_int, default=None, help="stop after this many frames")
    return p


def parse_args(argv: list[str] | None = None) -> RunConfig:
    a = build_parser().parse_args(argv)
    return RunConfig(a.source, a.output_dir, a.device, a.model, a.imgsz, a.max_frames)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = parse_args(argv)  # argparse exits with 2 on bad flags
    out = config.output_dir
    # Only point at a report this run wrote, never at an earlier run's results in a refused directory.
    preexisting = out.exists() and (not out.is_dir() or any(out.iterdir()))

    def partial_note(status: str) -> str:
        ours = not preexisting and (out / pipeline.REPORT).exists()
        return f"partial results and report (status: {status}) in {out}" if ours else ""

    try:
        report = pipeline.run(config)
    except MotionManError as e:
        print(f"error: {e}", file=sys.stderr)
        if note := partial_note("error"):
            print(note, file=sys.stderr)
        return e.exit_code
    except KeyboardInterrupt:
        note = partial_note("interrupted")
        print(f"interrupted{'; ' + note if note else ''}", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as e:
        traceback.print_exc()
        print(f"error: unexpected {type(e).__name__}: {e}", file=sys.stderr)
        if note := partial_note("error"):
            print(note, file=sys.stderr)
        return EXIT_FAILURE
    t = report["timing"]
    print(
        f"done: {report['processed_frames']} frames in {t['processing_wall_s']} s "
        f"({t['end_to_end_fps']} fps end to end, device {report['device']['used']}) → {config.output_dir}"
    )
    return EXIT_OK
