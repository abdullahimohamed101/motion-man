"""Command-line entry point (spec §5).

``python -m motionman --source …`` runs one video, unchanged since the MVP. A first argument naming a subcommand
(plan P2 D11) dispatches instead: ``benchmark``, ``compare``.

Exit codes: 0 success · 1 unexpected failure · 2 invalid arguments, input, device or model · 75 data folder
locked · 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
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
        epilog="Without --output-dir, each run is staged, validated and published to <data-root>/runs/<run_id>/.",
    )
    p.add_argument("--source", type=Path, required=True, help="input video (constant frame rate, even dimensions)")
    p.add_argument("--output-dir", type=Path, default=None, help="legacy mode: new or empty directory for the results")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="canonical mode data folder (default: $MOTIONMAN_DATA_ROOT or ./data)",
    )
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="default: cpu")
    p.add_argument("--model", default="yolo11n.pt", help="Ultralytics weights (default: yolo11n.pt)")
    p.add_argument("--imgsz", type=_imgsz, default=640, help="inference size in pixels (default: 640)")
    p.add_argument("--max-frames", type=_positive_int, default=None, help="stop after this many frames")
    p.add_argument("--telemetry", action="store_true", help="sample process memory about once a second")
    return p


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.output_dir is not None and a.data_root is not None:
        parser.error("--data-root applies to canonical runs; leave it out when using --output-dir")
    return RunConfig(a.source, a.output_dir, a.device, a.model, a.imgsz, a.max_frames, data_root=a.data_root,
                     telemetry=a.telemetry)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in SUBCOMMANDS:
        return SUBCOMMANDS[argv[0]](argv[1:])
    config = parse_args(argv)  # argparse exits with 2 on bad flags
    out = config.output_dir
    # Only point at a report this run wrote, never at an earlier run's results in a refused directory.
    # Canonical runs log where their unpublished diagnostics are kept.
    preexisting = out is not None and out.exists() and (not out.is_dir() or any(out.iterdir()))

    def partial_note(status: str) -> str:
        ours = out is not None and not preexisting and (out / pipeline.REPORT).exists()
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
        f"({t['end_to_end_fps']} fps end to end, device {report['device']['used']}) → {report['run_dir']}"
    )
    return EXIT_OK


def benchmark_main(argv: list[str]) -> int:
    """``motionman benchmark``: repeat one configuration with fresh inference; median and range (plan P2 D5)."""
    from motionman import benchmark

    p = argparse.ArgumentParser(prog="motionman benchmark", description=benchmark_main.__doc__.split(": ", 1)[1])
    p.add_argument("--source", type=Path, required=True, help="input video")
    p.add_argument("--repeats", type=_positive_int, default=3, help="independent trials (default: 3)")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="default: cpu")
    p.add_argument("--imgsz", type=_imgsz, default=640, help="inference size in pixels (default: 640)")
    p.add_argument("--max-frames", type=_positive_int, default=None, help="stop each trial after this many frames")
    p.add_argument("--telemetry", action="store_true", help="sample process memory in each trial")
    p.add_argument("--label", default=None, help="a note stored with the result, e.g. 'mains, lid open'")
    p.add_argument("--data-root", type=Path, default=None, help="default: $MOTIONMAN_DATA_ROOT or ./data")
    a = p.parse_args(argv)
    if not a.source.is_file():
        p.error(f"--source: no such file: {a.source}")

    def show(t: dict) -> None:
        power = t["power"]["source"] + (f" {t['power']['percent']}%" if "percent" in t["power"] else "")
        if t["status"] == "succeeded":
            print(f"trial {t['trial']}: {t['end_to_end_fps']} fps, {t['processed_frames']} frames, "
                  f"run {t['run_id']} ({power})")  # fmt: skip
        else:
            print(f"trial {t['trial']}: FAILED ({power}): {t['error']}", file=sys.stderr)

    try:
        path, doc = benchmark.run_benchmark(
            a.source, repeats=a.repeats, device=a.device, imgsz=a.imgsz, max_frames=a.max_frames,
            telemetry=a.telemetry, label=a.label, data_root=a.data_root, on_trial=show,
        )  # fmt: skip
    except KeyboardInterrupt:
        print("interrupted; finished trials are kept in the benchmark file", file=sys.stderr)
        return EXIT_INTERRUPTED
    s = doc["summary"]
    fps = s["fps"]
    if fps["median"] is not None:
        print(f"median {fps['median']} fps (range {fps['min']}–{fps['max']}) over {fps['trials']} trials; "
              f"{s['failed']} failed; power {', '.join(s['power_sources'])} → {path}")  # fmt: skip
    else:
        print(f"no trial succeeded → {path}", file=sys.stderr)
    return EXIT_OK if s["failed"] == 0 else EXIT_FAILURE


def compare_main(argv: list[str]) -> int:
    """``motionman compare``: what changed between two benchmarks, and what else did (plan P2 D6)."""
    from motionman import compare

    p = argparse.ArgumentParser(prog="motionman compare", description=compare_main.__doc__.split(": ", 1)[1])
    p.add_argument("a", help="baseline benchmark: an ID under <data root>/benchmarks or a benchmark.json path")
    p.add_argument("b", help="candidate benchmark, same forms")
    p.add_argument(
        "--intended", default="", help=f"comma-separated factors changed on purpose: {', '.join(compare.FACTORS)}"
    )
    p.add_argument("--data-root", type=Path, default=None, help="default: $MOTIONMAN_DATA_ROOT or ./data")
    a = p.parse_args(argv)
    intended = [f.strip() for f in a.intended.split(",") if f.strip()]
    try:
        doc = compare.compare(a.a, a.b, intended, a.data_root)
        path = compare.write(doc, a.data_root)
    except MotionManError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    for side in ("a", "b"):
        f = doc[side]["fps"]
        rng = f"range {f['min']}–{f['max']}, {f['trials']} trials" if f.get("median") is not None else f.get("reason")
        print(f"{side.upper()} {doc[side]['benchmark_id']}: median {f.get('median')} fps ({rng})")
    for f in doc["factors"]:
        if f["changed"]:
            tag = "intended" if f["intended"] else "CONFOUNDER"
            print(f"  {tag:<10} {f['name']}: {json.dumps(f['a'])} → {json.dumps(f['b'])}")
    for name in doc["unchanged_intended"]:
        print(f"  note: --intended {name} but it did not change")
    for note in doc["notes"]:
        print(f"  note: {note}")
    print(doc["verdict"])
    print(f"{compare.SPEED_ONLY} → {path}")
    return EXIT_OK  # a confounded comparison is information, not a failure (D6)


SUBCOMMANDS = {"benchmark": benchmark_main, "compare": compare_main}
