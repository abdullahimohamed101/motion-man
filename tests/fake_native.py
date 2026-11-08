"""A stand-in for ``motionman-native run`` (plan native-g2 step 6, INV-8): same arguments, outputs, status lines,
receipt and exit codes, with no GPU. ``FAKE_NATIVE_MODE`` picks a behaviour:

success · crash (dies mid-run, no receipt) · missing_records · short_records (receipt claims a frame the file lacks)
· short_video · bad_receipt (exit 0, receipt for another run) · status_error (exit 0, receipt says error)
· refused (exit 3) · interrupted (stops itself, as on SIGTERM) · slow (one frame per 50 ms, honours SIGINT)
· drained_tail (success, but the decoder gave the last 3 frames no timestamp)
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionman import records, store  # noqa: E402
from motionman.types import Track  # noqa: E402

STAGES = ("decode_ms", "preprocess_ms", "infer_ms", "track_ms", "render_ms", "write_ms")
stop = False


def on_signal(signum, frame):
    global stop
    stop = True


def status(**event) -> None:
    print(json.dumps(event), flush=True)


def main(argv: list[str]) -> int:
    mode = os.environ.get("FAKE_NATIVE_MODE", "success")
    assert argv[0] == "run"
    a = dict(zip(argv[1::2], argv[2::2], strict=True))
    out, run_id, source = Path(a["--out"]), a["--run-id"], Path(a["--source"])
    spec = json.loads(Path(a["--runspec"]).read_text(encoding="utf-8"))
    s = spec["settings"]
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    receipt = {
        "schema_version": "1.0", "status": "error", "error": None, "run_id": run_id,
        "runspec_sha256": store.sha256_file(Path(a["--runspec"])), "source_sha256": store.sha256_file(source),
        "frames": {"decoded": 0, "processed": 0, "records": 0, "timing_rows": 0, "video_written": 0,
                   "timestamps_missing": 3 if mode == "drained_tail" else 0},
        "engine": {"runtime": "tensorrt", "profile": s["backend"]["profile"], "precision": s["backend"]["precision"],
                   "engine_sha256": s["backend"]["engine_sha256"], "gpu": "Fake GPU 9000"},
        "timing": {"model_load_s": 0.5, "wall_s": 0.0, "processing_wall_s": 0.0},
    }  # fmt: skip

    def finish(code: int, state: str, error: str | None = None) -> int:
        receipt.update(status=state, error=error)
        (out / "native_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        status(event="finished", status=state, processed=receipt["frames"]["processed"], exit_code=code)
        return code

    if mode == "refused":
        return finish(3, "refused", "refusing engine (1 mismatches): tensorrt")
    w, h, fps = s["input"]["width"], s["input"]["height"], s["input"]["fps"]
    stop_at = s["input"]["frame_range"][1]
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*s["rendering"]["codec"]), fps, (w, h))
    tracks = (out / "tracks.jsonl").open("x", encoding="utf-8")
    timings = (out / "timings.jsonl").open("x", encoding="utf-8")
    status(event="started", run_id=run_id, frames_expected=stop_at, gpu="Fake GPU 9000")
    t0 = time.perf_counter()
    f = receipt["frames"]
    index = 0
    while stop_at is None or index < stop_at:
        if stop:
            break
        ok, image = cap.read()
        if not ok:
            break
        f["decoded"] += 1
        if mode == "crash" and index == 5:
            os._exit(1)  # no receipt, files left half written
        objs = [Track(1, 0, "person", 0.9, (10.0, 10.0, 30.0, 40.0))] if index % 2 else []
        line = json.dumps(records.frame_record(run_id, index, fps, w, h, objs))
        if not (mode == "short_video" and index == 3):
            writer.write(image)
        if not (mode == "short_records" and index == 4):
            tracks.write(line + "\n")
        timings.write(json.dumps({"source_frame_index": index, **dict.fromkeys(STAGES, 0.5)}) + "\n")
        for k in ("processed", "records", "timing_rows", "video_written"):
            f[k] += 1
        index += 1
        status(event="progress", processed=f["processed"], elapsed_s=time.perf_counter() - t0)
        if mode == "slow":
            time.sleep(0.05)
        if mode == "interrupted" and index == 6:
            break
    writer.release()
    tracks.close()
    timings.close()
    receipt["timing"].update(wall_s=time.perf_counter() - t0 + 0.6, processing_wall_s=time.perf_counter() - t0)
    if mode == "missing_records":
        (out / "tracks.jsonl").unlink()
    if stop or mode == "interrupted":
        return finish(130, "interrupted", f"stopped by SIGINT after frame {f['processed']}")
    if mode == "bad_receipt":
        receipt["run_id"] = "r_someone_else"
    if mode == "status_error":
        return finish(0, "error", "CUDA error: an illegal memory access")
    return finish(0, "success")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
