# MotionMan spec
Condensed from `docs/reference/design-brief.txt` (§1–§8) and `docs/reference/ui-reference.txt` (§9).
The reference files win on any disagreement. Grep for the `§N` you need; do not read whole.

## §1 Product
Local CLI: one recorded video in → detect + track objects → annotated MP4 (boxes, IDs, trails)
plus per-frame JSONL plus measured report JSON. Small enough to explain every module in an
interview. CPU must work; CUDA optional. Demo input: a 30–60 s clip of a few moving people or
vehicles, self-recorded or licensed for reuse.

## §2 Out of scope
Web UI beyond §9, API server, database, cloud, live camera/RTSP, multiple streams, training,
re-identification, custom tracker, C++, custom CUDA kernels, TensorRT, RT-DETR. No real-time FPS
requirement. Do not grow this into ServerFlow or Agent Black Box.

## §3 Stack
Python 3.11, OpenCV, Ultralytics `yolo11n.pt` + `bytetrack.yaml`, installed PyTorch backend, pytest.
Pin versions that pass the first smoke test. Config = argparse flags + one small dataclass.

## §4 Pipeline and modules
Synchronous loop: decode frame → detect/track → update trails → annotate → write video + JSONL →
update metrics. One model instance owns tracker state for the whole run, fed consecutive frames
via `model.track(frame, persist=True, tracker="bytetrack.yaml", ...)`. Never recreate the model
per frame, share tracker state across videos, or parallelise tracking calls.

| Module | Responsibility |
|---|---|
| `cli.py` | parse/validate flags, build `RunConfig`, return meaningful exit code |
| `video_io.py` | open/validate reader and writer, yield frames, release resources |
| `tracking.py` | wrap the model; convert results to plain `Track` records |
| `render.py` | bounded trails; draw annotations on a frame copy |
| `metrics.py` | accumulate timings/counts; write the report |
| `pipeline.py` | `run(config)`: the loop and cleanup |

Plus `__init__.py`, `__main__.py`, `tests/`, `README.md`, pinned `requirements.txt`.

## §5 CLI contract
`python -m motionman --source demo.mp4 --output-dir runs/demo --device cpu`
- Required: `--source`, `--output-dir`. Optional: `--device cpu|cuda|auto` (default cpu),
  `--model` (default `yolo11n.pt`), `--imgsz` (default 640), `--max-frames`.
- Outputs: `annotated.mp4`, `tracks.jsonl`, `report.json`.
- Output dir must be new or empty. Never overwrite the source or existing results.

## §6 Data contracts
- Frame: zero-based `index`, `timestamp_s = index / source_fps`, BGR image.
- Supported input: constant frame rate, valid FPS, even dimensions. Reject anything else clearly.
- Output keeps original dimensions and playback FPS, regardless of processing speed. No audio.
  Annotation coordinates are in original-frame pixel space.
- `update(frame) -> list[Track]`. `Track`: `track_id` (int or null), `class_id`, `class_name`,
  `confidence`, `bbox_xyxy` (4 pixel coords). Empty detections and missing `boxes.id` are normal.
  **Never invent IDs.** An object without an ID is drawn but gets no trail.
- JSONL: exactly one record per successfully processed frame: `frame_index`, `timestamp_s`,
  `objects` (list of Track, possibly empty). Stream to disk; never collect in memory.
  IDs are run-local, may switch/fragment, and are not unique-object counts.
- Trails: per valid ID, a deque of ≤30 centre points; drop the ID after 60 frames unseen.
  Retain no decoded frames, model result objects, or unbounded history.

## §7 Concurrency (optional, only if time remains)
Start with no threads. Optional decoder thread: `Queue(maxsize=2)`, blocking backpressure, no frame
drops; tracking/render/write stay in order on the main thread. Must propagate decoder errors and use
a stop event plus bounded waits so cancellation cannot deadlock.

## §8 Failure, cleanup, benchmark
Fail clearly on: missing input, unopenable video, invalid FPS, no decodable frames, explicit `cuda`
unavailable, writer cannot open. `auto` may fall back to CPU and must log it.
Release capture/writer and close JSONL in `finally`, including Ctrl-C. Partial reports have status
`interrupted` or `error`, never success. After success, reopen the output video and check its
decoded frame count equals `processed_frames`.

`report.json`: status, source metadata, processed_frames, model/weights id, imgsz, device, package
versions, machine/GPU description, precision (FP32), explicit conf/IoU thresholds.
- `processing_wall_s`: `perf_counter` from before first decode to after final writer release.
- `end_to_end_fps = processed_frames / processing_wall_s` (includes decode, track, draw, write).
- Model load time reported separately.
- `detect_track_ms` p50/p95 over the `model.track` call, excluding the first 10 frames (warmup;
  they still appear in outputs). ≤10 frames → percentiles null with a reason.
- On CUDA, synchronise around timings. Do not call wrapper timing "inference-only".
Benchmark: same clip + config 3×, report all three and the median, with hardware. No claims of
FPS targets, stream counts, or accuracy (accuracy needs labelled eval; MVP is visual inspection only).

## §9 Local UI (after the CLI is done; budget 2–3 h)
Thin local UI over the same `pipeline.run`; inference stays out of UI code; fresh tracker per run;
completed artifact paths kept in session state so interactions never re-run inference. Prefer
Streamlit theme settings and standard components; no React/Tailwind/API; no DOM hacking.
- Layout: header (wordmark, "Object Tracking Workbench", actual device, Idle/Processing/Complete/Error),
  ~280 px controls sidebar, video workspace; stack on narrow screens.
- Sidebar: upload MP4 or demo clip; source name/dims/duration/playback FPS; CPU/Auto/CUDA;
  confidence; one **Run tracking** button. Model fixed to YOLO11n, shown as metadata.
- Workspace: Original/Tracked toggle (no re-run), native playback, aspect ratio kept.
- Metrics strip (from current report only): processing FPS, detect+track p95, frames, duration.
  Before completion show "Pending"; unavailable latency is "N/A", never 0. Playback FPS labelled
  separately from processing FPS.
- Exports: neutral downloads for the 3 artifacts; collapsed Run details (model, imgsz, device, versions).
- States: processing shows frames + elapsed (percent only with a known total), duplicate runs disabled;
  errors give a next step; a failed run never shows the previous Complete badge or metrics.
- Style: canvas `#040506`, panel `#07080A`, inset `#111214`, raised `#1B1C1E`, border `#363739`,
  text `#F5F5F5`/`#A1A1AA`/`#8B8C91`, accent Motion Lime `#A3E635` (hover `#B5EF59`, pressed
  `#8CC927`, tint `#17210D`, text-on-lime `#0B1105`), warning `#FBBF24`, error `#F87171`.
  Lime only on Run tracking, focus (2 px outline, 2 px offset), selection, progress. Never state by
  colour alone. Inter / Geist Mono (mono for IDs, durations, numbers). Radii 8/6/16 px; spacing 8 px base.
- Track overlays: colour from track ID via a small palette (lime, cyan, amber, violet); trail matches
  box; ID + class label on a readable backing. No NVIDIA/Raycast logos or implied affiliation.

## §10 Done
Fresh env installs, `pytest` passes (no GPU, network, or weight download), documented command runs.
One clip yields a playable `annotated.mp4`, a JSONL record per processed frame, an honest
`report.json`. README: setup/run/test, module roles, real benchmark numbers + hardware, limitations,
sample-input attribution. Weights, clips, and `runs/` are gitignored. Describe it as a Python
object-tracking tool, not a C++/TensorRT or silicon-validation platform.
