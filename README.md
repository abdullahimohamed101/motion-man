# MotionMan

A local Python tool that reads one recorded video, detects and tracks objects with a pretrained YOLO11n model
and ByteTrack, and writes an identified, validated run:

| File | Contents |
|---|---|
| `annotated.mp4` | The video with boxes, track IDs, class labels and motion trails. Same size and frame rate as the source; no audio. |
| `tracks.jsonl` | One v1.0 record per decoded frame: `run_id`, `source_frame_index`, `source_t_us` (integer microseconds), source size, and `objects` (observation ID, track ID or null, class, confidence, `bbox_xyxy` in source pixels). |
| `report.json` | Run status, source metadata, settings, package versions, hardware, and measured timings. |
| `runspec.json` | Every resolved setting: input hash, weights and class-map hashes, the installed ByteTrack config, device, thresholds, codec, limits, plus the git commit and dependency hash. |
| `manifest.json` | Final status, the SHA-256 and size of every file, frame counts, and validation results. |

Records written before v1.0 (`frame_index`, `timestamp_s`) are still readable through an adapter in
`motionman/records.py`; they're never rewritten.

It runs on CPU. CUDA is supported by the code but has not been verified (see Limitations). A small local web
UI runs the same pipeline (see Local UI).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg (ffmpeg only to prepare the sample clip).

```bash
uv venv -p 3.11
uv pip install -r requirements-dev.txt
```

The first real run downloads `yolo11n.pt` (5.4 MB) from Ultralytics into the working directory.

## Sample clip

The demo uses a public-domain (CC0) clip from Wikimedia Commons (see Attribution). Download it and convert
it to a constant-frame-rate H.264 MP4:

```bash
mkdir -p clips/raw
curl -L -o clips/raw/DiagonalCrosswalkYongeDundas.webm https://upload.wikimedia.org/wikipedia/commons/d/d2/DiagonalCrosswalkYongeDundas.webm
ffmpeg -i clips/raw/DiagonalCrosswalkYongeDundas.webm -map 0:v:0 -an -c:v libx264 -crf 18 -pix_fmt yuv420p -r 24000/1001 -fps_mode cfr clips/crosswalk.mp4
```

## Run

```bash
.venv/bin/python -m motionman --source clips/crosswalk.mp4 --device cpu
```

This is a **canonical run**. It takes an exclusive lock on the data folder, imports the video into
`data/inputs/<sha256>/` (once per unique file), writes into `data/staging/<run_id>/`, then validates the
decoded video, the records and the unchanged source. It hashes every file into `manifest.json` and only then
renames the folder to `data/runs/<run_id>/`. A failed or interrupted attempt keeps its diagnostics in
`staging/` and is never published.

```bash
.venv/bin/python -m motionman --source clips/crosswalk.mp4 --output-dir runs/demo --device cpu
```

With `--output-dir` it's a **legacy run**: the MVP behaviour, writing in place into a new or empty folder,
plus the same `manifest.json` and `runspec.json`. Legacy runs don't take the lock.

| Flag | Default | Notes |
|---|---|---|
| `--source` | required | Constant-frame-rate video with even width and height (see Input limits) |
| `--output-dir` | none | Legacy mode: must be new or empty; existing results are never overwritten |
| `--data-root` | `$MOTIONMAN_DATA_ROOT` or `./data` | Canonical mode data folder |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto` (falls back to CPU and logs it) |
| `--model` | `yolo11n.pt` | Ultralytics weights |
| `--imgsz` | `640` | Multiple of 32, from 32 to 4096 |
| `--max-frames` | all | Stop early, useful while iterating |

Detection thresholds are fixed and recorded in the report: confidence 0.25, IoU 0.7, FP32.

Exit codes: `0` success, `1` tool failure (unexpected error, or output validation failed), `2` invalid
arguments, input, output directory, device or model, `75` another run holds the data folder's lock (try again
later), `130` interrupted. Most input problems are caught before any output folder exists; the few that only
show up while decoding (see Input limits) fail the run inside `staging/`, where it is never published.

A run only succeeds if decoding reached every frame the container reports, the written video decodes back to
exactly the processed frames, the records are this run's, complete and in order, and the source didn't change
during the run. If the container reports no frame count, completeness is recorded as `UNVERIFIED`, not passed. The manifest says `SUCCEEDED`, `FAILED`
or `INTERRUPTED`; `report.json` keeps the MVP words (`success`, `error`, `interrupted`) and, for a partial run,
`processed_frames`, `jsonl_records` and `output_frames_decoded` show what each file holds.

### Input limits

Exceeding a limit is a clear error, never a silent truncation:

- Constant frame rate, verified from frame timestamps. A clip whose timestamps drift more than half a frame
  from `index / fps` is rejected ("unsupported timebase"), even if it reports a plausible FPS. The first 240
  frames are checked before inference; later frames are checked as they decode.
- At most 512 MiB, 10 minutes, 1920×1080 and 36,000 frames, checked from the container before inference and
  again while decoding when the container reports no count; at least 2 GiB free for outputs, checked before
  writing, and the output size checked after.
- A decoder that stops before the container's frame count (a truncated or damaged file) fails the run.

## Local UI

```bash
scripts/ui.sh
```

Opens a Streamlit workbench at http://localhost:8501, listening on localhost only. Choose **Try demo**
(`clips/crosswalk.mp4`, or set `MOTIONMAN_DEMO_CLIP`) or **Choose video**, then press **Run tracking**.
Device, detection threshold and saved tracking presets sit under **Advanced settings**, closed by default.
While it runs you see frames, percentage and elapsed time. When it finishes the result opens:

- **Saved runs** at the top reopens any verified run in the data folder. Older pre-manifest folders under
  `runs/` appear only with **Show legacy runs** and are read-only and unverified.
- **Inspect**: the Tracked or Original video (native playback, not synchronized with the frame below), and an
  inspected frame. The **Source frame** slider shows that exact frame from the source video with its own
  saved detections; arrow keys step one frame. **View options** change labels, trail length and opacity, the
  class filter and the spotlight, and only re-render: saved results never change.
- **Object** panel: pick an object to see whether it is observed in this exact frame, the time intervals it
  was observed (not just first-to-last), **First seen** / **Last seen**, confidence, gaps and a contact sheet
  sampled only from frames where it was observed. Track IDs are local to one run and can fragment.
- Each run remembers its own frame, object and view options while the page is open; an ID from one run is
  never applied to another.
- **Performance**: the metrics from that run's `report.json`.
- **Export**: the annotated video, tracks JSONL and run report, plus **Create evidence PNG**, which writes the
  inspected frame with the current view to `data/exports/<id>/evidence.png` with an `export.json` recording
  the run, records hash, source frame and options.

- UI runs are canonical runs under the same data folder and lock as the CLI. If a command-line run holds
  the lock, the page says so and asks you to wait.
- Tracking runs on a background thread, one run at a time, so clicking around the page never stops it.
  Inspecting, switching views, exporting or downloading never re-runs inference.
- A failed attempt is not offered as a result: no partial downloads. Its files stay in `data/staging/`.
- Metrics show "Pending" until a run completes and "N/A" when a value can't be computed (for example
  latency on a run of 10 frames or fewer), never a placeholder zero. A failed run shows its error and a next
  step instead of the previous run's results.
- UI runs share the CPU with the browser playing video, so they are slower than CLI runs (about 20–24 fps
  against 31–32 fps on the demo clip, both on battery). Use the CLI for benchmarks.

## Test

```bash
scripts/check.sh
```

This runs ruff and the pytest suite (200 tests, about five seconds). The tests use synthetic clips and a fake
tracker; the UI page is tested with Streamlit's AppTest. They block network access and fail if torch or
Ultralytics gets imported, so they never need a GPU, the network, or the model weights.

## Modules

| Module | Responsibility |
|---|---|
| `cli.py` | Parse and validate flags, build `RunConfig`, map failures to exit codes |
| `pipeline.py` | `run(config)`: the synchronous decode → track → trails → annotate → write loop; canonical (staged, validated, published) and legacy modes; report and manifest on every exit |
| `video_io.py` | Open and validate the reader (limits, constant-frame-rate timestamps) and writer (H.264, falling back to `mp4v`), yield frames, re-count output frames |
| `records.py` | v1.0 frame records: writer, validation, and the reader that adapts pre-v1.0 output |
| `store.py` | Content hashes, canonical JSON, run IDs, resolved RunSpec, data root, execution lock, manifest, atomic publish |
| `tracking.py` | The only module that imports Ultralytics: one model instance per run, results converted to plain `Track` records; device resolution |
| `render.py` | Bounded trails (≤30 points per ID, dropped after 60 unseen frames) and drawing on a frame copy |
| `metrics.py` | Timing, latency percentiles, environment details, atomic `report.json` write |
| `types.py` | Plain records shared by every module (`Track`, `Frame`, `VideoInfo`, `RunConfig`) |
| `ui/app.py` | Streamlit page layout and state only; no inference code |
| `ui/runner.py` | Background canonical runs: one job per process plus the data-root lock, progress, errors mapped to a next step |
| `ui/view.py` | What the page shows for a given state ("Pending", "N/A", badges, progress text) |
| `ui/style.py` | The few style rules the Streamlit theme config can't express |

## Benchmark

Three runs of the same clip and configuration in each power state (`clips/crosswalk.mp4`: 1,356 frames,
1920×1080, 23.976 fps, 56.5 s; YOLO11n, imgsz 640, FP32, CPU).

**On mains power** (AC, battery 65% and charging):

| Run | End-to-end FPS | Processing wall time | detect+track p50 | detect+track p95 | Model load |
|---|---|---|---|---|---|
| 1 | 45.02 | 30.12 s | 19.89 ms | 21.94 ms | 2.85 s |
| 2 | 44.27 | 30.63 s | 20.27 ms | 22.38 ms | 2.93 s |
| 3 | 44.29 | 30.62 s | 20.18 ms | 22.12 ms | 2.99 s |
| **Median** | **44.29** | | | | |

**On battery** (27–35%):

| Run | End-to-end FPS | Processing wall time | detect+track p50 | detect+track p95 | Model load |
|---|---|---|---|---|---|
| 1 | 31.27 | 43.36 s | 27.26 ms | 35.65 ms | 3.88 s |
| 2 | 32.15 | 42.18 s | 26.88 ms | 31.73 ms | 4.07 s |
| 3 | 32.14 | 42.19 s | 27.14 ms | 30.40 ms | 4.10 s |
| **Median** | **32.14** | | | | |

Hardware: Apple M5 Pro (15 CPU cores, 24 GB RAM), macOS 26.6, Python 3.11.17, torch 2.15.0,
ultralytics 8.4.196, opencv-python 5.0.0.93. Every run exited 0 with 1,356 frames processed, 1,356 JSONL
records and 1,356 frames decoded from the output.

**Power state is the biggest variable.** The same code on the same clip runs about 38% faster on mains power,
and detect+track p50 drops from about 27 ms to about 20 ms; macOS throttles the CPU on battery. An earlier set
on battery, before the model warmup moved into model load, measured a median of 25.95 fps. Treat all of these
as one laptop's numbers, not a performance claim, and compare runs only within the same power state.

What the numbers mean:
- **End-to-end FPS** is processed frames divided by processing wall time, measured from before the first
  decode until the output writer is released. It includes decoding, detection, tracking, drawing and
  H.264 encoding at 1080p. It is not a real-time claim and not an inference-only number.
- **detect+track p50/p95** time the whole `model.track` call (YOLO detection plus ByteTrack association) plus
  conversion to records, excluding the first 10 frames as warmup. Those frames still appear in every output.
- **Model load** is reported separately and is not part of the wall time. Ultralytics loads lazily, so model load
  includes one prediction on a blank frame; without it, about 2.5 s of setup would land in the first frame of the
  wall time and make short runs look slow. The warmup runs in predict mode, so tracker state is untouched.

Where the time goes: on mains power a frame takes about 22.6 ms end to end, and the median detect+track call is
about 20 ms (about 31 ms and 27 ms on battery). So the model and tracker are most of the cost; decoding, drawing and encoding 1080p frames
make up most of the rest. A 48-frame run now measures 23 fps end to end (it was 11 fps before the warmup fix),
because a short run is still dominated by the slower first few frames.

## Limitations

- **Track IDs are not people counts.** IDs are local to one run and fragment under occlusion and in crowds. On
  the demo clip, 701 distinct IDs appear and the median ID lasts 11 frames, while static objects (traffic lights)
  keep one ID for the whole clip. ByteTrack's ID counter also skips numbers, so labels reach the thousands.
- **No accuracy claims.** The output was inspected visually; there is no labelled evaluation.
- **Crowded scenes are visually busy.** Labels overlap when many objects are close together.
- **CUDA is unverified.** The code path (explicit device check, CUDA synchronisation around timings) exists
  but has not been run: the development machine has no NVIDIA GPU. Apple's MPS backend is not supported.
- **Input limits.** Constant-frame-rate video with valid FPS and even dimensions only. Audio is dropped.
- **Single-threaded.** The optional decoder thread from the design was not built.
- **UI:** one run at a time per process. A run keeps going if the browser tab is closed. Original-view preview
  depends on the uploaded file's codec; the Tracked view is always H.264.
- If OpenCV on another machine lacks H.264, output falls back to `mp4v` with a warning; that plays in QuickTime
  but usually not in browsers.

## Attribution

Sample clip: [DiagonalCrosswalkYongeDundas.webm](https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm),
"Diagonal crosswalk at Yonge & Dundas in Toronto", by Raysonho @ Open Grid Scheduler / Grid Engine (own work, 2011),
released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Converted to H.264 MP4 without audio
for this project.

Detection model: [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/) with the
[ByteTrack](https://docs.ultralytics.com/modes/track/) tracker configuration shipped with Ultralytics.
