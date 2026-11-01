# MotionMan

A local Python command-line tool that reads one recorded video, detects and tracks objects with a
pretrained YOLO11n model and ByteTrack, and writes three artifacts:

| File | Contents |
|---|---|
| `annotated.mp4` | The video with boxes, track IDs, class labels and motion trails. Same size and frame rate as the source; no audio. |
| `tracks.jsonl` | One JSON record per processed frame: `frame_index`, `timestamp_s`, `objects` (track ID or null, class, confidence, `bbox_xyxy` in source pixels). |
| `report.json` | Run status, source metadata, settings, package versions, hardware, and measured timings. |

It runs on CPU. CUDA is supported by the code but has not been verified (see Limitations).

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
.venv/bin/python -m motionman --source clips/crosswalk.mp4 --output-dir runs/demo --device cpu
```

| Flag | Default | Notes |
|---|---|---|
| `--source` | required | Constant-frame-rate video with even width and height |
| `--output-dir` | required | Must be new or empty; existing results are never overwritten |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto` (falls back to CPU and logs it) |
| `--model` | `yolo11n.pt` | Ultralytics weights |
| `--imgsz` | `640` | Multiple of 32, from 32 to 4096 |
| `--max-frames` | all | Stop early, useful while iterating |

Detection thresholds are fixed and recorded in the report: confidence 0.25, IoU 0.7, FP32.

Exit codes: `0` success, `1` tool failure (unexpected error, or the output check below failed), `2` invalid
arguments, input, output directory, device or model, `130` interrupted. Those input problems are caught before
the output directory is created, so you can fix the flag and rerun with the same `--output-dir`.

If a run fails or is interrupted after it starts writing, the partial artifacts are kept and `report.json`
says `error` or `interrupted`, with `processed_frames`, `jsonl_records` and `output_frames_decoded` showing what
each file actually holds. A run only reports `success` if the written video decodes back to exactly
`processed_frames` frames.

## Test

```bash
scripts/check.sh
```

This runs ruff and the pytest suite (78 tests, about a second). The tests use synthetic clips and a fake
tracker. They block network access and fail if torch or Ultralytics gets imported, so they never need
a GPU, the network, or the model weights.

## Modules

| Module | Responsibility |
|---|---|
| `cli.py` | Parse and validate flags, build `RunConfig`, map failures to exit codes |
| `pipeline.py` | `run(config)`: the synchronous decode → track → trails → annotate → write loop, cleanup in `finally`, the report on every exit |
| `video_io.py` | Open and validate the reader and writer (H.264, falling back to `mp4v`), yield frames, re-count output frames |
| `tracking.py` | The only module that imports Ultralytics: one model instance per run, results converted to plain `Track` records; device resolution |
| `render.py` | Bounded trails (≤30 points per ID, dropped after 60 unseen frames) and drawing on a frame copy |
| `metrics.py` | Timing, latency percentiles, environment details, atomic `report.json` write |
| `types.py` | Plain records shared by every module (`Track`, `Frame`, `VideoInfo`, `RunConfig`) |

## Benchmark

Three runs of the same clip and configuration (`clips/crosswalk.mp4`: 1,356 frames, 1920×1080, 23.976 fps,
56.5 s; YOLO11n, imgsz 640, FP32, CPU):

| Run | End-to-end FPS | Processing wall time | detect+track p50 | detect+track p95 | Model load |
|---|---|---|---|---|---|
| 1 | 31.27 | 43.36 s | 27.26 ms | 35.65 ms | 3.88 s |
| 2 | 32.15 | 42.18 s | 26.88 ms | 31.73 ms | 4.07 s |
| 3 | 32.14 | 42.19 s | 27.14 ms | 30.40 ms | 4.10 s |
| **Median** | **32.14** | | | | |

Hardware: Apple M5 Pro (15 CPU cores, 24 GB RAM), macOS 26.6, Python 3.11.17, torch 2.15.0,
ultralytics 8.4.196, opencv-python 5.0.0.93. The laptop was on battery (27–35%) during these runs; results on
mains power may differ. Each run exited 0 with 1,356 frames processed, 1,356 JSONL records and 1,356 frames
decoded from the output.

**Run-to-run variance is real.** An earlier set of three runs on the same machine, before the model warmup
moved into model load, measured 25.95 / 25.19 / 26.01 fps (median 25.95) with a detect+track p50 near 31 ms.
The warmup explains only about 2.5 s of the 10 s difference in wall time; the rest is not isolated (power
state is the likely cause). Treat these as one laptop's numbers on one afternoon, not a performance claim.

What the numbers mean:
- **End-to-end FPS** is processed frames divided by processing wall time, measured from before the first
  decode until the output writer is released. It includes decoding, detection, tracking, drawing and
  H.264 encoding at 1080p. It is not a real-time claim and not an inference-only number.
- **detect+track p50/p95** time the whole `model.track` call (YOLO detection plus ByteTrack association) plus
  conversion to records, excluding the first 10 frames as warmup. Those frames still appear in every output.
- **Model load** is reported separately and is not part of the wall time. Ultralytics loads lazily, so model load
  includes one prediction on a blank frame; without it, about 2.5 s of setup would land in the first frame of the
  wall time and make short runs look slow. The warmup runs in predict mode, so tracker state is untouched.

Where the time goes: at the median, a frame takes about 31 ms end to end, and the median detect+track call is
about 27 ms. So the model and tracker are most of the cost; decoding, drawing and encoding 1080p frames
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
- If OpenCV on another machine lacks H.264, output falls back to `mp4v` with a warning; that plays in QuickTime
  but usually not in browsers.

## Attribution

Sample clip: [DiagonalCrosswalkYongeDundas.webm](https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm),
"Diagonal crosswalk at Yonge & Dundas in Toronto", by Raysonho @ Open Grid Scheduler / Grid Engine (own work, 2011),
released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Converted to H.264 MP4 without audio
for this project.

Detection model: [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/) with the
[ByteTrack](https://docs.ultralytics.com/modes/track/) tracker configuration shipped with Ultralytics.
