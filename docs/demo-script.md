# Demo recording script (about 3 minutes)
For a narrated screen recording (spec §5, Day 2). A silent 52-second cut assembled from real runs is generated
locally at `demo-video/motionman-demo.mp4` (gitignored). Record on mains power: battery runs about 38% slower.

## Before you record
```bash
ls clips/crosswalk.mp4 yolo11n.pt          # both present (README → Sample clip; first run downloads weights)
rm -rf runs/demo                           # the CLI refuses a non-empty output folder
scripts/check.sh                           # expect: ok ×3, 200 passed
```
Use a terminal with a large font, plus a browser window at about 1280×800.

## 1. The command (45 s)
Say: *"MotionMan is a Python object-tracking tool. One recorded video in, three artifacts out, on CPU."*
```bash
.venv/bin/python -m motionman --source clips/crosswalk.mp4 --output-dir runs/demo --device cpu
```
While it runs (about 30 s on mains), say: *"YOLO11n detects, ByteTrack tracks, one synchronous loop: decode,
track, draw, write video and JSONL, measure."* It ends with a line like
`done: 1356 frames in 29.8 s (45.5 fps end to end, device cpu)`; your numbers will differ.

## 2. The artifacts (45 s)
```bash
ls runs/demo
open runs/demo/annotated.mp4               # scrub to ~0:44 for the busy crossing
head -c 400 runs/demo/tracks.jsonl; echo
python3 -m json.tool runs/demo/report.json | head -40
```
Point at:
- **The video:** boxes, IDs and trails.
- **One JSONL line per frame**, with null IDs kept as null.
- **The report:** `processed_frames` = `jsonl_records` = `output_frames_decoded`. Model load is reported
  separately, and p50/p95 skip 10 warmup frames.

## 3. Failure handling (30 s)
```bash
.venv/bin/python -m motionman --source clips/crosswalk.mp4 --output-dir runs/demo   # rerun: refused, exit 2
.venv/bin/python -m motionman --source nope.mp4 --output-dir runs/x                 # clear error, nothing created
.venv/bin/python -m motionman --source clips/crosswalk.mp4 --output-dir runs/x --device cuda   # no GPU: exit 2
```
Say: *"Bad input fails before anything is written. Results are never overwritten. Ctrl-C mid-run leaves a
report marked interrupted whose counts still agree."*

## 4. The UI (45 s)
```bash
scripts/ui.sh                              # http://localhost:8501
```
1. Demo clip, then CPU, then **Run tracking**.
2. While it runs, click the device toggle. Say: *"Inference runs on a background thread. Streamlit stops its
   script on every click, so a run in the page would die; this one keeps going."*
3. At Complete: switch to **Tracked**, then scroll to the metrics. Say: *"These numbers come from this run's
   report.json."*

## 5. Close (15 s)
Say: *"200 offline tests run in CI on every PR. The limits are written down: track IDs aren't counts, there
are no accuracy claims without labelled data, and the CUDA path is unverified here."*

Then stop the UI with Ctrl-C in its terminal.
