# CLI MVP: video in, three artifacts out
Status: Active · Branch: `feat/cli-mvp` · Spec: §1–§8, §10

## Outcome
`python -m motionman --source clip.mp4 --output-dir runs/demo --device cpu` produces a playable
`annotated.mp4`, a `tracks.jsonl` with one record per frame, and an honestly measured `report.json`;
`pytest` passes offline with a fake tracker.

## Non-goals
UI (separate plan, spec §9), decoder thread (§7, only if time remains), CUDA verification (no GPU
on this Mac), anything in §2.

## Decisions (confirmed 2026-10-31)
- D1 Python 3.11 via `uv` (host has 3.9/3.12 only); `requirements.txt` pinned after step 1 passes.
- D2 Dev-only lint: `ruff` in `requirements-dev.txt`; pytest stays the only test tool.
- D3 Devices stay `cpu|cuda|auto` per §5. On this Mac `auto` → CPU (logged). Apple MPS is not added.
- D4 Writer codec `avc1` (H.264, yuv420p: browser + QuickTime), falling back to `mp4v` with a warning.
  Amended in step 2: this OpenCV build writes H.264, so no ffmpeg re-encode is needed.
- D5 Thresholds explicit and recorded: conf 0.25, IoU 0.7 (Ultralytics defaults), FP32.
- D6 Demo clip: owner supplies a 30–60 s people/vehicles clip with reuse rights (needed by step 2).

## Steps
1. [x] Env + smoke: uv venv 3.11, install ultralytics/opencv/pytest, one-frame detection on CPU.
   Pin `requirements.txt`. Confirm the installed `model.track` result API (`boxes.id/xyxy/cls/conf`).
2. [x] Vertical slice: `types`, `video_io`, `tracking`, `pipeline`, `__main__` with boxes + IDs; playable output.
3. [x] Trails + overlays (`render.py`): bounded deque, 60-frame expiry, colour from ID, label backing.
4. [x] JSONL streaming + `metrics.py` + `report.json` (wall time, e2e FPS, p50/p95 after warmup, versions).
5. [x] `cli.py` flags + validation + clear failures and exit codes; `finally` cleanup; interrupted/error status;
   post-run frame-count check.
6. [x] Tests: record conversion, missing IDs, empty frames, trail bounds/expiry, timing math, output frame
   count, invalid input, cleanup on failure. Synthetic clip + fake tracker. `scripts/check.sh`.
7. [x] Real-model demo + visual review; 3 benchmark runs; README with real numbers and hardware.

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| A1 | Offline tests pass | `scripts/check.sh` | PASS: ruff ok, 71 passed in 0.45 s; conftest blocks network and fails if torch/ultralytics load |
| A2 | Demo run exits 0, 3 artifacts | `python -m motionman --source clips/crosswalk.mp4 --output-dir runs/demo --device cpu` | PASS: exit 0, 3 artifacts, boxes/IDs/trails visually checked |
| A3 | JSONL lines == processed_frames == decoded output frames | `wc -l runs/demo/tracks.jsonl`; report fields | PASS: 1356 = 1356 = 1356 |
| A4 | Output keeps source dims + FPS | `ffprobe` source vs `annotated.mp4` | PASS: 1920x1080 both; 24000/1001 vs 2997/125 (both 23.976) |
| A5 | Non-empty output dir rejected, source untouched | run twice into `runs/a2` → non-zero exit | PASS (pan clip): exit 2, `output directory must be new or empty` |
| A6 | Missing / bad input → clear message, non-zero exit | `--source nope.mp4`; `--source AGENTS.md` | PASS: exit 2 each (not found / cannot open / odd 639x479); no dir created |
| A7 | `--device cuda` on this Mac → clear error | run with `--device cuda` | PASS: exit 2, `CUDA is not available ... use --device cpu` |
| A8 | Ctrl-C mid-run → report status `interrupted`, files closed | send SIGINT, read report | PASS: exit 130, one-line message; 188 = jsonl = decoded |
| A9 | 3 benchmark runs + median recorded | README table | PASS: 25.95 / 25.19 / 26.01 fps, median 25.95 (M5 Pro CPU, on battery) |
| A10 | Fresh env install + test from README | new venv, follow README | PASS: clean clone, uv install, check.sh 71 passed, 48-frame run exit 0 |

## Risks
- Another machine's OpenCV may lack H.264 → `mp4v` fallback logs a warning; browser playback then needs a re-encode.
- Ultralytics may download weights on first run → done in step 1 only; tests never load the model.
- CPU throughput on long clips is slow → use `--max-frames` while iterating.

## Log
- 2026-10-31 step 1: py3.11.17 venv; bus.jpg CPU: 5 dets, track IDs 1–5 stable over 2 frames, empty frame → `boxes.id` None; `boxes.xyxy/cls/conf` are tensors. Pinned. Weights download prints a long progress bar: use `verbose=False` and pre-fetched weights.
- 2026-10-31 step 2: package + slice on a 90-frame 640x480 pan of bus.jpg: exit 0, 90 frames written = 90 decoded, h264 30fps, boxes/IDs/labels correct. Missing/not-video source and non-empty dir → exit 2. Real-clip check waits on D6. Color-by-ID pulled in from step 3.
- 2026-10-31 step 3: `Trails` (≤30 pts, dropped at 60 unseen frames, none for null IDs); drawn only for objects visible this frame. tests/test_render.py 9 passed. Pan clip: 90/90 frames, trails visible in box colours.
- 2026-10-31 step 4: JSONL streamed, metrics.py, report.json (atomic write, written on every exit once the dir is ours). `half=False` passed so FP32 is true; CUDA sync in Tracker.update (UNVERIFIED (env)). 26 tests pass, torch not imported. Pan clip: 90 = jsonl = decoded, status success. SIGINT mid-loop (360-frame clip): interrupted, 126 = jsonl = decoded, video playable. Finding for step 7: the first model.track call does ~1–2 s of lazy Ultralytics imports inside the wall time; SIGINT then leaves a 0-frame mp4.
- 2026-10-31 step 5: MotionManError base; resolve_device (cpu never imports torch; auto→cpu logged; cuda w/o GPU → exit 2); ModelError on bad weights; --imgsz multiple of 32, --max-frames ≥1; exit codes 0/1/2/130; post-run decode check → `output_frames_decoded`. Ultralytics 8.4.196 deprecated `half` (warned per frame) → `quantize=32`. 42 tests pass. A5–A8 PASS.
- 2026-10-31 step 6: test_video_io (13) + test_pipeline (16, synthetic clip + scripted FakeTracker: artifacts agree, empty frames, null IDs, max-frames, null latency, fresh tracker, refused dir untouched, mid-run error/interrupt cleanup, no frames, count mismatch, writer failure releases reader). scripts/check.sh. INV-8 enforced in conftest (probe: exit 1). Fixed a flaky fps assertion (ms rounding on ~4 ms runs). 71 tests total.
- 2026-10-31 step 7: D6 resolved with CC0 Wikimedia clip (Yonge & Dundas crosswalk), converted to H.264 CFR 23.976, 1356 frames. Demo + 3 benchmarks (median 25.95 fps e2e, detect+track p50 ~31 ms). 701 distinct IDs, median life 11 frames (fragmentation documented). README written. A1–A10 all PASS.
