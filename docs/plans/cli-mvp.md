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
- D4 Writer codec `mp4v` via OpenCV; step 2 confirms the output plays in QuickTime and a browser.
  If it doesn't, re-encode with ffmpeg (ADR).
- D5 Thresholds explicit and recorded: conf 0.25, IoU 0.7 (Ultralytics defaults), FP32.
- D6 Demo clip: owner supplies a 30–60 s people/vehicles clip with reuse rights (needed by step 2).

## Steps
1. [x] Env + smoke: uv venv 3.11, install ultralytics/opencv/pytest, one-frame detection on CPU.
   Pin `requirements.txt`. Confirm the installed `model.track` result API (`boxes.id/xyxy/cls/conf`).
2. [ ] Vertical slice: `types`, `video_io`, `tracking`, `pipeline`, `__main__` with boxes + IDs; playable output.
3. [ ] Trails + overlays (`render.py`): bounded deque, 60-frame expiry, colour from ID, label backing.
4. [ ] JSONL streaming + `metrics.py` + `report.json` (wall time, e2e FPS, p50/p95 after warmup, versions).
5. [ ] `cli.py` flags + validation + clear failures and exit codes; `finally` cleanup; interrupted/error status;
   post-run frame-count check.
6. [ ] Tests: record conversion, missing IDs, empty frames, trail bounds/expiry, timing math, output frame
   count, invalid input, cleanup on failure. Synthetic clip + fake tracker. `scripts/check.sh`.
7. [ ] Real-model demo + visual review; 3 benchmark runs; README with real numbers and hardware.

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| A1 | Offline tests pass | `scripts/check.sh` | |
| A2 | Demo run exits 0, 3 artifacts | `python -m motionman --source <clip> --output-dir runs/a2 --device cpu; ls runs/a2` | |
| A3 | JSONL lines == processed_frames == decoded output frames | `wc -l runs/a2/tracks.jsonl`; report fields | |
| A4 | Output keeps source dims + FPS | `ffprobe` source vs `annotated.mp4` | |
| A5 | Non-empty output dir rejected, source untouched | run twice into `runs/a2` → non-zero exit | |
| A6 | Missing / bad input → clear message, non-zero exit | `--source nope.mp4`; `--source README.md` | |
| A7 | `--device cuda` on this Mac → clear error | run with `--device cuda` | |
| A8 | Ctrl-C mid-run → report status `interrupted`, files closed | send SIGINT, read report | |
| A9 | 3 benchmark runs + median recorded | README table | |
| A10 | Fresh env install + test from README | new venv, follow README | |

## Risks
- OpenCV `mp4v` may not play in browsers (matters for §9 UI) → D4.
- Ultralytics may download weights on first run → done in step 1 only; tests never load the model.
- CPU throughput on long clips is slow → use `--max-frames` while iterating.

## Log
- 2026-10-31 step 1: py3.11.17 venv; bus.jpg CPU: 5 dets, track IDs 1–5 stable over 2 frames, empty frame → `boxes.id` None; `boxes.xyxy/cls/conf` are tensors. Pinned. Weights download prints a long progress bar: use `verbose=False` and pre-fetched weights.
