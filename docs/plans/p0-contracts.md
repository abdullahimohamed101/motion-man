# P0: verify the baseline and formalize contracts
Status: Active · Branch: `feat/p0-contracts` · Spec: A§1–A§6, A§10 · Design: MD§20 P0, MD§25 first work package

## Outcome
The working MVP keeps working, and every new run becomes a reproducible, identified, validated artifact:
hashed input, resolved config, v1.0 frame records, a manifest with artifact hashes, published atomically
under a data root. P1 (inspect) and P2 (measure) build on these contracts.

## First report (MD§25)
**1. What exists and is verified (VERIFIED: commands and outputs recorded in `docs/plans/done/`):**
- CLI pipeline: ordered frames, one tracker per run, null IDs kept, bounded trails, H.264 output, three artifacts.
- Failure handling: bad input / device / model / output dir fail before writing (exit 2); Ctrl-C →
  `interrupted` with agreeing partial counts; success only if output decodes to `processed_frames`.
- Timing: wall time, end-to-end FPS, detect+track p50/p95 after 10 warmup frames, model load incl. warmup.
- Local Streamlit UI on a background worker; 132 offline tests; CI on every PR. Baseline real-clip runs:
  `runs/bench-ac1..3` (44.29 fps median, mains) and `runs/demo-recording`, all reopened and decoded.

**2. Present but not to Release A contract:**
- Frame records use `frame_index` / `timestamp_s` (s, float), no schema_version, run_id, observation_id,
  frame_status, source dims.
- `report.json` has counts and settings but no hashes, no resolved tracker config, no code revision.
- Only the UI has a job lock (in-process); the CLI has none. Outputs are written in place, not staged.
- Input validation checks FPS > 0 and even dims, but not that frames are actually constant-rate.

**3. Missing (acceptance criteria not met):** input_id / run_id / config_sha256; data root with
inputs/staging/runs; atomic publication; manifest with artifact hashes and validation; CFR verification;
resource limits (512 MiB, 10 min, 1920×1080, 36,000 frames, 2 GiB); a cross-process execution lease.

**4. Smallest proposed changes:** new `motionman/records.py` (v1.0 writer + legacy reader) and
`motionman/store.py` (hashing, canonical JSON, data root, staging, manifest, publish, lease); extend
`video_io.open_reader` (CFR + limits) and `pipeline.run` (resolved RunSpec, staging/publish); point the UI
runner at the data root. No new packages beyond these two modules; no rename of existing files.

**5. Validation commands:** listed in Acceptance below.

## Decisions (confirmed 2026-11-01)
- D1 Data root `./data` (gitignored), overridable with `--data-root` or `MOTIONMAN_DATA_ROOT`.
- D2 One command, two modes: `--output-dir` given → legacy mode (same meaning and filenames as today, plus
  `manifest.json`); omitted → canonical mode: generated run_id, staged under the data root, published to
  `data/runs/<run_id>/`. No subcommand renames.
- D3 Both modes write v1.0 records (one writer). Pre-v1.0 outputs on disk stay readable through the legacy
  reader; they are never rewritten or given invented hashes.
- D4 CFR check with OpenCV only (no ffprobe dependency): compare each frame's container timestamp with
  `i / fps`; preflight samples the first 240 frames and the run checks every frame. Drift > half a frame →
  reject with "This video uses an unsupported timebase…".
- D5 Canonical mode copies the source into `data/inputs/<input_id>/` once (deduplicated by hash); legacy
  mode hashes the source in place and re-hashes after the run to detect modification.
- D6 Lease: `fcntl.flock` on `data/.execution.lock`, non-blocking; a second run fails clearly. Legacy mode has
  no data root, so no lease (documented). Replaces the UI's in-process lock.
- D7 run_id = `r_<UTC yyyymmddThhmmssZ>_<6 hex>`; `config_sha256` over canonical resolved RunSpec JSON.
- D8 Limits from A§6 enforced in preflight (duration, dims, frames, free disk ≥ budget); output size checked
  after writing; exceeding either is a FAILED run with a message.

## Steps
1. [x] `records.py`: v1.0 frame record writer, validation (finite, positive-extent, in-bounds boxes),
   legacy reader adapter. Tests incl. reading `runs/demo-recording`-shaped legacy lines.
2. [x] `store.py`: streamed SHA-256, canonical JSON (sorted, no NaN), run_id, resolved RunSpec
   (weights hash, ByteTrack YAML contents + hash, code revision, requirements hash, env). Tests.
3. [ ] `video_io`: CFR timestamp check + limits in preflight and per frame. Tests with synthetic CFR and
   a deliberately VFR clip.
4. [ ] Pipeline canonical mode: staging dir → write → validate (order, counts, schema, decode) → hash →
   `manifest.json` (A§3) → atomic rename to `runs/`; legacy mode writes the same manifest in place. Lease.
5. [ ] UI runner uses canonical mode and the lease; UI behaviour otherwise unchanged (UI2 redesign is P1).
6. [ ] Failure tests (MD§25 P0.4): invalid input, empty detections, writer failure, interruption, lease
   held, limit exceeded, source modified mid-run; none produce a SUCCEEDED manifest or a published run.
7. [ ] Baseline (P0.2): real CPU clip in canonical mode; reopen artifacts; verify manifest hashes from disk.
8. [ ] Gate decision (P0.5): pass/fail per criterion, revised P1/P2 estimate, known limitations.

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| P0-1 | Offline gate passes, existing tests unchanged in intent | `scripts/check.sh` | |
| P0-2 | Canonical run publishes `data/runs/<run_id>/` with manifest; staging empty afterwards | `python -m motionman --source clips/crosswalk.mp4 --device cpu`; `ls data/runs data/staging` | |
| P0-3 | Manifest hashes match the files on disk; counts decoded = inferred = written | `shasum -a 256` vs manifest | |
| P0-4 | Records validate as v1.0; legacy `runs/demo-recording` still reads via adapter | test + script | |
| P0-5 | Legacy `--output-dir` mode unchanged for users, plus `manifest.json` | old command | |
| P0-6 | VFR clip and over-limit clip rejected before inference, nothing published | synthetic clips | |
| P0-7 | Second concurrent run refused by the lease; failed/interrupted runs never in `runs/` | two processes; SIGINT | |
| P0-8 | UI run lands in `data/runs/` and shows as before | browser | |

## Risks
- OpenCV timestamps on some containers are coarse or missing → D4 tolerance; if unavailable, reject with
  a clear message rather than assume CFR.
- Copying large inputs costs disk → dedup by hash; limits cap size.
- Changing record field names breaks external readers → only our own tools read them; adapter for old runs.

## Estimate
MD says 0.5–1 focused day. Because records, identity and publication contracts are all new, expect
1–1.5 focused days. Re-estimate P1 (1–2) and P2 (1.5–3) at the P0 gate.

## Log
- 2026-11-01 step 1: records.py (v1.0 writer with observation_ids and integer source_t_us, box policy: clip ≤0.5 px overrun, reject the rest; validate(); read_records() adapting pre-v1.0 lines with legacy=True and unknown dims/run_id left None). 20 tests. Real legacy file runs/demo-recording/tracks.jsonl: 1,356 records contiguous, t_us = formula exactly (0 µs diff), 25,361 objects. Pipeline still writes the old format until step 4. 152 tests.
- 2026-11-01 step 2: store.py (streamed sha256, canonical JSON rejecting NaN, run_id r_<UTC>_<6hex>, ByteTrack YAML read from package files without importing Ultralytics, git commit + dirty, resolve_runspec: config_sha256 over settings; reproducibility recorded but not hashed; missing weights null + reason). 11 tests. Real hashes match shasum (yolo11n.pt, crosswalk.mp4 81 MB in 0.03 s). 163 tests.
