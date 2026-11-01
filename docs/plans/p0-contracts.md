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
  no data root, so no lease (documented). Amended in step 5: the UI keeps its in-process lock too, so a double click
  is refused immediately; the lease covers other processes.
- D7 run_id = `r_<UTC yyyymmddThhmmssZ>_<6 hex>`; `config_sha256` over canonical resolved RunSpec JSON.
- D8 Limits from A§6 enforced in preflight (duration, dims, frames, free disk ≥ budget); output size checked
  after writing; exceeding either is a FAILED run with a message.

## Steps
1. [x] `records.py`: v1.0 frame record writer, validation (finite, positive-extent, in-bounds boxes),
   legacy reader adapter. Tests incl. reading `runs/demo-recording`-shaped legacy lines.
2. [x] `store.py`: streamed SHA-256, canonical JSON (sorted, no NaN), run_id, resolved RunSpec
   (weights hash, ByteTrack YAML contents + hash, code revision, requirements hash, env). Tests.
3. [x] `video_io`: CFR timestamp check + limits in preflight and per frame. Tests with synthetic CFR and
   a deliberately VFR clip.
4. [x] Pipeline canonical mode: staging dir → write → validate (order, counts, schema, decode) → hash →
   `manifest.json` (A§3) → atomic rename to `runs/`; legacy mode writes the same manifest in place. Lease.
5. [x] UI runner uses canonical mode and the lease; UI behaviour otherwise unchanged (UI2 redesign is P1).
6. [x] Failure tests (MD§25 P0.4): invalid input, empty detections, writer failure, interruption, lease
   held, limit exceeded, source modified mid-run; none produce a SUCCEEDED manifest or a published run.
7. [x] Baseline (P0.2): real CPU clip in canonical mode; reopen artifacts; verify manifest hashes from disk.
8. [x] Gate decision (P0.5): pass/fail per criterion, revised P1/P2 estimate, known limitations.

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| P0-1 | Offline gate passes, existing tests unchanged in intent | `scripts/check.sh` | PASS: ruff ok, 200 passed |
| P0-2 | Canonical run publishes `data/runs/<run_id>/` with manifest; staging empty afterwards | `python -m motionman --source clips/crosswalk.mp4 --device cpu`; `ls data/runs data/staging` | PASS: r_20261101T060226Z_90470c, 1356 frames, 45.19 fps, staging empty |
| P0-3 | Manifest hashes match the files on disk; counts decoded = inferred = written | `shasum -a 256` vs manifest | PASS: 4/4 artifacts hash+size = shasum; input_id = shasum(source); 1356 = 1356 = 1356 |
| P0-4 | Records validate as v1.0; legacy `runs/demo-recording` still reads via adapter | test + script | PASS: published run 1356 v1.0 records, one run_id; legacy file 1356 records via adapter, unchanged; both 25,361 observations |
| P0-5 | Legacy `--output-dir` mode unchanged for users, plus `manifest.json` | old command | PASS: same files + manifest/runspec, SUCCEEDED, hashes = shasum, data/ untouched; rerun refused exit 2 |
| P0-6 | VFR clip and over-limit clip rejected before inference, nothing published | synthetic clips | PASS: real VFR and 2560x1440 clips exit 2; data/ inputs/staging/runs counts unchanged |
| P0-7 | Second concurrent run refused by the lease; failed/interrupted runs never in `runs/` | two processes; SIGINT | PASS: second process exit 75; SIGINT exit 130, INTERRUPTED manifest in staging (118 frames), not published |
| P0-8 | UI run lands in `data/runs/` and shows as before | browser | PASS: r_20261101T060711Z_a0b0a2 published; Complete, N/A latency, 3 downloads; CLI-held lease → UI error + next step |

## Risks
- OpenCV timestamps on some containers are coarse or missing → D4 tolerance; if unavailable, reject with
  a clear message rather than assume CFR.
- Copying large inputs costs disk → dedup by hash; limits cap size.
- Changing record field names breaks external readers → only our own tools read them; adapter for old runs.

## Estimate
MD says 0.5–1 focused day. Because records, identity and publication contracts are all new, expect
1–1.5 focused days. Re-estimate P1 (1–2) and P2 (1.5–3) at the P0 gate.

## Gate decision (P0.5): PASS after review fixes
The first gate claim was overstated: the independent review found a P1 (a truncated MP4 published as SUCCEEDED
with 8 of 60 frames) and 12 P2s; all are fixed in `8f9ec5e` with regression tests (`tests/test_integrity.py`).
All eight acceptance rows pass; 214 offline tests. The baseline is now `data/runs/r_20261101T062222Z_2ad4d6`
at clean commit `8f9ec5e` (`dirty: false`): 1,356 frames reported = decoded = inferred = records = written,
`source_complete`/`records`/`video_decode` PASS, hashes = shasum, 44.67 fps on mains.

**Known limitations carried into P1:**
- Legacy `--output-dir` runs take no lease (no data root), so two legacy runs can overlap. Documented in D6.
- Failed and interrupted attempts stay in `data/staging/` indefinitely; there is no cleanup command yet
  (MD§17 wants explicit, preview-first deletion, not automatic).
- The output budget is checked after writing, not during; an oversized run writes, then fails.
- Completeness against the container's frame count is only as good as that count; when the container gives none,
  the manifest records `source_complete: UNVERIFIED` instead of passing.
- The constant-frame-rate check relies on OpenCV container timestamps; clips without usable timestamps are
  rejected, not assumed constant.
- `report.json` keeps the MVP status words (`success`/`error`/`interrupted`) while the manifest uses
  `SUCCEEDED`/`FAILED`/`INTERRUPTED`. The manifest is authoritative for publication.
- The lease uses `flock`, which is reliable on local disks; network filesystems are out of scope.
- `code.dirty` ignores untracked files. Inputs are copied into `data/inputs/` once per unique file (disk use).
- CUDA paths remain unverified on this machine (no NVIDIA GPU).

**Revised estimates:** P1 1.5–2.5 focused days (MD: 1–2). It adds the run shelf over `data/runs`, a sparse
records index, the exact-frame inspector with track summaries, and the UI2 guided-flow redesign, which
touches most of `app.py`. P2 stays at 1.5–3 days. Each phase still stops at its gate.

## Log
- 2026-11-01 step 1: records.py (v1.0 writer with observation_ids and integer source_t_us, box policy: clip ≤0.5 px overrun, reject the rest; validate(); read_records() adapting pre-v1.0 lines with legacy=True and unknown dims/run_id left None). 20 tests. Real legacy file runs/demo-recording/tracks.jsonl: 1,356 records contiguous, t_us = formula exactly (0 µs diff), 25,361 objects. Pipeline still writes the old format until step 4. 152 tests.
- 2026-11-01 step 2: store.py (streamed sha256, canonical JSON rejecting NaN, run_id r_<UTC>_<6hex>, ByteTrack YAML read from package files without importing Ultralytics, git commit + dirty, resolve_runspec: config_sha256 over settings; reproducibility recorded but not hashed; missing weights null + reason). 11 tests. Real hashes match shasum (yolo11n.pt, crosswalk.mp4 81 MB in 0.03 s). 163 tests.
- 2026-11-01 step 3: measured first: OpenCV POS_MSEC after grab() = current frame's PTS (CFR clips: 0.00-frame drift; ffmpeg VFR 30→15 fps clip: OpenCV reports a plausible 20.69 fps, timestamps drift 9.3 frames). Limits dataclass (A§6) on RunConfig; open_reader checks size/dims/frames/duration, then CFR on the first 240 frames, then reopens at frame 0; read_frames checks every frame and treats the frame limit as an error, never truncation. 15 tests (real ffmpeg VFR clip; skips without ffmpeg). Real CLI: crosswalk preflight 0.23 s (outside wall time), 60-frame run exit 0; VFR clip exit 2 'frame 2 is at 66.7 ms, expected 96.7 ms', no dir. 178 tests.
- 2026-11-01 step 4: store.py data root/import (dedup by hash, temp+verify+rename)/flock lease (LeaseError exit 75)/records-file validation/manifest/publish (rename, never overwrite); pipeline: canonical (lease, import, staging, publish) and legacy (in place + manifest) share one path; v1.0 records; runspec.json; validation before success (decode count, records order/count/schema, source unchanged, output budget); open_writer returns the codec; Tracker.class_names; CLI --output-dir optional, --data-root. Bug caught by test: budget breach returned early without raising; fixed. Removed dead types.frame_record/Track.to_dict. 190 tests. Real: see P0-2/3/7.
- 2026-11-01 step 5: UI runner runs canonical (data root, lease via pipeline); uploads written to data/.uploads temp, imported by hash, then removed; failed attempts are not offered as results (no partial downloads; diagnostics stay in staging, UI2 copy); LeaseError/StoreError mapped to next steps; runs/ui retired. Tests adapted + lease-held test. 191 tests. Browser: P0-8.
- 2026-11-01 step 6: audit of P0.4 + MD§18 fixture 9 per mode. Legacy already covered everything; canonical gaps filled in tests/test_failures.py (9): invalid input creates nothing in data/ (no import), all-empty run succeeds, writer failure and non-finalizing output (fixture 9: FAILED manifest, video_decode FAIL, not published), tracker box outside the image fails the run, lease released after error and interrupt. 200 tests.
- 2026-11-01 step 7: all acceptance rows recorded with real commands (P0-1..8 PASS). Baseline: r_20261101T060226Z_90470c (1356 frames, 45.19 fps, AC). The v1.0 run and the MVP-era run of the same clip have identical observation counts (25,361).
- 2026-11-01 step 8: gate PASS recorded with limitations and revised estimates (P1 1.5–2.5 days). README: canonical vs legacy runs, v1.0 records + adapter, runspec/manifest, exit 75, input limits, UI on the data root, 200 tests, new modules. interview.md and demo-script.md test counts and all 20 file:line references re-verified against the rewritten code.
- 2026-11-01 verify: reviewer confirmed P0-1/2/3/5/6/7 itself (extra CFR false-reject probes all accepted: B-frames, ts offset, 29.97, MKV), P0-4 by test, P0-8 unverified (no browser). P1: truncated faststart MP4 published as SUCCEEDED (8/60) — reproduced, fixed (completeness vs container count; container count measured exact on MP4/MKV/faststart). 12 P2s all fixed: real decoded/inferred/records counters; publish failure rewrites manifest FAILED; failed report run_dir = staging; managed input keyed by hash only, re-hashed on reuse, meta heals, upload display names; records check requires native v1.0 of this run/source; runtime duration limit; SourceChangedError + next step; --data-root with --output-dir refused; README wording; subprocess lease and canonical tamper tests. 15 regression tests, 214 total. Clean-commit baseline r_20261101T062222Z_2ad4d6.
