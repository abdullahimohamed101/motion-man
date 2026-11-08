# Native G2: serial slice (one video, batch 1, one slot)
Status: G2 gate reached 2026-11-07, awaiting the owner (approved 2026-11-07) · Branch: `feat/native-g2` (off `feat/native-g1`) · Spec: C§2, C§3,
C§7, C§8, ADR-001, ADR-003, ADR-006 (Accepted at G2)

## Outcome
On the PC, `python -m motionman --backend native --source <clip>` runs one video through `motionman-native`:
OpenCV CPU decode, the G1 C++ preprocessing, TensorRT FP32 RT-DETR, a native ByteTrack, records, an annotated
video and timings. Python resolves the config, launches the native run, validates the output and publishes it.
The published run passes the existing readers (`runs.open_run`, inspection, `analyze`, `evaluate`), and its
report says it came from TensorRT on the RTX 3060 Ti. Native ByteTrack matches the Python reference tracker on
replayed detections.

## Non-goals
- FP16 or any precision change (G3). Batching, several slots or streams, async pipelining, sessions (G4).
- Running from the Mac UI, remote jobs or automatic transfer (G5). G2 copies one run to the Mac by hand.
- Any speed claim. fps in the report is an observation of one run, not a benchmark (ADR-002, INV-7).
- GPU decode, NVENC, CUDA kernels for preprocessing or drawing. A pixel-identical match with Python drawing.
- Changing the Python `legacy` backend's behaviour or any A§ or B§ contract. UI changes.

## Decisions
- D1 Run contract (ADR-006):
  - Python owns input import, `runspec.json`, validation, `report.json`, `manifest.json` and publication.
  - Native gets the exact runspec bytes plus paths. It writes `tracks.jsonl`, `annotated.mp4`, `timings.jsonl`
    and `native_receipt.json` (status, counts, engine hash, host) into the staging folder.
  - Native writes JSON status lines to stdout and human logs to stderr.
  - A native exit code of 0 alone is never success (C§7).
- D2 `settings.model`, `settings.inference` and `settings.tracking` gain a `backend` block (tensorrt, profile,
  ONNX and engine hashes, precision, GPU). The legacy backend writes `backend: ultralytics`. Old runspecs without
  the block still open.
- D3 `report.json` gains `schema_version` "1.1": a `backend` section and per-stage native timings (preprocess,
  infer, track). Readers treat a missing version as 1.0. Both versions open in `runs` and the UI.
- D4 Reference tracker: Ultralytics BYTETracker 8.4.196 with `bytetrack.yaml` defaults, which the legacy
  backend already uses.
  - The native port follows the MIT ByteTrack algorithm (FoundationVision), with the licence kept in
    `native/third_party/NOTICE.md`.
  - Ultralytics code is AGPL, so it is not copied; it is only the behaviour we compare against.
  - The port includes: double-precision XYAH Kalman filter, IoU with `fuse_score`, Jonker-Volgenant assignment,
    two-stage association, unconfirmed tracks, the lost buffer and a per-instance ID allocator.
- D5 Class policy: `agnostic` reproduces the reference; `aware` blocks matches between classes (NA§8).
  - Parity is checked in `agnostic` mode.
  - Native runs default to `aware`, with its own fixtures. The runspec records which mode was used.
- D6 Detector admission and records:
  - Every RT-DETR detection with score > `track_low_thresh` (0.1) reaches the tracker, so the second
    association stage gets its low-confidence inputs (NA§5).
  - Records hold the tracker's output, as the reference does.
  - When the tracker returns nothing for a frame, the detections with score > `conf` (strict, as Ultralytics
    filters) are recorded with null IDs, unless an unconfirmed track is pending; then the frame records no
    objects. This matches Ultralytics 8.4.196 `on_predict_postprocess_end` and INV-2.
  - Detector boxes are clamped to the image in float32 at admission, and tracker boxes before recording, as
    Ultralytics' `clip_boxes` does. A box with nothing left inside the image is dropped and counted in the
    receipt; the Python writer would refuse it.
  - Records keep schema 1.0 exactly.
- D7 Native runtime for G2:
  - One TensorRT context and one non-blocking stream.
  - Device and pinned buffers allocated once at start.
  - Frames processed in order, with no per-frame allocation or global device sync (C§7).
  - SIGINT or SIGTERM stops after the current frame and writes the receipt as `interrupted`.
  - A CUDA error writes the receipt as `error`.
- D8 Annotated video is drawn in C++ with OpenCV, using the same colours per ID, labels and trail rules as
  `render.py` (≤30 points, expire after 60 unseen frames; INV-4). It is checked by frame count, not pixels.
- D9 Tests that need no GPU run on the Mac and in CI:
  - Tracker replay fixtures (C++ CTest `mac-cpu` and pytest).
  - Record and line formatting fixtures shared by both languages.
  - The Python controller, driven by a fake native executable (INV-8).
- D10 The PC needs the Python app's own environment for the controller and the reference tracker. Installing it
  is a download that waits for the owner's yes (CPU torch is enough; GPU work stays in native).

## Steps
1. [x] **[Mac]** ADR-006 Proposed → reviewed. Python adds backend metadata to runspec and report (D2, D3):
   `store.resolve_runspec(backend=…)` and `metrics.build_report` stop hardcoding Ultralytics and FP32. Report
   1.1 is readable next to 1.0. pytest for both versions; legacy runs are unchanged apart from the new fields.
2. [x] **[Mac]** Tracker replay fixtures:
   - `tools/tracker_replay/`: scenario generator and a reference runner that drives Ultralytics `BYTETracker`
     directly with cached detections.
   - `tests/fixtures/tracker/*.json` (small, committed) cover: entry, exit, reappearance inside and past the
     buffer, crossing, overlap, low-confidence recovery, empty frames, class conflict, assignment ties and two
     trackers with identical inputs.
   - pytest checks that the reference output is reproduced.
3. [x] **[Mac]** `native/src/track/` ByteTrack (D4, D5): Kalman, IoU, LAPJV, lifecycle, ID allocator,
   `class_policy`. CTest `tracker_replay` compares tracks with the fixtures after canonical ID renaming:
   association exact, boxes within 1e-3 px. Aware-mode fixtures stop cross-class updates. NOTICE file.
4. [x] **[Mac]** Native record and timing writers (portable): schema 1.0 lines with Python's rounding,
   `observation_id` order, null IDs, `source_t_us` from the G0 fixture. A fixture shared by pytest and CTest
   proves that both languages produce equal values. Trails logic (INV-4) unit-tested in C++.
5. [x] **[PC]** `motionman-native run --runspec … --source … --out …` (D1, D7, D8): decode, preprocess,
   TensorRT, admission, tracker, records streamed (INV-3), render, timings, receipt, status lines, signal
   handling. Identity refusal still applies. CTest gpu `native_run_smoke` on a short clip.
6. [x] **[Mac]** `motionman/native.py` controller and `--backend native` in the CLI:
   - Validate the input, import it and resolve the runspec, then launch and relay status.
   - After exit: check the receipt, run `validate_records_file`, check the video decode count and the source
     hash, then write the report and manifest and publish.
   - Interrupted, failed or inconsistent runs are never published (INV-5, INV-6).
   - pytest uses a fake native executable for: success, crash mid-run, missing records, wrong count, exit 0
     with a bad receipt, SIGINT.
7. [x] **[PC]** Owner's yes for the PC app environment (D10). Real replay parity: native dumps the crosswalk
   clip's admitted detections, which are replayed through the Python reference and the native tracker in
   `agnostic` mode and compared as in step 3.
8. [x] **[PC→Mac]** End to end on the crosswalk clip:
   - A native run publishes; `runs.open_run`, `analyze` and `evaluate` read it on the PC.
   - The run folder and source are copied to the Mac and opened in the UI.
   - An interrupted native run (SIGINT) stays unpublished and is reported as `INTERRUPTED`.
   - Observed fps is recorded as an observation.
9. [x] **[Mac]** G2 write-up: acceptance, ADR-006 Accepted or changed, G3–G6 re-estimate, STATE, PR.
   **G2 gate: stop for the owner.**

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| N2-1 | Python gate passes; legacy runs unchanged except new metadata | `scripts/check.sh` | step 1: ruff ok, 543 passed; step 2: 563; step 3: 564; step 4: 568; step 6: 582; steps 7–9: 583. Legacy runs only gain `backend` and report `schema_version` |
| N2-2 | Old (1.0) and new (1.1) reports and runspecs both open | `.venv/bin/pytest -q tests/test_runs.py tests/test_store.py` | step 1: passed incl. a run rewritten to 1.0 that still opens |
| N2-3 | Python reference reproduces the committed tracker fixtures | `.venv/bin/python tools/tracker_replay/reference.py --check` (imports Ultralytics, so outside pytest, INV-8) and `.venv/bin/pytest -q tests/test_tracker_replay.py` | step 2: 14/14 reproduced exactly; pytest 20 passed; a swapped ID is caught. Step 3: 16/16 (15 scenarios + lapjv) |
| N2-4 | Native ByteTrack matches the fixtures (agnostic) and aware fixtures pass | `ctest --preset mac-cpu -R 'tracker_replay\|lapjv'` | step 3 (Mac): 15/15 fixtures with raw IDs equal and max box error 0 px; LAPJV 400/400 vs lap incl. ties; aware checks pass; 6 mutations caught. Step 5 (PC, GCC 13): passes, no warnings |
| N2-5 | Native records equal Python's for the shared fixture | `ctest --preset mac-cpu -R native_records` and `.venv/bin/pytest -q tests/test_native_records.py` | step 4 (Mac): 130/130 record lines and 60/60 timing lines byte-identical, 3 bad boxes refused by both, 22/22 trail checkpoints; pytest 4 passed; 5 mutations caught. Step 5 (PC, GCC 13): passes |
| N2-6 | Controller publishes only verified runs (fake native, 6 failure cases) | `.venv/bin/pytest -q tests/test_native_controller.py` | step 6 (Mac): 14 passed. Success publishes and opens in `runs` and the UI views; crash, missing records, short records, short video, exit 0 with another run's receipt, exit 0 with status error, refused, self-interrupted and Ctrl-C are never published. 3/3 mutations caught |
| N2-7 | Native run smoke passes on the PC GPU | `ctest --preset linux-cuda-release -L gpu -R native_run_smoke` | step 5 (PC): passed. 60 frames, 60 records and timing rows; rerun into the same folder refused, unchanged; wrong engine exit 3, no outputs; SIGINT exit 130 after 24 whole frames. Two runs byte-identical (records and video). PC ctest 13/13 |
| N2-8 | Real replay parity on crosswalk detections: identical associations after renaming | `python tools/tracker_replay/compare.py --dump <crosswalk dets>` on the PC | step 7 (PC): 1356 frames, 392,691 admitted detections, 117,154 track outputs, 3,635 IDs: 0 problems, raw IDs equal (no renaming needed), max box error 0 px. The same file in aware mode gives 1569 problems, so the check discriminates |
| N2-9 | Published native run passes existing readers | on the PC: `python -m motionman --backend native --source <crosswalk>`, then `analyze` and `evaluate --dataset crosswak-eval@r2` on its run ID | step 8 (PC): `r_20261107T234147Z_84aaf4` SUCCEEDED, 1356 frames; `open_run` ok; `analyze` on 3 saved region sets ok (rs_1a46…: 47 crossings, far-kerb peak 29); `evaluate` person P 0.803 R 0.959 (TP 277, FP 68, FN 12, 46 frames). Copied to the Mac: all artifact hashes match, opens in the UI, H.264 video plays (56.6 s, 1920×1080), Performance tab renders |
| N2-10 | Interrupted native run is not published; manifest `INTERRUPTED` | SIGINT during a PC run, then `ls data/runs` and read the staging manifest | step 8 (PC): SIGINT to the process group after 15 s: controller exit 130, engine stopped after 275 whole frames, staging manifest `INTERRUPTED`, records = timing rows = video = 275, runs/ unchanged, no engine process left |
| N2-11 | Report shows backend tensorrt, engine hash, precision fp32, GPU name | `jq .backend data/runs/<id>/report.json` | step 8 (PC): report 1.1, backend tensorrt / fp32 / rtdetr_r18_640, engine sha256:50378763…, ONNX sha256:a14a2c99…, GPU NVIDIA GeForce RTX 3060 Ti, TensorRT 11.2.1.2; environment.gpu the same |
| N2-12 | Mac `mac-cpu` and PC CTest all pass; no weights, engines or footage in git | `ctest --preset mac-cpu`, `ctest --preset linux-cuda-release`, `git ls-files \| grep -E '\.(pth\|onnx\|engine\|plan\|mp4)$'` | step 9: Mac 7/7, PC 13/13 (step 8 build), grep finds nothing (exit 1); `tests/fixtures` 500 KB; the crosswalk replay fixture stays on the PC |

## G2 result and re-estimate
- G2 met: N2-1..N2-12 all recorded above; ADR-006 Accepted with the amendments it lists. `--backend native` runs
  RT-DETR in TensorRT FP32 from C++ with a ByteTrack equal to the Python reference (15 scenarios, full crosswalk
  clip) and publishes only what Python verifies. Calendar time 2026-11-07; focused hours not timed (no claim).
- Planning risks, resolved: tracker parity is exact; H.264 works on the PC (libx264); D6's low-score admission
  is intended but makes busy videos (gate note); keep inputs off `/mnt/c` (they were).
- Found: OpenCV 4.6's drained-tail timestamps (ADR-006); the UI does not show the backend (gate note); the PC
  app env is Python 3.12, the Mac and CI 3.11. One crosswalk run (observation, not a benchmark): infer p50
  ~20 ms, CPU preprocessing 6–9 ms, drawing ~14 ms, 22 fps, so G4 must overlap CPU stages, not just batch.
- Re-estimate (focused days, before → now): G3 3–5 → 3–5 (strongly typed FP16 export still the first unknown);
  G4 4–7 → 5–8 (CPU stages as slow as inference); G5 2–4 → 2–4; G6 3–6 → 3–6 (owner labelling not included).
  G3–G6 total 13–23, or 16–29 with 25% contingency. Estimates, not a deadline.

## Log
<!-- One line per step: date, step, commit, result. No pasted output. -->
- 2026-11-07 step 1: `settings.backend` (legacy default `ultralytics`/fp32 in `types.LEGACY_BACKEND`) and an optional `tracking` override in `resolve_runspec` (runspec 1.1); `build_report` takes `backend` and `tracker` (report 1.1 with `schema_version`, `backend`); `runspec_backend`/`report_backend` read 1.0 as legacy. compare gains a `backend` factor; bundle README names the backend. Legacy runs' config hash changes once (new key). Native stage timings deferred to step 6. Gate 543 passed.
- 2026-11-07 step 2: `tools/tracker_replay/` (`scenarios.py` 14 seeded scenarios on a float32-exact grid, `reference.py` drives Ultralytics `BYTETracker` directly with `--write`/`--check`, `fixtures.py` load/dump/compare with canonical ID renaming, no Ultralytics import). `tests/fixtures/tracker/` 14 fixtures (180 KB, one frame per line). Found in the reference: a 0.30 track never confirms under `fuse_score` and burns an ID per frame; later starts confirm one frame late; class conflict relabels the car track. D6 corrected: an empty tracker output records nothing while an unconfirmed track is pending. `--check` 14/14 exact; gate 563 passed.
- 2026-11-07 step 3: `native/src/track/` (portable, in `motionman_cpu`, `-ffp-contract=off`): `lapjv` (Jonker-Volgenant with lap's phases and scan order, `extend_cost`/`cost_limit` wrapper), `kalman` (XYAH, double, 4x4 partial-pivot solve), `bytetrack` (float32 box/IoU/fusion steps, unconfirmed not predicted, lost height velocity zeroed, stale-removed track kept one frame, removed-ID history of 1000, per-instance IDs from 1, `ClassPolicy` agnostic/aware with aware also gating duplicate suppression). Fixtures: `tests/fixtures/lapjv.json` (400 lap cases) and a 15th scenario `low_conf_jump` added after the 0.4 second-stage mutation survived the first 14. CTest `lapjv` 400/400, `tracker_replay` 15/15 (boxes bit-equal after the float32 cast), mutations caught: predicting unconfirmed, dropping removed at once, no fusion, second stage 0.4, keeping lost velocity. `native/third_party/NOTICE.md` (ByteTrack MIT, lap BSD-2; Ultralytics only as behaviour). Mac ctest 6/6, gate 564. PC GCC build waits for step 5.
- 2026-11-07 step 4: `native/include/motionman/records.hpp` + `src/records.cpp` (portable): `py_round` (printf correct rounding, ties to even, Python's shortest form), `checked_box` (Python's rules, -0.0 kept), `frame_record_line` and `timing_line` (exact `json.dumps` text, ASCII-only strings), `frame_objects` (the ADR-006 rule: tracker output clamped in float32; empty + unconfirmed → nothing; else detections > conf with null IDs; no-extent boxes dropped and counted), `Trails` (30 points, 60-frame expiry). `tools/native_records/make_fixture.py` builds `tests/fixtures/native_records.json` (205 KB) from Python's own `frame_record`, timings shape and `render.Trails`; pytest checks it stays current. Found: Ultralytics' conf filter is strict (`>`), D6 corrected; detector boxes clamp at admission. Mutation testing note: the Makefile build skips a restore within the same second, so mutants are now rebuilt by deleting the object; a clean rebuild re-confirmed steps 3–4. Mac ctest 7/7, gate 568.
- 2026-11-07 step 5: first PC GCC 13 build of steps 3–4, no warnings, CPU ctest 8/8. `motionman-native run --runspec --run-id --source --out --profile --model-dir` (`native/src/app/run.cpp`, `annotate.cpp`): reads the exact runspec, refuses (exit 3, receipt `refused`) on a source hash, engine hash, profile, GPU or class-list mismatch, CPU decode with the timestamp and limit checks of `video_io`, G1 preprocessing, TensorRT, admission > `track_low_thresh` clamped, ByteTrack, `frame_objects`, records and timings streamed, `render.annotate` rules in OpenCV, `native_receipt.json` written atomically on every exit after the folder check, JSON status lines on stdout. Exit 0/1/2/3/130. Never overwrites its outputs (INV-5). Contract details for step 6: the runspec must carry `tracking.class_policy` and `backend.{profile, engine_sha256}`, and `rendering.codec` must be one native's OpenCV opens (`mp4v` here; corrected in step 8: Ubuntu's 4.6 does write H.264 `avc1`). Smoke runspec `tests/fixtures/native_run/crosswalk_60.runspec.json` built by Python's `resolve_runspec` (`tools/native_run/make_smoke_runspec.py`). Observed on crosswalk, 60 frames (observation, not a benchmark): about 70 recorded objects per frame, 276 IDs recorded but IDs reach 2002, because low-score starts never confirm and use up IDs (the step 2 reference quirk; step 7 checks it against Python). PC ctest 13/13, Mac ctest 7/7, gate 568.
- 2026-11-07 step 6: `motionman/native.py` (`run`, `NativeConfig`, `native_runspec`, `backend_block`) and `--backend native` / `--class-policy` in the CLI (native defaults: device cuda, profile `rtdetr_r18_640`; `--telemetry` and `--device cpu` refused). The controller imports, writes the runspec, launches the engine, relays progress, forwards Ctrl-C and waits for the receipt, then checks the receipt (run ID, runspec and source hashes, engine, consistent counts) and validates records, decoded video, timing rows and the source before `pipeline._finish` writes report and manifest and publishes. Refusal maps to CLI exit 2, an engine interrupt to 130. Report 1.1 for native (D3): `detect_track_ms` = preprocess + infer + track per frame (valid while G2 is serial), plus `timing.native_stages`, native scope strings and the GPU from the receipt (no torch import); the UI shows a native run unchanged. Manifest validation adds `native_receipt` and `timings`. Receipt gains `timing.processing_wall_s`. `tests/fake_native.py` covers 10 behaviours. Smoke runspec now built by `native.native_runspec`. Gate 582, PC ctest 13/13.
- 2026-11-07 step 7: owner's yes. PC app env `~/motion-man/.venv`: system Python 3.12.3 venv + pip (no uv on the PC), torch 2.15.0+cpu and torchvision from the PyTorch CPU index, then `requirements-dev.txt`; ultralytics 8.4.196, lap 0.5.13, numpy 2.4.6, opencv-python 5.0.0; 1.9 GB. `reference.py --check` 16/16 on the PC env. `motionman-native run --dump-detections` (admitted detections, `%.9g` float32-exact), `tools/tracker_replay/compare.py` (dump → reference replay → fixture → `test_tracker_replay <file>`, a new single-file mode). Found: Ubuntu's OpenCV 4.6 reports 0 ms for the 20 frames its decoder drains after the last packet (opencv-python 5.0 does not), so the first full run failed the constant-fps check at frame 1336. Native now counts a trailing run of 0 ms frames as `frames.timestamps_missing` instead of failing; a timestamp after such a frame is still an error; the controller records manifest `validation.timestamps` `UNVERIFIED` when the count is non-zero (fake `drained_tail` test). Full crosswalk run: success, 1356 frames, 20 timestamps missing. Gate 583, PC ctest 13/13.
- 2026-11-07 step 8: synced the branch and `data/regions`, `data/datasets` to the PC. First end-to-end native run published but its `mp4v` video did not play in the browser (error 4, unsupported source). Probe: Ubuntu's OpenCV 4.6 does write H.264 via libx264 (avc1, H264, X264 all open and write `h264`), so the step 5 assumption was wrong; the controller now writes `avc1` like Python (smoke runspec regenerated). Rerun `r_20261107T234147Z_84aaf4`: SUCCEEDED, 1356 frames, H.264; observed 22.09 fps end to end, detect+track p50 25.6 ms (infer about 20 ms), processing 61.4 s, setup 1.95 s, model load about 1.2 s: one run, an observation, not a benchmark (ADR-002). analyze, evaluate, open_run, Mac UI all read it. SIGINT run unpublished, `INTERRUPTED`. Gate notes: the native run draws many low-score boxes (admission > 0.1 per D6, tracker keeps them), and the UI does not show which backend made a run. Gate 583, PC ctest 13/13.
- 2026-11-07 step 9: acceptance N2-1..N2-12 complete; ADR-006 Accepted with amendments (drained-tail timestamps, avc1, report 1.1 timing mapping, dump flag); spec C§11 updated; G3–G6 re-estimated (13–23 days); STATE; G2 PR stacked on the G1 PR. G2 gate.
