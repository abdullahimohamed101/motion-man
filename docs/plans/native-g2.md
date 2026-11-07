# Native G2: serial slice (one video, batch 1, one slot)
Status: Active (owner approved 2026-11-07) · Branch: `feat/native-g2` (off `feat/native-g1`) · Spec: C§2, C§3,
C§7, C§8, ADR-001, ADR-003, ADR-006 (Proposed)

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
  - Every RT-DETR detection with score ≥ `track_low_thresh` (0.1) reaches the tracker, so the second
    association stage gets its low-confidence inputs (NA§5).
  - Records hold the tracker's output, as the reference does.
  - When the tracker returns nothing for a frame, the detections with score ≥ `conf` are recorded with null
    IDs. This matches the reference behaviour and INV-2.
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
2. [ ] **[Mac]** Tracker replay fixtures:
   - `tools/tracker_replay/`: scenario generator and a reference runner that drives Ultralytics `BYTETracker`
     directly with cached detections.
   - `tests/fixtures/tracker/*.json` (small, committed) cover: entry, exit, reappearance inside and past the
     buffer, crossing, overlap, low-confidence recovery, empty frames, class conflict, assignment ties and two
     trackers with identical inputs.
   - pytest checks that the reference output is reproduced.
3. [ ] **[Mac]** `native/src/track/` ByteTrack (D4, D5): Kalman, IoU, LAPJV, lifecycle, ID allocator,
   `class_policy`. CTest `tracker_replay` compares tracks with the fixtures after canonical ID renaming:
   association exact, boxes within 1e-3 px. Aware-mode fixtures stop cross-class updates. NOTICE file.
4. [ ] **[Mac]** Native record and timing writers (portable): schema 1.0 lines with Python's rounding,
   `observation_id` order, null IDs, `source_t_us` from the G0 fixture. A fixture shared by pytest and CTest
   proves that both languages produce equal values. Trails logic (INV-4) unit-tested in C++.
5. [ ] **[PC]** `motionman-native run --runspec … --source … --out …` (D1, D7, D8): decode, preprocess,
   TensorRT, admission, tracker, records streamed (INV-3), render, timings, receipt, status lines, signal
   handling. Identity refusal still applies. CTest gpu `native_run_smoke` on a short clip.
6. [ ] **[Mac]** `motionman/native.py` controller and `--backend native` in the CLI:
   - Validate the input, import it and resolve the runspec, then launch and relay status.
   - After exit: check the receipt, run `validate_records_file`, check the video decode count and the source
     hash, then write the report and manifest and publish.
   - Interrupted, failed or inconsistent runs are never published (INV-5, INV-6).
   - pytest uses a fake native executable for: success, crash mid-run, missing records, wrong count, exit 0
     with a bad receipt, SIGINT.
7. [ ] **[PC]** Owner's yes for the PC app environment (D10). Real replay parity: native dumps the crosswalk
   clip's admitted detections, which are replayed through the Python reference and the native tracker in
   `agnostic` mode and compared as in step 3.
8. [ ] **[PC→Mac]** End to end on the crosswalk clip:
   - A native run publishes; `runs.open_run`, `analyze` and `evaluate` read it on the PC.
   - The run folder and source are copied to the Mac and opened in the UI.
   - An interrupted native run (SIGINT) stays unpublished and is reported as `INTERRUPTED`.
   - Observed fps is recorded as an observation.
9. [ ] **[Mac]** G2 write-up: acceptance, ADR-006 Accepted or changed, G3–G6 re-estimate, STATE, PR.
   **G2 gate: stop for the owner.**

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| N2-1 | Python gate passes; legacy runs unchanged except new metadata | `scripts/check.sh` | step 1: ruff ok, 543 passed |
| N2-2 | Old (1.0) and new (1.1) reports and runspecs both open | `.venv/bin/pytest -q tests/test_runs.py tests/test_store.py` | step 1: passed incl. a run rewritten to 1.0 that still opens |
| N2-3 | Python reference reproduces the committed tracker fixtures | `.venv/bin/pytest -q tests/test_tracker_replay.py` | |
| N2-4 | Native ByteTrack matches the fixtures (agnostic) and aware fixtures pass | `ctest --preset mac-cpu -R tracker_replay` | |
| N2-5 | Native records equal Python's for the shared fixture | `ctest --preset mac-cpu -R records` and `.venv/bin/pytest -q tests/test_native_records.py` | |
| N2-6 | Controller publishes only verified runs (fake native, 6 failure cases) | `.venv/bin/pytest -q tests/test_native_controller.py` | |
| N2-7 | Native run smoke passes on the PC GPU | `ctest --preset linux-cuda-release -L gpu -R native_run_smoke` | |
| N2-8 | Real replay parity on crosswalk detections: identical associations after renaming | `python tools/tracker_replay/compare.py --dump <crosswalk dets>` on the PC | |
| N2-9 | Published native run passes existing readers | on the PC: `python -m motionman --backend native --source <crosswalk>`, then `analyze` and `evaluate --dataset crosswak-eval@r2` on its run ID | |
| N2-10 | Interrupted native run is not published; manifest `INTERRUPTED` | SIGINT during a PC run, then `ls data/runs` and read the staging manifest | |
| N2-11 | Report shows backend tensorrt, engine hash, precision fp32, GPU name | `jq .backend data/runs/<id>/report.json` | |
| N2-12 | Mac `mac-cpu` and PC CTest all pass; no weights, engines or footage in git | `ctest --preset mac-cpu`, `ctest --preset linux-cuda-release`, `git ls-files \| grep -E '\.(pth\|onnx\|engine\|plan\|mp4)$'` | |

## Risks
- Exact tracker parity can fail on floating-point or assignment ties. Investigate and pin a fixture first, and
  never loosen the comparison silently. If Ultralytics-specific behaviour (e.g. duplicate removal) differs from
  MIT ByteTrack, port the behaviour, not the code.
- OpenCV 4.6 on Ubuntu may lack an H.264 writer: fall back to `mp4v` and record the codec, as Python does.
- Decode on WSL2 from `/mnt/c` is slow, so keep inputs in the Linux filesystem.
- Low-score admission (D6) changes which tracks appear compared with the legacy backend's conf-filtered input.
  That is intended and recorded, and is not a regression.
- Estimate: 4–6 focused days (G1 re-estimate).

## Log
<!-- One line per step: date, step, commit, result. No pasted output. -->
- 2026-11-07 step 1: `settings.backend` (legacy default `ultralytics`/fp32 in `types.LEGACY_BACKEND`) and an optional `tracking` override in `resolve_runspec` (runspec 1.1); `build_report` takes `backend` and `tracker` (report 1.1 with `schema_version`, `backend`); `runspec_backend`/`report_backend` read 1.0 as legacy. compare gains a `backend` factor; bundle README names the backend. Legacy runs' config hash changes once (new key). Native stage timings deferred to step 6. Gate 543 passed.
