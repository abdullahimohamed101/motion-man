# ADR-006: Native run contract, backend metadata and tracker semantics
Status: Accepted (G2, 2026-11-07), with the amendments below; the owner reviews them at the G2 gate
Date: 2026-11-07

## Context
ADR-001 makes a completed run the boundary between Python and `motionman-native`. G2 needs this boundary made
concrete:
- who writes which file;
- how a native run is labelled so it is never confused with the Python reference (C§3);
- what native ByteTrack must reproduce.

The legacy backend's records come from Ultralytics `model.track`. Ultralytics filters detections by `conf` before
the tracker. When the tracker returns no tracks, Ultralytics keeps the raw boxes with no IDs.

## Decision
- **Ownership:**
  - Python imports the input, writes `runspec.json` (canonical JSON, hashed once) and launches native with
    those exact bytes.
  - Native writes `tracks.jsonl`, `annotated.mp4`, `timings.jsonl` and `native_receipt.json` into staging.
  - Python validates them with the existing checks, then writes `report.json` and `manifest.json` and publishes.
  - The native exit code alone is never success.
- **Backend metadata:**
  - The runspec settings and `report.json` carry a `backend` block: `ultralytics` or `tensorrt`, the profile,
    the ONNX and engine hashes, precision and GPU.
  - `report.json` gains `schema_version` "1.1"; a report without one is read as 1.0.
  - Records stay schema 1.0.
- **Tracker:**
  - The native ByteTrack is ported from the MIT ByteTrack algorithm, with its licence kept.
  - Ultralytics BYTETracker 8.4.196 with `bytetrack.yaml` is the behavioural reference and is checked by
    replaying detections. No AGPL code is copied.
  - `class_policy` `agnostic` matches the reference; `aware` blocks matches between classes and is the default
    for native runs. The runspec records the mode.
- **Admission:**
  - Detections with score > `track_low_thresh` reach the tracker, their boxes clamped to the image.
  - Records hold the tracker's output, its boxes clamped to the image (as Ultralytics' `clip_boxes`).
  - A frame where the tracker returns nothing records its detections with score > `conf`, with null IDs (INV-2),
    unless an unconfirmed track is pending; then it records no objects (Ultralytics 8.4.196 behaviour).
  - A box with nothing left inside the image is dropped and counted in the receipt, never written.

## Amendments from building G2
- **Runspec for native:** `settings.backend` carries `profile`, `onnx_sha256`, `engine_sha256`, `gpu` and
  `tensorrt`; `settings.tracking.class_policy` is required; `rendering.codec` is `avc1` (H.264). Native refuses
  (exit 3) a runspec whose source hash, engine hash, profile, GPU or class list differs from what it loads.
- **Receipt** (`native_receipt.json`, schema 1.0): status `success | error | refused | interrupted`, the run ID,
  the runspec and source hashes, frame counts (decoded, processed, records, timing rows, video), object counts,
  engine identity and phase times. Written atomically on every exit once the output folder is checked. Python
  publishes only when the receipt matches the run and its own checks pass. Exit codes 0, 1, 2, 3, 130.
- **Timestamps:** a decoder may report no timestamp (0 ms) for the frames it drains after the last packet
  (Ubuntu's OpenCV 4.6 does on the crosswalk clip). An unbroken run of such frames at the end is counted in
  `frames.timestamps_missing`, and the manifest marks `validation.timestamps` `UNVERIFIED`. A timestamp after a
  missing one, or any other drift, still fails the run.
- **Report 1.1 timing for a serial native run:** `detect_track_ms` is preprocess + infer + track per frame, so
  existing views stay correct; `timing.native_stages` keeps the three apart and scope strings say what each
  covers. This sum is valid only while frames run one at a time; G4 must not reuse it for overlapped stages
  (C§8).
- **Replay input:** `motionman-native run --dump-detections` writes each frame's admitted detections, exact in
  float32, for parity checks against the Python reference.

## Alternatives
- Native writes report and manifest itself: duplicates validation and lets an exit code claim success. Rejected.
- Copy Ultralytics' tracker code into C++: licence (AGPL) and it is Python-specific. Rejected.
- `agnostic` as the native default: identical to the reference, but lets a person update a vehicle track
  (NA§8). Kept as the parity mode only.
- Filter at `conf` before the tracker, as the legacy backend does: disables ByteTrack's low-score stage. Rejected.

## Consequences
- `store.resolve_runspec` and `metrics.build_report` take backend metadata instead of hardcoding FP32 and
  Ultralytics. Old runs keep opening.
- Native and legacy runs of the same clip differ by model and admission policy; comparisons must say so.

## Revisit when
G4 introduces sessions and child runs (`session.json`), or the replay parity cannot be reached without
changing the reference.
