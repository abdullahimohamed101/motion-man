# ADR-006: Native run contract, backend metadata and tracker semantics
Status: Proposed (G2; accepted or changed at the G2 gate)
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
  - Detections with score ≥ `track_low_thresh` reach the tracker.
  - Records hold the tracker's output.
  - A frame where the tracker returns nothing records its detections with score ≥ `conf`, with null IDs (INV-2).

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
