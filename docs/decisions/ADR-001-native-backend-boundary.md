# ADR-001: Native engine as a whole-run backend
Status: Proposed (owner approval of `docs/spec-c.md`)
Date: 2026-11-04

## Context
The owner set a new direction: MotionMan should genuinely run C++, CUDA, TensorRT and OpenCV on an NVIDIA GPU
(`docs/reference/native-architecture.md`). Today the per-frame loop is Python (`pipeline.py`) calling Ultralytics
through `FrameTracker.update(image)`; the `cuda` option only picks the PyTorch device. The Mac has no NVIDIA GPU;
the owner's PC does. The Python app (store, records, UI, zones, review, evaluation) is tested and worth keeping.

## Decision
- Add a separate C++ executable, `motionman-native`, that owns the whole per-frame path on the PC: decode,
  preprocessing, TensorRT inference, tracking, rendering and record writing.
- The boundary between Python and native is a completed run: Python resolves a config, launches the job, then
  validates and publishes the staged artifacts. No per-frame call crosses the boundary or the network.
- Keep the existing Python pipeline as the named reference backend (`legacy`), never relabelled as native.
- One repository. Native code lives under `native/`; Python adapters under `motionman/backends/` later (G5).

## Alternatives
- TensorRT behind `FrameTracker.update`: the serial Python loop stays in control and nothing is native in the
  hot path. Rejected.
- Frame-by-frame RPC from the Mac to the PC: adds network cost per frame and changes nothing architecturally.
  Rejected.
- Rewrite the UI and analytics in C++ too: risk without advancing the GPU goal. Rejected.
- Python TensorRT runtime as the product: acceptable only as a G1 feasibility probe, not the finished engine.

## Consequences
- Existing saved runs and every A§/B§ contract stay valid; native child runs must pass the same readers.
- Report metadata gains backend, precision and host fields; `store.resolve_runspec` and `metrics.build_report`
  stop hardcoding Ultralytics and FP32 (G2).
- CI keeps its CPU-only gate; GPU tests run only on the PC.

## Revisit when
The native engine cannot produce schema-1.0 records without loss, or the PC becomes unavailable for good.
