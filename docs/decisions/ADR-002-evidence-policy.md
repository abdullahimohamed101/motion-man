# ADR-002: Performance and quality claims are measured, never assumed
Status: Proposed (owner approval of `docs/spec-c.md`)
Date: 2026-11-04

## Context
The native direction carries resume targets: 40+ FPS across 25+ streams, 45% lower latency, 1.8× throughput,
35% fewer identity switches under occlusion, RT-DETR in FP16. None has been measured in this repository. INV-7
already forbids fabricated numbers. Concurrent execution also breaks the current timing model, where stage
times are sequential and sum to at most the wall time.

## Decision
- Each target is a claim with a status: **met, unmet or inconclusive**, backed by raw trials, environment
  metadata and an exact reproduction command (spec C§10).
- Aggregate FPS = completed frames across all sources ÷ one shared processing interval; per-stream FPS is always
  reported beside it.
- A speed comparison names its baseline from the ladder H0/R0/N0–N3 and holds checkpoint, GPU, input, batch and
  timing boundary constant. H0 (Mac CPU YOLO11n) is never the denominator of a TensorRT speedup.
- At least 5 paired, alternating trials; median, range, p50/p95; failed trials are kept, never discarded.
- Identity-switch claims require TrackEval on labelled contiguous sequences, with absolute counts and IDF1/HOTA
  guardrails. Fewer switches from losing tracks is not an improvement.
- Native timing gets its own versioned schema; CUDA-event time, host stage time, queue wait and throughput are
  separate fields.

## Alternatives
- Carry the resume numbers forward as the spec: violates INV-7. Rejected.
- Tune configurations until a number is reached: changes the experiment silently. Rejected; any change is a new
  declared configuration.

## Consequences
A target may end the release unmet. That is a valid outcome and gets published as such.

## Revisit when
Never weakened; extended only when a new kind of claim appears.
