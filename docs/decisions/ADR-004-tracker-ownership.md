# ADR-004: Shared detector, one tracker per stream (amends INV-1)
Status: Proposed (takes effect at G4; G0–G3 keep INV-1 unchanged)
Date: 2026-11-04

## Context
INV-1 says one model instance owns tracker state per run, frames are fed in order, and track calls never run in
parallel. That fits the single-source Python loop. Multistream native execution (G4) needs one TensorRT engine
serving many sources; 25 sources should not mean 25 detector copies. Mutable track identities must still never
leak between sources.

## Decision
When G4 begins, INV-1 becomes:
> Detector resources (engine, weights) may be shared and immutable. Each stream owns its own tracker state and ID
> allocator. Within a stream, frames reach the tracker in source order, one at a time.
- At most one admitted inference frame per source until its tracking completes (no reorder buffer at first).
- Execution contexts are per GPU slot and never submitted concurrently.
- Track identity is `(session_id, stream_id, local_track_id)`; identical numbers in two streams are unrelated.

## Alternatives
- One detector and tracker per stream: memory grows with stream count and does not scale to 25. Rejected.
- One tracker shared across streams: mixes identities between videos. Rejected.

## Consequences
- AGENTS.md INV-1 text is updated in the G4 plan, not before.
- Tests: identical videos in two streams produce independent IDs; delayed GPU completions do not reorder a
  stream's tracking.

## Revisit when
Cross-camera re-identification is brought into scope (it is excluded in spec C§1).
