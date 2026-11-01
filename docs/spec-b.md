# MotionMan Release B spec, part 1 (P3–P4)
Condensed from `docs/reference/master-design.txt` (cited as MD§N) and `docs/reference/ui-reference-v2.txt` (UI2).
The reference files win on any disagreement. Extends `docs/spec-a.md`; every A§ contract still holds.
Grep for the `B§N` you need; do not read whole.

## B§1 Goal and boundary
Answer *when and where* something happened (P3), and make some quality statements *measurable* on a reviewed
subset (P4) (MD§21, MD§22). Order: P3 then P4, each with its own plan and gate (MD§19: one vertical slice at a
time). **Not in this spec:** P5 experiments, P6 queue/cancel/media exports, P7 masks/webcam, P8 TensorRT,
freehand polygons, real-world units (mph, metres), tracking metrics (HOTA/IDF1), COCO mAP, new models, any
network service. After each phase gate, stop for the owner's go-ahead.

## B§2 Shared rules (both phases)
- Analysis and evaluation read saved runs only. They never call the model (fake backend that raises, as A§8).
- Raw run artifacts are immutable: no P3/P4 action changes a run's records or manifest hashes (MD fixture 10).
- Every derived artifact records parent hashes (`records_sha256`, region/dataset hash), its canonical config,
  `implementation_version` and validation status. A failed derivation never invalidates the run (MD§11).
- Derived IDs hash canonical JSON (A§2). Same inputs and config give the same IDs and contents, byte for byte.
- Undefined numbers are `null` with a reason, never 0 (INV-7). All times are source time in µs from the
  records' `source_t_us`, never wall clock, so processing speed cannot change a result (MD fixture 5).
- Layout under the data root (JSON/JSONL, no database):
  `regions/<input_id>/<region_set_id>.json` · `analyses/<run_id>/<analysis_id>/` ·
  `datasets/<dataset_id>/{draft.json, revisions/r<N>.json}` · `evaluations/<run_id>/<evaluation_id>/` ·
  `disagreements/<id>/`. Each result folder has a `manifest.json` with artifact hashes. These are durable
  results, kept apart from `derived/`, which stays P1's disposable inspection cache.

## B§3 P3 regions (MD§10)
- Shapes: axis-aligned rectangles and oriented finite line segments only. Coordinates normalized to [0, 1] of
  the source frame, bound to `input_id`. Versioned region-set schema; invalid geometry (zero area, zero length,
  out of range) is refused with a message. Each region has an id, a display name and an optional class filter.
- Reference point: bounding-box centre. Any other policy is a later, explicit setting.
- Editing in the UI uses coordinate controls with a live overlay on the inspected frame (MD cut order: no
  freehand). Saving a region set never reruns inference.

## B§4 P3 temporal rules (MD§10)
- Occupancy = distinct non-null track IDs whose centre is inside, after the class filter; a centre on the edge
  counts as inside. Unassigned (null-ID) detections are reported separately and never enter events (INV-2).
- Region state UNKNOWN / OUTSIDE / INSIDE with a pending transition. Defaults, persisted in AnalysisSpec:
  deadband 3 source px, confirmation 100 ms and at least 2 observations, max gap 500 ms.
- A frame that was inferred without the track clears pending transitions and the crossing reference.
  Disappearance is not an exit. The first observation after reacquiring sets state without an event; a gap over
  the maximum resets to UNKNOWN.
- Line crossing: signed side per centre; keep last stable side outside the deadband; keep the *first* motion
  segment that intersects the finite line (direction + evidence interval); emit one event when confirmed on the
  opposite side; rearm only after stable movement outside the deadband. Touching, passing beyond an endpoint
  or jittering is not a crossing. No permanent counted-ID set: a genuine return crossing counts again.
- Dwell: sum adjacent observation intervals with both ends inside; never bridge a missing inferred frame or a
  gap over the maximum. Dwell-threshold and occupancy-threshold events fire once per upward transition.
- Analysis window `[start_us, end_us)`; events included by `confirming_t_us`; intervals clipped to the window.

## B§5 P3 events, heatmap and outputs (MD§11)
- AnalysisSpec: run and input IDs, `records_sha256`, region set and its hash, reference point, classes, window,
  deadband, confirmation, max gap, heatmap grid, implementation version. `analysis_id` = hash of it.
- Event: deterministic `event_id` (analysis, rule, track, ordinal), type (`LINE_CROSSING`, `ZONE_ENTRY`,
  `ZONE_EXIT`, `DWELL_THRESHOLD`, `OCCUPANCY_THRESHOLD`), rule id, track and class, direction where relevant,
  confirming frame and `confirming_t_us`, evidence interval. Re-running the same analysis duplicates nothing.
- Outputs: `analysis_spec.json`, `events.jsonl`, `summary.json` (per-rule counts, peak and time-weighted
  occupancy, dwell stats, coverage), `heatmap.json`, `manifest.json`. Editing a rule makes a new analysis; old
  ones stay so a count change can be explained.
- Heatmap: 256×144 grid of object-seconds (sample-and-hold per eligible interval). Colour scale and smoothing are
  display only. All-zero maps render a defined empty state. Labelled "image-space observed occupancy".
- Motion arrows on the inspected still in pixels/second from source-time differences, labelled as such.
- Bookmarks name their criterion (peak occupancy, first crossing per rule); they are not importance judgments.
- Exports: events CSV and JSON with IDs and units.

## B§6 P3 UI, CLI and gate
- UI2 section **Events & zones** (tab beside Inspect and Performance): region editor, Analyze button (the stage's
  one primary action), summary cards, event feed with filters (rule, class, track, type, time) that never change
  persisted totals, selecting an event opens its source frame in Inspect, heatmap with opacity control
  (re-render only), bookmarks, export. Plain-language copy explains occupancy versus cumulative crossings.
- CLI: `python -m motionman analyze --run <run_id> --regions <file|region_set_id> [--window a,b]`.
- Gate: MD fixtures 1–6 and 8 pass on synthetic records (one crossing; jitter gives none; return after rearm
  gives a second; a gap invents nothing; different processing speed gives identical results; a display filter
  leaves totals unchanged; identical config gives identical IDs). Changing regions reruns analysis only; opacity
  re-renders only. Real demo on the crosswalk clip: define a line, inspect a crossing at its source frame.

## B§7 P4 datasets and review (MD§13)
- Dataset revision: `dataset_id`, revision number, parent, `input_id`, per frame `source_frame_index`, status
  (`reviewed` | `excluded`), the classes fully reviewed on that frame, ignored regions; annotations with
  `annotation_id`, frame, class, `bbox_xyxy` (source px), origin (`accepted`, `added`, `corrected`,
  `imported`), optional note, optional ground-truth identity (distinct from predicted `track_id`).
- Draft vs frozen: `draft.json` is editable; *commit* writes an immutable `revisions/r<N>.json` with its hash.
  A revision carries a split (`eval` or `tuning`); the eval split is frozen apart from tuning material.
- Review queue: candidates from low confidence, track gaps, abrupt box changes and run disagreements, labelled
  as candidates, not errors. Compact review form per frame: accept, mark false positive, fix class or box with
  numeric fields and overlay preview, add a missed object, exclude the frame, mark classes fully reviewed.
- Import: MotionMan dataset JSON. (Other formats need the owner's go-ahead.)
- Ground truth comes only from human review. Labels are never generated and called reviewed (INV-7).

## B§8 P4 evaluation and disagreement (MD§13)
- Detection evaluation per class: sort predictions by confidence; match each to the highest-IoU unmatched
  eligible ground-truth box at or above the IoU threshold (default 0.5) with a documented tie-break;
  predictions at or below the recorded confidence threshold are dropped first. Predictions whose centre falls in
  an ignored region are neither TP nor FP. Only frames whose reviewed class scope covers the class are scored;
  partial frames are excluded. Output precision and recall (null + reason when undefined), TP/FP/FN, matched IoU
  distribution and example frames. Never called mAP. Every result names dataset revision, thresholds,
  evaluator version and run.
- Disagreement between two runs on the same `input_id`: per frame, class-agnostic one-to-one greedy matching by
  descending IoU above a threshold; report matched-same-class, class disagreements, unmatched A, unmatched B and
  localization IoU. Track IDs are never compared across runs and no winner is declared. Each item links to the
  frame and both predictions and can be sent to the review queue.
- Tracking metrics are deferred until temporal ground truth exists (and a pinned reference implementation).

## B§9 P4 UI, CLI and gate
- UI2 section **Quality lab**: Review (queue, form, commit revision), Evaluate (choose run and frozen revision),
  Compare runs (disagreements). Results show their dataset revision and thresholds on screen.
- CLI: `evaluate --run <id> --dataset <id>@r<N>`, `disagree <run_a> <run_b>`.
- Gate: partially reviewed frames are excluded unless their class scope is complete; an injected false positive
  or missed object changes the metric exactly as expected; swapped track IDs with the same boxes still match
  spatially (MD fixture 7); committing a correction leaves raw prediction hashes unchanged (fixture 10); every
  score names revision, threshold and evaluator version. Demo: review a disagreement, commit a revision,
  evaluate against the frozen subset, trace a number to its frame.

## B§10 Open decisions for the owner
1. Sequence: P3 then P4 (recommended; shared UI files) or both at once in separate worktrees.
2. P4 needs a human-reviewed subset: the owner reviews roughly 30–60 frames of the crosswalk clip in the review
   form (about an hour). Confirm the clip may be used, and who reviews.
3. Import formats beyond MotionMan JSON (for example a COCO detection subset): in or out for P4.
