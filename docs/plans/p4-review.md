# P4 Review and evaluation: reviewed datasets, detection metrics, run disagreement
Status: Active · Branch: `feat/p4-review` · Spec: B§2, B§7–B§9 (MD§13, MD§18 fixtures 7 and 10, MD§22)

## Outcome
Some quality statements become measurable. A person reviews a small set of crosswalk frames in a compact form
(accept, false positive, fix, add missed, exclude, mark classes complete) and commits a frozen revision. Any
compatible run can then be scored against it: per-class precision and recall at stated thresholds, with every
number tracing back to frames. Two runs of the same video can be compared box by box, by position, with the
differences offered as review candidates. Predictions are never changed and the model never runs.

## Non-goals
- Tracking metrics (HOTA, IDF1, MOTA) and temporal ground-truth identity beyond an optional field: deferred
  until temporal labels and a pinned reference implementation exist (B§8).
- mAP, PR curves, or any claim beyond the reviewed frames, classes and thresholds.
- Generated, model-assisted or "auto-accepted" ground truth. Labels exist only where a person reviewed (INV-7).
- Import formats other than MotionMan dataset JSON (COCO etc. need the owner's go-ahead).
- Freehand box drawing: boxes are edited with numeric fields and a live overlay (as in P3).
- Experiments and robustness recipes (P5).

## Decisions (confirmed 2026-11-01)
- D1 Storage: `<data root>/datasets/<dataset_id>/draft.json` (editable) and `revisions/r<N>.json` (immutable,
  written once by commit, hash recorded in `revisions/index.json`); `evaluations/<run_id>/<evaluation_id>/` and
  `disagreements/<disagreement_id>/`, each with a `manifest.json`, built in a temp folder and renamed in (as P3).
- D2 Dataset schema v1.0: `{schema_version, dataset_id, revision, parent_revision, input_id, split: "eval" |
  "tuning", source_run_id (the run whose predictions seeded review, for reference only), frames: [{frame,
  status: "reviewed" | "excluded", classes_complete: [names], ignored: [[x1,y1,x2,y2] px]}], annotations:
  [{annotation_id, frame, class_name, bbox_xyxy (source px), origin: "accepted" | "added" | "corrected" |
  "imported", note, gt_identity: int | null}]}`. `dataset_id` is a user-chosen slug `[a-z0-9-]{1,40}`.
  Annotations are allowed only on reviewed frames; a reviewed frame must name at least one complete class.
- D3 Seeding: a draft is started from a run and a frame list; nothing is labelled until the reviewer acts on a
  frame. "Accept" copies a prediction's class and box into an annotation with origin `accepted`; the run's
  records are only read.
- D4 Frame selection for the eval subset: evenly spaced frames (default every 30th: 45 of 1,356), plus frames
  the reviewer adds from the review queue. The selection is part of the draft, not computed later.
- D5 Review queue candidates (labelled "candidates, not errors"): confidence below 0.40; first frame after a
  track gap; abrupt box change (IoU < 0.5 between a track's consecutive boxes); disagreement items from D9.
  Kinds take turns (each in frame order) so one flood can't hide the rest; capped at 200 shown, totals kept.
- D6 Commit: validates the draft, writes `r<N>` with split chosen at commit, never overwrites. Evaluation only
  accepts `eval` revisions; `tuning` revisions are for choosing settings, so a number is never tuned on the set
  that scores it. The revision's hash and number are in every result.
- D7 Evaluator v1 (B§8): drop predictions with confidence below the threshold (default: the run's recorded
  conf); per class, sort predictions by confidence (ties: lower observation index first); match each to the
  unmatched eligible ground-truth box with the highest IoU ≥ threshold (default 0.5; ties: lower annotation_id).
  Predictions whose centre is in an ignored box are neither TP nor FP. A frame counts for a class only if that
  class is in its `classes_complete`; excluded frames count for nothing. Precision = TP/(TP+FP), recall =
  TP/(TP+FN), each null with a reason when the denominator is 0. Output per class and micro-total, matched IoU
  p10/p50/p90, and up to 12 example frames per error kind (FP, FN). Named "fixed-threshold detection
  precision/recall", never mAP. `evaluation_id` hashes run id, records hash, revision hash, thresholds,
  evaluator version.
- D8 Compatibility: a run can be evaluated only against a revision with the same `input_id` and frame size;
  frames outside the run's processed range are reported as not evaluated, not as misses.
- D9 Disagreement v1: two runs with the same `input_id`; per common frame, class-agnostic one-to-one greedy
  matching by descending IoU ≥ 0.5 (ties: lower observation index in A, then B). Items: matched same class,
  class disagreement, unmatched A, unmatched B, plus matched IoU distribution. Track IDs are never compared and
  no winner is declared. Each item has the frame and both observation ids; "Send to review" adds the frame to a
  chosen draft.
- D10 UI (per `docs/design/ui.md`): a **Quality lab** tab with three sections. Review: choose or create a
  dataset, frame list and queue, the per-frame form (predictions with Accept / False positive / Fix, Add
  missed with numeric box and overlay, Exclude, classes complete, ignore box), Commit (the one primary action
  while the draft has unsaved changes). Evaluate: choose a frozen eval revision; cards per class and total,
  thresholds, revision and evaluator version on screen; example frames open in Inspect. Compare runs: pick a
  second run of the same video; counts per kind and an item table with frames.
- D11 CLI: `dataset import FILE`, `dataset show ID[@rN]`, `evaluate --run ID --dataset ID@rN [--conf] [--iou]`,
  `disagree RUN_A RUN_B [--iou]`. Exit 0; 2 on invalid or incompatible input.

## Steps
1. [x] `datasets.py`: schema and validation, draft save/load, seeding from a run, commit to immutable
   revisions with an index, import/export. Tests: each refusal, revision immutability, raw records unchanged.
2. [x] `evaluate.py`: matching and metrics per D7, publish. Tests: exact match, duplicates, empty positives and
   negatives, ignored boxes, partial class scope excluded, null denominators with reasons, tie-breaks, an
   injected FP and a removed GT box change TP/FP/FN exactly as expected, identical inputs → identical id/bytes.
3. [x] `disagree.py`: spatial matching per D9, publish. Tests: fixture 7 (two runs with swapped track IDs and
   the same boxes match spatially), class disagreement, unmatched both ways, frame-range intersection.
4. [x] Review queue candidates (D5) as pure functions over the P1 index. Tests per heuristic and the cap.
5. [ ] CLI `dataset`, `evaluate`, `disagree`. Tests: happy paths and exit 2 refusals (tuning revision, other
   video, unknown revision, frame size mismatch).
6. [ ] UI part 1: Quality lab → Review (dataset, frames, queue, form, commit). AppTests: never infers;
   committing a correction leaves the run's hashes unchanged (fixture 10); a partially reviewed frame can't be
   committed without a complete class.
7. [ ] UI part 2: Evaluate and Compare runs, with revision/thresholds/version on screen, example frames to
   Inspect, Send to review. AppTests for each, and that nothing writes without its button.
8. [ ] **Owner review session (blocking):** the owner reviews the 45 evenly spaced frames (about an hour), then
   commits `crosswalk-eval@r1` as `eval`. Then: evaluate runs `ed3f02` (imgsz 640) and `6057dc` (imgsz 480);
   disagreement between them; a synthetic regression on the real revision (drop one reviewed box → recall
   falls by exactly one TP); trace one FP and one FN to their frames. README, ARCHITECTURE, demo script,
   acceptance table, STATE.

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| P4-1 | Offline gate passes; no weights, GPU or network in new tests | `scripts/check.sh`; CI | |
| P4-2 | Partially reviewed frames are excluded unless their class scope is complete | `.venv/bin/pytest -q tests/test_evaluate.py -k scope` | |
| P4-3 | An injected FP or missed object changes the metric exactly as expected | `.venv/bin/pytest -q tests/test_evaluate.py -k injected` | |
| P4-4 | Swapped track IDs with the same boxes still match spatially (fixture 7) | `.venv/bin/pytest -q tests/test_disagree.py -k swapped` | |
| P4-5 | Committing a correction leaves raw prediction hashes unchanged (fixture 10) | `.venv/bin/pytest -q tests/test_datasets.py tests/test_quality_ui.py -k unchanged` | |
| P4-6 | Every score names revision, thresholds and evaluator version; null with reason when undefined | `.venv/bin/pytest -q tests/test_evaluate.py -k "names or undefined"` | |
| P4-7 | Only frozen eval revisions are scored; tuning or incompatible revisions are refused | `.venv/bin/pytest -q tests/test_evaluate.py -k refused` | |
| P4-8 | Review, evaluation and comparison never call the model | `.venv/bin/pytest -q tests/test_quality_ui.py -k never_infers` | |
| P4-9 | Real: owner-reviewed revision, two runs evaluated, disagreement, synthetic regression, numbers traced | real run (Log) | |

## Risks
- The owner's review is the critical path: without it P4-9 can't pass, and no number may be invented (INV-7).
  Steps 1–7 don't depend on it; step 8 stops and asks.
- Reviewing small, distant people is slow and uncertain. The form lets a frame name only the classes it fully
  covers and mark hard areas as ignored, so uncertainty is recorded instead of guessed.
- 45 frames of one clip support statements about this clip and these thresholds only. README says so.
- ByteTrack IDs fragment, so a track-level view is avoided: everything here is per frame and per box.

## Log
<!-- One line per step: date, step, commit, result. No pasted output. -->
- 2026-11-01 step 1: datasets.py: even_frames (every 30th), new_draft from a verified run (video id, frame size, source run, selected frames; nothing labelled), set_frame (reviewed/excluded, classes complete, ignored boxes; excluding drops that frame's labels), add/accept/remove annotations (accept copies class and box and notes the source observation; records only read), commit (validates, next number skips any file already on disk, exclusive create so a revision is written once, read-only, sha256 in revisions/index.json), load_revision verifies the hash (changed or orphaned revisions refused), parse_ref `<id>@r<N>`, import of a MotionMan draft or revision as a new draft with origin imported (ground truth only after a person commits); full validation with named refusals. 18 tests (incl. raw run hashes unchanged after commit: fixture 10), 480 total.
- 2026-11-01 step 2: evaluate.py: pure score() per D7 (drop below the confidence threshold; per class, highest confidence first, ties by observation index; match the unmatched reviewed box with the highest IoU ≥ threshold, ties to the lower annotation id; centres in ignored boxes count neither way; only classes marked complete on reviewed frames are scored; excluded frames and frames the run never processed are reported, not counted as misses); precision/recall null with a reason on a zero denominator; matched IoU p10/p50/p90; up to 12 FP and FN examples with frames. evaluate(): refuses legacy runs, tuning revisions, another video, a different frame size, confidence below the run's own, IoU outside (0, 1], records changed since verification; id hashes run, records, revision hash, thresholds and evaluator version; published under evaluations/<run>/ev_*/ like P3 (temp folder, atomic rename, reuse, damaged rebuilt). Spec B§8 wording aligned to D7 (below the threshold is dropped). 13 tests, 493 total.
- 2026-11-01 step 3: disagree.py: match_frame (class-agnostic, one-to-one, every pair at IoU ≥ threshold best first, ties by observation index in A then B) → matched same class, class disagreement, unmatched A, unmatched B; compare() streams both runs' records in lockstep over the frames both processed and re-hashes both files in full; refuses legacy runs, different videos or frame sizes, bad IoU; dg_ id from both runs' ids and records hashes, IoU and version; disagreement.json (counts, per class, matched IoU, frames with differences, top 20 frames, settings of each side, a note that track IDs are never compared and neither run is assumed right) + items.jsonl (differences only, with both observations) + manifest, published and reused like P3/P4; datasets.select_frames for Send to review (frames only, nothing labelled). 6 tests (fixture 7 swapped IDs, class disagreement, unmatched both ways, best-IoU one-to-one, common frames only, reuse and refusals, send to review), 499 total.
- 2026-11-01 step 4: review.py: candidates over records (confidence < 0.40; first frame of a track's return after a gap; consecutive boxes with IoU < 0.5) plus run differences, labelled candidates not errors; up to 200 shown with totals per kind; frames_of for Add to review. Real check on run ed3f02 with the 640-vs-480 comparison (6057dc): 15,346 candidates (8,703 run differences, 5,462 low confidence, 1,151 track gaps, 30 box jumps); the planned kind-first ranking filled all 200 places with run differences from 36 frames, so kinds now take turns (57/30/57/56 across 82 frames; D5 updated). Comparison 640 vs 480 for the record: 18,215 boxes matched with the same class, 97 class disagreements, 7,049 only at 640, 1,557 only at 480, matched IoU p50 0.883, differences on 1,355 of 1,356 frames. 7 tests, 506 total.
