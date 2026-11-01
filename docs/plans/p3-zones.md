# P3 Zones and events: regions, temporal rules, events, heatmap
Status: Active · Branch: `feat/p3-zones` · Spec: B§2–B§6 (MD§10, MD§11, MD§18 fixtures, MD§21)

## Outcome
A user draws lines and rectangles on a saved run, presses Analyze, and sees when and where things happened:
direction-aware line crossings, zone entries and exits, occupancy over time, dwell, an event feed that jumps to
the source frame, a heatmap and labelled bookmarks, exportable as CSV/JSON. The same records and regions always
give the same events, no matter how fast the clip was processed, and the model never runs.

## Non-goals
- Polygons, freehand drawing, foot-point or mask reference points (centre only).
- Real-world units, speeds in mph/km/h, site-flow claims; heatmaps compare only within one run.
- Analysis of legacy (pre-P0) runs: they lack guaranteed source time; refused with a message.
- Event reels or any derived video (P6). Background analysis jobs (P6): analysis runs inline.
- Review, ground truth, evaluation (P4).

## Decisions (confirmed 2026-11-01)
- D1 Storage: `<data root>/regions/<input_id>/<region_set_id>.json` and
  `<data root>/analyses/<run_id>/<analysis_id>/{analysis_spec,summary,heatmap,manifest}.json + events.jsonl`.
  Built in a temp folder and renamed in atomically; an existing folder that verifies is reused, a damaged one
  is rebuilt. `derived/` stays the disposable P1 cache.
- D2 Region set schema v1.0: `{schema_version, input_id, name, regions: [{id, name, kind: "line"|"rect",
  points: [[x,y],[x,y]] | box: [x1,y1,x2,y2], classes: [names] | null, dwell_threshold_s: num | null,
  occupancy_threshold: int | null}]}`. Coordinates normalized to [0, 1]; ids are `[a-z0-9-]{1,32}`, unique.
  `region_set_id` = hash of the canonical set. Line direction: travelling from the right side to the left side
  of A→B is `NEGATIVE_TO_POSITIVE` (documented with a picture in README).
- D3 Units: every time and duration is integer µs; heatmap cells are integer object-µs, stored sparse
  (`[[col, row, value], …]`). No floats in identity-bearing files, so outputs are byte-identical on rerun.
- D4 Eligibility: v1.0 records only, from runs whose records hash matches the manifest. Null-ID detections are
  counted as unassigned per frame and never enter rules (INV-2). Class filter applies before everything.
- D5 Continuity: Release A infers every frame, so any frame where a track is absent breaks continuity: it
  clears pending transitions and the crossing reference, and the next observation re-establishes state with no
  event. A source-time gap over `max_gap_us` (default 500 ms) also sets UNKNOWN. Dwell never bridges a break.
- D6 Stability: a centre is *stably* on a side (line) or inside/outside (rect) only when at least `deadband_px`
  (default 3 source px) from the boundary; inside the band it neither confirms nor clears. A transition
  confirms on the first observation where the track has been stably on the new side for ≥ `confirm_us`
  (100 ms) and ≥ 2 consecutive observations. Occupancy uses raw centres, edge counts as inside.
- D7 Crossing evidence: keep the first observation pair whose motion segment intersects the finite line (proper
  segment intersection, endpoints-touching excluded) while leaving the stable side; the event's evidence
  interval is `[t of that pair's first sample, confirming t]`. Rearm only after the new side is stable.
- D8 Events: `event_id` = hash(analysis_id, rule_id, track_id, per-rule ordinal); types `LINE_CROSSING`,
  `ZONE_ENTRY`, `ZONE_EXIT`, `DWELL_THRESHOLD` (once per confirmed visit), `OCCUPANCY_THRESHOLD` (upward
  transitions only). Ordered by confirming frame, then rule id, then track id.
- D9 Summary: per region: crossings by direction or entries/exits, peak occupancy and its first frame,
  time-weighted mean occupancy, dwell count/median/max (µs), visits; plus unassigned detections, coverage
  (frames, window), and bookmarks (`peak occupancy`, `first crossing`, `first entry`) naming their criterion.
- D10 Heatmap: 256×144 cells over the source frame; each eligible interval between consecutive observations of
  one track adds its duration to the earlier centre's cell (sample-and-hold). Palette, smoothing and opacity
  are display-only. Empty map renders a labelled empty state.
- D11 Motion arrows: computed on the fly for the inspected frame from each track's previous continuous
  observation: (dx, dy)/dt in px/s, drawn on the still, labelled "px/s". Not part of the analysis files.
- D12 UI (per `docs/design/ui.md`): new **Events & zones** tab. Region editor = name, kind, numeric coordinates,
  classes and thresholds, with a live overlay on a frame; Save region set; **Analyze** is the tab's one primary
  action. Results: summary cards with a one-line note on occupancy versus cumulative crossings, an event table
  with display-only filters, a selected event's source frame still (regions + box drawn) and **Show in
  Inspect** (sets the Inspect slider), heatmap with opacity, bookmarks. Events CSV/JSON join the header's Export
  menu when an analysis is open. Streamlit can't switch tabs from code, so the button says where to look.
- D13 CLI: `python -m motionman analyze --run <run_id> --regions <file|region_set_id> [--window START_S,END_S]
  [--data-root]`; prints a summary and the analysis folder; exit 0, 2 on bad input or a refused run.

## Steps
1. [x] `regions.py`: schema, validation (zero length/area, range, ids, duplicate ids), px conversion, save
   (content-addressed) / load / list per input. Tests: valid and each refusal; same content gives same id.
2. [x] `analytics.py` rules, pure functions over observation streams: side and stability, segment intersection,
   line state machine, rect state machine, occupancy, dwell, thresholds. Tests on synthetic tracks: MD fixtures
   1–4, endpoint pass, touch-and-return, edge-inside occupancy, disappear-on-one-side-reappear-on-other.
3. [x] Analysis runner: AnalysisSpec, `analysis_id`, one streaming pass over records via the P1 index, outputs
   + manifest, atomic publish, reuse and rebuild, window and class filter, legacy refusal. Tests: fixture 5
   (same records, different processing speed and wall times give identical files), fixture 8 (identical config
   gives identical ids and bytes), run hashes unchanged, a tracker factory that raises is never called.
4. [x] Heatmap, summary and bookmarks in the runner; CSV/JSON export; `analyze` CLI. Tests: sample-and-hold
   totals equal summed dwell for a single track, empty map, export round-trip keeps ids and units, CLI codes.
5. [x] UI part 1: Events & zones tab, region editor with overlay, save/load region sets, Analyze. AppTest: the
   tab never infers, saving regions doesn't analyze, Analyze writes one analysis and reuses it on repeat.
6. [ ] UI part 2: summary cards, event table with filters (fixture 6: filters never change persisted totals),
   selected event still and Show in Inspect, Export menu items, bookmarks. AppTest coverage for each.
7. [ ] Heatmap overlay with opacity (display-only re-render), motion arrows in px/s on the Inspect still.
   Tests: changing opacity doesn't rewrite analysis files; arrow velocity from known synthetic motion.
8. [ ] Real verification on the crosswalk clip: a line across the crossing and a kerb zone, CLI and UI;
   check several events at their source frames by eye; record counts, timings and any deadband issue. README
   (regions, rules, direction picture, outputs), ARCHITECTURE, demo script, acceptance table, STATE.

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| P3-1 | Offline gate passes; no weights, GPU or network in new tests | `scripts/check.sh`; CI | |
| P3-2 | One crossing gives one event; jitter or touching gives none; return after rearm gives a second | `.venv/bin/pytest -q tests/test_analytics.py -k "crossing or jitter or touch or return"` | |
| P3-3 | A break in observation invents no crossing, exit or continuous dwell | `.venv/bin/pytest -q tests/test_analytics.py -k "gap or break"` | |
| P3-4 | Different processing speed, same records → identical analysis files | `.venv/bin/pytest -q tests/test_analysis.py -k speed` | |
| P3-5 | Identical config → identical analysis and event ids, byte-identical outputs | `.venv/bin/pytest -q tests/test_analysis.py -k identical` | |
| P3-6 | Display filters and opacity never change persisted totals or files | `.venv/bin/pytest -q tests/test_app.py -k "events and (filter or opacity)"` | |
| P3-7 | Analysis never calls the model and leaves run hashes unchanged | `.venv/bin/pytest -q tests/test_analysis.py -k "never_infers or immutable"` | |
| P3-8 | Changing regions makes a new analysis; the old one stays readable | `.venv/bin/pytest -q tests/test_analysis.py -k regions_change` | |
| P3-9 | CLI analyze works and refuses legacy or bad region input with exit 2 | `.venv/bin/pytest -q tests/test_analysis.py -k cli` | |
| P3-10 | Real crosswalk demo: line + zone, events checked at source frames, numbers in README | real run (Log) | |

## Risks
- Default deadband/confirmation are unvalidated on real footage (MD§10 says so): step 8 checks them on the
  crosswalk clip and records any change with evidence.
- ByteTrack ID switches split one person into two tracks: each may cross once. Documented, not hidden.
- Streamlit has no programmatic tab switch: "Show in Inspect" sets the frame and says to open Inspect.
- AppTest coverage of the region editor's overlay image is limited to "an image is shown"; the browser check
  in step 8 covers the visuals.

## Log
<!-- One line per step: date, step, commit, result. No pasted output. -->
- 2026-11-01 step 1: regions.py: RegionSet/Region (line A→B or rect, normalized coords rounded to 6 decimals so float noise keeps ids stable, optional class filter, rect-only dwell/occupancy thresholds); parse refuses unknown fields, bad schema/input_id, 0 or >32 regions, duplicate or malformed ids, zero-length lines, non-positive-area boxes, out-of-range/NaN/bool coords, bad thresholds and class lists; content-addressed save under regions/<input hex>/rs_<16 hex>.json (identical content → same file), load by id refuses hand-edited files, list per video skips unreadable ones. 31 tests, 393 total.
- 2026-11-01 step 2: analytics.py Engine: per-frame feed in order (refuses out-of-order), any absent frame or >500 ms gap drops a track's state (no event on reacquire, no bridged dwell), line side/deadband/confirmation (≥100 ms and ≥2 stable obs), crossing tested from the last point strictly on the stable side (found while testing: a centre exactly on the line otherwise slipped through as a silent side switch), endpoint pass = silent side change, rearm on confirm; rect INSIDE/OUTSIDE/UNKNOWN with the same deadband rules, edge-inclusive occupancy, time-weighted occupancy_us, per-visit dwell, dwell/occupancy thresholds on upward transitions only; null IDs only counted as unassigned. Integer µs throughout. 23 tests (fixtures 1–4, on-line centre, endpoint, unconfirmed excursion, long gap, class filter, ordering, determinism), 416 total.
- 2026-11-01 step 3: analysis.py: find_run (re-verifies the run), AnalysisSpec (run/input/records hash, frame size, fps, full region set + hash, centre reference, window, params, grid, impl version) → analysis_id; one streaming pass over records into the engine; events.jsonl (canonical JSON, event_id = hash(analysis, rule, track, per-rule ordinal)), summary.json (coverage, events by type, unassigned, per-region crossings or entries/exits/peak/occupancy_us/covered_us/visits/dwell, integers only), analysis_spec.json, manifest.json (no timestamps, so rebuilds are byte-identical); built in a temp folder and renamed in, identical analysis reused, damaged one moved aside and rebuilt, failure leaves nothing; refuses legacy runs, regions for another video, empty or inverted windows. 10 tests (byte-identical rebuild, slow vs fast tracker gives identical source-time results, never infers, run hashes unchanged, old analysis kept after a region change, damage rebuilt, window, params in identity, failure cleanup), 426 total.
- 2026-11-01 step 4: analytics.Heatmap (256×144 sample-and-hold object-µs, tracked objects of every class, same continuity as dwell, sparse row-major cells) written as heatmap.json and listed in the manifest; summary bookmarks (first crossing per line, first entry and peak occupancy per zone, each naming its criterion, sorted by frame); export_csv (13 columns, IDs, µs, value units) and export_json (spec + summary + events, units block); `analyze` CLI (--run, --regions file|rs_id with the file saved to the store, --window START_S,END_S, --csv/--json refusing existing files, exit 2 on refused input). 6 tests (heatmap total = observed dwell and lands in the expected cells, empty window, bookmarks, export round-trip, CLI paths and refusals), 432 total. Real smoke (crosswalk run 15bbeb, unchecked placeholder regions): 1,356 frames analyzed in 0.46 s wall; 128 visits vs 24 entries shows ByteTrack ID breaks splitting visits, to examine in step 8.
- 2026-11-01 step 5: Events & zones tab (between Inspect and Performance) in ui/zones.py; render.draw_regions (mist outlines, arrow A→B with "+" on the positive side, lime for the shape being edited). Editor: saved-set picker per video, region list with remove, Add a region (name, Line/Rectangle, coordinates in % of frame, classes, rect dwell/occupancy alerts), validation errors shown with their reason, Save regions (never analyzes), Analyze (primary only while stale, else a disabled "Up to date"; ui.md updated). Browser check found and fixed: the pending shape wasn't drawn until named; a just-added region was drawn twice; loading a saved set kept the old set name, which (being part of the set's identity) made an analyzed set look stale (regression test added). Also observed: analyses are per run, so a second run of the same clip shows Analyze for an already-analyzed set; step 6 will name the run. 7 AppTests (empty state, add without analyzing or inferring, invalid region refused, duplicate names and remove, save without analyzing, analyze once then reused as Up to date in a fresh session, loaded set name restored), 439 total.
