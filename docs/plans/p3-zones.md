# P3 Zones and events: regions, temporal rules, events, heatmap
Status: Draft · Branch: `feat/p3-zones` · Spec: B§2–B§6 (MD§10, MD§11, MD§18 fixtures, MD§21)

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

## Decisions (proposed; confirm with the plan)
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
1. [ ] `regions.py`: schema, validation (zero length/area, range, ids, duplicate ids), px conversion, save
   (content-addressed) / load / list per input. Tests: valid and each refusal; same content gives same id.
2. [ ] `analytics.py` rules, pure functions over observation streams: side and stability, segment intersection,
   line state machine, rect state machine, occupancy, dwell, thresholds. Tests on synthetic tracks: MD fixtures
   1–4, endpoint pass, touch-and-return, edge-inside occupancy, disappear-on-one-side-reappear-on-other.
3. [ ] Analysis runner: AnalysisSpec, `analysis_id`, one streaming pass over records via the P1 index, outputs
   + manifest, atomic publish, reuse and rebuild, window and class filter, legacy refusal. Tests: fixture 5
   (same records, different processing speed and wall times give identical files), fixture 8 (identical config
   gives identical ids and bytes), run hashes unchanged, a tracker factory that raises is never called.
4. [ ] Heatmap, summary and bookmarks in the runner; CSV/JSON export; `analyze` CLI. Tests: sample-and-hold
   totals equal summed dwell for a single track, empty map, export round-trip keeps ids and units, CLI codes.
5. [ ] UI part 1: Events & zones tab, region editor with overlay, save/load region sets, Analyze. AppTest: the
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
