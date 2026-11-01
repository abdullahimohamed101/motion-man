# P1: make completed work inspectable
Status: Done · Branch: `feat/p1-inspect` · Spec: A§7, A§8 · Design: MD§9, MD§11, MD§20 P1, UI2 · Estimate: 1.5–2.5 days

## Outcome
Opening and exploring a published run is useful without ever invoking the model: pick a run, scrub to an exact
source frame, select an object and see when it was observed, adjust overlays, and export an evidence PNG that
matches what's on screen. The UI follows UI2's guided flow (Choose video / Try demo → Run tracking → Inspect →
Export).

## Non-goals (A§1)
Zones, events, heatmaps (P3), stage timings, comparisons, doctor, bundles (P2), review/evaluation, derived
video exports (crops, highlights, redaction), clicking boxes on the still, a custom or synchronized player,
cancel buttons, run deletion.

## Findings (spike on the demo clip and baseline run)
- OpenCV seeks are **exact**: 8/8 frames, random order incl. backwards, both `POS_FRAMES` and `POS_MSEC`, match
  sequential decode byte-for-byte. But they're **slow, 31–396 ms**, because the clip has a keyframe only every
  ~10 s (6 in 56 s); a seek decodes forward from the previous keyframe.
- Run → source: `runspec.settings.input.input_id` → `data/inputs/<hex>/source`; display name in `meta.json`.
- Cheap enough to do on every open: verifying all artifact hashes against the manifest (36 ms for 62 MB),
  a sparse index every 128 records (11 offsets, 2 ms), summaries for all 701 tracks (24 ms).
- Legacy MVP runs record `source.path` (e.g. `clips/crosswalk.mp4`) but no hashes.

## Decisions (confirmed 2026-11-01)
- D1 Exact frames come from the run's managed source, never the annotated video. A `FrameSource` keeps one
  reader open: forward steps of ≤ 48 frames decode forward; other jumps seek. Every returned frame's timestamp
  is checked against `index / fps`; a mismatch shows "Couldn't position this frame exactly", never a wrong frame.
  LRU cache of 8 decoded frames (A§6).
- D2 The still's overlays are drawn from the frame records (`render`), so view options apply without
  re-rendering video. The annotated MP4 stays the playback artifact, labelled separately from the still.
- D3 Derived data in `data/derived/<key>/` with key = `sha256_json({kind, records_sha256, impl_version})`:
  `index.json` (byte offset every 128 records) and `tracks.json` (per track: class history, first/last frame,
  observation count, observed intervals, gaps, confidence p10/p50/p90), built in one streaming pass. Rebuilt
  if missing or mismatched; never the source of truth.
- D4 Run shelf: canonical runs in `data/runs/` with a SUCCEEDED manifest of a known schema. Opening a run verifies
  every artifact hash and the managed source hash against the manifest/RunSpec; a mismatch refuses the run
  with a clear message. Staging is never listed.
- D5 Legacy MVP runs: listed read-only from `./runs/*/` when `report.json` and `tracks.jsonl` exist, labelled
  "Legacy (unverified)"; the inspector works only if the recorded source path exists. First to cut if behind.
- D6 Evidence PNG: `data/exports/<export_id>/evidence.png` + `export.json` (run_id, source frame, t_us, view
  options, selected track, record hash, PNG hash); export_id = hash of that recipe. Downloads of existing
  artifacts stay; no other export types in P1.
- D7 Presets: `data/presets/<name>.json` with separate `view` (labels, trail length, trail opacity, spotlight,
  class filter) and `tracking` (device, confidence) sections; names `[A-Za-z0-9_-]{1,40}`. A view preset
  applies at once; a tracking preset only fills the next run's settings ("These settings apply to your next run").
- D8 UI per UI2: header (wordmark, source name, run selector, written status, neutral New video). Setup screen
  with Advanced settings collapsed. Result tabs **Inspect** and **Performance** (metrics + run details move
  there). One lime primary per stage: Run tracking in setup, Export in a result.
- D9 Per-run context in session state keyed by run_id (frame, track, view options); restored when returning to
  a run; selections that don't exist in the opened run are dropped.
- D10 Step 0: install ffmpeg in CI so the two real-media tests run on GitHub instead of skipping.

## Steps
0. [x] CI: install ffmpeg; confirm the real-media tests run (not skipped) on the PR.
1. [x] `motionman/runs.py`: discovery, open + verify (D4), legacy read-only listing (D5). Tests incl. tampered
   artifact, tampered source, staging never listed.
2. [x] `motionman/inspection.py` part 1: derived index + track summaries (D3), random record access by frame.
   Tests incl. stale index rebuild, gaps/intervals on synthetic records.
3. [x] `FrameSource` (D1): exact frames with timestamp verification, forward steps, LRU 8. Tests with fakes;
   real-clip script comparing seeked frames with sequential decode, plus step timings.
4. [x] Still rendering with view options (labels, trail length/opacity from prior records, spotlight, class
   filter) and evidence PNG export (D6). Tests: PNG equals the rendered still; overlays equal the records.
5. [x] `motionman/presets.py` (D7). Tests incl. name validation and view/tracking separation.
6. [x] UI restructure (D8, D9): header, setup screen, Inspect tab (playback + still + frame slider + object
   panel with intervals; confidence, gaps, contact sheet behind disclosure), Performance tab, Export disclosure.
   AppTest incl. a tracker factory that raises if called.
7. [x] Browser verification (P1-9, P1-10), README, log.

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| P1-1 | Offline gate passes; real-media tests run in CI | `scripts/check.sh`; PR check log | Local pass, 294 tests. CI real-media run: pending on the P1 PR |
| P1-2 | Inspecting never infers: slider, object, view, tab and run changes with a tracker that raises if called | AppTest + runner test | Pass: test_app::test_inspecting_never_runs_the_model (runner.start, default_tracker and Tracker all raise) across slider, object, labels, trail, playback and run switch |
| P1-3 | The still is exactly source frame i (byte-equal to sequential decode at sampled i on crosswalk); a mispositioned frame is refused | script + test | Pass: test_frames::test_real_clip_frames_match_sequential_decoding, ::test_a_mispositioned_seek_is_refused_not_shown; real 1080p crosswalk 13 mixed-order requests byte-exact (step 3 log) |
| P1-4 | Still overlays equal frame i's records; evidence PNG equals the shown still; export manifest hashes match | test + script | Pass: test_still::test_overlays_are_drawn_from_this_frames_record, ::test_evidence_png_is_exactly_the_shown_still; real baseline frame 1064 PNG pixel-identical, hash = export.json (step 4 log) |
| P1-5 | Index and summaries bound to the records hash; stale/missing rebuilt; tampered run refused | tests | Pass: test_inspection cache reuse/rebuild, corrupt/foreign cache rebuilt, records-hash mismatch refused; test_runs::test_a_tampered_artifact_refuses_the_run |
| P1-6 | Shelf lists only SUCCEEDED published runs; legacy runs read-only and labelled | tests + browser | Pass: test_runs listing tests (staging and non-SUCCEEDED never listed; legacy read-only, labelled); test_app::test_legacy_runs_are_hidden_until_asked_for; browser picker shows the baseline |
| P1-7 | Per-run context restores; IDs from another run never carry over | AppTest | Pass: test_app::test_each_run_keeps_its_own_context_and_ids_never_carry_over; test_workspace::test_restore_drops_what_this_run_does_not_have |
| P1-8 | Frame cache ≤ 8; a one-frame step forward is fast (measured, reported) | test + script | Pass: test_frames::test_cache_is_bounded_and_serves_repeats_without_decoding; real one-frame step median 1.0 ms (max 8.4), cache hit ~0.5 ms, far jumps 30–426 ms (step 3 log) |
| P1-9 | First run: Try demo → Run tracking → inspect an object → export PNG, without opening Advanced settings | browser | Pass: demo run r_20261101T065533Z_a53b2e (1,356 frames) opened itself; object picked; evidence.png + export.json written; Advanced never opened |
| P1-10 | Keyboard-only path and 320 px / tablet / desktop widths: no overflow, visible focus | browser | Pass: scrollWidth = viewport at 320/768/1440; every stop in bounds with a focus ring; run opened, object picked and frame stepped by keyboard only |

## Risks
- Large seeks stay up to ~0.4 s on sparse-keyframe clips: mitigated by forward stepping and the cache; reported,
  not hidden. A preview proxy with dense keyframes is a possible later change (needs an ADR).
- The UI restructure touches most of `app.py`: AppTest coverage first, then browser checks.
- Streamlit widget state across run switches is subtle: per-run context is keyed explicitly (D9).

## Cut order if behind (MD§20)
Contact-sheet styling, extra preset types, legacy listing (D5). Keep exact-frame access, identity semantics and
no-hidden-inference behaviour.

## Log
- 2026-11-01 plan: spike measured seek exactness (8/8 byte-equal) and cost (31–396 ms, keyframes every ~10 s),
  verify-on-open 36 ms, index 2 ms, 701 track summaries 24 ms.
- 2026-11-01 steps 0–1: ffmpeg added to the CI apt step (verified on the PR, P1-1). runs.py: list_runs (SUCCEEDED manifests only, newest first; legacy MVP folders read-only), open_run (re-hash every artifact + managed source, runspec hash checks; RunError on mismatch). Manifest times now ms precision (two runs in one second sorted arbitrarily — caught by test). 11 tests, 225 total. Real: 19 runs listed in 9 ms (3 canonical, 16 legacy); baseline opened + verified in 87 ms; legacy demo-recording opens with source found, unverified. UI note: hide legacy behind a toggle (16 mostly test runs).
- 2026-11-01 step 2: motionman/inspection.py (renamed from inspect.py: it shadowed the stdlib module). One streaming pass builds the sparse index (offset per 128 records) and TrackSummary per ID (intervals, gaps, class counts, confidence p10/p50/p90); null IDs counted separately, never a track; cache in data/derived/<sha256(records_sha, impl_version)>/, reused, rebuilt when stale or corrupt; records.parse_line shared with read_records. 8 tests, 233 total. Real baseline: build 144 ms, cached load 5 ms, random access 0.13 ms/frame (matches sequential); 701 tracks, 475 seen < 1 s; #10 traffic light observed in all 1356 frames.
- 2026-11-01 step 3: FrameSource in inspection.py (one open reader; forward moves ≤48 frames decode forward, others seek; timestamp checked on every decoded frame → FrameError and reader closed on mismatch or decode failure; LRU 8). 7 tests (fake capture incl. a seek that lands one frame late; real clip vs sequential decode). Real 1080p crosswalk via the managed source: 13 requests in mixed order all byte-exact; one-frame steps median 1.0 ms (max 8.4); cache hits ~0.5 ms; jumps 30–426 ms (keyframes every ~10 s). 240 tests. Feeds P1-3 and P1-8.
- 2026-11-01 step 4: render.ViewOptions (labels, trail length/opacity, spotlight, class filter) + annotate_still (MP4 path untouched); inspection: tracks_of, observed (this exact frame), trails_for (observed centres only, bounded), render_still (source frame + its own record), export_evidence (exports/<recipe hash>/evidence.png + export.json; reused for the same recipe). 7 tests, 247 total. Real: baseline frame 1064, spotlight #3262 (216 frames, 13 intervals): PNG pixel-identical to the still, hash = export.json, render 188 ms incl. a seek. Feeds P1-4.
- 2026-11-01 step 5: presets.py (data/presets/<name>.json; view = labels/trail length/opacity/classes, tracking = device/conf; allowlisted fields with type and range checks; names [A-Za-z0-9_-]{1,40}; explicit overwrite). Spotlight deliberately never saved: track IDs are run-local (D9). 21 tests, 268 total.
- 2026-11-01 step 6: app.py rewritten around the shelf. Header: run selector (legacy behind "Show legacy runs"), written status badge, New video. Setup: Try demo / Choose video, Advanced settings collapsed (tracking presets, device, detection threshold), Run tracking. Result: Inspect tab (native playback marked unsynchronized; exact still from the source frame + its own record; frame slider; view options; object panel with observed-in-this-frame, intervals, First/Last seen, details and contact sheet behind disclosures), Performance tab, Export popover (3 downloads + evidence PNG). ui/workspace.py: per-run context restored per run with stale frames/IDs dropped (D9). Bug caught by AppTest: two runs of one clip in the same minute had identical labels and Streamlit selectboxes match by label, so picking run B reopened A; labels now end with the run ID's tail. 22 tests (14 AppTest with a runner and model entry points that raise, 8 workspace), 281 total. Browser on the real baseline: open run, select #892, Last seen → frame 686 with spotlight; New video stays on setup; no console errors.
- 2026-11-01 step 7: browser on the real app. P1-9: Try demo → Run tracking (real model, 1,356 frames) → result opened itself → object picked → Create evidence PNG wrote exports/441052d0…/evidence.png; Advanced settings never opened. P1-10: no horizontal scroll at 320, 768 and 1440 px; full tab order recorded at 320 px, all stops in bounds with visible focus; Saved runs, object selector and Source frame slider (arrow keys: frame 0 → 5) driven by keyboard alone. Found and fixed: Streamlit's heading anchor links added four empty tab stops ("Link to heading"); hidden in style.py. README Local UI section rewritten for the workbench. 281 tests.
- 2026-11-01 verify: reviewer found no high issues; acceptance P1-2..P1-8 pass, P1-1 CI half pending on the PR. Fixed all 4 medium and 6 of 7 low findings: page no longer crashes on a broken legacy run (RecordError now caught; bad legacy/runspec timing → RunError), on a contact-sheet or evidence-export frame error, or on a run with 0 records; view presets now apply (selector in View options, re-render only) and Advanced lists only tracking presets; saving over a preset needs "Replace"; open requires the manifest to cover tracks/annotated/report + records hash and refuses paths outside the run folder; a cached run whose files change (size/mtime) is verified again before reuse; an evidence PNG is reused only if it matches export.json's hash; damaged cache entries rebuild; a still also checks its record's own time. 13 new tests, 294 total. Real runs: both current-format runs open; one early-P0 run (input stored as source.mp4) is refused as before.
