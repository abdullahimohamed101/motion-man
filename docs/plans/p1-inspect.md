# P1: make completed work inspectable
Status: Active · Branch: `feat/p1-inspect` · Spec: A§7, A§8 · Design: MD§9, MD§11, MD§20 P1, UI2 · Estimate: 1.5–2.5 days

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
3. [ ] `FrameSource` (D1): exact frames with timestamp verification, forward steps, LRU 8. Tests with fakes;
   real-clip script comparing seeked frames with sequential decode, plus step timings.
4. [ ] Still rendering with view options (labels, trail length/opacity from prior records, spotlight, class
   filter) and evidence PNG export (D6). Tests: PNG equals the rendered still; overlays equal the records.
5. [ ] `motionman/presets.py` (D7). Tests incl. name validation and view/tracking separation.
6. [ ] UI restructure (D8, D9): header, setup screen, Inspect tab (playback + still + frame slider + object
   panel with intervals; confidence, gaps, contact sheet behind disclosure), Performance tab, Export disclosure.
   AppTest incl. a tracker factory that raises if called.
7. [ ] Browser verification (P1-9, P1-10), README, log.

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| P1-1 | Offline gate passes; real-media tests run in CI | `scripts/check.sh`; PR check log | |
| P1-2 | Inspecting never infers: slider, object, view, tab and run changes with a tracker that raises if called | AppTest + runner test | |
| P1-3 | The still is exactly source frame i (byte-equal to sequential decode at sampled i on crosswalk); a mispositioned frame is refused | script + test | |
| P1-4 | Still overlays equal frame i's records; evidence PNG equals the shown still; export manifest hashes match | test + script | |
| P1-5 | Index and summaries bound to the records hash; stale/missing rebuilt; tampered run refused | tests | |
| P1-6 | Shelf lists only SUCCEEDED published runs; legacy runs read-only and labelled | tests + browser | |
| P1-7 | Per-run context restores; IDs from another run never carry over | AppTest | |
| P1-8 | Frame cache ≤ 8; a one-frame step forward is fast (measured, reported) | test + script | |
| P1-9 | First run: Try demo → Run tracking → inspect an object → export PNG, without opening Advanced settings | browser | |
| P1-10 | Keyboard-only path and 320 px / tablet / desktop widths: no overflow, visible focus | browser | |

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
