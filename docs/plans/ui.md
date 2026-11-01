# Local UI: tracking workbench
Status: Draft · Branch: `feat/ui` (stacked on `feat/cli-mvp`, PR #1) · Spec: §9 · Budget: 2–3 h

## Outcome
`scripts/ui.sh` opens a local Streamlit page: pick the demo clip or upload an MP4, choose device and
confidence, press **Run tracking**, watch progress, then play the Original or Tracked video, read metrics
from this run's `report.json`, and download the three artifacts. Same `pipeline.run` as the CLI.

## Non-goals (spec §2, §9)
Synchronised players, custom timeline, clickable objects, live camera, other models, run history or
database, accounts, API server, React/Tailwind, marketing hero. CLI behaviour and benchmarks stay as they are.

## Findings (spike, Streamlit 1.66.1, scratch venv)
- Theme config covers most tokens: `base`, `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`,
  `textColor`, `borderColor`, `baseRadius`, `buttonRadius`, `font`, `codeFont`, `metricValueFontSize`.
- **No option for primary-button text colour**: measured white on `#A3E635` (≈1.4:1), not the required
  `#0B1105` (≈12.7:1).
- **Any widget click mid-run kills the script**: a 60-step loop stopped at step 21 with
  `streamlit...StopException`, which is a `BaseException`, not an `Exception`. Running inference in the
  script thread would abort the run, and the current pipeline would write `status: error` with `error: null`.
- `st.fragment(run_every=…)`, `st.segmented_control`, `st.download_button`, `st.html` and
  `streamlit.testing.v1.AppTest` are all available.

## Decisions (confirm before build)
- D1 Streamlit 1.66.1, pinned in `requirements.txt`.
- D2 Inference runs in **one background worker thread** (`motionman/ui/runner.py`), one job at a time
  (INV-1). The page polls job state with `st.fragment(run_every=0.5 s)`. Reruns never touch the job.
  Completed artifact paths live in `st.session_state`, so switching views or downloading never re-runs inference.
- D3 Styling is theme config first. One small scoped CSS block (`st.html`) for what config can't do:
  primary button text `#0B1105`, lime hover/pressed, 2 px lime `:focus-visible` outline with 2 px offset,
  mono tabular figures in metrics. Selectors limited to `data-testid` attributes; nothing else in the DOM.
- D4 `pipeline.run` gains an optional `on_progress(processed, total_or_None, elapsed_s)` callback, called
  at most every 0.25 s. The CLI passes nothing, so its behaviour and timings are unchanged.
- D5 `pipeline.run` also records non-`Exception` exits (`StopException`, `SystemExit`) with their type
  name instead of `error: null`.
- D6 Each run gets a new directory `runs/ui/<UTC timestamp>/` (uploaded source copied in, outputs in
  `out/`), so INV-5 holds without asking the user for a path.
- D7 Demo clip = `clips/crosswalk.mp4` when present; otherwise the button is disabled with the README hint.
- D8 Fonts: `Inter`, then the system sans stack; `Geist Mono`, then `ui-monospace, Menlo`. No font files committed.
- D9 Confidence slider 0.05–0.95, step 0.05, default 0.25. Model fixed to YOLO11n and shown as metadata.

## Steps
1. [ ] Pin Streamlit; `.streamlit/config.toml` theme; `scripts/ui.sh`; empty page loads (health 200).
2. [ ] `pipeline.run`: `on_progress` callback (D4) and BaseException status (D5), with tests.
3. [ ] `ui/runner.py`: worker thread, job states idle/processing/complete/error, run dirs, duplicate-run
   guard, and errors mapped to a short message plus next step. Tests use the fake tracker.
4. [ ] `ui/app.py`: header (wordmark, subtitle, actual device, state badge with text), ~280 px sidebar
   controls, Original/Tracked workspace, metrics strip, exports, collapsed Run details. No inference code.
5. [ ] CSS block (D3) and overlay checks: lime only on Run tracking, focus, selection and progress.
6. [ ] AppTest tests for the states (Pending/N/A, processing disables Run, error clears old metrics);
   all in `scripts/check.sh`.
7. [ ] Browser verification (U2–U10), screenshots; README UI section; log.

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| U1 | Gate passes offline, UI tests included, CLI tests unchanged | `scripts/check.sh` | |
| U2 | App starts | `scripts/ui.sh`; `curl -f localhost:8501/_stcore/health` | |
| U3 | Demo run reaches Complete; every metric equals this run's report.json | browser DOM vs `report.json` | |
| U4 | Before a run: badge Idle, metrics "Pending", playback FPS labelled separately | browser DOM | |
| U5 | During a run: "Processing", frame count and elapsed rise, Run disabled; clicking other widgets doesn't stop it | browser; report `status: success` | |
| U6 | Tracked video plays in the browser | `<video>` readyState ≥ 2, duration ≈ 56.5 s | |
| U7 | Original/Tracked switch and downloads never re-run inference | run count and report mtime unchanged | |
| U8 | Bad upload or `CUDA` → short error plus next step; no stale Complete badge or metrics | browser DOM | |
| U9 | The 3 downloads serve the run's files | downloaded bytes == files on disk | |
| U10 | Run button text `#0B1105` on `#A3E635`; visible 2 px lime focus ring; unavailable latency shows N/A | computed styles, keyboard Tab | |

## Risks
- Streamlit's `data-testid` names can change between versions: pinned (D1), and the CSS is under 30 lines.
- A large upload is copied to disk once (81 MB for the demo clip); `server.maxUploadSize` stays 200 MB.
- The worker thread outlives a closed tab; acceptable locally, and only one job can run at a time.
- Original-view playback depends on the uploaded file's codec; non-H.264 uploads may not preview (Tracked always does).

## Log
- 2026-10-31 plan: spike in a scratch venv (Streamlit 1.66.1): primary text measured white on lime; mid-run click → StopException at step 21/60.
