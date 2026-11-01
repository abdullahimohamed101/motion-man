# Architecture
Status: MVP (CLI + UI), Release A P0–P2 (contracts, inspection, measurement) and Release B P3 (events & zones).
Contracts: `docs/spec.md` §4–§8, `docs/spec-a.md`, `docs/spec-b.md`; plans in `docs/plans/` and `docs/plans/done/`.

## Flow (one synchronous loop, spec §4)
```text
cli.py ──RunConfig──▶ pipeline.run(config)
                        │  for each frame (video_io.read_frames):
                        │    tracks = tracker.update(frame)        tracking.py
                        │    trails.update(frame_index, tracks)    render.py
                        │    out = render.annotate(frame, tracks, trails)
                        │    writer.write(out); jsonl.write(record) video_io.py
                        │    metrics.record(timings)               metrics.py
                        └─ finally: release reader/writer, close JSONL, write report.json
ui/app.py ──▶ ui/runner.py (worker thread) ──▶ the same pipeline.run(config, on_progress=…)
```

## Dependency direction
`cli`, `ui` → `pipeline` → `video_io`, `tracking`, `render`, `metrics`, `records`, `store`, `telemetry` → `types`.
Tools beside the run (P1/P2): `runs`, `inspection`, `presets` (inspect published runs; never infer), and
`benchmark` → `pipeline`, `compare`, `doctor`, `bundle` (measure and diagnose), all reached from `cli` subcommands
(`benchmark`, `compare`, `doctor`, `bundle`) or the UI. Events & zones (P3): `analysis` → `analytics`, `regions`,
`records`, `runs`, `store` (reads verified runs only; never infers), reached from the `analyze` subcommand and
`ui/zones`. `analytics` is pure (no I/O). Leaf modules never import `pipeline`.
Only `tracking.py` imports Ultralytics (and `doctor`'s device probe imports torch lazily); OpenCV is used by
`video_io`, `render`, `inspection` and `doctor`.

## Seams for tests
`pipeline.run` takes a tracker factory, so tests inject a fake tracker and a synthetic
OpenCV-written clip: no weights, network or GPU (INV-8).

## Layout
```text
motionman/  __init__ __main__ cli pipeline video_io tracking render metrics types records store
            runs inspection presets telemetry benchmark compare doctor bundle regions analytics analysis
  ui/      app (Streamlit layout) runner (worker thread) view (display rules) workspace (per-run context) style (CSS)
           zones (Events & zones tab)
tests/      unit tests per module + test_pipeline (synthetic video, fake tracker); conftest enforces INV-8
scripts/    check.sh ui.sh
data/       inputs/ staging/ runs/ derived/ exports/ presets/ benchmarks/ comparisons/ regions/ analyses/ (gitignored)
```
