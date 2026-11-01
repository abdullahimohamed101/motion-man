# Architecture
Status: slice + trails built (no JSONL or report yet). Contracts live in `docs/spec.md` §4–§8.

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
ui/app.py (spec §9, later) ──▶ the same pipeline.run(config)
```

## Dependency direction
`cli`, `ui` → `pipeline` → `video_io`, `tracking`, `render`, `metrics` → `types` (Track, Frame, RunConfig).
Leaf modules never import each other or `pipeline`. Only `tracking.py` imports Ultralytics;
only `video_io.py` and `render.py` import OpenCV.

## Seams for tests
`pipeline.run` takes a tracker factory, so tests inject a fake tracker and a synthetic
OpenCV-written clip: no weights, network or GPU (INV-8).

## Layout
```text
motionman/  __init__ __main__ cli pipeline video_io tracking render metrics types
tests/      unit tests per module + test_pipeline (synthetic video, fake tracker)
scripts/    check.sh
```
