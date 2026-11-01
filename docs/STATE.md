# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- **Release A complete** (`docs/spec-a.md`, P0–P2), all merged to `main`:
  - P0 contracts (`docs/plans/done/p0-contracts.md`), PR #6.
  - P1 inspect (`docs/plans/done/p1-inspect.md`), PR #7.
  - P2 measure (`docs/plans/done/p2-measure.md`, P2-1..P2-10 PASS), PR #8: stage timings, telemetry,
    `benchmark`, `compare`, `doctor`, `bundle`.
- No active plan. Anything past Release A (P3 zones/events/heatmaps, P4 review/eval, P5 experiments, P6 job
  queue/exports, P7 masks/webcam, P8 TensorRT; A§1) is out of scope until the owner opens a new release spec.
- Env: `.venv` (py3.11.17); `yolo11n.pt`, `clips/`, `data/` local and gitignored. psutil 7.2.2 pinned.
- Known: `data/runs/r_20261101T060711Z_a0b0a2` is an early-P0 run; listed but refused on open. Safe to delete locally.

## Next step (exact)
1. Owner decides what follows Release A (a Release B spec, or hardening issues via `gh issue list -l bug`).

## Blocked / needs owner
- Scope for the next release.

## Last check
- `scripts/check.sh`: ruff ok, 360 passed; CI green on PR #8.
- Benchmarks (crosswalk, M5 Pro CPU): MVP median 44.29 fps mains / 32.14 battery; P2 `benchmark` on battery 45.86
  (imgsz 640) and 54.21 (imgsz 480), controlled +18.2%.
