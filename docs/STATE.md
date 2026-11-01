# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- MVP done on `main`. Release A (P0–P2, `docs/spec-a.md`) in progress.
- **P0 done** (`docs/plans/done/p0-contracts.md`, gate PASS after review fixes): v1.0 records, content-hash identities,
  resolved RunSpec, CFR + limits, data root with staging → validate → manifest → atomic publish, lease, UI on the
  data root. Merged as PR #6 (CI: 212 passed, 2 real-media tests skipped: no ffmpeg on the runner).
- Baseline: `data/runs/r_20261101T062222Z_2ad4d6` at clean commit `8f9ec5e` (1,356 frames, 44.67 fps, mains).
- Active plan: `docs/plans/p1-inspect.md` (Active; D1–D10 confirmed; steps 0–4 done). Branch `feat/p1-inspect` (local).
- Env: `.venv` (py3.11.17); `yolo11n.pt`, `clips/`, `data/` local and gitignored.

## Next step (exact)
1. P1 step 5: `motionman/presets.py` (D7): data/presets/<name>.json with separate view and tracking sections;
   name validation; view applies now, tracking only to the next run. Then step 6, the UI restructure.

## Blocked / needs owner
- none

## Last check
- `scripts/check.sh`: ruff ok, 247 passed; CI green on main.
- Benchmarks (CLI, crosswalk, M5 Pro CPU): median 44.29 fps on mains, 32.14 on battery (README tables).
