# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- MVP done on `main`. Release A (P0–P2, `docs/spec-a.md`) in progress.
- **P0 done** (`docs/plans/done/p0-contracts.md`, gate PASS after review fixes): v1.0 records, content-hash identities,
  resolved RunSpec, CFR + limits, data root with staging → validate → manifest → atomic publish, lease, UI on the
  data root. Shipped as a PR from `feat/p0-contracts` into `main` (awaiting owner review/merge).
- Baseline: `data/runs/r_20261101T062222Z_2ad4d6` at clean commit `8f9ec5e` (1,356 frames, 44.67 fps, mains).
- Env: `.venv` (py3.11.17); `yolo11n.pt`, `clips/`, `data/` local and gitignored.

## Next step (exact)
1. Owner reviews and merges the P0 PR (never merged by an agent).
2. On the owner's go-ahead: write `docs/plans/p1-inspect.md` (A§8: run shelf, sparse index, exact-frame inspector,
   track summaries, evidence PNG, presets, UI2 guided flow) on a new branch off `main`. Estimate 1.5–2.5 focused days.

## Blocked / needs owner
- P0 PR review and merge; go-ahead for P1.

## Last check
- `scripts/check.sh`: ruff ok, 214 passed; CI green on main.
- Benchmarks (CLI, crosswalk, M5 Pro CPU): median 44.29 fps on mains, 32.14 on battery (README tables).
