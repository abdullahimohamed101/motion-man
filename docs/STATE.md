# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- MVP done on `main` (CLI #1, UI #4, CI #3, handoff docs #5). Owner chose **Release A** (P0–P2) from the
  master design (`docs/reference/master-design.txt`, condensed in `docs/spec-a.md`).
- Active plan: `docs/plans/p0-contracts.md` (Active; steps 1–7 done; P0-1..8 all PASS). Branch `feat/p0-contracts`
  (local, not pushed).
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/` local and gitignored; future data root `./data` gitignored.

## Next step (exact)
1. P0 step 8: gate decision + known limitations + revised P1/P2 estimate in the plan; README (two modes, data root,
   manifest, limits, exit 75); then verify (reviewer) and ship P0, and stop for the owner's P1 go-ahead.
2. After the P0 gate: write `docs/plans/p1-inspect.md` (A§8), then P2 (A§9). Stop at each gate.

## Blocked / needs owner
- none

## Last check
- `scripts/check.sh`: ruff ok, 200 passed; CI green on main.
- Benchmarks (CLI, crosswalk, M5 Pro CPU): median 44.29 fps on mains, 32.14 on battery (README tables).
