# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- MVP done on `main` (CLI #1, UI #4, CI #3, handoff docs #5). Owner chose **Release A** (P0–P2) from the
  master design (`docs/reference/master-design.txt`, condensed in `docs/spec-a.md`).
- Active plan: `docs/plans/p0-contracts.md` (Active; all 8 steps done; gate PASS; not yet reviewed or shipped). Branch `feat/p0-contracts`
  (local, not pushed).
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/` local and gitignored; future data root `./data` gitignored.

## Next step (exact)
1. `verify` P0 with the reviewer subagent (`main...feat/p0-contracts`); fix P0/P1 findings.
2. `ship` P0: archive the plan, PR into main. Then stop for the owner's P1 go-ahead (write `docs/plans/p1-inspect.md`).

## Blocked / needs owner
- none

## Last check
- `scripts/check.sh`: ruff ok, 200 passed; CI green on main.
- Benchmarks (CLI, crosswalk, M5 Pro CPU): median 44.29 fps on mains, 32.14 on battery (README tables).
