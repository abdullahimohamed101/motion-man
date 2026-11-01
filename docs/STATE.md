# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- MVP done on `main` (CLI #1, UI #4, CI #3, handoff docs #5). Owner chose **Release A** (P0–P2) from the
  master design (`docs/reference/master-design.txt`, condensed in `docs/spec-a.md`).
- Active plan: `docs/plans/p0-contracts.md` (Draft; decisions D1–D8 await owner). Branch `feat/p0-contracts`
  (local, not pushed).
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/` local and gitignored; future data root `./data` gitignored.

## Next step (exact)
1. Owner confirms P0 decisions D1–D8, then `build` P0 step 1 (`motionman/records.py`, v1.0 records + legacy reader).
2. After the P0 gate: write `docs/plans/p1-inspect.md` (A§8), then P2 (A§9). Stop at each gate.

## Blocked / needs owner
- P0 decisions D1–D8.

## Last check
- `scripts/check.sh`: ruff ok, 132 passed; CI green on main.
- Benchmarks (CLI, crosswalk, M5 Pro CPU): median 44.29 fps on mains, 32.14 on battery (README tables).
