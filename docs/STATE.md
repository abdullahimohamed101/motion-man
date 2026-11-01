# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- MVP done on `main` (CLI #1, UI #4, CI #3, handoff docs #5). Owner chose **Release A** (P0–P2) from the
  master design (`docs/reference/master-design.txt`, condensed in `docs/spec-a.md`).
- Active plan: `docs/plans/p0-contracts.md` (Active; D1–D8 confirmed; steps 1–5 done; P0-2/3/7/8 PASS). Branch `feat/p0-contracts`
  (local, not pushed).
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/` local and gitignored; future data root `./data` gitignored.

## Next step (exact)
1. P0 step 6: audit failure tests against MD§25 P0.4 (invalid input, empty detections, writer failure, interruption, lease,
   limits, source modified); fill gaps. Then step 7 baseline record (P0-4/5/6), step 8 gate + README + revised estimate.
2. After the P0 gate: write `docs/plans/p1-inspect.md` (A§8), then P2 (A§9). Stop at each gate.

## Blocked / needs owner
- none

## Last check
- `scripts/check.sh`: ruff ok, 191 passed; CI green on main.
- Benchmarks (CLI, crosswalk, M5 Pro CPU): median 44.29 fps on mains, 32.14 on battery (README tables).
