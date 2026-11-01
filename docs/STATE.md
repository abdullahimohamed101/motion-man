# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- MVP done on `main`. Release A (P0–P2, `docs/spec-a.md`): P0 and P1 done, P2 planned.
- **P0 done** (`docs/plans/done/p0-contracts.md`): v1.0 records, content-hash identities, RunSpec, CFR + limits,
  staging → validate → manifest → atomic publish, lease. PR #6.
- **P1 done** (`docs/plans/done/p1-inspect.md`, P1-1..P1-10 PASS): run shelf with verified open, derived index
  + track summaries, exact frames, stills + evidence PNG, presets, workbench UI. PR #7 (CI: 294 passed, 0 skipped).
- Baseline: `data/runs/r_20261101T062222Z_2ad4d6` at clean commit `8f9ec5e` (1,356 frames, 44.67 fps, mains).
- Active plan: `docs/plans/p2-measure.md` (Active; D1–D12 confirmed; steps 1–5 done). Branch `feat/p2-measure`.
- Env: `.venv` (py3.11.17); `yolo11n.pt`, `clips/`, `data/` local and gitignored. psutil 7.2.2 pinned.
- Known: `data/runs/r_20261101T060711Z_a0b0a2` is an early-P0 run (input stored as `source.mp4`); listed but
  refused on open. Safe to delete locally.

## Next step (exact)
1. P2 step 6: `python -m motionman doctor` (D7) + input-quality hints (D8): read-only checks, own probes only.

## Blocked / needs owner
- none

## Last check
- `scripts/check.sh`: ruff ok, 329 passed; CI green on main (PR #7).
- Benchmarks (CLI, crosswalk, M5 Pro CPU): median 44.29 fps on mains, 32.14 on battery (README tables).
