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
- Active plan: `docs/plans/p2-measure.md` (Draft; D1–D12 awaiting owner confirmation). Branch `feat/p2-measure`.
- Env: `.venv` (py3.11.17); `yolo11n.pt`, `clips/`, `data/` local and gitignored. psutil 7.2.2 present via
  Ultralytics (to be pinned in step 3).
- Known: `data/runs/r_20261101T060711Z_a0b0a2` is an early-P0 run (input stored as `source.mp4`); listed but
  refused on open. Safe to delete locally.

## Next step (exact)
1. Owner confirms P2 decisions D1–D12, then P2 step 1: stage timings in the pipeline, `timings.jsonl`,
   report timing fields and warmup policy.

## Blocked / needs owner
- P2 decisions D1–D12 (plan) need confirmation.

## Last check
- `scripts/check.sh`: ruff ok, 294 passed; CI green on main (PR #7).
- Benchmarks (CLI, crosswalk, M5 Pro CPU): median 44.29 fps on mains, 32.14 on battery (README tables).
