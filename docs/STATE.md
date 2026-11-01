# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-10-31

## Now
- Active plan: `docs/plans/cli-mvp.md`: all 7 steps done, A1–A10 PASS, reviewer findings fixed. Not shipped.
- Branch: `feat/cli-mvp`, stacked on `feat/project-docs` (neither pushed nor merged)
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/crosswalk.mp4` local and gitignored (README has the commands)

## Next step (exact)
1. `ship`: move the plan to `docs/plans/done/`, draft the PR text, ask before pushing.
2. Then write the UI plan (spec §9).

## Blocked / needs owner
- none

## Last check
- `scripts/check.sh`: ruff ok, 78 passed. Crosswalk benchmarks after fixes: exit 0 ×3, median 32.14 fps (CPU, battery).
