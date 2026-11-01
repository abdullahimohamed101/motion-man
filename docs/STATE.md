# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-10-31

## Now
- Active plan: `docs/plans/cli-mvp.md`: all 7 steps done, A1–A10 PASS. Not yet verified by the reviewer or shipped.
- Branch: `feat/cli-mvp`, stacked on `feat/project-docs` (neither pushed nor merged)
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/crosswalk.mp4` local and gitignored (README has the commands)

## Next step (exact)
1. `verify` skill via the `reviewer` subagent on `main...feat/cli-mvp`; fix any P0/P1.
2. `ship`: move the plan to `docs/plans/done/`, draft the PR text, ask before pushing.
3. Then write the UI plan (spec §9).

## Blocked / needs owner
- none

## Last check
- `scripts/check.sh`: ruff ok, 71 passed. Crosswalk demo + 3 benchmarks exit 0 (median 25.95 fps, CPU, on battery).
