# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-10-31

## Now
- Active plan: none. CLI MVP shipped for review: `docs/plans/done/cli-mvp.md` (A1–A10 PASS, reviewer findings fixed).
- Branch: `feat/cli-mvp` → PR into `main` (includes the docs commits; `feat/project-docs` is superseded)
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/crosswalk.mp4` local and gitignored (README has the commands)

## Next step (exact)
1. Owner reviews and merges the PR (never merged by an agent).
2. `plan` the local UI (spec §9) as `docs/plans/ui.md`, on a new branch off `main` after the merge.

## Blocked / needs owner
- PR review and merge

## Last check
- `scripts/check.sh`: ruff ok, 78 passed. Crosswalk benchmarks: exit 0 ×3, median 32.14 fps (CPU, battery).
