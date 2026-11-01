# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-10-31

## Now
- Active plan: `docs/plans/ui.md` (Active; all 7 steps done, U1–U10 PASS; not yet reviewed or shipped). CLI MVP in review: PR #1, `docs/plans/done/cli-mvp.md`.
- Branch: `feat/ui`, stacked on `feat/cli-mvp` (PR #1, open). `feat/project-docs` is superseded.
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/crosswalk.mp4` local and gitignored (README has the commands)

## Next step (exact)
1. `verify` the UI branch with the reviewer subagent (`feat/cli-mvp...feat/ui`), fix P0/P1.
2. `ship`: archive `docs/plans/ui.md`, PR `feat/ui` → `main` once PR #1 merges (or stacked on `feat/cli-mvp`).
3. Owner reviews and merges PR #1 (never merged by an agent); rebase `feat/ui` on `main` afterwards.

## Blocked / needs owner
- PR #1 review and merge

## Last check
- `scripts/check.sh`: ruff ok, 127 passed. UI U1–U10 PASS in browser. Crosswalk benchmarks: exit 0 ×3, median 32.14 fps (CPU, battery).
