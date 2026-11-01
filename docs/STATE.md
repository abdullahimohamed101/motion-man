# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- Active plan: none. Both MVP plans shipped for review:
  - CLI: PR #1 `feat/cli-mvp` → `main` (`docs/plans/done/cli-mvp.md`, A1–A10 PASS)
  - UI: PR #2 `feat/ui` → `feat/cli-mvp` (`docs/plans/done/ui.md`, U1–U10 PASS)
- `feat/project-docs` is superseded (never pushed).
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/` local and gitignored (README has the commands)

## Next step (exact)
1. Owner reviews and merges PR #1, then PR #2 (never merged by an agent). After #1 merges, retarget PR #2 to `main`
   (GitHub usually does this when the base branch is deleted; otherwise `gh pr edit 2 --base main`).
2. Optional, from spec: decoder prefetch thread (§7); CUDA smoke test on a machine with an NVIDIA GPU.

## Blocked / needs owner
- PR #1 and PR #2 review and merge

## Last check
- `scripts/check.sh`: ruff ok, 132 passed. UI U1–U10 PASS in browser. Crosswalk benchmarks: exit 0 ×3, median 32.14 fps (CPU, battery).
