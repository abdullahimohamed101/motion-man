# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-10-31

## Now
- Active plan: `docs/plans/ui.md` (Active; steps 1–6 done, U1–U2 PASS, U10 partial). CLI MVP in review: PR #1, `docs/plans/done/cli-mvp.md`.
- Branch: `feat/ui`, stacked on `feat/cli-mvp` (PR #1, open). `feat/project-docs` is superseded.
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/crosswalk.mp4` local and gitignored (README has the commands)

## Next step (exact)
1. UI step 7: browser verification U3–U10 (demo run vs report, clicks mid-run, Tracked playback, view switch/downloads
   don't rerun, bad upload + CUDA errors, N/A on a ≤10-frame clip, download bytes), README UI section, log.
2. Owner reviews and merges PR #1 (never merged by an agent); rebase `feat/ui` on `main` afterwards.

## Blocked / needs owner
- PR #1 review and merge

## Last check
- `scripts/check.sh`: ruff ok, 127 passed. Crosswalk benchmarks: exit 0 ×3, median 32.14 fps (CPU, battery).
