# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-10-31

## Now
- Active plan: `docs/plans/cli-mvp.md` (Active; D1–D6 confirmed; step 1 done)
- Branch: `feat/cli-mvp`, stacked on `feat/project-docs` (neither pushed nor merged)
- Env: `.venv` (py3.11.17) from `requirements-dev.txt`; `yolo11n.pt` in repo root (gitignored)

## Next step (exact)
1. Plan step 2: vertical slice (`types`, `video_io`, `tracking`, `pipeline`, `__main__`), boxes + IDs, playable output.
   Can start on a synthetic OpenCV-written clip; the real clip is needed to finish step 2.

## Blocked / needs owner
- Demo clip: 30–60 s people/vehicles, reuse rights (the local Valorant MP4 doesn't fit spec §1).

## Last check
- Smoke test (CPU, bus.jpg): detect + track OK. `scripts/check.sh` does not exist yet (plan step 6).
