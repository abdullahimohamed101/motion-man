# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-10-31

## Now
- Active plan: `docs/plans/cli-mvp.md` (Active; steps 1–6 done; A1, A5–A8 pass)
- Branch: `feat/cli-mvp`, stacked on `feat/project-docs` (neither pushed nor merged)
- Env: `.venv` (py3.11.17) from `requirements-dev.txt`; `yolo11n.pt` in repo root (gitignored)

## Next step (exact)
1. Plan step 7 (needs the demo clip): real-model run + visual review (A2–A4), 3 benchmark runs + median (A9),
   README with setup/run/test, module roles, real numbers + hardware, limitations, clip attribution; fresh-env check (A10).

## Blocked / needs owner
- Demo clip: 30–60 s people/vehicles, reuse rights (the local Valorant MP4 doesn't fit spec §1).

## Last check
- `scripts/check.sh`: ruff ok, 71 passed (0.45 s, offline, network blocked, torch never imported).
