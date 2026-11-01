# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-10-31

## Now
- Active plan: `docs/plans/cli-mvp.md` (Active; steps 1–5 done; A5–A8 pass)
- Branch: `feat/cli-mvp`, stacked on `feat/project-docs` (neither pushed nor merged)
- Env: `.venv` (py3.11.17) from `requirements-dev.txt`; `yolo11n.pt` in repo root (gitignored)

## Next step (exact)
1. Plan step 6: pipeline integration tests (synthetic OpenCV clip + fake tracker: frame counts, empty frames, missing IDs,
   cleanup on failure, no-frames input) and `scripts/check.sh` (ruff + pytest, failures only).
2. When the real clip arrives: run the slice on it and confirm playback (finishes step 2's real-clip check).

## Blocked / needs owner
- Demo clip: 30–60 s people/vehicles, reuse rights (the local Valorant MP4 doesn't fit spec §1).

## Last check
- `pytest -q`: 42 passed (0.6 s, offline). A5–A8 PASS on synthetic clips. ruff clean.
