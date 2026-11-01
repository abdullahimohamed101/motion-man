# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-10-31

## Now
- Active plan: `docs/plans/cli-mvp.md` (Active; steps 1–2 done; D4 amended to H.264)
- Branch: `feat/cli-mvp`, stacked on `feat/project-docs` (neither pushed nor merged)
- Env: `.venv` (py3.11.17) from `requirements-dev.txt`; `yolo11n.pt` in repo root (gitignored)

## Next step (exact)
1. Plan step 3: trails in `render.py` (deque ≤30 per ID, expire after 60 unseen frames, no trail for null IDs).
2. When the real clip arrives: run the slice on it and confirm playback (finishes step 2's real-clip check).

## Blocked / needs owner
- Demo clip: 30–60 s people/vehicles, reuse rights (the local Valorant MP4 doesn't fit spec §1).

## Last check
- Slice on synthetic pan clip: 90/90 frames, H.264, exit 0; ruff clean. No tests yet; `scripts/check.sh` is plan step 6.
