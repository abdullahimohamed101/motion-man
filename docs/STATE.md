# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- **Release A complete** (`docs/spec-a.md`, P0–P2), merged (PRs #6, #7, #8). UI redesign merged (PRs #10, #11):
  top bar = brand + Library menu + New video; title once with status; bone primary; lime as a signal only.
  Rules in `docs/design/ui.md`.
- **Release B part 1 spec approved** (`docs/spec-b.md`, P3 then P4; owner reviews the P4 frames; MotionMan JSON
  import only). Active plan: `docs/plans/p3-zones.md` (Draft, D1–D13, 8 steps) on `feat/p3-zones`.
- Env: `.venv` (py3.11.17); `yolo11n.pt`, `clips/`, `data/` local and gitignored. psutil 7.2.2 pinned.
- Known: `data/runs/r_20261101T060711Z_a0b0a2` is an early-P0 run; listed but refused on open. Safe to delete locally.

## Next step (exact)
1. Owner approves the P3 plan (D1–D13). Then build step 1 (`regions.py`).

## Blocked / needs owner
- P3 plan approval.

## Last check
- `scripts/check.sh` on `main` after PR #11: ruff ok, 362 passed; CI green on PR #11.
- Benchmarks (crosswalk, M5 Pro CPU): MVP median 44.29 fps mains / 32.14 battery; P2 `benchmark` on battery 45.86
  (imgsz 640) and 54.21 (imgsz 480), controlled +18.2%.
