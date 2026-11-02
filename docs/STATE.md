# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- Release A complete (`docs/spec-a.md`), merged (PRs #6–#9). UI redesign merged (PRs #10, #11); rules in
  `docs/design/ui.md`.
- **Release B part 1** (`docs/spec-b.md`). **P3 zones and events done**, merged PR #12
  (`docs/plans/done/p3-zones.md`, P3-1..P3-10 pass): `regions`, `analytics`, `analysis`, `ui/zones`, `analyze` CLI.
- **P4 review and evaluation**: `docs/plans/p4-review.md` (Active) on `feat/p4-review`; steps 1–7 built and
  committed (datasets, evaluator, run comparison, review queue, CLI, Quality lab Review / Evaluate / Compare runs).
  P4-2..P4-8 pass. Not pushed. **Step 8 is blocked on the owner's review** of 45 crosswalk frames.
- Env: `.venv` (py3.11.17); `yolo11n.pt`, `clips/`, `data/` local and gitignored. psutil 7.2.2 pinned.
- Known: `data/runs/r_20261101T060711Z_a0b0a2` is an early-P0 run; listed but refused on open. Safe to delete locally.

## Next step (exact)
1. Owner: `scripts/ui.sh` → open the latest crosswalk run → Quality lab → Review → New dataset `crosswalk-eval`
   (every 30th frame) → review each frame (classes complete, Accept / add / fix / ignore / exclude) → Commit as eval.
2. Then step 8: evaluate runs `ed3f02` (640) and `6057dc` (480), disagreement, synthetic regression, docs, acceptance.

## Blocked / needs owner
- The 45-frame review session (step 8): only a person can create ground truth (INV-7).
- Side request: portfolio website update; waiting for its folder/repo and what to change.

## Last check
- `scripts/check.sh` on `feat/p4-review`: ruff ok, 519 passed.
- Crosswalk P3 demo (run ed3f02): 22 crossings, 7 entries, peak 9, analyzed in 0.46 s.
- Benchmarks (crosswalk, M5 Pro CPU): P2 `benchmark` on battery 45.86 fps (imgsz 640), 54.21 (imgsz 480).
