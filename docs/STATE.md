# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-04

## Now
- Release A complete (`docs/spec-a.md`), merged (PRs #6–#9). UI redesign merged (PRs #10, #11); rules in
  `docs/design/ui.md`. P3 zones and events merged (PR #12, `docs/plans/done/p3-zones.md`).
- **P4 review and evaluation: built, all 8 steps done** on `feat/p4-review` (`docs/plans/p4-review.md`, Done;
  P4-2..P4-9 pass, P4-1 needs CI on the PR). Not pushed. Reviewer pass not yet run.
- Real eval set: `crosswak-eval@r2` (name has the owner's typo; kept). 46 frames, every box accepted from the 640
  run, none added, ignore band y 500–630 on every frame. So scores are agreement with 640, not true accuracy;
  README Limitations says so. Frame 60 has a known wrong accepted box (merges two people and a car).
- Results: 640 run 1.000/1.000 by construction; 480 run P 0.962, R 0.785 (227/9/62). Evaluator now `detection-v2`
  (reviewed boxes in ignored areas aren't graded).
- Env: `.venv` (py3.11.17); `yolo11n.pt`, `clips/`, `data/` local and gitignored. UI: `scripts/ui.sh`.
- Known: `data/runs/r_20261101T060711Z_a0b0a2` is an early-P0 run; listed but refused on open. Safe to delete locally.

## Next step (exact)
1. Reviewer subagent on `feat/p4-review`, fix findings, then PR when the owner asks.
2. Optional, owner: fix the frame 60 box in the draft (Replaces a53) and commit `r3`; rerun the evaluations.
3. After P4 merges: Release C G0 (`docs/plans/native-g0-g1.md` on `feat/native-plan`, approved).

## After P4: Release C (approved 2026-11-04)
- Native C++/CUDA/TensorRT/RT-DETR engine on the owner's NVIDIA PC. Branch `feat/native-plan` (worktree
  `.worktrees/native-plan`, off `main`, not pushed): `docs/spec-c.md`, ADR-001/002 Accepted, ADR-003/004 Proposed,
  `docs/plans/native-g0-g1.md` (Approved), source `docs/reference/native-architecture.md`.
- Needs owner: PC inventory output (`nvidia-smi`, `nvcc --version`, `cmake --version`, OS, disk) and SSH access;
  yes to `brew install cmake` on the Mac.

## Blocked / needs owner
- Side request: portfolio website update; waiting for its folder/repo and what to change.

## Last check
- `scripts/check.sh` on `feat/p4-review`: ruff ok, 520 passed.
- Crosswalk P3 demo (run ed3f02): 22 crossings, 7 entries, peak 9, analyzed in 0.46 s.
- Benchmarks (crosswalk, M5 Pro CPU): P2 `benchmark` on battery 45.86 fps (imgsz 640), 54.21 (imgsz 480).
