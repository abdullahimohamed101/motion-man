# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- Release A complete (`docs/spec-a.md`), merged (PRs #6–#9). UI redesign merged (PRs #10, #11); rules in
  `docs/design/ui.md`.
- **Release B part 1** (`docs/spec-b.md`, P3 then P4). **P3 zones and events built** on `feat/p3-zones`
  (`docs/plans/p3-zones.md`, steps 1–8 done, P3-2..P3-10 pass, P3-1 CI pending). Not pushed, no PR yet.
  - Modules: `regions.py` (region sets), `analytics.py` (pure rules + heatmap), `analysis.py` (deterministic
    published results, CSV/JSON), `ui/zones.py` (Events & zones tab), `analyze` CLI, motion arrows in Inspect.
  - Data: `data/regions/<input>/rs_*.json`, `data/analyses/<run>/a_*/`.
  - Known limits (README): box-centre artefact when a box changes shape; observed stays split by detection
    flicker; heatmap counts every class.
- Env: `.venv` (py3.11.17); `yolo11n.pt`, `clips/`, `data/` local and gitignored. psutil 7.2.2 pinned.
- Known: `data/runs/r_20261101T060711Z_a0b0a2` is an early-P0 run; listed but refused on open. Safe to delete locally.

## Next step (exact)
1. P3 verified (reviewer: 13 findings fixed). Ship as a PR when the owner asks (P3-1 CI).
2. Then stop for the owner's go-ahead on P4 (plan `docs/plans/p4-review.md`). P4 needs the owner to review
   about 30–60 crosswalk frames in the review form.

## Blocked / needs owner
- Go-ahead for the P3 PR; P4 start after that.
- Side request: portfolio website update; waiting for its folder/repo and what to change.

## Last check
- `scripts/check.sh` on `feat/p3-zones`: ruff ok, 462 passed.
- Crosswalk P3 demo (run ed3f02): 22 crossings, 7 entries, peak 9, analyzed in 0.46 s.
- Benchmarks (crosswalk, M5 Pro CPU): P2 `benchmark` on battery 45.86 fps (imgsz 640), 54.21 (imgsz 480).
