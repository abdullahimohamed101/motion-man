# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- Active plan: none. Both MVP plans are done and on `main`: CLI (#1, `docs/plans/done/cli-mvp.md`), UI (#4,
  `docs/plans/done/ui.md`), CI (#3). PR #2 merged into the old `feat/cli-mvp` by mistake; #4 re-landed it.
- Merged branches deleted locally and on GitHub; only `main` (+ this docs branch) remain.
- Handoff docs: `docs/interview.md` (5-minute walkthrough + the brief's five questions), `docs/demo-script.md`.
  Silent demo video: `demo-video/motionman-demo.mp4` (local, gitignored, built from real runs).
- Env: `.venv` (py3.11.17); `yolo11n.pt` and `clips/` local and gitignored (README has the commands)

## Next step (exact)
1. Owner records the narrated demo with `docs/demo-script.md` (on mains power).
2. Optional, from spec: decoder prefetch thread (§7, `Queue(maxsize=2)`, blocking puts, stop event);
   CUDA smoke test on a machine with an NVIDIA GPU (hold model, imgsz, FP32 and output constant).

## Blocked / needs owner
- Nothing blocking. CUDA work needs an NVIDIA machine.

## Last check
- `scripts/check.sh`: ruff ok, 132 passed; CI green on #3 and #4.
- Benchmarks (CLI, crosswalk, M5 Pro CPU): median 44.29 fps on mains, 32.14 on battery (README tables).
