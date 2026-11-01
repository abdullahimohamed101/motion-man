# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-01

## Now
- **Release A complete** (`docs/spec-a.md`, P0–P2), all merged to `main` (PRs #6, #7, #8). UI redesign merged (PR #10).
- **UI flow fix on `feat/ui-flow`** (not pushed, no PR): the owner rejected the PR #10 header (repeated clip name,
  hash in the run picker, floating status pill, loud lime Export). Design in `docs/design/ui.md`; built in
  `motionman/ui/app.py`, `style.py`, `view.py`, `.streamlit/config.toml`:
  - top bar = brand + **Library** menu (runs as "clip · Today, 7:45 PM · 1,356 frames", legacy filter inside) +
    New video; page header = title once + live status pill + Export (bone primary) + ⋯ details (run ID, folder);
  - acid lime is now a signal only (focus ring, active tab, progress); `primaryColor` is bone `#E8E9E3`.
- No active plan. Anything past Release A (A§1 P3–P8) is out of scope until the owner opens a new release spec.
- Env: `.venv` (py3.11.17); `yolo11n.pt`, `clips/`, `data/` local and gitignored. psutil 7.2.2 pinned.
- Known: `data/runs/r_20261101T060711Z_a0b0a2` is an early-P0 run; listed but refused on open. Safe to delete locally.

## Next step (exact)
1. Owner reviews `feat/ui-flow` in the running app (`scripts/ui.sh`), then asks to push / open the PR.
2. Owner decides what follows Release A (a Release B spec, or hardening issues via `gh issue list -l bug`).

## Blocked / needs owner
- Go-ahead on the `feat/ui-flow` header; scope for the next release.

## Last check
- `scripts/check.sh` on `feat/ui-flow`: ruff ok, 362 passed. CI green on PR #10 (`main`).
- Benchmarks (crosswalk, M5 Pro CPU): MVP median 44.29 fps mains / 32.14 battery; P2 `benchmark` on battery 45.86
  (imgsz 640) and 54.21 (imgsz 480), controlled +18.2%.
