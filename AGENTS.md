# Motion Man: agent guide

Loaded every session. Only rules an agent would not follow by default belong here.
Hard cap: 80 lines. Details go in `docs/`, read on demand.

## Start of session
1. Read `docs/STATE.md` (≤60 lines: current task, next step, blockers). Nothing else up front.
2. If STATE names an active plan, read its **Steps** and **Acceptance** sections only.
3. `git status -sb`. Do not run the full test suite to "check the baseline".
4. Read other docs only when the task touches them (see Map).

## End of session
Rewrite `docs/STATE.md` in place (do not append history; git keeps history).

## Workflow
`plan` → `build` → `verify` → `ship` (skills in `.agents/skills/`). Small fixes skip `plan`.
- Plans: `docs/plans/<slug>.md`, ≤150 lines. Progress notes go in the plan's
  `## Log` as one line per step, never pasted command output.
- Done = each acceptance criterion has a command and a one-line result.
- Known issues are GitHub issues (`gh issue list -l bug`), not a markdown file.

## Scope
Two-day Python MVP (spec §1–§2). Anything not in `docs/spec.md` is out of scope: say so, don't build it.

## Invariants (never break without an ADR)
- INV-1 One model instance owns tracker state per run; frames fed in order; no parallel track calls.
- INV-2 Never invent track IDs. Null ID → drawn, no trail.
- INV-3 Exactly one JSONL record per processed frame, streamed to disk.
- INV-4 Bounded memory: trails ≤30 points, expire after 60 unseen frames; no frame/result retention.
- INV-5 Never overwrite the source or a non-empty output dir.
- INV-6 Cleanup in `finally`; a partial run is reported `interrupted`/`error`, never success.
- INV-7 No fabricated numbers: benchmarks, test results and GPU work are real or marked UNVERIFIED.
- INV-8 `pytest` needs no GPU, network or weight download (fake tracker); real-model checks are separate.

## Commands
- Setup: `uv venv -p 3.11 && uv pip install -r requirements-dev.txt`
- Focused test: `.venv/bin/pytest -q tests/test_<module>.py`
- Gate: `scripts/check.sh` (ruff + pytest, failures only). Run before `ship`. CI runs the same gate on every PR
  and push to `main` (`.github/workflows/check.yml`, ubuntu-24.04, CPU-only torch).
- Demo: `.venv/bin/python -m motionman --source <clip>.mp4 --output-dir runs/<name> --device cpu`
- Host is Apple Silicon: no CUDA here. CUDA paths are UNVERIFIED (env) unless run elsewhere.

## Git
Branch `feat/<slug>` off `main`. Commit per step. Push / PR / merge only when asked.

## Map (read on demand, never preload)
- `ARCHITECTURE.md`: modules and dependency direction (≤150 lines)
- `docs/spec.md`: product spec; grep for the `§N` you need, never read whole
- `docs/reference/`: original briefs; open only to settle a spec ambiguity
- `docs/decisions/INDEX.md`: one line per ADR; open an ADR only for the area you touch
- `docs/plans/`: active plans; `docs/plans/done/` is archive, ignore unless asked
