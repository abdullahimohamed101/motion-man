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

## Invariants (never break without an ADR)
<!-- 3–8 project-specific rules. Delete this comment once filled. -->
- INV-1 …

## Commands
<!-- Fill as the stack lands. Every command must be quiet by default. -->
- Focused test: `…`  (one file/package; use this while iterating)
- Gate: `scripts/check.sh` (lint, types, tests; prints failures only). Run before `ship`.

## Git
Branch `feat/<slug>` off `main`. Commit per step. Push / PR / merge only when asked.

## Map (read on demand, never preload)
- `ARCHITECTURE.md`: modules and dependency direction (≤150 lines)
- `docs/spec.md`: product spec; grep for the `§N` you need, never read whole
- `docs/decisions/INDEX.md`: one line per ADR; open an ADR only for the area you touch
- `docs/plans/`: active plans; `docs/plans/done/` is archive, ignore unless asked
