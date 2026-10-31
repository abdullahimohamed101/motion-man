---
name: plan
description: Plan a non-trivial change before coding. Writes docs/plans/<slug>.md from the template. Skip for small fixes.
---

No production code in this step.

1. Restate the outcome; list non-goals.
2. Read only what the change touches: the relevant code, the cited spec `§N` (grep it),
   ADRs listed in `docs/decisions/INDEX.md` for that area, and open issues
   (`gh issue list -l bug --search "<area>"`).
3. Note invariants (AGENTS.md) the change could break.
4. Copy `docs/plans/TEMPLATE.md`. Fill it in, ≤150 lines. Every acceptance criterion
   must be checkable by a command.
5. A decision a future reader would question gets an ADR (`adr` skill).
6. Set STATE.md "Active plan". Ask the owner only about real blockers.
