---
name: build
description: Implement the active plan step by step, with tests, on its feature branch.
---

1. Work the next unchecked step in the plan. Read existing patterns before adding abstractions.
2. Tests ship in the same commit as the code.
3. While iterating, run the **focused** test for what you touched, not the gate.
4. Tick the step and add one line to the plan's `## Log` (date, step, commit, result).
5. New dependency? Say why the stdlib or an existing dependency is not enough.
6. Repeat. When all steps are done, hand off to `verify`.
