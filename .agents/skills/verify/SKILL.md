---
name: verify
description: Independently verify and review a finished change against its plan, then simplify. Combines verification, code review and hardening in one pass.
---

Prefer running this in the `reviewer` subagent so the diff reading stays out of the main context.

1. Run `scripts/check.sh`. Fill the plan's Acceptance table: one-line result per row,
   `PASS` only with observed output, otherwise `FAIL` or `UNVERIFIED (<why>)`.
2. Read `git diff main...HEAD`. Check correctness, edge cases, invariants, security,
   needless complexity, dead code.
3. UI changes: load the page in a browser and check the console.
4. Report findings as P0/P1/P2, each one line: `file:line: problem → fix`.
   Fix P0/P1 now. File P2s as GitHub issues; do not write a review document.
