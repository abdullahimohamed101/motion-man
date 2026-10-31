---
name: ship
description: Close out a verified plan and prepare the PR text. Never pushes or opens a PR without approval.
disable-model-invocation: true
---

1. Confirm `verify` left no open P0/P1 and the Acceptance table is filled.
2. Set plan Status: Done; move it to `docs/plans/done/`.
3. Rewrite `docs/STATE.md`. Update `ARCHITECTURE.md` only if module boundaries changed.
4. Draft the PR body: Summary, Why, Testing (commands + results), Risks, Rollback. Link plan and ADRs.
5. Ask before pushing or opening the PR.
