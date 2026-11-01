#!/usr/bin/env bash
# The gate: lint, format, offline tests. One line per passing step; full output only on failure.
set -uo pipefail
cd "$(dirname "$0")/.."
bin=.venv/bin
status=0

step() {
  local name=$1; shift
  local out
  if out=$("$@" 2>&1); then
    local summary
    summary=$(printf '%s\n' "$out" | tail -n 1)
    echo "ok   $name${summary:+ ($summary)}"
  else
    echo "FAIL $name"
    printf '%s\n' "$out"
    status=1
  fi
}

step "ruff check"  "$bin/ruff" check -q .
step "ruff format" "$bin/ruff" format -q --check .
step "pytest"      "$bin/pytest" -q -p no:cacheprovider
exit $status
