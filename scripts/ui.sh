#!/usr/bin/env bash
# Start the local MotionMan UI on http://localhost:8501 (spec §9).
set -euo pipefail
cd "$(dirname "$0")/.."
# Run from the repo root so .streamlit/config.toml applies and `motionman` imports resolve.
exec env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/streamlit run motionman/ui/app.py "$@"
