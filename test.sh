#!/usr/bin/env bash
# Runs the unit tests (pytest via uv; no engine venv needed).
set -euo pipefail
cd "$(dirname "$0")"
unset VIRTUAL_ENV
exec uv run --no-project --python 3.12 --with pytest python -m pytest -q tests "$@"
