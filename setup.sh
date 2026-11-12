#!/usr/bin/env bash
# Usage: ./setup.sh [kokoro] [dia]   - creates engines/<name>/.venv (needs uv: https://docs.astral.sh/uv/)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
for engine in "${@:-kokoro}"; do
  case "$engine" in
    kokoro)
      cd "$ROOT/engines/kokoro"
      uv venv --python 3.12 .venv
      VIRTUAL_ENV="$PWD/.venv" uv pip install "kokoro>=0.9.4" soundfile \
        'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
      ;;
    dia)
      cd "$ROOT/engines/dia"
      uv venv --python 3.12 .venv
      VIRTUAL_ENV="$PWD/.venv" uv pip install "mlx-audio==0.6.5" soundfile
      ;;
    *) echo "no setup recipe for $engine (see README)" >&2; exit 1 ;;
  esac
done
