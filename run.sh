#!/usr/bin/env bash
# Usage: ./run.sh <transcript.txt> [engine ...]   (default engine: kokoro)
# Writes out/<transcript-name>.<engine>.mp3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 1 ]] || { echo "usage: $0 <transcript.txt> [kokoro ...]" >&2; exit 2; }
TRANSCRIPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
shift
ENGINES=("${@:-kokoro}")
NAME="$(basename "${TRANSCRIPT%.*}")"

# Keep every model download inside this folder.
export HF_HOME="$ROOT/models/hf"
export TORCH_HOME="$ROOT/models/torch"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_DISABLE_TELEMETRY=1
unset VIRTUAL_ENV  # never let an engine install into a caller's venv

command -v ffmpeg >/dev/null || { echo "ffmpeg not found (brew install ffmpeg)" >&2; exit 1; }

for engine in "${ENGINES[@]}"; do
  py="$ROOT/engines/$engine/.venv/bin/python"
  [[ -x "$py" ]] || { echo "$engine: venv missing, run ./setup.sh $engine" >&2; exit 1; }
  out="$ROOT/out/$NAME.$engine.mp3"
  echo "== $engine"
  start=$(date +%s)
  "$py" "$ROOT/engines/$engine/generate.py" "$TRANSCRIPT" "$out"
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out")
  echo "$engine: audio ${dur%.*}s, wall $(( $(date +%s) - start ))s -> $out"
done
