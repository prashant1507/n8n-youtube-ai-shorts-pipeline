#!/usr/bin/env bash
# Wrapper for n8n — runs the narration video pipeline and prints JSON to stdout.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LANG="${1:?language required: en or hi}"
THEME="${2:-}"
DURATION="${3:-45}"
TIER="${4:-flux}"
THEMES_CSV="${5:-}"

if [[ "$LANG" != "en" && "$LANG" != "hi" ]]; then
  echo '{"error": "language must be en or hi"}' >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false

# Full pipeline uses flux-venv (TTS + optional wan via mlxgen subprocess)
# shellcheck source=/dev/null
source flux-venv/bin/activate

CMD=(python -m src.pipeline --lang "$LANG" --duration "$DURATION" --tier "$TIER")
if [[ -n "$THEME" && "$THEME" != "auto" ]]; then
  CMD+=(--theme "$THEME")
fi
if [[ -n "$THEMES_CSV" ]]; then
  CMD+=(--themes-csv "$THEMES_CSV")
fi

exec "${CMD[@]}"
