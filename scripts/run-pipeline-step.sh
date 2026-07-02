#!/usr/bin/env bash
# Run one pipeline stage for n8n — prints JSON to stdout.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAGE="${1:?stage required: script|voice|music|images|clips|video|subtitles|audio_mix|final}"
RUN_ID="${2:-}"
LANG="${3:-}"
THEME="${4:-}"
DURATION="${5:-45}"
TIER="${6:-flux}"
THEMES_CSV="${7:-}"

export TOKENIZERS_PARALLELISM=false
PYTHON="$ROOT/flux-venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo '{"error": "flux-venv not found — run: python3.12 -m venv flux-venv && pip install -r requirements.txt"}' >&2
  exit 1
fi

CMD=("$PYTHON" -m src.pipeline --stage "$STAGE")

if [[ -n "$RUN_ID" ]]; then
  CMD+=(--from-run "$ROOT/output/$RUN_ID")
fi

if [[ "$STAGE" == "script" ]]; then
  if [[ "$LANG" != "en" && "$LANG" != "hi" ]]; then
    echo '{"error": "language must be en or hi"}' >&2
    exit 1
  fi
  CMD+=(--lang "$LANG" --duration "$DURATION" --tier "$TIER")
  if [[ -n "$THEME" && "$THEME" != "auto" ]]; then
    CMD+=(--theme "$THEME")
  fi
  if [[ -n "$THEMES_CSV" ]]; then
    CMD+=(--themes-csv "$THEMES_CSV")
  fi
fi

exec "${CMD[@]}"
