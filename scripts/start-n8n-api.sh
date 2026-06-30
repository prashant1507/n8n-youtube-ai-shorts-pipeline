#!/usr/bin/env bash
# Start the local HTTP API for n8n (keep this running in a terminal or as a service).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export N8N_API_HOST="${N8N_API_HOST:-0.0.0.0}"
export N8N_API_PORT="${N8N_API_PORT:-8765}"
export N8N_STEP_MAX_TRIES="${N8N_STEP_MAX_TRIES:-3}"
export N8N_STEP_RETRY_WAIT_SEC="${N8N_STEP_RETRY_WAIT_SEC:-1800}"

if curl -sf "http://127.0.0.1:${N8N_API_PORT}/health" >/dev/null 2>&1; then
  echo "Pipeline API already running on port ${N8N_API_PORT}."
  echo "Health: http://127.0.0.1:${N8N_API_PORT}/health"
  echo "To restart: kill \$(lsof -ti :${N8N_API_PORT}) && $0"
  exit 0
fi

# API imports pipeline code (needs pyyaml, pydantic, etc.) — use flux-venv, not system python3.
VENV_PYTHON="$ROOT/flux-venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "flux-venv not found at $ROOT/flux-venv" >&2
  echo "Create it: python3.12 -m venv flux-venv && source flux-venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi
exec "$VENV_PYTHON" -m src.n8n_api
