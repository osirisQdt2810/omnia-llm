#!/usr/bin/env bash
# Start the gateway in the foreground. systemd (see omnia-llm.service) is what normally runs it.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/hf-cache}"
exec python -m uvicorn gateway.app:app --host 127.0.0.1 --port "${1:-8721}" --log-level info
