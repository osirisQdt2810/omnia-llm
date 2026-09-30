#!/usr/bin/env bash
# Run the gateway in the foreground with one config. systemd (deploy/systemd) runs this too.
#   scripts/run.sh configs/nvidia-24gb.toml
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-${OMNIA_LLM_CONFIG:?usage: scripts/run.sh configs/<name>.toml}}"
exec "${VENV:-.venv}/bin/omnia-llm" serve -c "$CONFIG"
