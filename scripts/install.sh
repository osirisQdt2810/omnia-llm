#!/usr/bin/env bash
# Install omnia-llm and ONE platform's engines into a virtualenv.
#
#   scripts/install.sh apple     # Apple Silicon: MLX
#   scripts/install.sh nvidia    # NVIDIA: vLLM + diffusers (pinned for a CUDA 12.6 driver)
#   scripts/install.sh cpu       # the gateway only; bring llama.cpp's `llama-server` yourself
#
# Environment: PYTHON (default python3), VENV (default .venv).
set -euo pipefail
cd "$(dirname "$0")/.."
PLATFORM="${1:?usage: scripts/install.sh apple|nvidia|cpu}"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

[ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel

case "$PLATFORM" in
  apple)
    "$VENV/bin/pip" install -q -e ".[apple,dev]"
    ;;
  nvidia)
    # The reference box's driver is CUDA 12.6 with no root to upgrade it, which pins the stack:
    # vLLM 0.16.0 -> torch 2.9.1 -> cu126 (0.17+ needs torch 2.10+, which has no cu126 build).
    # torch goes in FIRST from the cu126 index, or pip resolves a cu13 wheel that imports fine
    # and dies at init_device. diffusers>=0.40 needs huggingface-hub>=1.23, which forces
    # transformers>=5, which vLLM 0.16 refuses — hence 0.35.1 / 4.57 / 0.36.
    "$VENV/bin/pip" install -q torch==2.9.1 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126
    "$VENV/bin/pip" install -q vllm==0.16.0 "diffusers==0.35.1" "transformers==4.57.*" \
        "huggingface-hub==0.36.*" accelerate safetensors pillow
    "$VENV/bin/pip" install -q -e ".[dev]"
    ;;
  cpu)
    "$VENV/bin/pip" install -q -e ".[dev]"
    command -v llama-server >/dev/null \
        || echo "note: put llama.cpp's llama-server on PATH (a release, or: brew install llama.cpp)"
    ;;
  *)
    echo "unknown platform: $PLATFORM (apple|nvidia|cpu)" >&2
    exit 2
    ;;
esac
echo "installed into $VENV — next: $VENV/bin/omnia-llm devices"
