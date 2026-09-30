#!/usr/bin/env bash
# diffusers is pure Python and rides the torch already pinned for CUDA 12.6 — no second stack.
set -euo pipefail
cd ~/workspaces/omnia-llm
source .venv/bin/activate
pip install -q 'diffusers>=0.31' transformers accelerate safetensors pillow
python -c 'import diffusers, transformers; print("diffusers", diffusers.__version__, "| transformers", transformers.__version__)'
