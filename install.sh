#!/usr/bin/env bash
# Driver on this box is CUDA 12.6 and we have no root to change it, so the whole stack is
# pinned to the newest vLLM whose torch still ships a cu126 build:
#   vllm 0.16.0 -> torch 2.9.1 -> cu126  (0.17+ moves to torch 2.10/2.11/2.13, cu128+ only)
# Installing torch FIRST from the cu126 index is what stops pip resolving the default cu13
# wheel and leaving an engine that imports fine and dies on init_device.
set -euo pipefail
cd ~/workspaces/omnia-llm
source .venv/bin/activate
pip install -q --upgrade pip wheel
pip install -q torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -q vllm==0.16.0 'fastapi>=0.110' 'uvicorn[standard]' httpx tomli
python - <<'PY'
import torch, vllm
print('torch', torch.__version__, '| vllm', vllm.__version__, '| cuda ok:', torch.cuda.is_available())
PY
