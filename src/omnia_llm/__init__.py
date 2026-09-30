"""omnia-llm: serve self-hosted models to Omnia through one lazy, token-guarded gateway.

The layers, each open for extension through a registry (``registry.py``):

* ``devices``  — where a model can run: NVIDIA, ROCm, Apple Silicon, CPU (``@register_device_probe``).
* ``engines``  — how a model is served: vLLM, diffusers, MLX, llama.cpp (``@register_engine``).
* ``server``   — the OpenAI-compatible gateway: tokens, lockout, lazy start, idle stop.
"""

__version__ = "0.2.0"
