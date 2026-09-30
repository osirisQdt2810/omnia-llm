"""Engines: the abstraction, and the engines that run on more than one platform.

    vllm       text   NVIDIA (CUDA), AMD (ROCm build of vLLM)
    diffusers  image  NVIDIA, AMD, Apple (MPS), CPU
    llamacpp   text   every platform (GGUF; llama.cpp's own server)

An engine that runs on ONE platform only lives with that platform, in
``omnia_llm.platforms.<platform>`` (MLX, under ``apple``). Either way it registers by name with
``@register_engine`` and says where it runs in ``backends``.
"""

from omnia_llm.engines.base import Engine, EngineBusy, ProcessEngine
from omnia_llm.engines.registry import ENGINES, register_engine

# isort: split
# The portable engines register themselves on import, after the names they subclass exist.
from omnia_llm.engines import diffusers, llamacpp, vllm  # noqa: E402, F401 - registration

__all__ = ["ENGINES", "Engine", "EngineBusy", "ProcessEngine", "register_engine"]
