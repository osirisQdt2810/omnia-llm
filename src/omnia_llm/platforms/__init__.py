"""Hardware platforms, one package each: how to find its devices, and any engine that runs
ONLY there.

    nvidia/  NVIDIA GPUs, found with nvidia-smi
    rocm/    AMD GPUs, found with rocm-smi
    apple/   Apple Silicon: unified memory, plus MLX, which runs nowhere else
    cpu/     any machine: the fallback when no accelerator is found

An engine that runs on several platforms (vLLM, diffusers, llama.cpp) lives in
``omnia_llm.engines`` instead. Importing this package registers every probe and every
platform-only engine.
"""

from omnia_llm.platforms import apple, cpu, nvidia, rocm  # noqa: F401 - registration
