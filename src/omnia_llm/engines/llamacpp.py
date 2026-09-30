"""llama.cpp's ``llama-server``: GGUF models on CPU, Metal, CUDA or ROCm — the portable engine.

Needs the ``llama-server`` binary (a llama.cpp release, or ``brew install llama.cpp``).
"""

from __future__ import annotations

import dataclasses

from omnia_llm.devices import Device
from omnia_llm.engines.base import ProcessEngine
from omnia_llm.engines.registry import register_engine


@dataclasses.dataclass(frozen=True)
class LlamaCppOptions:
    #: A local .gguf path, or — with ``hf_repo`` — the file name inside that repo.
    model: str = ""
    #: e.g. "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M" — downloaded by llama-server itself.
    hf_repo: str = ""
    binary: str = "llama-server"
    ctx_size: int = 4096
    #: Layers offloaded to the GPU (0 on CPU; a large number offloads everything).
    gpu_layers: int = 0
    extra_args: tuple[str, ...] = ()


@register_engine("llamacpp")
class LlamaCppEngine(ProcessEngine):
    Options = LlamaCppOptions
    backends = ("cpu", "apple", "nvidia", "rocm")

    def command(self, device: Device) -> list[str]:
        o = self.options
        source = ["-hf", o.hf_repo] if o.hf_repo else ["-m", o.model]
        return [
            o.binary, *source, "--host", "127.0.0.1", "--port", str(self.spec.port),
            "--alias", self.id, "-c", str(o.ctx_size),
            "-ngl", str(0 if device.backend == "cpu" else o.gpu_layers),
            *o.extra_args,
        ]

    def environment(self, device: Device) -> dict[str, str]:
        env = super().environment(device)
        # ``-hf`` downloads into llama.cpp's own cache, not Hugging Face's.
        env["LLAMA_CACHE"] = str(self.paths.llama_cache)
        return env
