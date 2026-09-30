"""vLLM: high-throughput text serving on a GPU (CUDA, or ROCm with vLLM's ROCm build)."""

from __future__ import annotations

import dataclasses

from omnia_llm.devices import Device
from omnia_llm.engines.base import ProcessEngine
from omnia_llm.engines.registry import register_engine


@dataclasses.dataclass(frozen=True)
class VllmOptions:
    #: HuggingFace id or local path. AWQ 4-bit of a 14B leaves room on 24 GB for a KV cache.
    model: str
    max_model_len: int = 4096
    #: vLLM PREALLOCATES this share of the card; keep headroom on a shared GPU.
    gpu_memory_utilization: float = 0.90
    extra_args: tuple[str, ...] = ()


@register_engine("vllm")
class VllmEngine(ProcessEngine):
    Options = VllmOptions
    backends = ("nvidia", "rocm")

    def command(self, device: Device) -> list[str]:
        o = self.options
        return self.titled(
            "llm-engine", "vllm.entrypoints.cli.main", "serve", o.model,
            "--host", "127.0.0.1", "--port", str(self.spec.port),
            # Served under the id clients use, so swapping weights never breaks a client.
            "--served-model-name", self.id,
            "--max-model-len", str(o.max_model_len),
            "--gpu-memory-utilization", str(o.gpu_memory_utilization),
            # One device, always — the gateway's promise is that it takes exactly one.
            "--tensor-parallel-size", "1",
            *o.extra_args,
        )
