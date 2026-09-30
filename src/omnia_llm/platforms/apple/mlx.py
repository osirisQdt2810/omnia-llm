"""MLX: Apple's framework for Apple Silicon — fast, quantized text models on a Mac."""

from __future__ import annotations

import dataclasses

from omnia_llm.devices import Device
from omnia_llm.engines.base import ProcessEngine
from omnia_llm.engines.registry import register_engine


@dataclasses.dataclass(frozen=True)
class MlxOptions:
    #: An MLX model repo or path, e.g. "mlx-community/Qwen2.5-1.5B-Instruct-4bit".
    model: str
    #: Default cap on generated tokens when a request sets none.
    max_tokens: int = 512


@register_engine("mlx")
class MlxEngine(ProcessEngine):
    Options = MlxOptions
    backends = ("apple",)

    @property
    def upstream_model(self) -> str:
        # mlx_lm.server treats an unknown `model` as a model to LOAD — so every request is
        # pointed at the one it already serves, whatever id the client used.
        return self.options.model

    def command(self, device: Device) -> list[str]:
        o = self.options
        return self.titled(
            "llm-engine", "mlx_lm.server", "--model", o.model,
            "--host", "127.0.0.1", "--port", str(self.spec.port),
            "--max-tokens", str(o.max_tokens),
        )
