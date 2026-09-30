"""diffusers: text-to-image, served by our own worker in the OpenAI images shape."""

from __future__ import annotations

import dataclasses

from omnia_llm.devices import Device
from omnia_llm.engines.base import ProcessEngine
from omnia_llm.engines.registry import register_engine


@dataclasses.dataclass(frozen=True)
class DiffusersOptions:
    #: e.g. "stabilityai/sdxl-turbo" — distilled to 4 steps, ~7 GB in fp16.
    model: str
    steps: int = 4
    #: Keep components in system RAM and move each to the card only for its step: ~1.5 GB
    #: resident instead of ~9.5, at a few seconds per image. CUDA only.
    cpu_offload: bool = True


@register_engine("diffusers")
class DiffusersEngine(ProcessEngine):
    Options = DiffusersOptions
    backends = ("nvidia", "rocm", "apple", "cpu")

    def command(self, device: Device) -> list[str]:
        o = self.options
        return self.titled(
            "image-engine", "omnia_llm.workers.diffusers_server",
            "--model", o.model, "--host", "127.0.0.1", "--port", str(self.spec.port),
            "--steps", str(o.steps), "--device", device.backend,
            *(() if o.cpu_offload else ("--no-cpu-offload",)),
        )
