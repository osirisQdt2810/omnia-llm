"""The CPU probe: the fallback that hides every GPU."""

from __future__ import annotations

from omnia_llm.devices import detect


class TestCpu:
    def test_the_cpu_hides_every_gpu(self):
        cpu = detect("cpu").list_devices()[0]
        assert dict(cpu.env) == {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
