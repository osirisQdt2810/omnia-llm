"""The CPU and system RAM — always available, the fallback when there is no accelerator."""

from __future__ import annotations

import os
import platform

from omnia_llm.devices.base import Device, DeviceProbe
from omnia_llm.devices.registry import register_device_probe


def _total_ram_mib() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except (ValueError, OSError, AttributeError):  # pragma: no cover - platform dependent
        return 0


@register_device_probe("cpu")
class CpuProbe(DeviceProbe):
    priority = 90

    @classmethod
    def available(cls) -> bool:
        return True

    def list_devices(self) -> list[Device]:
        # Hide every GPU from the process, so a library cannot pick one up on its own.
        return [Device(backend="cpu", index=0, name=platform.processor() or "CPU",
                       memory_total_mib=_total_ram_mib(), exclusive=True,
                       env=(("CUDA_VISIBLE_DEVICES", ""), ("HIP_VISIBLE_DEVICES", "")))]
