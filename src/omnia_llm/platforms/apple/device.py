"""Apple Silicon: one device, the unified memory shared by CPU and GPU (Metal / MPS / MLX)."""

from __future__ import annotations

import platform
import subprocess

from omnia_llm.devices.base import Device, DeviceError, DeviceProbe
from omnia_llm.devices.registry import register_device_probe


@register_device_probe("apple")
class AppleSiliconProbe(DeviceProbe):
    priority = 30

    @classmethod
    def available(cls) -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def list_devices(self) -> list[Device]:
        try:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], check=True,
                                       capture_output=True, text=True, timeout=10).stdout)
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=True,
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(f"could not read the Mac's memory: {exc}") from exc
        # A laptop is one person's machine: there is no colleague to protect, so the busy
        # signals do not apply and the device is exclusive.
        return [Device(backend="apple", index=0, name=chip or "Apple Silicon",
                       memory_total_mib=total // (1024 * 1024), exclusive=True)]
