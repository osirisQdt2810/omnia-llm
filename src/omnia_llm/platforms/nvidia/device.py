"""NVIDIA GPUs, read with ``nvidia-smi``; a process is pinned with CUDA_VISIBLE_DEVICES."""

from __future__ import annotations

import shutil
import subprocess

from omnia_llm.devices.base import Device, DeviceError, DeviceProbe
from omnia_llm.devices.registry import register_device_probe


def _run(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout


@register_device_probe("nvidia")
class NvidiaProbe(DeviceProbe):
    priority = 10

    @classmethod
    def available(cls) -> bool:
        return shutil.which("nvidia-smi") is not None

    def list_devices(self) -> list[Device]:
        try:
            rows = _run([
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,uuid",
                "--format=csv,noheader,nounits",
            ])
            apps = _run([
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader",
            ])
        except Exception as exc:  # noqa: BLE001 - one clear failure for the caller
            raise DeviceError(f"could not read nvidia-smi: {exc}") from exc
        return self.parse(rows, apps)

    @staticmethod
    def parse(rows: str, apps: str) -> list[Device]:
        busy = {line.split(",")[0].strip() for line in apps.splitlines() if line.strip()}
        devices = []
        for line in rows.splitlines():
            if not line.strip():
                continue
            index, name, total, used, util, uuid = (p.strip() for p in line.split(","))
            devices.append(Device(
                backend="nvidia",
                index=int(index),
                name=name,
                memory_total_mib=int(total),
                memory_used_mib=int(used),
                utilization_pct=int(util),
                has_compute_apps=uuid in busy,
                # The WHOLE process, including anything a library does at import time, then
                # cannot see the other cards — a flag would leave room for a stray allocation.
                env=(("CUDA_VISIBLE_DEVICES", index),),
            ))
        return devices
