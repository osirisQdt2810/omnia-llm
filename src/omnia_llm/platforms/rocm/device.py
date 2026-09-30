"""AMD GPUs, read with ``rocm-smi --json``; a process is pinned with HIP_VISIBLE_DEVICES."""

from __future__ import annotations

import json
import shutil
import subprocess

from omnia_llm.devices.base import Device, DeviceError, DeviceProbe
from omnia_llm.devices.registry import register_device_probe

_MIB = 1024 * 1024


@register_device_probe("rocm")
class RocmProbe(DeviceProbe):
    priority = 20

    @classmethod
    def available(cls) -> bool:
        return shutil.which("rocm-smi") is not None

    def list_devices(self) -> list[Device]:
        try:
            raw = subprocess.run(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showuse",
                 "--showpids", "--json"],
                check=True, capture_output=True, text=True, timeout=30,
            ).stdout
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(f"could not read rocm-smi: {exc}") from exc
        return self.parse(raw)

    @staticmethod
    def parse(raw: str) -> list[Device]:
        """rocm-smi's JSON: one ``cardN`` object per GPU, VRAM in bytes, use in percent.

        Keys vary a little across ROCm releases, so each is read defensively; a card whose
        memory cannot be read is reported as busy rather than guessed free.
        """
        data = json.loads(raw or "{}")
        busy_cards = set()
        for key, value in data.items():
            if key == "system" and isinstance(value, dict):
                for proc in value.values():
                    if isinstance(proc, str) and "," in proc:
                        # "name, gpu(s), vram, sdma, cu" — the second field lists card indices
                        for idx in proc.split(",")[1].split():
                            busy_cards.add(idx.strip())
        devices = []
        for key in sorted((k for k in data if k.startswith("card")), key=lambda k: int(k[4:])):
            card = data[key]
            index = int(key[4:])
            total = int(card.get("VRAM Total Memory (B)", 0)) // _MIB
            used = int(card.get("VRAM Total Used Memory (B)", 0)) // _MIB
            util = int(float(str(card.get("GPU use (%)", "0")).strip("%") or 0))
            devices.append(Device(
                backend="rocm",
                index=index,
                name=str(card.get("Card series") or card.get("Card SKU") or key),
                memory_total_mib=total,
                memory_used_mib=used if total else 1 << 30,
                utilization_pct=util,
                has_compute_apps=str(index) in busy_cards,
                env=(("HIP_VISIBLE_DEVICES", str(index)),),
            ))
        return devices
