"""The device-probe registry, and ``detect()`` for ``probe = "auto"``."""

from __future__ import annotations

from omnia_llm.devices.base import DeviceError, DeviceProbe
from omnia_llm.registry import Registry

DEVICE_PROBES: Registry[DeviceProbe] = Registry("device probe")
register_device_probe = DEVICE_PROBES.register


def detect(name: str = "auto") -> DeviceProbe:
    """The probe named ``name``, or for ``auto`` the first available one by priority."""
    import omnia_llm.platforms  # noqa: F401 - registers every built-in platform
    if name != "auto":
        cls = DEVICE_PROBES.get(name)
        if not cls.available():
            raise DeviceError(f"device probe {name!r} is configured but not available here")
        return cls()
    for _name, cls in sorted(DEVICE_PROBES.items(), key=lambda item: item[1].priority):
        if cls.available():
            return cls()
    raise DeviceError("no device probe is available on this machine")
