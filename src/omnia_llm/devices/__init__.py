"""Devices: the abstraction. Each platform's probe lives in ``omnia_llm.platforms.<platform>``."""

from omnia_llm.devices.base import Device, DeviceError, DeviceProbe
from omnia_llm.devices.registry import DEVICE_PROBES, detect, register_device_probe

__all__ = ["DEVICE_PROBES", "Device", "DeviceError", "DeviceProbe", "detect",
           "register_device_probe"]
