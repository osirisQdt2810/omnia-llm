"""Finding this machine's probe."""

from __future__ import annotations

import pytest

from omnia_llm.devices import DeviceError, detect
from omnia_llm.platforms.nvidia.device import NvidiaProbe


class TestDetect:
    def test_cpu_is_always_there(self):
        assert type(detect("cpu")).registry_name == "cpu"

    def test_a_configured_probe_that_is_absent_is_an_error(self, monkeypatch):
        monkeypatch.setattr(NvidiaProbe, "available", classmethod(lambda cls: False))
        with pytest.raises(DeviceError):
            detect("nvidia")
