"""Shared fakes: a device probe with no hardware, and a spec builder."""

from __future__ import annotations

import pytest

from omnia_llm.config import ModelSpec, PathsConfig
from omnia_llm.devices import Device, DeviceProbe


def gpu(index, used=10, util=0, busy=False, total=24564, backend="nvidia"):
    return Device(backend=backend, index=index, name="A5000", memory_total_mib=total,
                  memory_used_mib=used, utilization_pct=util, has_compute_apps=busy,
                  env=(("CUDA_VISIBLE_DEVICES", str(index)),))


class FakeProbe(DeviceProbe):
    """Reports whatever devices the test gives it, as if it were ``backend``'s probe."""

    registry_name = "nvidia"

    def __init__(self, devices=None, backend="nvidia"):
        self.devices = list(devices if devices is not None else [gpu(7)])
        type(self).registry_name = backend

    @classmethod
    def available(cls):
        return True

    def list_devices(self):
        return list(self.devices)


@pytest.fixture
def paths(tmp_path):
    return PathsConfig(state=tmp_path / "state", model_cache=tmp_path / "model-cache")


def spec(id="omnia-local", kind="text", engine="vllm", port=8722, required_mib=0, **options):
    options.setdefault("model", "test/model")
    return ModelSpec(id=id, kind=kind, engine=engine, port=port, required_mib=required_mib,
                     options=options)
