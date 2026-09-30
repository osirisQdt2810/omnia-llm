"""Fakes shared by the tests: devices with no hardware behind them, engines with no processes
behind them, and a model-spec builder."""

from __future__ import annotations

import asyncio
import socket
import time

from omnia_llm.config import ModelSpec
from omnia_llm.devices import Device, DeviceProbe


def free_port() -> int:
    """A loopback port nothing listens on, for a test that starts a real process."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def gpu(index, used=10, util=0, busy=False, total=24564, backend="nvidia"):
    return Device(backend=backend, index=index, name="A5000", memory_total_mib=total,
                  memory_used_mib=used, utilization_pct=util, has_compute_apps=busy,
                  env=(("CUDA_VISIBLE_DEVICES", str(index)),))


class _FakeProbe(DeviceProbe):
    registry_name = "nvidia"

    def __init__(self, devices=None):
        self.devices = list(devices if devices is not None else [gpu(7)])

    @classmethod
    def available(cls):
        return True

    def list_devices(self):
        return list(self.devices)


def fake_probe(devices=None, backend="nvidia"):
    """A probe reporting ``devices`` as if it were ``backend``'s.

    A subclass per call, not one shared class: engines read the backend off the probe's CLASS,
    so setting it on a shared class would silently change every probe a test already holds.
    """
    probe_class = type(f"Fake{backend.title()}Probe", (_FakeProbe,), {"registry_name": backend})
    return probe_class(devices)


def spec(id="omnia-local", kind="text", engine="vllm", port=8722, required_mib=0, **options):
    options.setdefault("model", "test/model")
    return ModelSpec(id=id, kind=kind, engine=engine, port=port, required_mib=required_mib,
                     options=options)


class FakeGroup:
    """Stands in for a model's processes: alive until something ends them."""

    def __init__(self):
        self.code = None

    def poll(self):
        return self.code


def fake_lifecycle(engine, *, ready=True, on_launch=None):
    """Give ``engine`` fake processes; returns the ids of the devices it is launched on.

    ``on_launch(device)`` runs at each launch: a fake probe can then show the device as taken.
    """
    launched = []

    def launch(device):
        launched.append(device.id)
        if on_launch is not None:
            on_launch(device)
        engine._group = FakeGroup()
        engine._device = device
        engine._started_at = engine._last_used = time.monotonic()
        return engine._group

    async def wait_ready(group):
        if not ready:
            raise RuntimeError("exited with code 1 during startup")

    async def stop(reason):
        if engine._group is not None:
            engine._group.code = 0
        engine._group = engine._device = None

    engine._launch, engine._wait_until_ready, engine.stop = launch, wait_ready, stop
    return launched


async def until(condition, timeout=10.0):
    """Wait for ``condition()`` to hold, failing the test after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        await asyncio.sleep(0.01)
