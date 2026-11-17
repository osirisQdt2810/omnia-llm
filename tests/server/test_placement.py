"""Every model on one card: side by side when they fit, taking turns on it when they do not."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from omnia_llm.config import (
    AppConfig,
    CpuBusyOverride,
    DevicesConfig,
    ModelSpec,
    PathsConfig,
    ServerConfig,
)
from omnia_llm.engines import EngineBusy
from omnia_llm.server.manager import ModelManager
from tests.helpers import cpu, fake_lifecycle, fake_probe, gpu, until

#: As configs/nvidia-24gb-gemma4.toml has them: 15.5 GB with some experts in RAM, 19 GB without.
TEXT = ModelSpec("gemma", "text", "llamacpp", 8722, required_mib=15500,
                 options={"model": "g.gguf", "extra_args": ["--n-cpu-moe", "8"]},
                 when_cpu_busy=CpuBusyOverride(required_mib=19000, options={"extra_args": []}))
IMAGE = ModelSpec("sdxl", "image", "diffusers", 8723, required_mib=7000, options={"model": "i"})
FREE, BUSY = 4.0, 73.0  # load averages on 48 CPUs


class _Box:
    """A gateway on two idle 24 GB cards, its engines faked. A running model holds what it
    declared on its card (or ``actual[id]``), and answers each request once its ``hold`` is
    released, if one was set for it."""

    def __init__(self, tmp_path, *, load, single_device=True):
        cfg = AppConfig(ServerConfig(), PathsConfig(state=tmp_path, model_cache=tmp_path / "mc"),
                        DevicesConfig(single_device=single_device), (TEXT, IMAGE))
        self.load = load
        self.actual: dict[str, int] = {}
        #: Memory somebody else holds, by device id.
        self.foreign: dict[str, int] = {}
        probe = fake_probe([gpu(6), gpu(7)])
        probe.list_devices = self._devices
        self.manager = ModelManager(cfg, probe, cpu_load=lambda: cpu(self.load))
        self.manager.placement.drain_poll_seconds = 0.01
        self.launched: list[tuple[str, str]] = []
        self.served: list[str] = []
        self._holds: dict[str, list[asyncio.Event]] = {}
        for engine in self.manager.engines.values():
            fake_lifecycle(engine,
                           on_launch=lambda d, e=engine: self.launched.append((e.id, d.id)))
            engine._client = httpx.AsyncClient(
                transport=httpx.MockTransport(self._upstream(engine.id)))

    def engine(self, model_id):
        return self.manager.engines[model_id]

    def hold(self, model_id) -> asyncio.Event:
        """Keep the next request to ``model_id`` in flight until the event is set."""
        event = asyncio.Event()
        self._holds.setdefault(model_id, []).append(event)
        return event

    async def ask(self, model_id) -> int:
        engine = self.engine(model_id)
        path = "/v1/chat/completions" if engine.kind == "text" else "/v1/images/generations"
        return (await self.manager.forward(engine, "POST", path, b"{}", {})).status_code

    def running(self) -> list[str]:
        return [e.id for e in self.manager.engines.values() if e.running]

    def _devices(self):
        devices = []
        for index in (6, 7):
            device_id = f"nvidia:{index}"
            ours = [e for e in self.manager.engines.values()
                    if e.running and e.device.id == device_id]
            used = (10 + self.foreign.get(device_id, 0)
                    + sum(self.actual.get(e.id, e.required_mib) for e in ours))
            devices.append(gpu(index, used=used, busy=bool(ours) or device_id in self.foreign))
        return devices

    def _upstream(self, model_id):
        async def respond(request):
            holds = self._holds.get(model_id)
            if holds:
                await holds.pop(0).wait()
            await asyncio.sleep(0.02)  # a request takes a moment: long enough to overlap
            self.served.append(model_id)
            return httpx.Response(200, json={"ok": True})
        return respond


async def test_on_a_free_cpu_both_models_share_the_card(tmp_path):
    box = _Box(tmp_path, load=FREE)
    assert await box.ask("gemma") == await box.ask("sdxl") == 200
    assert box.launched == [("gemma", "nvidia:7"), ("sdxl", "nvidia:7")]
    assert box.running() == ["gemma", "sdxl"]
    assert box.engine("gemma").required_mib == 15500


async def test_on_a_busy_cpu_they_take_turns_on_the_one_card(tmp_path):
    """Gemma whole leaves no room for the image model: never a second card, but a turn each."""
    box = _Box(tmp_path, load=BUSY)
    for model_id in ("gemma", "sdxl", "gemma"):
        assert await box.ask(model_id) == 200
    assert box.launched == [("gemma", "nvidia:7"), ("sdxl", "nvidia:7"), ("gemma", "nvidia:7")]
    assert box.served == ["gemma", "sdxl", "gemma"]
    assert box.running() == ["gemma"] and box.engine("gemma").required_mib == 19000


async def test_an_image_model_between_images_still_counts_what_it_declared(tmp_path):
    """Idle, it holds a fraction of the 7 GB it takes while drawing. Packed into the gap, Gemma
    would leave it no room for its next image: the failure that made the card a single one."""
    box = _Box(tmp_path, load=BUSY)
    box.actual["sdxl"] = 2000
    await box.ask("sdxl")
    await box.ask("gemma")
    assert box.running() == ["gemma"]
    assert box.launched == [("sdxl", "nvidia:7"), ("gemma", "nvidia:7")]


async def test_a_model_that_stopped_every_other_starts_on_the_card_it_emptied(tmp_path):
    """Nothing runs once the last model on the card is stopped, but the card is not up for a
    fresh pick: one ranked above it, here the idle card 6 over card 7 still showing a little
    memory held, would make it a second card."""
    box = _Box(tmp_path, load=BUSY)
    await box.ask("gemma")
    box.foreign["nvidia:7"] = 600  # what a stopped model's card can still show for a moment
    await box.ask("sdxl")
    assert box.launched == [("gemma", "nvidia:7"), ("sdxl", "nvidia:7")]
    assert box.running() == ["sdxl"]


async def test_a_model_in_use_finishes_its_requests_before_it_is_stopped(tmp_path):
    box = _Box(tmp_path, load=BUSY)
    released = box.hold("gemma")
    text = asyncio.create_task(box.ask("gemma"))
    await until(lambda: box.manager.placement.in_flight(box.engine("gemma")) == 1)
    state = box.manager.status()["gemma"]
    assert (state["in_flight"], state["profile"]) == (1, "when_cpu_busy")

    image = asyncio.create_task(box.ask("sdxl"))
    await asyncio.sleep(0.1)
    assert box.running() == ["gemma"] and not image.done()

    released.set()
    assert await text == await image == 200
    assert box.served == ["gemma", "sdxl"]
    assert box.launched == [("gemma", "nvidia:7"), ("sdxl", "nvidia:7")]


async def test_new_requests_for_a_model_waiting_to_stop_wait_their_turn(tmp_path):
    """Let straight in, a steady stream of text requests would keep the image model waiting
    until it timed out."""
    box = _Box(tmp_path, load=BUSY)
    released = box.hold("gemma")
    first = asyncio.create_task(box.ask("gemma"))
    await until(lambda: box.manager.placement.in_flight(box.engine("gemma")) == 1)
    image = asyncio.create_task(box.ask("sdxl"))
    await asyncio.sleep(0.05)  # the image model now waits for gemma's request
    second = asyncio.create_task(box.ask("gemma"))
    await asyncio.sleep(0.05)
    assert box.manager.placement.in_flight(box.engine("gemma")) == 1, "it was let in"

    released.set()
    assert await asyncio.wait_for(asyncio.gather(first, image, second), timeout=5) == [200] * 3
    assert box.served == ["gemma", "sdxl", "gemma"]


async def test_two_models_asked_at_once_are_both_served_in_turn(tmp_path):
    """Each in the other's way: serialised, one waits for the other instead of both for each
    other."""
    box = _Box(tmp_path, load=BUSY)
    asked = ["gemma", "sdxl", "gemma", "sdxl"]
    codes = await asyncio.wait_for(asyncio.gather(*(box.ask(m) for m in asked)), timeout=5)
    assert codes == [200] * 4
    assert box.served == asked
    assert box.launched == [(m, "nvidia:7") for m in asked]


async def test_a_model_busy_for_too_long_is_a_refusal_not_a_hang(tmp_path):
    box = _Box(tmp_path, load=BUSY)
    box.manager.placement.drain_timeout_seconds = 0.1
    released = box.hold("gemma")
    text = asyncio.create_task(box.ask("gemma"))
    await until(lambda: box.manager.placement.in_flight(box.engine("gemma")) == 1)

    with pytest.raises(EngineBusy, match="still had requests in flight"):
        await box.ask("sdxl")
    assert box.running() == ["gemma"]
    # No longer waited for: its next request goes straight in.
    assert await asyncio.wait_for(box.ask("gemma"), timeout=1) == 200
    released.set()
    assert await text == 200


async def test_a_warm_up_never_takes_a_running_models_turn(tmp_path):
    box = _Box(tmp_path, load=BUSY)
    await box.ask("gemma")
    assert box.manager.warm(["image"]) == ["sdxl"]
    await until(lambda: not box.manager._warming)
    assert box.running() == ["gemma"]


async def test_a_warm_up_starts_a_model_that_fits_beside_the_other(tmp_path):
    box = _Box(tmp_path, load=FREE)
    await box.ask("gemma")
    box.manager.warm(["image"])
    await until(lambda: not box.manager._warming)
    assert box.running() == ["gemma", "sdxl"]


async def test_with_no_model_running_the_next_may_start_on_another_card(tmp_path):
    box = _Box(tmp_path, load=BUSY)
    await box.ask("gemma")
    await box.engine("gemma").stop("idle")
    box.foreign["nvidia:7"] = 3000  # somebody took card 7 meanwhile
    await box.ask("sdxl")
    assert box.launched[-1] == ("sdxl", "nvidia:6")


async def test_without_single_device_a_model_that_does_not_fit_takes_a_second_card(tmp_path):
    """Today's behaviour, unchanged for every config that does not ask for one card; the busy
    CPU settings apply all the same."""
    box = _Box(tmp_path, load=BUSY, single_device=False)
    await box.ask("gemma")
    await box.ask("sdxl")
    assert box.launched == [("gemma", "nvidia:7"), ("sdxl", "nvidia:6")]
    assert box.running() == ["gemma", "sdxl"]
    assert box.engine("gemma").required_mib == 19000
