import asyncio
import contextlib
import json
import signal
import subprocess
import time

import httpx

from omnia_llm.config import AppConfig, DevicesConfig, ModelSpec, PathsConfig, ServerConfig
from omnia_llm.server.manager import ModelManager
from tests.helpers import (
    IGNORES_SIGTERM,
    fake_lifecycle,
    fake_probe,
    free_port,
    gpu,
    group_alive,
    until,
)


def _manager(tmp_path, *models, probe=None, server=None):
    cfg = AppConfig(server or ServerConfig(),
                    PathsConfig(state=tmp_path, model_cache=tmp_path / "model-cache"),
                    DevicesConfig(), tuple(models))
    return ModelManager(cfg, probe or fake_probe([gpu(7)]))


TEXT = ModelSpec("omnia-local", "text", "vllm", 8722, options={"model": "t"})
TEXT2 = ModelSpec("small", "text", "vllm", 8724, options={"model": "s"})
IMAGE = ModelSpec("sdxl-turbo", "image", "diffusers", 8723, options={"model": "i"})


def test_a_model_is_routed_by_its_id(tmp_path):
    m = _manager(tmp_path, TEXT, TEXT2, IMAGE)
    assert m.route("text", "small").id == "small"


def test_an_unknown_id_falls_back_to_the_first_of_its_kind(tmp_path):
    """An older client config ("sdxl" for "sdxl-turbo") keeps working."""
    m = _manager(tmp_path, TEXT, IMAGE)
    assert m.route("image", "sdxl").id == "sdxl-turbo"
    assert m.route("text", "").id == "omnia-local"


def test_an_id_of_the_wrong_kind_is_not_used_for_the_other(tmp_path):
    m = _manager(tmp_path, TEXT, IMAGE)
    assert m.route("text", "sdxl-turbo").id == "omnia-local"


def test_no_model_of_a_kind_is_none(tmp_path):
    assert _manager(tmp_path, TEXT).route("image", "x") is None


def test_the_listing_carries_each_kind_and_starts_nothing(tmp_path):
    m = _manager(tmp_path, TEXT, IMAGE)
    assert [(e["id"], e["kind"]) for e in m.listing()] == [("omnia-local", "text"),
                                                           ("sdxl-turbo", "image")]
    assert not any(e.running for e in m.engines.values())


async def test_a_client_still_using_an_old_id_is_answered(tmp_path):
    """The shipped ids were renamed after their models. A client configured before that is
    routed to the first text model, and must reach it under the name vLLM serves it as."""
    served = "qwen2.5-14b-instruct-awq"
    m = _manager(tmp_path, ModelSpec(served, "text", "vllm", 8722, options={"model": "t"}))
    engine = m.route("text", "omnia-local")

    async def running(prefer=()):
        pass

    def vllm(request):
        asked = json.loads(request.content)["model"]
        if asked != served:
            return httpx.Response(404, json={"error": {"message": f"model {asked} not found"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ephemeral"}}]})

    engine.ensure_running = running
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(vllm))
    body = json.dumps({"model": "omnia-local", "messages": []}).encode()
    response = await m.forward(engine, "POST", "/v1/chat/completions", body, {})
    assert response.status_code == 200


async def test_a_second_model_starts_on_the_card_the_first_is_using(tmp_path):
    """The pair shares one card: the one in use, which no longer looks free, before an idle
    second card, which would take two cards from the pool for one pair of models."""
    probe = fake_probe([gpu(2), gpu(7)])
    m = _manager(tmp_path, TEXT, IMAGE, probe=probe)

    def taken(device):  # as on a real card, the model's own memory and context mark it busy
        probe.devices = [gpu(d.index, used=13800, busy=True) if d.id == device.id else d
                         for d in probe.devices]

    for engine in m.engines.values():
        fake_lifecycle(engine, on_launch=taken)
        engine._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    text, image = m.engines["omnia-local"], m.engines["sdxl-turbo"]
    await m.forward(text, "POST", "/v1/chat/completions", b"{}", {})
    await m.forward(image, "POST", "/v1/images/generations", b"{}", {})
    assert text.device.id == image.device.id == "nvidia:7"


async def test_the_reaper_stops_a_model_left_idle(tmp_path):
    """The promise to everyone else on the machine: an idle model hands its card back."""
    m = _manager(tmp_path, TEXT,
                 server=ServerConfig(idle_timeout_minutes=1, reaper_interval_seconds=0.01))
    engine = m.engines["omnia-local"]
    fake_lifecycle(engine)
    await engine.ensure_running()
    engine._last_used = time.monotonic() - 120
    m.start_reaper()
    try:
        await until(lambda: not engine.running, timeout=5)
    finally:
        await m.shutdown()


async def test_a_shutdown_during_the_reapers_stop_still_ends_the_model(tmp_path):
    """The shutdown cancels the reaper. Cancelled mid-stop, the stop used to take the group with
    it, and a model that ignores SIGTERM outlived the gateway, holding its memory and port."""
    m = _manager(tmp_path, ModelSpec("omnia-local", "text", "vllm", free_port(),
                                     options={"model": "t"}),
                 server=ServerConfig(idle_timeout_minutes=1, reaper_interval_seconds=0.01))
    engine = m.engines["omnia-local"]
    engine.command = lambda device: IGNORES_SIGTERM
    engine.shutdown_grace_seconds = 0.5
    group = engine._launch(gpu(7))
    try:
        await until(lambda: "ready" in (tmp_path / "logs" / "omnia-local.log").read_text())
        engine._last_used = time.monotonic() - 120
        m.start_reaper()
        await until(lambda: engine._stopping is not None)  # the reaper's stop, mid-grace
        await m.shutdown()
        assert not group_alive(group.leader.pid)
    finally:
        if group.leader.returncode is None:  # unreaped: the id is still the group's own
            group.signal(signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                group.leader.wait(timeout=5)


async def test_every_model_is_stopped_at_once_and_one_failure_skips_none(tmp_path):
    """One after another, a shutdown took the sum of every engine's grace period, and one engine
    raising left the rest running."""
    m = _manager(tmp_path, TEXT, TEXT2, IMAGE)
    in_flight, peak, stopped = 0, 0, []

    def stopper(engine_id):
        async def stop(reason):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            stopped.append(engine_id)
            if engine_id == "small":
                raise RuntimeError("its stop failed")
        return stop

    for engine_id, engine in m.engines.items():
        engine.stop = stopper(engine_id)
    await m.stop_all("gateway shutting down")

    assert peak == 3
    assert sorted(stopped) == ["omnia-local", "sdxl-turbo", "small"]


async def test_a_shutdown_cancels_a_warm_up_still_starting(tmp_path):
    """A warm-up that finished its launch after stop_all had looked would leave an engine holding
    its device once the gateway is gone."""
    m = _manager(tmp_path, TEXT)
    started = asyncio.Event()

    async def slow_start(prefer=()):
        started.set()
        await asyncio.Event().wait()  # never finishes on its own

    m.engines["omnia-local"].ensure_running = slow_start
    m.warm(["text"])
    await started.wait()
    warm_up = m._warming["omnia-local"]
    try:
        await asyncio.wait_for(m.shutdown(), timeout=5)

        assert warm_up.cancelled()
        assert "omnia-local" not in m._warming
    finally:
        warm_up.cancel()  # so a failure here cannot leave the test's loop waiting on it

