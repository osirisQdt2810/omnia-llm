"""Engines: the lazy lifecycle (start on demand, once; stop when idle) and each backend's command."""

from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest
from conftest import FakeProbe, gpu, spec

import omnia_llm.platforms  # noqa: F401 - registration
from omnia_llm.config import ConfigError
from omnia_llm.engines import ENGINES, EngineBusy


class _FakeProcess:
    """Stands in for the model's process: alive until something kills it."""

    pid = 4242

    def __init__(self):
        self.code = None

    def poll(self):
        return self.code


def _engine(paths, *, probe=None, ready=True, **spec_kw):
    s = spec(**spec_kw)
    engine = ENGINES.get(s.engine)(s, probe or FakeProbe([gpu(7)]), paths)
    launched = []

    def launch(device):
        launched.append(device.id)
        engine._process = _FakeProcess()
        engine._device = device
        engine._started_at = engine._last_used = time.monotonic()

    async def wait_ready():
        if not ready:
            raise RuntimeError("exited with code 1 during startup")

    async def stop(reason):
        if engine._process is not None:
            engine._process.code = 0
        engine._process = engine._device = None

    engine._launch, engine._wait_until_ready, engine.stop = launch, wait_ready, stop
    return engine, launched


class TestItStartsOnlyWhenAsked:
    async def test_nothing_runs_until_the_first_request(self, paths):
        engine, launched = _engine(paths)
        assert not engine.running and launched == []

    async def test_the_first_request_starts_it_on_the_free_card(self, paths):
        engine, launched = _engine(paths)
        await engine.ensure_running()
        assert launched == ["nvidia:7"] and engine.device.id == "nvidia:7"

    async def test_a_second_request_reuses_it(self, paths):
        engine, launched = _engine(paths)
        await engine.ensure_running()
        await engine.ensure_running()
        assert launched == ["nvidia:7"]

    async def test_a_request_arriving_mid_start_waits_for_it(self, paths):
        """The process exists from the moment it is launched, long before it answers. A second
        request let through then would be forwarded to a model still loading: a refused
        connection, which the user sees as a failed generation."""
        engine, launched = _engine(paths)
        ready, order = asyncio.Event(), []

        async def loading():
            await ready.wait()
            order.append("ready")

        engine._wait_until_ready = loading

        async def request(n):
            await engine.ensure_running()
            order.append(n)

        requests = [asyncio.create_task(request(n)) for n in range(3)]
        await asyncio.sleep(0.01)
        assert order == [], "a request got through while the model was loading"
        ready.set()
        await asyncio.gather(*requests)
        assert order[0] == "ready" and sorted(order[1:]) == [0, 1, 2]
        assert launched == ["nvidia:7"]

    async def test_every_card_busy_is_a_plain_refusal(self, paths):
        engine, _ = _engine(paths, probe=FakeProbe([gpu(0, busy=True)]))
        with pytest.raises(EngineBusy):
            await engine.ensure_running()

    async def test_it_prefers_the_card_another_model_holds(self, paths):
        probe = FakeProbe([gpu(2), gpu(7, used=13800, busy=True)])
        engine, launched = _engine(paths, probe=probe, required_mib=5000)
        await engine.ensure_running(prefer=["nvidia:7"])
        assert launched == ["nvidia:7"]


class TestItGivesTheCardBack:
    async def test_it_stops_after_the_idle_timeout(self, paths):
        engine, _ = _engine(paths)
        await engine.ensure_running()
        engine._last_used = time.monotonic() - 100
        assert await engine.stop_if_idle(60) and not engine.running

    async def test_it_stays_up_while_requests_keep_arriving(self, paths):
        engine, _ = _engine(paths)
        await engine.ensure_running()
        engine._last_used = time.monotonic() - 10
        assert not await engine.stop_if_idle(60) and engine.running

    async def test_a_request_arriving_while_it_waits_to_stop_keeps_it_up(self, paths):
        """Idleness is checked twice: once to decide, then again under the lock. A request that
        came in between must win, or the model is killed mid-generation."""
        engine, _ = _engine(paths)
        await engine.ensure_running()
        engine._last_used = time.monotonic() - 100
        async with engine._lock:  # something else has the engine, e.g. a start
            reap = asyncio.create_task(engine.stop_if_idle(60))
            await asyncio.sleep(0.01)  # the reaper decided "idle" and waits for the lock
            engine._last_used = time.monotonic()  # a request arrives meanwhile
        assert not await reap and engine.running

    async def test_a_failed_start_leaves_nothing_holding_the_card(self, paths):
        engine, _ = _engine(paths, ready=False)
        with pytest.raises(RuntimeError):
            await engine.ensure_running()
        assert not engine.running and engine.status()["device"] is None


class TestConfigIsChecked:
    def test_an_engine_refuses_a_platform_it_does_not_run_on(self, paths):
        with pytest.raises(ConfigError, match="runs on apple"):
            ENGINES.get("mlx")(spec(engine="mlx"), FakeProbe(backend="nvidia"), paths)

    def test_an_unknown_option_is_refused(self, paths):
        with pytest.raises(ConfigError, match="unknown option"):
            ENGINES.get("vllm")(spec(max_len=9), FakeProbe(), paths)

    def test_a_missing_required_option_is_refused(self, paths):
        s = spec()
        s.options.pop("model")
        with pytest.raises(ConfigError):
            ENGINES.get("vllm")(s, FakeProbe(), paths)


class TestCommands:
    def test_vllm_serves_under_the_client_id_on_one_card(self, paths):
        engine, _ = _engine(paths)
        argv = engine.command(gpu(7))
        assert argv[1:5] == ["-m", "omnia_llm.workers.titled", "llm-engine",
                             "vllm.entrypoints.cli.main"]
        assert argv[argv.index("--served-model-name") + 1] == "omnia-local"
        assert argv[argv.index("--tensor-parallel-size") + 1] == "1"

    def test_diffusers_is_told_the_device_kind(self, paths):
        engine = ENGINES.get("diffusers")(spec(id="img", kind="image", engine="diffusers"),
                                          FakeProbe(), paths)
        argv = engine.command(gpu(7))
        assert argv[3] == "image-engine" and argv[argv.index("--device") + 1] == "nvidia"

    def test_mlx_always_asks_for_the_model_it_loaded(self, paths):
        """mlx_lm.server would LOAD an unknown model name — so every request is rewritten."""
        engine = ENGINES.get("mlx")(spec(engine="mlx", model="mlx-community/x-4bit"),
                                    FakeProbe(backend="apple"), paths)
        assert b'"model": "mlx-community/x-4bit"' in engine.rewrite(b'{"model": "omnia-local"}')

    def test_llamacpp_uses_no_gpu_layers_on_a_cpu(self, paths):
        from omnia_llm.devices import Device

        engine = ENGINES.get("llamacpp")(spec(engine="llamacpp", model="m.gguf", gpu_layers=99),
                                         FakeProbe(backend="cpu"), paths)
        cpu = Device(backend="cpu", index=0, name="cpu", memory_total_mib=1, exclusive=True)
        argv = engine.command(cpu)
        assert argv[argv.index("-ngl") + 1] == "0" and argv[argv.index("--alias") + 1] == "omnia-local"

    def test_llamacpp_downloads_into_the_model_cache_too(self, paths):
        engine = ENGINES.get("llamacpp")(spec(engine="llamacpp", hf_repo="x/y-GGUF:Q4_K_M"),
                                         FakeProbe(backend="cpu"), paths)
        env = engine.environment(gpu(0))
        assert env["LLAMA_CACHE"] == str(paths.model_cache / "llama.cpp")

    def test_the_child_is_pinned_and_keeps_weights_in_state(self, paths):
        engine, _ = _engine(paths)
        env = engine.environment(gpu(3))
        assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["HF_HOME"] == str(paths.hf_home)

    def test_a_body_that_is_not_json_is_forwarded_untouched(self, paths):
        engine = ENGINES.get("mlx")(spec(engine="mlx"), FakeProbe(backend="apple"), paths)
        assert engine.rewrite(b"not json") == b"not json"


def test_starting_a_model_leaks_no_file_descriptor(paths, monkeypatch):
    """The parent must not hold the log open: the child has its own copy, and a descriptor kept
    per start would run a gateway that restarts idle models all day out of them."""
    engine = ENGINES.get("vllm")(spec(), FakeProbe([gpu(7)]), paths)
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    monkeypatch.setattr(engine, "command", lambda device: sleeper)
    before = len(os.listdir("/dev/fd"))
    engine._launch(gpu(7))
    try:
        assert len(os.listdir("/dev/fd")) == before
        assert "=== omnia-local on nvidia:7" in (paths.logs / "omnia-local.log").read_text()
    finally:
        engine._process.kill()
        engine._process.wait()
