"""The lazy lifecycle: start on demand, one at a time, give the card back when idle."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from gateway import gpus
from gateway.engine import Engine, EngineBusy
from gateway.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        model="test/model",
        idle_timeout_minutes=0.05,  # 3 s
        reaper_interval_seconds=0.05,
        startup_timeout_seconds=5.0,
        log_dir=tmp_path,
    )


class _FakeProcess:
    """Stands in for the vLLM subprocess: alive until something kills it."""

    def __init__(self):
        self.pid = 4242
        self._code = None

    def poll(self):
        return self._code

    def die(self, code=0):
        self._code = code


def _fake_engine(settings, monkeypatch, *, free_gpu=7, ready=True):
    engine = Engine(settings)
    launched: list[int] = []

    monkeypatch.setattr(
        gpus,
        "pick_shared_or_free",
        lambda *_a, **_k: None
        if free_gpu is None
        else gpus.Gpu(free_gpu, "A5000", 24564, 10, 0, False),
    )

    def launch(index):
        launched.append(index)
        engine._process = _FakeProcess()
        engine._gpu_index = index
        import time as _t

        engine._started_at = engine._last_used = _t.monotonic()

    monkeypatch.setattr(engine, "_launch", launch)

    async def wait_ready():
        if not ready:
            raise RuntimeError("vLLM exited with code 1 during startup")

    monkeypatch.setattr(engine, "_wait_until_ready", wait_ready)
    return engine, launched


class TestItStartsOnlyWhenAsked:
    @pytest.mark.asyncio
    async def test_nothing_runs_until_the_first_request(self, settings, monkeypatch):
        engine, launched = _fake_engine(settings, monkeypatch)

        assert not engine.running
        assert launched == []

    @pytest.mark.asyncio
    async def test_the_first_request_starts_it_on_the_free_card(self, settings, monkeypatch):
        engine, launched = _fake_engine(settings, monkeypatch, free_gpu=7)

        await engine.ensure_running()

        assert launched == [7]
        assert engine.status()["gpu"] == 7

    @pytest.mark.asyncio
    async def test_a_second_request_reuses_the_running_engine(self, settings, monkeypatch):
        engine, launched = _fake_engine(settings, monkeypatch)

        await engine.ensure_running()
        await engine.ensure_running()

        assert launched == [7], "it started a second engine on top of the first"

    @pytest.mark.asyncio
    async def test_concurrent_cold_starts_launch_once(self, settings, monkeypatch):
        """The failure the lock prevents costs an out-of-memory crash.

        Two requests arriving together must not both conclude the engine is down and both
        launch vLLM on the same card.
        """
        engine, launched = _fake_engine(settings, monkeypatch)

        await asyncio.gather(*(engine.ensure_running() for _ in range(5)))

        assert launched == [7]

    @pytest.mark.asyncio
    async def test_every_card_busy_is_a_plain_refusal_not_a_crash(self, settings, monkeypatch):
        engine, launched = _fake_engine(settings, monkeypatch, free_gpu=None)

        with pytest.raises(EngineBusy):
            await engine.ensure_running()

        assert launched == [], "it started something anyway"


class TestItGivesTheCardBack:
    @pytest.mark.asyncio
    async def test_it_stops_after_the_idle_timeout(self, settings, monkeypatch):
        engine, _ = _fake_engine(settings, monkeypatch)
        await engine.ensure_running()
        stopped: list[str] = []
        monkeypatch.setattr(
            engine, "stop", lambda reason: stopped.append(reason) or asyncio.sleep(0)
        )

        reaper = asyncio.create_task(engine.reap_when_idle())
        await asyncio.sleep(settings.idle_timeout_seconds + 0.4)
        reaper.cancel()

        assert stopped, "the card was never handed back"

    @pytest.mark.asyncio
    async def test_it_stays_up_while_requests_keep_arriving(self, settings, monkeypatch):
        """The timer measures silence, not uptime.

        Resetting only on the way in would let the timer fire during a long generation and
        kill the engine answering it.
        """
        engine, _ = _fake_engine(settings, monkeypatch)
        await engine.ensure_running()
        stopped: list[str] = []
        monkeypatch.setattr(
            engine, "stop", lambda reason: stopped.append(reason) or asyncio.sleep(0)
        )

        reaper = asyncio.create_task(engine.reap_when_idle())
        for _ in range(6):
            await asyncio.sleep(settings.idle_timeout_seconds / 3)
            engine._last_used = asyncio.get_event_loop().time() and __import__(
                "time"
            ).monotonic()
        reaper.cancel()

        assert stopped == [], f"it stopped a busy engine: {stopped}"

    @pytest.mark.asyncio
    async def test_the_timeout_is_configurable(self, settings):
        assert Settings(idle_timeout_minutes=30.0).idle_timeout_seconds == 1800.0
        assert dataclasses.replace(settings, idle_timeout_minutes=5).idle_timeout_seconds == 300


class TestStartupFailure:
    @pytest.mark.asyncio
    async def test_a_failed_start_leaves_nothing_holding_the_card(self, settings, monkeypatch):
        engine, _ = _fake_engine(settings, monkeypatch, ready=False)

        with pytest.raises(RuntimeError):
            await engine.ensure_running()

        # The next request must be able to try again rather than find a phantom engine.
        assert engine.status()["running"] is False


class TestTheTwoEnginesShareOneCard:
    """Text and image are separate processes and are meant to live on ONE card.

    Separate processes because killing a process is the only reliable way to give GPU memory
    back — unloading a pipeline inside a long-lived server leaves the allocator fragmented and
    the card still spoken for, which would defeat the idle timer entirely.

    One card because the machine is shared: taking a second would double what a colleague
    loses, for a model that fits in what the first one has left.
    """

    @pytest.fixture
    def settings(self, tmp_path):
        return Settings(
            model="test/text",
            image_model="test/image",
            image_required_mib=7000,
            log_dir=tmp_path,
        )

    def test_the_image_engine_prefers_the_card_the_text_one_holds(
        self, settings, monkeypatch
    ):
        from gateway import gpus

        asked: dict = {}

        def pick(required, prefer_index=None):
            asked["required"], asked["prefer"] = required, prefer_index
            return gpus.Gpu(7, "A5000", 24564, 14425, 30, True)

        monkeypatch.setattr(gpus, "pick_shared_or_free", pick)
        images = Engine(settings, "image")
        images._share_with = 7

        assert images._pick_gpu().index == 7
        assert asked == {"required": 7000, "prefer": 7}

    def test_the_text_engine_shares_too(self, settings, monkeypatch):
        """Symmetric, and it was not at first.

        Making only the image engine share meant whichever started FIRST claimed a card and the
        second took another — so asking for a picture before any text quietly cost two cards.
        Order of use is not something a user should have to think about to keep a promise the
        service made. Caught on a real run, not by this test.
        """
        from gateway import gpus

        asked: dict = {}

        def pick(required, prefer_index=None):
            asked["required"], asked["prefer"] = required, prefer_index
            return gpus.Gpu(7, "A5000", 24564, 9513, 20, True)

        monkeypatch.setattr(gpus, "pick_shared_or_free", pick)
        text = Engine(settings, "text")
        text._share_with = 7  # the image engine is already there

        assert text._pick_gpu().index == 7
        assert asked["prefer"] == 7
        assert asked["required"] == settings.text_required_mib

    def test_each_role_asks_for_its_own_budget(self, settings, monkeypatch):
        # vLLM preallocates a share of the whole card; a diffusion pipeline just loads weights.
        from gateway import gpus

        seen: list = []
        monkeypatch.setattr(
            gpus,
            "pick_shared_or_free",
            lambda required, prefer_index=None: seen.append(required) or None,
        )
        Engine(settings, "text")._pick_gpu()
        Engine(settings, "image")._pick_gpu()

        assert seen == [settings.text_required_mib, settings.image_required_mib]

    def test_no_engine_shows_this_project_s_path(self, settings):
        # The box is shared: the visible command line is the title, set before anything runs.
        for role, title in (("text", "llm-engine"), ("image", "image-engine")):
            argv = Engine(settings, role)._command(7)
            assert argv[1:4] == ["-m", "gateway.titled", title]

    def test_each_role_runs_its_own_command(self, settings):
        text = Engine(settings, "text")._command(7)
        image = Engine(settings, "image")._command(7)

        assert "serve" in text and settings.model in text
        assert "gateway.imaged" in image and settings.image_model in image

    def test_each_role_has_its_own_port(self, settings):
        assert Engine(settings, "text").port == settings.engine_port
        assert Engine(settings, "image").port == settings.image_port
        assert settings.engine_port != settings.image_port

    def test_each_role_logs_to_its_own_file(self, settings, monkeypatch):
        # One log for both would interleave two models' startup, which is exactly the moment
        # you need to read one of them.
        monkeypatch.setattr(
            "subprocess.Popen", lambda *a, **k: _FakeProcess()
        )
        Engine(settings, "text")._launch(7)
        Engine(settings, "image")._launch(7)

        names = {p.name for p in settings.log_dir.glob("*.log")}
        assert {"text.log", "image.log"} <= names

    def test_they_report_separately(self, settings):
        assert Engine(settings, "text").status()["role"] == "text"
        assert Engine(settings, "image").status()["model"] == "test/image"
