"""The models this gateway serves: building their engines, routing to them, sharing devices."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Iterable, Optional

from omnia_llm.config import AppConfig
from omnia_llm.devices import DeviceProbe
from omnia_llm.engines import ENGINES, Engine


class ModelManager:
    """One engine per configured model, and the rules for which one answers a request."""

    def __init__(self, config: AppConfig, probe: DeviceProbe) -> None:
        import omnia_llm.platforms  # noqa: F401 - registers the platform-only engines

        self.config = config
        self.probe = probe
        self.engines: dict[str, Engine] = {
            spec.id: ENGINES.get(spec.engine)(spec, probe, config.paths) for spec in config.models
        }
        self._warming: dict[str, asyncio.Task] = {}
        self._reaper: Optional[asyncio.Task] = None

    # --- routing -----------------------------------------------------------------------------
    def route(self, kind: str, requested: str = "") -> Optional[Engine]:
        """The engine for ``kind`` that serves ``requested``; the first of that kind otherwise.

        Lenient on purpose: a client configured with an older or shorter name ("sdxl" for
        "sdxl-turbo") keeps working, and a server with one model of a kind has no ambiguity to
        resolve. Exact ids win when they match.
        """
        engine = self.engines.get(requested)
        if engine is not None and engine.kind == kind:
            return engine
        return next((e for e in self.engines.values() if e.kind == kind), None)

    def held_devices(self, exclude: Engine) -> list[str]:
        """Devices other running models hold — tried first, so models share one device."""
        return [e.device.id for e in self.engines.values()
                if e is not exclude and e.running and e.device is not None]

    def listing(self) -> list[dict]:
        """``GET /v1/models``: from config, without starting anything."""
        return [{"id": e.id, "object": "model", "owned_by": "omnia-llm", "kind": e.kind}
                for e in self.engines.values()]

    # --- lifecycle ---------------------------------------------------------------------------
    async def forward(self, engine: Engine, method: str, path: str, body: bytes, headers: dict):
        return await engine.forward(method, path, body, headers, self.held_devices(engine))

    def warm(self, kinds: Iterable[str]) -> list[str]:
        """Start every model of ``kinds`` in the background; returns what is warming."""
        for engine in self.engines.values():
            if engine.kind in kinds and not engine.running and engine.id not in self._warming:
                self._warming[engine.id] = asyncio.create_task(self._warm_one(engine))
        return sorted(self._warming)

    async def _warm_one(self, engine: Engine) -> None:
        try:
            await engine.ensure_running(self.held_devices(engine))
        except Exception as exc:  # noqa: BLE001 - reported in /status, never raised into the loop
            print(f"[{engine.id}] warm-up failed: {exc}", flush=True)
        finally:
            self._warming.pop(engine.id, None)

    def status(self) -> dict:
        out = {}
        for engine_id, engine in self.engines.items():
            state = engine.status()
            state["starting"] = bool(state.get("starting")) or engine_id in self._warming
            out[engine_id] = state
        return out

    async def stop_all(self, reason: str) -> None:
        """Stop every engine at once, so a shutdown takes as long as the slowest stop rather than
        the sum of them all, and one that fails does not leave the others running."""
        engines = list(self.engines.values())
        results = await asyncio.gather(*(e.stop(reason) for e in engines), return_exceptions=True)
        for engine, result in zip(engines, results, strict=True):
            if isinstance(result, BaseException):
                print(f"[{engine.id}] stop failed: {result!r}", flush=True)

    def start_reaper(self) -> None:
        self._reaper = asyncio.create_task(self._reap_forever())

    async def _reap_forever(self) -> None:
        timeout = self.config.server.idle_timeout_seconds
        while True:
            await asyncio.sleep(self.config.server.reaper_interval_seconds)
            for engine in self.engines.values():
                stop_if_idle = getattr(engine, "stop_if_idle", None)
                if stop_if_idle is not None:
                    await stop_if_idle(timeout)

    async def shutdown(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
        # A warm-up still starting could otherwise finish its launch after stop_all has looked,
        # and leave an engine holding its device once the gateway is gone. A start cancelled
        # part-way stops what it launched (Engine.ensure_running).
        warming = list(self._warming.values())
        for task in warming:
            task.cancel()
        await asyncio.gather(*warming, return_exceptions=True)
        await self.stop_all("gateway shutting down")
