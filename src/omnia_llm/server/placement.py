"""Starting a model: with which settings, on which device, and making room there for it.

Both are decided at the moment a model starts, never before:

- Its settings: its ``when_cpu_busy`` profile when the host's CPU is busy then, its own otherwise
  (:class:`~omnia_llm.devices.CpuLoad`). A model that computes part of itself on the CPU is fast
  on an idle one and crawls on a saturated one.
- Its device: as the probe picks it, or with ``[devices] single_device`` the one the running
  models hold and no other. A model that does not fit there waits until the models on it have
  no request in flight, stops them, and takes their place: the models take turns on the device.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import time
from typing import AsyncIterator, Callable, ClassVar, Mapping, Optional

from omnia_llm.devices import CpuLoad, DeviceProbe
from omnia_llm.engines import Engine, EngineBusy


@dataclasses.dataclass(frozen=True)
class Start:
    """How an engine is to start when a request finds it not running: the arguments of its
    :meth:`~omnia_llm.engines.Engine.ensure_running`."""

    prefer: tuple[str, ...]
    single_device: bool
    cpu: Optional[CpuLoad]


class Placement:
    """Where the models start, and which requests are in flight on each."""

    #: How long a start waits for the models in its way to finish their requests.
    drain_timeout_seconds: ClassVar[float] = 300.0
    #: How often a waiting start asks whether they have.
    drain_poll_seconds: ClassVar[float] = 0.1

    def __init__(self, engines: Mapping[str, Engine], probe: DeviceProbe, *,
                 single_device: bool, cpu_load: Callable[[], Optional[CpuLoad]]) -> None:
        self.engines = engines
        self.probe = probe
        self.single_device = single_device
        #: The host's CPU load now; injectable, so a test decides whether the CPU is busy.
        self.cpu_load = cpu_load
        #: One start at a time on the single device. Two models racing for it would each find
        #: the other in the way and wait for it to stop, or both fit by the same measurement
        #: and overfill it.
        self._lock = asyncio.Lock()
        self._in_flight: collections.Counter[str] = collections.Counter()
        #: Models a start is waiting to stop. Their new requests queue behind that start: let
        #: in, a steady stream of them would keep the waiting model from ever getting its turn.
        self._draining: set[str] = set()

    def held_devices(self, exclude: Engine) -> list[str]:
        """Devices other running models hold — tried first, so models share one device."""
        return [e.device.id for e in self.engines.values()
                if e is not exclude and e.running and e.device is not None]

    def in_flight(self, engine: Engine) -> int:
        return self._in_flight[engine.id]

    @contextlib.asynccontextmanager
    async def admitted(self, engine: Engine) -> AsyncIterator[Start]:
        """One request for ``engine``: counted in flight, so not stopped to make room until it is
        done, and given how the engine is to start if it is not running.

        On a single device the model is started here, and counted with no wait between the
        start and the count: a start queued behind this one cannot stop the model before the
        request that started it is sent. Otherwise the engine starts itself, as it always has.
        """
        if self.single_device and (not engine.running or engine.id in self._draining):
            await self.start(engine, make_room=True)
        self._in_flight[engine.id] += 1
        try:
            yield Start(tuple(self.held_devices(engine)), self.single_device,
                        None if engine.running else self.cpu_load())
        finally:
            self._in_flight[engine.id] -= 1

    async def start(self, engine: Engine, *, make_room: bool = False) -> None:
        """Start ``engine`` unless it runs, with the settings the CPU calls for now.

        On a single device, ``make_room`` lets it stop the models in its way; without, a model
        that does not fit beside them is refused (a warm-up should not take a model's turn).

        Raises:
            EngineBusy: when there is no room, or the models in the way stay busy too long.
        """
        if not self.single_device:
            if not engine.running:
                await engine.ensure_running(self.held_devices(engine), cpu=self.cpu_load())
            return
        async with self._lock:
            if engine.running:
                return
            cpu = self.cpu_load()
            need = engine.spec.profile(cpu_busy=cpu is not None and cpu.busy).required_mib
            prefer = await self._room_for(engine, need, make_room)
            await engine.ensure_running(prefer, single_device=True, cpu=cpu)

    async def _room_for(self, engine: Engine, need: int, make_room: bool) -> tuple[str, ...]:
        """The device ``engine`` must start on, as ``prefer``, with ``need`` MiB there for it:
        the one the running models hold, emptied of them as far as it takes (all of them
        included), or none when no model held one as the start began (the engine then picks a
        free device)."""
        deadline = time.monotonic() + self.drain_timeout_seconds
        told = False
        #: The device this start stopped models on. Emptied by it, it is still the one device:
        #: picked afresh, a card ranked above it would become a second.
        emptied: Optional[str] = None
        try:
            while True:
                residents = [e for e in self.engines.values()
                             if e is not engine and e.running and e.device is not None]
                if not residents:
                    return (emptied,) if emptied else ()
                device_id = residents[0].device.id
                residents = [e for e in residents if e.device.id == device_id]
                if self._room(device_id, residents) >= need:
                    return (device_id,)
                names = ", ".join(repr(e.id) for e in residents)
                if not make_room:
                    raise EngineBusy(f"{engine.id!r} does not fit beside {names} on {device_id}; "
                                     "it starts when it is asked for")
                self._draining.update(e.id for e in residents)
                # Waiting, only the requests in flight change, not what the card holds: watch
                # them, and measure the card again only once a model has left it. Measured on
                # every poll, a long request kept the GPU tool forking on the event loop, which
                # stalled the very request being waited for.
                idle = None
                while all(e.running for e in residents):
                    idle = next((e for e in residents if not self._in_flight[e.id]), None)
                    if idle is not None:
                        break
                    if time.monotonic() >= deadline:
                        raise EngineBusy(f"{names} still had requests in flight after "
                                         f"{self.drain_timeout_seconds:.0f}s; {engine.id!r} was "
                                         "not started")
                    if not told:
                        told = True
                        print(f"[{engine.id}] waiting for {names} to finish on {device_id}, to "
                              "take its place", flush=True)
                    await asyncio.sleep(self.drain_poll_seconds)
                emptied = device_id
                if idle is not None:
                    # No wait between the check and the stop: no request can slip in between.
                    await idle.stop(f"making room for {engine.id!r}")
        finally:
            self._draining.clear()

    def _room(self, device_id: str, residents: list[Engine]) -> int:
        """MiB a new model can count on there: what is free now, and no more than the models on
        it leave of what they declared. A model holding less than that now (an image model
        between images) grows back into it, and a newcomer packed into the gap runs it out."""
        device = next((d for d in self.probe.list_devices() if d.id == device_id), None)
        if device is None:
            return 0
        declared = sum(e.required_mib for e in residents)
        return min(device.free_mib, device.memory_total_mib - declared)
