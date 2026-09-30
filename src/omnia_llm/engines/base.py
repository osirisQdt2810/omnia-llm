"""How a model is served: the :class:`Engine` interface, and :class:`ProcessEngine`.

An engine owns ONE model: it starts it on a device when asked, forwards requests to it, and
stops it when told. Almost every backend is "a server process on a loopback port", so
:class:`ProcessEngine` implements everything that is hard about that once — one start at a time,
a real readiness wait, killing the whole process group, the idle clock — and a backend only says
which command to run (:meth:`ProcessEngine.command`) and what options it takes (``Options``).

Add a backend: subclass ProcessEngine (or Engine, for something that is not a process),
declare an ``Options`` dataclass, and decorate the class with ``@register_engine("name")``.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, ClassVar, Iterable, Optional

import httpx

from omnia_llm.config import ConfigError, ModelSpec, PathsConfig
from omnia_llm.devices import Device, DeviceProbe


class EngineBusy(RuntimeError):
    """No device has room right now. A thing to wait out (503), not a bug to report."""


@dataclasses.dataclass(frozen=True)
class NoOptions:
    """For an engine that takes none."""


class Engine(abc.ABC):
    """One served model. The gateway talks only to this interface."""

    #: The engine's settings, filled from the model's ``[models.options]`` table.
    Options: ClassVar[type] = NoOptions
    #: The device backends (probe names) this engine can run on.
    backends: ClassVar[tuple[str, ...]] = ()
    registry_name: ClassVar[str] = ""

    def __init__(self, spec: ModelSpec, probe: DeviceProbe, paths: PathsConfig) -> None:
        backend = type(probe).registry_name
        if backend not in self.backends:
            raise ConfigError(
                f"model {spec.id!r}: engine {spec.engine!r} runs on "
                f"{', '.join(self.backends) or 'nothing'}, but this machine's devices are "
                f"{backend!r} — pick another engine, or set [devices] probe"
            )
        self.spec = spec
        self.probe = probe
        self.paths = paths
        self.options = self._parse_options(spec)

    @classmethod
    def _parse_options(cls, spec: ModelSpec) -> Any:
        known = {f.name for f in dataclasses.fields(cls.Options)}
        unknown = set(spec.options) - known
        if unknown:
            raise ConfigError(
                f"model {spec.id!r} ({spec.engine}): unknown option(s) "
                f"{', '.join(sorted(unknown))}; known: {', '.join(sorted(known)) or 'none'}"
            )
        try:
            return cls.Options(**spec.options)
        except TypeError as exc:  # a required option is missing
            raise ConfigError(f"model {spec.id!r} ({spec.engine}): {exc}") from None

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    @abc.abstractmethod
    def running(self) -> bool: ...

    @property
    @abc.abstractmethod
    def device(self) -> Optional[Device]: ...

    @abc.abstractmethod
    async def ensure_running(self, prefer: Iterable[str] = ()) -> None:
        """Start the model if it is not up; ``prefer`` = device ids other models already hold.

        Raises:
            EngineBusy: when no device has room.
        """

    @abc.abstractmethod
    async def forward(self, method: str, path: str, body: bytes, headers: dict,
                      prefer: Iterable[str] = ()) -> httpx.Response: ...

    @abc.abstractmethod
    async def stop(self, reason: str) -> None: ...

    @abc.abstractmethod
    def idle_seconds(self) -> float:
        """Seconds since the last request finished (0 when never used)."""

    @abc.abstractmethod
    def status(self) -> dict: ...


class ProcessEngine(Engine):
    """A model served by a child process on a loopback port."""

    #: GET here answers 200 once the process can take requests.
    health_path: ClassVar[str] = "/health"
    startup_timeout_seconds: ClassVar[float] = 900.0
    shutdown_grace_seconds: ClassVar[float] = 20.0

    def __init__(self, spec: ModelSpec, probe: DeviceProbe, paths: PathsConfig, *,
                 python: str = sys.executable) -> None:
        super().__init__(spec, probe, paths)
        self.python = python
        self._process: Optional[subprocess.Popen] = None
        self._device: Optional[Device] = None
        self._lock = asyncio.Lock()
        self._last_used = 0.0
        self._started_at = 0.0
        self._starting = False
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    # --- what a backend declares -------------------------------------------------------------
    @abc.abstractmethod
    def command(self, device: Device) -> list[str]:
        """The argv that serves this model on ``self.spec.port``."""

    def environment(self, device: Device) -> dict[str, str]:
        """The child's environment: pinned to ``device``, downloads kept in the model cache."""
        env = dict(os.environ)
        env.update(dict(device.env))
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env["HF_HOME"] = str(self.paths.hf_home)
        return env

    @property
    def upstream_model(self) -> Optional[str]:
        """What to put in a request's ``model`` before forwarding (None: leave it)."""
        return None

    def titled(self, title: str, module: str, *args: str) -> list[str]:
        """Run ``module``'s ``main()`` under a neutral process title (see workers/titled.py)."""
        return [self.python, "-m", "omnia_llm.workers.titled", title, module, *args]

    # --- state -------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def device(self) -> Optional[Device]:
        return self._device if self.running else None

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used if self._last_used else 0.0

    def status(self) -> dict:
        return {
            "kind": self.kind,
            "engine": self.registry_name,
            "running": self.running,
            "starting": self._starting,
            "device": self._device.id if self.running and self._device else None,
            "idle_seconds": round(self.idle_seconds(), 1),
            "uptime_seconds": round(time.monotonic() - self._started_at, 1) if self.running else 0.0,
        }

    # --- lifecycle ---------------------------------------------------------------------------
    async def ensure_running(self, prefer: Iterable[str] = ()) -> None:
        """Start the model if it is not up. Safe to call from every request.

        Holds the lock across the whole cold start, readiness wait included: a second request
        arriving mid-start must queue behind it, not conclude the model is down and launch a
        second copy onto the same device.
        """
        async with self._lock:
            if self.running:
                return
            device = self.probe.pick(self.spec.required_mib, prefer)
            if device is None:
                raise EngineBusy(f"no device has {self.spec.required_mib} MiB free right now "
                                 f"for {self.id!r} — nothing was started")
            self._starting = True
            try:
                self._launch(device)
                await self._wait_until_ready()
            except BaseException:
                # A start that did not finish holds no device — the invariant lives here, at the
                # only place that owns the lifecycle, so no future failure mode can forget it.
                await self.stop("startup failed")
                raise
            finally:
                self._starting = False

    def _launch(self, device: Device) -> None:
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        self.paths.hf_home.mkdir(parents=True, exist_ok=True)
        # The child gets its own copy of the log's descriptor. The parent's is closed here rather
        # than left to the garbage collector.
        with open(self.paths.logs / f"{self.id}.log", "ab", buffering=0) as log:
            log.write(f"\n=== {self.id} on {device.id} at {time.strftime('%F %T')} ===\n".encode())
            # Its own process group, so stopping it takes the worker children too.
            self._process = subprocess.Popen(self.command(device), env=self.environment(device),
                                             stdout=log, stderr=log, start_new_session=True)
        self._device = device
        self._started_at = self._last_used = time.monotonic()

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        url = f"http://127.0.0.1:{self.spec.port}{self.health_path}"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"{self.id!r} exited with code {self._process.returncode} "
                                   f"during startup — see {self.paths.logs / (self.id + '.log')}")
            with contextlib.suppress(Exception):  # not up yet is the normal case here
                if (await self._client.get(url, timeout=5.0)).status_code == 200:
                    return
            await asyncio.sleep(2.0)
        raise RuntimeError(f"{self.id!r} did not become ready within "
                           f"{self.startup_timeout_seconds:.0f}s")

    async def stop(self, reason: str) -> None:
        """SIGTERM the process GROUP, SIGKILL after a grace period — a leaked worker keeps the
        device memory allocated, which would defeat the idle timer."""
        process, self._process = self._process, None
        device, self._device = self._device, None
        if process is None or process.poll() is not None:
            return
        print(f"[{self.id}] stopping ({device.id if device else '?'}): {reason}", flush=True)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        for _ in range(int(self.shutdown_grace_seconds * 2)):
            if process.poll() is not None:
                return
            await asyncio.sleep(0.5)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    # --- traffic -----------------------------------------------------------------------------
    def rewrite(self, body: bytes) -> bytes:
        """Set the request's ``model`` to :attr:`upstream_model`, when the backend needs it."""
        target = self.upstream_model
        if not target or not body:
            return body
        try:
            payload = json.loads(body)
        except ValueError:
            return body
        if not isinstance(payload, dict):
            return body
        payload["model"] = target
        return json.dumps(payload).encode()

    async def forward(self, method: str, path: str, body: bytes, headers: dict,
                      prefer: Iterable[str] = ()) -> httpx.Response:
        """Send one request, starting the model first if needed.

        The idle clock is reset on the way in AND out: only in would let the timer fire during a
        long generation; only out would let a stalled request look idle.
        """
        await self.ensure_running(prefer)
        self._last_used = time.monotonic()
        try:
            return await self._client.request(method, f"http://127.0.0.1:{self.spec.port}{path}",
                                              content=self.rewrite(body), headers=headers)
        finally:
            self._last_used = time.monotonic()

    async def stop_if_idle(self, timeout_seconds: float) -> bool:
        """Stop when idle for ``timeout_seconds`` — re-checked under the lock, so a request that
        arrived meanwhile is never killed mid-generation."""
        if not self.running or not self._last_used or self.idle_seconds() < timeout_seconds:
            return False
        async with self._lock:
            if self.running and self.idle_seconds() >= timeout_seconds:
                await self.stop(f"idle for {self.idle_seconds() / 60:.1f} min")
                return True
        return False
