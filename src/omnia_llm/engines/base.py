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
import socket
import subprocess
import sys
import time
import typing
from typing import Any, ClassVar, Iterable, Optional

import httpx

from omnia_llm.config import ConfigError, ModelSpec, PathsConfig
from omnia_llm.devices import Device, DeviceProbe


class EngineBusy(RuntimeError):
    """No device has room right now. A thing to wait out (503), not a bug to report."""


def _say(message: str) -> None:
    """Print one lifecycle line, and never raise.

    Closing the terminal the gateway runs in is the SIGHUP that starts its shutdown, and from
    then on stdout is a hung-up tty where a print is an OSError. Raised inside a stop, it would
    leave that model's processes, and every model stopped after it, running.
    """
    with contextlib.suppress(OSError):
        print(message, flush=True)


class _ProcessGroup:
    """A launched model's processes: the leader, and every child it started.

    The group's id is the leader's pid (``start_new_session``), so the group can be signalled
    with the leader gone, and must be: vLLM's EngineCore outlives its launcher, holding the GPU.
    But only while the leader is unreaped. Reaping frees that pid, and the kernel may give it to
    any new process group of this user, a shell job, say. So the leader is reaped here only, in
    :meth:`poll`, which kills what is left of the group at that instant, the last one at which
    the id is surely the group's own, and the group is never signalled again.
    """

    def __init__(self, leader: subprocess.Popen) -> None:
        self.leader = leader
        #: The leader is reaped: the id may be another group's by now.
        self.over = False

    def poll(self) -> Optional[int]:
        """The leader's exit code, or None while it runs."""
        code = self.leader.poll()
        if code is not None and not self.over:
            self.over = True
            # macOS answers EPERM, not ESRCH, for a group left with zombies only.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.leader.pid, signal.SIGKILL)
        return code

    def signal(self, sig: int) -> None:
        """Send ``sig`` to the whole group, unless it is over.

        Safe while the leader is unreaped, even in the moment after it exits: its pid stays
        taken until it is reaped.
        """
        if self.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.leader.pid, sig)

    async def exited(self, within: float) -> bool:
        """Whether the group is over, its leader reaped and the rest killed, within ``within``
        seconds."""
        deadline = time.monotonic() + within
        while self.poll() is None:
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.1)
        return True


@dataclasses.dataclass(frozen=True)
class NoOptions:
    """For an engine that takes none."""


#: What an option of each type takes, in the words an error uses.
_WANTED = {bool: "true or false", int: "a whole number", float: "a number", str: "a string"}


def _fits(hint: Any, value: Any) -> bool:
    """Whether a value, as TOML gives it, is a ``hint``: a bool is no number here, whatever
    Python's subclassing says, and a whole number is a fine float."""
    if isinstance(value, bool):
        return hint is bool
    return isinstance(value, (int, float) if hint is float else hint)


def _option(where: str, name: str, hint: Any, value: Any) -> Any:
    """``value`` as the option ``name``, of type ``hint``, takes it.

    A list becomes a tuple where the option is one, and a whole number a float. Nothing else is
    converted: a string where a list belongs is refused, not split into characters.

    Raises:
        ConfigError: naming the option and what it takes.
    """
    if typing.get_origin(hint) is tuple:
        item = typing.get_args(hint)[0]
        if isinstance(value, (list, tuple)) and all(_fits(item, v) for v in value):
            return tuple(value)
        wanted = f"a list, each item {_WANTED[item]}"
    elif hint in _WANTED:
        if _fits(hint, value):
            return float(value) if hint is float else value
        wanted = _WANTED[hint]
    else:
        return value  # a type with no rule here is the engine's own business
    raise ConfigError(f"{where}: option {name!r} must be {wanted}, not {value!r}")


class Engine(abc.ABC):
    """One served model. The gateway talks only to this interface."""

    #: The engine's settings, filled from the model's ``[models.options]`` table.
    Options: ClassVar[type] = NoOptions
    #: The device backends (probe names) this engine can run on.
    backends: ClassVar[tuple[str, ...]] = ()
    registry_name: ClassVar[str] = ""

    def __init__(self, spec: ModelSpec, probe: DeviceProbe, paths: PathsConfig) -> None:
        self.check_platform(spec, type(probe).registry_name)
        self.spec = spec
        self.probe = probe
        self.paths = paths
        self.options = self.check_options(spec)

    @classmethod
    def check_platform(cls, spec: ModelSpec, backend: str) -> None:
        """Refuse ``spec`` on a machine whose devices are ``backend``, if this engine does not
        run there. Public, like :meth:`check_options`, so ``omnia-llm models`` refuses it too."""
        if backend not in cls.backends:
            raise ConfigError(
                f"model {spec.id!r}: engine {spec.engine!r} runs on "
                f"{', '.join(cls.backends) or 'nothing'}, but this machine's devices are "
                f"{backend!r} — pick another engine, or set [devices] probe"
            )

    @classmethod
    def check_options(cls, spec: ModelSpec) -> Any:
        """``spec``'s options as this engine's ``Options``, or ConfigError naming the mistake.

        Public so that ``omnia-llm models`` checks a config exactly as ``serve`` will: a mistake
        that passes the one and fails the other is a service in a restart loop.
        """
        where = f"model {spec.id!r} ({spec.engine})"
        hints = typing.get_type_hints(cls.Options)
        known = {f.name: hints[f.name] for f in dataclasses.fields(cls.Options)}
        unknown = set(spec.options) - set(known)
        if unknown:
            raise ConfigError(
                f"{where}: unknown option(s) {', '.join(sorted(unknown))}; "
                f"known: {', '.join(sorted(known)) or 'none'}"
            )
        values = {name: _option(where, name, known[name], value)
                  for name, value in spec.options.items()}
        try:
            return cls.Options(**values)
        except (TypeError, ValueError) as exc:  # a required option missing; a bad combination
            raise ConfigError(f"{where}: {exc}") from None

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
    #: How often a start asks whether the process answers yet.
    startup_poll_seconds: ClassVar[float] = 2.0
    shutdown_grace_seconds: ClassVar[float] = 20.0

    def __init__(self, spec: ModelSpec, probe: DeviceProbe, paths: PathsConfig, *,
                 python: str = sys.executable) -> None:
        super().__init__(spec, probe, paths)
        self.python = python
        self._group: Optional[_ProcessGroup] = None
        self._device: Optional[Device] = None
        self._lock = asyncio.Lock()
        #: Set while a stop is under way: a /stop and the shutdown do not take the lock.
        self._stopping: Optional[asyncio.Event] = None
        self._last_used = 0.0
        self._started_at = 0.0
        self._starting = False
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    # --- what a backend declares -------------------------------------------------------------
    @abc.abstractmethod
    def command(self, device: Device) -> list[str]:
        """The argv that serves this model on ``self.spec.port``."""

    def environment(self, device: Device) -> dict[str, str]:
        """The child's environment: pinned to ``device``, downloads kept in the model cache.

        The hub cache moves, HF_HOME does not: that is also where ``huggingface-cli login``
        keeps its token, and without the token a gated model is a 401.
        """
        env = dict(os.environ)
        env.update(dict(device.env))
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env["HF_HUB_CACHE"] = str(self.paths.hf_hub_cache)
        return env

    @property
    def upstream_model(self) -> str:
        """The name the engine serves its model under, given to every request it is sent.

        The model's id, which is vLLM's ``--served-model-name`` and llama.cpp's ``--alias``.
        Always rewritten, because the manager hands an unknown id to the first model of its
        kind: a client still using an old id then reaches the engine under a name it knows,
        instead of getting its 404.
        """
        return self.spec.id

    def titled(self, title: str, module: str, *args: str) -> list[str]:
        """Run ``module``'s ``main()`` under a neutral process title (see workers/titled.py)."""
        return [self.python, "-m", "omnia_llm.workers.titled", title, module, *args]

    # --- state -------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._group is not None and self._group.poll() is None

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
            if self._stopping is not None:
                # A /stop is still ending the last run, whose processes hold the port and the
                # device until they are gone: started now, this would fail on the port.
                await self._stopping.wait()
            if self._group is not None:
                # It exited on its own (a crash, the OOM killer), and what it left behind was
                # killed when that was found: only its bookkeeping is left to clear.
                await self.stop("exited")
            device = self.probe.pick(self.spec.required_mib, prefer)
            if device is None:
                raise EngineBusy(f"no device has {self.spec.required_mib} MiB free right now "
                                 f"for {self.id!r} — nothing was started")
            self._starting = True
            try:
                await self._wait_until_ready(self._launch(device))
            except BaseException:
                # A start that did not finish holds no device — the invariant lives here, at the
                # only place that owns the lifecycle, so no future failure mode can forget it.
                await self.stop("startup failed")
                raise
            finally:
                self._starting = False

    def _launch(self, device: Device) -> _ProcessGroup:
        self._check_port_is_free()
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        self.paths.hf_hub_cache.mkdir(parents=True, exist_ok=True)
        # The child gets its own copy of the log's descriptor. The parent's is closed here rather
        # than left to the garbage collector.
        with open(self.paths.logs / f"{self.id}.log", "ab", buffering=0) as log:
            log.write(f"\n=== {self.id} on {device.id} at {time.strftime('%F %T')} ===\n".encode())
            # Its own process group, so stopping it takes the worker children too.
            group = _ProcessGroup(subprocess.Popen(
                self.command(device), env=self.environment(device), stdout=log, stderr=log,
                start_new_session=True))
        self._group, self._device = group, device
        self._started_at = self._last_used = time.monotonic()
        return group

    def _check_port_is_free(self) -> None:
        """Refuse a port something already listens on.

        Most likely an engine left behind by a gateway that died: it would answer the new
        start's health check, then take its traffic, from a device nobody accounts for.
        Bound with SO_REUSEADDR, as the engines' own servers bind: a connection from the last
        run still in TIME_WAIT is not a listener.

        No defence against other users of this machine. The engines' ports take no token, so
        anyone here can reach a model directly, or bind its port between this check and the
        engine's own bind; and on macOS, a listener on 0.0.0.0 passes the check.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", self.spec.port))
            except OSError:
                raise RuntimeError(f"port {self.spec.port} is in use (a leftover engine?); "
                                   f"nothing was started for {self.id!r}") from None

    async def _wait_until_ready(self, group: _ProcessGroup) -> None:
        """Poll until ``group`` answers its health check.

        Asked of the group THIS start launched, not of whatever ``_group`` holds now: a /stop
        does not wait for the lock a start holds, and a start that only checked ``_group``
        would go on polling a dead port for the whole startup timeout.
        """
        deadline = time.monotonic() + self.startup_timeout_seconds
        url = f"http://127.0.0.1:{self.spec.port}{self.health_path}"
        while time.monotonic() < deadline:
            ready = False
            with contextlib.suppress(Exception):  # not up yet is the normal case here
                ready = (await self._client.get(url, timeout=5.0)).status_code == 200
            # After the health check, which a stop may have happened during.
            if self._group is not group:
                raise RuntimeError(f"{self.id!r} was stopped while starting")
            code = group.poll()
            if code is not None:
                # Relative on purpose: this reaches every token holder in a 502, and the full
                # path would tell them the server's user name.
                raise RuntimeError(f"{self.id!r} exited with code {code} during startup; see "
                                   f"logs/{self.id}.log in the server's state folder")
            if ready:
                return
            await asyncio.sleep(self.startup_poll_seconds)
        raise RuntimeError(f"{self.id!r} did not become ready within "
                           f"{self.startup_timeout_seconds:.0f}s")

    async def stop(self, reason: str) -> None:
        """End the process GROUP: SIGTERM, then SIGKILL once the grace period is up.

        The group, not only the leader: a worker left running keeps the device's memory, which
        would defeat the idle timer. A group that is already over (see :class:`_ProcessGroup`)
        is not signalled at all, and only its bookkeeping is cleared. Killed first, logged
        after (see :func:`_say`).
        """
        group, self._group = self._group, None
        device, self._device = self._device, None
        if group is None:
            if self._stopping is not None:
                # Another stop has the group: done is when that one is, or a shutdown arriving
                # during a /stop would leave the old engine to nobody.
                await self._stopping.wait()
            return
        stopped = self._stopping = asyncio.Event()
        try:
            group.signal(signal.SIGTERM)
            _say(f"[{self.id}] stopping ({device.id if device else '?'}): {reason}")
            if not await group.exited(within=self.shutdown_grace_seconds):
                group.signal(signal.SIGKILL)
                # SIGKILL cannot be refused: this only waits for the leader to go, and reaps it.
                await group.exited(within=5.0)
        finally:
            stopped.set()
            if self._stopping is stopped:
                self._stopping = None

    # --- traffic -----------------------------------------------------------------------------
    def rewrite(self, body: bytes) -> bytes:
        """Set the request's ``model`` to :attr:`upstream_model`; anything but a JSON object is
        forwarded untouched."""
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
