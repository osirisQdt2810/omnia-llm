"""Engines: the lazy lifecycle (start on demand, once; stop when idle), options and platform checks."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import re
import signal
import socket
import subprocess
import sys
import time

import pytest

import omnia_llm.platforms  # noqa: F401 - registration
from omnia_llm.config import ConfigError
from omnia_llm.engines import ENGINES, EngineBusy
from tests.helpers import fake_probe, free_port, gpu, spec


class _FakeGroup:
    """Stands in for the model's processes: alive until something ends them."""

    def __init__(self):
        self.code = None

    def poll(self):
        return self.code


def _engine(paths, *, probe=None, ready=True, **spec_kw):
    s = spec(**spec_kw)
    engine = ENGINES.get(s.engine)(s, probe or fake_probe([gpu(7)]), paths)
    launched = []

    def launch(device):
        launched.append(device.id)
        engine._group = _FakeGroup()
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

        async def loading(group):
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
        engine, _ = _engine(paths, probe=fake_probe([gpu(0, busy=True)]))
        with pytest.raises(EngineBusy):
            await engine.ensure_running()

    async def test_it_prefers_the_card_another_model_holds(self, paths):
        probe = fake_probe([gpu(2), gpu(7, used=13800, busy=True)])
        engine, launched = _engine(paths, probe=probe, required_mib=5000)
        await engine.ensure_running(prefer=["nvidia:7"])
        assert launched == ["nvidia:7"]

    async def test_a_model_that_died_is_stopped_before_it_starts_again(self, paths):
        """A leader that exited on its own (a crash, the OOM killer) may have left children
        holding the device the new start is about to pick."""
        engine, launched = _engine(paths)
        reasons, stop = [], engine.stop

        async def recorded(reason):
            reasons.append(reason)
            await stop(reason)

        engine.stop = recorded
        await engine.ensure_running()
        engine._group.code = -9
        await engine.ensure_running()
        assert reasons == ["exited"] and launched == ["nvidia:7", "nvidia:7"]


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


SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]
#: Exits at once, as a model server that crashed.
CRASHES = [sys.executable, "-c", "raise SystemExit(1)"]
#: Exits at once, leaving a child behind in its process group, as vLLM's launcher can.
LEAVES_A_CHILD = [sys.executable, "-c", "import subprocess, sys; "
                  "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])"]
IGNORES_SIGTERM = [sys.executable, "-c", "import signal, time; "
                   "signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); "
                   "time.sleep(60)"]


def _alive(group: int) -> bool:
    """Whether any process in ``group`` is alive. macOS answers EPERM, not ESRCH, for a group
    left with zombies only."""
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


async def _until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        await asyncio.sleep(0.01)


class _HungUpTerminal:
    """stdout after the terminal closed: every write is an I/O error."""

    def write(self, _text):
        raise OSError(errno.EIO, "Input/output error")

    def flush(self):
        raise OSError(errno.EIO, "Input/output error")


@pytest.fixture
def real(paths):
    """Engines that launch a real process (``command``) on a free port; none outlives its test."""
    groups = []

    def make(command, **spec_kw):
        spec_kw.setdefault("port", free_port())
        engine = ENGINES.get("vllm")(spec(**spec_kw), fake_probe([gpu(7)]), paths)
        engine.command = lambda device: command
        engine.startup_poll_seconds = 0.05
        launch = engine._launch

        def launched(device):
            group = launch(device)
            groups.append(group)
            return group

        engine._launch = launched
        return engine

    yield make
    for group in groups:
        # Only an unreaped leader keeps the id the group's own; a reaped one's is not ours.
        if group.leader.returncode is None:
            group.signal(signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                group.leader.wait(timeout=5)


class TestItsProcesses:
    async def test_a_stop_during_a_start_ends_the_start_at_once(self, real):
        """/stop does not wait for the lock a start holds. The start must see that its process
        is gone and give up, not poll a dead port for the whole startup timeout."""
        engine = real(SLEEPER)
        start = asyncio.create_task(engine.ensure_running())
        await _until(lambda: engine._group is not None)
        await engine.stop("asked to stop")
        with pytest.raises(RuntimeError, match="stopped while starting"):
            await asyncio.wait_for(start, timeout=5)
        assert not engine.running and engine.status()["device"] is None

    async def test_a_stop_ends_the_children_even_when_the_leader_has_exited(self, real):
        """vLLM's EngineCore outlives its launcher. Returning early because the leader was gone
        left it running, holding the GPU."""
        engine = real(LEAVES_A_CHILD)
        leader = engine._launch(gpu(7)).leader
        leader.wait(timeout=10)
        assert _alive(leader.pid), "the child should outlive its leader"
        await engine.stop("idle")
        await _until(lambda: not _alive(leader.pid))

    async def test_what_a_crashed_model_left_behind_is_killed_when_the_crash_is_found(self, real):
        """The instant its leader is reaped is the last at which the group's id is surely its
        own: its children (vLLM's EngineCore, holding the GPU) are killed then, not at some
        later stop."""
        engine = real(LEAVES_A_CHILD)
        group = engine._launch(gpu(7)).leader.pid
        await _until(lambda: not engine.running)  # as /status or the reaper would find it
        await _until(lambda: not _alive(group))

    async def test_a_crashed_model_is_never_signalled_once_it_is_reaped(self, real):
        """Reaping the leader frees its pid, and the kernel may give it to any new process group
        of this user. Signalled later by a /stop, a shutdown or the next start, that group would
        be somebody else's: here, a stand-in given the same id."""
        engine = real(CRASHES)
        engine._launch(gpu(7))
        await _until(lambda: not engine.running)
        victim = subprocess.Popen(SLEEPER, start_new_session=True)
        try:
            engine._group.leader.pid = victim.pid
            await engine.stop("asked to stop")
            await asyncio.sleep(0.2)
            assert victim.poll() is None, "a process group that is not the model's was signalled"
        finally:
            victim.kill()
            victim.wait()

    async def test_a_process_that_ignores_sigterm_is_killed_after_the_grace_period(self, real,
                                                                                    paths):
        engine = real(IGNORES_SIGTERM)
        engine.shutdown_grace_seconds = 0.3
        group = engine._launch(gpu(7)).leader.pid
        await _until(lambda: "ready" in (paths.logs / "omnia-local.log").read_text())
        await engine.stop("idle")
        assert not _alive(group)

    async def test_a_terminal_that_went_away_does_not_interrupt_a_stop(self, real, monkeypatch):
        """Closing the terminal is the SIGHUP that starts a shutdown, and from then on a print
        is an OSError: raised mid-stop, it would leave this model, and every model stopped
        after it, running."""
        engine = real(LEAVES_A_CHILD)
        group = engine._launch(gpu(7)).leader.pid
        monkeypatch.setattr(sys, "stdout", _HungUpTerminal())
        await engine.stop("gateway shutting down")
        await _until(lambda: not _alive(group))

    async def test_a_port_in_use_is_refused_before_anything_starts(self, real):
        """Most likely an engine a gateway that died left behind: it would answer the new
        start's health check, then its traffic, from a device nobody accounts for."""
        with socket.socket() as squatter:
            squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            squatter.bind(("127.0.0.1", 0))
            squatter.listen()
            port = squatter.getsockname()[1]
            engine = real(SLEEPER, port=port)
            engine.startup_timeout_seconds = 1
            with pytest.raises(RuntimeError,
                               match=re.escape(f"port {port} is in use (a leftover engine?)")):
                await engine.ensure_running()
        assert engine._group is None and engine.status()["device"] is None

    async def test_a_start_that_exits_names_its_log_but_not_where_it_lives(self, real, paths):
        """The message reaches token holders in a 502, and an absolute path would tell them the
        server's user name."""
        engine = real([sys.executable, "-c", "raise SystemExit(3)"])
        with pytest.raises(RuntimeError) as failure:
            await engine.ensure_running()
        message = str(failure.value)
        assert "exited with code 3" in message
        assert "see logs/omnia-local.log in the server's state folder" in message
        assert str(paths.state) not in message


class TestConfigIsChecked:
    def test_an_engine_refuses_a_platform_it_does_not_run_on(self, paths):
        with pytest.raises(ConfigError, match="runs on apple"):
            ENGINES.get("mlx")(spec(engine="mlx"), fake_probe(backend="nvidia"), paths)

    def test_an_unknown_option_is_refused(self, paths):
        with pytest.raises(ConfigError, match="unknown option"):
            ENGINES.get("vllm")(spec(max_len=9), fake_probe(), paths)

    def test_a_missing_required_option_is_refused(self, paths):
        s = spec()
        s.options.pop("model")
        with pytest.raises(ConfigError):
            ENGINES.get("vllm")(s, fake_probe(), paths)

    def test_a_list_option_is_read_as_a_tuple(self, paths):
        engine = ENGINES.get("vllm")(spec(extra_args=["--enforce-eager"]), fake_probe(), paths)
        assert engine.options.extra_args == ("--enforce-eager",)

    @pytest.mark.parametrize("options", [
        {"extra_args": "--threads 4"},  # split into characters, it would pass eleven arguments
        {"extra_args": ["--threads", 4]},
        {"max_model_len": "4096"},
        {"max_model_len": 4096.0},
        {"max_model_len": True},
        {"gpu_memory_utilization": "0.5"},
        {"model": 14},
    ])
    def test_an_option_of_the_wrong_type_is_refused(self, paths, options):
        with pytest.raises(ConfigError, match=f"option {next(iter(options))!r} must be"):
            ENGINES.get("vllm")(spec(**options), fake_probe(), paths)

    def test_a_switch_takes_true_or_false_only(self, paths):
        with pytest.raises(ConfigError, match="true or false"):
            ENGINES.get("diffusers")(spec(kind="image", engine="diffusers", cpu_offload="no"),
                                     fake_probe(), paths)

    def test_a_whole_number_is_a_fine_fraction(self, paths):
        engine = ENGINES.get("vllm")(spec(gpu_memory_utilization=1), fake_probe(), paths)
        assert isinstance(engine.options.gpu_memory_utilization, float)


class TestTheChild:
    def test_the_child_is_pinned_and_keeps_weights_in_the_model_cache(self, paths):
        engine, _ = _engine(paths)
        env = engine.environment(gpu(3))
        assert env["CUDA_VISIBLE_DEVICES"] == "3"
        assert env["HF_HUB_CACHE"] == str(paths.model_cache / "huggingface" / "hub")

    def test_a_hugging_face_login_stays_visible_to_the_child(self, paths, monkeypatch):
        """HF_HOME is also where `huggingface-cli login` keeps its token: moved, a gated model
        is a 401."""
        monkeypatch.setenv("HF_HOME", "/opt/hf-home")
        engine, _ = _engine(paths)
        assert engine.environment(gpu(3))["HF_HOME"] == "/opt/hf-home"


def test_starting_a_model_leaks_no_file_descriptor(paths, monkeypatch):
    """The parent must not hold the log open: the child has its own copy, and a descriptor kept
    per start would run a gateway that restarts idle models all day out of them."""
    engine = ENGINES.get("vllm")(spec(port=free_port()), fake_probe([gpu(7)]), paths)
    monkeypatch.setattr(engine, "command", lambda device: SLEEPER)
    before = len(os.listdir("/dev/fd"))
    leader = engine._launch(gpu(7)).leader
    try:
        assert len(os.listdir("/dev/fd")) == before
        assert "=== omnia-local on nvidia:7" in (paths.logs / "omnia-local.log").read_text()
    finally:
        leader.kill()
        leader.wait()
