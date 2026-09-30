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
import types

import pytest

import omnia_llm.platforms  # noqa: F401 - registration
from omnia_llm.config import ConfigError
from omnia_llm.engines import ENGINES, EngineBusy
from omnia_llm.engines.base import _ProcessGroup
from tests.helpers import (
    IGNORES_SIGTERM,
    fake_lifecycle,
    fake_probe,
    free_port,
    gpu,
    group_alive,
    spec,
    until,
)


def _engine(paths, *, probe=None, ready=True, **spec_kw):
    s = spec(**spec_kw)
    engine = ENGINES.get(s.engine)(s, probe or fake_probe([gpu(7)]), paths)
    return engine, fake_lifecycle(engine, ready=ready)


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
#: Answers /health on {port}, and takes a second to exit on SIGTERM, holding the port meanwhile,
#: as vLLM does while it winds down.
SLOW_TO_EXIT = [sys.executable, "-c", """
import os, signal, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass

signal.signal(signal.SIGTERM, lambda *_: threading.Timer(1.0, os._exit, (0,)).start())
HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
""", "{port}"]


#: Binds argv[1], then forks a worker that inherits the listening socket and ignores SIGTERM,
#: as vLLM's API server binds before it forks EngineCore. The leader goes first: on SIGTERM, or
#: on its own after argv[2] seconds, saying "crashing".
_PORT_TO_A_CHILD = """
import os, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass

server = HTTPServer(("127.0.0.1", int(sys.argv[1])), Health)
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.1)
signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[2]) or 3600)
print("crashing", flush=True)
os._exit(9)
"""


def _hands_its_port_to_a_child(crash_after: float = 0.0) -> list[str]:
    return [sys.executable, "-c", _PORT_TO_A_CHILD, "{port}", str(crash_after)]


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
        argv = [part.replace("{port}", str(spec_kw["port"])) for part in command]
        engine.command = lambda device: argv
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
        await until(lambda: engine._group is not None)
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
        assert group_alive(leader.pid), "the child should outlive its leader"
        await engine.stop("idle")
        await until(lambda: not group_alive(leader.pid))

    async def test_what_a_crashed_model_left_behind_is_killed_when_the_crash_is_found(self, real):
        """The instant its leader is reaped is the last at which the group's id is surely its
        own: its children (vLLM's EngineCore, holding the GPU) are killed then, not at some
        later stop."""
        engine = real(LEAVES_A_CHILD)
        group = engine._launch(gpu(7)).leader.pid
        await until(lambda: not engine.running)  # as /status or the reaper would find it
        await until(lambda: not group_alive(group))

    async def test_a_forced_quit_during_a_stop_still_ends_the_model(self, real, paths):
        """A second Ctrl+C makes the interpreter cancel every task, the ending itself included,
        and a model that ignores SIGTERM used to be left running with its device mid-grace."""
        engine = real(IGNORES_SIGTERM)
        engine.shutdown_grace_seconds = 30
        group = engine._launch(gpu(7))
        await until(lambda: "ready" in (paths.logs / "omnia-local.log").read_text())
        stop = asyncio.create_task(engine.stop("gateway shutting down"))
        await until(lambda: engine._stopping is not None)
        await asyncio.sleep(0.1)  # SIGTERM sent and ignored: inside the grace period now
        engine._stopping.cancel()  # what asyncio.run's teardown does after a forced quit
        with contextlib.suppress(asyncio.CancelledError):
            await stop
        # Killed, not merely asked: it ignores SIGTERM. Reaped here because the task that would
        # have reaped it is the one cancelled; after a real forced quit, init reaps it (on
        # Linux a zombie still counts as a member of its group until then).
        assert group.leader.wait(timeout=5) == -signal.SIGKILL
        assert not group_alive(group.leader.pid)

    async def test_a_crashed_model_is_never_signalled_once_it_is_reaped(self, real):
        """Reaping the leader frees its pid, and the kernel may give it to any new process group
        of this user. Signalled later by a /stop, a shutdown or the next start, that group would
        be somebody else's: here, a stand-in given the same id."""
        engine = real(CRASHES)
        engine.kill_wait_seconds = 0.3  # the stand-in keeps gone() waiting out its bound
        engine._launch(gpu(7))
        await until(lambda: not engine.running)
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
        await until(lambda: "ready" in (paths.logs / "omnia-local.log").read_text())
        await engine.stop("idle")
        assert not group_alive(group)

    async def test_a_terminal_that_went_away_does_not_interrupt_a_stop(self, real, monkeypatch):
        """Closing the terminal is the SIGHUP that starts a shutdown, and from then on a print
        is an OSError: raised mid-stop, it would leave this model, and every model stopped
        after it, running."""
        engine = real(LEAVES_A_CHILD)
        group = engine._launch(gpu(7)).leader.pid
        monkeypatch.setattr(sys, "stdout", _HungUpTerminal())
        await engine.stop("gateway shutting down")
        await until(lambda: not group_alive(group))

    async def test_a_request_during_a_stop_waits_for_the_old_engine_to_go(self, real):
        """A /stop does not take the lock. Let through at once, the next request found the old
        engine still holding its port as it wound down, and failed blaming a leftover."""
        engine = real(SLOW_TO_EXIT)
        await engine.ensure_running()
        stopping = asyncio.create_task(engine.stop("asked to stop"))
        await asyncio.sleep(0.2)  # signalled, and waiting for the old engine to exit
        await engine.ensure_running()
        assert engine.running
        await stopping

    async def test_a_stop_is_over_as_soon_as_the_engine_has_exited(self, real):
        """Not at the end of the grace period: a /stop, and a request waiting on it, take as long
        as the engine takes to wind down, and no longer."""
        engine = real(SLOW_TO_EXIT)
        await engine.ensure_running()
        started = time.monotonic()
        await engine.stop("asked to stop")
        assert time.monotonic() - started < engine.shutdown_grace_seconds / 2

    async def test_a_request_given_up_during_a_stop_leaves_the_stop_running(self, real, paths):
        """A client that goes away cancels its request. Waiting on the stop, the request must
        not take the stop with it, or a model ignoring SIGTERM is never killed."""
        engine = real(IGNORES_SIGTERM)
        engine.shutdown_grace_seconds = 0.5
        group = engine._launch(gpu(7))
        await until(lambda: "ready" in (paths.logs / "omnia-local.log").read_text())
        stopping = asyncio.create_task(engine.stop("asked to stop"))
        await until(lambda: engine._stopping is not None)
        request = asyncio.create_task(engine.ensure_running())
        await asyncio.sleep(0.05)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await stopping
        assert not group_alive(group.leader.pid)

    async def test_a_second_stop_waits_for_the_first_to_finish(self, real):
        """The gateway shutting down while a /stop is under way must not leave before the old
        engine is gone: nothing would be left to finish the stop."""
        engine = real(SLOW_TO_EXIT)
        await engine.ensure_running()
        group = engine._group
        first = asyncio.create_task(engine.stop("asked to stop"))
        await asyncio.sleep(0.2)
        await engine.stop("gateway shutting down")
        assert group.over
        await first

    async def test_a_request_right_after_a_stop_starts_a_fresh_engine(self, real):
        """The worker the leader forked holds the leader's port. Killed once the leader is gone,
        it keeps the port until it is gone too, and a stop that returned before then sent the
        next request into "port in use"."""
        engine = real(_hands_its_port_to_a_child())
        await engine.ensure_running()
        await engine.stop("asked to stop")
        await engine.ensure_running()
        assert engine.running

    async def test_a_request_that_finds_a_crash_starts_a_fresh_engine(self, real, paths):
        """The same worker, left behind by a crash the request itself discovers: the request
        must wait for the killed worker to be gone before it picks a device and a port."""
        engine = real(_hands_its_port_to_a_child(crash_after=1.0))
        await engine.ensure_running()
        await until(lambda: "crashing" in (paths.logs / "omnia-local.log").read_text())
        await asyncio.sleep(0.2)  # the leader has exited, and nothing has asked about it yet
        await engine.ensure_running()
        assert engine.running

    async def test_a_stop_returns_only_once_its_group_is_gone(self, real, monkeypatch):
        """What was killed with or after the leader may still be dying, holding the device's
        memory, which a start let through would count as taken."""
        events = []

        async def gone(group, within):
            events.append("waiting")
            await asyncio.sleep(0.2)
            events.append("gone")
            return True

        monkeypatch.setattr(_ProcessGroup, "gone", gone)
        engine = real(SLEEPER)
        engine._launch(gpu(7))
        await engine.stop("asked to stop")
        assert events == ["waiting", "gone"]

    async def test_a_stop_waits_for_its_port_to_come_free(self, real):
        """macOS frees a dead listener's port up to a millisecond after its process is gone: a
        start straight after would find it taken, and blame a leftover."""
        engine = real(SLEEPER)
        engine._launch(gpu(7))
        with socket.socket() as late:  # the port, still held a moment after the engine is gone
            late.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            late.bind(("127.0.0.1", engine.spec.port))
            late.listen()
            asyncio.get_running_loop().call_later(0.2, late.close)
            await engine.stop("asked to stop")
            assert engine._port_is_free()

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


class TestTheGroupOnceItIsOver:
    """After the leader is reaped a real signal may reach a stranger, so the group is only ever
    asked with signal 0 again. No process is signalled here: os.killpg is a stand-in."""

    @pytest.fixture
    def killpg(self, monkeypatch):
        sent, answers = [], []

        def fake(pgid, sig):
            sent.append(sig)
            answer = answers.pop(0) if answers else None
            if answer is not None:
                raise answer

        monkeypatch.setattr(os, "killpg", fake)
        return sent, answers

    @pytest.mark.parametrize("end", [ProcessLookupError, PermissionError])
    async def test_its_members_are_waited_for_with_signal_zero_only(self, killpg, end):
        """PermissionError: macOS, for a group of zombies only."""
        sent, answers = killpg
        answers.extend([None, None, end()])
        group = _ProcessGroup(types.SimpleNamespace(pid=4242))
        assert await group.gone(within=5.0)
        assert sent == [0, 0, 0]

    async def test_the_wait_gives_up_at_its_deadline(self, killpg):
        """A member alive past it, or an id given to another group, only ends the wait."""
        sent, _ = killpg
        group = _ProcessGroup(types.SimpleNamespace(pid=4242))
        assert not await group.gone(within=0.2)
        assert set(sent) == {0}


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
