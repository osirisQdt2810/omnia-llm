"""Start vLLM on demand, hand it requests, and stop it once nobody is asking.

The point of the whole gateway: the GPU is shared, so holding a card open between study
sessions is taking it from a colleague for nothing. vLLM therefore runs only while it is being
used, and the idle timer is what gives the card back without anybody remembering to.

One engine at a time. Concurrency is handled by a single lock around start/stop rather than by
reference counting, because the failure the lock prevents — two requests both deciding the
engine is down and both launching vLLM on the same card — costs an out-of-memory crash, and
the cost of serialising two cold starts is one wait nobody notices behind a 60s model load.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import time
from typing import Optional

import httpx

from gateway import gpus
from gateway.settings import Settings


class EngineBusy(RuntimeError):
    """Every GPU is in use, so there is nothing to start on.

    Distinct from a crash on purpose: the caller turns this into a 503 with a plain sentence,
    because "someone else is using the machine" is a thing to wait out, not a bug to report.
    """


class Engine:
    """The vLLM subprocess, its lifetime, and the only door to it."""

    def __init__(self, settings: Settings, role: str = "text") -> None:
        """
        Args:
            settings: The gateway's configuration.
            role: ``"text"`` (vLLM) or ``"image"`` (the diffusion server). One class rather
                than two because everything that is hard here — one start at a time, a real
                readiness wait, killing the process GROUP, the idle timer — is identical. Only
                the command line and which card it may share differ.
        """
        self._settings = settings
        self._role = role
        #: Set by the gateway to the card the OTHER engine is on, so this one can share it.
        #: A plain attribute rather than a constructor argument because it changes over the
        #: gateway's life: the text engine comes and goes, and what it holds moves with it.
        self._share_with: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        self._gpu_index: Optional[int] = None
        self._lock = asyncio.Lock()
        self._last_used = 0.0
        self._started_at = 0.0
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    # --- state ---------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def model(self) -> str:
        return (
            self._settings.image_model
            if self._role == "image"
            else self._settings.model
        )

    @property
    def port(self) -> int:
        return (
            self._settings.image_port
            if self._role == "image"
            else self._settings.engine_port
        )

    def status(self) -> dict:
        """What the gateway reports about itself — used by ``/status`` and the logs."""
        idle = time.monotonic() - self._last_used if self._last_used else 0.0
        return {
            "role": self._role,
            "running": self.running,
            "gpu": self._gpu_index,
            "model": self.model,
            "idle_seconds": round(idle, 1),
            "idle_timeout_seconds": self._settings.idle_timeout_seconds,
            "uptime_seconds": (
                round(time.monotonic() - self._started_at, 1) if self.running else 0.0
            ),
        }

    # --- lifecycle -----------------------------------------------------------------------
    async def ensure_running(self) -> None:
        """Start vLLM if it is not up. Safe to call from every request.

        Holds the lock across the whole cold start, including the readiness wait: a second
        request arriving mid-start must queue behind it rather than conclude the engine is down
        and launch a second one on the same card.
        """
        async with self._lock:
            if self.running:
                return
            gpu = self._pick_gpu()
            if gpu is None:
                raise EngineBusy(
                    "every GPU on this machine is in use right now — nothing was started"
                )
            self._launch(gpu.index)
            try:
                await self._wait_until_ready()
            except BaseException:
                # The invariant lives HERE, at the only place that owns the lifecycle: a start
                # that did not finish holds no card. Leaving it to the branches inside the wait
                # means every future failure mode has to remember — and the one that forgets
                # leaves a phantom engine pinning a GPU nobody can see is taken.
                await self.stop("startup failed")
                raise

    def _pick_gpu(self) -> "gpus.Gpu | None":
        """Which card to start on: the one the OTHER engine holds, if it has the room.

        Symmetric on purpose, and it was not at first. Making only the image engine share meant
        whichever engine happened to start FIRST claimed a card and the second took another —
        so asking for a picture before any text quietly cost two cards out of a pool eight
        people draw from. Order of use is not something a user should have to think about to
        keep a promise the service made.

        Falls back to an idle card when the held one genuinely has no room: "the other model
        left no space" is not a reason to refuse to generate.
        """
        required = (
            self._settings.image_required_mib
            if self._role == "image"
            else self._settings.text_required_mib
        )
        return gpus.pick_shared_or_free(required, self._share_with)

    def _launch(self, gpu_index: int) -> None:
        """Spawn vLLM pinned to ``gpu_index`` and nothing else.

        ``CUDA_VISIBLE_DEVICES`` rather than vLLM's own device flag: it is the only way to make
        the *whole process* — including anything a library does at import time — unable to see
        the other seven cards. A flag would leave the door open for one stray allocation to
        land on somebody's job.
        """
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        # Keep the ~10 GB of weights inside the project rather than in ~/.cache, so everything
        # this service owns can be found — and deleted — in one place.
        env["HF_HOME"] = str(self._settings.hf_home)
        args = self._command(gpu_index)
        log = open(self._settings.log_dir / f"{self._role}.log", "ab", buffering=0)
        log.write(
            f"\n=== launching on GPU {gpu_index} at {time.strftime('%F %T')} ===\n".encode()
        )
        # start_new_session so the engine gets its own process group: stopping it must kill
        # the worker children too, and signalling the group is the only reliable way.
        self._process = subprocess.Popen(
            args, env=env, stdout=log, stderr=log, start_new_session=True
        )
        self._gpu_index = gpu_index
        self._started_at = time.monotonic()
        self._last_used = time.monotonic()

    def _command(self, gpu_index: int) -> list[str]:
        """The argv for this engine's role."""
        # Both run under a neutral title (gateway/titled.py): the box is shared and `ps` /
        # `nvidia-smi` show every user this project's path otherwise.
        titled = [self._settings.python_binary, "-m", "gateway.titled"]
        if self._role == "image":
            return titled + [
                "image-engine",
                "gateway.imaged",
                "--model",
                self._settings.image_model,
                "--host",
                "127.0.0.1",
                "--port",
                str(self._settings.image_port),
                "--steps",
                str(self._settings.image_steps),
            ] + ([] if self._settings.image_cpu_offload else ["--no-cpu-offload"])
        return titled + [
            "llm-engine",
            "vllm.entrypoints.cli.main",
            "serve",
            self._settings.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._settings.engine_port),
            "--served-model-name",
            self._settings.served_model_name,
            "--max-model-len",
            str(self._settings.max_model_len),
            "--gpu-memory-utilization",
            str(self._settings.gpu_memory_utilization),
            # One card, always. The gateway's whole promise is that it takes exactly one.
            "--tensor-parallel-size",
            "1",
        ] + list(self._settings.extra_vllm_args)

    async def _wait_until_ready(self) -> None:
        """Poll vLLM's health endpoint until it answers, then return; raise if it never does.

        A model load is tens of seconds and a first-ever run also downloads weights, so the
        budget is generous. This method only decides READY or NOT — cleaning up a start that
        failed belongs to :meth:`ensure_running`, which is what owns the card.
        """
        deadline = time.monotonic() + self._settings.startup_timeout_seconds
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"{self._role} engine exited with code {self._process.returncode} "
                    f"during startup — see logs/{self._role}.log"
                )
            try:
                if (await self._client.get(url, timeout=5.0)).status_code == 200:
                    return
            except Exception:  # noqa: BLE001 - not up yet is the normal case here
                pass
            await asyncio.sleep(2.0)
        raise RuntimeError(
            f"the {self._role} engine did not become ready within "
            f"{self._settings.startup_timeout_seconds}s"
        )

    async def stop(self, reason: str) -> None:
        """Terminate vLLM and give the card back.

        SIGTERM to the process GROUP, then SIGKILL after a grace period. vLLM spawns workers
        that outlive a bare SIGTERM to the parent, and a leaked worker keeps the GPU memory
        allocated — which would defeat the entire point of the idle timer.
        """
        process, self._process = self._process, None
        gpu, self._gpu_index = self._gpu_index, None
        if process is None or process.poll() is not None:
            return
        print(f"[engine] stopping (gpu={gpu}): {reason}", flush=True)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        for _ in range(int(self._settings.shutdown_grace_seconds * 2)):
            if process.poll() is not None:
                return
            await asyncio.sleep(0.5)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    # --- traffic -------------------------------------------------------------------------
    async def forward(self, method: str, path: str, body: bytes, headers: dict) -> httpx.Response:
        """Send one request to vLLM, starting it first if needed, and keep it alive.

        The idle clock is reset on the way IN and again on the way OUT. Only resetting on the
        way in would let the timer fire during a long generation and kill the engine answering
        it; only on the way out would let a stalled request look idle.
        """
        await self.ensure_running()
        self._last_used = time.monotonic()
        try:
            response = await self._client.request(
                method,
                f"http://127.0.0.1:{self.port}{path}",
                content=body,
                headers=headers,
            )
        finally:
            self._last_used = time.monotonic()
        return response

    async def reap_when_idle(self) -> None:
        """Background loop: stop the engine once nobody has asked for anything.

        Runs for the gateway's whole life. The gateway itself costs no GPU, so it stays up and
        the card is what comes and goes.
        """
        while True:
            await asyncio.sleep(self._settings.reaper_interval_seconds)
            if not self.running or not self._last_used:
                continue
            idle = time.monotonic() - self._last_used
            if idle >= self._settings.idle_timeout_seconds:
                async with self._lock:
                    # Re-checked under the lock: a request may have arrived while we waited,
                    # and killing the engine out from under it would fail a live generation.
                    if (
                        self.running
                        and time.monotonic() - self._last_used
                        >= self._settings.idle_timeout_seconds
                    ):
                        await self.stop(f"idle for {idle / 60:.1f} min")
