"""Gateway configuration: one TOML file, every knob the operator actually turns.

Read once at startup. The values that matter most are ``model`` and ``idle_timeout_minutes`` —
the second is the promise this whole service makes to the other people on the box.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

try:  # stdlib from 3.11; this box runs 3.10, where the same parser ships as `tomli`
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter, not the code
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent


@dataclasses.dataclass(frozen=True)
class Settings:
    """Everything the gateway needs, with defaults that suit one 24 GB card."""

    #: HuggingFace id or local path. AWQ 4-bit of a 14B leaves room for a long KV cache on
    #: 24 GB, which matters more for this workload than the last few points of benchmark.
    model: str = "Qwen/Qwen3-14B-AWQ"
    #: What clients call it. Kept stable so swapping the weights does not break Omnia's config.
    served_model_name: str = "omnia-local"

    #: The gateway: always up, no GPU, this is what the SSH tunnel points at.
    gateway_port: int = 8721
    #: vLLM: bound to localhost only, reached exclusively through the gateway.
    engine_port: int = 8722

    #: Text-to-image. Empty DISABLES it: /v1/images/generations then answers a plain "not
    #: configured" rather than starting something the operator never asked for, which matters
    #: on a shared card where an accidental 7 GB is somebody else's problem.
    image_model: str = ""
    image_port: int = 8723
    #: Denoising steps. SDXL-Turbo and FLUX-schnell are distilled to a handful; a full SDXL
    #: wants ~30. The one knob that trades seconds for quality.
    image_steps: int = 4
    #: What the TEXT engine needs on a card. vLLM preallocates ``gpu_memory_utilization`` of
    #: the card's TOTAL memory, so this is that share plus a little room, and it is what decides
    #: whether the text model fits beside an image model already loaded.
    text_required_mib: int = 14500
    #: Keep the pipeline in system RAM and move each component to the card only for the step
    #: that needs it: ~4 GiB resident instead of ~9.5, at a few seconds per image. On by
    #: default because this card is shared and the text model is on it too.
    image_cpu_offload: bool = True
    #: What the image engine needs on a card. Used to decide whether it fits beside the text
    #: model on the card we already hold, or has to take one of its own.
    image_required_mib: int = 5000

    #: Minutes without a single request before the card is handed back.
    idle_timeout_minutes: float = 30.0
    #: How often the reaper looks. Cheap, so it can be frequent enough to be punctual.
    reaper_interval_seconds: float = 20.0

    #: A model load is tens of seconds; the FIRST ever run also downloads the weights.
    startup_timeout_seconds: float = 900.0
    #: How long a terminated engine gets to exit before it is killed outright.
    shutdown_grace_seconds: float = 20.0

    #: Skip cards smaller than this. Every card here is 24 GB; the guard is for the day one is not.
    minimum_gpu_mib: int = 20000
    #: vLLM's share of the card. Below 1.0 so a colleague's stray allocation does not instantly
    #: OOM us, and so the driver has room of its own.
    gpu_memory_utilization: float = 0.90
    #: Context window. Larger costs KV cache, which is the scarce thing at 4-bit weights.
    max_model_len: int = 16384
    #: Anything else to pass through to `vllm serve`, verbatim.
    extra_vllm_args: tuple[str, ...] = ()

    vllm_binary: str = str(ROOT / ".venv" / "bin" / "vllm")
    #: The venv's interpreter — the image engine runs as ``python -m gateway.imaged``, so it
    #: must be THIS venv's python and not whatever happens to be on PATH.
    python_binary: str = str(ROOT / ".venv" / "bin" / "python")
    log_dir: Path = ROOT / "logs"
    #: The API key file. Generated on first start, 0600, never checked in.
    api_key_file: Path = ROOT / "api-key.txt"
    #: Named tokens, hashed (see tokens.py). The old api-key.txt is adopted into it as "legacy".
    tokens_file: Path = ROOT / "tokens.json"
    #: Where the weights land. Inside the project on purpose: this is ~10 GB, and a model
    #: cache that hides in ~/.cache is one nobody finds when /home fills up — which on this
    #: box, at 99% used, is the failure that actually happens.
    hf_home: Path = ROOT / "hf-cache"

    @property
    def idle_timeout_seconds(self) -> float:
        return self.idle_timeout_minutes * 60.0

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        """Build from ``config.toml`` beside the package, falling back to the defaults.

        A missing file is not an error: the defaults are a working configuration, and requiring
        a config file to start a service whose whole job is to be unattended would be a poor
        trade.
        """
        path = path or Path(os.environ.get("OMNIA_LLM_CONFIG", ROOT / "config.toml"))
        data: dict = {}
        if path.is_file():
            data = tomllib.loads(path.read_text(encoding="utf-8")).get("gateway", {})
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "extra_vllm_args" in kwargs:
            kwargs["extra_vllm_args"] = tuple(kwargs["extra_vllm_args"])
        for key in ("log_dir", "hf_home", "api_key_file", "tokens_file"):
            if key in kwargs:
                kwargs[key] = Path(kwargs[key])
        settings = cls(**kwargs)
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        settings.hf_home.mkdir(parents=True, exist_ok=True)
        return settings
