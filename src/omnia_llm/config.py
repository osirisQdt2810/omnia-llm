"""The gateway's configuration: one TOML file (``configs/*.toml``) read into typed objects.

Relative paths resolve against the REPOSITORY ROOT, not the working directory, so a service
started from anywhere finds the same state. Unknown sections, keys, engines or device probes
are errors at startup — a typo that silently falls back to a default is a model that quietly
never runs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional

try:  # stdlib from 3.11; the reference box runs 3.10, where the same parser is `tomli`
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("text", "image")


class ConfigError(ValueError):
    """The configuration cannot be run as written."""


@dataclasses.dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8721
    #: Minutes with no request before a model is stopped and its device handed back.
    idle_timeout_minutes: float = 30.0
    reaper_interval_seconds: float = 20.0

    @property
    def idle_timeout_seconds(self) -> float:
        return self.idle_timeout_minutes * 60.0


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    #: What this deployment owns at runtime: tokens, the legacy key, logs.
    state: Path = ROOT / "state"
    #: Downloaded model weights. Inside the repository, so everything the server fetched sits
    #: in one folder that can be measured, moved or deleted, rather than in a home-wide cache.
    model_cache: Path = ROOT / ".model-cache"

    @property
    def tokens_file(self) -> Path:
        return self.state / "tokens.json"

    @property
    def legacy_key_file(self) -> Path:
        return self.state / "api-key.txt"

    @property
    def logs(self) -> Path:
        return self.state / "logs"

    @property
    def hf_hub_cache(self) -> Path:
        """Hugging Face downloads (vLLM, MLX, diffusers): the hub that HF_HOME set to
        <model_cache>/huggingface used to fill, so nothing downloaded before is fetched again."""
        return self.model_cache / "huggingface" / "hub"

    @property
    def vllm_cache(self) -> Path:
        """vLLM's own cache: its compiled graphs, kept with the weights."""
        return self.model_cache / "vllm"

    @property
    def llama_cache(self) -> Path:
        """llama.cpp's own downloads (``-hf``)."""
        return self.model_cache / "llama.cpp"


@dataclasses.dataclass(frozen=True)
class DevicesConfig:
    #: ``auto`` picks the first available probe by priority (nvidia, rocm, apple, cpu).
    probe: str = "auto"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """One model the gateway serves: what clients call it, what it is for, and how it runs."""

    id: str
    kind: str
    engine: str
    #: Where the engine process listens (loopback only). Each model needs its own.
    port: int
    #: Device memory the model needs, so a device without the room is never chosen.
    required_mib: int = 0
    #: Engine-specific settings, validated by the engine's own ``Options`` type.
    options: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class AppConfig:
    server: ServerConfig
    paths: PathsConfig
    devices: DevicesConfig
    models: tuple[ModelSpec, ...]
    source: Optional[Path] = None

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"no config file at {path}")
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, source=path)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: Optional[Path] = None) -> "AppConfig":
        unknown = set(data) - {"server", "paths", "devices", "models"}
        if unknown:
            raise ConfigError(f"unknown section(s): {', '.join(sorted(unknown))}")
        server = _build(ServerConfig, data.get("server", {}), "server")
        paths_raw = dict(data.get("paths", {}))
        for key in ("state", "model_cache"):
            if key in paths_raw:
                value = Path(paths_raw[key]).expanduser()
                paths_raw[key] = value if value.is_absolute() else ROOT / value
        paths = _build(PathsConfig, paths_raw, "paths")
        devices = _build(DevicesConfig, data.get("devices", {}), "devices")
        models = tuple(_model(raw, i) for i, raw in enumerate(data.get("models", [])))
        if not models:
            raise ConfigError("no [[models]] configured — there is nothing to serve")
        seen: set[str] = set()
        ports: set[int] = {server.port}
        for model in models:
            if model.id in seen:
                raise ConfigError(f"two models are called {model.id!r}")
            if model.port in ports:
                raise ConfigError(f"model {model.id!r}: port {model.port} is already in use")
            seen.add(model.id)
            ports.add(model.port)
        return cls(server, paths, devices, models, source)

    def models_of(self, kind: str) -> list[ModelSpec]:
        return [m for m in self.models if m.kind == kind]


def _build(cls: type, raw: dict[str, Any], where: str):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"[{where}]: unknown key(s): {', '.join(sorted(unknown))}")
    return cls(**raw)


def _model(raw: dict[str, Any], index: int) -> ModelSpec:
    where = f"[[models]] #{index + 1}"
    for key in ("id", "kind", "engine", "port"):
        if key not in raw:
            raise ConfigError(f"{where}: missing {key!r}")
    if raw["kind"] not in KINDS:
        raise ConfigError(f"{where}: kind must be one of {KINDS}, not {raw['kind']!r}")
    spec = _build(ModelSpec, raw, where)
    return dataclasses.replace(spec, options=dict(spec.options))
