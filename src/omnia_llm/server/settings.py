"""Settings a client may change while the gateway runs, kept in ``state/settings.json``.

Today that is one: the idle timeout. The config file's ``[server] idle_timeout_minutes`` is the
default; a value set over HTTP overrides it and survives a restart. A missing file means the
default; an unreadable one is logged and means the default too, never a gateway that will not
start.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from omnia_llm.config import AppConfig

log = logging.getLogger(__name__)

#: The range a client may set, in minutes: at least one, at most a day.
MIN_IDLE_MINUTES = 1
MAX_IDLE_MINUTES = 1440
_KEY = "idle_timeout_minutes"
_RANGE = f'"{_KEY}" must be a number between {MIN_IDLE_MINUTES} and {MAX_IDLE_MINUTES}'


def _valid_minutes(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and MIN_IDLE_MINUTES <= value <= MAX_IDLE_MINUTES)


class RuntimeSettings:
    """The idle timeout in force: the config default, or the override a client last set."""

    def __init__(self, path: Path, default_idle_timeout_minutes: float) -> None:
        self._path = path
        self._default = default_idle_timeout_minutes
        self._minutes = self._read()

    @classmethod
    def from_config(cls, config: AppConfig) -> "RuntimeSettings":
        return cls(config.paths.settings_file, config.server.idle_timeout_minutes)

    @property
    def idle_timeout_minutes(self) -> float:
        return self._minutes

    @property
    def idle_timeout_seconds(self) -> float:
        return self._minutes * 60.0

    def to_dict(self) -> dict:
        """What ``GET /settings`` answers."""
        return {_KEY: self._minutes, "default_idle_timeout_minutes": self._default}

    def update(self, body: Any) -> None:
        """Apply and persist ``{"idle_timeout_minutes": N}``.

        Raises:
            ValueError: for anything else — another key, a missing one, or N not a number in
                range — with a message naming the range. Nothing is changed then.
        """
        if not isinstance(body, dict) or set(body) != {_KEY} or not _valid_minutes(body[_KEY]):
            raise ValueError(f'the body must be {{"{_KEY}": N}}: {_RANGE}')
        self._write(body[_KEY])
        self._minutes = body[_KEY]

    def _read(self) -> float:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._default
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if isinstance(data, dict) and _valid_minutes(data.get(_KEY)):
            return data[_KEY]
        log.warning("%s is unreadable or out of range; using the config's idle timeout (%s min)",
                    self._path, self._default)
        return self._default

    def _write(self, minutes: float) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        # 0600 from creation, then an atomic rename: never readable by others, never half-written.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({_KEY: minutes}, handle)
        os.replace(tmp, self._path)
