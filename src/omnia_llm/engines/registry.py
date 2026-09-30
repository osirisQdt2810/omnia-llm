"""The engine registry: ``[[models]] engine = "<name>"`` names a class registered here."""

from __future__ import annotations

from omnia_llm.engines.base import Engine
from omnia_llm.registry import Registry

ENGINES: Registry[Engine] = Registry("engine")
register_engine = ENGINES.register
