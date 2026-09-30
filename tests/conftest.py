"""Fixtures shared by every test package. The fakes themselves live in ``tests/helpers.py``."""

from __future__ import annotations

import pytest

from omnia_llm.config import PathsConfig


@pytest.fixture
def paths(tmp_path):
    """Runtime state and the model cache, both inside the test's own temporary folder."""
    return PathsConfig(state=tmp_path / "state", model_cache=tmp_path / "model-cache")
