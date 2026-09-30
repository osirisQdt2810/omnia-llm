"""MLX: every request asks for the model it loaded."""

from __future__ import annotations

import omnia_llm.platforms  # noqa: F401 - registration
from omnia_llm.engines import ENGINES
from tests.helpers import fake_probe, spec


class TestRewrite:
    def test_mlx_always_asks_for_the_model_it_loaded(self, paths):
        """mlx_lm.server would LOAD an unknown model name — so every request is rewritten."""
        engine = ENGINES.get("mlx")(spec(engine="mlx", model="mlx-community/x-4bit"),
                                    fake_probe(backend="apple"), paths)
        assert b'"model": "mlx-community/x-4bit"' in engine.rewrite(b'{"model": "omnia-local"}')

    def test_a_body_that_is_not_json_is_forwarded_untouched(self, paths):
        engine = ENGINES.get("mlx")(spec(engine="mlx"), fake_probe(backend="apple"), paths)
        assert engine.rewrite(b"not json") == b"not json"
