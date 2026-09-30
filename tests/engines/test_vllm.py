"""vLLM: one card, under the client's model id."""

from __future__ import annotations

from omnia_llm.engines import ENGINES
from tests.helpers import fake_probe, gpu, spec


class TestCommand:
    def test_vllm_serves_under_the_client_id_on_one_card(self, paths):
        engine = ENGINES.get("vllm")(spec(), fake_probe(), paths)
        argv = engine.command(gpu(7))
        assert argv[1:5] == ["-m", "omnia_llm.workers.titled", "llm-engine",
                             "vllm.entrypoints.cli.main"]
        assert argv[argv.index("--served-model-name") + 1] == "omnia-local"
        assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
