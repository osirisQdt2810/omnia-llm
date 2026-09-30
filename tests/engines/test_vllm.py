"""vLLM: one card, under the client's model id, and its caches kept with the weights."""

from __future__ import annotations

import json

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

    def test_vllms_compile_cache_stays_in_the_model_cache(self, paths):
        engine = ENGINES.get("vllm")(spec(), fake_probe(), paths)
        assert engine.environment(gpu(7))["VLLM_CACHE_ROOT"] == str(paths.model_cache / "vllm")


class TestRewrite:
    def test_vllm_is_always_asked_for_the_name_it_serves(self, paths):
        """The manager hands an unknown id to the first model of its kind; asked under that id,
        vLLM would answer 404."""
        engine = ENGINES.get("vllm")(spec(id="qwen2.5-14b-instruct-awq"), fake_probe(), paths)
        for asked in ("omnia-local", "", "qwen2.5-14b-instruct-awq"):
            body = json.dumps({"model": asked, "messages": []}).encode()
            assert json.loads(engine.rewrite(body))["model"] == "qwen2.5-14b-instruct-awq"

    def test_a_body_that_is_not_json_is_forwarded_untouched(self, paths):
        engine = ENGINES.get("vllm")(spec(), fake_probe(), paths)
        assert engine.rewrite(b"not json") == b"not json"
