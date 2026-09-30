"""llama.cpp: where its model comes from, no GPU layers on a CPU, downloads in the model cache."""

from __future__ import annotations

import pytest

from omnia_llm.config import ConfigError
from omnia_llm.devices import Device
from omnia_llm.engines import ENGINES
from tests.helpers import fake_probe, gpu, spec

CPU = Device(backend="cpu", index=0, name="cpu", memory_total_mib=1, exclusive=True)


def _llamacpp(paths, **options):
    return ENGINES.get("llamacpp")(spec(engine="llamacpp", **options), fake_probe(backend="cpu"),
                                   paths)


class TestCommand:
    def test_llamacpp_uses_no_gpu_layers_on_a_cpu(self, paths):
        argv = _llamacpp(paths, model="m.gguf", gpu_layers=99).command(CPU)
        assert argv[argv.index("-ngl") + 1] == "0" and argv[argv.index("--alias") + 1] == "omnia-local"

    def test_llamacpp_downloads_into_the_model_cache_too(self, paths):
        env = _llamacpp(paths, hf_repo="x/y-GGUF:Q4_K_M").environment(gpu(0))
        assert env["LLAMA_CACHE"] == str(paths.model_cache / "llama.cpp")


class TestWhereTheModelComesFrom:
    def test_a_file_in_a_repo_is_passed_with_its_repo(self, paths):
        """Without -hff, llama-server took the file the repo's tag names and ignored ``model``."""
        argv = _llamacpp(paths, hf_repo="x/y-GGUF", model="y-q8_0.gguf").command(CPU)
        assert argv[1:5] == ["-hf", "x/y-GGUF", "-hff", "y-q8_0.gguf"]

    def test_a_repo_alone_is_passed_alone(self, paths):
        argv = _llamacpp(paths, hf_repo="x/y-GGUF:Q4_K_M", model="").command(CPU)
        assert argv[1:3] == ["-hf", "x/y-GGUF:Q4_K_M"] and "-hff" not in argv and "-m" not in argv

    def test_a_local_file_is_passed_as_one(self, paths):
        argv = _llamacpp(paths, model="/models/m.gguf").command(CPU)
        assert argv[1:3] == ["-m", "/models/m.gguf"] and "-hf" not in argv

    def test_no_model_at_all_is_refused_when_the_config_loads(self, paths):
        with pytest.raises(ConfigError, match="hf_repo"):
            _llamacpp(paths, model="")
