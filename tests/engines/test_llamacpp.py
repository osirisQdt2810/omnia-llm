"""llama.cpp: no GPU layers on a CPU, and downloads kept in the model cache."""

from __future__ import annotations

from omnia_llm.devices import Device
from omnia_llm.engines import ENGINES
from tests.helpers import fake_probe, gpu, spec


class TestCommand:
    def test_llamacpp_uses_no_gpu_layers_on_a_cpu(self, paths):

        engine = ENGINES.get("llamacpp")(spec(engine="llamacpp", model="m.gguf", gpu_layers=99),
                                         fake_probe(backend="cpu"), paths)
        cpu = Device(backend="cpu", index=0, name="cpu", memory_total_mib=1, exclusive=True)
        argv = engine.command(cpu)
        assert argv[argv.index("-ngl") + 1] == "0" and argv[argv.index("--alias") + 1] == "omnia-local"

    def test_llamacpp_downloads_into_the_model_cache_too(self, paths):
        engine = ENGINES.get("llamacpp")(spec(engine="llamacpp", hf_repo="x/y-GGUF:Q4_K_M"),
                                         fake_probe(backend="cpu"), paths)
        env = engine.environment(gpu(0))
        assert env["LLAMA_CACHE"] == str(paths.model_cache / "llama.cpp")
