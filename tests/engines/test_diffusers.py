"""diffusers: the worker is told which kind of device it is on."""

from __future__ import annotations

from omnia_llm.engines import ENGINES
from tests.helpers import fake_probe, gpu, spec


class TestCommand:
    def test_diffusers_is_told_the_device_kind(self, paths):
        engine = ENGINES.get("diffusers")(spec(id="img", kind="image", engine="diffusers"),
                                          fake_probe(), paths)
        argv = engine.command(gpu(7))
        assert argv[3] == "image-engine" and argv[argv.index("--device") + 1] == "nvidia"
