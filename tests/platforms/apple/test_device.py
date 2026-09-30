"""The Apple Silicon probe: the Mac's unified memory, as one device that is always ours."""

from __future__ import annotations

import subprocess
import types

import pytest

import omnia_llm.platforms.apple.device as mod
from omnia_llm.devices import DeviceError
from omnia_llm.platforms.apple.device import AppleSiliconProbe

_SIXTEEN_GB = 16 * 1024**3


def _sysctl(monkeypatch, memsize=f"{_SIXTEEN_GB}\n", chip="Apple M4\n"):
    answers = {"hw.memsize": memsize, "machdep.cpu.brand_string": chip}

    def run(args, **_kwargs):
        return types.SimpleNamespace(stdout=answers[args[-1]])

    monkeypatch.setattr(mod.subprocess, "run", run)


class TestAppleSilicon:
    def test_the_mac_is_one_device_with_its_memory_in_mib(self, monkeypatch):
        _sysctl(monkeypatch)

        [mac] = AppleSiliconProbe().list_devices()

        assert (mac.backend, mac.index, mac.name) == ("apple", 0, "Apple M4")
        assert mac.memory_total_mib == 16384
        assert mac.exclusive and mac.is_free

    def test_a_model_that_fits_the_memory_is_placed(self, monkeypatch):
        """Reported in the wrong unit, a 16 GB Mac would have room for nothing that says how
        much memory it needs, and every request would be a 503."""
        _sysctl(monkeypatch)

        assert AppleSiliconProbe().pick(required_mib=4096) is not None
        assert AppleSiliconProbe().pick(required_mib=32768) is None

    def test_an_unnamed_chip_still_has_a_name(self, monkeypatch):
        _sysctl(monkeypatch, chip="\n")

        assert AppleSiliconProbe().list_devices()[0].name == "Apple Silicon"

    def test_an_unreadable_sysctl_is_an_error_not_a_guess(self, monkeypatch):
        def refuse(*_args, **_kwargs):
            raise subprocess.CalledProcessError(1, "sysctl")

        monkeypatch.setattr(mod.subprocess, "run", refuse)

        with pytest.raises(DeviceError):
            AppleSiliconProbe().list_devices()

    @pytest.mark.parametrize(
        "system, machine, expected",
        [
            ("Darwin", "arm64", True),
            ("Darwin", "x86_64", False),
            ("Linux", "arm64", False),
            ("Linux", "x86_64", False),
        ],
    )
    def test_it_is_only_available_on_apple_silicon(self, monkeypatch, system, machine, expected):
        """An Intel Mac has no Metal device MLX can use; left to `auto`, it falls to the CPU."""
        monkeypatch.setattr(mod.platform, "system", lambda: system)
        monkeypatch.setattr(mod.platform, "machine", lambda: machine)

        assert AppleSiliconProbe.available() is expected
