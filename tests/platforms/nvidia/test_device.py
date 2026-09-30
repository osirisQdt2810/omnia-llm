"""The NVIDIA probe: nvidia-smi's output, and pinning a card."""

from __future__ import annotations

import pytest

from omnia_llm.devices import DeviceError
from omnia_llm.platforms.nvidia.device import NvidiaProbe


class TestNvidia:
    ROWS = ("0, NVIDIA RTX A5000, 24564, 52, 0, GPU-aaa\n"
            "7, NVIDIA RTX A5000, 24564, 13825, 0, GPU-hhh\n")
    APPS = "GPU-hhh, 4016306\n"

    def test_it_reads_the_real_format(self):
        cards = NvidiaProbe.parse(self.ROWS, self.APPS)
        assert [(c.index, c.memory_used_mib, c.has_compute_apps) for c in cards] == [
            (0, 52, False), (7, 13825, True)]

    def test_a_card_is_pinned_with_cuda_visible_devices(self):
        assert dict(NvidiaProbe.parse(self.ROWS, "")[1].env) == {"CUDA_VISIBLE_DEVICES": "7"}

    def test_an_unreadable_tool_is_an_error_not_a_guess(self, monkeypatch):
        import omnia_llm.platforms.nvidia.device as mod

        def boom(_args):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(mod, "_run", boom)
        with pytest.raises(DeviceError):
            NvidiaProbe().list_devices()
