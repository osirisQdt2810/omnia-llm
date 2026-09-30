"""The ROCm probe: rocm-smi's JSON, and pinning a card."""

from __future__ import annotations

import json

from omnia_llm.platforms.rocm.device import RocmProbe


class TestRocm:
    RAW = json.dumps({
        "card0": {"Card series": "Instinct MI210", "VRAM Total Memory (B)": str(64 * 1024**3),
                  "VRAM Total Used Memory (B)": str(10 * 1024**2), "GPU use (%)": "0"},
        "card1": {"Card series": "Instinct MI210", "VRAM Total Memory (B)": str(64 * 1024**3),
                  "VRAM Total Used Memory (B)": str(30 * 1024**3), "GPU use (%)": "95"},
        "system": {"PID 123": "python, 1, 30000000000, 0, 0"},
    })

    def test_it_reads_memory_use_and_pids(self):
        cards = RocmProbe.parse(self.RAW)
        assert [(c.index, c.memory_total_mib, c.is_free) for c in cards] == [
            (0, 65536, True), (1, 65536, False)]
        assert cards[1].has_compute_apps

    def test_a_card_is_pinned_with_hip_visible_devices(self):
        assert dict(RocmProbe.parse(self.RAW)[0].env) == {"HIP_VISIBLE_DEVICES": "0"}

    def test_a_card_whose_memory_cannot_be_read_is_busy_not_free(self):
        cards = RocmProbe.parse(json.dumps({"card0": {"Card series": "x"}}))
        assert not cards[0].is_free
