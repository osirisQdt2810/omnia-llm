"""Devices: what counts as free, which one is taken, and reading each platform's tool."""

import json

import pytest
from conftest import FakeProbe, gpu

from omnia_llm.devices import Device, DeviceError, detect
from omnia_llm.platforms.nvidia.device import NvidiaProbe
from omnia_llm.platforms.rocm.device import RocmProbe


class TestWhatCountsAsFree:
    def test_an_untouched_card_is_free(self):
        assert gpu(0, used=0).is_free

    def test_the_driver_holding_a_few_mib_is_still_free(self):
        assert gpu(0, used=38).is_free

    def test_a_compute_process_makes_it_busy_whatever_the_memory(self):
        assert not gpu(0, used=0, busy=True).is_free

    def test_holding_memory_makes_it_busy(self):
        assert not gpu(0, used=13264).is_free

    def test_load_makes_it_busy(self):
        assert not gpu(0, used=0, util=40).is_free

    def test_a_personal_machine_is_always_ours(self):
        mac = Device(backend="apple", index=0, name="M4", memory_total_mib=16384,
                     memory_used_mib=9000, utilization_pct=90, exclusive=True)
        assert mac.is_free


class TestPickingOne:
    def test_none_when_every_card_is_busy(self):
        assert FakeProbe([gpu(0, busy=True), gpu(1, used=9000)]).pick(5000) is None

    def test_the_highest_free_index(self):
        assert FakeProbe([gpu(0), gpu(3), gpu(7), gpu(5, busy=True)]).pick(5000).index == 7

    def test_a_card_too_small_is_skipped(self):
        assert FakeProbe([gpu(6), gpu(7, total=4000)]).pick(5000).index == 6

    def test_the_card_we_hold_is_reused_when_it_has_room(self):
        probe = FakeProbe([gpu(2), gpu(7, used=13800, busy=True)])
        assert probe.pick(5000, prefer=["nvidia:7"]).index == 7

    def test_a_full_held_card_falls_back_to_an_idle_one(self):
        probe = FakeProbe([gpu(2), gpu(7, used=24000, busy=True)])
        assert probe.pick(5000, prefer=["nvidia:7"]).index == 2

    def test_somebody_elses_busy_card_is_never_taken(self):
        assert FakeProbe([gpu(1, used=3000, busy=True)]).pick(100) is None


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


class TestDetect:
    def test_cpu_is_always_there(self):
        assert type(detect("cpu")).registry_name == "cpu"

    def test_a_configured_probe_that_is_absent_is_an_error(self, monkeypatch):
        monkeypatch.setattr(NvidiaProbe, "available", classmethod(lambda cls: False))
        with pytest.raises(DeviceError):
            detect("nvidia")

    def test_the_cpu_hides_every_gpu(self):
        cpu = detect("cpu").list_devices()[0]
        assert dict(cpu.env) == {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
