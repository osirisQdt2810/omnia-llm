"""What counts as a free device, and how one is picked."""

from __future__ import annotations

from omnia_llm.devices import Device
from tests.helpers import fake_probe, gpu


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
        assert fake_probe([gpu(0, busy=True), gpu(1, used=9000)]).pick(5000) is None

    def test_the_highest_free_index(self):
        assert fake_probe([gpu(0), gpu(3), gpu(7), gpu(5, busy=True)]).pick(5000).index == 7

    def test_a_card_too_small_is_skipped(self):
        assert fake_probe([gpu(6), gpu(7, total=4000)]).pick(5000).index == 6

    def test_the_card_we_hold_is_reused_when_it_has_room(self):
        probe = fake_probe([gpu(2), gpu(7, used=13800, busy=True)])
        assert probe.pick(5000, prefer=["nvidia:7"]).index == 7

    def test_a_full_held_card_falls_back_to_an_idle_one(self):
        probe = fake_probe([gpu(2), gpu(7, used=24000, busy=True)])
        assert probe.pick(5000, prefer=["nvidia:7"]).index == 2

    def test_somebody_elses_busy_card_is_never_taken(self):
        assert fake_probe([gpu(1, used=3000, busy=True)]).pick(100) is None
