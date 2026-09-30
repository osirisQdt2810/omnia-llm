"""What counts as a free GPU — the one decision that can cost somebody else their job."""

from __future__ import annotations

import pytest

from gateway import gpus
from gateway.gpus import Gpu


def _gpu(index=1, used=10, util=0, apps=False, total=24564):
    return Gpu(
        index=index,
        name="NVIDIA RTX A5000",
        memory_total_mib=total,
        memory_used_mib=used,
        utilization_pct=util,
        has_compute_apps=apps,
    )


class TestWhatCountsAsFree:
    """Three signals, all of which must agree, because each misses a case the others catch."""

    def test_an_untouched_card_is_free(self):
        assert _gpu(used=15, util=0, apps=False).is_free

    def test_the_driver_holding_a_few_mib_is_still_free(self):
        # GPU 0 on this box idles at ~52 MiB with a display server attached.
        assert _gpu(used=52).is_free

    def test_a_card_with_a_compute_process_is_not(self):
        """Even at zero memory and zero utilization.

        A job that has just started holds a CUDA context before it allocates anything. Taking
        the card then is how you kill a run in its first second.
        """
        assert not _gpu(used=0, util=0, apps=True).is_free

    def test_a_card_holding_memory_is_not(self):
        # A leaked context has no live process to report, but the memory is still gone.
        assert not _gpu(used=16799, apps=False).is_free

    def test_a_busy_card_between_batches_is_not(self):
        # Utilization dips to 0 between batches; the memory is what gives it away.
        assert not _gpu(used=11971, util=0, apps=False).is_free

    def test_a_card_under_load_is_not(self):
        assert not _gpu(used=100, util=97).is_free


class TestPickingOne:
    def _patch(self, monkeypatch, cards):
        monkeypatch.setattr(gpus, "query", lambda: cards)

    def test_none_when_every_card_is_busy(self, monkeypatch):
        self._patch(monkeypatch, [_gpu(index=i, used=16000, apps=True) for i in range(8)])

        assert gpus.pick_shared_or_free(7000) is None

    def test_it_prefers_the_highest_free_index(self, monkeypatch):
        """GPU 0 is where a display server and any unpinned script land by default.

        Starting from the other end keeps us out of the way of exactly the work most likely to
        show up without warning.
        """
        self._patch(
            monkeypatch,
            [_gpu(index=0), _gpu(index=3, used=16000, apps=True), _gpu(index=7)],
        )

        assert gpus.pick_shared_or_free(7000).index == 7

    def test_a_card_too_small_for_the_model_is_skipped(self, monkeypatch):
        # "Too small" is now expressed as "not enough free memory", which is the same question
        # asked in the units that actually decide whether a model fits.
        self._patch(
            monkeypatch,
            [_gpu(index=7, total=8192, used=10), _gpu(index=2, total=24564, used=10)],
        )

        assert gpus.pick_shared_or_free(20000).index == 2

    def test_it_never_guesses_when_nvidia_smi_is_unreadable(self, monkeypatch):
        """Raises rather than returning a card.

        Picking blind is precisely the failure this module exists to prevent — the caller
        turns this into "did not start", which is the safe end.
        """
        def boom():
            raise RuntimeError("could not read nvidia-smi: not found")

        monkeypatch.setattr(gpus, "query", boom)

        with pytest.raises(RuntimeError):
            gpus.pick_shared_or_free(7000)


class TestParsingRealOutput:
    def test_it_reads_the_machines_actual_format(self, monkeypatch):
        """Verbatim from `nvidia-smi` on this box, with GPUs 3-6 busy."""
        rows = "\n".join(
            [
                "0, NVIDIA RTX A5000, 24564, 52, 0, GPU-4b4d15cf",
                "1, NVIDIA RTX A5000, 24564, 15, 0, GPU-cbbd6f05",
                "3, NVIDIA RTX A5000, 24564, 16799, 100, GPU-0d8b9519",
                "7, NVIDIA RTX A5000, 24564, 15, 0, GPU-273c50b6",
            ]
        )
        apps = "GPU-0d8b9519, 2641185\n"
        monkeypatch.setattr(
            gpus, "_run", lambda args: apps if "compute-apps" in args[1] else rows
        )

        found = {g.index: g for g in gpus.query()}

        assert found[0].is_free and found[1].is_free and found[7].is_free
        assert not found[3].is_free
        assert found[3].has_compute_apps


class TestSharingOneCard:
    """Two engines are meant to share a card, and a card we hold is not "free".

    Asking `pick_free` for the second one would take a SECOND card from the pool — the opposite
    of sharing, and the opposite of what was asked for.
    """

    def _patch(self, monkeypatch, cards):
        monkeypatch.setattr(gpus, "query", lambda: cards)

    def test_it_reuses_the_card_we_already_hold(self, monkeypatch):
        # GPU 7 holds our 14 GB text model; 10 GB of its 24 are free, and SDXL needs 7.
        held = _gpu(index=7, used=14425, util=30, apps=True)
        self._patch(monkeypatch, [held, _gpu(index=2)])

        assert gpus.pick_shared_or_free(7000, prefer_index=7).index == 7

    def test_it_does_not_take_a_second_card_when_one_will_do(self, monkeypatch):
        held = _gpu(index=7, used=14425, util=30, apps=True)
        self._patch(monkeypatch, [held, _gpu(index=2), _gpu(index=1)])

        assert gpus.pick_shared_or_free(7000, prefer_index=7).index != 2

    def test_a_full_card_falls_back_to_an_idle_one(self, monkeypatch):
        """"The model we already loaded left no room" is not a reason to refuse to generate."""
        full = _gpu(index=7, used=23000, util=90, apps=True)
        self._patch(monkeypatch, [full, _gpu(index=2)])

        assert gpus.pick_shared_or_free(7000, prefer_index=7).index == 2

    def test_a_busy_card_of_somebody_elses_is_never_shared(self, monkeypatch):
        """The preference is for a card WE hold. A colleague's job keeps the whole card."""
        theirs = _gpu(index=3, used=9000, util=100, apps=True)
        self._patch(monkeypatch, [theirs, _gpu(index=2)])

        assert gpus.pick_shared_or_free(7000, prefer_index=None).index == 2

    def test_nothing_anywhere_is_none(self, monkeypatch):
        self._patch(monkeypatch, [_gpu(index=i, used=23000, apps=True) for i in range(4)])

        assert gpus.pick_shared_or_free(7000, prefer_index=0) is None

    def test_free_mib_is_what_is_left_not_whether_it_is_idle(self):
        assert _gpu(used=14425, total=24564).free_mib == 10139
