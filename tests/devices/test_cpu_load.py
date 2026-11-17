"""When the host's CPU counts as busy, and reading it."""

from __future__ import annotations

import os

from omnia_llm.devices import CpuLoad
from tests.helpers import cpu


def test_a_load_below_the_threshold_per_cpu_is_free():
    assert not cpu(4).busy


def test_a_load_at_the_threshold_per_cpu_is_busy():
    assert cpu(36, cpus=48, busy_from=0.75).busy


def test_a_saturated_machine_is_busy():
    """The load other users put on the machine when Gemma fell under one token a second."""
    assert cpu(73, cpus=48).busy


def test_it_says_what_it_decided_and_from_what():
    said = str(cpu(73, cpus=48))
    assert said.startswith("CPU busy") and "73.0 on 48 CPUs" in said and "0.70" in said


def test_it_reads_the_one_minute_load(monkeypatch):
    monkeypatch.setattr(os, "getloadavg", lambda: (12.0, 30.0, 40.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert CpuLoad.read(0.7) == CpuLoad(load=12.0, cpus=16, busy_from=0.7)


def test_a_load_it_cannot_read_is_none(monkeypatch):
    def unreadable():
        raise OSError("no load average here")

    monkeypatch.setattr(os, "getloadavg", unreadable)
    assert CpuLoad.read(0.7) is None
