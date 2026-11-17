"""How busy the host's CPUs are, the other half of "free" on a machine other people share.

A model that keeps part of itself in system memory (llama.cpp's ``--n-cpu-moe``) computes that
part on the CPU. It is fast while the CPUs are idle and crawls when other users saturate them,
so whether to start a model that way is decided from the load at the moment it starts.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional


@dataclasses.dataclass(frozen=True)
class CpuLoad:
    """The host's CPU load at one moment, and whether that counts as busy."""

    #: The 1-minute load average: processes running or waiting for a CPU.
    load: float
    cpus: int
    #: Load per CPU at or above which the CPU counts as busy (``[server] cpu_busy_load``).
    busy_from: float

    @classmethod
    def read(cls, busy_from: float) -> Optional["CpuLoad"]:
        """The load now, or None where the platform cannot tell (it has no load average)."""
        try:
            load = os.getloadavg()[0]
        except (OSError, AttributeError):  # AttributeError: no getloadavg on this platform
            return None
        return cls(load=load, cpus=os.cpu_count() or 1, busy_from=busy_from)

    @property
    def per_cpu(self) -> float:
        return self.load / self.cpus

    @property
    def busy(self) -> bool:
        return self.per_cpu >= self.busy_from

    def __str__(self) -> str:
        return (f"CPU {'busy' if self.busy else 'free'}: load {self.load:.1f} on {self.cpus} "
                f"CPUs, {self.per_cpu:.2f} per CPU (busy from {self.busy_from:.2f})")
