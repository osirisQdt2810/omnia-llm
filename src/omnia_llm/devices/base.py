"""Where a model can run, and the definition of "free" on a machine other people share.

A :class:`DeviceProbe` lists the devices of one kind of hardware and knows how to pin a process
to one of them. Everything else — "is it free", "which one do we take", "share the card we
already hold" — is decided HERE, once, so NVIDIA, ROCm, Apple Silicon and CPU cannot disagree.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import ClassVar, Iterable, Optional

#: A device holding less than this many MiB counts as idle. Not zero: the driver itself, and a
#: display server on GPU 0, hold a few tens of MiB on a device nobody is computing on.
IDLE_MEMORY_MIB = 512
#: Utilisation at or above this means somebody is computing, whatever the memory says.
IDLE_UTILIZATION_PCT = 10


class DeviceError(RuntimeError):
    """The devices could not be read. Never guess one in that case."""


@dataclasses.dataclass(frozen=True)
class Device:
    """One device, as its probe reports it."""

    backend: str
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int = 0
    utilization_pct: int = 0
    #: True when some process holds a compute context here, whatever its memory says.
    has_compute_apps: bool = False
    #: Environment that pins a process to exactly this device (e.g. CUDA_VISIBLE_DEVICES).
    env: tuple[tuple[str, str], ...] = ()
    #: A device only this user runs on (a laptop). Busy signals do not apply there.
    exclusive: bool = False

    @property
    def id(self) -> str:
        return f"{self.backend}:{self.index}"

    @property
    def free_mib(self) -> int:
        """Memory not allocated here — what a SECOND load has to fit into."""
        return max(0, self.memory_total_mib - self.memory_used_mib)

    @property
    def is_free(self) -> bool:
        """Whether we may take this device.

        On a shared GPU, three signals must all agree, because each misses a case the others
        catch: a just-started process holds a context before it allocates; a process between
        batches shows 0% utilisation while holding gigabytes; a leaked context holds memory
        with no live process. A wrong answer fails towards NOT starting.
        """
        if self.exclusive:
            return True
        return (
            not self.has_compute_apps
            and self.memory_used_mib < IDLE_MEMORY_MIB
            and self.utilization_pct < IDLE_UTILIZATION_PCT
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "utilization_pct": self.utilization_pct,
            "free": self.is_free,
        }


class DeviceProbe(abc.ABC):
    """Lists the devices of one kind of hardware. Subclass + ``@register_device_probe``."""

    #: Lower runs first when the config says ``probe = "auto"``.
    priority: ClassVar[int] = 100
    registry_name: ClassVar[str] = ""

    @classmethod
    @abc.abstractmethod
    def available(cls) -> bool:
        """Whether this hardware is present here (cheap: a binary on PATH, a platform check)."""

    @abc.abstractmethod
    def list_devices(self) -> list[Device]:
        """Every device, in index order.

        Raises:
            DeviceError: when they cannot be read at all.
        """

    def pick(self, required_mib: int, prefer: Iterable[str] = ()) -> Optional[Device]:
        """A device with ``required_mib`` to spare, preferring one we ALREADY hold.

        Two models are meant to share one device: a device we hold is not "free" by any honest
        definition, and asking for a free one would take a SECOND from the pool. So fill the one
        we are on until it cannot take the load, then fall back to an idle one — the highest
        index, leaving low indices (often the display GPU) for people.
        """
        devices = {d.id: d for d in self.list_devices()}
        for wanted in prefer:
            held = devices.get(wanted)
            if held is not None and held.free_mib >= required_mib:
                return held
        free = [d for d in devices.values() if d.is_free and d.free_mib >= required_mib]
        return max(free, key=lambda d: d.index) if free else None
