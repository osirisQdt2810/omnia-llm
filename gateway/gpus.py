"""Which GPU this machine can lend us, and the definition of "can".

The box has eight A5000s and colleagues working on them. Everything here is written from that
constraint: taking a card somebody is using does not merely slow them down, it can push their
job into an out-of-memory death several hours into a run. So the test for "free" is
deliberately strict, and a wrong answer fails towards *not starting*.
"""

from __future__ import annotations

import dataclasses
import subprocess
from typing import Optional

#: A card holding less than this many MiB counts as idle. Not zero: the driver itself, and a
#: display server on GPU 0, hold a few tens of MiB on a card nobody is computing on.
IDLE_MEMORY_MIB = 512


@dataclasses.dataclass(frozen=True)
class Gpu:
    """One card, as ``nvidia-smi`` reports it."""

    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_pct: int
    #: True when some process holds a CUDA context on this card, whatever its memory says.
    has_compute_apps: bool

    @property
    def free_mib(self) -> int:
        """Memory not currently allocated on this card.

        What a SECOND load has to fit into, which is a different question from whether the card
        is idle: a card holding a 14B model has 10 GB free and is not free at all.
        """
        return max(0, self.memory_total_mib - self.memory_used_mib)

    @property
    def is_free(self) -> bool:
        """Whether we may take this card.

        Three signals, all of which must agree, because each misses a case the others catch:
        a process that has just started holds a context before it allocates; a process between
        batches reports 0% utilization while holding gigabytes; and a leaked context can hold
        memory with no live process. Requiring all three is what makes a false "free" unlikely.
        """
        return (
            not self.has_compute_apps
            and self.memory_used_mib < IDLE_MEMORY_MIB
            and self.utilization_pct < 10
        )


def _run(args: list[str]) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=30
    ).stdout


def query() -> list[Gpu]:
    """Every GPU on the box, in index order.

    Raises:
        RuntimeError: when ``nvidia-smi`` cannot be read at all. The caller must not guess a
            card in that case — picking one blind is exactly the failure this module exists to
            avoid.
    """
    try:
        rows = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,uuid",
                "--format=csv,noheader,nounits",
            ]
        )
        busy_uuids = {
            line.split(",")[0].strip()
            for line in _run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader",
                ]
            ).splitlines()
            if line.strip()
        }
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as one clear failure
        raise RuntimeError(f"could not read nvidia-smi: {exc}") from exc

    gpus: list[Gpu] = []
    for line in rows.splitlines():
        if not line.strip():
            continue
        index, name, total, used, util, uuid = (p.strip() for p in line.split(","))
        gpus.append(
            Gpu(
                index=int(index),
                name=name,
                memory_total_mib=int(total),
                memory_used_mib=int(used),
                utilization_pct=int(util),
                has_compute_apps=uuid in busy_uuids,
            )
        )
    return gpus


def pick_shared_or_free(
    required_mib: int, prefer_index: Optional[int] = None
) -> Optional[Gpu]:
    """A card with ``required_mib`` to spare, preferring one we are ALREADY using.

    The reason this is not simply :func:`pick_free`: two engines on this machine are meant to
    share one card, and a card we hold is not "free" by any honest definition — it has a
    compute process on it and gigabytes allocated. Asking for a free card would take a SECOND
    one from the pool, which is the opposite of what sharing means and the opposite of what was
    asked for.

    So the preference is deliberate rather than incidental: fill the card we are on until it
    genuinely cannot take the load, and only then reach for another. A colleague loses one card
    to us, not two.

    Falling back to an idle card rather than failing, because "the model we already loaded left
    no room" is not a reason to refuse to generate — it is a reason to use the spare capacity
    the machine actually has.

    Args:
        required_mib: How much free memory the new load needs on the card.
        prefer_index: A card we already hold. Tried first; used when it has the room.
    """
    cards = {gpu.index: gpu for gpu in query()}
    preferred = cards.get(prefer_index) if prefer_index is not None else None
    if preferred is not None and preferred.free_mib >= required_mib:
        return preferred
    free = [g for g in cards.values() if g.is_free and g.free_mib >= required_mib]
    return max(free, key=lambda g: g.index) if free else None
