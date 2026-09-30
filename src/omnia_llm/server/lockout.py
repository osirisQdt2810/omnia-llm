"""Refuse a client that keeps presenting wrong tokens.

A 32-byte token cannot be guessed, so this is not what keeps it secret. It keeps a public URL
from being a free way to make this machine do work: every wrong attempt costs a request, a log
line and a hash, and someone scripting them should be told to stop, not answered forever.
"""

from __future__ import annotations

import time

MAX_FAILURES = 10          # wrong tokens within the window...
WINDOW_SECONDS = 600.0     # ...of ten minutes...
BLOCK_SECONDS = 900.0      # ...refuse that client for fifteen.
#: Past this many remembered clients, the aged-out ones are dropped at the next failure.
SWEEP_ABOVE = 256


class Lockout:
    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._failures: dict[str, list[float]] = {}
        self._blocked_until: dict[str, float] = {}

    def blocked(self, client: str) -> float:
        """Seconds left on ``client``'s block, or 0."""
        left = self._blocked_until.get(client, 0.0) - self._clock()
        if left <= 0:
            self._blocked_until.pop(client, None)
            return 0.0
        return left

    def failed(self, client: str) -> None:
        now = self._clock()
        if len(self._failures) + len(self._blocked_until) > SWEEP_ABOVE:
            self._sweep(now)
        recent = [t for t in self._failures.get(client, []) if now - t < WINDOW_SECONDS]
        recent.append(now)
        if len(recent) >= MAX_FAILURES:
            self._blocked_until[client] = now + BLOCK_SECONDS
            recent = []
        self._failures[client] = recent

    def succeeded(self, client: str) -> None:
        self._failures.pop(client, None)

    def _sweep(self, now: float) -> None:
        """Forget clients whose failures have all aged out, and blocks that have run out: a
        public URL would otherwise keep one entry for every address that ever guessed."""
        self._failures = {c: ts for c, ts in self._failures.items()
                          if ts and now - ts[-1] < WINDOW_SECONDS}
        self._blocked_until = {c: t for c, t in self._blocked_until.items() if t > now}
