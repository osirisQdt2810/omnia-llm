"""The lockout: repeated wrong tokens from one client."""

from __future__ import annotations

from omnia_llm.server.lockout import BLOCK_SECONDS, MAX_FAILURES, Lockout


def test_a_client_is_blocked_after_repeated_failures():
    now = [0.0]
    lock = Lockout(clock=lambda: now[0])
    for _ in range(MAX_FAILURES - 1):
        lock.failed("1.2.3.4")
    assert lock.blocked("1.2.3.4") == 0
    lock.failed("1.2.3.4")
    assert lock.blocked("1.2.3.4") > 0
    assert lock.blocked("5.6.7.8") == 0, "one client's failures must not block another"
    now[0] += BLOCK_SECONDS + 1
    assert lock.blocked("1.2.3.4") == 0


def test_success_clears_the_count():
    lock = Lockout(clock=lambda: 0.0)
    for _ in range(MAX_FAILURES - 1):
        lock.failed("c")
    lock.succeeded("c")
    lock.failed("c")
    assert lock.blocked("c") == 0
