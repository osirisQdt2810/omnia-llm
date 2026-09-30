"""The lockout: repeated wrong tokens from one client."""

from __future__ import annotations

from omnia_llm.server.lockout import (
    BLOCK_SECONDS,
    MAX_FAILURES,
    SWEEP_ABOVE,
    WINDOW_SECONDS,
    Lockout,
)


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


def test_clients_whose_failures_have_aged_out_are_forgotten():
    """On a public URL, one entry for every address that ever guessed would grow for good."""
    now = [0.0]
    lockout = Lockout(clock=lambda: now[0])
    for i in range(SWEEP_ABOVE + 1):
        lockout.failed(f"10.0.{i // 256}.{i % 256}")

    now[0] = WINDOW_SECONDS + BLOCK_SECONDS + 1
    lockout.failed("192.0.2.1")

    assert set(lockout._failures) == {"192.0.2.1"}
    assert lockout._blocked_until == {}


def test_a_sweep_keeps_a_block_that_has_not_run_out():
    now = [0.0]
    lockout = Lockout(clock=lambda: now[0])
    for _ in range(MAX_FAILURES):
        lockout.failed("203.0.113.9")
    for i in range(SWEEP_ABOVE + 1):
        lockout.failed(f"10.1.{i // 256}.{i % 256}")

    now[0] = WINDOW_SECONDS + 1  # every failure has aged out; the block has not
    lockout.failed("198.51.100.7")

    assert lockout.blocked("203.0.113.9") > 0

