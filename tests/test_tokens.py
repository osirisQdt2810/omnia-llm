import json
import stat

from gateway.lockout import BLOCK_SECONDS, MAX_FAILURES, Lockout
from gateway.tokens import TokenStore


def test_an_issued_token_verifies_under_its_name(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    token = store.issue("phuc-mac")
    assert store.verify(token) == "phuc-mac"
    assert store.verify(token + "x") is None
    assert store.verify("") is None


def test_the_file_holds_hashes_only_and_is_private(tmp_path):
    path = tmp_path / "tokens.json"
    token = TokenStore(path).issue("a")
    assert token not in path.read_text()
    assert json.loads(path.read_text())["tokens"][0]["sha256"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_revoking_takes_effect_without_a_restart(tmp_path):
    path = tmp_path / "tokens.json"
    gateway = TokenStore(path)          # what the running gateway holds
    token = TokenStore(path).issue("a")
    assert gateway.verify(token) == "a"
    assert TokenStore(path).revoke("a")  # the script, a separate process
    assert gateway.verify(token) is None


def test_one_revocation_leaves_the_others(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    a, b = store.issue("a"), store.issue("b")
    store.revoke("a")
    assert store.verify(a) is None and store.verify(b) == "b"


def test_names_are_unique_and_required(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    store.issue("a")
    for bad in ("a", " "):
        try:
            store.issue(bad)
        except ValueError:
            continue
        raise AssertionError(f"issued a duplicate or blank name: {bad!r}")


def test_the_old_single_key_is_adopted_once(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    store.adopt("legacy", "omnia-old")
    store.adopt("legacy", "omnia-old")
    assert store.verify("omnia-old") == "legacy"
    assert [n["name"] for n in store.names()] == ["legacy"]


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
