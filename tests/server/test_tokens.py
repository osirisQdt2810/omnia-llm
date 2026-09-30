"""The token store: hashed, private, revocable at once."""

from __future__ import annotations

import json
import stat

import pytest

from omnia_llm.server.tokens import TokenStore


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


def test_a_wrong_token_or_a_prefix_of_one_is_refused(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    token = store.issue("laptop")

    assert store.verify(token[:-1] + ("a" if token[-1] != "a" else "b")) is None
    assert store.verify(token[:8]) is None


@pytest.mark.parametrize("issued", [False, True])
def test_presenting_nothing_is_never_a_match(tmp_path, issued):
    """Also with no tokens issued at all: a gateway with an empty store must not then accept
    every request that simply sends no key."""
    store = TokenStore(tmp_path / "tokens.json")
    if issued:
        store.issue("laptop")

    assert store.verify("") is None
