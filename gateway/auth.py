"""The gateway's API key: generated here, never checked in, required on every call.

The service binds to loopback and is reached through an SSH tunnel, so this is not the only
thing standing between the model and the internet — it is the thing standing between the model
and everyone else who has a shell on this shared machine. Eight people share the box; loopback
is not a privacy boundary among them.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

#: Long enough that guessing is not a strategy, short enough to paste into a settings box.
_KEY_BYTES = 32


def load_or_create(path: Path) -> str:
    """Return the gateway's key, generating and storing one the first time.

    Generated rather than configured so there is no default to forget to change, and no key
    that ever existed in a file somebody might commit. Written ``0600`` — on a machine with
    other users, a world-readable key file is the same as no key.

    Args:
        path: Where the key lives. Inside the project directory, beside the service it belongs
            to, so it is obvious what it is and obvious to delete.

    Returns:
        The key, as the client must present it.
    """
    if path.is_file():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            _harden(path)
            return existing

    key = "omnia-" + secrets.token_urlsafe(_KEY_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Created 0600 from the start rather than written then chmod'ed: the gap between the two
    # is a window in which the key is readable by everyone on the machine.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    return key


def _harden(path: Path) -> None:
    """Narrow an existing key file to 0600 if it is looser than that."""
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        path.chmod(0o600)


def presented_key(authorization: str | None, x_api_key: str | None) -> str:
    """Pull the key out of whichever header the client used.

    ``Authorization: Bearer …`` is what an OpenAI-compatible client sends, which is the whole
    point: Omnia's existing provider already puts its configured api_key there, so no add-on
    code changes to authenticate. ``X-API-Key`` is accepted too for hand-testing with curl.
    """
    if authorization:
        # split on ANY run of whitespace, after stripping: a header is allowed more than one
        # space after the scheme, and a client that adds one should not be told its key is
        # wrong — the least useful possible error message.
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
    return (x_api_key or "").strip()


def matches(presented: str, expected: str) -> bool:
    """Constant-time comparison — a normal ``==`` leaks the key one character at a time."""
    if not presented or not expected:
        return False
    return secrets.compare_digest(presented, expected)
