"""Reading the presented token, and the single key of the first version.

Tokens are issued and checked by ``tokens.py``. Before named tokens existed, a deployment had one
key, in ``state/api-key.txt``. Where that file exists it is adopted as the token ``legacy``, so
clients configured with it keep working until it is revoked. A new install gets no such key:
every token it accepts is one somebody issued.
"""

from __future__ import annotations

import stat
from pathlib import Path

#: The name the first version's key carries in the token store.
LEGACY_TOKEN = "legacy"


def read_legacy_key(path: Path) -> str:
    """The first version's key, or "" where there is none (a new install, or it was retired).

    A file looser than ``0600`` is narrowed: on a machine with other users, a world-readable key
    file is the same as no key. A blank file is no key — a truncated file must not become an
    empty key that then matches a request presenting none.
    """
    if not path.is_file():
        return ""
    key = path.read_text(encoding="utf-8").strip()
    if key:
        _harden(path)
    return key


def retire_legacy_key(path: Path) -> bool:
    """Delete the first version's key file; True if there was one.

    Revoking ``legacy`` has to do this too. The gateway adopts the file at every start, so a
    revocation that left the file behind would be undone by the next restart.
    """
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


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
