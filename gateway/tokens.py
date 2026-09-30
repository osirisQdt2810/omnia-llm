"""Named access tokens: issued by an operator on this machine, stored only as hashes, revocable.

The gateway is reachable from the internet (through ngrok), so a token is the whole boundary.
That is acceptable because a token can only be MADE here, by someone with a shell on this box
and read access to this directory — and it is made safe by four properties:

* **Random and long** — 32 bytes from ``secrets``; guessing is not a strategy, and the lockout
  (``lockout.py``) stops anyone trying at scale.
* **Stored as SHA-256 only.** ``tokens.json`` never holds a usable token; the plaintext is
  printed once, at issue, and nowhere else. Reading the file grants nothing.
* **Named, one per client**, so a leaked one is revoked without disturbing the others.
* **Revocation is immediate**: the file is re-read whenever it changes, no restart needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

_TOKEN_BYTES = 32
PREFIX = "omnia-"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenStore:
    """The set of live tokens, backed by a 0600 JSON file of hashes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mtime: Optional[float] = None
        self._entries: list[dict] = []

    def _load(self) -> list[dict]:
        try:
            st = self._path.stat()
        except FileNotFoundError:
            self._mtime, self._entries = None, []
            return self._entries
        # Inode and size as well as mtime: two writes can land inside one timestamp tick, and a
        # revocation missed that way would leave a revoked token working. Every save is an
        # atomic rename, so the inode always changes.
        mtime = (st.st_mtime_ns, st.st_ino, st.st_size)
        if mtime != self._mtime:
            data = json.loads(self._path.read_text(encoding="utf-8") or "{}")
            self._entries = list(data.get("tokens", []))
            self._mtime = mtime
        return self._entries

    def names(self) -> list[dict]:
        """Every live token without its hash: name and when it was issued."""
        return [{"name": e["name"], "created": e.get("created", "")} for e in self._load()]

    def verify(self, presented: str) -> Optional[str]:
        """The name of the token ``presented`` is, or None."""
        if not presented:
            return None
        digest = _digest(presented)
        match = None
        for entry in self._load():
            # Every entry is compared, not stopping at the first hit, so timing does not tell
            # where in the list a token sits.
            if secrets.compare_digest(digest, entry["sha256"]):
                match = entry["name"]
        return match

    def _save(self, entries: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        # 0600 from creation, then an atomic rename: never readable by others, never half-written.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"tokens": entries}, handle, indent=2)
        os.replace(tmp, self._path)
        self._mtime = None

    def issue(self, name: str) -> str:
        """Create a token called ``name``; return its plaintext — the only time it exists."""
        name = name.strip()
        if not name:
            raise ValueError("a token needs a name (who or what it is for)")
        entries = self._load()
        if any(e["name"] == name for e in entries):
            raise ValueError(f"there is already a token called {name!r}; revoke it first")
        token = PREFIX + secrets.token_urlsafe(_TOKEN_BYTES)
        entry = {"name": name, "sha256": _digest(token), "created": time.strftime("%F %T")}
        self._save(entries + [entry])
        return token

    def adopt(self, name: str, token: str) -> None:
        """Record an EXISTING token under ``name`` (carries the old single key across)."""
        entries = self._load()
        if any(e["name"] == name or e["sha256"] == _digest(token) for e in entries):
            return
        entry = {"name": name, "sha256": _digest(token), "created": time.strftime("%F %T")}
        self._save(entries + [entry])

    def revoke(self, name: str) -> bool:
        """Remove ``name``; effective on the gateway's next request. False if unknown."""
        entries = self._load()
        kept = [e for e in entries if e["name"] != name]
        if len(kept) == len(entries):
            return False
        self._save(kept)
        return True
