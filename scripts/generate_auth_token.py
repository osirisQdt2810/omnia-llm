#!/usr/bin/env python3
"""Issue, list and revoke access tokens — a shortcut for ``omnia-llm token``.

    scripts/generate_auth_token.py <name>             issue a token for <name>; printed ONCE
    scripts/generate_auth_token.py --list             live tokens (names and dates, no secrets)
    scripts/generate_auth_token.py --revoke <name>    revoke it; effective on the next request

Tokens live in state/. Only a config that moved it needs -c/--config (or OMNIA_LLM_CONFIG).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnia_llm.cli import main  # noqa: E402


def run() -> int:
    parser = argparse.ArgumentParser(description="Issue, list and revoke gateway tokens.")
    parser.add_argument("name", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--revoke", metavar="NAME")
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    config = ["-c", args.config] if args.config else []
    if args.list:
        return main(["token", "list", *config])
    if args.revoke:
        return main(["token", "revoke", args.revoke, *config])
    if not args.name:
        parser.print_help()
        return 2
    return main(["token", "issue", args.name, *config])


if __name__ == "__main__":
    sys.exit(run())
