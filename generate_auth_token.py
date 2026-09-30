#!/usr/bin/env python3
"""Issue, list and revoke the tokens that open this gateway.

    python generate_auth_token.py <name>           issue a token for <name>; printed ONCE
    python generate_auth_token.py --list           live tokens (names and dates, no secrets)
    python generate_auth_token.py --revoke <name>  revoke it; effective on the next request

Only someone with a shell here can run this, which is what makes a public URL acceptable: the
URL alone opens nothing. The gateway keeps only a hash of each token; lose one, issue another.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gateway.settings import Settings  # noqa: E402
from gateway.tokens import TokenStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue, list and revoke gateway tokens.")
    parser.add_argument("name", nargs="?", help="who or what the token is for, e.g. phuc-mac")
    parser.add_argument("--list", action="store_true", help="list live tokens")
    parser.add_argument("--revoke", metavar="NAME", help="revoke the token called NAME")
    args = parser.parse_args()
    store = TokenStore(Settings.load().tokens_file)

    if args.list:
        rows = store.names()
        for row in rows:
            print(f"{row['name']:24} issued {row['created']}")
        if not rows:
            print("(no tokens)")
        return 0
    if args.revoke:
        if store.revoke(args.revoke):
            print(f"revoked {args.revoke!r}")
            return 0
        print(f"no token called {args.revoke!r}", file=sys.stderr)
        return 1
    if not args.name:
        parser.print_help()
        return 2
    try:
        token = store.issue(args.name)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(token)
    print(f"\nIssued for {args.name!r}. Shown this once only — paste it into Omnia as the "
          "endpoint's API key.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
