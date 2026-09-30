"""``omnia-llm`` — serve models, manage tokens, and inspect what this machine can run.

    omnia-llm serve   -c configs/<name>.toml
    omnia-llm token   issue <name> | list | revoke <name>   [-c configs/<name>.toml]
    omnia-llm devices [--probe auto|nvidia|rocm|apple|cpu]
    omnia-llm models  -c configs/<name>.toml

The config can also come from ``OMNIA_LLM_CONFIG``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omnia_llm.config import ROOT, AppConfig, ConfigError


def _config(path: str | None) -> AppConfig:
    chosen = path or os.environ.get("OMNIA_LLM_CONFIG")
    if not chosen:
        found = ", ".join(sorted(p.name for p in (ROOT / "configs").glob("*.toml")))
        raise ConfigError(f"no config given: pass -c configs/<name>.toml (available: {found})")
    candidate = Path(chosen)
    if not candidate.is_absolute() and not candidate.is_file():
        candidate = ROOT / chosen
    return AppConfig.load(candidate)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from omnia_llm.server.app import create_app

    config = _config(args.config)
    uvicorn.run(create_app(config), host=args.host or config.server.host,
                port=args.port or config.server.port, log_level="info")
    return 0


def _token(args: argparse.Namespace) -> int:
    from omnia_llm.config import PathsConfig
    from omnia_llm.server import auth
    from omnia_llm.server.tokens import TokenStore

    # Tokens depend only on where state/ is, which every shipped config leaves at its default:
    # a config is needed only by a deployment that moved it.
    given = args.config or os.environ.get("OMNIA_LLM_CONFIG")
    paths = _config(given).paths if given else PathsConfig()
    store = TokenStore(paths.tokens_file)
    if args.action == "list":
        rows = store.names()
        for row in rows:
            print(f"{row['name']:24} issued {row['created']}")
        if not rows:
            print("(no tokens)")
        return 0
    if not args.name:
        print(f"token {args.action}: a name is required", file=sys.stderr)
        return 2
    if args.action == "revoke":
        revoked = store.revoke(args.name)
        # The gateway adopts the old key file at every start: leave it and a restart un-revokes.
        retired = args.name == auth.LEGACY_TOKEN and auth.retire_legacy_key(
            paths.legacy_key_file)
        if revoked or retired:
            print(f"revoked {args.name!r} — effective on the next request")
            return 0
        print(f"no token called {args.name!r}", file=sys.stderr)
        return 1
    token = store.issue(args.name)
    print(token)
    print(f"\nIssued for {args.name!r}. Shown this once only — paste it into Omnia as the "
          "endpoint's API key.", file=sys.stderr)
    return 0


def _devices(args: argparse.Namespace) -> int:
    from omnia_llm.devices import detect

    probe = detect(args.probe)
    print(f"probe: {type(probe).registry_name}")
    for device in probe.list_devices():
        state = "free" if device.is_free else "busy"
        print(f"  {device.id:10} {device.name:28} {device.memory_used_mib:>6}/"
              f"{device.memory_total_mib} MiB  {device.utilization_pct:>3}%  {state}")
    return 0


def _models(args: argparse.Namespace) -> int:
    import omnia_llm.platforms  # noqa: F401 - registration
    from omnia_llm.devices import detect
    from omnia_llm.engines import ENGINES

    config = _config(args.config)
    backend = type(detect(config.devices.probe)).registry_name
    for spec in config.models:
        cls = ENGINES.get(spec.engine)
        fits = "ok" if backend in cls.backends else f"NOT on {backend}"
        print(f"  {spec.id:18} {spec.kind:6} {spec.engine:10} port {spec.port}  [{fits}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omnia-llm", description="Serve self-hosted models to Omnia.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the gateway")
    serve.add_argument("-c", "--config")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=_serve)

    token = sub.add_parser("token", help="issue, list or revoke access tokens")
    token.add_argument("action", choices=("issue", "list", "revoke"))
    token.add_argument("name", nargs="?")
    token.add_argument("-c", "--config")
    token.set_defaults(func=_token)

    devices = sub.add_parser("devices", help="what this machine can run on")
    devices.add_argument("--probe", default="auto")
    devices.set_defaults(func=_devices)

    models = sub.add_parser("models", help="the configured models and whether they fit here")
    models.add_argument("-c", "--config")
    models.set_defaults(func=_models)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, KeyError, ValueError, RuntimeError) as exc:
        print(f"omnia-llm: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
