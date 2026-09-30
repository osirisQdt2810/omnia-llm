"""The gateway: an OpenAI-compatible HTTP front for the configured models.

Every path but ``/health`` needs a token, checked in ONE place, a middleware that runs before
any route, so no route can forget it. A request that is authorised may start its model (lazily,
on a free device); nothing unauthorised ever can. What a client may rely on:
https://github.com/osirisQdt2810/omnia/blob/main/docs/guidance/local-server/server-contract.md
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from omnia_llm import __version__
from omnia_llm.config import AppConfig
from omnia_llm.devices import DeviceError, DeviceProbe, detect
from omnia_llm.engines import Engine, EngineBusy
from omnia_llm.server import auth
from omnia_llm.server.lockout import Lockout
from omnia_llm.server.manager import ModelManager
from omnia_llm.server.tokens import TokenStore

#: The only paths answered without a token. Matched exactly: "/health/" is not one of them.
OPEN_PATHS = frozenset({"/health"})
#: Hop-by-hop headers a proxy must not copy through, plus the ones httpx sets itself.
_SKIP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}


def _error(status: int, message: str, kind: str, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status,
                        headers=headers)


def _client(request: Request) -> str:
    """Who is asking, for the lockout.

    Behind ngrok every request arrives from loopback, so the address comes from X-Forwarded-For:
    the LAST hop across ALL such headers. ngrok forwards the client's own header untouched and
    adds its own as a SECOND header — earlier hops are whatever the client chose to send, and
    keying on them would let a guesser dodge the lockout by sending a new one each time.
    """
    hops = [h.strip() for v in request.headers.getlist("x-forwarded-for") for h in v.split(",")]
    hops = [h for h in hops if h]
    if hops:
        return hops[-1]
    return request.client.host if request.client else "unknown"


def create_app(config: AppConfig, probe: Optional[DeviceProbe] = None) -> FastAPI:
    probe = probe or detect(config.devices.probe)
    manager = ModelManager(config, probe)
    tokens = TokenStore(config.paths.tokens_file)
    legacy = auth.read_legacy_key(config.paths.legacy_key_file)
    if legacy:
        tokens.adopt(auth.LEGACY_TOKEN, legacy)
    lockout = Lockout()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        manager.start_reaper()
        try:
            yield
        finally:
            # Never leave a device held by a gateway that is going away.
            await manager.shutdown()

    # No schema and no docs pages: no client needs them, and they would be one more route to
    # keep behind the token.
    app = FastAPI(title="omnia-llm", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.manager, app.state.tokens, app.state.lockout = manager, tokens, lockout

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        """Let a request in or not: the one place that decides, and the lockout's one witness.

        The lockout records THIS decision: a missing or wrong token is a failure, a good one a
        success, an open path neither. Guessed from status codes instead, any path answering
        below 400 without a token (FastAPI's docs, the 307 for "/health/") cleared a guesser's
        count, and interleaving one made guessing free.
        """
        client = _client(request)
        left = lockout.blocked(client)
        if left:
            return _error(429, "too many wrong tokens; try again later", "locked",
                          {"Retry-After": str(int(left) + 1)})
        if request.url.path in OPEN_PATHS:
            return await call_next(request)
        presented = auth.presented_key(request.headers.get("authorization"),
                                       request.headers.get("x-api-key"))
        if tokens.verify(presented) is None:
            lockout.failed(client)
            # The same body whether the key was missing, malformed or wrong.
            return _error(401, "invalid or missing API key", "unauthorized",
                          {"WWW-Authenticate": "Bearer"})
        lockout.succeeded(client)
        return await call_next(request)

    @app.get("/health")
    async def health() -> JSONResponse:
        """Liveness only: no token, and nothing in it worth having."""
        return JSONResponse({"ok": True})

    @app.get("/status")
    async def status() -> JSONResponse:
        try:
            devices = [d.to_dict() for d in probe.list_devices()]
        except DeviceError as exc:
            devices = [{"error": str(exc)}]
        return JSONResponse({"models": manager.status(), "devices": devices,
                             "time": time.strftime("%F %T")})

    @app.post("/warm")
    async def warm(request: Request) -> JSONResponse:
        """Start models now and return at once (202); poll /status to see them become ready."""
        body = await _json(request)
        kinds = body.get("kinds") or (["text"] if body.get("images") is False else ["text", "image"])
        return JSONResponse({"warming": manager.warm(kinds), "models": manager.status()},
                            status_code=202)

    @app.post("/stop")
    async def stop() -> JSONResponse:
        """Hand every device back now, without waiting out the idle timer."""
        await manager.stop_all("asked to stop")
        return JSONResponse({"stopped": True})

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def v1(path: str, request: Request) -> Response:
        """Everything OpenAI-shaped, routed on the path's segments.

        Empty segments are dropped: Omnia asks for ``<base>/models``, and a Base URL typed with
        a trailing slash makes that ``/v1//models``. Dot segments are refused, not resolved:
        forwarded, ``/v1/%2e%2e/metrics`` walks out of /v1 into the engine's own endpoints.
        """
        segments = [s for s in path.split("/") if s]
        if "." in segments or ".." in segments:
            return _error(400, "a path may not contain '.' or '..' segments", "invalid_request")
        if segments[:1] == ["models"]:
            return _models(manager, "/".join(segments[1:]))
        kind = "image" if segments == ["images", "generations"] else "text"
        body = await request.body()
        engine = manager.route(kind, _model_of(body))
        if engine is None:
            return _error(501, f"no {kind} model is configured on this gateway", "not_configured")
        return await _forward(manager, engine, request, body, "/v1/" + "/".join(segments))

    return app


def _models(manager: ModelManager, wanted: str) -> JSONResponse:
    """``/v1/models`` and ``/v1/models/<id>``, answered from config WITHOUT starting anything:
    clients call them to check a connection, and a cold start for that would burn a minute
    and a GPU."""
    listing = manager.listing()
    if not wanted:
        return JSONResponse({"object": "list", "data": listing})
    for entry in listing:
        if entry["id"] == wanted:
            return JSONResponse(entry)
    return _error(404, f"no model called {wanted!r} on this gateway", "not_found")


async def _json(request: Request) -> dict:
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001 - an empty or non-JSON body means "the defaults"
        return {}
    return data if isinstance(data, dict) else {}


def _model_of(body: bytes) -> str:
    try:
        data = json.loads(body) if body else {}
    except ValueError:
        return ""
    return str(data.get("model") or "") if isinstance(data, dict) else ""


def _described(exc: Exception) -> str:
    """The exception's message, or its type where it has none (httpx's timeouts have none)."""
    return str(exc) or type(exc).__name__


async def _forward(manager: ModelManager, engine: Engine, request: Request, body: bytes,
                   path: str) -> Response:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_HEADERS}
    # Our token is ours; the engine is not asked to verify it and must not see it.
    headers.pop("authorization", None)
    headers.pop("x-api-key", None)
    try:
        upstream = await manager.forward(engine, request.method, path, body, headers)
    except EngineBusy as exc:
        # 503 + Retry-After: the honest answer is "come back", not "something broke".
        return _error(503, str(exc), "gpu_unavailable", {"Retry-After": "120"})
    except httpx.HTTPError as exc:
        # It started, then did not answer, or not in time: not the same thing to go and fix.
        return _error(502, f"the local engine did not answer: {_described(exc)}", "engine_error")
    except Exception as exc:  # noqa: BLE001 - one clear message, never a traceback
        return _error(502, f"local engine failed to start: {_described(exc)}", "engine_error")
    passthrough = {k: v for k, v in upstream.headers.items() if k.lower() not in _SKIP_HEADERS}
    if "text/event-stream" in upstream.headers.get("content-type", ""):
        return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code,
                                 headers=passthrough,
                                 media_type=upstream.headers.get("content-type"))
    return Response(content=upstream.content, status_code=upstream.status_code,
                    headers=passthrough)
