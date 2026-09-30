"""The always-on front door: an OpenAI-compatible endpoint that owns a GPU only when busy.

Omnia's ``openai_compatible`` provider talks to this and cannot tell the difference from a
hosted API, which is the point — no new provider, no new code in the add-on, just a base URL.

Everything under ``/v1`` is proxied to vLLM, starting it on the first request. Two endpoints
are answered here instead, because both must work while the engine is DOWN: ``/status`` (so
the operator can see what is happening without waking a GPU) and ``/v1/models`` (so a client
probing for the model list does not cold-start one).
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from gateway import auth, gpus
from gateway.lockout import Lockout
from gateway.tokens import TokenStore
from gateway.engine import Engine, EngineBusy
from gateway.settings import Settings

settings = Settings.load()
TOKENS = TokenStore(settings.tokens_file)
# The single key this gateway started with keeps working, as a token named "legacy" — so the SSH
# tunnel and every client already configured are not cut off — and can now be revoked by name.
TOKENS.adopt("legacy", auth.load_or_create(settings.api_key_file))
LOCKOUT = Lockout()
engine = Engine(settings, "text")
images = Engine(settings, "image")
app = FastAPI(title="omnia-llm gateway", version="1.0")

#: Answered without a key, because a health probe must not need a secret to say "up". It
#: reveals nothing: no model name, no GPU state, no hostname.
_OPEN_PATHS = {"/health"}


def _unauthorized() -> JSONResponse:
    """The same body whether the key was missing, malformed or wrong.

    Distinguishing them tells an attacker which half they got right.
    """
    return JSONResponse(
        {"error": {"message": "invalid or missing API key", "type": "unauthorized"}},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _authorized(authorization: str | None, x_api_key: str | None) -> bool:
    return TOKENS.verify(auth.presented_key(authorization, x_api_key)) is not None


def _client(request: Request) -> str:
    """Who is asking. Behind ngrok every request arrives from loopback, so the address comes
    from X-Forwarded-For — its LAST hop, the one ngrok appended. Every earlier hop is whatever
    the client chose to send, and keying the lockout on those would let a guesser dodge it by
    sending a new one each time."""
    #
    # getlist, not get: ngrok sends the client's own header UNTOUCHED and adds a SECOND one with
    # the real address, and .get() returns only the first — the spoofed one.
    hops = [h.strip() for v in request.headers.getlist("x-forwarded-for") for h in v.split(",")]
    hops = [h for h in hops if h]
    if hops:
        return hops[-1]
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def _lockout(request: Request, call_next):
    """Refuse a client that keeps failing auth; count every 401 the routes return."""
    client = _client(request)
    left = LOCKOUT.blocked(client)
    if left:
        return JSONResponse(
            {"error": {"message": "too many wrong tokens; try again later", "type": "locked"}},
            status_code=429,
            headers={"Retry-After": str(int(left) + 1)},
        )
    response = await call_next(request)
    if response.status_code == 401:
        LOCKOUT.failed(client)
    elif response.status_code < 400 and request.url.path != "/health":
        LOCKOUT.succeeded(client)
    return response

#: Hop-by-hop headers a proxy must not copy through, plus the ones httpx will set itself.
_SKIP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}


@app.on_event("startup")
async def _start_reaper() -> None:
    app.state.reaper = asyncio.create_task(engine.reap_when_idle())
    app.state.image_reaper = asyncio.create_task(images.reap_when_idle())


@app.on_event("shutdown")
async def _shutdown() -> None:
    """Never leave a card held by a gateway that is going away."""
    for name in ("reaper", "image_reaper"):
        task = getattr(app.state, name, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    await engine.stop("gateway shutting down")
    await images.stop("gateway shutting down")


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness only. No key, and nothing in it worth having."""
    return JSONResponse({"ok": True})


@app.get("/status")
async def status(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> JSONResponse:
    """What the engine is doing and what the machine looks like — without waking anything.

    Behind the key: it reports the model, the card in use and every GPU's occupancy, which is
    other people's business on a shared machine.
    """
    if not _authorized(authorization, x_api_key):
        return _unauthorized()
    try:
        cards = [
            {
                "index": g.index,
                "memory_used_mib": g.memory_used_mib,
                "utilization_pct": g.utilization_pct,
                "free": g.is_free,
            }
            for g in gpus.query()
        ]
    except RuntimeError as exc:
        cards = [{"error": str(exc)}]
    return JSONResponse(
        {
            **_engine_states(),
            "gpus": cards,
            "time": time.strftime("%F %T"),
        }
    )


@app.post("/stop")
async def stop_now(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> JSONResponse:
    """Hand the card back immediately, without waiting out the idle timer.

    For the moment a colleague needs the GPU now: one curl beats explaining the timeout.
    """
    if not _authorized(authorization, x_api_key):
        return _unauthorized()
    await engine.stop("asked to stop")
    await images.stop("asked to stop")
    return JSONResponse({"stopped": True})


#: Warm-ups in flight, so /status can say "starting" and a second /warm does not start another.
_warming: dict[str, asyncio.Task] = {}


async def _warm_one(role: str, target: Engine) -> None:
    other = images if target is engine else engine
    target._share_with = other.status()["gpu"]
    try:
        await target.ensure_running()
    finally:
        _warming.pop(role, None)


@app.post("/warm")
async def warm(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> JSONResponse:
    """Start the engines NOW, and return at once; poll /status to see them become ready.

    A cold start takes a minute and a half. A client that knows it is about to need the model —
    a batch about to begin — asks here first instead of letting its first real request sit in a
    timeout. ``{"images": false}`` in the body wakes only the text engine.
    """
    if not _authorized(authorization, x_api_key):
        return _unauthorized()
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - an empty or non-JSON body means "the defaults"
        body = {}
    roles = [("text", engine)]
    if body.get("images", True) and settings.image_model:
        roles.append(("image", images))
    for role, target in roles:
        if role not in _warming and not target.status()["running"]:
            _warming[role] = asyncio.create_task(_warm_one(role, target))
    return JSONResponse({"warming": sorted(_warming), **_engine_states()}, status_code=202)


def _engine_states() -> dict:
    return {
        "engine": {**engine.status(), "starting": "text" in _warming},
        "images": {**images.status(), "starting": "image" in _warming},
    }


@app.get("/v1/models")
async def models(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> JSONResponse:
    """Answered WITHOUT starting the engine.

    Clients list models to check a connection. Cold-starting a 14B because something probed
    the endpoint would burn a minute and a GPU for a question this can answer from config.
    """
    if not _authorized(authorization, x_api_key):
        return _unauthorized()
    # "kind" is Omnia's extension to the OpenAI shape: it is how a client offers the image
    # model in its image picker and the text one in its text picker. Stock clients ignore it.
    data = [
        {"id": settings.served_model_name, "object": "model", "owned_by": "omnia-llm",
         "kind": "text"},
    ]
    if settings.image_model:
        data.append({"id": settings.image_model.rsplit("/", 1)[-1], "object": "model",
                     "owned_by": "omnia-llm", "kind": "image"})
    return JSONResponse({"object": "list", "data": data})


@app.post("/v1/images/generations")
async def images_generations(request: Request) -> Response:
    """Text-to-image, in the shape Omnia's provider already speaks.

    Routed to its own engine rather than to vLLM, which cannot serve a diffusion model — but
    the caller never learns that, which is the point: one base URL, one key, and the add-on
    needs no idea which kind of model is behind each path.

    Declared BEFORE the catch-all below, because FastAPI matches in declaration order and
    ``/v1/{path}`` would otherwise swallow it and hand an image request to the text engine.
    """
    if not _authorized(request.headers.get("authorization"), request.headers.get("x-api-key")):
        return _unauthorized()
    if not settings.image_model:
        # An honest refusal rather than a timeout: nothing is configured to answer this, and
        # starting something the operator never asked for on a shared card is not the fix.
        return JSONResponse(
            {
                "error": {
                    "message": (
                        "no image model is configured on this gateway — set image_model in "
                        "config.toml"
                    ),
                    "type": "not_configured",
                }
            },
            status_code=501,
        )
    return await _forward(images, request, "/v1/images/generations")


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request) -> Response:
    """Forward anything else to vLLM, starting it if it is not up."""
    if not _authorized(request.headers.get("authorization"), request.headers.get("x-api-key")):
        return _unauthorized()
    return await _forward(engine, request, f"/v1/{path}")


async def _forward(target: Engine, request: Request, path: str) -> Response:
    """Send one already-authorised request to ``target``, starting it if it is not up.

    Each engine is told, every time, which card the other is on — read fresh rather than
    remembered, because what the other holds changes as it starts and is reaped. That is what
    keeps the two on ONE card whichever of them is asked for first.

    The key is checked by the CALLER, before this is reached, so an unauthenticated request can
    never cause a model load — otherwise anyone on the box could take a GPU for 30 minutes by
    curling with no credentials at all.
    """
    other = images if target is engine else engine
    target._share_with = other.status()["gpu"]
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_HEADERS}
    # Our key is ours; the engine is not asked to verify it and must not see it.
    headers.pop("authorization", None)
    headers.pop("x-api-key", None)
    try:
        upstream = await target.forward(request.method, path, body, headers)
    except EngineBusy as exc:
        # 503 + Retry-After: the honest answer is "come back", not "something broke".
        return JSONResponse(
            {"error": {"message": str(exc), "type": "gpu_unavailable"}},
            status_code=503,
            headers={"Retry-After": "120"},
        )
    except Exception as exc:  # noqa: BLE001 - boundary: one clear message, never a traceback
        return JSONResponse(
            {"error": {"message": f"local engine failed to start: {exc}", "type": "engine_error"}},
            status_code=502,
        )
    passthrough = {
        k: v for k, v in upstream.headers.items() if k.lower() not in _SKIP_HEADERS
    }
    if "text/event-stream" in upstream.headers.get("content-type", ""):
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=passthrough,
            media_type=upstream.headers.get("content-type"),
        )
    return Response(
        content=upstream.content, status_code=upstream.status_code, headers=passthrough
    )
