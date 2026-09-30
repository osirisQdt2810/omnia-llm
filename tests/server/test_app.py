"""The HTTP gateway: what is open, what needs a token, the lockout, and the /v1 paths."""

import socket
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from omnia_llm.config import AppConfig, DevicesConfig, ModelSpec, PathsConfig, ServerConfig
from omnia_llm.server.app import create_app
from omnia_llm.server.lockout import MAX_FAILURES
from tests.helpers import fake_probe, gpu

TEXT = ModelSpec("omnia-local", "text", "vllm", 8722, options={"model": "t"})
IMAGE = ModelSpec("sdxl-turbo", "image", "diffusers", 8723, options={"model": "i"})


@pytest.fixture
def client(tmp_path):
    cfg = AppConfig(ServerConfig(), PathsConfig(state=tmp_path / "state"), DevicesConfig(),
                    (TEXT,))
    app = create_app(cfg, fake_probe([gpu(7)]))
    token = app.state.tokens.issue("tester")
    return TestClient(app), token


class _Gateway:
    """A text and an image model whose engines only record: every start, every request."""

    def __init__(self, tmp_path):
        cfg = AppConfig(ServerConfig(), PathsConfig(state=tmp_path / "state"), DevicesConfig(),
                        (TEXT, IMAGE))
        self.app = create_app(cfg, fake_probe([gpu(7)]))
        self.token = self.app.state.tokens.issue("tester")
        self.http = TestClient(self.app)
        self.started: list[str] = []
        self.forwarded: list[httpx.Request] = []
        for engine in self.app.state.manager.engines.values():
            engine.ensure_running = self._start(engine.id)
            self.answer(engine.id, lambda request: httpx.Response(200, json={"ok": True}))

    def _start(self, model_id):
        async def ensure_running(prefer=()):
            self.started.append(model_id)
        return ensure_running

    def answer(self, model_id, respond):
        """What the engine serving ``model_id`` answers, in place of a real one."""
        def upstream(request):
            self.forwarded.append(request)
            return respond(request)
        engine = self.app.state.manager.engines[model_id]
        engine._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))

    def request(self, method, path, **kwargs):
        headers = {"Authorization": f"Bearer {self.token}", **kwargs.pop("headers", {})}
        return self.http.request(method, path, headers=headers, **kwargs)


@pytest.fixture
def gateway(tmp_path):
    return _Gateway(tmp_path)


def test_health_needs_no_token(client):
    c, _ = client
    assert c.get("/health").json() == {"ok": True}


@pytest.mark.parametrize("route", ["/status", "/v1/models"])
def test_everything_else_needs_one(client, route):
    c, token = client
    assert c.get(route).status_code == 401
    assert c.get(route, headers={"Authorization": f"Bearer {token}"}).status_code == 200


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@pytest.mark.parametrize("path", ["/status", "/warm", "/stop", "/v1/models",
                                  "/v1/chat/completions", "/v1/images/generations",
                                  "/openapi.json", "/docs"])
def test_no_path_but_health_answers_without_a_token(gateway, path, method):
    """Refused before routing, whatever the method: a route cannot forget the check, and a
    caller without a token learns nothing, not even which methods a path takes."""
    r = gateway.http.request(method, path, content=b'{"model": "omnia-local"}')
    assert r.status_code == 401
    assert gateway.started == [] and gateway.forwarded == []


def test_the_model_listing_has_kinds(client):
    c, token = client
    data = c.get("/v1/models", headers={"Authorization": f"Bearer {token}"}).json()["data"]
    assert data == [{"id": "omnia-local", "object": "model", "owned_by": "omnia-llm",
                     "kind": "text"}]


def test_no_image_model_is_an_honest_501(client):
    c, token = client
    r = c.post("/v1/images/generations", json={"prompt": "x"},
               headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 501


@pytest.mark.parametrize("header", ["Authorization", "X-API-Key"])
def test_the_clients_token_never_reaches_the_engine(gateway, header):
    """Our token is ours: the engine is not asked to verify it and must not see it."""
    value = f"Bearer {gateway.token}" if header == "Authorization" else gateway.token
    r = gateway.http.post("/v1/chat/completions", json={"model": "omnia-local"},
                          headers={header: value, "X-Trace": "kept"})
    assert r.status_code == 200
    sent = gateway.forwarded[0].headers
    assert "authorization" not in sent and "x-api-key" not in sent and sent["x-trace"] == "kept"


@pytest.mark.parametrize("path", ["/v1/models", "/v1//models", "/v1/models/", "/v1//models//"])
def test_the_listing_answers_from_config_however_it_is_slashed(gateway, path):
    """Omnia asks for ``<base>/models``; a Base URL typed with a trailing slash makes that
    ``//models``, which must list rather than cold-start a model to be told 404."""
    r = gateway.request("GET", path)
    assert r.status_code == 200
    assert [m["id"] for m in r.json()["data"]] == ["omnia-local", "sdxl-turbo"]
    assert gateway.started == [] and gateway.forwarded == []


def test_one_model_is_described_from_config(gateway):
    r = gateway.request("GET", "/v1/models/sdxl-turbo")
    assert r.json() == {"id": "sdxl-turbo", "object": "model", "owned_by": "omnia-llm",
                        "kind": "image"}
    assert gateway.started == []


def test_an_unknown_model_is_a_404_not_a_cold_start(gateway):
    r = gateway.request("GET", "/v1/models/nothing-here")
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found"
    assert gateway.started == [] and gateway.forwarded == []


@pytest.mark.parametrize("method, path", [
    ("GET", "/v1/"), ("GET", "/v1//"), ("POST", "/v1/"),
    ("GET", "/v1/images"), ("GET", "/v1/images/edits"), ("POST", "/v1/images/edits"),
    ("POST", "/v1/images/variations"), ("POST", "/v1/images/generations/extra"),
])
def test_a_path_nothing_here_serves_is_a_404_not_a_cold_start(gateway, method, path):
    """Sent on, the text model would answer these with a 404 of its own, after a cold start."""
    r = gateway.request(method, path, json={"prompt": "a lighthouse"})
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found"
    assert gateway.started == [] and gateway.forwarded == []


@pytest.mark.parametrize("path", ["/v1/%2e%2e/metrics", "/v1/models/..%2fmetrics",
                                  "/v1/chat/%2e/completions"])
def test_a_path_with_dot_segments_is_refused_not_forwarded(gateway, path):
    """Resolved on the way out, ``/v1/../metrics`` is the engine's own ``/metrics``."""
    r = gateway.request("GET", path)
    assert r.status_code == 400
    assert gateway.started == [] and gateway.forwarded == []


@pytest.mark.parametrize("path, model, upstream", [
    ("/v1//chat//completions", "omnia-local", "/v1/chat/completions"),
    ("/v1//images/generations/", "sdxl-turbo", "/v1/images/generations"),
])
def test_repeated_slashes_collapse_before_a_request_is_routed(gateway, path, model, upstream):
    r = gateway.request("POST", path, json={"prompt": "a lighthouse"})
    assert r.status_code == 200
    assert gateway.started == [model] and gateway.forwarded[0].url.path == upstream


def test_an_engine_that_does_not_answer_is_not_called_a_failed_start(gateway):
    """httpx's timeouts carry no message; the type is then the only clue there is."""
    def times_out(request):
        raise httpx.ReadTimeout("")
    gateway.answer("omnia-local", times_out)
    r = gateway.request("POST", "/v1/chat/completions", json={})
    assert r.status_code == 502
    assert r.json()["error"]["message"] == "the local engine did not answer: ReadTimeout"


def test_a_start_that_fails_is_reported_as_one(gateway):
    async def fails(prefer=()):
        raise RuntimeError("")
    gateway.app.state.manager.engines["omnia-local"].ensure_running = fails
    r = gateway.request("POST", "/v1/chat/completions", json={})
    assert r.status_code == 502
    assert r.json()["error"]["message"] == "local engine failed to start: RuntimeError"


def test_a_port_in_use_is_a_502_that_says_so_and_holds_no_device(tmp_path):
    with socket.socket() as squatter:
        squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        port = squatter.getsockname()[1]
        cfg = AppConfig(ServerConfig(), PathsConfig(state=tmp_path / "state"), DevicesConfig(),
                        (ModelSpec("omnia-local", "text", "vllm", port, options={"model": "t"}),))
        app = create_app(cfg, fake_probe([gpu(7)]))
        engine = app.state.manager.engines["omnia-local"]
        engine.command = lambda device: [sys.executable, "-c", "import time; time.sleep(60)"]
        engine.startup_timeout_seconds = 1
        token = app.state.tokens.issue("tester")
        r = TestClient(app).post("/v1/chat/completions", json={},
                                 headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 502
    assert f"port {port} is in use (a leftover engine?)" in r.json()["error"]["message"]
    assert app.state.manager.status()["omnia-local"]["device"] is None


def _config_file(tmp_path):
    path = tmp_path / "gateway.toml"
    path.write_text(f"""
[paths]
state = "{tmp_path / 'state'}"
model_cache = "{tmp_path / 'model-cache'}"

[[models]]
id = "omnia-local"
kind = "text"
engine = "vllm"
port = 8722
[models.options]
model = "t"
""", encoding="utf-8")
    return path


def _restart(config_file):
    return TestClient(create_app(AppConfig.load(config_file), fake_probe([gpu(7)])))


def test_a_new_install_accepts_no_token_it_did_not_issue(tmp_path):
    """No key is generated behind the operator's back: the only way in is `token issue`."""
    c = _restart(_config_file(tmp_path))
    assert not (tmp_path / "state" / "api-key.txt").exists()
    assert c.app.state.tokens.names() == []


def test_the_first_versions_key_keeps_working(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "api-key.txt").write_text("omnia-oldkey\n")
    c = _restart(_config_file(tmp_path))
    assert c.get("/status", headers={"Authorization": "Bearer omnia-oldkey"}).status_code == 200


def test_revoking_the_first_versions_key_survives_a_restart(tmp_path):
    """The gateway adopts the old key file at every start, so revoking has to retire the file,
    or the next restart would quietly let the key back in."""
    from omnia_llm import cli

    config_file = _config_file(tmp_path)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "api-key.txt").write_text("omnia-oldkey\n")
    _restart(config_file)  # adopted as "legacy"

    assert cli.main(["token", "revoke", "legacy", "-c", str(config_file)]) == 0
    after = _restart(config_file)
    assert after.get("/status", headers={"Authorization": "Bearer omnia-oldkey"}).status_code == 401


def test_a_guesser_is_locked_out_even_when_spoofing_forwarded_for(client):
    """ngrok forwards the client's X-Forwarded-For untouched and adds its own as a SECOND
    header; keying on the first would let each guess pretend to be someone new."""
    c, token = client
    codes = [c.get("/status", headers=[("Authorization", "Bearer wrong"),
                                       ("X-Forwarded-For", f"10.0.0.{i}"),
                                       ("X-Forwarded-For", "58.0.0.1")]).status_code
             for i in range(MAX_FAILURES + 1)]
    assert codes[-1] == 429 and set(codes[:MAX_FAILURES]) == {401}
    blocked = c.get("/status", headers=[("Authorization", f"Bearer {token}"),
                                        ("X-Forwarded-For", "58.0.0.1")])
    assert blocked.status_code == 429


def test_requests_without_a_token_never_clear_a_guessers_count(client):
    """Judged by status codes, a path answering below 400 without a token (FastAPI's docs, the
    307 for "/health/") counted as a success, so interleaving one made guessing free."""
    c, _ = client
    wrong = {"Authorization": "Bearer wrong"}
    for _ in range(MAX_FAILURES):
        c.get("/status", headers=wrong)
        c.get("/openapi.json")
        c.get("/health/", follow_redirects=False)
    assert c.get("/status", headers=wrong).status_code == 429


def test_health_never_clears_a_guessers_count(client):
    """Open, so no success either: interleaved with the guesses, it must not reset them."""
    c, _ = client
    wrong = {"Authorization": "Bearer wrong"}
    for _ in range(MAX_FAILURES):
        c.get("/status", headers=wrong)
        c.get("/health")
    assert c.get("/status", headers=wrong).status_code == 429


def test_health_counts_for_nothing_in_the_lockout(client):
    """Open, so neither a failure nor a success, whatever token it carries."""
    c, token = client
    for _ in range(MAX_FAILURES * 2):
        c.get("/health", headers={"Authorization": "Bearer wrong"})
    assert c.get("/status", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_another_client_is_not_locked_out_by_one_guesser(client):
    c, token = client
    for _ in range(MAX_FAILURES):
        c.get("/status", headers=[("Authorization", "Bearer wrong"), ("X-Forwarded-For", "1.1.1.1")])
    ok = c.get("/status", headers=[("Authorization", f"Bearer {token}"),
                                   ("X-Forwarded-For", "2.2.2.2")])
    assert ok.status_code == 200


def test_a_gateway_going_away_gives_its_devices_back(tmp_path):
    cfg = AppConfig(ServerConfig(), PathsConfig(state=tmp_path / "state"), DevicesConfig(),
                    (TEXT,))
    app = create_app(cfg, fake_probe([gpu(7)]))
    stopped = []

    async def stop(reason):
        stopped.append(reason)

    app.state.manager.engines["omnia-local"].stop = stop
    with TestClient(app):
        assert app.state.manager._reaper is not None and not app.state.manager._reaper.done()
    assert stopped == ["gateway shutting down"]
