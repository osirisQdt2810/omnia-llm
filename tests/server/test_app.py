"""The HTTP gateway: what is open, what needs a token, and the lockout."""

import pytest
from fastapi.testclient import TestClient

from omnia_llm.config import AppConfig, DevicesConfig, ModelSpec, PathsConfig, ServerConfig
from omnia_llm.server.app import create_app
from omnia_llm.server.lockout import MAX_FAILURES
from tests.helpers import fake_probe, gpu


@pytest.fixture
def client(tmp_path):
    cfg = AppConfig(ServerConfig(), PathsConfig(state=tmp_path / "state"), DevicesConfig(),
                    (ModelSpec("omnia-local", "text", "vllm", 8722, options={"model": "t"}),))
    app = create_app(cfg, fake_probe([gpu(7)]))
    token = app.state.tokens.issue("tester")
    return TestClient(app), token


def test_health_needs_no_token(client):
    c, _ = client
    assert c.get("/health").json() == {"ok": True}


@pytest.mark.parametrize("route", ["/status", "/v1/models"])
def test_everything_else_needs_one(client, route):
    c, token = client
    assert c.get(route).status_code == 401
    assert c.get(route, headers={"Authorization": f"Bearer {token}"}).status_code == 200


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


def test_another_client_is_not_locked_out_by_one_guesser(client):
    c, token = client
    for _ in range(MAX_FAILURES):
        c.get("/status", headers=[("Authorization", "Bearer wrong"), ("X-Forwarded-For", "1.1.1.1")])
    ok = c.get("/status", headers=[("Authorization", f"Bearer {token}"),
                                   ("X-Forwarded-For", "2.2.2.2")])
    assert ok.status_code == 200


def test_a_gateway_going_away_gives_its_devices_back(tmp_path):
    cfg = AppConfig(ServerConfig(), PathsConfig(state=tmp_path / "state"), DevicesConfig(),
                    (ModelSpec("omnia-local", "text", "vllm", 8722, options={"model": "t"}),))
    app = create_app(cfg, fake_probe([gpu(7)]))
    stopped = []

    async def stop(reason):
        stopped.append(reason)

    app.state.manager.engines["omnia-local"].stop = stop
    with TestClient(app):
        assert app.state.manager._reaper is not None and not app.state.manager._reaper.done()
    assert stopped == ["gateway shutting down"]
