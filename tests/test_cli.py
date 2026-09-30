import functools
import signal
import subprocess
import sys
import time

import httpx

from omnia_llm import cli, config
from tests.helpers import free_port


def test_tokens_need_no_config_while_state_is_where_it_always_is(tmp_path, monkeypatch, capsys):
    """Every shipped config leaves state/ at its default, so asking for one is only friction."""
    monkeypatch.delenv("OMNIA_LLM_CONFIG", raising=False)
    monkeypatch.setattr(config, "PathsConfig",
                        functools.partial(config.PathsConfig, state=tmp_path / "state"))

    assert cli.main(["token", "issue", "laptop"]) == 0
    token = capsys.readouterr().out.strip()
    assert cli.main(["token", "list"]) == 0
    assert "laptop" in capsys.readouterr().out
    assert token and (tmp_path / "state" / "tokens.json").is_file()


def _models_config(tmp_path, **options):
    path = tmp_path / "models.toml"
    table = "\n".join(f"{key} = {value!r}" for key, value in options.items())
    path.write_text(f"""
[devices]
probe = "cpu"

[[models]]
id = "m"
kind = "text"
engine = "llamacpp"
port = 8742
[models.options]
{table}
""", encoding="utf-8")
    return path


def test_models_names_an_option_serve_would_refuse(tmp_path, capsys):
    """Passed here and refused by `serve`, a typo is a service in a restart loop."""
    config_file = _models_config(tmp_path, model="m.gguf", ctx_sise=4096)
    assert cli.main(["models", "-c", str(config_file)]) == 1
    assert "ctx_sise" in capsys.readouterr().err


def test_models_passes_a_config_serve_would_run(tmp_path, capsys):
    assert cli.main(["models", "-c", str(_models_config(tmp_path, model="m.gguf"))]) == 0
    assert "[ok]" in capsys.readouterr().out


def _gateway_config(tmp_path, port):
    path = tmp_path / "gateway.toml"
    path.write_text(f"""
[server]
port = {port}

[paths]
state = "{tmp_path / 'state'}"
model_cache = "{tmp_path / 'model-cache'}"

[devices]
probe = "cpu"

[[models]]
id = "m"
kind = "text"
engine = "llamacpp"
port = {port + 1 if port < 65535 else port - 1}
[models.options]
model = "m.gguf"
""", encoding="utf-8")
    return path


def _wait_for_health(gateway, port):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        assert gateway.poll() is None, gateway.stdout.read()
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise AssertionError("the gateway never answered /health")


def test_closing_the_terminal_shuts_the_gateway_down_gracefully(tmp_path):
    """The terminal's SIGHUP used to kill it on the spot, without the lifespan shutdown: the
    models, each in a session of its own, went on holding their devices."""
    port = free_port()
    gateway = subprocess.Popen(
        [sys.executable, "-m", "omnia_llm", "serve", "-c", str(_gateway_config(tmp_path, port))],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_for_health(gateway, port)
        gateway.send_signal(signal.SIGHUP)
        output, _ = gateway.communicate(timeout=30)
    finally:
        if gateway.poll() is None:
            gateway.kill()
            gateway.wait()
    assert "Application shutdown complete" in output
    assert gateway.returncode == 0
