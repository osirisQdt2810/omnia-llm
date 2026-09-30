import functools

from omnia_llm import cli, config


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
