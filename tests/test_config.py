import pytest

from omnia_llm.config import ROOT, AppConfig, ConfigError


def _cfg(**overrides):
    data = {"server": {"port": 8721},
            "models": [{"id": "m", "kind": "text", "engine": "vllm", "port": 8722,
                        "options": {"model": "x"}}]}
    data.update(overrides)
    return data


def test_every_shipped_config_loads():
    configs = sorted((ROOT / "configs").glob("*.toml"))
    assert configs
    for path in configs:
        assert AppConfig.load(path).models


def test_downloaded_models_stay_inside_the_repo_by_default():
    paths = AppConfig.from_dict(_cfg()).paths
    assert paths.model_cache == ROOT / ".model-cache"
    assert paths.llama_cache.parent == paths.vllm_cache.parent == ROOT / ".model-cache"


def test_hugging_face_downloads_stay_where_the_last_layout_put_them():
    """HF_HOME used to be <model_cache>/huggingface, so its hub is where the weights already are."""
    paths = AppConfig.from_dict(_cfg()).paths
    assert paths.hf_hub_cache == ROOT / ".model-cache" / "huggingface" / "hub"


def test_a_relative_model_cache_resolves_against_the_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = AppConfig.from_dict(_cfg(paths={"model_cache": "weights"})).paths
    assert paths.model_cache == ROOT / "weights"


def test_relative_state_resolves_against_the_repo_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert AppConfig.from_dict(_cfg(paths={"state": "state"})).paths.state == ROOT / "state"


@pytest.mark.parametrize("bad, message", [
    ({"serverr": {}}, "unknown section"),
    ({"server": {"prot": 1}}, "unknown key"),
    ({"models": []}, "nothing to serve"),
    ({"models": [{"id": "m", "kind": "audio", "engine": "vllm", "port": 1}]}, "kind"),
    ({"models": [{"id": "m", "kind": "text", "engine": "vllm"}]}, "missing 'port'"),
    ({"models": [{"id": "m", "kind": "text", "engine": "e", "port": 9},
                 {"id": "m", "kind": "text", "engine": "e", "port": 10}]}, "two models"),
    ({"models": [{"id": "m", "kind": "text", "engine": "e", "port": 8721}]}, "already in use"),
])
def test_a_config_that_cannot_run_is_refused_at_startup(bad, message):
    with pytest.raises(ConfigError, match=message):
        AppConfig.from_dict(_cfg(**bad))


def test_models_of_a_kind():
    cfg = AppConfig.from_dict(_cfg(models=[
        {"id": "t", "kind": "text", "engine": "vllm", "port": 1},
        {"id": "i", "kind": "image", "engine": "diffusers", "port": 2}]))
    assert [m.id for m in cfg.models_of("image")] == ["i"]
