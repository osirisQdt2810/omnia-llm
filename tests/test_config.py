import pytest

import omnia_llm.platforms  # noqa: F401 - registers MLX, which a shipped config uses
from omnia_llm.config import ROOT, AppConfig, ConfigError
from omnia_llm.engines import ENGINES


def _cfg(**overrides):
    data = {"server": {"port": 8721},
            "models": [{"id": "m", "kind": "text", "engine": "vllm", "port": 8722,
                        "options": {"model": "x"}}]}
    data.update(overrides)
    return data


def test_every_shipped_config_loads():
    """Options included, as each model's engine reads them: `serve` refuses what they refuse."""
    configs = sorted((ROOT / "configs").glob("*.toml"))
    assert configs
    for path in configs:
        models = AppConfig.load(path).models
        assert models
        for spec in models:
            ENGINES.get(spec.engine).check_options(spec)


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


def _model(**fields):
    return {"id": "m", "kind": "text", "engine": "vllm", "port": 8722, **fields}


@pytest.mark.parametrize("bad", ["Qwen/Qwen2.5-1.5B", "", ".hidden", "-x", "a b", "a" * 65, 7])
def test_an_id_that_cannot_name_a_log_file_is_refused(bad):
    """The id names the model's log file: "Qwen/Qwen2.5-1.5B" loaded, and every start failed."""
    with pytest.raises(ConfigError, match="id"):
        AppConfig.from_dict(_cfg(models=[_model(id=bad)]))


@pytest.mark.parametrize("good", ["qwen2.5-14b-instruct-awq", "qwen2.5-1.5b-instruct-q4_k_m",
                                  "M", "a" * 64])
def test_an_id_made_of_file_name_characters_is_accepted(good):
    assert AppConfig.from_dict(_cfg(models=[_model(id=good)])).models[0].id == good


@pytest.mark.parametrize("server", [
    {"port": 0}, {"port": 65536}, {"port": "8721"}, {"port": True},
    {"idle_timeout_minutes": 0}, {"idle_timeout_minutes": -5}, {"idle_timeout_minutes": "30"},
    {"reaper_interval_seconds": 0}, {"reaper_interval_seconds": False},
    {"cpu_busy_load": 0}, {"cpu_busy_load": "0.7"}, {"cpu_busy_load": True},
])
def test_a_server_setting_out_of_range_is_refused(server):
    with pytest.raises(ConfigError, match=next(iter(server))):
        AppConfig.from_dict(_cfg(server=server))


@pytest.mark.parametrize("port", [0, 65536, -1, "8722", 8722.5, True])
def test_a_model_port_out_of_range_is_refused(port):
    with pytest.raises(ConfigError, match="port"):
        AppConfig.from_dict(_cfg(models=[_model(port=port)]))


@pytest.mark.parametrize("required", ["14500", -1, 1.5, True])
def test_the_memory_a_model_needs_is_a_whole_number(required):
    """Accepted as "14500", it loaded, and then every start failed comparing it with a number."""
    with pytest.raises(ConfigError, match="required_mib"):
        AppConfig.from_dict(_cfg(models=[_model(required_mib=required)]))


def test_a_model_may_need_no_memory_at_all():
    assert AppConfig.from_dict(_cfg(models=[_model(required_mib=0)])).models[0].required_mib == 0


def test_fractional_minutes_are_fine():
    cfg = AppConfig.from_dict(_cfg(server={"port": 8721, "idle_timeout_minutes": 0.5}))
    assert cfg.server.idle_timeout_seconds == 30.0


def test_a_config_without_the_cpu_and_single_device_keys_keeps_todays_behaviour():
    cfg = AppConfig.from_dict(_cfg())
    assert cfg.devices.single_device is False and cfg.server.cpu_busy_load == 0.7
    model = cfg.models[0]
    assert model.when_cpu_busy is None and model.profile(cpu_busy=True) is model


@pytest.mark.parametrize("value", ["yes", 1, 0])
def test_single_device_is_true_or_false(value):
    with pytest.raises(ConfigError, match="single_device must be true or false"):
        AppConfig.from_dict(_cfg(devices={"single_device": value}))


def test_single_device_can_be_turned_on():
    assert AppConfig.from_dict(_cfg(devices={"single_device": True})).devices.single_device


def _busy(**override):
    return _model(required_mib=15500,
                  options={"model": "x", "extra_args": ["--n-cpu-moe", "8", "-np", "2"],
                           "max_model_len": 8192},
                  when_cpu_busy=override)


def test_the_busy_cpu_settings_are_laid_over_the_models_own():
    """Shallowly: a list it names replaces the model's whole, an option it does not name stays."""
    model = AppConfig.from_dict(_cfg(models=[
        _busy(required_mib=19000, options={"extra_args": ["-np", "2"]})])).models[0]
    busy = model.profile(cpu_busy=True)
    assert (busy.required_mib, busy.options) == (
        19000, {"model": "x", "extra_args": ["-np", "2"], "max_model_len": 8192})
    assert model.profile(cpu_busy=False) is model
    assert model.options["extra_args"] == ["--n-cpu-moe", "8", "-np", "2"]


def test_busy_cpu_settings_without_required_mib_keep_the_models_own():
    model = AppConfig.from_dict(_cfg(models=[_busy(options={"max_model_len": 4096})])).models[0]
    assert model.profile(cpu_busy=True).required_mib == 15500


@pytest.mark.parametrize("override, message", [
    ({"required_mib": 19000, "extra_args": []}, "unknown key"),
    ({"required_mib": "19000"}, "required_mib"),
    ({"required_mib": -1}, "required_mib"),
    ({"options": ["-np", "2"]}, "options must be a table"),
])
def test_busy_cpu_settings_that_cannot_run_are_refused(override, message):
    with pytest.raises(ConfigError, match=f"when_cpu_busy.*{message}"):
        AppConfig.from_dict(_cfg(models=[_busy(**override)]))


def test_busy_cpu_settings_must_be_a_table():
    with pytest.raises(ConfigError, match="when_cpu_busy: must be a table"):
        AppConfig.from_dict(_cfg(models=[_model(when_cpu_busy=19000)]))


def test_models_of_a_kind():
    cfg = AppConfig.from_dict(_cfg(models=[
        {"id": "t", "kind": "text", "engine": "vllm", "port": 1},
        {"id": "i", "kind": "image", "engine": "diffusers", "port": 2}]))
    assert [m.id for m in cfg.models_of("image")] == ["i"]
