from conftest import FakeProbe, gpu

from omnia_llm.config import AppConfig, ModelSpec, PathsConfig, ServerConfig, DevicesConfig
from omnia_llm.server.manager import ModelManager


def _manager(tmp_path, *models):
    cfg = AppConfig(ServerConfig(), PathsConfig(state=tmp_path), DevicesConfig(), tuple(models))
    return ModelManager(cfg, FakeProbe([gpu(7)]))


TEXT = ModelSpec("omnia-local", "text", "vllm", 8722, options={"model": "t"})
TEXT2 = ModelSpec("small", "text", "vllm", 8724, options={"model": "s"})
IMAGE = ModelSpec("sdxl-turbo", "image", "diffusers", 8723, options={"model": "i"})


def test_a_model_is_routed_by_its_id(tmp_path):
    m = _manager(tmp_path, TEXT, TEXT2, IMAGE)
    assert m.route("text", "small").id == "small"


def test_an_unknown_id_falls_back_to_the_first_of_its_kind(tmp_path):
    """An older client config ("sdxl" for "sdxl-turbo") keeps working."""
    m = _manager(tmp_path, TEXT, IMAGE)
    assert m.route("image", "sdxl").id == "sdxl-turbo"
    assert m.route("text", "").id == "omnia-local"


def test_an_id_of_the_wrong_kind_is_not_used_for_the_other(tmp_path):
    m = _manager(tmp_path, TEXT, IMAGE)
    assert m.route("text", "sdxl-turbo").id == "omnia-local"


def test_no_model_of_a_kind_is_none(tmp_path):
    assert _manager(tmp_path, TEXT).route("image", "x") is None


def test_the_listing_carries_each_kind_and_starts_nothing(tmp_path):
    m = _manager(tmp_path, TEXT, IMAGE)
    assert [(e["id"], e["kind"]) for e in m.listing()] == [("omnia-local", "text"),
                                                           ("sdxl-turbo", "image")]
    assert not any(e.running for e in m.engines.values())
