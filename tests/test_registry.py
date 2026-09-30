import pytest

from omnia_llm.registry import Registry


class _A: ...
class _B: ...


def test_a_registered_class_is_found_by_name_and_knows_it():
    reg = Registry("thing")
    reg.register("a")(_A)
    assert reg.get("a") is _A and _A.registry_name == "a"


def test_an_unknown_name_says_what_is_known():
    reg = Registry("thing")
    reg.register("a")(_A)
    with pytest.raises(KeyError, match="known: a"):
        reg.get("nope")


def test_two_classes_cannot_share_a_name():
    reg = Registry("thing")
    reg.register("a")(_A)
    with pytest.raises(ValueError):
        reg.register("a")(_B)


def test_the_built_ins_are_all_registered():
    import omnia_llm.platforms  # noqa: F401
    from omnia_llm.devices import DEVICE_PROBES
    from omnia_llm.engines import ENGINES

    assert {"nvidia", "rocm", "apple", "cpu"} <= set(DEVICE_PROBES.names())
    assert {"vllm", "diffusers", "mlx", "llamacpp"} <= set(ENGINES.names())
