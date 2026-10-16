"""The runtime settings a client may change: the idle timeout, its range, and its file."""

import json
import logging
import os
import stat

import pytest

from omnia_llm.server.settings import RuntimeSettings


def test_without_a_file_the_config_value_is_used(tmp_path):
    settings = RuntimeSettings(tmp_path / "settings.json", 30)
    assert settings.idle_timeout_minutes == 30
    assert settings.idle_timeout_seconds == 1800
    assert settings.to_dict() == {"idle_timeout_minutes": 30, "default_idle_timeout_minutes": 30}


def test_a_change_is_written_privately_and_read_back(tmp_path):
    path = tmp_path / "state" / "settings.json"
    RuntimeSettings(path, 30).update({"idle_timeout_minutes": 7.5})

    assert json.loads(path.read_text()) == {"idle_timeout_minutes": 7.5}
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert not path.with_suffix(".tmp").exists()
    assert RuntimeSettings(path, 30).idle_timeout_minutes == 7.5


@pytest.mark.parametrize("raw", ["{not json", '{"idle_timeout_minutes": "ten"}',
                                 '{"idle_timeout_minutes": 0}', "[5]", ""])
def test_a_corrupt_file_falls_back_to_the_config_value(tmp_path, caplog, raw):
    path = tmp_path / "settings.json"
    path.write_text(raw)
    with caplog.at_level(logging.WARNING):
        settings = RuntimeSettings(path, 30)
    assert settings.idle_timeout_minutes == 30
    assert "settings.json" in caplog.text


@pytest.mark.parametrize("body", [
    {"idle_timeout_minutes": 0.5}, {"idle_timeout_minutes": 1441}, {"idle_timeout_minutes": True},
    {"idle_timeout_minutes": "10"}, {"idle_timeout_minutes": None}, {},
    {"idle_timeout_minutes": 10, "reaper": 1}, [10], 10, {"idle_timeout_minutes": float("nan")},
    {"idle_timeout_minutes": float("inf")},
])
def test_a_bad_change_is_refused_and_changes_nothing(tmp_path, body):
    path = tmp_path / "settings.json"
    settings = RuntimeSettings(path, 30)
    with pytest.raises(ValueError, match="between 1 and 1440"):
        settings.update(body)
    assert settings.idle_timeout_minutes == 30 and not path.exists()


@pytest.mark.parametrize("minutes", [1, 1440, 2.25])
def test_the_range_ends_and_fractions_are_allowed(tmp_path, minutes):
    settings = RuntimeSettings(tmp_path / "settings.json", 30)
    settings.update({"idle_timeout_minutes": minutes})
    assert settings.idle_timeout_minutes == minutes
