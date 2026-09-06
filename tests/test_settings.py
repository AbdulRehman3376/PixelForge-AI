# tests/test_settings.py -- PHASE 14 unit test coverage: settings persistence

import json


def test_defaults_returned_when_no_config_file_exists(isolated_settings):
    assert isolated_settings.get_setting("theme") == "dark"
    assert isolated_settings.get_setting("tutorial_seen") == "false"


def test_set_and_get_roundtrip(isolated_settings):
    isolated_settings.set_setting("theme", "light")
    assert isolated_settings.get_setting("theme") == "light"


def test_set_setting_coerces_bool_to_string(isolated_settings):
    stored = isolated_settings.set_setting("scene_classifier_enabled", False)
    assert stored == "false"
    assert isolated_settings.get_setting("scene_classifier_enabled") == "false"


def test_set_setting_rejects_empty_key(isolated_settings):
    import pytest
    with pytest.raises(ValueError):
        isolated_settings.set_setting("", "x")


def test_set_many_and_get_all(isolated_settings):
    isolated_settings.set_many({"theme": "light", "export_format": "jpg"})
    data = isolated_settings.get_all()
    assert data["theme"] == "light"
    assert data["export_format"] == "jpg"


def test_reset_settings_restores_defaults(isolated_settings):
    isolated_settings.set_setting("theme", "light")
    isolated_settings.reset_settings()
    assert isolated_settings.get_setting("theme") == "dark"


def test_corrupted_config_falls_back_to_defaults(isolated_settings, tmp_path):
    """Config/settings validation (Phase 14 requirement): a corrupted
    config.json must never crash the app -- it should fall back to
    defaults instead."""
    isolated_settings._CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    isolated_settings._CONFIG_PATH.write_text("{not valid json!!", encoding="utf-8")
    assert isolated_settings.get_setting("theme") == "dark"


def test_config_holding_a_list_instead_of_dict_falls_back(isolated_settings):
    isolated_settings._CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    isolated_settings._CONFIG_PATH.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert isolated_settings.get_setting("theme") == "dark"


def test_write_is_atomic_no_leftover_temp_files(isolated_settings):
    isolated_settings.set_setting("theme", "light")
    leftovers = list(isolated_settings._CONFIG_PATH.parent.glob("config_*"))
    assert leftovers == []


def test_config_path_returns_a_string(isolated_settings):
    assert isinstance(isolated_settings.config_path(), str)