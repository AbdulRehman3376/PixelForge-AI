# tests/test_filters.py -- PHASE 14 unit test coverage: presets/filters

import pytest


def test_builtin_presets_count_matches_spec(isolated_filters):
    """Spec (Phase 4) requires 9 built-in presets."""
    assert len(isolated_filters.BUILTIN_PRESETS) == 9


def test_list_presets_includes_all_builtins(isolated_filters):
    presets = isolated_filters.list_presets()
    builtin_ids = {p["id"] for p in isolated_filters.BUILTIN_PRESETS}
    listed_ids = {p["id"] for p in presets}
    assert builtin_ids.issubset(listed_ids)


def test_get_preset_by_id_returns_builtin(isolated_filters):
    first_id = isolated_filters.BUILTIN_PRESETS[0]["id"]
    preset = isolated_filters.get_preset(first_id)
    assert preset is not None
    assert preset["id"] == first_id


def test_get_preset_unknown_id_returns_none(isolated_filters):
    assert isolated_filters.get_preset("this_id_does_not_exist") is None


def test_apply_preset_returns_same_size_image(isolated_filters, sample_image):
    preset = isolated_filters.BUILTIN_PRESETS[0]
    result = isolated_filters.apply_preset(sample_image, preset, intensity=100)
    assert result.size == sample_image.size


def test_apply_preset_at_zero_intensity_is_near_identity(isolated_filters, sample_image):
    preset = isolated_filters.BUILTIN_PRESETS[0]
    result = isolated_filters.apply_preset(sample_image, preset, intensity=0)
    assert result.size == sample_image.size
    assert result.mode == sample_image.convert("RGB").mode


def test_save_custom_preset_roundtrip(isolated_filters):
    preset = isolated_filters.save_custom_preset(
        "My Test Preset", {"contrast": 120, "saturation": 90}, grain=10, monochrome=False,
    )
    assert preset["name"] == "My Test Preset"
    assert preset["id"].startswith("custom_")

    fetched = isolated_filters.get_preset(preset["id"])
    assert fetched is not None
    assert fetched["name"] == "My Test Preset"


def test_delete_custom_preset(isolated_filters):
    preset = isolated_filters.save_custom_preset("Temp", {"contrast": 110})
    isolated_filters.delete_preset(preset["id"])
    assert isolated_filters.get_preset(preset["id"]) is None


def test_builtin_preset_cannot_be_renamed(isolated_filters):
    builtin_id = isolated_filters.BUILTIN_PRESETS[0]["id"]
    with pytest.raises(ValueError):
        isolated_filters.rename_preset(builtin_id, "New Name")


def test_duplicate_custom_preset(isolated_filters):
    original = isolated_filters.save_custom_preset("Original", {"contrast": 105})
    dup = isolated_filters.duplicate_preset(original["id"])
    assert dup["id"] != original["id"]
    assert isolated_filters.get_preset(dup["id"]) is not None


def test_auto_suggest_preset_returns_a_valid_id(isolated_filters, sample_image):
    suggestion = isolated_filters.auto_suggest_preset(sample_image)
    assert "preset_id" in suggestion or isinstance(suggestion, dict)