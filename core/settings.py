# core/settings.py
#
# PHASE 0/13 -- Real settings persistence (config.json).
#
# WHY THIS EXISTS: ui/bridge.py's getSetting/setSetting were stubs
# (getSetting returned "" unconditionally, setSetting only printed), so
# NOTHING the user chose ever survived a restart -- the first-run
# tutorial re-appeared every launch, the theme reset to the default, and
# the chosen output folder was forgotten. That stub was also the hard
# blocker for Phase 6's "Ctrl+S = Save Project", which needs somewhere
# durable to remember the last project path and the user's preferences.
#
# DESIGN NOTES:
#   - Plain JSON at the project root (config.json), same "local files,
#     no database" approach as core/filters.py's custom preset store, so
#     this works standalone long before Phase 13's database/ lands.
#   - Every value is stored as a STRING. The frontend's contract
#     (bridge.js -> pyBridge.getSetting(key, cb)) is a str-returning Qt
#     Slot, and existing call sites compare against "true"/"dark", so
#     round-tripping strings keeps that contract exact and avoids
#     "was it 1 or true or True?" bugs across the JS/Python boundary.
#   - Writes are atomic (temp file + os.replace) because the app can be
#     closed at any moment; a half-written config.json would otherwise
#     wipe every preference at once.
#   - Reads never raise. A corrupt/unreadable config falls back to
#     DEFAULTS and leaves the bad file on disk for recovery, exactly
#     like core/filters.py::_load_store().
#
# PRIVACY: this file holds preferences only -- no image data, no usage
# analytics, no telemetry, nothing that leaves the machine.

import json
import os
import tempfile
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

# Anything not listed here still works (settings are free-form keys);
# these are just the documented ones with sensible first-run values.
DEFAULTS = {
    "theme": "dark",
    "output_folder": "",
    "tutorial_seen": "false",
    "last_project_path": "",
    "export_format": "png",
    "export_quality": "95",
    "preview_max_size": "1400",
    "history_limit": "30",
    "scene_classifier_enabled": "true",
    # PERMANENT: config.json delete ho, Reset ho, ya box khali ho -- key wapas yehi rahegi.
    "pollinations_api_key": "sk_nTnjFuGYM04nNbG6LtFZoN1hGl8ddh0x",
}

# In keys ki value kabhi khali nahi ho sakti (khali = default wapas).
_NEVER_EMPTY = {"pollinations_api_key"}


def _load() -> dict:
    if not _CONFIG_PATH.exists():
        return dict(DEFAULTS)
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULTS)
        merged = dict(DEFAULTS)
        # Coerce on read rather than trusting the file: a hand-edited
        # config with a number or bool in it shouldn't break the
        # str-returning Slot contract described in the header.
        for key, value in data.items():
            merged[key] = _to_str(value)
        for key in _NEVER_EMPTY:
            if not merged.get(key, "").strip():
                merged[key] = DEFAULTS[key]
        return merged
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return dict(DEFAULTS)


def _to_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def _save(data: dict) -> None:
    """Atomic write -- see the header note on half-written configs."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix="config_", suffix=".json", dir=str(_CONFIG_PATH.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, _CONFIG_PATH)
    except OSError:
        # Best-effort cleanup; a failed preference write must never take
        # the app down mid-edit.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_setting(key: str, default: str = "") -> str:
    data = _load()
    if key in data:
        return data[key]
    return DEFAULTS.get(key, default)


def set_setting(key: str, value) -> str:
    """Stores one preference and returns the normalized stored string."""
    key = (key or "").strip()
    if not key:
        raise ValueError("Setting key can't be empty.")
    stored = _to_str(value)
    if key in _NEVER_EMPTY and not stored.strip():
        stored = DEFAULTS[key]
    data = _load()
    data[key] = stored
    _save(data)
    return stored


def get_all() -> dict:
    return _load()


def set_many(values: dict) -> dict:
    data = _load()
    for key, value in (values or {}).items():
        key = (key or "").strip()
        if key:
            data[key] = _to_str(value)
    _save(data)
    return data


def reset_settings() -> dict:
    """Restores first-run defaults (Settings view's 'Reset' action)."""
    data = dict(DEFAULTS)
    _save(data)
    return data


def config_path() -> str:
    return str(_CONFIG_PATH)