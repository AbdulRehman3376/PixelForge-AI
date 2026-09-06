# tests/conftest.py
#
# PHASE 14 -- shared pytest fixtures.
#
# IMPORTANT ISOLATION NOTE: several core/ modules (settings.py,
# filters.py, database.py) read/write real files next to the project
# root (config.json, presets/custom_presets.json, database/pixelforge.db)
# using module-level path constants computed at import time. Tests must
# NEVER touch the user's real data files, so every fixture below that
# exercises one of those modules monkeypatches its path constant to a
# pytest tmp_path first. This is why individual test files import the
# *module* (e.g. `from core import settings`) rather than only its
# functions -- monkeypatch needs the module object's attribute to patch.

import sys
from pathlib import Path

import pytest
from PIL import Image

# Make the project root importable when pytest is run from anywhere
# (e.g. `pytest` from the project root, or `pytest tests/`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_image():
    """A small, fast, deterministic RGB test image (no file I/O)."""
    img = Image.new("RGB", (64, 48), color=(120, 140, 160))
    # A bit of structure so sharpness/contrast/color reads aren't all
    # flat-zero -- a simple gradient plus a bright/dark corner block.
    pixels = img.load()
    for x in range(64):
        for y in range(48):
            pixels[x, y] = (min(255, x * 3), min(255, y * 4), 128)
    return img


@pytest.fixture
def sample_image_path(tmp_path, sample_image):
    """Same image, saved to a temp file -- for functions that need a path."""
    path = tmp_path / "sample.png"
    sample_image.save(path)
    return str(path)


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """core.settings, redirected to a temp config.json for this test only."""
    from core import settings
    monkeypatch.setattr(settings, "_CONFIG_PATH", tmp_path / "config.json")
    return settings


@pytest.fixture
def isolated_filters(tmp_path, monkeypatch):
    """core.filters, redirected to a temp custom-presets store for this test only."""
    from core import filters
    monkeypatch.setattr(filters, "_STORE_PATH", tmp_path / "presets" / "custom_presets.json")
    return filters


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    """
    core.database, redirected to a temp SQLite file for this test only.
    The module keeps a single cached connection in a module-level
    `_conn` global (see core/database.py::_get_conn), so this also
    resets that global before/after the test to guarantee a fresh
    connection against the temp path rather than reusing a connection
    opened by an earlier test against the real database.
    """
    from core import database as db
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "pixelforge_test.db")
    monkeypatch.setattr(db, "_conn", None)
    yield db
    try:
        db.close()
    except Exception:
        pass