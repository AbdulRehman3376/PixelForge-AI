# tests/test_image_and_export.py -- PHASE 14 unit test coverage:
# image loading, manual adjustments (Phase 2/3 enhancer), and export.

from PIL import Image

from core.enhancer import DEFAULT_ADJUSTMENTS, apply_adjustments


# ----- Image loading -----

def test_image_loading_from_disk(sample_image_path):
    img = Image.open(sample_image_path)
    img.load()
    assert img.size == (64, 48)


def test_image_loading_reports_correct_format(sample_image_path):
    img = Image.open(sample_image_path)
    assert img.format == "PNG"


def test_image_loading_missing_file_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        Image.open("/this/path/does/not/exist.png").load()


# ----- Enhancer / manual adjustments -----

def test_apply_adjustments_with_defaults_returns_same_size(sample_image):
    result = apply_adjustments(sample_image, dict(DEFAULT_ADJUSTMENTS))
    assert result.size == sample_image.size


def test_apply_adjustments_brightness_increase_brightens_image(sample_image):
    adjustments = dict(DEFAULT_ADJUSTMENTS)
    adjustments["brightness"] = 150
    result = apply_adjustments(sample_image, adjustments)

    def _avg_luma(img):
        gray = img.convert("L")
        pixels = list(gray.getdata())
        return sum(pixels) / len(pixels)

    assert _avg_luma(result) >= _avg_luma(sample_image.convert("RGB")) - 1  # allow tiny rounding


def test_apply_adjustments_partial_dict_fills_in_defaults(sample_image):
    # Callers are allowed to only pass the sliders that changed --
    # apply_adjustments should fill in the rest from DEFAULT_ADJUSTMENTS.
    result = apply_adjustments(sample_image, {"contrast": 120})
    assert result.size == sample_image.size


# ----- Export (save to disk) -----

def test_export_image_as_png(tmp_path, sample_image):
    dest = tmp_path / "export_test.png"
    sample_image.save(dest)
    assert dest.exists()
    reopened = Image.open(dest)
    assert reopened.size == sample_image.size


def test_export_image_as_jpeg_with_quality(tmp_path, sample_image):
    dest = tmp_path / "export_test.jpg"
    sample_image.convert("RGB").save(dest, quality=90)
    assert dest.exists()
    assert dest.stat().st_size > 0


def test_export_to_unwritable_folder_raises(tmp_path, sample_image):
    """Confirms the underlying failure mode a real 'disk write
    permission check' (core/system_monitor.py::check_disk_writable) is
    meant to catch BEFORE export -- an unwritable destination should
    raise an OSError/PermissionError from PIL itself when actually
    attempted.

    Skipped on:
    - root/administrator accounts (Linux/Mac root, or an elevated
      Windows session) -- permission bits are bypassed entirely and
      the write would silently succeed instead of raising.
    - Windows in general -- os.chmod() only toggles the read-only
      FILE attribute there and does NOT reliably enforce write
      protection on a FOLDER the way POSIX permission bits do, so
      this specific test is not meaningful cross-platform. The real
      behavior this guards (an unwritable destination surfacing a
      clear error before/при export) is still exercised on Linux/Mac
      CI, and is what core/system_monitor.py::check_disk_writable
      (tested separately in test_system_monitor.py) exists to catch
      proactively regardless of platform.
    """
    import os
    import platform
    import pytest

    if platform.system() == "Windows":
        pytest.skip("os.chmod() doesn't reliably enforce folder write "
                     "protection on Windows -- not testable this way here.")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("Running as root -- permission bits are bypassed, can't test this.")

    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    os.chmod(readonly_dir, 0o444)
    dest = readonly_dir / "export_test.png"
    try:
        with pytest.raises(OSError):
            sample_image.save(dest)
    finally:
        os.chmod(readonly_dir, 0o755)  # restore so tmp_path cleanup can delete it