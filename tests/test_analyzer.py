# tests/test_analyzer.py -- PHASE 14 unit test coverage: analyzer

from core.analyzer import analyze_image


def test_analyze_image_returns_expected_top_level_keys(sample_image):
    result = analyze_image(sample_image)
    expected_keys = {
        "width", "height", "megapixels", "orientation",
        "face_detected", "face_count", "faces",
        "subject_type", "indoor_outdoor", "lighting",
        "light_quality", "brightness", "contrast",
        "dominant_colors", "quality", "sky", "background",
    }
    assert expected_keys.issubset(result.keys())


def test_analyze_image_reports_correct_dimensions(sample_image):
    result = analyze_image(sample_image)
    assert result["width"] == sample_image.width
    assert result["height"] == sample_image.height


def test_analyze_image_orientation_is_valid_label(sample_image):
    result = analyze_image(sample_image)
    assert result["orientation"] in ("landscape", "portrait", "square")


def test_analyze_image_no_faces_in_flat_color_image():
    from PIL import Image
    flat = Image.new("RGB", (100, 100), color=(128, 128, 128))
    result = analyze_image(flat)
    assert result["face_detected"] is False
    assert result["face_count"] == 0


def test_analyze_image_brightness_and_contrast_in_range(sample_image):
    result = analyze_image(sample_image)
    assert 0 <= result["brightness"] <= 100
    assert 0 <= result["contrast"] <= 100


def test_analyze_image_dominant_colors_present(sample_image):
    result = analyze_image(sample_image)
    assert isinstance(result["dominant_colors"], list)
    assert len(result["dominant_colors"]) > 0
    assert "hex" in result["dominant_colors"][0]


def test_analyze_image_never_raises_on_tiny_image():
    from PIL import Image
    tiny = Image.new("RGB", (2, 2), color=(255, 0, 0))
    result = analyze_image(tiny)
    assert result["width"] == 2 and result["height"] == 2