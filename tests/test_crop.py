# tests/test_crop.py -- PHASE 14 unit test coverage: straighten/crop/transpose

import pytest

from core.crop import crop_to_box, rotate_image, transpose_image


def test_rotate_image_zero_degrees_returns_same_object(sample_image):
    result = rotate_image(sample_image, 0)
    assert result is sample_image


def test_rotate_image_expands_canvas_for_nonzero_angle(sample_image):
    result = rotate_image(sample_image, 5)
    # A rotated image with expand=True must be at least as large in
    # both dimensions as the original (corners need somewhere to go).
    assert result.width >= sample_image.width
    assert result.height >= sample_image.height


def test_rotate_image_converts_to_rgb(sample_image):
    rgba = sample_image.convert("RGBA")
    result = rotate_image(rgba, 3)
    assert result.mode == "RGB"


def test_crop_to_box_basic(sample_image):
    result = crop_to_box(sample_image, (10, 10, 40, 30))
    assert result.size == (30, 20)


def test_crop_to_box_clamps_out_of_range_box(sample_image):
    w, h = sample_image.size
    # Way beyond the image bounds -- must clamp, not raise.
    result = crop_to_box(sample_image, (-100, -100, w + 500, h + 500))
    assert result.size == (w, h)


def test_crop_to_box_never_produces_empty_box(sample_image):
    # A degenerate/zero-size request should still clamp to a 1px-minimum box.
    result = crop_to_box(sample_image, (5, 5, 5, 5))
    assert result.width >= 1 and result.height >= 1


def test_transpose_rotate90_swaps_dimensions(sample_image):
    result = transpose_image(sample_image, "rotate90")
    assert result.size == (sample_image.height, sample_image.width)


def test_transpose_rotate180_keeps_dimensions(sample_image):
    result = transpose_image(sample_image, "rotate180")
    assert result.size == sample_image.size


def test_transpose_flip_h_keeps_dimensions(sample_image):
    result = transpose_image(sample_image, "flip_h")
    assert result.size == sample_image.size


def test_transpose_preserves_mode_for_rgba(sample_image):
    rgba = sample_image.convert("RGBA")
    result = transpose_image(rgba, "flip_v")
    assert result.mode == "RGBA"


def test_transpose_unknown_op_raises_value_error(sample_image):
    with pytest.raises(ValueError):
        transpose_image(sample_image, "not_a_real_op")