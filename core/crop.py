# core/crop.py
#
# PHASE 3 (on-request addition) -- Straighten (rotate) + Crop for the
# main Image Editor.
#
# Classic deterministic geometry (PIL only) -- no trained model, same
# module-split rule as core/enhancer.py and core/object_remover.py
# (ai/ is reserved for trained-model features).
#
# Scope: this is meant for FIXING a slightly crooked photo (small
# angles, -45..45 degrees) and then cropping it, not free-form
# arbitrary-angle art rotation. The two operations are exposed as two
# separate, independently-callable functions (rotate_image,
# crop_to_box) rather than one combined call, because the editor
# applies them as two separate, user-confirmed steps (see
# ui/bridge.py::straightenImage / cropImage and frontend/js/editor.js):
# the user drags the Straighten slider and applies it FIRST, then opens
# Manual Crop against the now-straightened photo. Keeping them separate
# functions means each step's result becomes the new working image
# (same hand-off pattern already used by "Continue in Remove BG" after
# an export) -- so a crop box the user draws always lines up exactly
# with what they see on screen, with no rotate/crop coordinate math to
# get wrong.

from PIL import Image


def rotate_image(image: Image.Image, angle_degrees: float) -> Image.Image:
    """
    Rotates `image` by `angle_degrees` to straighten a slightly tilted
    photo. Positive = clockwise (matches the on-screen slider direction
    the user sees, opposite of PIL's default counter-clockwise-positive
    convention, so it's negated below).

    The canvas expands to fit the full rotated frame (no corners get
    silently clipped) -- the newly-exposed corners are filled with
    white, since a straighten rotation is nearly always followed by a
    crop that trims them away. Uses bicubic resampling for a clean edge
    on a small-angle rotation (nearest/bilinear would look noticeably
    aliased on straight lines like a horizon).
    """
    if not angle_degrees:
        return image
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    return rgb.rotate(
        -angle_degrees,
        resample=Image.BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )


def crop_to_box(image: Image.Image, box) -> Image.Image:
    """
    Crops `image` to `box` = (left, top, right, bottom) in the image's
    own pixel coordinates (i.e. the coordinates the frontend already
    computed against this exact image's width/height -- see
    naturalBoxFromCropBoxEl() in editor.js).

    Coordinates are clamped to the image bounds and guaranteed to
    produce a non-empty box, so a slightly out-of-range box (rounding
    at the very edge) never raises instead of just quietly clamping.
    """
    left, top, right, bottom = box
    left = max(0, min(int(round(left)), image.width - 1))
    top = max(0, min(int(round(top)), image.height - 1))
    right = max(left + 1, min(int(round(right)), image.width))
    bottom = max(top + 1, min(int(round(bottom)), image.height))
    return image.crop((left, top, right, bottom))


# Missing-feature #1 (this session): Rotate 90 / Flip H / Flip V.
#
# Deliberately a SEPARATE function from rotate_image() above, not a
# special-case of it, for two reasons:
#   1. rotate_image() always converts to RGB and expands the canvas
#      with a white fill -- correct for straightening (which leaves
#      angled slivers to fill), but wrong here: a 90/flip is exact, so
#      converting a transparent PNG to RGB or filling corners that
#      don't exist would be a real quality regression.
#   2. PIL's transpose() is lossless and mode-preserving (RGBA stays
#      RGBA) -- no resampling at all, unlike the BICUBIC fine-angle
#      rotate above.
_TRANSPOSE_OPS = {
    "rotate90": Image.ROTATE_270,  # PIL rotates counter-clockwise; users expect clockwise
    "rotate180": Image.ROTATE_180,
    "rotate270": Image.ROTATE_90,
    "flip_h": Image.FLIP_LEFT_RIGHT,
    "flip_v": Image.FLIP_TOP_BOTTOM,
}


def transpose_image(image: Image.Image, op: str) -> Image.Image:
    """
    Lossless 90-degree rotation / mirroring. `op` is one of:
    "rotate90" (clockwise), "rotate180", "rotate270" (counter-clockwise
    90), "flip_h" (mirror left-right), "flip_v" (mirror top-bottom).
    """
    transpose_const = _TRANSPOSE_OPS.get(op)
    if transpose_const is None:
        raise ValueError(f"Unknown transpose op: {op!r}")
    return image.transpose(transpose_const)