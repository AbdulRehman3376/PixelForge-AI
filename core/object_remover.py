# core/object_remover.py
#
# PHASE 2A -- Object Remover (mask-brush + inpaint).
#
# Classic deterministic OpenCV inpainting, no trained model -- belongs
# in core/ per the project's module split (see core/enhancer.py's header
# note; ai/ is reserved for trained-model features like ai/bg_remover.py).
#
# The frontend lets the user paint a rough mask over an unwanted object
# (a stray person, a trash can, a power line) on an HTML <canvas>; that
# mask comes back here as a base64 PNG (white = erase, black = keep) and
# cv2.inpaint fills the masked region using the surrounding pixels.
#
# This is a first-version tool, not a generative object-removal model --
# it works well for small/medium objects on relatively simple
# backgrounds (sky, grass, walls, water) and struggles on complex
# textures or large masked areas, same as any classic inpainting
# algorithm. That's an inherent limitation of the algorithm, not a bug.

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

INPAINT_METHODS = {
    "telea": cv2.INPAINT_TELEA,          # fast, edge-based -- better default for most photos
    "navier_stokes": cv2.INPAINT_NS,     # fluid-dynamics based -- sometimes smoother on textures
}

# Radius of the circular neighborhood cv2.inpaint samples from. Larger =
# smoother but blurrier fill; kept modest since these are photos, not
# small scratches/dust the algorithm was originally designed for.
_INPAINT_RADIUS = 5


def _decode_mask_data_url(mask_data_url: str, target_size: tuple[int, int]) -> np.ndarray:
    """
    Decodes a base64 "data:image/png;base64,...." string (from the
    frontend's mask-brush <canvas>) into a single-channel uint8 mask
    resized to target_size (width, height), thresholded to strict 0/255.
    """
    if "," in mask_data_url:
        mask_data_url = mask_data_url.split(",", 1)[1]
    raw = base64.b64decode(mask_data_url)
    mask_img = Image.open(BytesIO(raw)).convert("L")
    if mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.NEAREST)
    mask = np.array(mask_img)
    # Anti-aliased brush strokes leave semi-transparent edge pixels;
    # threshold at the midpoint so cv2.inpaint gets a clean binary mask
    # rather than treating "barely painted" pixels as fully masked.
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def erase_object(image: Image.Image, mask_data_url: str, method: str = "telea") -> Image.Image:
    """
    Fills the masked region of `image` using OpenCV inpainting.

    image:          PIL Image (any mode; converted to RGB).
    mask_data_url:  base64 PNG data URL from the mask-brush canvas,
                    same pixel dimensions as the (possibly downscaled)
                    preview the user was painting on -- resized to match
                    `image` here so full-res export and low-res preview
                    both work from the same brush strokes.
    method:         "telea" (default) or "navier_stokes".

    Returns a new PIL Image (RGB) with the masked region filled in.
    """
    if method not in INPAINT_METHODS:
        raise ValueError(f"Unknown inpaint method: {method!r} (expected one of {list(INPAINT_METHODS)})")

    rgb = image.convert("RGB")
    cv_img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    mask = _decode_mask_data_url(mask_data_url, target_size=rgb.size)
    if not mask.any():
        # Nothing painted -- return the image unchanged rather than
        # spending time calling inpaint with an empty mask.
        return rgb

    result = cv2.inpaint(cv_img, mask, _INPAINT_RADIUS, INPAINT_METHODS[method])
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))