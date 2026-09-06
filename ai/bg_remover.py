# ai/bg_remover.py
#
# PHASE 2A -- Background Remover.
#
# Thin, deliberately dumb wrapper around `rembg` (U^2-Net, ONNX runtime
# backend). All the "AI" here is a pretrained third-party model doing
# foreground/background segmentation -- this file just: loads the model
# once, runs it, and gives back either a cutout (transparent PNG) or a
# cutout composited onto a new background (color / blur / custom image).
#
# Kept in ai/ (not core/) per the project's module split: core/ is
# classic deterministic image processing (see core/enhancer.py), ai/ is
# for anything backed by a trained model. rembg is CPU-friendly and has
# no CUDA requirement, so it fits the spec's "CPU-first, no mandatory
# cloud" rules even though it lives under ai/.
#
# Session/model lifecycle: rembg downloads its ONNX model (~176MB for
# the default "u2net") to a local cache directory on first use, then
# reuses it -- no network call after that, satisfying the "no mandatory
# cloud" privacy rule (the download itself happens once, locally cached,
# not per-image).
#
# PHASE 2A EXTRAS (still "dumb wrapper" territory, no new models):
#   - hair_refine on remove_background()      -> AI Hair & Edge Refinement
#     (rembg's built-in alpha-matting pass, same model/session)
#   - detect_subject_bbox()                   -> AI Subject Detection
#     (read off the alpha mask, not a second detector)
#   - smart_crop() / manual_crop()            -> Smart Crop / Manual Crop
#   - shadow_preserve() / shadow_remove()     -> Shadow Preserve/Remove
#     (heuristic dark/low-sat pixel recovery, not a shadow detector)

from io import BytesIO

import base64
import numpy as np
from PIL import Image, ImageFilter

_session = None  # lazily created; rembg's InferenceSession load is the
                  # slow part (~1-2s), so it's done once and reused for
                  # every image in the process, not per-call.


def _get_session():
    global _session
    if _session is None:
        # Imported lazily so the rest of the app (and its tests) don't
        # need onnxruntime/rembg installed just to import this module --
        # only Phase 2A's Remove BG screen actually needs it.
        from rembg import new_session

        _session = new_session("u2net")
    return _session


def remove_background(
    image: Image.Image, edge_feather: int = 0, hair_refine: bool = False
) -> Image.Image:
    """
    Returns a new RGBA PIL Image with the background cut out (alpha=0
    where the model decided "background").

    edge_feather: 0-100, blurs the alpha mask's edge so the cutout
    doesn't look like it was cut with scissors -- 0 leaves rembg's raw
    mask edge untouched, higher values soften it. Values above ~40 start
    visibly eating into fine detail (hair strands, fur), so the frontend
    slider should treat that as "soft" territory rather than "default".

    hair_refine: AI Hair & Edge Refinement. False (default) uses u2net's
    plain binary mask -- fast, but loose hair strands / fuzzy fur edges
    get cut clean off since a pixel is either "subject" or "not". True
    re-runs rembg with its built-in alpha-matting post-process (still
    the same model/session, no second network call), which produces a
    soft/partial alpha in fine-detail regions instead of a hard edge.
    Noticeably slower per image, so this is opt-in, not the default.

    Callers that need to re-feather the SAME cutout repeatedly (e.g. a
    live edge-refinement slider) should call remove_background() once
    with edge_feather=0, cache the raw cutout, and call feather_alpha()
    directly on each slider change instead of re-running the model.
    """
    from rembg import remove

    session = _get_session()

    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")

    if hair_refine:
        # alpha_matting thresholds are rembg's own defaults, tuned for
        # "confidently foreground" / "confidently background" cutoffs
        # with everything in between treated as a soft matting region
        # (hair, fur, motion-blurred edges).
        cutout_bytes = remove(
            buf.getvalue(),
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=270,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=11,
        )
    else:
        cutout_bytes = remove(buf.getvalue(), session=session)

    cutout = Image.open(BytesIO(cutout_bytes)).convert("RGBA")

    if edge_feather and edge_feather > 0:
        cutout = feather_alpha(cutout, edge_feather)

    return cutout


def feather_alpha(rgba: Image.Image, amount: int) -> Image.Image:
    """Softens the cutout's alpha-channel edge with a small Gaussian blur."""
    r, g, b, a = rgba.split()
    # 0-100 maps to a 0-6px blur radius -- enough to soften a hard cutout
    # edge without visibly eroding fine detail at low/mid settings.
    radius = max(0.0, min(100, amount) / 100.0 * 6.0)
    a = a.filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.merge("RGBA", (r, g, b, a))


def expand_contract_alpha(rgba: Image.Image, amount: int) -> Image.Image:
    """
    Grows or shrinks the cutout's silhouette by dilating/eroding its
    alpha channel -- lets the user push the cutout edge in or out a few
    pixels when rembg trims too tight (common on thin straps, jewelry)
    or leaves a rim of background color (common on busy backgrounds).

    amount: -30..30. Positive expands (grows the subject / dilate),
    negative contracts (shrinks the subject / erode). 0 is a no-op.
    """
    import cv2

    amount = max(-30, min(30, int(amount or 0)))
    if amount == 0:
        return rgba

    r, g, b, a = rgba.split()
    alpha = np.array(a)
    radius = abs(amount)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    if amount > 0:
        alpha = cv2.dilate(alpha, kernel)
    else:
        alpha = cv2.erode(alpha, kernel)
    a = Image.fromarray(alpha)
    return Image.merge("RGBA", (r, g, b, a))


def apply_manual_mask(rgba: Image.Image, mask_data_url: str) -> Image.Image:
    """
    Applies user-painted Keep/Remove brush strokes on top of the auto
    cutout's alpha channel, for touching up spots the automatic model
    got wrong (e.g. a stray finger cut off, or a chunk of background
    left behind near hair). This never re-runs the model -- it's a
    cheap alpha edit over the existing cutout.

    mask_data_url: base64 PNG data URL, same pixel dimensions as `rgba`,
    painted by the frontend with:
      - white  (255) = force KEEP (alpha -> 255) at that pixel
      - black  (0)   = force REMOVE (alpha -> 0) at that pixel
      - mid-gray (127ish, i.e. untouched/transparent brush layer) =
        leave the existing cutout alpha alone

    The frontend paints on a transparent canvas and only strokes actual
    brush pixels, so "untouched" here specifically means "this canvas
    pixel's own alpha is 0" (nothing painted there), not "gray colored".
    """
    if not mask_data_url:
        return rgba

    raw = mask_data_url.split(",", 1)[1] if "," in mask_data_url else mask_data_url
    mask_img = Image.open(BytesIO(base64.b64decode(raw))).convert("RGBA")
    if mask_img.size != rgba.size:
        mask_img = mask_img.resize(rgba.size, Image.NEAREST)

    mask_rgb = np.array(mask_img.convert("RGB"))
    mask_alpha = np.array(mask_img.split()[3])  # which pixels were actually painted

    r, g, b, a = rgba.split()
    alpha = np.array(a).astype(np.uint8)

    painted = mask_alpha > 10
    is_keep_stroke = painted & (mask_rgb[..., 0] > 128)   # white-ish strokes = keep
    is_remove_stroke = painted & (mask_rgb[..., 0] <= 128)  # black-ish strokes = remove

    alpha[is_keep_stroke] = 255
    alpha[is_remove_stroke] = 0

    return Image.merge("RGBA", (r, g, b, Image.fromarray(alpha)))


def replace_background(cutout: Image.Image, mode: str, **kwargs) -> Image.Image:
    """
    Composites an RGBA cutout (from remove_background) onto a new
    background and flattens to RGB. `mode` is one of:

      "transparent" -- no compositing, returns the RGBA cutout as-is
                        (still useful to route through here so callers
                        don't need a special case).
      "color"       -- kwargs["color"] = (r, g, b)
      "blur"        -- kwargs["original"] = the original PIL Image (RGB),
                        kwargs["blur_radius"] = int, blurs a copy of the
                        original photo and uses it as the backdrop, so
                        the subject appears to pop off a softly-out-of-
                        focus version of its own scene (a common "portrait
                        mode" look).
      "gradient"    -- kwargs["color1"], kwargs["color2"] = (r, g, b),
                        kwargs["angle"] = degrees (0 = left-to-right,
                        90 = top-to-bottom, etc). A straight 2-color
                        linear gradient backdrop.
      "image"       -- kwargs["background_path"] = path to a custom
                        background image, resized/cropped to cover the
                        cutout's dimensions. Optional kwargs["scale"]
                        (>=1.0, default 1.0) and kwargs["offset_x"]/
                        kwargs["offset_y"] (0-100, default 50/50 =
                        centered) let the user zoom into and pan the
                        background image within the cover-crop, same
                        idea as CSS background-position on top of
                        background-size: cover.
    """
    if mode == "transparent":
        return cutout

    width, height = cutout.size

    if mode == "color":
        color = kwargs.get("color", (255, 255, 255))
        backdrop = Image.new("RGB", (width, height), tuple(color))

    elif mode == "blur":
        original = kwargs.get("original")
        if original is None:
            raise ValueError("replace_background(mode='blur') requires kwargs['original']")
        radius = kwargs.get("blur_radius", 18)
        backdrop = original.convert("RGB").resize((width, height), Image.LANCZOS)
        backdrop = backdrop.filter(ImageFilter.GaussianBlur(radius=radius))

    elif mode == "gradient":
        color1 = tuple(int(c) for c in kwargs.get("color1", (30, 30, 40)))
        color2 = tuple(int(c) for c in kwargs.get("color2", (200, 200, 220)))
        angle = kwargs.get("angle", 90)
        backdrop = _linear_gradient(width, height, color1, color2, angle)

    elif mode == "image":
        bg_path = kwargs.get("background_path")
        if not bg_path:
            raise ValueError("replace_background(mode='image') requires kwargs['background_path']")
        scale = kwargs.get("scale", 1.0)
        offset_x = kwargs.get("offset_x", 50)
        offset_y = kwargs.get("offset_y", 50)
        backdrop = _cover_resize(
            Image.open(bg_path).convert("RGB"), width, height,
            scale=scale, offset_x_pct=offset_x, offset_y_pct=offset_y,
        )

    else:
        raise ValueError(f"Unknown replace_background mode: {mode!r}")

    result = Image.new("RGB", (width, height))
    result.paste(backdrop, (0, 0))
    result.paste(cutout, (0, 0), mask=cutout.split()[3])  # alpha channel as mask
    return result


def _linear_gradient(width: int, height: int, color1, color2, angle_degrees) -> Image.Image:
    """
    Straight 2-color linear gradient, `angle_degrees` measured the way
    a CSS `linear-gradient(<angle>deg, ...)` would (0deg = bottom-to-top,
    90deg = left-to-right) -- flipped once at the end vs. the raw math
    below so the frontend's angle slider matches what the user expects
    from every other gradient tool they've used.
    """
    theta = np.deg2rad(90 - (angle_degrees % 360))
    dx, dy = np.cos(theta), -np.sin(theta)

    xs = np.linspace(0, 1, width, dtype=np.float32)
    ys = np.linspace(0, 1, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Project every pixel onto the gradient direction, normalized to 0-1
    # across the whole image regardless of angle/aspect ratio.
    proj = grid_x * dx + grid_y * dy
    proj_min, proj_max = proj.min(), proj.max()
    t = (proj - proj_min) / max(1e-6, (proj_max - proj_min))
    t = t[..., None]

    c1 = np.array(color1, dtype=np.float32)
    c2 = np.array(color2, dtype=np.float32)
    arr = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _cover_resize(img: Image.Image, target_w: int, target_h: int, scale: float = 1.0,
                   offset_x_pct: float = 50, offset_y_pct: float = 50) -> Image.Image:
    """
    Resizes+crops `img` to exactly cover target_w x target_h (like CSS
    background-size: cover), then optionally zooms in further and pans
    within that cover-crop -- `scale` >= 1.0 zooms in (1.0 = no extra
    zoom, matching the original always-centered behaviour), `offset_x_pct`/
    `offset_y_pct` (0-100, 50 = centered) shift which part of the
    now-larger-than-frame image is kept, same idea as CSS
    background-position.
    """
    src_w, src_h = img.size
    base_scale = max(target_w / src_w, target_h / src_h)
    total_scale = base_scale * max(1.0, float(scale or 1.0))
    new_w, new_h = round(src_w * total_scale), round(src_h * total_scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    max_left = max(0, new_w - target_w)
    max_top = max(0, new_h - target_h)
    offset_x_pct = max(0, min(100, float(offset_x_pct or 50)))
    offset_y_pct = max(0, min(100, float(offset_y_pct or 50)))
    left = round(max_left * (offset_x_pct / 100.0))
    top = round(max_top * (offset_y_pct / 100.0))
    return img.crop((left, top, left + target_w, top + target_h))


def detect_subject_bbox(rgba: Image.Image, alpha_threshold: int = 10) -> dict | None:
    """
    AI Subject Detection.

    Deliberately NOT a second detector model -- it reads the bounding
    box straight out of the alpha channel that remove_background()
    already produced (i.e. "where the segmentation model decided the
    subject is"), so there's no extra inference cost. Good enough for
    every consumer of this in the app so far: Smart Crop and any
    frontend "subject outline" preview overlay.

    Returns {"left", "top", "right", "bottom"} in pixel coords (right/
    bottom are exclusive, PIL-crop-box style), or None if the cutout is
    fully transparent (model found nothing).
    """
    alpha = np.array(rgba.split()[3])
    ys, xs = np.where(alpha > alpha_threshold)
    if xs.size == 0:
        return None
    return {
        "left": int(xs.min()),
        "top": int(ys.min()),
        "right": int(xs.max()) + 1,
        "bottom": int(ys.max()) + 1,
    }


def smart_crop(image: Image.Image, rgba_cutout: Image.Image, padding: int = 20) -> Image.Image:
    """
    Smart Crop -- crops `image` to the subject's bounding box (via
    detect_subject_bbox) plus a pixel margin, instead of a fixed
    aspect-ratio or center crop. `image` can be the original photo or
    the RGBA cutout itself; `rgba_cutout` is only used to find WHERE the
    subject is.

    padding: extra pixels kept around the detected bounding box on all
    sides, so the crop doesn't hug the subject's silhouette exactly.

    Falls back to returning `image` unchanged if nothing was detected
    (empty cutout), rather than raising.
    """
    bbox = detect_subject_bbox(rgba_cutout)
    if bbox is None:
        return image

    w, h = image.size
    left = max(0, bbox["left"] - padding)
    top = max(0, bbox["top"] - padding)
    right = min(w, bbox["right"] + padding)
    bottom = min(h, bbox["bottom"] + padding)
    return image.crop((left, top, right, bottom))


def manual_crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """
    Manual Crop -- plain user-specified crop from the frontend's crop
    handles. box = (left, top, right, bottom) in the image's own pixel
    coordinates, PIL-crop-box style (right/bottom exclusive).

    Coordinates are clamped to the image bounds rather than raising, so
    a slightly-dragged-past-the-edge crop handle from the UI doesn't
    need frontend-side clamping too.
    """
    w, h = image.size
    left, top, right, bottom = box
    left = max(0, min(int(left), w))
    top = max(0, min(int(top), h))
    right = max(left, min(int(right), w))
    bottom = max(top, min(int(bottom), h))
    return image.crop((left, top, right, bottom))


def shadow_preserve(image: Image.Image, rgba_cutout: Image.Image, strength: int = 50) -> Image.Image:
    """
    Shadow Preserve.

    rembg's mask is binary (subject vs. everything else), so a
    subject's cast shadow on the floor/table normally gets cut away
    along with the rest of the background. This adds a soft, partial-
    alpha shadow back in by looking for dark, low-saturation pixels in
    the ORIGINAL photo that sit outside the raw cutout's silhouette
    (alpha ~0) and blending them back in at reduced opacity.

    This is a heuristic, not a shadow detector -- it works well for the
    common case (simple/plain floor or backdrop under the subject) and
    will misfire on busy/dark backgrounds (may pull in dark background
    detail as "shadow"). For those, leave strength at 0 or use
    shadow_remove().

    strength: 0-100. 0 = identical to a plain cutout (no shadow, same
    as calling remove_background() alone).
    """
    if not strength:
        return rgba_cutout

    import cv2

    orig_rgb = np.array(image.convert("RGB")).astype(np.float32)
    alpha = np.array(rgba_cutout.split()[3]).astype(np.float32)

    hsv = cv2.cvtColor(orig_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    value, sat = hsv[..., 2], hsv[..., 1]

    # "Shadow-like" = dark, not very saturated, and NOT already part of
    # the raw cutout (alpha near 0 there).
    shadow_like = (value < 140) & (sat < 60) & (alpha < 10)
    darkness = np.clip((140 - value) / 140.0, 0.0, 1.0)
    shadow_alpha = np.where(shadow_like, darkness * (strength / 100.0) * 180.0, 0.0)

    combined_alpha = np.maximum(alpha, shadow_alpha).astype(np.uint8)

    out_arr = np.array(rgba_cutout).copy()
    shadow_mask_bool = shadow_alpha > 0
    # Recolor the recovered shadow pixels from the ORIGINAL photo (the
    # cutout's own RGB is meaningless where alpha was 0).
    out_arr[..., :3][shadow_mask_bool] = orig_rgb[shadow_mask_bool]
    out_arr[..., 3] = combined_alpha

    return Image.fromarray(out_arr, mode="RGBA")


def shadow_remove(rgba_cutout: Image.Image) -> Image.Image:
    """
    Shadow Remove.

    rembg's default segmentation already excludes cast shadows (they're
    classified as background), so there's nothing to strip out here --
    this is a no-op that exists purely so the frontend's Shadow
    Preserve/Remove toggle has a real function to call on both sides
    instead of special-casing "off" as "don't call anything".
    """
    return rgba_cutout


# Phase 2A "Missing Professional Features" additions (this session):
# Edge Decontamination, Background Gradient, Background Position/Scale.
# All 🧮 classical -- no new model, same "dumb wrapper around numpy/cv2"
# territory as everything else in this file.

def decontaminate_edges(rgba: Image.Image, strength: int = 100) -> Image.Image:
    """
    Edge Decontamination ("Remove Color Fringing").

    A cutout's semi-transparent edge pixels (hair strands, soft fabric
    edges, motion blur) are a blend of the subject's true color AND
    whatever the original background behind them was -- rembg (and
    Feather Edge above) only ever touch the ALPHA channel, so that
    background-color "fringe"/halo stays baked into the RGB of every
    partial-alpha pixel untouched. On a white/bright background this
    shows up as a faint white-ish glow around hair; on a green/colored
    background it's a visible color spill.

    This estimates, for each partially-transparent edge pixel, what the
    local background color behind it probably was (a large blur of the
    confidently-background region, i.e. alpha < 15 -- cheap and good
    enough since decontamination only matters right at the edge, not
    deep into the background), then un-mixes it out using the standard
    alpha-compositing formula in reverse:

        observed = alpha*fg + (1-alpha)*bg
        =>   fg  = (observed - (1-alpha)*bg) / alpha

    strength: 0-100, how much of the estimated background tint to pull
    out (100 = full correction, 0 = no-op). Only ever touches pixels
    with 0 < alpha < 255 -- fully-opaque subject pixels and fully-
    transparent background pixels are left exactly as they were.
    """
    if not strength:
        return rgba

    import cv2

    strength = max(0, min(100, int(strength))) / 100.0

    r, g, b, a = rgba.split()
    rgb = np.array(Image.merge("RGB", (r, g, b))).astype(np.float32)
    alpha = np.array(a).astype(np.float32) / 255.0

    edge_mask = (alpha > 0.02) & (alpha < 0.98)
    if not edge_mask.any():
        return rgba

    # Estimate the local background color from confidently-background
    # pixels only, then fill the rest of the frame by heavily blurring
    # that sparse estimate (cv2.inpaint-lite via repeated large blur is
    # overkill for what's only needed right at the silhouette edge, so
    # a big Gaussian blur of the zeroed-elsewhere background is enough).
    bg_only = rgb.copy()
    bg_only[alpha >= 0.02] = 0
    bg_weight = (alpha < 0.02).astype(np.float32)
    ksize = 61
    blurred_bg = cv2.GaussianBlur(bg_only, (ksize, ksize), 0)
    blurred_weight = cv2.GaussianBlur(bg_weight, (ksize, ksize), 0)
    blurred_weight = np.clip(blurred_weight, 1e-3, None)
    bg_estimate = blurred_bg / blurred_weight[..., None]

    safe_alpha = np.clip(alpha, 0.05, 1.0)[..., None]  # avoid divide-by-near-zero blowing up color
    decontaminated = (rgb - (1.0 - safe_alpha) * bg_estimate) / safe_alpha
    decontaminated = np.clip(decontaminated, 0, 255)

    blend = edge_mask.astype(np.float32)[..., None] * strength
    out_rgb = rgb * (1 - blend) + decontaminated * blend
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

    r2, g2, b2 = Image.fromarray(out_rgb).split()
    return Image.merge("RGBA", (r2, g2, b2, a))