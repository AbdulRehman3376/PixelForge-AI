# core/enhancer.py
#
# PHASE 2/3 -- Manual/professional adjustment engine.
#
# This is the real per-pixel processing behind the Image Editor's manual
# controls. Kept separate from ui/bridge.py (which should stay a thin
# JS<->Python relay) per the project's module responsibilities.
#
# IMPORTANT SCOPE NOTE: everything in here is classic, deterministic
# image processing (numpy/Pillow/OpenCV math) -- no trained models, no
# "AI" in the marketing sense. Features that genuinely need a trained
# model (face detection, deblur, super-resolution, scene classification)
# are NOT implemented here on purpose -- they belong to Phase 5/6 (Analyzer
# / Smart Pipeline) and Phase 9/10 (Real-ESRGAN / GFPGAN), once those
# dependencies are intentionally installed. Labeling a CSS/numpy trick as
# "AI Deblur" here would be misleading, so it's left out rather than faked.
#
# All adjustment values use consistent ranges so the frontend sliders map
# 1:1 onto this dict:
#
#   brightness   0-200   (100 = no change)   multiplicative
#   contrast     0-200   (100 = no change)   multiplicative
#   saturation   0-200   (100 = no change)   multiplicative
#   exposure     -100..100 (0 = no change)   stops-ish
#   highlights   -100..100 (0 = no change)
#   shadows      -100..100 (0 = no change)
#   whites       -100..100 (0 = no change)
#   blacks       -100..100 (0 = no change)
#   temperature  -100..100 (0 = no change)   negative=cooler, positive=warmer
#   tint         -100..100 (0 = no change)   negative=green, positive=magenta
#   vibrance     -100..100 (0 = no change)   selective saturation
#   sharpness    0-100     (0 = no change)
#   clarity      0-100     (0 = no change)   local/mid-tone contrast
#   noise_reduction 0-100  (0 = no change)
#   vignette     0-100     (0 = no change)   PHASE 3 -- radial edge darkening
#   smoke        0-100     (0 = no change)   PHASE 4 -- procedural
#                       atmospheric smoke/haze overlay (layered,
#                       Gaussian-blurred noise, screen-blended --
#                       classic generation, not a trained model). This
#                       is an intentional creative effect, distinct
#                       from Phase 9's "smoke/haze artifact control"
#                       (which suppresses unwanted AI-upscaling
#                       artifacts, not add one).
#
# PHASE 3 (manual controls extension) -- added below, all still classic
# deterministic processing (no trained models), consistent with the
# scope note above:
#
#   highlight_recovery 0-100 (0 = no change) targeted recovery of
#                       near-clipped highlight detail -- distinct from
#                       the "highlights" tone-curve push above, which
#                       nudges the whole upper tonal range. This one
#                       specifically compresses the top of the range
#                       with a soft knee so blown skies/windows regain
#                       texture instead of staying flat white.
#   shadow_recovery  0-100 (0 = no change) same idea for the bottom of
#                       the range -- opens up near-black shadow detail
#                       with a lift curve instead of a flat push.
#   skin_protect     0/1 (0 = off)         when on, vibrance/saturation/
#                       temperature-tint shifts are attenuated over
#                       detected skin-tone pixels so portraits don't
#                       drift orange/plastic-looking when those sliders
#                       are pushed hard for the rest of the scene.
#   sharpen_protect  0/1 (0 = off)         when on, sharpening/clarity
#                       are applied through an edge-aware + skin-aware
#                       mask so flat/noisy areas and skin don't pick up
#                       halos and pore/texture artifacts.
#   local_adjustments  list (default [])   PHASE 3 -- Selective/Local
#                       Adjustments. Each entry is a single radial
#                       ("Lightroom radial filter"-style) region:
#                         {
#                           "cx": 0-1, "cy": 0-1,     # center, fraction of image size
#                           "rx": 0-1, "ry": 0-1,     # radii, fraction of image size
#                           "feather": 0-100,          # edge softness
#                           "invert": bool,            # true = adjust OUTSIDE the ellipse
#                           "adjust": {"exposure": -100..100,
#                                      "saturation": 0..200,
#                                      "sharpness": 0..100}
#                         }
#                       Applied last, as a masked blend on top of the
#                       already fully-processed image -- i.e. a "brush"
#                       for pushing exposure/saturation/sharpness over
#                       just one region (a sky, a face, a product)
#                       without touching the rest of the photo.

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DEFAULT_ADJUSTMENTS = {
    "brightness": 100,
    "contrast": 100,
    "saturation": 100,
    "exposure": 0,
    "highlights": 0,
    "shadows": 0,
    "whites": 0,
    "blacks": 0,
    "temperature": 0,
    "tint": 0,
    "vibrance": 0,
    "sharpness": 0,
    "clarity": 0,
    "noise_reduction": 0,
    "vignette": 0,
    "smoke": 0,
    "highlight_recovery": 0,
    "shadow_recovery": 0,
    "skin_protect": 0,
    "sharpen_protect": 0,
    "local_adjustments": [],
}


def _get(adj, key):
    return adj.get(key, DEFAULT_ADJUSTMENTS[key])


# ===================== Phase 3 extension: helpers =====================

def _skin_tone_mask(arr):
    """
    Soft 0..1 mask of skin-like pixels, computed in YCbCr space (the
    standard space for this heuristic -- luminance-independent, so it
    works across skin tones and lighting). Rather than a hard in/out
    threshold (which bands visibly once blended back in), each pixel
    gets a continuous score based on how close its (Cb, Cr) pair is to
    the center of the typical skin-tone range, so the protection fades
    smoothly at the edges instead of cutting a hard-edged mask into the
    photo.
    """
    rgb = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]

    # Typical skin-tone band in Cb/Cr (well-established heuristic range).
    cb_center, cr_center = 111.0, 154.0
    cb_spread, cr_spread = 18.0, 20.0

    d = ((cb - cb_center) / cb_spread) ** 2 + ((cr - cr_center) / cr_spread) ** 2
    mask = np.clip(1.0 - d / 2.0, 0.0, 1.0) ** 1.5
    return mask[..., None]


def _edge_energy_mask(gray_u8):
    """
    0..1 mask of "how much real detail is here" from local Laplacian
    variance -- high over edges/texture, near-zero over flat/smooth
    areas (sky, out-of-focus background, skin). Used to keep sharpening
    away from areas where it can only amplify noise or create halos,
    instead of adding detail that doesn't exist.
    """
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    energy = cv2.GaussianBlur(np.abs(lap), (0, 0), sigmaX=3)
    if energy.max() > 1e-6:
        energy = energy / np.percentile(energy, 98).clip(min=1e-6)
    return np.clip(energy, 0.0, 1.0)


def _apply_highlight_shadow_recovery(arr, highlight_recovery, shadow_recovery):
    """
    Targeted recovery of near-clipped detail, separate from the
    highlights/shadows tone-curve sliders. Uses a soft-knee compression
    on the top of the range (highlight_recovery) and a lift curve on the
    bottom (shadow_recovery), each weighted by luminance so midtones are
    left alone -- only the parts of the image actually at risk of
    clipping are pulled back.
    """
    if not highlight_recovery and not shadow_recovery:
        return arr

    luminance = arr.mean(axis=2, keepdims=True)

    if highlight_recovery:
        amt = highlight_recovery / 100.0
        # Soft knee starting at 70% luminance: compress values above the
        # knee toward it instead of letting them ride up to hard white.
        knee = 0.70
        over = np.clip(luminance - knee, 0.0, None)
        compressed = knee + over / (1.0 + over * 6.0 * amt)
        gain = np.divide(compressed, luminance, out=np.ones_like(luminance), where=luminance > 1e-4)
        weight = np.clip((luminance - knee) / (1.0 - knee), 0.0, 1.0)
        arr = arr * (1.0 - weight) + (arr * gain) * weight

    if shadow_recovery:
        amt = shadow_recovery / 100.0
        # Lift curve below 30% luminance -- opens shadow detail without
        # touching anything already in the midtones/highlights.
        floor = 0.30
        weight = np.clip((floor - luminance) / floor, 0.0, 1.0)
        lift = weight * amt * 0.45
        arr = arr + lift

    return arr


def _apply_temperature_tint(arr, temperature, tint, protect_mask=None):
    if not temperature and not tint:
        return arr
    t = temperature / 100.0
    tt = tint / 100.0
    arr = arr.copy()
    dr = t * 0.12          # Red   (warm+)
    db = -t * 0.12         # Blue  (cool-)
    dg = -tt * 0.08        # Green (positive tint = more magenta)

    if protect_mask is not None:
        # Skin-tone Protection: halve the color cast over skin-like
        # pixels instead of zeroing it out entirely, so a strong
        # temperature push still reads as "warmer photo" while faces
        # don't swing as far toward orange/green.
        strength = 1.0 - protect_mask * 0.6
        arr[..., 0] += dr * strength[..., 0]
        arr[..., 1] += dg * strength[..., 0]
        arr[..., 2] += db * strength[..., 0]
    else:
        arr[..., 0] += dr
        arr[..., 1] += dg
        arr[..., 2] += db
    return arr


def _apply_vibrance(arr, vibrance, protect_mask=None):
    if not vibrance:
        return arr
    maxc = arr.max(axis=2, keepdims=True)
    minc = arr.min(axis=2, keepdims=True)
    sat = (maxc - minc) / (maxc + 1e-6)
    # Boost less-saturated pixels more than already-saturated ones, so
    # skin tones (usually already reasonably saturated) shift less than
    # a dull sky or muted grass -- a cheap approximation of a proper
    # vibrance control.
    boost = (1.0 - sat) * (vibrance / 100.0) * 0.6
    if protect_mask is not None:
        # Skin-tone Protection: further damp the boost specifically over
        # detected skin, on top of vibrance's own built-in skin leniency,
        # so a heavy vibrance push on foliage/sky doesn't also push
        # faces toward oversaturated orange.
        boost = boost * (1.0 - protect_mask * 0.75)
    return minc + (arr - minc) * (1.0 + boost)


def _apply_tone_curve(arr, highlights, shadows, whites, blacks):
    """Simple luminance-masked tone adjustments (Levels/Curves-lite)."""
    if not any([highlights, shadows, whites, blacks]):
        return arr

    luminance = arr.mean(axis=2, keepdims=True)

    if shadows:
        shadow_mask = np.clip(1.0 - luminance * 2.0, 0.0, 1.0)
        arr = arr + (shadows / 100.0) * 0.35 * shadow_mask

    if highlights:
        highlight_mask = np.clip((luminance - 0.5) * 2.0, 0.0, 1.0)
        arr = arr + (highlights / 100.0) * 0.35 * highlight_mask

    if whites:
        arr = arr + (whites / 100.0) * 0.3 * (luminance ** 3)

    if blacks:
        arr = arr - (blacks / 100.0) * 0.3 * ((1.0 - luminance) ** 3)

    return arr


def _apply_exposure(arr, exposure):
    if not exposure:
        return arr
    stops = (exposure / 100.0) * 1.2
    return arr * (2.0 ** stops)


def _apply_vignette(arr, amount):
    """
    PHASE 3 -- radial edge darkening. amount: 0 (none, default) .. 100
    (strong). Purely geometric (distance from center), independent of
    the photo's own content -- this is the classic "Vignette" slider,
    not a subject-aware effect (that distinction matters here since
    Phase 3's remaining, genuinely subject/scene-aware behavior is
    deliberately deferred to Phase 5/6's Analyzer + Smart Pipeline --
    see the spec note on this phase).
    """
    if not amount:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    max_dist = np.sqrt(cx ** 2 + cy ** 2) or 1.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_dist

    # Smooth falloff rather than a hard circle: stays fully bright out to
    # ~35% of the way to the corner, then eases down toward the corners
    # so the effect reads as a natural lens vignette, not a spotlight.
    falloff = np.clip((dist - 0.35) / 0.65, 0.0, 1.0)
    strength = (amount / 100.0) * 0.85  # cap so amount=100 dims corners, never blacks them out
    multiplier = 1.0 - falloff * strength
    return arr * multiplier[..., None]


def _apply_smoke(arr, amount):
    """
    PHASE 4 -- procedural atmospheric smoke/haze overlay. amount: 0
    (none, default) .. 100 (heavy). Purely a synthesized soft-noise
    texture (layered low-res random noise, upsampled + Gaussian
    blurred into soft cloud-like wisps, screen-blended over the
    photo) -- classic deterministic generation, not an AI-generated
    smoke asset or a trained model, consistent with this module's
    scope note above. Deterministic seed so the same amount always
    reproduces the same wisp pattern rather than re-randomizing on
    every slider tick.

    BUGFIX (this session): the old shaping (gamma=2.2, blur sigma
    scaled fairly wide, ground_bias floor 0.4) left the texture
    continuous and only lightly faded everywhere -- so even the
    "clear" parts of the frame still had a faint haze tint, and the
    net effect read as a flat wash covering the WHOLE photo instead of
    a few distinct wisps with real clear gaps between them. Fixed by:
    a much steeper gamma (pushes far more of the frame to true 0,
    not just near-0), a hard threshold that zeroes out anything below
    it (creates actual clear gaps rather than a continuous fade),
    tighter blur (wisps stay more localized instead of smearing into
    one broad haze sheet), and a much lower ground_bias floor at the
    top of the frame (top of the photo now stays essentially untouched
    instead of carrying ~40% of the effect everywhere).
    """
    if not amount:
        return arr
    h, w = arr.shape[:2]

    rng = np.random.default_rng(20240601)  # fixed seed -- stable look
    texture = np.zeros((h, w), dtype=np.float32)
    # A few octaves of low-res noise upsampled with cubic interpolation
    # reads as soft, irregular wisps rather than a flat fog sheet.
    octaves = [(6, 0.45), (14, 0.30), (30, 0.25)]
    for grid, weight in octaves:
        gh = max(2, grid)
        gw = max(2, int(round(grid * w / max(h, 1))))
        small = rng.random((gh, gw), dtype=np.float32)
        layer = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        texture += layer * weight
    # Tighter blur than before -- keeps wisps looking like distinct
    # patches instead of blurring into one broad haze sheet that
    # touches every pixel.
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=max(w, h) * 0.006 + 2)
    texture -= texture.min()
    if texture.max() > 1e-6:
        texture /= texture.max()

    # Steeper gamma than before (2.2 -> 3.6): pushes the large majority
    # of the frame to true 0, leaving only the brightest noise peaks as
    # visible wisps -- real clear sky/background should stay clear.
    texture = texture ** 3.6

    # Hard threshold: anything still faint after the gamma curve gets
    # zeroed out completely rather than left as a barely-visible tint.
    # This is what actually creates clear GAPS between wisps instead of
    # a continuous (if subtle) haze over the entire picture.
    SMOKE_THRESHOLD = 0.22
    texture = np.where(texture > SMOKE_THRESHOLD, (texture - SMOKE_THRESHOLD) / (1 - SMOKE_THRESHOLD), 0.0)

    # Bias toward the lower half of the frame so it reads as ground
    # haze/smoke drifting up, not a uniform screen-tint. Floor lowered
    # from 0.4 to 0.08 so the top of the frame stays essentially clear
    # instead of still carrying a chunk of the effect.
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    ground_bias = 0.08 + 0.92 * yy
    texture = np.clip(texture * ground_bias, 0.0, 1.0)

    strength = (amount / 100.0) * 0.45  # cap well below 1.0 -- even amount=100 keeps the photo visible through the smoke
    smoke_tone = 0.92  # slightly warm-neutral smoke/haze color, not pure white
    alpha = (texture * strength)[..., None]
    return arr + (smoke_tone - arr) * alpha


def _apply_noise_reduction(pil_img, amount):
    if not amount:
        return pil_img
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    strength = max(1, round((amount / 100.0) * 10))
    denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, strength, strength, 7, 21)
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))


def _protection_mask(pil_img, skin_protect):
    """
    Builds the combined 0..1 mask used by Smart Sharpen Protection:
    high (protect strongly) over flat/low-detail areas and, if
    skin_protect is also on, over detected skin -- low (sharpen freely)
    over genuine high-frequency detail (eyes, hair strands, fabric
    weave, edges). Returned as an HxWx1 float array.
    """
    rgb = np.asarray(pil_img).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detail = _edge_energy_mask(gray)               # 1 = real detail, 0 = flat
    flat_protect = 1.0 - detail

    if skin_protect:
        skin = _skin_tone_mask(rgb.astype(np.float32) / 255.0)[..., 0]
        flat_protect = np.clip(flat_protect + skin * 0.7, 0.0, 1.0)

    return flat_protect[..., None]


def _apply_sharpness(pil_img, amount, sharpen_protect=False, skin_protect=False):
    if not amount:
        return pil_img
    percent = int((amount / 100.0) * 150)
    sharpened = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))

    if not sharpen_protect:
        return sharpened

    # Smart Sharpen Protection: blend the sharpened result back with the
    # original, weighted by the protection mask, so noise/flat areas
    # (and skin, if that toggle is also on) don't pick up halos/grain
    # while genuine edges still get the full sharpening amount.
    mask = _protection_mask(pil_img, skin_protect)
    original = np.asarray(pil_img).astype(np.float32)
    sharp_arr = np.asarray(sharpened).astype(np.float32)
    blended = sharp_arr * (1.0 - mask) + original * mask
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _apply_clarity(pil_img, amount, sharpen_protect=False, skin_protect=False):
    """Local/mid-tone contrast via a wide-radius unsharp mask."""
    if not amount:
        return pil_img
    percent = int((amount / 100.0) * 80)
    result = pil_img.filter(ImageFilter.UnsharpMask(radius=24, percent=percent, threshold=2))

    if not sharpen_protect:
        return result

    # Same protection idea as sharpness above, but skin is protected
    # more aggressively here since clarity's wide radius is exactly what
    # makes pores/wrinkles look harsh if left unprotected on a portrait.
    mask = _protection_mask(pil_img, skin_protect)
    if skin_protect:
        skin = _skin_tone_mask(np.asarray(pil_img).astype(np.float32) / 255.0)[..., 0]
        mask = np.clip(mask[..., 0] + skin * 0.3, 0.0, 1.0)[..., None]
    original = np.asarray(pil_img).astype(np.float32)
    result_arr = np.asarray(result).astype(np.float32)
    blended = result_arr * (1.0 - mask) + original * mask
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# ===================== Phase 3: Selective/Local Adjustments =====================

def _ellipse_mask(h, w, cx, cy, rx, ry, feather):
    """
    0..1 soft ellipse mask in image space. cx/cy/rx/ry are fractions of
    (w, h) so the same mask definition scales correctly between the
    downsized live preview and the full-resolution export. `feather`
    (0-100) controls how wide the soft edge is, matching Lightroom's
    radial-filter feather slider.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx_px, cy_px = cx * w, cy * h
    rx_px = max(rx * w, 1.0)
    ry_px = max(ry * h, 1.0)

    d = np.sqrt(((xx - cx_px) / rx_px) ** 2 + ((yy - cy_px) / ry_px) ** 2)
    # d <= 1 is inside the ellipse. Feather widens the falloff band
    # straddling the boundary instead of a hard cutoff at d == 1.
    band = max(feather / 100.0, 0.02)
    mask = 1.0 - np.clip((d - (1.0 - band)) / (2 * band), 0.0, 1.0)
    return mask


def _apply_local_adjustments(image: Image.Image, local_adjustments) -> Image.Image:
    """
    Applies each Selective/Local Adjustment region on top of the
    already fully-processed image. Kept as its own pass (rather than
    folded into the main array pipeline) because it operates on the
    final rendered result -- exactly like a Lightroom radial filter
    brush stroke, not a per-slider global operation.
    """
    if not local_adjustments:
        return image

    for region in local_adjustments:
        try:
            cx = float(region.get("cx", 0.5))
            cy = float(region.get("cy", 0.5))
            rx = float(region.get("rx", 0.25))
            ry = float(region.get("ry", 0.25))
            feather = float(region.get("feather", 50))
            invert = bool(region.get("invert", False))
            local_adj = region.get("adjust", {}) or {}
        except (TypeError, ValueError):
            continue

        w, h = image.size
        mask = _ellipse_mask(h, w, cx, cy, rx, ry, feather)
        if invert:
            mask = 1.0 - mask
        if mask.max() <= 0.0:
            continue

        region_exposure = float(local_adj.get("exposure", 0) or 0)
        region_saturation = float(local_adj.get("saturation", 100) or 100)
        region_sharpness = float(local_adj.get("sharpness", 0) or 0)

        adjusted = image
        if region_exposure:
            arr = np.asarray(adjusted).astype(np.float32) / 255.0
            arr = _apply_exposure(arr, region_exposure)
            adjusted = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        if region_saturation != 100:
            adjusted = ImageEnhance.Color(adjusted).enhance(region_saturation / 100.0)
        if region_sharpness:
            adjusted = _apply_sharpness(adjusted, region_sharpness)

        base_arr = np.asarray(image).astype(np.float32)
        adj_arr = np.asarray(adjusted).astype(np.float32)
        blended = base_arr * (1.0 - mask[..., None]) + adj_arr * mask[..., None]
        image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    return image


# ===================== Phase 3: Auto White Balance =====================

def auto_white_balance(image: Image.Image) -> dict:
    """
    Dedicated Auto White Balance button. Uses a "gray-world on reliable
    midtones" estimate: pixels that are near-black, near-white, or
    heavily saturated (likely a colorful subject rather than a neutral
    surface) are excluded, then the remaining pixels' per-channel means
    are compared -- a scene with a correct white balance should average
    out close to neutral gray across R/G/B. The channel imbalance is
    converted back into this app's -100..100 temperature/tint scale
    (the exact inverse of _apply_temperature_tint's per-channel deltas)
    so the result can be dropped straight into the existing sliders.
    """
    img = image.convert("RGB") if image.mode != "RGB" else image
    small = img.copy()
    small.thumbnail((400, 400), Image.LANCZOS)
    arr = np.asarray(small).astype(np.float32) / 255.0

    luminance = arr.mean(axis=2)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = (maxc - minc) / (maxc + 1e-6)

    reliable = (luminance > 0.08) & (luminance < 0.92) & (saturation < 0.4)
    if reliable.sum() < 200:  # not enough neutral-ish pixels -- fall back to everything
        reliable = np.ones_like(luminance, dtype=bool)

    r_mean = float(arr[..., 0][reliable].mean())
    g_mean = float(arr[..., 1][reliable].mean())
    b_mean = float(arr[..., 2][reliable].mean())

    # Inverse of _apply_temperature_tint: temperature pushed R up/B down
    # by t*0.12 each; tint pushed G down by tt*0.08. Solve for the t/tt
    # that would bring (r_mean, g_mean, b_mean) back to neutral.
    warm_bias = (r_mean - b_mean) / 2.0
    temperature = int(np.clip(-warm_bias / 0.12 * 100.0, -100, 100))
    tint_bias = g_mean - (r_mean + b_mean) / 2.0
    tint = int(np.clip(tint_bias / 0.08 * 100.0, -100, 100))

    if abs(temperature) < 3 and abs(tint) < 3:
        summary = "White balance already looks neutral -- little to correct."
    else:
        cast = "warm/orange" if temperature < 0 else "cool/blue" if temperature > 0 else None
        parts = []
        if cast:
            parts.append(f"a {cast} cast")
        if abs(tint) >= 3:
            parts.append("a green/magenta shift" if abs(tint) >= abs(temperature) else "a slight tint shift")
        summary = f"Detected {' and '.join(parts)} in the neutral tones -- corrected."

    return {"ok": True, "temperature": temperature, "tint": tint, "summary": summary}


def apply_adjustments(image: Image.Image, adjustments: dict) -> Image.Image:
    """
    Applies the full manual-control pipeline to a PIL Image and returns
    a new PIL Image. Order roughly follows a real raw-editing pipeline:
    exposure -> tone curve -> highlight/shadow recovery -> color
    (skin-protected if enabled) -> vignette -> smoke/haze overlay ->
    noise reduction -> global brightness/contrast/saturation ->
    clarity -> sharpness
    (sharpen-protected if enabled) -> Selective/Local Adjustments last,
    as a masked brush on top of the fully-rendered result.
    """
    adj = {**DEFAULT_ADJUSTMENTS, **(adjustments or {})}
    skin_protect = bool(_get(adj, "skin_protect"))
    sharpen_protect = bool(_get(adj, "sharpen_protect"))

    img = image.convert("RGB") if image.mode not in ("RGB",) else image
    arr = np.asarray(img).astype(np.float32) / 255.0

    protect_mask = _skin_tone_mask(arr) if skin_protect else None

    arr = _apply_exposure(arr, _get(adj, "exposure"))
    arr = _apply_tone_curve(
        arr, _get(adj, "highlights"), _get(adj, "shadows"), _get(adj, "whites"), _get(adj, "blacks")
    )
    arr = _apply_highlight_shadow_recovery(
        arr, _get(adj, "highlight_recovery"), _get(adj, "shadow_recovery")
    )
    arr = _apply_temperature_tint(arr, _get(adj, "temperature"), _get(adj, "tint"), protect_mask)
    arr = _apply_vibrance(arr, _get(adj, "vibrance"), protect_mask)
    arr = _apply_vignette(arr, _get(adj, "vignette"))
    arr = _apply_smoke(arr, _get(adj, "smoke"))

    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")

    img = _apply_noise_reduction(img, _get(adj, "noise_reduction"))

    brightness = _get(adj, "brightness") / 100.0
    contrast = _get(adj, "contrast") / 100.0
    saturation = _get(adj, "saturation") / 100.0
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)

    img = _apply_clarity(img, _get(adj, "clarity"), sharpen_protect, skin_protect)
    img = _apply_sharpness(img, _get(adj, "sharpness"), sharpen_protect, skin_protect)

    img = _apply_local_adjustments(img, _get(adj, "local_adjustments"))

    return img