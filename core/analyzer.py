# core/analyzer.py
#
# PHASE 5 -- Analyzer.
#
# Detects: subject type, portrait/landscape orientation, indoor/outdoor,
# lighting, brightness, contrast, dominant colors, face detected,
# orientation, approximate quality -- exactly the feature list in the
# spec's Phase 5 section. Output is a single JSON-ready dict, meant to
# be fed to Phase 6's Smart Pipeline once that exists (this module does
# NOT make any "apply preset X" recommendation itself -- see the note at
# the bottom of this file for why that's deliberately kept out).
#
# PHASE 5 CLOSEOUT ADDITIONS (this pass):
#   - Dedicated Sky Detection (`_detect_sky`) -- previously the only sky
#     signal in this file was a throwaway cue buried inside
#     `_detect_scene`'s indoor/outdoor vote (see that function's
#     docstring). It answered "is there enough sky up there to call this
#     outdoor" and discarded the number. Highlight protection needs a
#     different question answered -- WHERE is the sky, how much of the
#     frame is it, and is it already at risk of blowing out to flat
#     white -- so that's now its own read, independent of indoor/outdoor.
#   - AI Quality Analysis expanded: `quality` now also reports an
#     exposure score, a dynamic-range score, and one blended overall
#     score/grade, alongside the existing sharpness + noise reads.
#   - AI Lighting Analysis expanded: a new `light_quality` read
#     classifies the CHARACTER of the light (harsh / soft / uneven /
#     even), separate from `lighting` above (which reads overall
#     EXPOSURE: bright/normal/dark/backlit). A photo can be bright AND
#     harsh, or dark AND soft -- these are two different axes, so they
#     stay two different fields rather than overloading `lighting`.
#   - `lighting` and `quality`'s existing keys (sharpness_score, rating,
#     noise_score) are UNCHANGED -- Phase 6's core/smart_pipeline.py
#     reads those by exact value, so nothing here was renamed or
#     repurposed, only added to.
#
# MODULE SPLIT (same rule stated in core/enhancer.py, core/crop.py,
# core/object_remover.py, core/filters.py): core/ holds orchestration +
# classic deterministic image statistics; ai/ holds anything backed by
# an actual trained model. Face detection is the one genuinely
# AI-powered piece of this phase, so it lives in ai/face_detector.py and
# is imported here -- everything else below (lighting, indoor/outdoor,
# dominant colors, quality, orientation) is honest classical image
# processing (histograms, color-space math, k-means color clustering),
# same as Phase 2's Auto Enhance and Phase 4's presets. See the
# "WHERE AI IS ACTUALLY USED" section of the spec doc.

from PIL import Image

import cv2
import numpy as np

from ai.face_detector import detect_faces
from ai.scene_classifier import classify as classify_scene


# ===================== ORIENTATION =====================

def _classify_orientation(width: int, height: int) -> str:
    """Frame orientation from raw pixel dimensions -- not to be confused
    with `subject_type` below (a landscape-oriented photo can still have
    "portrait" as its *subject*, e.g. a tall crop of a person)."""
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


# ===================== LIGHTING =====================

def _classify_lighting(l_channel: np.ndarray) -> tuple:
    """
    Classifies overall lighting from the LAB L (lightness) channel,
    0-255 scale. Returns (lighting_label, brightness_0_100).

    "backlit" is detected separately from plain brightness: it's not
    about the average being dark or bright, it's a photo where the
    subject/center is noticeably darker than the frame's edges (a
    classic silhouette-against-a-bright-window/sky shot) -- mean
    brightness alone can't tell that apart from a normal dim interior,
    so this compares the center region against the outer border.
    """
    mean_l = float(l_channel.mean())
    brightness_0_100 = round(mean_l / 255 * 100)

    h, w = l_channel.shape
    cy0, cy1 = round(h * 0.3), round(h * 0.7)
    cx0, cx1 = round(w * 0.3), round(w * 0.7)
    center = l_channel[cy0:cy1, cx0:cx1]
    # Border = everything outside a slightly larger inset box, sampled
    # via a mask rather than 4 separate slices so odd aspect ratios
    # don't bias the border average toward the longer edge pair.
    border_mask = np.ones_like(l_channel, dtype=bool)
    by0, by1 = round(h * 0.15), round(h * 0.85)
    bx0, bx1 = round(w * 0.15), round(w * 0.85)
    border_mask[by0:by1, bx0:bx1] = False

    center_mean = float(center.mean()) if center.size else mean_l
    border_mean = float(l_channel[border_mask].mean()) if border_mask.any() else mean_l

    if border_mean - center_mean > 45:
        return "backlit", brightness_0_100

    if mean_l >= 175:
        return "bright", brightness_0_100
    if mean_l <= 70:
        return "dark", brightness_0_100
    return "normal", brightness_0_100


def _contrast_score(l_channel: np.ndarray) -> int:
    """0-100 scale, from the L channel's standard deviation. A real
    photo's L std dev rarely exceeds ~75-80 even at maximum contrast, so
    that's used as the top of the scale rather than the theoretical max
    of 127.5 (which nothing realistic ever reaches, making every photo
    read as artificially "low contrast" if used as the divisor)."""
    std = float(l_channel.std())
    return max(0, min(100, round(std / 78 * 100)))


# ===================== LIGHT QUALITY (harsh / soft / uneven) =====================

def _classify_light_quality(l_channel: np.ndarray) -> dict:
    """
    PHASE 5 addition: reads the CHARACTER of the light, not its
    exposure. `_classify_lighting` above already answers "is this photo
    bright/dark/backlit overall" -- that's an EXPOSURE read. This
    answers a different question: is the light itself hard and
    directional (crisp shadow edges, a bright-sun-at-noon look), soft
    and diffused (an overcast day, open shade, a softbox), or falling
    unevenly across the frame (one side lit, one side dark -- a single
    off-camera light or a window lighting only half the room)? A photo
    can be bright-and-harsh or dark-and-soft, so this is intentionally a
    second, independent field rather than more values crammed into
    `lighting`.

    Three classical cues, none of them requiring a trained model:

      1. REGIONAL SPREAD -- split the frame into a 4x4 grid and look at
         how much the block means disagree with each other. Even light
         keeps every block close to the frame average; directional light
         leaves one side clearly brighter than the other.
      2. HARD-EDGE FRACTION -- hard light casts crisp, high-gradient
         shadow boundaries; soft light's shadow edges are gradual. A
         Sobel gradient magnitude map's fraction of very-strong edges is
         a cheap proxy for "how many hard shadow boundaries exist".
      3. BIMODALITY -- hard light typically produces a photo with BOTH
         deep, near-black shadows AND blown, near-white highlights at
         once (not just "high contrast" generally -- a well-lit portrait
         can have high contrast without either extreme). Soft light
         rarely produces either extreme.

    Returns {"label": "harsh"|"soft"|"uneven"|"even",
             "harshness_score": 0-100, "evenness_score": 0-100} --
    both scores are returned (not just the label) so Phase 6 can weigh a
    borderline read instead of trusting a single bucketed word.
    """
    h, w = l_channel.shape

    # ---- 1. Regional spread (evenness) ----
    gy = np.linspace(0, h, 5).astype(int)
    gx = np.linspace(0, w, 5).astype(int)
    block_means = [
        float(l_channel[gy[i]:gy[i + 1], gx[j]:gx[j + 1]].mean())
        for i in range(4) for j in range(4)
        if l_channel[gy[i]:gy[i + 1], gx[j]:gx[j + 1]].size
    ]
    regional_spread = float(np.std(block_means)) if block_means else 0.0

    # ---- 2. Hard shadow-edge fraction ----
    sobel_x = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobel_x, sobel_y)
    hard_edge_frac = float((edge_mag > 90).mean())

    # ---- 3. Bimodality (deep shadow AND blown highlight together) ----
    dark_frac = float((l_channel <= 25).mean())
    bright_frac = float((l_channel >= 230).mean())
    bimodal = dark_frac + bright_frac

    harshness_raw = hard_edge_frac * 55.0 + bimodal * 45.0
    harshness_score = max(0, min(100, round(harshness_raw * 3.2)))
    evenness_score = max(0, min(100, round(100 - min(100.0, regional_spread / 45.0 * 100.0))))

    if regional_spread >= 26 and harshness_score < 35:
        label = "uneven"
    elif harshness_score >= 40:
        label = "harsh"
    elif harshness_score <= 12 and regional_spread < 16:
        label = "soft"
    else:
        label = "even"

    return {"label": label, "harshness_score": harshness_score, "evenness_score": evenness_score}


# ===================== INDOOR / OUTDOOR =====================

def _detect_scene(rgb_np: np.ndarray) -> tuple:
    """
    Returns (label, confidence_0_100, source) where label is
    "indoor" | "outdoor" | "uncertain".

    TWO PATHS, in priority order:

    1. 🤖 ai/scene_classifier.py -- a real (user-installed) ONNX scene
       classifier. Used when a model is present at
       models/scene_classifier.onnx. Nothing is ever downloaded; see that
       module's header.

    2. 🧮 The multi-cue heuristic below. The previous version of this
       function read ONE signal ("is the top of the frame blue?"), which
       misread a blue wall as a sky and any overcast day as indoors. It
       now weighs FOUR independent pieces of evidence for outdoors and
       TWO for indoors, and -- importantly -- returns a CONFIDENCE, so
       Phase 6 can weight a shaky scene read lower instead of treating a
       guess as a fact:

         outdoor evidence
           - sky-blue coverage in the upper third
           - bright, low-saturation "open air / overcast" coverage up there
           - greenery (foliage/grass hue band) anywhere in the frame
           - a bright-top / darker-bottom luminance gradient (sky above ground)
         indoor evidence
           - a warm red-over-blue cast across the frame (tungsten/LED)
           - an even, mid-tone, low-saturation frame with no sky signal
             at all (walls and ceilings under artificial light)

    Still a heuristic, still reported as "uncertain" when the evidence is
    genuinely split -- that's more useful downstream than a confident
    coin-flip.
    """
    classified = classify_scene(rgb_np)
    if classified:
        return classified["label"], classified["confidence"], "classifier"

    # Work on a small copy -- every fraction below is a coverage ratio,
    # so resolution changes nothing except the CPU bill.
    small = cv2.resize(rgb_np, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    upper = slice(0, 90)          # top ~35% -- where sky lives
    lower = slice(154, 256)       # bottom ~40% -- ground/floor

    sky_blue = float(((hue[upper] >= 95) & (hue[upper] <= 135)
                      & (sat[upper] >= 40) & (val[upper] >= 90)).mean())
    open_air = float(((val[upper] >= 190) & (sat[upper] <= 60)).mean())
    greenery = float(((hue >= 30) & (hue <= 90) & (sat >= 60) & (val >= 40)).mean())
    gradient = float(val[upper].mean() - val[lower].mean())
    gradient_cue = max(0.0, min(1.0, gradient / 55.0))

    r_mean = float(small[:, :, 0].mean())
    b_mean = float(small[:, :, 2].mean())
    warm_cast = max(0.0, min(1.0, (r_mean - b_mean) / 30.0))
    even_midtones = float(((val >= 45) & (val <= 195) & (sat <= 95)).mean())
    no_sky = 1.0 - min(1.0, (sky_blue + open_air) / 0.10)

    outdoor_score = (sky_blue * 2.2) + (open_air * 1.0) + (greenery * 1.3) + (gradient_cue * 0.5)
    indoor_score = (warm_cast * 0.9) + (even_midtones * no_sky * 0.9)

    margin = outdoor_score - indoor_score
    if margin >= 0.30:
        return "outdoor", int(round(min(95, 55 + margin * 45))), "heuristic"
    if -margin >= 0.30:
        return "indoor", int(round(min(95, 55 + (-margin) * 45))), "heuristic"
    return "uncertain", int(round(35 + abs(margin) * 40)), "heuristic"


# ===================== SKY DETECTION (dedicated) =====================

def _detect_sky(rgb_np: np.ndarray) -> dict:
    """
    PHASE 5 addition: a DEDICATED sky read, kept deliberately separate
    from the sky-blue cue buried inside `_detect_scene` above. That cue
    exists only to help decide indoor-vs-outdoor and is thrown away
    afterward -- it never reports where the sky is, how much of the
    frame it covers, or whether it's in good shape. This function exists
    for a different downstream consumer: HIGHLIGHT PROTECTION. A blown,
    flat-white sky is the single most common highlight-clipping problem
    in outdoor photography, and core/enhancer.py's `highlight_recovery`
    slider (see that module's header) is built to fix exactly that --
    but nothing upstream currently tells it (or Phase 6's Smart
    Pipeline) whether a sky is even present, let alone whether it's
    already clipped. This is that signal.

    Looks at the upper ~43% of the frame (a generous band -- wide-angle
    and horizon-heavy compositions can carry sky lower than the top
    third `_detect_scene` uses) for three sky appearances:
      - clear blue sky (hue/saturation/brightness band)
      - bright, low-saturation overcast/hazy sky
      - warm sunset/sunrise sky (orange-to-pink hue band)
    and unions them into one sky mask. Within that mask, it separately
    measures how much is ALREADY near-white and desaturated -- the
    textureless "blown sky" pattern -- as the highlight-risk read.

    Returns:
        {
            "present": bool,                # sky covers a meaningful share of the frame
            "coverage_percent": 0-100,       # sky pixels as a % of the WHOLE frame
            "confidence": 0-100,             # how sure this heuristic is
            "highlight_risk": bool,          # sky region already looks clipped
            "clipped_percent": 0-100,        # % of the detected sky that's near-white/flat
        }

    Still a classical heuristic (no trained model), same honesty note as
    `_detect_scene` -- it can miss unusual compositions (sky reflected in
    water, a sky-colored backdrop) and that's fine: `present: false` with
    a low confidence is a legitimate, useful answer, not a failure.
    """
    small = cv2.resize(rgb_np, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    band_h = 110  # top ~43% of the 256px working copy
    upper = slice(0, band_h)
    hue_u, sat_u, val_u = hue[upper], sat[upper], val[upper]

    blue_sky = (hue_u >= 95) & (hue_u <= 135) & (sat_u >= 35) & (val_u >= 80)
    overcast_sky = (val_u >= 195) & (sat_u <= 55)
    sunset_sky = (hue_u <= 30) & (sat_u >= 60) & (val_u >= 120)
    sky_mask_upper = blue_sky | overcast_sky | sunset_sky

    band_coverage = float(sky_mask_upper.mean())  # fraction of the SEARCH BAND
    frame_coverage = band_coverage * (band_h / small.shape[0])  # rescaled to whole-frame %

    present = frame_coverage >= 0.05  # ~5% of the whole frame reads as sky
    confidence = int(round(min(95, 30 + band_coverage * 130))) if present else int(round(band_coverage * 90))

    if sky_mask_upper.any():
        sky_val = val_u[sky_mask_upper]
        sky_sat = sat_u[sky_mask_upper]
        clipped_frac = float(((sky_val >= 250) & (sky_sat <= 20)).mean())
    else:
        clipped_frac = 0.0

    return {
        "present": present,
        "coverage_percent": round(frame_coverage * 100, 1),
        "confidence": confidence,
        "highlight_risk": present and clipped_frac >= 0.20,
        "clipped_percent": round(clipped_frac * 100, 1),
    }


# ===================== COLOUR STRENGTH =====================

def _saturation_score(rgb_np: np.ndarray) -> int:
    """
    How colourful the photo actually is, 0-100, from the HSV S channel's
    mean weighted toward pixels bright enough for their colour to read
    (near-black pixels have meaningless hue/saturation and would drag a
    plain mean down on every night shot).

    Phase 6 needs this for three separate decisions -- "colours look
    dull", "colours already very strong" (so don't push further), and
    "this is basically monochrome already" -- none of which the old
    dominant-colour list could answer reliably.
    """
    small = cv2.resize(rgb_np, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    weights = np.clip((val - 25.0) / 205.0, 0.0, 1.0)
    total = float(weights.sum())
    if total < 1.0:
        return 0
    weighted_mean = float((sat * weights).sum() / total)
    return max(0, min(100, round(weighted_mean / 255 * 100)))


# ===================== DOMINANT COLORS =====================

def _dominant_colors(rgb_np: np.ndarray, k: int = 5) -> list:
    """
    Top `k` dominant colors via k-means clustering (cv2.kmeans) on a
    downsized copy of the photo -- clustering the full-res pixel array
    is wasted CPU for a color summary and slow on the target hardware
    for no visible difference in the result. Returns a list of
    {"hex": "#rrggbb", "percent": float} sorted largest cluster first.
    """
    small = cv2.resize(rgb_np, (120, 120), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    k = max(1, min(k, len(pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)

    total = len(labels)
    colors = []
    for idx in order:
        r, g, b = (int(round(c)) for c in centers[idx])
        colors.append(
            {
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "percent": round(float(counts[idx]) / total * 100, 1),
            }
        )
    return colors


# ===================== QUALITY / SHARPNESS =====================

def _exposure_score(l_channel: np.ndarray) -> int:
    """
    PHASE 5 addition: 0-100, 100 = well exposed. Distinct from
    `brightness` (a plain 0-100 mean reading) in that this specifically
    penalizes CLIPPING -- actual lost detail at either end of the range
    -- rather than just "the photo happens to be dark or bright". A
    deliberately dark, moody photo with no clipped pixels should NOT
    score badly here even though its `brightness` reads low; a
    snapshot with a big blown-out window should score badly here even
    if its average brightness looks unremarkable.

    Two components:
      - clip_penalty: fraction of pixels sitting at the extreme floor
        (<=3) or ceiling (>=252) of the 0-255 range -- true clipping,
        not just "dark" or "bright" pixels.
      - center_penalty: a gentle, secondary penalty for the frame mean
        drifting far from a broad well-exposed middle band -- real
        photos are "well exposed" across a wide range, so this is a
        soft distance term, not a hard target of 128.
    """
    clipped_shadow = float((l_channel <= 3).mean())
    clipped_highlight = float((l_channel >= 252).mean())
    clip_penalty = (clipped_shadow + clipped_highlight) * 140.0

    mean_l = float(l_channel.mean())
    center_penalty = abs(mean_l - 128.0) / 128.0 * 35.0

    return max(0, min(100, round(100.0 - clip_penalty - center_penalty)))


def _dynamic_range_score(l_channel: np.ndarray) -> int:
    """
    PHASE 5 addition: 0-100, how much of the available tonal range the
    photo actually uses, from the 5th-to-95th-percentile spread of the
    L channel (percentiles rather than raw min/max so a handful of
    stray noise pixels at either extreme can't fake a wide range out of
    an otherwise flat photo). A hazy, low-contrast, or heavily backlit-
    and-crushed photo uses only a narrow slice of the 0-255 scale even
    when it isn't technically clipped; a photo with rich, separated
    tones from shadow to highlight uses nearly all of it. ~200 of
    percentile spread is already about as wide as a normal, healthy
    photo gets, so that's the top of the scale rather than the
    theoretical max of 255 (which would make every real photo read as
    artificially "low range").
    """
    p5, p95 = np.percentile(l_channel, [5, 95])
    spread = float(p95 - p5)
    return max(0, min(100, round(spread / 200.0 * 100.0)))


def _quality(gray: np.ndarray, l_channel: np.ndarray, contrast: int) -> dict:
    """
    Same Laplacian-variance blur read used by Phase 2's Auto Enhance
    (bridge.py::autoEnhanceAnalyze), factored out here as its own
    reusable reading rather than re-deriving sharpness sliders --
    Analyzer only needs a rating, not a slider value.

    Also returns a NOISE score (0-100). Phase 6 needs it to answer
    "should Denoise run?" honestly: without a measured noise number the
    only options are to always denoise (which softens clean photos) or
    never denoise (which leaves night shots grainy). The measurement is a
    median-blur residual -- median filtering removes speckle while
    preserving edges, so what's left over is mostly noise rather than
    detail. Deliberately separate from the sharpness read: a photo can be
    sharp AND noisy (high ISO), and treating those as one axis is exactly
    how "sharpen + denoise fighting each other" bugs happen.

    PHASE 5 EXPANSION -- three more reads, folded into this same dict
    since they're all "how good is this photo, technically" questions:
      - exposure_score (see `_exposure_score`)
      - dynamic_range (see `_dynamic_range_score`)
      - overall_score / grade -- a single blended number + label, for
        the Analyzer's "final output" summary view, weighted toward the
        reads that matter most for a usable photo (sharpness and
        exposure) without letting any one weak read alone tank the
        score the way a strict minimum would.
    """
    resized = cv2.resize(gray, (512, 512))
    laplacian_var = float(cv2.Laplacian(resized, cv2.CV_64F).var())
    score = max(0, min(100, round(laplacian_var / 300 * 100)))

    if laplacian_var >= 120:
        rating = "sharp"
    elif laplacian_var >= 40:
        rating = "soft"
    else:
        rating = "blurry"

    residual = resized.astype(np.float32) - cv2.medianBlur(resized, 3).astype(np.float32)
    # ~6.0 of residual std is already visibly grainy on screen, so that's
    # the top of the scale rather than a theoretical maximum nothing hits.
    noise_score = max(0, min(100, round(float(residual.std()) / 6.0 * 100)))

    exposure_score = _exposure_score(l_channel)
    dynamic_range = _dynamic_range_score(l_channel)

    overall_score = round(
        score * 0.30
        + (100 - noise_score) * 0.15
        + exposure_score * 0.30
        + dynamic_range * 0.10
        + contrast * 0.15
    )
    overall_score = max(0, min(100, overall_score))

    if overall_score >= 80:
        grade = "excellent"
    elif overall_score >= 60:
        grade = "good"
    elif overall_score >= 40:
        grade = "fair"
    else:
        grade = "poor"

    return {
        "sharpness_score": score,
        "rating": rating,
        "noise_score": noise_score,
        "exposure_score": exposure_score,
        "dynamic_range": dynamic_range,
        "overall_score": overall_score,
        "grade": grade,
    }


def _background_clutter(rgb_np: np.ndarray, faces: list) -> dict:
    """
    PHASE 5/6 addition: a classical (no model) proxy for "would this
    photo benefit from Remove Background?" -- used by Phase 6's Smart
    Pipeline to SUGGEST the Remove BG tool for busy-background product/
    object shots, rather than guessing blindly.

    There's no trained subject/product detector here (that's a real
    model and out of this phase's scope -- see the module header), so
    this reads the OUTER RING of the frame only (the 12% border, which
    on a typical product/object photo is background, not subject) and
    scores two classical signals there:
      - edge density (Sobel magnitude)   -- high = visually busy/cluttered
      - color variation (channel std)    -- high = not a plain/seamless backdrop
    Both are combined into one 0-100 "clutter" score. A face-detected
    photo is excluded from the caller's decision (portraits have their
    own treatment), not from this measurement itself.
    """
    h, w = rgb_np.shape[:2]
    border = max(4, int(min(h, w) * 0.12))

    # Build a boolean ring mask: True in the outer border, False in the
    # center -- cheaper and clearer than slicing four separate strips.
    ring = np.zeros((h, w), dtype=bool)
    ring[:border, :] = True
    ring[-border:, :] = True
    ring[:, :border] = True
    ring[:, -border:] = True

    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobel_x, sobel_y)

    ring_edges = edge_mag[ring]
    ring_pixels = rgb_np[ring]

    edge_score = max(0.0, min(100.0, float(ring_edges.mean()) / 45.0 * 100.0))
    color_std = float(ring_pixels.reshape(-1, 3).std(axis=0).mean())
    color_score = max(0.0, min(100.0, color_std / 55.0 * 100.0))

    clutter = round(edge_score * 0.6 + color_score * 0.4)
    return {
        "clutter_score": clutter,
        "busy": clutter >= 55 and not faces,
    }


# ===================== SUBJECT TYPE =====================

def _classify_subject(faces: list, orientation: str, indoor_outdoor: str) -> str:
    """
    Best-effort label for what the *photo is of*, distinct from its
    pixel `orientation` above. Rule order: a face large enough to be the
    clear subject wins first (portrait), then falls back to the
    indoor/outdoor read plus frame orientation for a landscape/scene
    guess, and otherwise reports "general" rather than forcing a guess
    the data doesn't support.
    """
    if faces:
        largest_area = max(f["w"] * f["h"] for f in faces)
        if largest_area >= 0.03 or len(faces) >= 2:
            return "portrait"

    if indoor_outdoor == "outdoor" and orientation in ("landscape", "square"):
        return "landscape_scene"

    return "general"


# ===================== SUMMARY TEXT =====================

def _build_summary(result: dict) -> str:
    bits = []

    if result["face_count"] == 1:
        bits.append("one face detected")
    elif result["face_count"] > 1:
        bits.append(f"{result['face_count']} faces detected")

    if result["subject_type"] == "portrait":
        bits.append("looks like a portrait")
    elif result["subject_type"] == "landscape_scene":
        bits.append("looks like an outdoor/landscape scene")

    if result["indoor_outdoor"] != "uncertain":
        bits.append(result["indoor_outdoor"])

    lighting_phrases = {
        "bright": "brightly lit",
        "dark": "dimly lit",
        "backlit": "backlit (subject darker than the background)",
        "normal": "evenly lit",
    }
    bits.append(lighting_phrases[result["lighting"]])

    light_quality_phrases = {
        "harsh": "harsh light",
        "soft": "soft, diffused light",
        "uneven": "unevenly lit across the frame",
    }
    lq = result["light_quality"]["label"]
    if lq in light_quality_phrases:
        bits.append(light_quality_phrases[lq])

    bits.append(f"{result['quality']['rating']} focus")

    if result["sky"]["present"] and result["sky"]["highlight_risk"]:
        bits.append("sky looks blown out")

    if not bits:
        return "Analysis complete."
    return "Photo analysis: " + ", ".join(bits) + f" ({result['quality']['grade']} overall)."


# ===================== PUBLIC ENTRY POINT =====================

def analyze_image(image: Image.Image) -> dict:
    """
    Runs the full Phase 5 analysis pipeline on `image` and returns a
    single JSON-ready dict:

        {
            "width": int, "height": int, "megapixels": float,
            "orientation": "landscape"|"portrait"|"square",
            "face_detected": bool, "face_count": int,
            "faces": [{"x","y","w","h"}, ...]  # 0-1 fractions of the frame
            "subject_type": "portrait"|"landscape_scene"|"general",
            "indoor_outdoor": "indoor"|"outdoor"|"uncertain",
            "lighting": "bright"|"normal"|"dark"|"backlit",
            "light_quality": {                                  # PHASE 5 addition
                "label": "harsh"|"soft"|"uneven"|"even",
                "harshness_score": 0-100, "evenness_score": 0-100,
            },
            "brightness": 0-100, "contrast": 0-100,
            "dominant_colors": [{"hex","percent"}, ...],  # largest first
            "quality": {                                        # PHASE 5 expanded
                "sharpness_score": 0-100, "rating": "sharp"|"soft"|"blurry",
                "noise_score": 0-100,
                "exposure_score": 0-100,
                "dynamic_range": 0-100,
                "overall_score": 0-100,
                "grade": "excellent"|"good"|"fair"|"poor",
            },
            "sky": {                                            # PHASE 5 addition (dedicated)
                "present": bool, "coverage_percent": 0-100,
                "confidence": 0-100, "highlight_risk": bool,
                "clipped_percent": 0-100,
            },
            "background": {"clutter_score": 0-100, "busy": bool},  # PHASE 6: Remove BG suggestion input
            "summary": str,
        }

    Pure function, no I/O -- callers (ui/bridge.py::analyzeImageAsync)
    own opening the file and turning this into JSON.
    """
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    rgb_np = np.array(rgb)

    if rgb_np.size == 0:
        raise ValueError("Empty image.")

    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

    orientation = _classify_orientation(image.width, image.height)
    lighting, brightness = _classify_lighting(l_channel)
    light_quality = _classify_light_quality(l_channel)
    contrast = _contrast_score(l_channel)
    indoor_outdoor, scene_confidence, scene_source = _detect_scene(rgb_np)
    sky = _detect_sky(rgb_np)
    saturation = _saturation_score(rgb_np)
    dominant_colors = _dominant_colors(rgb_np, k=5)
    quality = _quality(gray, l_channel, contrast)

    # Face detection (ai/face_detector.py) -- runs on a version of the
    # image already resized internally by that module, so no extra
    # downscale needed here.
    faces = detect_faces(rgb)
    subject_type = _classify_subject(faces, orientation, indoor_outdoor)
    background = _background_clutter(rgb_np, faces)

    result = {
        "width": image.width,
        "height": image.height,
        "megapixels": round(image.width * image.height / 1_000_000, 1),
        "orientation": orientation,
        "face_detected": len(faces) > 0,
        "face_count": len(faces),
        "faces": faces,
        "subject_type": subject_type,
        "indoor_outdoor": indoor_outdoor,
        "scene_confidence": scene_confidence,
        "scene_source": scene_source,
        "lighting": lighting,
        "light_quality": light_quality,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "dominant_colors": dominant_colors,
        "quality": quality,
        "sky": sky,
        "background": background,
    }
    result["summary"] = _build_summary(result)
    return result


# ===================== SCOPE NOTE (spec Phase 5 vs. Phase 6) =====================
#
# This module deliberately stops at DESCRIBING the photo. It does not
# recommend a preset, adjustment values, or a processing pipeline --
# that's explicitly Phase 6's job (core/smart_pipeline.py, still a
# stub): "Takes analyzer output -> recommends a processing pipeline...
# User can always override." Folding recommendation logic in here would
# duplicate Phase 6 before it exists and make this module's output
# harder to reuse once Phase 6 lands (it would have to unpick
# recommendation logic back out of the analysis). Phase 4's deferred
# "Auto Filter" (core/filters.py) and Phase 2/3's "true subject/scene-
# aware enhancement" note both point at this module's output as their
# eventual input -- both are still correctly unblocked-but-not-built,
# same as before this phase started.
#
# Same rule applies to this pass's new `sky` field: it reports facts
# (present, coverage, confidence, highlight risk) and stops there. It
# does NOT set core/enhancer.py's `highlight_recovery` slider itself --
# that decision (how much recovery, and whether to suggest it at all)
# belongs to Phase 6's core/smart_pipeline.py, same separation of
# concerns as every other field in this file.
#
# OBJECT DETECTION NOTE (unchanged from the spec): Person/Car/Animal/etc.
# detection remains deliberately out of scope for this phase -- see the
# top-level project notes. Nothing in this pass adds it.