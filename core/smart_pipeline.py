# core/smart_pipeline.py
#
# PHASE 6 -- Smart Pipeline (v2: a real recipe builder, not a filter picker).
#
# WHAT CHANGED AND WHY
# --------------------
# v1 of this module was, honestly, a "Smart Filter Picker": a first-match-wins
# rule table that returned ONE preset name plus a HARD-CODED intensity
# (90%, 95%, 80%...). It never looked at what was actually wrong with the
# photo, never explained itself, never skipped work that wasn't needed,
# and gave every image of a given category the same strength. That's a
# lookup table wearing a trench coat.
#
# v2 does the thing the name promises:
#
#   Analyzer output          (Phase 5 -- core/analyzer.py)
#        |
#        v
#   1. UNDERSTAND   -> normalized cues (brightness/contrast/sharpness/
#                      saturation/noise/scene/faces)
#        |
#        v
#   2. DIAGNOSE     -> problems (with severity + the number behind them)
#                      AND strengths (what is already good, i.e. what we
#                      must NOT touch)
#        |
#        v
#   3. CHOOSE LOOK  -> every candidate look scored 0-100 against the
#                      cues, with reasons for AND against, so the user
#                      sees alternatives, not a verdict
#        |
#        v
#   4. BUILD RECIPE -> an explicit, ordered, per-step plan. Each step is
#                      either "apply, and here's the value" or "SKIP, and
#                      here's why" (already sharp / no face / no noise).
#        |
#        v
#   5. ADAPT        -> intensity computed from how much this photo
#                      actually needs, per image (not a constant)
#        |
#        v
#   6. SAFETY CHECK -> clipping, crushed shadows, over-saturation,
#                      over-sharpening, denoise+sharpen conflict
#        |
#        v
#   7. PREVIEW / CUSTOMIZE / APPLY
#
# WHERE THE AI IS (and isn't)
# ---------------------------
# There is deliberately NO model in this file. The layering the spec
# asks for is:
#
#   Phase 5  ai/face_detector.py + ai/scene_classifier.py  -> perception
#   Phase 6  THIS FILE (rules over Phase 5's numbers)      -> decision
#   Phase 2/3 core/enhancer.py                             -> pixels
#   Phase 4  core/filters.py                               -> base looks
#
# Adding a heavy model here would be the wrong place: better perception
# (Phase 5) automatically makes this file smarter for free, while a model
# here would just be a slower way to write an if-statement.
#
# BACKWARD COMPATIBILITY: recommend_pipeline() still returns every key v1
# returned (id, name, preset_id, preset_name, intensity, reason, signals,
# confidence, adjustments, monochrome, grain), so existing call sites in
# frontend/editor.js and frontend/filters.js keep working untouched. The
# v2 data (problems, strengths, recipe, safety, alternatives, match_score,
# explanation) is added alongside them.

from PIL import Image

from core.enhancer import DEFAULT_ADJUSTMENTS, apply_adjustments
from core.filters import add_grain, get_preset, to_monochrome

# Safe ranges for every adjustment key, mirrored from core/enhancer.py's
# header docstring (the single source of truth for what each slider
# means). Anything this module computes -- a preset's own values, this
# module's recipe deltas, or the two combined -- is clamped to these
# ranges before it is ever returned, so a recommended pipeline can never
# push a value further than a manual slider drag already could.
_ADJUSTMENT_RANGES = {
    "brightness": (0, 200),
    "contrast": (0, 200),
    "saturation": (0, 200),
    "exposure": (-100, 100),
    "highlights": (-100, 100),
    "shadows": (-100, 100),
    "whites": (-100, 100),
    "blacks": (-100, 100),
    "temperature": (-100, 100),
    "tint": (-100, 100),
    "vibrance": (-100, 100),
    "sharpness": (0, 100),
    "clarity": (0, 100),
    "noise_reduction": (0, 100),
    "vignette": (0, 100),
}

# core/enhancer.py::_apply_exposure does stops = (exposure / 100) * 1.2,
# so this converts the internal 0-100 slider unit into the EV number
# photographers actually recognize ("Exposure +0.18") for display.
_STOPS_PER_UNIT = 1.2 / 100.0

# Severity -> how much weight a problem carries when computing intensity.
_SEVERITY_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}


def _clamp(key, value):
    lo, hi = _ADJUSTMENT_RANGES[key]
    return max(lo, min(hi, value))


def _clamp_all(adjustments: dict) -> dict:
    return {k: _clamp(k, v) for k, v in adjustments.items() if k in _ADJUSTMENT_RANGES}


def _merge(base: dict, extra: dict) -> dict:
    """
    Merges recipe deltas onto a base adjustment dict. Multiplicative keys
    (brightness/contrast/saturation, where 100 = no change) accumulate
    around 100; additive keys (everything else, where 0 = no change)
    simply add. Same rule as v1 -- getting this backwards is how a
    "+2 contrast" nudge silently becomes "contrast = 2", i.e. a
    near-flat image.
    """
    out = dict(base)
    for key, value in (extra or {}).items():
        if key not in _ADJUSTMENT_RANGES:
            continue
        if DEFAULT_ADJUSTMENTS[key] == 100:
            out[key] = out.get(key, 100) + (value - 100)
        else:
            out[key] = out.get(key, 0) + value
    return _clamp_all(out)


def _resolve_preset_adjustments(preset: dict, intensity: float) -> dict:
    """
    Blends `preset`'s adjustment overrides toward DEFAULT_ADJUSTMENTS by
    `intensity` (0-100) -- identical maths to core/filters.py's
    _scaled_adjustments, so a pipeline preview and the Filters view's own
    preview of the same preset at the same intensity match exactly.
    """
    t = max(0.0, min(1.0, intensity / 100.0))
    out = dict(DEFAULT_ADJUSTMENTS)
    for key, preset_val in (preset.get("adjustments") or {}).items():
        if key not in DEFAULT_ADJUSTMENTS:
            continue
        default_val = DEFAULT_ADJUSTMENTS[key]
        out[key] = default_val + (preset_val - default_val) * t
    return out


# ===================== 1. UNDERSTAND: normalized cues =====================

def _cues(analysis: dict) -> dict:
    """
    Flattens Analyzer output into the handful of numbers every rule below
    reasons about. Everything is 0-100 so thresholds read the same way
    across cues, and every field has a defined fallback -- an older
    analysis dict (or one from a build without the newer saturation/noise
    readings) must never make this module raise.
    """
    quality = analysis.get("quality") or {}
    brightness = float(analysis.get("brightness", 50) or 0)
    contrast = float(analysis.get("contrast", 50) or 0)
    sharpness = float(quality.get("sharpness_score", 50) or 0)
    saturation = analysis.get("saturation")
    if saturation is None:
        saturation = _estimate_saturation(analysis)
    noise = quality.get("noise_score")
    if noise is None:
        # No measured noise reading available: dark photos are the ones
        # that are usually noisy, so fall back to a conservative read
        # from brightness rather than claiming "quality already good".
        noise = 55.0 if brightness < 30 else (35.0 if brightness < 45 else 20.0)

    lighting = analysis.get("lighting", "normal")
    scene = analysis.get("indoor_outdoor", "uncertain")
    scene_confidence = analysis.get("scene_confidence")
    if scene_confidence is None:
        scene_confidence = 50 if scene == "uncertain" else 75

    face_count = int(analysis.get("face_count", 0) or 0)
    subject = analysis.get("subject_type", "general")

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": float(sharpness),
        "saturation": float(saturation),
        "noise": float(noise),
        "lighting": lighting,
        "scene": scene,
        "scene_confidence": float(scene_confidence),
        "subject": subject,
        "face_count": face_count,
        "has_face": face_count > 0,
        "is_portrait_subject": subject == "portrait",
        "is_landscape_subject": subject == "landscape_scene",
        "orientation": analysis.get("orientation", "landscape"),
        "rating": quality.get("rating", "soft"),
        "dark": brightness < 45 or lighting == "dark",
        "bright": brightness > 78 or lighting == "bright",
        "backlit": lighting == "backlit",
        "outdoor": scene == "outdoor",
        "indoor": scene == "indoor",
        "flat": contrast < 45,
        "contrasty": contrast >= 60,
        "soft": float(sharpness) < 40,
        "already_sharp": float(sharpness) >= 55,
        "muted": float(saturation) < 30,
        "near_monochrome": float(saturation) < 10,
        "noisy": float(noise) >= 45,
        "clean": float(noise) < 30,
        # PHASE 6 addition: Remove BG suggestion input (core/analyzer.py::_background_clutter).
        # Older analysis dicts (pre-upgrade) won't have "background" -- default to "not busy"
        # rather than ever suggesting Remove BG off missing data.
        "background_clutter": float((analysis.get("background") or {}).get("clutter_score", 0) or 0),
        "background_busy": bool((analysis.get("background") or {}).get("busy", False)),
        "is_general_subject": subject == "general",
    }


def _estimate_saturation(analysis: dict) -> float:
    """
    Fallback colourfulness read for analysis dicts that predate the
    Analyzer's own `saturation` field: derived from how far the dominant
    colour clusters sit from grey. Rough, but good enough to keep the
    "muted / near-monochrome" branches from misfiring.
    """
    colors = analysis.get("dominant_colors") or []
    if not colors:
        return 45.0
    total_weight = 0.0
    total_sat = 0.0
    for color in colors:
        hex_value = (color.get("hex") or "").lstrip("#")
        if len(hex_value) != 6:
            continue
        try:
            r, g, b = (int(hex_value[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            continue
        high, low = max(r, g, b), min(r, g, b)
        sat = 0.0 if high == 0 else (high - low) / high * 100.0
        weight = float(color.get("percent", 1) or 1)
        total_sat += sat * weight
        total_weight += weight
    if not total_weight:
        return 45.0
    return round(total_sat / total_weight, 1)


# ===================== 2. DIAGNOSE: problems + strengths =====================

def detect_problems(analysis: dict) -> dict:
    """
    Reads the cues and reports WHAT IS WRONG with this specific photo,
    plus -- just as important -- WHAT IS ALREADY GOOD.

    The "strengths" half is what makes auto-skip honest: the pipeline
    skips sharpening because the photo measured 68/100 sharp, not because
    a rule row happened to omit sharpening.
    """
    c = _cues(analysis)
    problems = []
    strengths = []

    # ---- Exposure ----
    if c["brightness"] < 30:
        problems.append(_problem("exposure", "Severely underexposed", "high",
                                 f"brightness {round(c['brightness'])}/100"))
    elif c["brightness"] < 45:
        problems.append(_problem("exposure", "Underexposed", "high",
                                 f"brightness {round(c['brightness'])}/100"))
    elif c["brightness"] < 52:
        problems.append(_problem("exposure", "Slightly dark", "low",
                                 f"brightness {round(c['brightness'])}/100"))
    elif c["brightness"] > 82:
        problems.append(_problem("exposure_high", "Overexposed", "medium",
                                 f"brightness {round(c['brightness'])}/100"))
    else:
        strengths.append(_strength("exposure", "Exposure already good",
                                   f"brightness {round(c['brightness'])}/100"))

    # ---- Shadows / backlight ----
    if c["backlit"]:
        problems.append(_problem("shadows", "Subject darker than the background", "high",
                                 "backlit scene detected"))
    elif c["brightness"] < 50:
        problems.append(_problem("shadows", "Shadow detail is buried", "medium",
                                 f"brightness {round(c['brightness'])}/100"))

    # ---- Highlights ----
    if c["bright"] and c["contrasty"]:
        problems.append(_problem("highlights", "Highlights close to clipping", "medium",
                                 f"brightness {round(c['brightness'])}/100"))

    # ---- Contrast ----
    if c["contrast"] < 32:
        problems.append(_problem("contrast", "Very flat / hazy", "high",
                                 f"contrast {round(c['contrast'])}/100"))
    elif c["contrast"] < 45:
        problems.append(_problem("contrast", "Low contrast", "medium",
                                 f"contrast {round(c['contrast'])}/100"))
    else:
        strengths.append(_strength("contrast", "Contrast already good",
                                   f"contrast {round(c['contrast'])}/100"))

    # ---- Colour ----
    if c["near_monochrome"]:
        strengths.append(_strength("color", "Image is already near-monochrome",
                                   f"saturation {round(c['saturation'])}/100"))
    elif c["muted"]:
        problems.append(_problem("color", "Colours look dull", "medium",
                                 f"saturation {round(c['saturation'])}/100"))
    elif c["saturation"] > 72:
        problems.append(_problem("color_high", "Colours already very strong", "low",
                                 f"saturation {round(c['saturation'])}/100"))
    else:
        strengths.append(_strength("color", "Colour balance looks natural",
                                   f"saturation {round(c['saturation'])}/100"))

    # ---- Detail ----
    if c["sharpness"] < 18:
        problems.append(_problem("sharpness", "Looks blurry", "high",
                                 f"sharpness {round(c['sharpness'])}/100"))
    elif c["sharpness"] < 40:
        problems.append(_problem("sharpness", "Slightly soft", "medium",
                                 f"sharpness {round(c['sharpness'])}/100"))
    else:
        strengths.append(_strength("sharpness", "Detail is already sharp",
                                   f"sharpness {round(c['sharpness'])}/100"))

    # ---- Noise ----
    if c["noise"] >= 60:
        problems.append(_problem("noise", "Heavy noise / grain", "high",
                                 f"noise {round(c['noise'])}/100"))
    elif c["noise"] >= 45:
        problems.append(_problem("noise", "Visible noise", "medium",
                                 f"noise {round(c['noise'])}/100"))
    else:
        strengths.append(_strength("noise", "Quality already good",
                                   f"noise {round(c['noise'])}/100"))

    # ---- Background (Remove BG suggestion, not an intensity-affecting "problem") ----
    if c["background_busy"] and c["is_general_subject"] and not c["has_face"]:
        problems.append(_problem("background", "Busy/cluttered background", "low",
                                 f"background clutter {round(c['background_clutter'])}/100 -- "
                                 "looks like a product/object shot with a messy backdrop"))

    return {"problems": problems, "strengths": strengths, "cues": c}


def _problem(pid, label, severity, detail):
    return {"id": pid, "label": label, "severity": severity, "detail": detail}


def _strength(sid, label, detail):
    return {"id": sid, "label": label, "detail": detail}


def _problem_load(problems: list) -> float:
    """
    0..1 measure of "how much work does this photo actually need", used
    to make intensity adaptive per image instead of a constant.
    """
    total = sum(_SEVERITY_WEIGHT.get(p["severity"], 0.3) for p in problems)
    return max(0.0, min(1.0, total / 2.6))


# ===================== 3. CHOOSE LOOK: scored candidates =====================
#
# A "look" is a GOAL (what should this photo end up feeling like), not a
# filter. Each one names a Phase 4 preset as its BASE LOOK and adds its
# own stylistic bias on top of whatever corrections the recipe decides
# the photo needs. The scoring cues below each return 0..1 ("how well
# does this photo fit this look"), are weighted, and are turned into a
# 0-100 match score -- so the user gets a ranked shortlist with reasons,
# not a single take-it-or-leave-it verdict.

def _fit(value, good, ok=0.6, poor=0.3):
    """Tiny helper: True -> 1.0, else a graded partial fit."""
    return 1.0 if value else (ok if good else poor)


LOOKS = [
    {
        "id": "cinematic_landscape",
        "name": "Cinematic Landscape",
        "preset_id": "cinematic",
        "base_intensity": 64,
        "goal": "A wide, filmic outdoor grade -- deep shadows, controlled highlights, rich but believable colour.",
        "bias": {"contrast": 102, "highlights": -6, "clarity": 8, "vibrance": 6, "vignette": 10},
        "cues": [
            ("scene", 2.0, lambda c: 1.0 if (c["is_landscape_subject"] or c["outdoor"]) else 0.3,
             "outdoor / landscape scene", "not an outdoor scene"),
            ("no_face", 1.4, lambda c: 0.25 if c["is_portrait_subject"] else 1.0,
             "no face to protect, so the grade can be pushed", "portrait subject -- skin tones limit how far the grade can go"),
            ("mood", 1.6, lambda c: 1.0 if c["brightness"] < 52 else (0.75 if c["brightness"] < 70 else 0.5),
             "darker base tones suit a cinematic grade", "already bright, so there is little mood to build"),
            ("structure", 1.0, lambda c: 1.0 if c["contrasty"] else 0.55,
             "contrast is strong enough to carry a filmic curve", "flat contrast needs rebuilding first"),
            ("detail", 0.8, lambda c: 1.0 if c["sharpness"] >= 45 else 0.5,
             "detail holds up at this size", "soft detail limits the look"),
            ("certainty", 0.9, lambda c: max(0.4, min(1.0, c["scene_confidence"] / 100.0)),
             "scene read with good confidence", "scene type is uncertain"),
        ],
    },
    {
        "id": "travel_vibrant",
        "name": "Travel Vibrant",
        "preset_id": "travel",
        "base_intensity": 68,
        "goal": "Punchy, postcard-ready colour and clarity for travel and street shots.",
        "bias": {"vibrance": 8, "clarity": 6, "saturation": 103},
        "cues": [
            ("scene", 1.8, lambda c: 1.0 if (c["outdoor"] or c["is_landscape_subject"]) else 0.4,
             "outdoor subject benefits from a vibrant grade", "indoor scenes rarely suit a travel grade"),
            ("colour_room", 1.5, lambda c: 1.0 if c["saturation"] < 60 else 0.35,
             "colour has room to grow", "colours are already strong -- pushing further risks looking artificial"),
            ("light", 1.2, lambda c: 1.0 if 40 <= c["brightness"] <= 78 else 0.55,
             "well-lit enough for a bright, open look", "lighting is too extreme for a bright grade"),
            ("detail", 1.0, lambda c: 1.0 if c["sharpness"] >= 40 else 0.45,
             "sharp enough for the extra clarity", "clarity would emphasise softness"),
            ("no_face", 0.7, lambda c: 0.45 if c["is_portrait_subject"] else 1.0,
             "no skin tones to keep natural", "saturation this high is unkind to skin"),
        ],
    },
    {
        "id": "natural_outdoor",
        "name": "Natural Outdoor",
        "preset_id": "travel",
        "base_intensity": 42,
        "goal": "A restrained, true-to-life correction: fix the exposure, leave the character alone.",
        "bias": {"vibrance": -4, "saturation": 99, "clarity": 4},
        "cues": [
            ("scene", 1.4, lambda c: 1.0 if (c["outdoor"] or c["is_landscape_subject"]) else 0.5,
             "outdoor scene", "scene is not clearly outdoor"),
            ("already_decent", 1.6, lambda c: 1.0 if (45 <= c["brightness"] <= 78 and c["contrast"] >= 45) else 0.6,
             "the photo is already close -- it only needs a light touch", "the photo needs more correction than a natural look provides"),
            ("honest_colour", 1.2, lambda c: 1.0 if 25 <= c["saturation"] <= 70 else 0.5,
             "colour is believable as-is", "colour is too far off for a hands-off look"),
            ("safe", 0.8, lambda c: 1.0,
             "lowest risk option -- hardest to overcook", ""),
        ],
    },
    {
        "id": "natural_portrait",
        "name": "Natural Portrait",
        "preset_id": "portrait",
        "base_intensity": 62,
        "goal": "Flattering, skin-safe correction: gentle contrast, soft warmth, protected skin tones.",
        "cues": [
            ("face", 2.4, lambda c: 1.0 if c["is_portrait_subject"] else (0.6 if c["has_face"] else 0.1),
             "face detected -- skin tones drive every decision", "no face detected"),
            ("light", 1.3, lambda c: 1.0 if 40 <= c["brightness"] <= 80 else 0.5,
             "workable light on the subject", "lighting on the subject is difficult"),
            ("not_extreme", 1.0, lambda c: 0.5 if c["backlit"] else 1.0,
             "no backlight fighting the subject", "backlit subject needs recovery first"),
            ("clean", 0.8, lambda c: 1.0 if c["clean"] else 0.5,
             "clean enough that skin stays smooth", "noise on skin needs handling first"),
        ],
        "bias": {"highlights": -4, "clarity": 4},
    },
    {
        "id": "indoor_portrait",
        "name": "Indoor Portrait",
        "preset_id": "portrait",
        "base_intensity": 70,
        "goal": "Fix indoor light: lift the subject, cool the artificial-light cast, keep skin natural.",
        "bias": {"temperature": -4, "shadows": 6, "noise_reduction": 8},
        "cues": [
            ("face", 2.2, lambda c: 1.0 if c["is_portrait_subject"] else (0.55 if c["has_face"] else 0.1),
             "face detected", "no face detected"),
            ("indoor", 1.8, lambda c: 1.0 if c["indoor"] else (0.5 if not c["outdoor"] else 0.15),
             "indoor lighting detected", "scene reads as outdoor"),
            ("needs_lift", 1.2, lambda c: 1.0 if c["brightness"] < 62 else 0.5,
             "the subject needs lifting out of the ambient light", "already bright enough indoors"),
        ],
    },
    {
        "id": "night_portrait",
        "name": "Night Portrait",
        "preset_id": "night",
        "base_intensity": 76,
        "goal": "Rescue a low-light portrait: lift exposure, recover shadows, clean noise without plastic skin.",
        "bias": {"noise_reduction": 12, "shadows": 8, "sharpness": 6},
        "cues": [
            ("face", 2.2, lambda c: 1.0 if c["is_portrait_subject"] else (0.55 if c["has_face"] else 0.1),
             "face detected in low light", "no face detected"),
            ("dark", 2.0, lambda c: 1.0 if c["brightness"] < 38 else (0.6 if c["brightness"] < 50 else 0.15),
             "very low light", "there is enough light that a night rescue isn't needed"),
            ("noise", 1.0, lambda c: 1.0 if c["noisy"] else 0.5,
             "noise present, as expected at night", "noise is already low"),
        ],
    },
    {
        "id": "night_scene",
        "name": "Night Scene",
        "preset_id": "night",
        "base_intensity": 72,
        "goal": "Low-light scene recovery: exposure lift, shadow detail, noise control, no colour shift.",
        "bias": {"noise_reduction": 10, "shadows": 8, "clarity": 4},
        "cues": [
            ("dark", 2.2, lambda c: 1.0 if c["brightness"] < 38 else (0.6 if c["brightness"] < 50 else 0.15),
             "very low light", "the scene isn't dark enough to need night recovery"),
            ("no_face", 1.4, lambda c: 0.35 if c["is_portrait_subject"] else 1.0,
             "no face -- noise reduction can be more aggressive", "portrait subject needs gentler noise handling"),
            ("noise", 1.0, lambda c: 1.0 if c["noisy"] else 0.55,
             "noise present, as expected in low light", "noise is already low"),
        ],
    },
    {
        "id": "moody_dramatic",
        "name": "Moody Dramatic",
        "preset_id": "moody",
        "base_intensity": 62,
        "goal": "Lean into the drama: deeper shadows, restrained colour, a cool cast.",
        "bias": {"shadows": -4, "vignette": 8},
        "cues": [
            ("drama", 2.0, lambda c: 1.0 if (c["backlit"] or c["contrasty"]) else 0.4,
             "strong contrast or backlight to work with", "the photo is too even for a dramatic grade"),
            ("dark_ok", 1.4, lambda c: 1.0 if c["brightness"] < 62 else 0.45,
             "base tones are dark enough for the mood", "too bright for a moody grade"),
            ("colour_restraint", 0.9, lambda c: 1.0 if c["saturation"] < 65 else 0.5,
             "muted colour suits the look", "colours are loud for a moody grade"),
            ("no_face", 0.7, lambda c: 0.5 if c["is_portrait_subject"] else 1.0,
             "no skin tones to darken", "darkening shadows on a face is risky"),
        ],
    },
    {
        "id": "film_classic",
        "name": "Film Classic",
        "preset_id": "film",
        "base_intensity": 58,
        "goal": "A classic film treatment: soft contrast curve, gentle grain, warm shadows.",
        "bias": {"contrast": 103, "temperature": 3},
        "cues": [
            ("flat_base", 1.6, lambda c: 1.0 if c["flat"] else 0.5,
             "a flat base takes a film curve well", "contrast is already strong"),
            ("colour_room", 1.2, lambda c: 1.0 if c["saturation"] < 62 else 0.5,
             "colour is restrained enough for a film look", "colours are too loud for film"),
            ("light", 1.0, lambda c: 1.0 if 35 <= c["brightness"] <= 80 else 0.5,
             "usable base exposure", "exposure is too extreme"),
            ("detail", 0.7, lambda c: 1.0 if c["sharpness"] >= 30 else 0.5,
             "grain will sit on real detail", "grain on a blurry photo just adds mush"),
        ],
    },
    {
        "id": "mono_dramatic",
        "name": "Monochrome Dramatic",
        "preset_id": "black_and_white",
        "base_intensity": 70,
        "goal": "Full monochrome conversion with punchy tonal separation.",
        "bias": {"clarity": 6},
        "cues": [
            ("colourless", 2.2, lambda c: 1.0 if c["near_monochrome"] else (0.6 if c["muted"] else 0.2),
             "there is almost no colour information to lose", "the photo's colour is a big part of it"),
            ("tonal_range", 1.3, lambda c: 1.0 if c["contrast"] >= 45 else 0.5,
             "enough tonal range for a mono conversion", "flat tones make weak monochrome"),
        ],
    },
]

_LOOK_BY_ID = {look["id"]: look for look in LOOKS}


def score_looks(analysis: dict) -> list:
    """
    Scores EVERY look against this photo and returns them ranked, each
    with a 0-100 match score plus the reasons for and against. This is
    what powers "Other Recommendations" and the "Why not?" dropdown --
    the user sees the shortlist and the trade-offs, not just the winner.
    """
    c = _cues(analysis)
    scored = []
    for look in LOOKS:
        total_weight = 0.0
        earned = 0.0
        reasons = []
        counter = []
        for _cue_id, weight, test, pro, con in look["cues"]:
            value = max(0.0, min(1.0, float(test(c))))
            earned += weight * value
            total_weight += weight
            if value >= 0.85 and pro:
                reasons.append(pro)
            elif value <= 0.55 and con:
                counter.append(con)

        ratio = (earned / total_weight) if total_weight else 0.0
        # Calibration: a flawless-on-paper match lands in the mid-90s
        # rather than a suspicious 100%, and a poor match still gets a
        # real number instead of 0 -- these are recommendations, not
        # certainties, and the score should read that way.
        score = int(round(52 + ratio * 45))
        score = max(8, min(97, score))

        scored.append({
            "id": look["id"],
            "name": look["name"],
            "preset_id": look["preset_id"],
            "goal": look["goal"],
            "match_score": score,
            "reasons": reasons,
            "why_not": counter,
        })

    scored.sort(key=lambda item: item["match_score"], reverse=True)
    return scored


# ===================== 4. BUILD RECIPE =====================
#
# Steps are listed in the order core/enhancer.py::apply_adjustments
# actually applies them, so the "Recipe Preview" the user reads top to
# bottom is the real processing order, not a prettier fiction.

def build_recipe(analysis: dict, look_id: str = "") -> dict:
    """
    Builds the explicit, per-step plan for this photo under the chosen
    look. Every step comes back either:
        status "apply" -- with the value AND why it's needed, or
        status "skip"  -- with the measured reason it isn't
    Nothing is silently omitted; a skipped step is a decision the user
    can see and override in Customize.
    """
    diagnosis = detect_problems(analysis)
    c = diagnosis["cues"]
    problems = diagnosis["problems"]
    strengths = diagnosis["strengths"]
    by_id = {p["id"]: p for p in problems}

    look = _LOOK_BY_ID.get(look_id) or _LOOK_BY_ID[score_looks(analysis)[0]["id"]]
    steps = []

    # ---- 1. Exposure ----
    if "exposure" in by_id:
        target = 62 if c["is_portrait_subject"] else 54
        deficit = max(0.0, target - c["brightness"])
        units = round(min(34.0, deficit * 0.95), 1)
        steps.append(_step("exposure", "Exposure", "apply", {"exposure": units},
                           f"Exposure {_ev(units)}",
                           f"Needed -- {by_id['exposure']['detail']}"))
    elif "exposure_high" in by_id:
        units = -round(min(24.0, (c["brightness"] - 78) * 0.8), 1)
        steps.append(_step("exposure", "Exposure", "apply", {"exposure": units},
                           f"Exposure {_ev(units)}",
                           f"Pulling back -- {by_id['exposure_high']['detail']}"))
    else:
        steps.append(_step("exposure", "Exposure", "skip", {}, "No change",
                           f"Skipped -- exposure already good ({round(c['brightness'])}/100)"))

    # ---- 2. Shadow recovery ----
    if "shadows" in by_id:
        if c["backlit"]:
            amount = 22
            why = "Needed -- backlit subject, lifting the shadows off the subject"
        else:
            amount = int(round(min(26, (50 - c["brightness"]) * 0.85)))
            why = f"Needed -- {by_id['shadows']['detail']}"
        steps.append(_step("shadows", "Shadow Recovery", "apply", {"shadows": amount},
                           f"Shadows +{amount}", why))
    else:
        steps.append(_step("shadows", "Shadow Recovery", "skip", {}, "No change",
                           "Skipped -- shadow detail is already readable"))

    # ---- 3. Highlight protection ----
    highlight_bias = (look.get("bias") or {}).get("highlights", 0)
    if "highlights" in by_id:
        amount = -int(round(min(18, (c["brightness"] - 70) * 0.6))) or -6
        steps.append(_step("highlights", "Highlight Recovery", "apply", {"highlights": amount},
                           f"Highlights {amount}",
                           f"Needed -- {by_id['highlights']['detail']}"))
    elif highlight_bias:
        steps.append(_step("highlights", "Highlight Recovery", "apply",
                           {"highlights": highlight_bias},
                           f"Highlights {highlight_bias:+d}",
                           f"Part of the {look['name']} look -- keeps bright areas from going flat"))
    else:
        steps.append(_step("highlights", "Highlight Recovery", "skip", {}, "No change",
                           "Skipped -- no highlight risk detected"))

    # ---- 4. Colour correction (white balance) ----
    wb = {}
    wb_note = ""
    if c["indoor"] and not c["outdoor"]:
        wb = {"temperature": -5}
        wb_note = "Needed -- cooling the warm cast typical of indoor artificial light"
    elif c["lighting"] == "dark":
        wb = {"temperature": -3}
        wb_note = "Needed -- low light usually skews warm/muddy"
    elif (look.get("bias") or {}).get("temperature"):
        wb = {"temperature": look["bias"]["temperature"]}
        wb_note = f"Part of the {look['name']} look"
    if wb:
        temp = wb["temperature"]
        steps.append(_step("color_correction", "Colour Correction", "apply", wb,
                           f"Temperature {temp:+d}", wb_note))
    else:
        steps.append(_step("color_correction", "Colour Correction", "skip", {}, "No change",
                           "Skipped -- white balance already neutral"))

    # ---- 5. Vibrance ----
    vibrance_bias = (look.get("bias") or {}).get("vibrance", 0)
    if "color" in by_id:
        base = int(round(min(20, (40 - c["saturation"]) * 0.55)))
        amount = max(0, base) + vibrance_bias
        steps.append(_step("vibrance", "Vibrance", "apply", {"vibrance": amount},
                           f"Vibrance {amount:+d}",
                           f"Needed -- {by_id['color']['detail']} (vibrance protects skin tones better than saturation)"))
    elif vibrance_bias > 0 and not c["near_monochrome"]:
        steps.append(_step("vibrance", "Vibrance", "apply", {"vibrance": vibrance_bias},
                           f"Vibrance {vibrance_bias:+d}",
                           f"Part of the {look['name']} look"))
    elif vibrance_bias < 0:
        steps.append(_step("vibrance", "Vibrance", "apply", {"vibrance": vibrance_bias},
                           f"Vibrance {vibrance_bias:+d}",
                           f"Pulled back for the {look['name']} look -- keeps colour believable"))
    else:
        detail = by_id.get("color_high", {}).get("detail") or f"saturation {round(c['saturation'])}/100"
        steps.append(_step("vibrance", "Vibrance", "skip", {}, "No change",
                           f"Skipped -- colour is already strong ({detail})"))

    # ---- 6. Saturation ----
    sat_bias = (look.get("bias") or {}).get("saturation", 100)
    if look["preset_id"] == "black_and_white":
        steps.append(_step("saturation", "Saturation", "skip", {}, "No change",
                           "Skipped -- this look converts to monochrome anyway"))
    elif "color" in by_id and not c["near_monochrome"]:
        amount = int(round(min(10, (40 - c["saturation"]) * 0.25))) + (sat_bias - 100)
        value = 100 + max(0, amount)
        steps.append(_step("saturation", "Saturation", "apply", {"saturation": value},
                           f"Saturation {value - 100:+d}",
                           "Needed -- a small global lift on top of vibrance"))
    elif sat_bias != 100 and not c["near_monochrome"]:
        steps.append(_step("saturation", "Saturation", "apply", {"saturation": sat_bias},
                           f"Saturation {sat_bias - 100:+d}",
                           f"Part of the {look['name']} look"))
    else:
        steps.append(_step("saturation", "Saturation", "skip", {}, "No change",
                           f"Skipped -- saturation is already where it should be ({round(c['saturation'])}/100)"))

    # ---- 7. Contrast ----
    contrast_bias = (look.get("bias") or {}).get("contrast", 100)
    if "contrast" in by_id:
        boost = int(round(min(18, (48 - c["contrast"]) * 0.5)))
        value = 100 + boost + (contrast_bias - 100)
        steps.append(_step("contrast", "Contrast", "apply", {"contrast": value},
                           f"Contrast {value - 100:+d}",
                           f"Needed -- {by_id['contrast']['detail']}"))
    elif contrast_bias != 100:
        steps.append(_step("contrast", "Contrast", "apply", {"contrast": contrast_bias},
                           f"Contrast {contrast_bias - 100:+d}",
                           f"Small nudge for the {look['name']} look -- contrast is already good "
                           f"({round(c['contrast'])}/100), so it stays a nudge"))
    else:
        steps.append(_step("contrast", "Contrast", "skip", {}, "No change",
                           f"Skipped -- contrast already good ({round(c['contrast'])}/100)"))

    # ---- 8. Clarity ----
    clarity_bias = (look.get("bias") or {}).get("clarity", 0)
    if c["flat"] or clarity_bias:
        amount = clarity_bias + (6 if c["flat"] else 0)
        if c["soft"]:
            # Clarity on a soft photo mostly amplifies the softness --
            # halve it rather than pretending it adds detail.
            amount = int(round(amount * 0.5))
        if amount > 0:
            steps.append(_step("clarity", "Clarity", "apply", {"clarity": amount},
                               f"Clarity +{amount}",
                               "Needed -- adds mid-tone separation without touching global contrast"
                               if c["flat"] else f"Part of the {look['name']} look"))
        else:
            steps.append(_step("clarity", "Clarity", "skip", {}, "No change",
                               "Skipped -- mid-tone contrast is already fine"))
    else:
        steps.append(_step("clarity", "Clarity", "skip", {}, "No change",
                           "Skipped -- mid-tone contrast is already fine"))

    # ---- 9. Sharpening ----  (the headline auto-skip)
    sharp_bias = (look.get("bias") or {}).get("sharpness", 0)
    if c["already_sharp"]:
        steps.append(_step("sharpening", "Sharpening", "skip", {}, "No change",
                           f"Skipped -- already sharp ({round(c['sharpness'])}/100). "
                           "Sharpening a sharp photo only adds halos."))
    elif c["sharpness"] < 18:
        steps.append(_step("sharpening", "Sharpening", "apply", {"sharpness": 14},
                           "Sharpening +14",
                           f"Light touch only -- the photo reads as blurry ({round(c['sharpness'])}/100) "
                           "and unsharp masking can't recover real detail that isn't there"))
    else:
        amount = int(round(min(30, (52 - c["sharpness"]) * 0.7))) + sharp_bias
        steps.append(_step("sharpening", "Sharpening", "apply", {"sharpness": amount},
                           f"Sharpening +{amount}",
                           f"Needed -- slightly soft ({round(c['sharpness'])}/100)"))

    # ---- 10. Denoise ----
    denoise_bias = (look.get("bias") or {}).get("noise_reduction", 0)
    if "noise" in by_id:
        amount = int(round(min(45, (c["noise"] - 30) * 0.9))) + denoise_bias
        steps.append(_step("denoise", "Denoise", "apply", {"noise_reduction": amount},
                           f"Denoise +{amount}",
                           f"Needed -- {by_id['noise']['detail']}"))
    else:
        steps.append(_step("denoise", "Denoise", "skip", {}, "No change",
                           f"Skipped -- quality already good (noise {round(c['noise'])}/100). "
                           "Denoising a clean photo just softens it."))

    # ---- 11. Skin protection ----
    if c["has_face"]:
        # Not a mask -- an honest constraint on the recipe: with a face
        # in frame, the aggressive controls get capped instead of the
        # pipeline pretending it can retouch skin selectively (that's
        # Phase 10/GFPGAN territory, not this phase's).
        steps.append(_step("skin_protection", "Skin Protection", "apply",
                           {"clarity": -2, "sharpness": -4},
                           f"Capped clarity & sharpening ({c['face_count']} face"
                           f"{'s' if c['face_count'] != 1 else ''})",
                           "Needed -- clarity and sharpening are pulled back so skin doesn't go crunchy"))
    else:
        steps.append(_step("skin_protection", "Skin Protection", "skip", {}, "No change",
                           "Skipped -- no face detected"))

    # ---- 11b. Background (Remove BG suggestion -- see core/analyzer.py::_background_clutter) ----
    # A THIRD step status, "suggested": this pipeline never runs Remove
    # BG itself (that's a separate, mask/model-driven tool the user
    # should look at and confirm, not something safe to auto-apply
    # blind), but it tells the user when the photo looks like a
    # candidate for it, with the reasoning that led there.
    if "background" in by_id:
        steps.append(_step("background", "Background", "suggested", {},
                           "Try Remove Background",
                           f"Suggested -- {by_id['background']['detail']}. "
                           "Not applied automatically -- open Remove BG to confirm the cutout."))
    else:
        steps.append(_step("background", "Background", "skip", {}, "No change",
                           "Skipped -- background doesn't look like it needs cleanup"))

    # ---- 12. Final look (vignette / grain / mono come from the preset) ----
    vignette_bias = (look.get("bias") or {}).get("vignette", 0)
    if vignette_bias:
        steps.append(_step("final_look", "Final Look", "apply", {"vignette": vignette_bias},
                           f"Vignette +{vignette_bias}",
                           f"Part of the {look['name']} look -- draws the eye inward"))
    else:
        steps.append(_step("final_look", "Final Look", "apply", {},
                           f"{_preset_name(look['preset_id'])} base look",
                           f"The {look['name']} grade itself, applied at the intensity below"))

    intensity = _adaptive_intensity(look, problems, strengths, c)

    recipe = {
        "look_id": look["id"],
        "look_name": look["name"],
        "preset_id": look["preset_id"],
        "preset_name": _preset_name(look["preset_id"]),
        "goal": look["goal"],
        "steps": steps,
        "intensity": intensity,
        "applied_count": sum(1 for s in steps if s["status"] == "apply"),
        "skipped_count": sum(1 for s in steps if s["status"] == "skip"),
    }
    return recipe


def _step(step_id, label, status, adjustments, action, reason):
    return {
        "id": step_id,
        "label": label,
        "status": status,          # "apply" | "skip"
        "action": action,          # human-readable value, e.g. "Exposure +0.18"
        "reason": reason,          # WHY -- always populated, including for skips
        "adjustments": adjustments,
        "editable": bool(adjustments) or status == "apply",
    }


def _ev(units: float) -> str:
    """0-100 slider units -> the EV number a photographer recognises."""
    stops = units * _STOPS_PER_UNIT
    return f"{stops:+.2f}"


def _preset_name(preset_id: str) -> str:
    preset = get_preset(preset_id)
    return preset["name"] if preset else preset_id


def _adaptive_intensity(look, problems, strengths, cues) -> int:
    """
    Per-image strength, which is the whole point of "adaptive": a photo
    with several serious problems gets pushed harder than one that's
    nearly right, and a photo with lots of measured strengths gets a
    lighter touch. Never a constant.
    """
    load = _problem_load(problems)
    base = look.get("base_intensity", 65)
    value = base + (load - 0.5) * 30

    # Each already-good measurement pulls the strength back a little --
    # "don't fix what isn't broken", expressed numerically.
    value -= min(6.0, len(strengths) * 1.5)

    # A face in frame caps how far the grade can go, for the same reason
    # Skin Protection exists.
    if cues["has_face"]:
        value = min(value, 78)
    # Blurry photos can't take a strong grade -- it just looks processed.
    if cues["sharpness"] < 18:
        value = min(value, 60)

    return int(max(30, min(95, round(value))))


# ===================== 5. RECIPE -> ADJUSTMENTS =====================

def recipe_to_adjustments(recipe: dict) -> dict:
    """
    Collapses the recipe's applied steps into one adjustment delta dict.
    Skipped steps contribute nothing -- that's what "skip" means, and
    it's why the skip list is trustworthy rather than cosmetic.
    """
    delta = {}
    for step in recipe.get("steps", []):
        if step.get("status") != "apply":
            continue
        for key, value in (step.get("adjustments") or {}).items():
            if key not in _ADJUSTMENT_RANGES:
                continue
            if DEFAULT_ADJUSTMENTS[key] == 100:
                delta[key] = delta.get(key, 100) + (value - 100)
            else:
                delta[key] = delta.get(key, 0) + value
    return delta


def resolve_pipeline_adjustments(recommendation: dict, intensity=None) -> dict:
    """
    The full 15-key adjustment dict this pipeline will actually apply:

        BASE LOOK (Phase 4 preset, scaled)  +  SMART ADJUSTMENTS (recipe)
                            both scaled by the final strength

    This is exactly the "preset as base look + analyzer-driven smart
    adjustments + one final strength" model -- the Filters view no longer
    has to choose between "a preset" and "the pipeline's corrections",
    because they're the same render.
    """
    recipe = recommendation.get("recipe") or {}
    strength = recipe.get("intensity", recommendation.get("intensity", 70))
    if intensity is not None:
        strength = intensity
    strength = max(0.0, min(100.0, float(strength)))
    t = strength / 100.0

    preset = get_preset(recommendation.get("preset_id", "")) or {"adjustments": {}}
    base = _resolve_preset_adjustments(preset, strength)

    delta = recipe_to_adjustments(recipe)
    scaled_delta = {}
    for key, value in delta.items():
        if DEFAULT_ADJUSTMENTS[key] == 100:
            scaled_delta[key] = 100 + (value - 100) * t
        else:
            scaled_delta[key] = value * t

    return {k: round(v, 1) for k, v in _merge(base, scaled_delta).items()}


def apply_pipeline(image: Image.Image, recommendation: dict, intensity=None) -> Image.Image:
    """
    Renders a recommendation onto an image: merged adjustments first,
    then the preset's monochrome/grain treatment (same order as
    core/filters.py::apply_preset, so a pipeline result and a plain
    preset result of the same look are visually consistent).
    """
    recipe = recommendation.get("recipe") or {}
    strength = intensity if intensity is not None else recipe.get(
        "intensity", recommendation.get("intensity", 70)
    )
    strength = max(0.0, min(100.0, float(strength)))
    t = strength / 100.0

    adjustments = resolve_pipeline_adjustments(recommendation, strength)
    result = apply_adjustments(image, adjustments)

    preset = get_preset(recommendation.get("preset_id", "")) or {}
    if preset.get("monochrome") or recommendation.get("monochrome"):
        mono = to_monochrome(result)
        result = Image.blend(result.convert("RGB"), mono, t) if t < 1.0 else mono

    grain = preset.get("grain", recommendation.get("grain", 0)) or 0
    if grain:
        result = add_grain(result, round(grain * t))

    return result


# ===================== 6. SAFETY CHECK =====================

def evaluate_safety(analysis: dict, adjustments: dict, recipe: dict) -> dict:
    """
    Pre-flight check, run BEFORE Apply. Catches the five ways an
    otherwise sensible recipe can still damage a photo. Warnings are
    advisory -- the user can always apply anyway -- but they're shown,
    not swallowed.
    """
    c = _cues(analysis)
    warnings = []

    exposure = adjustments.get("exposure", 0)
    whites = adjustments.get("whites", 0)
    highlights = adjustments.get("highlights", 0)
    shadows = adjustments.get("shadows", 0)
    blacks = adjustments.get("blacks", 0)
    saturation = adjustments.get("saturation", 100)
    vibrance = adjustments.get("vibrance", 0)
    sharpness = adjustments.get("sharpness", 0)
    denoise = adjustments.get("noise_reduction", 0)

    headroom = 100 - c["brightness"]
    if exposure + whites + max(0, highlights) > max(8, headroom * 0.9):
        warnings.append(_warning(
            "clipping", "Highlights may clip",
            f"brightness is already {round(c['brightness'])}/100 and the recipe lifts exposure "
            f"{_ev(exposure)}. Bright areas could lose detail permanently.",
            "Lower the intensity, or push Highlight Recovery further negative."))

    if (shadows < 0 or blacks > 0) and c["brightness"] < 42:
        warnings.append(_warning(
            "crushed_shadows", "Shadows may become too dark",
            f"the photo is already dark ({round(c['brightness'])}/100) and the recipe deepens shadows. "
            "Dark areas could go solid black.",
            "Reduce the intensity or set Shadow Recovery to a positive value."))

    effective_sat = (saturation - 100) + vibrance
    if c["saturation"] > 65 and effective_sat > 12:
        warnings.append(_warning(
            "oversaturation", "Colours may look excessive",
            f"saturation already measures {round(c['saturation'])}/100 and the recipe adds "
            f"{round(effective_sat)} more. Skies and skin can turn neon.",
            "Skip the Saturation step and keep Vibrance only."))
    elif c["has_face"] and effective_sat > 20:
        warnings.append(_warning(
            "oversaturation", "Skin tones may look excessive",
            f"{c['face_count']} face(s) detected and the recipe adds {round(effective_sat)} colour.",
            "Reduce Saturation -- Vibrance alone is kinder to skin."))

    if sharpness > 45 and c["already_sharp"]:
        warnings.append(_warning(
            "oversharpening", "Sharpening may be excessive",
            f"the photo already measures {round(c['sharpness'])}/100 sharp; more will show halos on edges.",
            "Set the Sharpening step to Skip."))

    if denoise >= 20 and sharpness >= 30:
        warnings.append(_warning(
            "denoise_conflict", "Denoise and sharpening are fighting each other",
            f"denoise +{round(denoise)} smooths detail, then sharpening +{round(sharpness)} tries to "
            "rebuild it -- the usual result is a plastic, over-processed look.",
            "Keep one of the two: denoise for noisy photos, sharpening for soft ones."))

    return {
        "safe": not warnings,
        "warnings": warnings,
        "label": "Pipeline safe to apply" if not warnings else (
            f"{len(warnings)} thing{'s' if len(warnings) != 1 else ''} to check before applying"
        ),
    }


def _warning(wid, title, detail, fix):
    return {"id": wid, "title": title, "detail": detail, "fix": fix}


# ===================== 7. THE PUBLIC ENTRY POINT =====================

def recommend_pipeline(analysis: dict) -> dict:
    """
    The one call the bridge makes. Returns the full v2 payload:

        {
          # --- v1 keys, unchanged, so old call sites keep working ---
          "id", "name", "preset_id", "preset_name", "intensity",
          "reason", "signals", "confidence", "adjustments",
          "monochrome", "grain",

          # --- v2 ---
          "match_score":   0-100 for the chosen look
          "explanation":   plain-language "why this recommendation?"
          "problems":      [{id,label,severity,detail}, ...]
          "strengths":     [{id,label,detail}, ...]
          "recipe":        {steps:[{label,action,status,reason}], intensity, ...}
          "safety":        {safe, warnings:[{title,detail,fix}], label}
          "alternatives":  [{id,name,match_score,reasons,why_not}, ...]
          "cues":          the normalized numbers every decision came from
        }
    """
    ranked = score_looks(analysis)
    best = ranked[0]

    recipe = build_recipe(analysis, best["id"])
    diagnosis = detect_problems(analysis)
    adjustments = resolve_pipeline_adjustments(
        {"preset_id": recipe["preset_id"], "recipe": recipe}, recipe["intensity"]
    )
    safety = evaluate_safety(analysis, adjustments, recipe)
    preset = get_preset(recipe["preset_id"]) or {}

    explanation = _explain(analysis, best, recipe, diagnosis)

    return {
        # ---- v1-compatible keys ----
        "id": best["id"],
        "name": best["name"],
        "preset_id": recipe["preset_id"],
        "preset_name": recipe["preset_name"],
        "intensity": recipe["intensity"],
        "reason": explanation,
        "signals": _signals(analysis),
        "confidence": best["match_score"],
        "adjustments": adjustments,
        "monochrome": bool(preset.get("monochrome", False)),
        "grain": preset.get("grain", 0),

        # ---- v2 additions ----
        "match_score": best["match_score"],
        "goal": best["goal"],
        "explanation": explanation,
        "problems": diagnosis["problems"],
        "strengths": diagnosis["strengths"],
        "recipe": recipe,
        "safety": safety,
        "alternatives": ranked[1:4],
        "ranked": ranked,
        "cues": diagnosis["cues"],
    }


def customize_pipeline(analysis: dict, look_id: str, overrides: dict = None) -> dict:
    """
    Rebuilds a recommendation under the user's own choices -- the
    "Customize" button, and also how "pick a different look from Other
    Recommendations" is handled (same code path, no special case).

    overrides:
        {"intensity": 0-100,
         "steps": {"sharpening": "apply"|"skip", ...},
         "step_values": {"exposure": {"exposure": 12}, ...}}

    "User can always override" is the spec's own wording for this phase;
    this is that promise, implemented -- toggling a step off genuinely
    removes its contribution, because recipe_to_adjustments() only reads
    steps whose status is "apply".
    """
    overrides = overrides or {}
    ranked = score_looks(analysis)
    chosen = next((r for r in ranked if r["id"] == look_id), ranked[0])

    recipe = build_recipe(analysis, chosen["id"])

    status_overrides = overrides.get("steps") or {}
    value_overrides = overrides.get("step_values") or {}
    for step in recipe["steps"]:
        if step["id"] in status_overrides:
            wanted = status_overrides[step["id"]]
            if wanted in ("apply", "skip") and wanted != step["status"]:
                step["status"] = wanted
                step["reason"] = ("Turned on by you" if wanted == "apply"
                                  else "Turned off by you")
                if wanted == "apply" and not step["adjustments"]:
                    step["action"] = "No value set -- adjust it in the Editor"
        if step["id"] in value_overrides:
            merged = {k: v for k, v in (value_overrides[step["id"]] or {}).items()
                      if k in _ADJUSTMENT_RANGES}
            if merged:
                step["adjustments"] = {k: _clamp(k, v) for k, v in merged.items()}
                step["reason"] = "Value set by you"
                step["action"] = ", ".join(
                    f"{k.replace('_', ' ').title()} {v:+g}" for k, v in step["adjustments"].items()
                )

    if overrides.get("intensity") is not None:
        recipe["intensity"] = int(max(0, min(100, round(float(overrides["intensity"])))))

    recipe["applied_count"] = sum(1 for s in recipe["steps"] if s["status"] == "apply")
    recipe["skipped_count"] = sum(1 for s in recipe["steps"] if s["status"] == "skip")

    diagnosis = detect_problems(analysis)
    adjustments = resolve_pipeline_adjustments(
        {"preset_id": recipe["preset_id"], "recipe": recipe}, recipe["intensity"]
    )
    safety = evaluate_safety(analysis, adjustments, recipe)
    preset = get_preset(recipe["preset_id"]) or {}
    explanation = _explain(analysis, chosen, recipe, diagnosis)

    return {
        "id": chosen["id"],
        "name": chosen["name"],
        "preset_id": recipe["preset_id"],
        "preset_name": recipe["preset_name"],
        "intensity": recipe["intensity"],
        "reason": explanation,
        "signals": _signals(analysis),
        "confidence": chosen["match_score"],
        "adjustments": adjustments,
        "monochrome": bool(preset.get("monochrome", False)),
        "grain": preset.get("grain", 0),
        "match_score": chosen["match_score"],
        "goal": chosen["goal"],
        "explanation": explanation,
        "problems": diagnosis["problems"],
        "strengths": diagnosis["strengths"],
        "recipe": recipe,
        "safety": safety,
        "alternatives": [r for r in ranked if r["id"] != chosen["id"]][:3],
        "ranked": ranked,
        "cues": diagnosis["cues"],
        "customized": bool(status_overrides or value_overrides or overrides.get("intensity") is not None),
    }


# ===================== EXPLANATION + SIGNALS =====================

def _explain(analysis, look, recipe, diagnosis) -> str:
    """
    The "Why this recommendation?" paragraph -- written from the same
    numbers the decision used, in the order a person would explain it:
    what I see -> what's wrong -> what's already fine -> what I'll do.
    """
    c = diagnosis["cues"]
    problems = diagnosis["problems"]
    strengths = diagnosis["strengths"]

    scene_bits = []
    if c["is_portrait_subject"]:
        scene_bits.append(f"{c['face_count']} face{'s' if c['face_count'] != 1 else ''} in frame")
    elif c["is_landscape_subject"]:
        scene_bits.append("an outdoor landscape scene")
    elif c["scene"] != "uncertain":
        scene_bits.append(f"an {c['scene']} scene")
    else:
        scene_bits.append("a general scene")
    if not c["has_face"]:
        scene_bits.append("no faces detected")

    sentences = [f"I see {' with '.join(scene_bits)}."]

    if problems:
        top = ", ".join(f"{p['label'].lower()} ({p['detail']})" for p in problems[:3])
        sentences.append(f"The main problems are {top}.")
    else:
        sentences.append("Nothing is clearly wrong with this photo.")

    if strengths:
        keep = ", ".join(f"{s['label'].lower()} ({s['detail']})" for s in strengths[:3])
        sentences.append(f"Already fine, so left alone: {keep}.")

    sentences.append(
        f"{look['name']} fits because {look['goal'][0].lower() + look['goal'][1:]} "
        f"{recipe['applied_count']} of {len(recipe['steps'])} steps will run at "
        f"{recipe['intensity']}% strength; {recipe['skipped_count']} are skipped."
    )
    return " ".join(sentences)


def _signals(analysis: dict) -> list:
    """
    Short badge strings for the Analyzer facts a recommendation leaned
    on -- kept from v1 because #pipeline-signals in the frontend renders
    these directly.
    """
    c = _cues(analysis)
    signals = []
    if c["has_face"]:
        signals.append(f"{c['face_count']} face{'s' if c['face_count'] != 1 else ''}")
    else:
        signals.append("no face")
    if c["scene"] != "uncertain":
        signals.append(c["scene"])
    if c["is_landscape_subject"]:
        signals.append("landscape scene")
    signals.append(f"{c['lighting']} light")
    signals.append(f"brightness {round(c['brightness'])}")
    signals.append(f"contrast {round(c['contrast'])}")
    signals.append(f"sharpness {round(c['sharpness'])}")
    if c["saturation"]:
        signals.append(f"saturation {round(c['saturation'])}")
    if c["noise"] >= 45:
        signals.append(f"noise {round(c['noise'])}")
    if c["background_busy"]:
        signals.append("busy background")
    return signals


# ===================== INTROSPECTION =====================

def list_pipeline_rules() -> list:
    """
    The look catalogue, for the Settings/docs view -- so the decision
    logic is inspectable rather than a black box the user has to trust.
    """
    return [
        {
            "id": look["id"],
            "name": look["name"],
            "preset_id": look["preset_id"],
            "preset_name": _preset_name(look["preset_id"]),
            "goal": look["goal"],
            "base_intensity": look.get("base_intensity", 65),
            "cues": [cue[0] for cue in look["cues"]],
        }
        for look in LOOKS
    ]