# core/filters.py
#
# PHASE 4 -- Presets (Smart Filters).
#
# Classic deterministic image processing -- reuses core/enhancer.py's
# adjustment pipeline for all tone/color work (including the "smoke"
# atmospheric-haze slider -- see core/enhancer.py::_apply_smoke),
# plus two small effects enhancer.py doesn't have: monochrome
# conversion and film grain. No
# trained model here -- same module-split rule as core/enhancer.py,
# core/crop.py, core/object_remover.py (ai/ is reserved for
# trained-model features like ai/bg_remover.py).
#
# Two kinds of presets:
#   - BUILTIN_PRESETS: the 9 required presets from the spec (Cinematic,
#     Luxury, Moody, Vintage, Black & White, Portrait, Travel, Night,
#     Film). Fixed -- not editable/deletable -- but can be favorited.
#   - Custom presets: user-created (typically "save the Editor's
#     current sliders as a preset"), stored locally as JSON. No
#     database dependency on purpose -- matches the spec's "stored
#     locally as JSON" line for Phase 4 and keeps this module Qt-free
#     like every other core/ module, so it works standalone before
#     Phase 13's database/ lands. Supports save, rename, duplicate,
#     delete, export (single preset -> a shareable .json file),
#     import, and favorite.
#
# Every preset's adjustment values live inside the same "safe range"
# every manual slider in core/enhancer.py already enforces, so applying
# a preset can never destroy a photo the way an unclamped one-shot
# filter could -- see apply_preset()'s intensity blending below, which
# keeps that guarantee even at partial strength.

import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.enhancer import DEFAULT_ADJUSTMENTS, apply_adjustments

# Local JSON store for custom presets + favorites (favorites can be
# either a builtin or a custom preset id). Lives next to the other
# local-storage folders (input/output/cache/logs) at the project root
# -- no SQLite dependency, so Phase 4 works standalone before Phase 13.
_STORE_PATH = Path(__file__).resolve().parent.parent / "presets" / "custom_presets.json"

_EMPTY_STORE = {"presets": [], "favorites": []}


# ===================== BUILT-IN PRESETS (Phase 4 required 9) =====================
#
# Each "adjustments" dict only needs to list values that differ from
# core/enhancer.py's DEFAULT_ADJUSTMENTS -- apply_preset() below fills
# in the rest via _scaled_adjustments().
BUILTIN_PRESETS = [
    {
        "id": "cinematic",
        "name": "Cinematic",
        "description": "Controlled contrast, a cool-leaning grade, and a soft vignette.",
        "adjustments": {
            "contrast": 112, "highlights": -12, "shadows": 8,
            "temperature": -6, "tint": 4, "vibrance": 10,
            "clarity": 14, "vignette": 28,
        },
        "grain": 0,
        "monochrome": False,
    },
    {
        "id": "luxury",
        "name": "Luxury",
        "description": "Clean contrast, warm highlights, crisp detail.",
        "adjustments": {
            "contrast": 110, "highlights": 6, "temperature": 8,
            "vibrance": 8, "clarity": 18, "sharpness": 22,
        },
        "grain": 0,
        "monochrome": False,
    },
    {
        "id": "moody",
        "name": "Moody",
        "description": "Darker shadows, pulled-back saturation, a cool tone.",
        "adjustments": {
            "exposure": -6, "shadows": -14, "blacks": 10,
            "saturation": 82, "temperature": -10, "contrast": 108,
            "vignette": 22, "smoke": 16,
        },
        "grain": 0,
        "monochrome": False,
    },
    {
        "id": "vintage",
        "name": "Vintage",
        "description": "Faded colors, a warm tint, and a hint of film grain.",
        "adjustments": {
            "blacks": -14, "contrast": 92, "saturation": 86,
            "temperature": 14, "tint": 6, "vignette": 18,
        },
        "grain": 22,
        "monochrome": False,
    },
    {
        "id": "black_and_white",
        "name": "Black & White",
        "description": "Full monochrome conversion with punchy contrast and optional grain.",
        "adjustments": {
            "contrast": 116, "clarity": 16, "highlights": -6, "shadows": 6,
        },
        "grain": 10,
        "monochrome": True,
    },
    {
        "id": "portrait",
        "name": "Portrait",
        "description": "Skin-friendly correction -- gentle contrast, soft warmth, light sharpening.",
        "adjustments": {
            "contrast": 106, "highlights": -8, "shadows": 6,
            "temperature": 5, "vibrance": 6, "clarity": 8, "sharpness": 14,
        },
        "grain": 0,
        "monochrome": False,
    },
    {
        "id": "travel",
        "name": "Travel",
        "description": "Vibrant colors and extra clarity for landscapes and street shots.",
        "adjustments": {
            "contrast": 110, "saturation": 114, "vibrance": 20,
            "clarity": 22, "sharpness": 18,
        },
        "grain": 0,
        "monochrome": False,
    },
    {
        "id": "night",
        "name": "Night",
        "description": "Noise reduction, exposure lift, and shadow recovery for low light.",
        "adjustments": {
            "exposure": 14, "shadows": 22, "blacks": 8,
            "noise_reduction": 34, "temperature": -4, "clarity": 6,
            "smoke": 10,
        },
        "grain": 0,
        "monochrome": False,
    },
    {
        "id": "film",
        "name": "Film",
        "description": "Cinematic contrast with a classic film treatment and grain.",
        "adjustments": {
            "contrast": 114, "highlights": -10, "shadows": 10,
            "saturation": 92, "temperature": 6, "vignette": 16,
        },
        "grain": 18,
        "monochrome": False,
    },
]

_BUILTIN_BY_ID = {p["id"]: p for p in BUILTIN_PRESETS}


# ===================== EFFECTS NOT IN enhancer.py =====================

def to_monochrome(image: Image.Image) -> Image.Image:
    """
    Desaturates `image` to grayscale, then expands it back to a
    3-channel RGB image (flat R=G=B) so it still flows through the rest
    of the pipeline -- and still exports as a normal JPEG/PNG -- exactly
    like the color path.
    """
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    gray = rgb.convert("L")
    return Image.merge("RGB", (gray, gray, gray))


def add_grain(image: Image.Image, amount: int) -> Image.Image:
    """
    Adds monochrome film grain. amount: 0 (none) .. 100 (heavy). Noise
    is generated in luminance only (the same random delta added to all
    3 channels) rather than per-channel color noise, so it reads as
    real film grain instead of RGB static/color speckling.
    """
    if not amount:
        return image
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    arr = np.asarray(rgb).astype(np.float32)

    rng = np.random.default_rng()
    strength = (amount / 100.0) * 22.0  # stddev in 0-255 units, capped modest
    noise = rng.normal(0.0, strength, size=arr.shape[:2]).astype(np.float32)
    arr = arr + noise[..., None]

    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


# ===================== APPLYING A PRESET =====================

def _scaled_adjustments(preset_adjustments: dict, intensity: float) -> dict:
    """
    Blends a preset's adjustment overrides toward DEFAULT_ADJUSTMENTS by
    `intensity` (0-100, 100 = full preset strength) so the frontend's
    Intensity slider can dial a preset back without a second processing
    path. A blended value can never leave the range between the
    default and the preset's own (already safe-range) value, so this
    keeps the same "can't destroy the photo" guarantee at any strength.
    """
    t = max(0.0, min(1.0, intensity / 100.0))
    out = {}
    for key, preset_val in preset_adjustments.items():
        default_val = DEFAULT_ADJUSTMENTS[key]
        out[key] = default_val + (preset_val - default_val) * t
    return out


def apply_preset(image: Image.Image, preset: dict, intensity: float = 100) -> Image.Image:
    """
    Applies one preset (a dict shaped like a BUILTIN_PRESETS entry, or a
    custom preset loaded from the JSON store) to `image` at the given
    intensity (0-100).

    Order: tone/color adjustments (enhancer.py) -> monochrome -> grain,
    so grain lands last (same "sharpen/grain last" reasoning as
    enhancer.py -- it shouldn't get smoothed out by anything upstream).
    """
    adjustments = _scaled_adjustments(preset.get("adjustments", {}), intensity)
    result = apply_adjustments(image, adjustments)

    t = max(0.0, min(1.0, intensity / 100.0))
    if preset.get("monochrome"):
        mono = to_monochrome(result)
        result = Image.blend(result.convert("RGB"), mono, t) if t < 1.0 else mono

    grain_amount = preset.get("grain", 0)
    if grain_amount:
        result = add_grain(result, round(grain_amount * t))

    return result


# ===================== PHASE 4: Auto Filter =====================
#
# Suggests the single builtin preset that best matches this specific
# photo, plus an intensity -- same "read the photo, don't guess" spirit
# as core/enhancer.py's auto_white_balance and ui/bridge.py's
# autoEnhanceAnalyze, just scoped to "which of the 9 looks fits" rather
# than per-slider values. Deterministic feature scoring, no trained
# model (consistent with every other core/ module).

def _photo_features(image: Image.Image) -> dict:
    """A small, cheap feature read of the photo used to score presets."""
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    small = rgb.copy()
    small.thumbnail((400, 400), Image.LANCZOS)
    arr = np.asarray(small).astype(np.float32)

    lab = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0

    mean_l = float(L.mean()) / 255.0          # 0 (dark) .. 1 (bright)
    contrast = float(L.std()) / 255.0          # 0 (flat) .. ~0.3+ (punchy)
    warmth = float(B.mean())                   # negative = cool/blue, positive = warm/yellow

    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = float(((maxc - minc) / (maxc + 1e-6)).mean())  # 0 (gray) .. 1 (vivid)

    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edge_density = float(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)).mean()) / 40.0
    edge_density = min(edge_density, 1.0)

    skin_ratio = 0.0
    try:
        from core.enhancer import _skin_tone_mask
        skin_ratio = float((_skin_tone_mask(arr / 255.0)[..., 0] > 0.5).mean())
    except Exception:  # noqa: BLE001
        pass

    return {
        "mean_l": mean_l, "contrast": contrast, "warmth": warmth,
        "sat": sat, "edge_density": edge_density, "skin_ratio": skin_ratio,
    }


def _preset_affinity(preset_id: str, f: dict) -> float:
    """
    Hand-tuned affinity score (higher = better fit) for one builtin
    preset given this photo's features. Each rule targets the trait
    that preset is actually built around (see BUILTIN_PRESETS'
    "adjustments" above), so a dark/low-contrast photo scores "Night"
    higher, a portrait-heavy warm photo scores "Portrait" higher, etc.
    """
    scores = {
        "night": (1.0 - f["mean_l"]) * 1.4 + (1.0 - f["contrast"]) * 0.6,
        "black_and_white": (1.0 - f["sat"]) * 1.6 + f["contrast"] * 0.4,
        "portrait": f["skin_ratio"] * 1.8 + max(0.0, f["warmth"]) * 0.02,
        "travel": f["sat"] * 1.1 + f["edge_density"] * 0.8 + f["mean_l"] * 0.3,
        "moody": (1.0 - f["mean_l"]) * 0.9 + f["contrast"] * 0.7 - f["sat"] * 0.3,
        "cinematic": f["contrast"] * 1.0 + (0.5 - abs(f["mean_l"] - 0.5)) * 0.5,
        "vintage": (1.0 - f["contrast"]) * 0.8 + max(0.0, f["warmth"]) * 0.03 - f["sat"] * 0.2,
        "luxury": f["contrast"] * 0.6 + max(0.0, f["warmth"]) * 0.02 + f["sat"] * 0.3,
        "film": f["contrast"] * 0.7 + (1.0 - f["sat"]) * 0.3,
    }
    return scores.get(preset_id, 0.0)


def auto_suggest_preset(image: Image.Image) -> dict:
    """
    Returns {"ok": true, "preset_id": ..., "intensity": 0-100,
    "summary": "..."} -- the best-fit builtin preset for this photo and
    a suggested strength. Intensity leans higher for a flatter/duller
    photo (more room to benefit from a strong look) and lower for a
    photo that's already got good contrast/color, so Auto Filter never
    overcooks a photo that didn't need much help.
    """
    f = _photo_features(image)
    scored = sorted(BUILTIN_PRESETS, key=lambda p: _preset_affinity(p["id"], f), reverse=True)
    best = scored[0]

    # "How much does this photo need help" -- low contrast + low
    # saturation = more room for a preset to make a visible difference,
    # so it gets applied stronger; an already punchy/vivid photo gets a
    # lighter touch so the preset accents it instead of overpowering it.
    need = (1.0 - f["contrast"]) * 0.6 + (1.0 - f["sat"]) * 0.4
    intensity = int(round(55 + need * 40))  # roughly 55-95
    intensity = max(40, min(100, intensity))

    summary = f'"{best["name"]}" matches this photo best'
    if f["skin_ratio"] > 0.15:
        summary += " (portrait detected)"
    elif f["mean_l"] < 0.35:
        summary += " (low-light scene)"
    elif f["sat"] < 0.15:
        summary += " (mostly desaturated)"
    summary += f" -- suggested at {intensity}% strength."

    return {"ok": True, "preset_id": best["id"], "intensity": intensity, "summary": summary}


# ===================== PHASE 4: Filter Stacking =====================
#
# Combines 2+ presets (each at its own intensity) into a single
# effective adjustment set instead of literally re-running apply_preset
# once per layer (which would double-process JPEG-style artifacts and
# compound each preset's own clamping oddly). "Multiplicative" sliders
# (brightness/contrast/saturation, where 100 = no change) stack by
# multiplying their ratios; "additive" sliders (everything based at 0,
# e.g. exposure/temperature/vibrance) stack by summing the deltas --
# then the combined result is clamped back into each slider's normal
# safe range, so a stack can never leave "destroy the photo" territory
# any more than a single preset could.
_MULTIPLICATIVE_KEYS = {"brightness", "contrast", "saturation"}
_SLIDER_RANGES = {
    "brightness": (50, 150), "contrast": (50, 150), "saturation": (0, 200),
    "exposure": (-100, 100), "highlights": (-100, 100), "shadows": (-100, 100),
    "whites": (-100, 100), "blacks": (-100, 100),
    "highlight_recovery": (0, 100), "shadow_recovery": (0, 100),
    "temperature": (-100, 100), "tint": (-100, 100), "vibrance": (-100, 100),
    "sharpness": (0, 100), "clarity": (0, 100),
    "noise_reduction": (0, 100), "vignette": (0, 100),
}


def _merge_stack_adjustments(scaled_layers: list) -> dict:
    combined = dict(DEFAULT_ADJUSTMENTS)
    for key in DEFAULT_ADJUSTMENTS:
        if key not in _SLIDER_RANGES:
            continue  # skip non-numeric keys (local_adjustments, protect toggles)
        default_val = DEFAULT_ADJUSTMENTS[key]
        if key in _MULTIPLICATIVE_KEYS:
            ratio = 1.0
            for layer in scaled_layers:
                v = layer.get(key, default_val)
                ratio *= (v / default_val) if default_val else 1.0
            value = default_val * ratio
        else:
            delta = 0.0
            for layer in scaled_layers:
                delta += layer.get(key, default_val) - default_val
            value = default_val + delta
        lo, hi = _SLIDER_RANGES[key]
        combined[key] = max(lo, min(hi, value))
    return combined


def apply_preset_stack(image: Image.Image, layers: list) -> Image.Image:
    """
    layers: [{"preset_id": ..., "intensity": 0-100}, ...] (2+ entries).
    Applies every layer's tone/color adjustments as one merged pass
    (see _merge_stack_adjustments), then layers monochrome (on if ANY
    layer requests it, blended by that layer's own intensity) and grain
    (summed across layers, capped at 100) last -- same ordering
    reasoning as apply_preset.
    """
    scaled_layers = []
    mono_t = 0.0
    total_grain = 0
    for layer in layers:
        preset = get_preset(layer.get("preset_id"))
        if not preset:
            continue
        intensity = layer.get("intensity", 100)
        scaled_layers.append(_scaled_adjustments(preset.get("adjustments", {}), intensity))
        t = max(0.0, min(1.0, intensity / 100.0))
        if preset.get("monochrome"):
            mono_t = max(mono_t, t)
        total_grain += round(preset.get("grain", 0) * t)

    if not scaled_layers:
        return image

    combined = _merge_stack_adjustments(scaled_layers)
    result = apply_adjustments(image, combined)

    if mono_t > 0:
        mono = to_monochrome(result)
        result = Image.blend(result.convert("RGB"), mono, mono_t) if mono_t < 1.0 else mono

    if total_grain:
        result = add_grain(result, min(100, total_grain))

    return result


# ===================== PRESET LOOKUP / LISTING =====================

def _load_store() -> dict:
    if not _STORE_PATH.exists():
        return dict(_EMPTY_STORE)
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("presets", [])
        data.setdefault("favorites", [])
        return data
    except (json.JSONDecodeError, OSError):
        # Corrupt/unreadable store -- fail safe to empty rather than
        # crashing the Filters view. The bad file is left on disk as-is
        # in case the user wants to recover it by hand.
        return dict(_EMPTY_STORE)


def _save_store(store: dict) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def get_preset(preset_id: str):
    """Looks up a preset (builtin or custom) by id, or None if not found."""
    if preset_id in _BUILTIN_BY_ID:
        return _BUILTIN_BY_ID[preset_id]
    store = _load_store()
    for p in store["presets"]:
        if p["id"] == preset_id:
            return p
    return None


def list_presets() -> list:
    """
    Returns every preset (9 builtins first, then custom presets in
    creation order), each annotated with is_builtin/is_favorite -- ready
    to hand straight to the frontend as JSON.
    """
    store = _load_store()
    favorites = set(store["favorites"])

    out = []
    for p in BUILTIN_PRESETS:
        out.append({**p, "is_builtin": True, "is_favorite": p["id"] in favorites})
    for p in store["presets"]:
        out.append({**p, "is_builtin": False, "is_favorite": p["id"] in favorites})
    return out


# ===================== CUSTOM PRESET CRUD =====================

def save_custom_preset(name: str, adjustments: dict, grain: int = 0, monochrome: bool = False) -> dict:
    """
    Creates a new custom preset from a name + adjustment dict (typically
    "save the Editor/Filters view's current effective sliders as a
    preset" from the frontend -- only keys that differ from
    DEFAULT_ADJUSTMENTS are kept, same shape as a BUILTIN_PRESETS entry).
    Returns the new preset dict.
    """
    name = (name or "").strip() or "Untitled Preset"
    clean_adjustments = {
        k: v for k, v in (adjustments or {}).items()
        if k in DEFAULT_ADJUSTMENTS and v != DEFAULT_ADJUSTMENTS[k]
    }
    preset = {
        "id": f"custom_{uuid.uuid4().hex[:10]}",
        "name": name,
        "description": "",
        "adjustments": clean_adjustments,
        "grain": max(0, min(100, int(grain or 0))),
        "monochrome": bool(monochrome),
        "created_at": int(time.time()),
    }
    store = _load_store()
    store["presets"].append(preset)
    _save_store(store)
    return preset


def rename_preset(preset_id: str, new_name: str) -> dict:
    if preset_id in _BUILTIN_BY_ID:
        raise ValueError("Built-in presets can't be renamed.")
    store = _load_store()
    for p in store["presets"]:
        if p["id"] == preset_id:
            p["name"] = (new_name or "").strip() or p["name"]
            _save_store(store)
            return p
    raise ValueError("Preset not found.")


def duplicate_preset(preset_id: str) -> dict:
    """
    Works for both a builtin (copies it into a new, editable custom
    preset) and an existing custom preset.
    """
    source = get_preset(preset_id)
    if not source:
        raise ValueError("Preset not found.")
    copy = {
        "id": f"custom_{uuid.uuid4().hex[:10]}",
        "name": f"{source['name']} Copy",
        "description": source.get("description", ""),
        "adjustments": dict(source.get("adjustments", {})),
        "grain": source.get("grain", 0),
        "monochrome": source.get("monochrome", False),
        "created_at": int(time.time()),
    }
    store = _load_store()
    store["presets"].append(copy)
    _save_store(store)
    return copy


def delete_preset(preset_id: str) -> None:
    if preset_id in _BUILTIN_BY_ID:
        raise ValueError("Built-in presets can't be deleted.")
    store = _load_store()
    before = len(store["presets"])
    store["presets"] = [p for p in store["presets"] if p["id"] != preset_id]
    if len(store["presets"]) == before:
        raise ValueError("Preset not found.")
    store["favorites"] = [f for f in store["favorites"] if f != preset_id]
    _save_store(store)


def set_favorite(preset_id: str, favorite: bool) -> None:
    """Favoriting works on builtin AND custom presets."""
    if not get_preset(preset_id):
        raise ValueError("Preset not found.")
    store = _load_store()
    is_fav = preset_id in store["favorites"]
    if favorite and not is_fav:
        store["favorites"].append(preset_id)
        _save_store(store)
    elif not favorite and is_fav:
        store["favorites"] = [f for f in store["favorites"] if f != preset_id]
        _save_store(store)


def export_preset(preset_id: str, dest_path: str) -> None:
    """Writes a single preset out as a standalone, shareable .json file."""
    preset = get_preset(preset_id)
    if not preset:
        raise ValueError("Preset not found.")
    exportable = {k: v for k, v in preset.items() if k not in ("is_builtin", "is_favorite")}
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(exportable, f, indent=2)


def import_preset(source_path: str) -> dict:
    """
    Reads a preset .json file (as written by export_preset) and adds it
    as a new custom preset -- always given a fresh id, so importing the
    same file twice, or a file that collides with an existing preset's
    id, never overwrites anything already saved.
    """
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "adjustments" not in data or "name" not in data:
        raise ValueError("Not a valid PixelForge preset file.")

    return save_custom_preset(
        name=data.get("name", "Imported Preset"),
        adjustments=data.get("adjustments", {}),
        grain=data.get("grain", 0),
        monochrome=data.get("monochrome", False),
    )