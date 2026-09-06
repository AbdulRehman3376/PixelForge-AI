# core/prompt_engine.py
#
# PHASE 7 -- Local Prompt Engine.
#
# WHAT THIS IS
# ------------
# A local, deterministic, keyword/rule-based parser that turns a plain
# English (or mixed-language, best-effort) editing/generation prompt
# into a structured spec: subject, style, mood, lighting, color palette,
# background, composition and quality cues -- WITHOUT requiring an LLM
# or a mandatory paid API. This is the "Prompt Understanding" half of
# Phase 7; ai/image_generator.py (the "AI Image Generation" half) reads
# the structured spec this module produces.
#
# WHERE THE AI IS (and isn't)
# ----------------------------
# There is deliberately NO model in this file, same placement decision
# already made for core/smart_pipeline.py (Phase 6): this is a rule
# table over words, not pattern recognition over pixels. Keeping prompt
# UNDERSTANDING classical keeps the "usable without paid APIs" success
# criterion true even if a user never wants to touch the (optional,
# opt-in) cloud image-generation provider in ai/image_generator.py.
#
# FEATURES (spec checklist)
# --------------------------
#   - Synonym recognition   -> _SYNONYMS / _CATEGORY_LOOKUP, resolve_token()
#   - Prompt preview         -> preview_prompt()
#   - Conflict detection     -> _CONFLICT_PAIRS, detect_conflicts()
#   - Unknown instruction handling -> "unrecognized" list in parse_prompt(),
#     never raises on odd input
#   - Prompt history          -> add_history()/get_history()/
#     delete_history_entry()/clear_history(), persisted to disk
#
# DESIGN NOTES
# ------------
#   - Plain JSON store at data/prompt_history.json, same "local files,
#     no database" approach as core/filters.py's custom preset store
#     and core/settings.py's config.json -- consistent with this
#     project's pattern until Phase 13's real database/ lands.
#   - Parsing never raises. Bad/empty input returns a well-formed empty
#     spec rather than throwing, so a Slot in ui/bridge.py never needs a
#     bespoke try/except per case -- the existing generic one is enough.
#   - Word matching is intentionally simple (lowercase, punctuation-
#     stripped, whole-word/phrase matching over a fixed vocabulary) --
#     fast, predictable, and fully offline, matching this project's
#     classical-first placement strategy (see the top of the spec's
#     "AI PLACEMENT STRATEGY" section).

import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "prompt_history.json"
_HISTORY_LIMIT = 200  # hard ceiling; UI/settings can request fewer

# ============================================================
# SYNONYM / VOCABULARY TABLES
# ============================================================
# Each category maps a CANONICAL value -> list of words/phrases that
# should resolve to it. Longer phrases are checked before single words
# (see _sorted_phrases) so e.g. "golden hour" wins over the bare word
# "hour" (which isn't even in the table, but the principle matters for
# things like "soft light" vs "light").

_SYNONYMS = {
    "style": {
        "photorealistic": ["photorealistic", "realistic", "photo realistic", "real life", "lifelike", "hyperrealistic", "hyper realistic"],
        "digital art": ["digital art", "digital painting", "digital illustration", "concept art"],
        "oil painting": ["oil painting", "oil paint", "painterly", "classical painting"],
        "watercolor": ["watercolor", "watercolour", "water color"],
        "anime": ["anime", "manga", "japanese animation"],
        "cartoon": ["cartoon", "cartoonish", "toon"],
        "3d render": ["3d render", "3d rendered", "cgi", "octane render", "unreal engine", "blender render"],
        "sketch": ["sketch", "pencil sketch", "line drawing", "line art", "doodle"],
        "cyberpunk": ["cyberpunk", "cyber punk", "neon futuristic"],
        "fantasy art": ["fantasy art", "fantasy style", "mythical", "storybook"],
        "minimalist": ["minimalist", "minimal", "clean and simple"],
        "vintage": ["vintage", "retro", "old school", "nostalgic"],
        "pixel art": ["pixel art", "8 bit", "8-bit", "16 bit", "16-bit", "pixelated"],
    },
    "mood": {
        "happy": ["happy", "joyful", "cheerful", "upbeat", "positive"],
        "sad": ["sad", "melancholic", "melancholy", "gloomy", "somber", "sombre"],
        "calm": ["calm", "peaceful", "serene", "tranquil", "relaxed"],
        "dramatic": ["dramatic", "intense", "powerful", "epic"],
        "mysterious": ["mysterious", "eerie", "enigmatic", "cryptic"],
        "romantic": ["romantic", "dreamy", "loving", "tender"],
        "energetic": ["energetic", "vibrant", "lively", "dynamic"],
        "dark": ["dark mood", "ominous", "sinister", "foreboding"],
        "whimsical": ["whimsical", "playful", "quirky", "fun"],
        "nostalgic": ["nostalgic", "sentimental", "wistful"],
    },
    "lighting": {
        "golden hour": ["golden hour", "sunset light", "sunrise light", "magic hour"],
        "soft light": ["soft light", "soft lighting", "diffused light", "gentle light"],
        "hard light": ["hard light", "harsh light", "harsh lighting", "direct sunlight"],
        "backlit": ["backlit", "back lit", "rim light", "rim lighting", "silhouette light"],
        "studio light": ["studio light", "studio lighting", "softbox"],
        "neon light": ["neon light", "neon lighting", "neon glow"],
        "candlelight": ["candlelight", "candle light", "warm glow"],
        "moonlight": ["moonlight", "moon light", "night light"],
        "overcast": ["overcast", "cloudy light", "flat light"],
        "dramatic lighting": ["dramatic lighting", "chiaroscuro", "high contrast light"],
        "bright": ["bright", "well lit", "well-lit", "sunny"],
        "dim": ["dim", "low light", "dimly lit", "dark lighting"],
    },
    "color": {
        "warm tones": ["warm tones", "warm colors", "warm colours", "warm palette"],
        "cool tones": ["cool tones", "cool colors", "cool colours", "cool palette"],
        "monochrome": ["monochrome", "black and white", "grayscale", "greyscale"],
        "pastel": ["pastel", "pastel colors", "pastel colours", "soft colors"],
        "vibrant colors": ["vibrant", "vivid colors", "vivid colours", "saturated colors"],
        "muted colors": ["muted", "desaturated", "faded colors", "faded colours"],
        "earth tones": ["earth tones", "earthy colors", "natural palette"],
        "neon colors": ["neon colors", "neon colours", "neon palette"],
        "sepia": ["sepia", "sepia tone"],
    },
    "background": {
        "studio backdrop": ["studio background", "studio backdrop", "plain background", "solid background"],
        "outdoor nature": ["outdoor", "nature background", "forest background", "outdoors"],
        "urban city": ["city background", "urban background", "cityscape", "street background"],
        "beach": ["beach background", "beach", "seaside", "ocean background"],
        "night sky": ["night sky", "starry sky", "starry background"],
        "blurred background": ["blurred background", "bokeh background", "bokeh", "out of focus background"],
        "indoor room": ["indoor", "interior background", "room background"],
        "abstract background": ["abstract background", "gradient background", "textured background"],
        "no background": ["no background", "transparent background", "isolated on white", "isolated on plain"],
    },
    "composition": {
        "close-up": ["close up", "close-up", "closeup", "macro shot"],
        "portrait framing": ["portrait shot", "headshot", "bust shot"],
        "full body": ["full body", "full-body", "head to toe"],
        "wide shot": ["wide shot", "wide angle", "landscape shot", "establishing shot"],
        "centered": ["centered", "centred", "center composition"],
        "rule of thirds": ["rule of thirds", "off center", "off-center composition"],
        "top-down": ["top down", "top-down", "bird's eye", "birds eye", "aerial view"],
        "low angle": ["low angle", "worm's eye", "worms eye", "from below"],
        "symmetrical": ["symmetrical", "symmetric composition"],
    },
    "quality": {
        "high detail": ["high detail", "highly detailed", "intricate detail", "fine detail"],
        "sharp focus": ["sharp focus", "in focus", "crisp"],
        "high resolution": ["high resolution", "4k", "8k", "hd", "ultra hd", "high res"],
        "professional quality": ["professional", "professional quality", "studio quality"],
        "cinematic": ["cinematic", "movie still", "film still"],
        "award winning": ["award winning", "award-winning", "masterpiece"],
    },
}

# Flattened lookup: phrase (lowercase) -> (category, canonical value).
# Built once at import time from _SYNONYMS above.
_CATEGORY_LOOKUP = {}
for _category, _canon_map in _SYNONYMS.items():
    for _canonical, _phrases in _canon_map.items():
        for _phrase in _phrases:
            _CATEGORY_LOOKUP[_phrase.lower()] = (_category, _canonical)

# Phrases sorted longest-first so multi-word phrases are matched before
# any shorter word inside them is considered on its own.
_SORTED_PHRASES = sorted(_CATEGORY_LOOKUP.keys(), key=len, reverse=True)

# Words that are common English filler/connective words in an editing
# prompt and shouldn't be reported back as "unrecognized instructions"
# even though they don't map to any category above.
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "but",
    "to", "for", "is", "are", "very", "please", "make", "it", "this",
    "photo", "image", "picture", "shot", "style", "some", "more", "less",
}

# Pairs of canonical values (any category) that contradict each other.
# Checked symmetrically. Kept as (category, value) tuples so a clash is
# only flagged within a semantically meaningful axis (e.g. two
# lighting choices), not across unrelated categories.
_CONFLICT_PAIRS = [
    (("lighting", "bright"), ("lighting", "dim")),
    (("lighting", "golden hour"), ("lighting", "moonlight")),
    (("lighting", "hard light"), ("lighting", "soft light")),
    (("mood", "happy"), ("mood", "sad")),
    (("mood", "calm"), ("mood", "dramatic")),
    (("mood", "happy"), ("mood", "dark")),
    (("color", "warm tones"), ("color", "cool tones")),
    (("color", "vibrant colors"), ("color", "muted colors")),
    (("color", "monochrome"), ("color", "vibrant colors")),
    (("style", "photorealistic"), ("style", "cartoon")),
    (("style", "photorealistic"), ("style", "anime")),
    (("style", "photorealistic"), ("style", "pixel art")),
    (("style", "minimalist"), ("style", "fantasy art")),
    (("composition", "close-up"), ("composition", "wide shot")),
    (("composition", "close-up"), ("composition", "full body")),
    (("background", "no background"), ("background", "outdoor nature")),
    (("background", "no background"), ("background", "urban city")),
    (("background", "no background"), ("background", "beach")),
]

_WORD_SPLIT_RE = re.compile(r"[^a-z0-9]+")


_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")


def _normalize(text: str) -> str:
    """
    Lowercases, strips punctuation (commas, periods, etc. -- so "warm
    tones," matches the phrase "warm tones" the same as "warm tones"
    would), and collapses whitespace.
    """
    lowered = (text or "").strip().lower()
    no_punct = _PUNCT_RE.sub(" ", lowered)
    return re.sub(r"\s+", " ", no_punct).strip()


def resolve_token(phrase: str):
    """
    Looks a single phrase up in the synonym table. Returns
    (category, canonical) or None. Exposed standalone so other modules
    (or a future UI "type ahead") can resolve one term without running
    the full parser.
    """
    return _CATEGORY_LOOKUP.get(_normalize(phrase))


def _extract_matches(normalized_text: str):
    """
    Scans normalized_text for every known phrase (longest-first so
    multi-word phrases win over a shorter word they contain), removing
    matched spans as it goes so nothing is double-counted. Returns
    (matches, remaining_text) where matches is a list of
    {"category", "value", "matched_text"} dicts in first-seen order.
    """
    working = f" {normalized_text} "
    matches = []
    seen_spans = []

    for phrase in _SORTED_PHRASES:
        needle = f" {phrase} "
        idx = working.find(needle)
        if idx == -1:
            continue
        category, canonical = _CATEGORY_LOOKUP[phrase]
        matches.append({"category": category, "value": canonical, "matched_text": phrase})
        # Blank out the matched phrase (keep the surrounding spaces) so
        # a shorter phrase contained within it can't also fire, and so
        # leftover-word detection below doesn't see it either.
        working = working.replace(needle, "  " + " " * len(phrase) + "  ", 1)

    return matches, working.strip()


def _leftover_words(remaining_text: str):
    words = [w for w in _WORD_SPLIT_RE.split(remaining_text) if w]
    return [w for w in words if w not in _STOPWORDS]


def detect_conflicts(matches):
    """
    Given the list of {"category","value",...} matches from a single
    prompt (or prompt+negative-prompt pair), returns a list of
    human-readable conflict warning dicts:
    {"a": "...", "b": "...", "message": "..."}.
    Contradictory instructions are surfaced as a WARNING rather than
    silently picking one side, per the spec's "Conflict detection"
    requirement.
    """
    present = {(m["category"], m["value"]) for m in matches}
    warnings = []
    for left, right in _CONFLICT_PAIRS:
        if left in present and right in present:
            warnings.append(
                {
                    "a": left[1],
                    "b": right[1],
                    "category": left[0],
                    "message": f"\"{left[1]}\" and \"{right[1]}\" contradict each other -- pick one.",
                }
            )
    return warnings


def parse_prompt(prompt: str, negative_prompt: str = "") -> dict:
    """
    The core entry point. Parses a free-text prompt (and optional
    negative prompt) into a structured spec.

    Returns:
    {
      "ok": True,
      "raw_prompt": "...", "raw_negative_prompt": "...",
      "categories": {"style": ["photorealistic"], "lighting": [...], ...},
      "negative_categories": {...},   # same shape, parsed from negative_prompt
      "matches": [{"category","value","matched_text"}, ...],
      "unrecognized": ["foo", "bar"],   # words that matched no category
      "conflicts": [{"a","b","category","message"}, ...],
      "is_empty": bool,
    }
    Never raises -- empty/garbage input yields a well-formed, mostly
    empty spec (is_empty=True) rather than an exception, per the spec's
    "Unknown instruction handling" requirement (graceful, not crashing).
    """
    normalized = _normalize(prompt)
    normalized_negative = _normalize(negative_prompt)

    if not normalized:
        return {
            "ok": True,
            "raw_prompt": prompt or "",
            "raw_negative_prompt": negative_prompt or "",
            "categories": {},
            "negative_categories": {},
            "matches": [],
            "unrecognized": [],
            "conflicts": [],
            "is_empty": True,
        }

    matches, remaining = _extract_matches(normalized)
    unrecognized = _leftover_words(remaining)

    neg_matches = []
    if normalized_negative:
        neg_matches, _ = _extract_matches(normalized_negative)

    categories = {}
    for m in matches:
        categories.setdefault(m["category"], [])
        if m["value"] not in categories[m["category"]]:
            categories[m["category"]].append(m["value"])

    negative_categories = {}
    for m in neg_matches:
        negative_categories.setdefault(m["category"], [])
        if m["value"] not in negative_categories[m["category"]]:
            negative_categories[m["category"]].append(m["value"])

    conflicts = detect_conflicts(matches)

    return {
        "ok": True,
        "raw_prompt": prompt or "",
        "raw_negative_prompt": negative_prompt or "",
        "categories": categories,
        "negative_categories": negative_categories,
        "matches": matches,
        "unrecognized": unrecognized,
        "conflicts": conflicts,
        "is_empty": False,
    }


_CATEGORY_LABELS = {
    "style": "Style",
    "mood": "Mood",
    "lighting": "Lighting",
    "color": "Color",
    "background": "Background",
    "composition": "Composition",
    "quality": "Quality",
}


def preview_prompt(prompt: str, negative_prompt: str = "") -> dict:
    """
    Builds the "Prompt preview" the spec asks for: a friendly, ordered
    summary of what was detected, ready to show the user BEFORE
    anything is applied/generated. Wraps parse_prompt() and adds a
    plain-English summary line + per-category display list.

    Returns parse_prompt()'s dict plus:
      "summary": "Style: Photorealistic. Lighting: Golden Hour. ..."
      "summary_lines": ["Style: Photorealistic", ...]
      "has_warnings": bool
      "has_unrecognized": bool
    """
    spec = parse_prompt(prompt, negative_prompt)
    if spec["is_empty"]:
        spec["summary"] = "Nothing to preview yet -- type a prompt above."
        spec["summary_lines"] = []
        spec["has_warnings"] = False
        spec["has_unrecognized"] = False
        return spec

    lines = []
    for category in ("style", "mood", "lighting", "color", "background", "composition", "quality"):
        values = spec["categories"].get(category)
        if values:
            label = _CATEGORY_LABELS.get(category, category.title())
            lines.append(f"{label}: {', '.join(v.title() for v in values)}")

    if not lines and not spec["unrecognized"]:
        lines.append("No specific style/mood/lighting keywords detected -- the prompt will be used as-is.")

    spec["summary_lines"] = lines
    spec["summary"] = ". ".join(lines) + ("." if lines else "")
    spec["has_warnings"] = bool(spec["conflicts"])
    spec["has_unrecognized"] = bool(spec["unrecognized"])
    return spec


# ============================================================
# PROMPT HISTORY
# ============================================================
# Same "plain JSON on disk" pattern as core/settings.py's config.json
# and core/filters.py's custom_presets.json -- see those modules'
# headers for the reasoning (atomic writes, never-raise reads).

def _load_history() -> list:
    if not _HISTORY_PATH.exists():
        return []
    try:
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []


def _save_history(entries: list) -> None:
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="prompt_history_", suffix=".json", dir=str(_HISTORY_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp_path, _HISTORY_PATH)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def add_history(prompt: str, negative_prompt: str = "", settings: dict = None) -> dict:
    """
    Saves one prompt (with its negative prompt and generation settings,
    e.g. aspect ratio/provider/model) to history so it can be reused
    later ("Prompt history" requirement). De-duplicates back-to-back
    identical prompts (re-running the same prompt just bumps it to the
    top with a fresh timestamp) rather than piling up copies. Returns
    the stored entry.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Can't save an empty prompt to history.")

    entries = _load_history()
    entries = [e for e in entries if e.get("prompt") != prompt or e.get("negative_prompt", "") != (negative_prompt or "")]

    entry = {
        "id": uuid.uuid4().hex[:12],
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "settings": settings or {},
        "created_at": time.time(),
    }
    entries.insert(0, entry)
    entries = entries[:_HISTORY_LIMIT]
    _save_history(entries)
    return entry


def get_history(limit: int = 50) -> list:
    entries = _load_history()
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    if limit <= 0:
        limit = 50
    return entries[:limit]


def delete_history_entry(entry_id: str) -> bool:
    entries = _load_history()
    remaining = [e for e in entries if e.get("id") != entry_id]
    changed = len(remaining) != len(entries)
    if changed:
        _save_history(remaining)
    return changed


def clear_history() -> None:
    _save_history([])


def history_path() -> str:
    return str(_HISTORY_PATH)