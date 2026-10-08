# ai/image_generator.py
#
# PHASE 7 -- AI Image Generation
#
# Two no-mandatory-cost providers: Pollinations.ai (free hosted, no
# signup/API key) and a local offline diffusion model (see
# ai/local_generator.py). A previous "AgentRouter" cloud integration was
# removed after its API key was found to be unauthorized/blocked by the
# provider -- the dead code for it (status check, config reader, and
# the request function itself) has been fully deleted rather than left
# unused; only the one-line backward-compat redirect in generate_image()
# remains, so an old config.json with a leftover "agentrouter"/"cloud"
# provider value still starts up cleanly on Pollinations instead.
#

import base64
import io
import json
import uuid
import urllib.request
import urllib.error
from pathlib import Path

from PIL import Image

from core.settings import get_setting


_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "cache" / "generated"


# UI "style" id -> (real Pollinations model, style words added to the prompt).
#
# BUGFIX: "flux-realism" and "flux-anime" are NOT real Pollinations model
# names any more (current ones: flux, zimage, turbo, gptimage, seedream,
# kontext ...). Sending them made the server fall back to a random default,
# which is why results looked unrelated to the prompt. The UI ids are kept
# so the frontend doesn't change, but they are translated here: the real
# model is "flux" and the *style* is pushed through the prompt instead.
STYLE_PRESETS = {
    "flux-realism": (
        "flux",
        "photorealistic, ultra detailed, sharp focus, natural lighting, "
        "realistic skin and textures, shot on 85mm lens, high resolution",
    ),
    "flux": ("flux", ""),
    "flux-anime": (
        "flux",
        "anime style illustration, clean line art, vibrant colors, "
        "detailed, studio quality",
    ),
    "turbo": ("turbo", ""),
    # Newer hosted models (best quality; usually need a free API key).
    "zimage": ("zimage", ""),
    "seedream": ("seedream", ""),
    "gptimage": ("gptimage", ""),
}

POLLINATIONS_MODELS = {k: v[0] for k, v in STYLE_PRESETS.items()}

# Tried in order if the requested model errors out.
_MODEL_FALLBACKS = ("flux", "turbo")

POLLINATIONS_KEY_SETTING = "pollinations_api_key"
_GEN_URL = "https://gen.pollinations.ai/image/"
_LEGACY_URL = "https://image.pollinations.ai/prompt/"


ASPECT_RATIOS = {
    "square": (1024, 1024),
    "portrait": (832, 1024),
    "portrait_tall": (768, 1024),
    "landscape": (1024, 832),
    "landscape_wide": (1024, 768),
}


_MAX_DIMENSION = 1024
_MIN_DIMENSION = 256
DEFAULT_MODEL = "flux"
DEFAULT_STEPS = 30


class GenerationError(Exception):
    """AI image generation failed."""
    pass


def _snap8(value: int) -> int:
    """Keep dimensions within limits and aligned to multiples of 8."""
    value = max(_MIN_DIMENSION, min(_MAX_DIMENSION, int(value)))
    return value - (value % 8) if value % 8 else value


def resolve_dimensions(
    aspect_ratio: str,
    custom_width=None,
    custom_height=None,
) -> tuple:
    """Resolve UI aspect ratio into output dimensions."""
    if aspect_ratio == "custom" and custom_width and custom_height:
        return _snap8(custom_width), _snap8(custom_height)

    width, height = ASPECT_RATIOS.get(
        aspect_ratio,
        ASPECT_RATIOS["square"],
    )
    return _snap8(width), _snap8(height)


# ============================================================
# PROVIDER STATUS
# ============================================================

def _pollinations_key() -> str:
    key = (get_setting(POLLINATIONS_KEY_SETTING, "") or "").strip()
    if not key:
        import os
        key = (os.environ.get("POLLINATIONS_API_KEY", "") or "").strip()
    return key


def cloud_provider_status() -> dict:
    """Pollinations status. Works without a key, but anonymous requests are
    throttled and may carry a watermark; a free key removes both."""
    if _pollinations_key():
        return {
            "available": True,
            "reason": "key_set",
            "message": "Pollinations ready with API key (no watermark).",
            "provider": "pollinations",
            "has_key": True,
        }
    return {
        "available": True,
        "reason": "no_key",
        "message": (
            "Pollinations works without a key, but images may have a "
            "watermark. Add a free key from enter.pollinations.ai to remove it."
        ),
        "provider": "pollinations",
        "has_key": False,
    }


def local_model_status() -> dict:
    """Status for the offline/local provider. Delegates to
    ai.local_generator.local_status()."""
    from ai.local_generator import local_status

    return local_status()


# ============================================================
# RESPONSE / IMAGE DECODING
# ============================================================

def _extract_image_items(response_data: dict) -> list:
    """
    Accept common image response structures without changing the
    external UI/output architecture.
    """

    if not isinstance(response_data, dict):
        return []

    images = response_data.get("images")

    if isinstance(images, list):
        return images

    data = response_data.get("data")

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        nested_images = data.get("images")
        if isinstance(nested_images, list):
            return nested_images

    return []


def _download_or_decode_image(
    image_item,
    token: str,
) -> Image.Image:
    """Convert a URL/base64/dict image response into a PIL image."""

    image_source = image_item

    if isinstance(image_item, dict):
        image_source = (
            image_item.get("url")
            or image_item.get("image_url")
            or image_item.get("image")
            or image_item.get("b64_json")
            or image_item.get("base64")
        )

        if isinstance(image_source, dict):
            image_source = image_source.get("url")

    if not isinstance(image_source, str) or not image_source.strip():
        raise GenerationError(
            "Pollinations returned an invalid image result."
        )

    image_source = image_source.strip()

    # Generated image URL.
    if image_source.startswith(("http://", "https://")):
        request = urllib.request.Request(
            image_source,
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "PixelForge/Phase7",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                image_bytes = response.read()
        except urllib.error.HTTPError:
            # Some generated URLs are public and do not accept the auth header.
            request = urllib.request.Request(
                image_source,
                headers={
                    "User-Agent": "PixelForge/Phase7",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                image_bytes = response.read()

        return Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    # data:image/png;base64,...
    if image_source.startswith("data:") and "," in image_source:
        image_source = image_source.split(",", 1)[1]

    # Raw base64.
    try:
        image_bytes = base64.b64decode(
            image_source,
            validate=False,
        )
        return Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except Exception as exc:
        raise GenerationError(
            f"Could not decode generated image: {exc}"
        ) from exc


# ============================================================
# POLLINATIONS.AI
# ============================================================

def _looks_like_image(data: bytes) -> bool:
    return (
        data[:8] == b"\x89PNG\r\n\x1a\n"
        or data[:3] == b"\xff\xd8\xff"
        or data[:4] == b"RIFF"       # webp
        or data[:6] in (b"GIF87a", b"GIF89a")
    )


def _fetch_image(url: str, key: str, timeout: int = 150) -> Image.Image:
    """
    One HTTP request -> validated PIL image.

    Raises GenerationError with a readable message for auth/rate-limit/
    server problems, and for "200 OK but not actually an image" (Pollinations
    sometimes returns a tiny error picture or JSON/HTML instead).
    """
    import time

    headers = {"User-Agent": "PixelForge/1.1", "Accept": "image/*"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    last_error = "unknown error"
    for attempt in range(3):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
        except urllib.error.HTTPError as exc:
            code = exc.code
            if code in (401, 402, 403):
                raise GenerationError(
                    f"Pollinations refused the request (HTTP {code}). "
                    "This model needs a valid API key -- get a free one at "
                    "enter.pollinations.ai and paste it in the API key box."
                ) from exc
            last_error = f"HTTP {code}"
            if code == 429 or code >= 500:
                wait = 4 * (attempt + 1)
                try:
                    wait = max(wait, int(exc.headers.get("Retry-After", 0)))
                except (TypeError, ValueError):
                    pass
                time.sleep(min(wait, 20))
                continue
            raise GenerationError(
                f"Pollinations request failed (HTTP {code})."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            time.sleep(2 * (attempt + 1))
            continue

        if len(data) < 2000 or not _looks_like_image(data):
            snippet = data[:120].decode("utf-8", "ignore").strip()
            last_error = f"server did not return an image ({snippet or 'empty'})"
            if attempt >= 1:
                break
            time.sleep(2)
            continue

        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            last_error = f"corrupt image data: {exc}"
            continue

    raise GenerationError(f"Pollinations failed: {last_error}")


def _generate_pollinations(spec: dict, on_progress=None) -> list:
    """
    Online generation through Pollinations.

    With an API key (setting "pollinations_api_key") the new
    gen.pollinations.ai endpoint is used -> no watermark, higher limits.
    Without a key it still works via the same endpoint / the legacy one,
    but the result may be watermarked. Result includes a "notice" flag so
    the UI can tell the user why.
    """
    import random
    import urllib.parse

    prompt = (spec.get("prompt") or "").strip()
    if not prompt:
        raise GenerationError("Prompt is empty.")

    style_id = spec.get("model_id") or DEFAULT_MODEL
    if style_id not in STYLE_PRESETS:
        style_id = DEFAULT_MODEL
    real_model, style_words = STYLE_PRESETS[style_id]

    # Add the style words once, unless the prompt already says so.
    full_prompt = prompt
    if style_words and style_words.split(",")[0].strip().lower() not in prompt.lower():
        full_prompt = f"{prompt}, {style_words}"

    negative_prompt = (spec.get("negative_prompt") or "").strip()
    width = spec.get("width", 1024)
    height = spec.get("height", 1024)
    num_images = spec.get("num_images", 1)
    key = _pollinations_key()

    models_to_try = [real_model] + [
        m for m in _MODEL_FALLBACKS if m != real_model
    ]

    if on_progress:
        on_progress(-1, "Requesting image from Pollinations...")

    encoded = urllib.parse.quote(full_prompt, safe="")
    base_seed = spec.get("seed") or random.randint(0, 2**31 - 1)
    results = []

    for i in range(num_images):
        seed_i = int(base_seed) + i
        image = None
        last_exc = None

        for model in models_to_try:
            params = {
                "model": model,
                "width": width,
                "height": height,
                "seed": seed_i,
                "nologo": "true",
                "safe": "false",
            }
            if negative_prompt:
                params["negative_prompt"] = negative_prompt
            query = urllib.parse.urlencode(params)

            urls = [f"{_GEN_URL}{encoded}?{query}"]
            if not key:
                # No key: also try the older anonymous endpoint.
                urls.append(f"{_LEGACY_URL}{encoded}?{query}")

            for url in urls:
                try:
                    image = _fetch_image(url, key)
                    break
                except GenerationError as exc:
                    last_exc = exc
                    # Auth problem with a key set is final -- no point retrying.
                    if key and "API key" in str(exc):
                        raise
            if image is not None:
                break

        if image is None:
            raise last_exc or GenerationError("Pollinations returned no image.")

        results.append({"image": image, "seed": seed_i})

        if on_progress:
            pct = int(((i + 1) / max(num_images, 1)) * 100)
            on_progress(pct, f"Generated {i + 1}/{num_images}")

    return results


# ============================================================
# LOCAL MODE
# ============================================================

def _generate_local(spec: dict, on_progress=None) -> list:
    """
    Offline generation on this PC (no watermark, no API).

    The model is chosen by the setting "local_image_model":
    tiny / small / turbo / realvis (realistic photos) / flux (best, heavy).
    If the setting was never changed, the best model for this PC is picked
    automatically (realvis on an 8GB+ NVIDIA GPU, otherwise turbo).
    """
    from ai.local_generator import (
        MODEL_INFO,
        REALISM_PROMPT,
        LocalGenerationError,
        effective_model,
        generate_local,
        local_status,
        recommended_model,
    )

    status = local_status()
    if not status["available"]:
        raise GenerationError(status["message"])

    model_key = (get_setting("local_image_model", "") or "").strip().lower()
    if model_key not in MODEL_INFO:
        model_key = recommended_model()
    # Too-heavy model for this PC -> swapped for a safe one (no freezing).
    model_key, _note = effective_model(model_key)
    info = MODEL_INFO[model_key]

    # Each model has a native resolution (SD-1.5 class: 512, SDXL/FLUX: 1024).
    # Going above it makes SD-1.5 models repeat/duplicate subjects.
    width, height = spec["width"], spec["height"]
    native = info["native"]
    longest = max(width, height)
    if longest > native:
        scale = native / longest
        width = max(256, int(width * scale) // 8 * 8)
        height = max(256, int(height * scale) // 8 * 8)

    prompt = spec["prompt"]
    # Realistic look: add photo words for realistic models, or whenever the
    # user picked the "Realistic Photo" style.
    if info["realistic"] or spec.get("model_id") == "flux-realism":
        if "photorealistic" not in prompt.lower():
            prompt = f"{prompt}, {REALISM_PROMPT}"

    seed = spec.get("seed")

    try:
        images = generate_local(
            prompt=prompt,
            negative_prompt=spec.get("negative_prompt", ""),
            width=width,
            height=height,
            seed=seed,
            steps=spec.get("steps"),
            num_images=spec["num_images"],
            model_key=model_key,
            on_progress=on_progress,
        )
    except MemoryError as exc:
        raise GenerationError(
            "Not enough memory. Close other programs and try again."
        ) from exc
    except LocalGenerationError as exc:
        raise GenerationError(str(exc)) from exc

    return [{"image": img, "seed": seed} for img in images]


# ============================================================
# PUBLIC API
# ============================================================

def generate(
    prompt: str,
    negative_prompt: str = "",
    aspect_ratio: str = "square",
    custom_width=None,
    custom_height=None,
    num_images: int = 1,
    steps: int = None,
    seed=None,
    provider: str = "pollinations",
    model_id: str = None,
    on_progress=None,
) -> dict:
    """
    Generate images.

    Pollinations (free, online):
        True prompt-based AI image generation, no API key needed.
        No local model download.

    Local:
        No automatic model download.
        If no verified local generative model is installed, a clear
        message is returned instead of pretending enhancement is
        image generation.
    """

    prompt = (prompt or "").strip()

    if not prompt:
        raise GenerationError(
            "Prompt likho - image ka description daalo!"
        )

    if len(prompt) > 2000:
        raise GenerationError(
            "Prompt bohat lamba hai (2000 characters maximum)."
        )

    width, height = resolve_dimensions(
        aspect_ratio,
        custom_width,
        custom_height,
    )

    spec = {
        "prompt": prompt,
        "negative_prompt": (
            negative_prompt or ""
        ).strip(),
        "width": width,
        "height": height,
        "num_images": max(
            1,
            min(int(num_images), 4),
        ),
        "steps": (
            steps
            if steps is not None
            else DEFAULT_STEPS
        ),
        "seed": seed,
        "model_id": (
            model_id
            or get_setting(
                "ai_image_default_model",
                DEFAULT_MODEL,
            )
            or DEFAULT_MODEL
        ),
    }

    provider = (
        provider
        or get_setting(
            "ai_image_generator_provider",
            "pollinations",
        )
    ).strip().lower()

    # Cloud/AgentRouter removed -- any old saved setting now falls
    # through to the free Pollinations provider instead.
    if provider in ("agentrouter", "cloud"):
        provider = "pollinations"

    if provider == "local":
        results = _generate_local(
            spec,
            on_progress=on_progress,
        )

    elif provider == "pollinations":
        results = _generate_pollinations(
            spec,
            on_progress=on_progress,
        )

    else:
        raise GenerationError(
            f"Unknown AI image provider '{provider}'."
        )

    _OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = []
    batch_id = uuid.uuid4().hex[:10]

    for index, item in enumerate(results):
        out_path = (
            _OUTPUT_DIR
            / f"gen_{batch_id}_{index}.png"
        )

        item["image"].save(
            out_path,
            format="PNG",
        )

        saved.append({
            "path": str(out_path),
            "seed": item.get("seed"),
            "width": width,
            "height": height,
        })

    if on_progress:
        on_progress(
            100,
            "Complete! ✓",
        )

    notice = ""
    if provider == "pollinations" and not _pollinations_key():
        notice = (
            "No API key set: image may carry a Pollinations watermark. "
            "Add a free key (enter.pollinations.ai) to remove it."
        )

    return {
        "ok": True,
        "notice": notice,
        "provider": provider,
        "prompt": prompt,
        "negative_prompt": spec["negative_prompt"],
        "width": width,
        "height": height,
        "model_id": spec["model_id"],
        "images": saved,
    }


def regenerate(
    previous_spec: dict,
    on_progress=None,
) -> dict:
    """Repeat a previous generation request."""

    return generate(
        prompt=previous_spec.get(
            "prompt",
            "",
        ),
        negative_prompt=previous_spec.get(
            "negative_prompt",
            "",
        ),
        aspect_ratio=previous_spec.get(
            "aspect_ratio",
            "custom",
        ),
        custom_width=previous_spec.get(
            "width",
        ),
        custom_height=previous_spec.get(
            "height",
        ),
        num_images=previous_spec.get(
            "num_images",
            1,
        ),
        steps=previous_spec.get(
            "steps",
        ),
        seed=previous_spec.get(
            "seed",
        ),
        provider=previous_spec.get(
            "provider",
            "pollinations",
        ),
        model_id=previous_spec.get(
            "model_id",
        ),
        on_progress=on_progress,
    )


def cleanup_generated_cache(
    keep_latest: int = 50,
) -> int:
    """Delete old generated images and keep the latest files."""

    try:
        files = sorted(
            _OUTPUT_DIR.glob("gen_*.png"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return 0

    removed = 0

    for file_path in files[max(keep_latest, 0):]:
        try:
            file_path.unlink()
            removed += 1
        except OSError:
            pass

    return removed