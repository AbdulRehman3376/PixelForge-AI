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


# Models exposed by the current PixelForge UI/config.
# These are Pollinations.ai's real hosted model identifiers.
POLLINATIONS_MODELS = {
    "flux": "FLUX.1 [schnell] -- Black Forest Labs' high-quality, fast general-purpose model (default)",
    "flux-realism": "Photorealistic generation",
    "flux-anime": "Anime-style generation",
    "turbo": "Fast SDXL-Turbo-class generation",
}


ASPECT_RATIOS = {
    "square": (1024, 1024),
    "portrait": (832, 1024),
    "portrait_tall": (768, 1024),
    "landscape": (1024, 832),
    "landscape_wide": (1024, 768),
}


_MAX_DIMENSION = 1024
_MIN_DIMENSION = 256
DEFAULT_MODEL = "flux-realism"
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

def cloud_provider_status() -> dict:
    """
    Status for the app's actual online provider (Pollinations). Pollinations
    needs no key and is free/always reachable, so this is a simple
    always-ready status.
    """
    return {
        "available": True,
        "reason": "ready",
        "message": "Pollinations (free, online) is ready -- no API key needed.",
        "provider": "pollinations",
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
# POLLINATIONS.AI -- FREE, NO API KEY REQUIRED
# ============================================================
def _generate_pollinations(spec: dict, on_progress=None) -> list:
    """
    Free image generation via pollinations.ai -- no signup, no API key.
    Uses their hosted Flux/SDXL-class model over a simple GET request.

    Passes model / enhance / negative through to Pollinations, which
    previously weren't sent at all -- so every request silently used
    Pollinations' bare default regardless of what the app picked, and
    the negative prompt the user typed was never actually applied.
    """
    import random
    import urllib.parse

    prompt = (spec.get("prompt") or "").strip()
    if not prompt:
        raise GenerationError("Prompt is empty.")

    negative_prompt = (spec.get("negative_prompt") or "").strip()
    model = spec.get("model_id") or DEFAULT_MODEL
    if model not in POLLINATIONS_MODELS:
        model = DEFAULT_MODEL

    width = spec.get("width", 1024)
    height = spec.get("height", 1024)
    num_images = spec.get("num_images", 1)

    if on_progress:
        on_progress(-1, "Requesting image from Pollinations...")

    results = []
    encoded_prompt = urllib.parse.quote(prompt)

    for i in range(num_images):
        # Random seed per image so multiple requests don't return the same result.
        seed = spec.get("seed") or random.randint(0, 2**31 - 1)
        seed_i = seed + i

        params = {
            "width": width,
            "height": height,
            "seed": seed_i,
            "model": model,
            # Pollinations' own server-side prompt-enhancement (LLM rewrite
            # of the prompt before generation) -- noticeably sharper,
            # more detailed results for the same input prompt.
            "enhance": "true",
            "nologo": "true",
        }
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?{urllib.parse.urlencode(params)}"
        )

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "PixelForge/1.0"},
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                image_bytes = response.read()
        except urllib.error.URLError as exc:
            raise GenerationError(
                f"Pollinations request failed: {exc}"
            ) from exc

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results.append({"image": img})

        if on_progress:
            pct = int(((i + 1) / max(num_images, 1)) * 100)
            on_progress(pct, f"Generated {i + 1}/{num_images}")

    return results


# ============================================================
# LOCAL MODE
# ============================================================

def _generate_local(spec: dict, on_progress=None) -> list:
    """
    Low-bandwidth local mode.

    Uses a small distilled Stable Diffusion checkpoint (segmind/tiny-sd,
    ~500MB, or segmind/small-sd, ~730MB) instead of full SD-Turbo/FLUX
    (4-7GB). Real prompt-to-image generation, smaller download, slightly
    lower fidelity than the full-size models.

    Requires: pip install diffusers torch transformers accelerate safetensors
    Model itself downloads once on first use (cached afterwards).
    """
    from ai.local_generator import generate_local, local_status, LocalGenerationError

    status = local_status()
    if not status["available"]:
        raise GenerationError(status["message"])

    model_key = get_setting("local_image_model", "tiny")

    try:
        images = generate_local(
            prompt=spec["prompt"],
            width=spec["width"],
            height=spec["height"],
            steps=min(spec.get("steps", 20), 25),  # small models need fewer steps
            num_images=spec["num_images"],
            model_key=model_key,
            on_progress=on_progress,
        )
    except LocalGenerationError as exc:
        raise GenerationError(str(exc)) from exc

    # Match the {"image": PIL.Image, ...} shape the caller expects
    # (same shape _generate_pollinations's items use).
    return [{"image": img} for img in images]


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

    return {
        "ok": True,
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