# ai/local_generator.py
#
# OFFLINE / LOW-BANDWIDTH IMAGE GENERATION
#
# For users who can't do a multi-GB download (SD-Turbo/FLUX/full SD are
# 4-7GB). This uses a small distilled Stable Diffusion checkpoint instead.
#
# Model: segmind/small-sd (~730MB total, fp16 ~370MB) OR
#        segmind/tiny-sd  (~500MB, fastest, slightly lower quality)
#
# Both are real diffusers checkpoints on Hugging Face -- not a toy/fake
# model. Quality is a step below full SD/FLUX but is genuine prompt-based
# generation, good enough for drafts, previews, and casual use.
#
# First run downloads the model once (cached under ~/.cache/huggingface).
# After that, generation is fully offline -- no network, no API key.

import os
from pathlib import Path

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "cache" / "generated"

# Pick "tiny" for the smallest download, "small" for a bit more quality.
MODEL_CHOICES = {
    "tiny": "segmind/tiny-sd",     # ~500MB
    "small": "segmind/small-sd",   # ~730MB
}

DEFAULT_LOCAL_MODEL = "tiny"

_pipe_cache = {}


class LocalGenerationError(Exception):
    pass


def _load_pipeline(model_key: str = DEFAULT_LOCAL_MODEL):
    """Lazily loads and caches the diffusion pipeline in memory."""
    if model_key in _pipe_cache:
        return _pipe_cache[model_key]

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise LocalGenerationError(
            "Local generation needs 'diffusers' and 'torch'.\n"
            "Install with:\n"
            "  pip install diffusers torch transformers accelerate safetensors"
        ) from exc

    model_id = MODEL_CHOICES.get(model_key, MODEL_CHOICES[DEFAULT_LOCAL_MODEL])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,   # keep footprint small; UI handles moderation
    )
    pipe = pipe.to(device)

    if device == "cpu":
        pipe.enable_attention_slicing()

    _pipe_cache[model_key] = pipe
    return pipe


def generate_local(
    prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    num_images: int = 1,
    model_key: str = DEFAULT_LOCAL_MODEL,
    on_progress=None,
) -> list:
    """
    Generates images fully offline using a small local diffusion model.
    Returns a list of saved file paths.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise LocalGenerationError("Prompt is empty.")

    if on_progress:
        on_progress(-1, f"Loading local model ({model_key})... first run downloads it once.")

    pipe = _load_pipeline(model_key)

    if on_progress:
        on_progress(10, "Generating locally...")

    def _cb(step, timestep, latents):
        if on_progress and steps > 0:
            pct = int(10 + (step / steps) * 85)
            on_progress(pct, f"Step {step}/{steps}")

    result = pipe(
        prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        num_images_per_prompt=num_images,
        callback=_cb,
        callback_steps=1,
    )

    if on_progress:
        on_progress(100, "Done.")

    # Return raw PIL images -- image_generator.py's caller handles
    # saving/naming for both providers uniformly.
    return list(result.images)


def local_status() -> dict:
    """Reports whether local generation dependencies are installed."""
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError:
        return {
            "available": False,
            "reason": "missing_deps",
            "message": (
                "Local generation needs extra packages (~200MB, not the model itself):\n"
                "  pip install diffusers torch transformers accelerate safetensors\n\n"
                "The model itself (500-730MB) downloads once on first generation."
            ),
        }
    return {"available": True}