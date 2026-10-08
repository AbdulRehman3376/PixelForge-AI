# ai/local_generator.py
#
# OFFLINE IMAGE GENERATION (runs on your own PC, no API, no watermark)
#
# Models (downloaded once from Hugging Face, cached in ~/.cache/huggingface):
#
#   tiny     segmind/tiny-sd             ~0.5 GB  draft quality, any PC
#   small    segmind/small-sd            ~0.7 GB  draft quality, any PC
#   turbo    stabilityai/sd-turbo        ~2 GB    decent, very fast (4 steps)
#   realvis  SG161222/RealVisXL_V4.0     ~7 GB    REALISTIC photos (recommended
#                                                 if you have an NVIDIA GPU, 8GB+ VRAM)
#   flux     black-forest-labs/FLUX.1-schnell ~24 GB  best quality, needs a strong
#                                                 GPU (16GB+ VRAM) and 32GB+ RAM
#
# Without an NVIDIA GPU, realvis/flux still run on CPU but take many minutes
# per image -- use "turbo" there.

from pathlib import Path

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "cache" / "generated"

MODEL_INFO = {
    "tiny": dict(
        repo="segmind/tiny-sd", kind="sd", native=512,
        steps=20, guidance=7.5, fixed_steps=False, realistic=False,
        size="~0.5 GB", label="Tiny (draft)",
    ),
    "small": dict(
        repo="segmind/small-sd", kind="sd", native=512,
        steps=20, guidance=7.5, fixed_steps=False, realistic=False,
        size="~0.7 GB", label="Small (draft)",
    ),
    "turbo": dict(
        repo="stabilityai/sd-turbo", kind="sd", native=512,
        steps=4, guidance=0.0, fixed_steps=True, realistic=False,
        size="~2 GB", label="SD-Turbo (fast)",
    ),
    "realvis": dict(
        repo="SG161222/RealVisXL_V4.0", kind="sdxl", native=1024,
        steps=28, guidance=6.0, fixed_steps=False, realistic=True,
        size="~7 GB", label="RealVisXL (realistic photos)",
    ),
    "flux": dict(
        repo="black-forest-labs/FLUX.1-schnell", kind="flux", native=1024,
        steps=4, guidance=0.0, fixed_steps=True, realistic=True,
        size="~24 GB", label="FLUX.1 schnell (best, heavy)",
    ),
}

# Backward-compatible name used elsewhere.
MODEL_CHOICES = {k: v["repo"] for k, v in MODEL_INFO.items()}
DEFAULT_LOCAL_MODEL = "tiny"

REALISM_PROMPT = (
    "photorealistic, ultra detailed, natural lighting, realistic skin texture, "
    "sharp focus, shot on DSLR, 85mm lens, high resolution"
)
DEFAULT_NEGATIVE = (
    "blurry, low quality, deformed, bad anatomy, extra limbs, extra fingers, "
    "cartoon, illustration, painting, 3d render, watermark, text, cropped"
)

_pipe_cache = {}


class LocalGenerationError(Exception):
    pass


def _device_info() -> dict:
    """GPU / VRAM info so the UI can recommend a model honestly."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "device": "cuda",
                "gpu": props.name,
                "vram_gb": round(props.total_memory / 1024 ** 3, 1),
            }
    except Exception:  # noqa: BLE001
        pass
    return {"device": "cpu", "gpu": "", "vram_gb": 0.0}


def _ram_gb() -> float:
    """Total system RAM in GB (psutil is already in requirements)."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001
        return 8.0


# Minimum hardware each model needs to run WITHOUT hanging the PC.
#   gpu_vram: NVIDIA VRAM in GB (0 = runs fine on CPU)
#   ram: total system RAM in GB
_REQUIREMENTS = {
    "tiny":    dict(gpu_vram=0,  ram=6),
    "small":   dict(gpu_vram=0,  ram=6),
    "turbo":   dict(gpu_vram=0,  ram=8),
    "realvis": dict(gpu_vram=8,  ram=16),
    "flux":    dict(gpu_vram=16, ram=32),
}


def is_model_safe(model_key: str) -> bool:
    req = _REQUIREMENTS.get(model_key)
    if req is None:
        return False
    dev = _device_info()
    if req["gpu_vram"] and (dev["device"] != "cuda" or dev["vram_gb"] < req["gpu_vram"]):
        return False
    return _ram_gb() >= req["ram"]


def safe_models() -> list:
    return [k for k in MODEL_INFO if is_model_safe(k)]


def effective_model(model_key: str):
    """
    Returns (model_key_to_use, note). If the requested model would be too
    heavy for this PC (e.g. RealVisXL on a laptop with no NVIDIA GPU) it is
    swapped for the best model that is safe, instead of freezing the PC.
    """
    if model_key in MODEL_INFO and is_model_safe(model_key):
        return model_key, ""
    fallback = recommended_model()
    if model_key in MODEL_INFO:
        note = (
            f"'{MODEL_INFO[model_key]['label']}' is too heavy for this PC "
            f"(needs a bigger GPU/RAM) -- using {MODEL_INFO[fallback]['label']} instead."
        )
    else:
        note = ""
    return fallback, note


def recommended_model() -> str:
    """Best model that is safe on this machine."""
    for key in ("realvis", "turbo", "small", "tiny"):
        if is_model_safe(key):
            return key
    return "tiny"


def _load_pipeline(model_key: str = DEFAULT_LOCAL_MODEL):
    """Lazily loads and caches the diffusion pipeline in memory."""
    if model_key in _pipe_cache:
        return _pipe_cache[model_key]

    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:
        raise LocalGenerationError(
            "Local generation needs 'diffusers' and 'torch'.\n"
            "Install with:\n"
            "  pip install diffusers torch transformers accelerate safetensors"
        ) from exc

    info = MODEL_INFO.get(model_key) or MODEL_INFO[DEFAULT_LOCAL_MODEL]
    dev = _device_info()
    use_cuda = dev["device"] == "cuda"
    device = "cuda" if use_cuda else "cpu"

    # Only drop all other cached pipelines: SDXL/FLUX are big, keeping two
    # of them in memory at once would exhaust RAM/VRAM.
    _pipe_cache.clear()
    if use_cuda:
        torch.cuda.empty_cache()

    try:
        if info["kind"] == "flux":
            from diffusers import FluxPipeline
            dtype = torch.bfloat16 if use_cuda else torch.float32
            pipe = FluxPipeline.from_pretrained(info["repo"], torch_dtype=dtype)
            if use_cuda:
                pipe.enable_model_cpu_offload()   # fits 16GB cards
            else:
                pipe = pipe.to("cpu")
        else:
            dtype = torch.float16 if use_cuda else torch.float32
            kwargs = dict(torch_dtype=dtype, use_safetensors=True)
            if info["kind"] == "sd":
                kwargs["safety_checker"] = None
            try:
                if use_cuda:
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        info["repo"], variant="fp16", **kwargs
                    )
                else:
                    raise ValueError("cpu: skip fp16 variant")
            except Exception:  # noqa: BLE001  (repo has no fp16 variant)
                pipe = AutoPipelineForText2Image.from_pretrained(info["repo"], **kwargs)

            if use_cuda:
                if info["kind"] == "sdxl" and dev["vram_gb"] < 10:
                    pipe.enable_model_cpu_offload()
                else:
                    pipe = pipe.to("cuda")
                if info["kind"] == "sdxl":
                    pipe.enable_vae_tiling()
            else:
                pipe = pipe.to("cpu")
                pipe.enable_attention_slicing()
    except LocalGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LocalGenerationError(
            f"Could not load local model '{model_key}' ({info['repo']}): {exc}\n"
            "Check your internet connection (first run downloads "
            f"{info['size']}) and free disk space."
        ) from exc

    _pipe_cache[model_key] = pipe
    return pipe


def generate_local(
    prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = None,
    num_images: int = 1,
    model_key: str = DEFAULT_LOCAL_MODEL,
    on_progress=None,
    negative_prompt: str = "",
    seed=None,
) -> list:
    """Generates images fully offline. Returns a list of PIL images."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise LocalGenerationError("Prompt is empty.")

    import os
    import torch

    dev = _device_info()
    on_cpu = dev["device"] != "cuda"

    # Never run a model this PC can't handle -- swap to a safe one.
    model_key, note = effective_model(model_key)
    info = MODEL_INFO[model_key]

    # CPU-only protection: small size, one image at a time. Larger sizes /
    # batches multiply RAM use and time and are what makes laptops freeze.
    if on_cpu:
        longest = max(width, height)
        if longest > 512:
            scale = 512 / longest
            width = max(256, int(width * scale) // 8 * 8)
            height = max(256, int(height * scale) // 8 * 8)
        num_images = 1
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2)))
        except Exception:  # noqa: BLE001
            pass

    if on_progress:
        msg = f"Loading {info['label']}... first run downloads {info['size']} once."
        if note:
            msg = note + " " + msg
        if on_cpu:
            msg += " (CPU mode: ~1-3 min per image, please wait.)"
        on_progress(-1, msg)

    pipe = _load_pipeline(model_key)

    if info["fixed_steps"] or not steps:
        steps = info["steps"]
    else:
        steps = max(10, min(int(steps), 50 if info["kind"] == "sdxl" else 30))
        if info["kind"] == "sdxl":
            steps = max(steps, info["steps"] - 6)

    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        except (TypeError, ValueError):
            generator = None

    if on_progress:
        on_progress(10, "Generating locally...")

    def _cb(pipeline, step, timestep, callback_kwargs):
        if on_progress and steps > 0:
            pct = int(10 + ((step + 1) / steps) * 85)
            on_progress(min(pct, 95), f"Step {step + 1}/{steps}")
        return callback_kwargs

    kwargs = dict(
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=info["guidance"],
        num_images_per_prompt=num_images,
        generator=generator,
        callback_on_step_end=_cb,
    )
    if info["kind"] != "flux" and info["guidance"] > 0:
        kwargs["negative_prompt"] = negative_prompt or DEFAULT_NEGATIVE
    if info["kind"] == "flux":
        kwargs["max_sequence_length"] = 256

    try:
        result = pipe(prompt, **kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise LocalGenerationError(
            "GPU ran out of memory. Pick a smaller size / 1 image at a time, "
            "or choose the 'turbo' local model."
        ) from exc

    if on_progress:
        on_progress(100, "Done.")

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
                "Local generation needs extra packages (not the model itself):\n"
                "  pip install diffusers torch transformers accelerate safetensors\n\n"
                "The model downloads once on first generation."
            ),
        }
    dev = _device_info()
    return {
        "available": True,
        "device": dev["device"],
        "gpu": dev["gpu"],
        "vram_gb": dev["vram_gb"],
        "recommended": recommended_model(),
        "safe_models": safe_models(),
        "ram_gb": _ram_gb(),
        "models": {
            k: {"label": v["label"], "size": v["size"]}
            for k, v in MODEL_INFO.items()
        },
    }