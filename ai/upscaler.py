# ai/upscaler.py
#
# PHASE 9 -- AI Upscaling.
#
# Genuinely AI-powered: Real-ESRGAN (RRDBNet), a trained super-resolution
# model. Same "ai/ = trained-model features, core/ = orchestration +
# classic deterministic processing" split as ai/face_detector.py and
# ai/bg_remover.py -- see their header notes for the full rationale.
#
# ---------------------------------------------------------------------
# WHY ONNX RUNTIME, NOT torch/realesrgan/basicsr (per the spec's Phase 9
# "Core" checklist note): `basicsr`, the package the original torch/
# realesrgan inference path depends on, is unmaintained and fails to
# build on this project's installed Python (3.14) -- `pip install
# basicsr` errors with `KeyError: '__version__'` inside its own setup
# script. That's an environment incompatibility, not a project bug.
# `onnxruntime` is already a pinned dependency (requirements.txt) and
# Real-ESRGAN's RRDBNet architecture has ONNX-exported weights that run
# on it directly -- no torch/torchvision/realesrgan/basicsr needed for
# inference. `torch` itself stays installed (other phases may still want
# it) but is no longer a hard requirement for this module.
#
# WHICH MODEL / DOWNLOAD SOURCE: RealESRGAN_x4plus -- the general-
# purpose RRDBNet checkpoint the upstream Real-ESRGAN README itself
# recommends as the default for everyday photos (as opposed to the
# anime-specialised variants). ONNX export sourced from Qualcomm AI Hub
# Models' official Hugging Face mirror, a 1:1 export of Xintao Wang's
# original weights (same BSD-3-Clause license chain as the .pth
# original). ~65MB, matching the spec's estimate. Downloads once,
# locally cached afterwards under cache/models/upscale/ (same
# "cache/generated/" convention ai/image_generator.py and
# ai/local_generator.py already use for their own first-run downloads,
# and the same "no mandatory cloud after first use" rule rembg's u2net
# already follows in ai/bg_remover.py).
#
# HONESTY NOTE (if MODEL_URL ever goes stale): model hosting on
# third-party sites can move. If the download fails, this module raises
# a clear UpscaleError explaining exactly that, rather than a bare
# network traceback -- update MODEL_URL below and it's fixed everywhere.
# Nothing else in this module cares where the bytes came from as long as
# the file loads as a valid ONNX RRDBNet x4 model.
#
# ONLY ONE CHECKPOINT IS DOWNLOADED. "2x" and "Custom" targets do not
# get a second model -- the x4 checkpoint is the only trained pass that
# ever touches the pixels. For a 2x (or smaller-than-4x custom) request,
# the SAME AI x4 pass runs first and the AI output is then Lanczos-
# resampled down to the exact requested size. This is a deliberate,
# honestly-documented trade-off: every output still gets genuine AI
# detail recovery feeding it (never a plain resize pretending to be an
# AI upscale), it avoids a second multi-tens-of-MB download for one
# dropdown option, and it mirrors a technique the upstream Real-ESRGAN
# CLI itself uses internally (its `--outscale` flag also doesn't require
# outscale to equal the model's native factor).
#
# DENOISE / ARTIFACT REDUCTION: the official Real-ESRGAN "denoise
# strength" slider blends between two *paired* trained checkpoints
# (x4plus vs. its no-denoise sibling). This module only ships one
# checkpoint, so "denoise strength" and "artifact reduction" here are
# implemented as a classical (non-AI) post-pass -- OpenCV's
# fastNlMeansDenoisingColored / bilateral filter -- blended over the AI
# output at a strength the user controls. This is stated plainly rather
# than implied to be a second trained model, matching this project's
# existing pattern of documenting where the line between "AI" and
# "classical" actually sits (see core/enhancer.py's header, and
# ai/face_detector.py's docstring).
#
# FACE-AWARE ENHANCEMENT: reuses ai/face_detector.py's real trained
# detector (Haar Cascade) -- no second face model. Detected face regions
# get a mild secondary sharpening pass blended in with a feathered edge,
# so restored detail is concentrated on faces without a hard seam.
#
# SAFETY: tile-based processing (default 256px tiles + 16px overlap,
# blended at the seams) keeps peak RAM bounded regardless of source
# image size -- required on the 16GB / no-CUDA target hardware. A
# megapixel cap, a rough RAM headroom check, and a disk-space check (the
# same shutil.disk_usage pattern as core/batch_processor.py::
# check_disk_space) all run before any pixels are touched, and every
# long loop below checks a cancel Event so the frontend's Cancel button
# actually stops mid-run instead of only hiding the progress bar.

from pathlib import Path

import gc
import os
import shutil
import tempfile
import threading
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - onnxruntime is a pinned dependency
    ort = None

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a pinned dependency
    psutil = None

from ai.face_detector import detect_faces

# ===================== CONSTANTS =====================

from core.paths import model_dir as _model_dir
_CACHE_DIR = _model_dir("upscale")
_MODEL_FILENAME = "RealESRGAN_x4plus.onnx"
_MODEL_PATH = _CACHE_DIR / _MODEL_FILENAME

# See the "WHICH MODEL / DOWNLOAD SOURCE" header note above for why this
# specific URL. Update here (only) if the mirror ever moves.
MODEL_URL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/real_esrgan_x4plus/releases/v0.61.0/real_esrgan_x4plus-onnx-float.zip"
MODEL_NATIVE_SCALE = 4
_MODEL_APPROX_MB = 65  # matches the spec's own estimate; used for the "downloading ~65MB" UI copy

# "Maximum output resolution / megapixel safety cap" -- keeps a runaway
# custom-resolution request (or a huge source photo at 4x) from trying
# to allocate an output bitmap that would exhaust RAM on the 16GB target
# hardware. 60 MP is generous headroom above a real 8K frame (33.2MP)
# while still catching accidental/typo'd custom sizes.
MAX_OUTPUT_MEGAPIXELS = 60

# Tile size chosen for the i7-7600U / HD 620 / no-CUDA target hardware --
# small enough that a single tile's forward pass stays fast and RAM-light
# on CPU, large enough that tile seams (blended anyway, see _blend below)
# stay unnoticeable. 16px overlap on each side gives the blend real
# gradient room to work with.
DEFAULT_TILE_SIZE = 128
DEFAULT_TILE_OVERLAP = 16

SCALE_CHOICES = ("original", "2x", "4x", "custom")
OUTPUT_FORMATS = ("jpg", "png", "webp", "tiff")


class UpscaleError(Exception):
    """Raised for any expected/explainable failure -- surfaced verbatim to the UI."""


class UpscaleCancelled(Exception):
    """Raised internally when a cancel Event fires mid-run; caught by the caller."""


# ===================== MODEL STATUS / DOWNLOAD =====================

def is_model_cached() -> bool:
    return _MODEL_PATH.is_file() and _MODEL_PATH.stat().st_size > 1_000_000


def _available_providers() -> list:
    if ort is None:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        return []


def model_status() -> dict:
    """
    Reports whether the Real-ESRGAN weights are cached, and which
    onnxruntime execution provider would be used (GPU vs CPU), so the
    UI can show an honest "Model ready (CPU)" / "First use downloads
    ~65MB" state instead of only discovering it on first Upscale click
    -- same pattern as ai/image_generator.py::local_model_status.
    """
    providers = _available_providers()
    gpu_provider = next(
        (p for p in providers if p in ("CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider")),
        None,
    )
    return {
        "onnxruntime_installed": ort is not None,
        "cached": is_model_cached(),
        "model_path": str(_MODEL_PATH) if is_model_cached() else "",
        "approx_download_mb": _MODEL_APPROX_MB,
        "provider": "gpu" if gpu_provider else "cpu",
        "provider_name": gpu_provider or "CPUExecutionProvider",
        "cpu_only": gpu_provider is None,
    }


def _download_model(on_progress=None, cancel_event: threading.Event = None) -> None:
    """
    Streams MODEL_URL to a temp file with real progress (not a silent
    stall -- "Model download progress + caching" requirement), then
    atomically renames it into place so a cancelled/failed download
    never leaves a half-written file that looks cached.
    """
    import requests  # pinned dependency (requirements.txt)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _CACHE_DIR / f".{_MODEL_FILENAME}.part"

    try:
        with requests.get(MODEL_URL, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or (_MODEL_APPROX_MB * 1_000_000)
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if cancel_event is not None and cancel_event.is_set():
                        raise UpscaleCancelled("Download cancelled.")
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        pct = min(99, int(downloaded / total * 100)) if total else -1
                        on_progress(pct, f"Downloading AI Upscale model... ({downloaded // 1_000_000}MB)")
        if tmp_path.stat().st_size < 1_000_000:
            raise UpscaleError("Downloaded model file looks incomplete or invalid.")

        # The official Qualcomm AI Hub asset is a ZIP containing the ONNX
        # model and its external weights (.data file). Extract them to the cache path.
        try:
            with zipfile.ZipFile(tmp_path, "r") as zf:
                onnx_names = [
                    name for name in zf.namelist()
                    if name.lower().endswith(".onnx")
                ]
                if not onnx_names:
                    raise UpscaleError("Downloaded model archive does not contain an ONNX model.")
                
                for item_name in zf.namelist():
                    if item_name.endswith("/"):
                        continue
                    
                    if item_name == onnx_names[0]:
                        out_path = _MODEL_PATH
                    else:
                        out_path = _CACHE_DIR / Path(item_name).name

                    with zf.open(item_name, "r") as src_f, open(out_path, "wb") as dst_f:
                        while True:
                            if cancel_event is not None and cancel_event.is_set():
                                raise UpscaleCancelled("Download cancelled.")
                            chunk = src_f.read(1024 * 1024)
                            if not chunk:
                                break
                            dst_f.write(chunk)
        except UpscaleCancelled:
            _safe_unlink(_MODEL_PATH)
            raise
        except zipfile.BadZipFile as exc:
            raise UpscaleError("Downloaded AI Upscale model archive is invalid.") from exc
        finally:
            _safe_unlink(tmp_path)

        if not _MODEL_PATH.is_file() or _MODEL_PATH.stat().st_size < 1_000_000:
            _safe_unlink(_MODEL_PATH)
            raise UpscaleError("Extracted AI Upscale model file looks incomplete or invalid.")
    except UpscaleCancelled:
        _safe_unlink(tmp_path)
        raise
    except Exception as exc:  # noqa: BLE001
        _safe_unlink(tmp_path)
        raise UpscaleError(
            "Couldn't download the AI Upscale model "
            f"(~{_MODEL_APPROX_MB}MB, one-time only). Check your internet "
            f"connection and try again. ({exc})"
        ) from exc


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            os.unlink(path)
    except OSError:
        pass


# ===================== SESSION (lazy, cached, GPU->CPU fallback) =====================

_session = None
_session_provider = None


def _get_session(on_progress=None, cancel_event: threading.Event = None):
    """
    Lazily loads the ONNX InferenceSession, downloading the model first
    if this is the very first use. Reused for every image in the
    process afterwards (loading is the slow part, same convention as
    ai/bg_remover.py::_get_session for rembg).

    GPU/CPU automatic fallback: tries GPU-capable providers first (if
    onnxruntime was installed with GPU support and the machine actually
    has one), and transparently falls back to CPU -- both if no GPU
    provider is available at all, AND if creating the GPU session itself
    raises (e.g. an out-of-memory/driver error), per the spec's
    "GPU/CPU automatic fallback ... on an out-of-memory GPU error, fall
    back to CPU automatically instead of crashing" requirement.
    """
    global _session, _session_provider

    if ort is None:
        raise UpscaleError(
            "onnxruntime isn't installed. Run: pip install onnxruntime "
            "(it's already listed in requirements.txt)."
        )

    if _session is not None:
        return _session, _session_provider

    if not is_model_cached():
        _download_model(on_progress=on_progress, cancel_event=cancel_event)

    if cancel_event is not None and cancel_event.is_set():
        raise UpscaleCancelled("Upscale cancelled.")

    if on_progress:
        on_progress(-1, "Loading AI Upscale model...")

    available = _available_providers()
    gpu_providers = [p for p in available if p in ("CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider")]

    session = None
    provider_used = "cpu"
    if gpu_providers:
        try:
            session = ort.InferenceSession(str(_MODEL_PATH), providers=gpu_providers + ["CPUExecutionProvider"])
            used = session.get_providers()[0] if session.get_providers() else ""
            provider_used = "gpu" if used in gpu_providers else "cpu"
        except Exception:  # noqa: BLE001 -- GPU init failed (driver/OOM/etc): fall back silently
            session = None

    if session is None:
        # THERMAL/CPU-LOAD NOTE: with no SessionOptions, onnxruntime
        # defaults intra_op_num_threads to the machine's full logical
        # core count and pins them all near 100% for the whole
        # inference call -- unlike the FFmpeg video-encode path
        # elsewhere in this app, this is genuine sustained neural-net
        # math on every core at once, which is why this feature runs
        # noticeably hotter than anything else here on CPU-only/no-CUDA
        # hardware (this module's own SAFETY note above already flags
        # "16GB / no-CUDA target hardware" as the design target). Capping
        # to (core_count - 1) leaves one core free for the UI/OS so the
        # app doesn't feel frozen and the chip has a little headroom,
        # at the cost of the upscale itself taking a bit longer.
        # Override with PIXELFORGE_UPSCALE_THREADS=<n> (e.g. set lower
        # on a thin/fanless laptop that runs hot even at core_count-1,
        # or =0 to go back to onnxruntime's unrestricted default).
        opts = ort.SessionOptions()
        env_threads = os.environ.get("PIXELFORGE_UPSCALE_THREADS")
        if env_threads is not None:
            n_threads = int(env_threads) or None  # "0" -> None -> onnxruntime's own default
        else:
            n_threads = max(1, (os.cpu_count() or 2) - 1)
        if n_threads:
            opts.intra_op_num_threads = n_threads
        session = ort.InferenceSession(str(_MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
        provider_used = "cpu"

    _session = session
    _session_provider = provider_used
    return _session, _session_provider


def unload_model() -> None:
    """Frees the cached session (e.g. Settings > Cache > Clear could call this)."""
    global _session, _session_provider
    _session = None
    _session_provider = None
    gc.collect()


def clear_model_cache() -> dict:
    """Deletes the downloaded weights from disk (Settings > Cache > Clear)."""
    unload_model()
    try:
        if _MODEL_PATH.exists():
            os.unlink(_MODEL_PATH)
        for data_file in _CACHE_DIR.glob("*.data"):
            os.unlink(data_file)
        return {"ok": True}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


# ===================== SAFETY CHECKS =====================

def estimate_output_size(width: int, height: int, scale: str, custom_width=None, custom_height=None) -> tuple:
    """Returns (out_w, out_h) for the requested scale, WITHOUT running any model."""
    if scale == "original":
        return width, height
    if scale == "2x":
        return width * 2, height * 2
    if scale == "4x":
        return width * MODEL_NATIVE_SCALE, height * MODEL_NATIVE_SCALE
    if scale == "custom":
        cw = int(custom_width) if custom_width else width
        ch = int(custom_height) if custom_height else height
        return max(1, cw), max(1, ch)
    raise UpscaleError(f"Unknown scale option: {scale!r}")


def safety_check(width: int, height: int, scale: str, output_folder: str = "",
                  custom_width=None, custom_height=None) -> dict:
    """
    Runs every "Safety & Performance" checklist item that can be
    evaluated BEFORE any pixel work starts: megapixel cap, RAM headroom,
    disk space. Returns {"ok": bool, "warnings": [...], "blocking": [...],
    "estimated_output": {"width", "height", "megapixels"}} -- "blocking"
    entries should stop the run, "warnings" are shown but don't.
    """
    out_w, out_h = estimate_output_size(width, height, scale, custom_width, custom_height)
    megapixels = round((out_w * out_h) / 1_000_000, 1)

    warnings, blocking = [], []

    if megapixels > MAX_OUTPUT_MEGAPIXELS:
        blocking.append(
            f"Requested output ({out_w}x{out_h}, {megapixels}MP) exceeds the "
            f"{MAX_OUTPUT_MEGAPIXELS}MP safety cap. Choose a smaller scale or custom size."
        )

    if psutil is not None:
        try:
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            # Rough rule of thumb: tiled processing keeps peak RAM well
            # below the full output size, but very large outputs still
            # need headroom for the assembled result + intermediate
            # buffers. ~4 bytes/px (RGBA float intermediate) x safety factor.
            needed_gb = (out_w * out_h * 4 * 3) / (1024 ** 3)
            if avail_gb < needed_gb:
                warnings.append(
                    f"Low available RAM ({avail_gb:.1f}GB free, ~{needed_gb:.1f}GB recommended for "
                    f"this output size). Processing may be slow or the app may need to be restarted."
                )
        except Exception:  # noqa: BLE001
            pass

    out_folder = output_folder or str(Path(tempfile.gettempdir()))
    try:
        # Same shutil.disk_usage pattern as core/batch_processor.py::check_disk_space.
        free_bytes = shutil.disk_usage(out_folder).free
        estimated_bytes = out_w * out_h * 3  # rough uncompressed estimate; real files are usually smaller
        if free_bytes < estimated_bytes:
            warnings.append("Low disk space on the output drive for an image this size.")
    except OSError:
        pass

    return {
        "ok": not blocking,
        "warnings": warnings,
        "blocking": blocking,
        "estimated_output": {"width": out_w, "height": out_h, "megapixels": megapixels},
    }


# ===================== CORE INFERENCE (tiled) =====================

def _run_tile(session, tile_rgb: np.ndarray) -> np.ndarray:
    """One forward pass: HWC uint8 RGB tile -> HWC uint8 RGB tile at 4x."""
    inp = tile_rgb.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]  # NCHW
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: inp})[0]
    out = np.clip(output[0], 0.0, 1.0)
    out = np.transpose(out, (1, 2, 0))
    return (out * 255.0).round().astype(np.uint8)


def _upscale_ai(session, image_rgb: np.ndarray, tile_size: int, tile_overlap: int,
                 on_progress=None, cancel_event: threading.Event = None,
                 progress_range: tuple = (0, 100)) -> np.ndarray:
    
    h, w = image_rgb.shape[:2]
    scale = MODEL_NATIVE_SCALE
    out_h, out_w = h * scale, w * scale

    # Helper function: 128x128 fixed size requirement ko pora karne ke liye padding
    def process_padded_tile(tile_img):
        real_th, real_tw = tile_img.shape[:2]
        pad_h = max(0, tile_size - real_th)
        pad_w = max(0, tile_size - real_tw)
        
        if pad_h > 0 or pad_w > 0:
            # Kinaron ko reflect kar ke pad karein taake AI weird artifacts na banaye
            padded = np.pad(tile_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            out = _run_tile(session, padded)
            # Model output scale (4x) ho chuka hai, is liye crop bhi 4x se hoga
            return out[:real_th * scale, :real_tw * scale]
        
        return _run_tile(session, tile_img)

    # Fast path: Image itni choti hai ke aik hi tile mein aagayi
    if h <= tile_size and w <= tile_size:
        if on_progress:
            on_progress(progress_range[0], "Running AI model...")
        result = process_padded_tile(image_rgb)
        if on_progress:
            on_progress(progress_range[1], "Running AI model...")
        return result

    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

    stride = tile_size - tile_overlap
    tiles_y = list(range(0, h, stride))
    tiles_x = list(range(0, w, stride))
    total_tiles = len(tiles_y) * len(tiles_x)
    done = 0

    # Default cooldown deliberately set higher than a "just take the
    # edge off" pause -- explicitly requested to prioritize a cooler
    # laptop over speed, even if that means the run takes noticeably
    # longer wall-clock.
    _cooldown_env = os.environ.get("PIXELFORGE_UPSCALE_COOLDOWN")
    cooldown_seconds = float(_cooldown_env) if _cooldown_env is not None else 0.15

    def _feather_mask(th, tw):
        wy = np.ones(th, dtype=np.float32)
        wx = np.ones(tw, dtype=np.float32)
        ramp = min(tile_overlap * scale, th // 2, tw // 2)
        if ramp > 0:
            fall = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp))
            wy[:ramp] = fall
            wy[-ramp:] = fall[::-1]
            wx[:ramp] = fall
            wx[-ramp:] = fall[::-1]
        return (wy[:, None] * wx[None, :])[..., None]

    for ty in tiles_y:
        for tx in tiles_x:
            if cancel_event is not None and cancel_event.is_set():
                raise UpscaleCancelled("Upscale cancelled.")

            y0, x0 = ty, tx
            y1, x1 = min(ty + tile_size, h), min(tx + tile_size, w)
            tile = image_rgb[y0:y1, x0:x1]

            # Naya padded tile processor yahan call hoga
            tile_out = process_padded_tile(tile)
            
            th, tw = tile_out.shape[:2]
            mask = _feather_mask(th, tw)

            oy0, ox0 = y0 * scale, x0 * scale
            output[oy0:oy0 + th, ox0:ox0 + tw] += tile_out.astype(np.float32) * mask
            weight[oy0:oy0 + th, ox0:ox0 + tw] += mask

            done += 1
            if on_progress:
                pct = progress_range[0] + int((done / total_tiles) * (progress_range[1] - progress_range[0]))
                on_progress(pct, f"AI upscaling tile {done}/{total_tiles}...")

            # THERMAL COOLDOWN: a deliberate pause after every tile so the
            # CPU gets a breather between forward passes instead of
            # sitting pinned at max load for the whole tiled run --
            # trades real wall-clock time for a lower sustained
            # temperature, which is what was explicitly asked for on this
            # thin-and-light laptop (a fan-cooled tower doesn't need this;
            # a 15W ultrabook chip running back-to-back CNN inference for
            # 49+ tiles straight does). Checks cancel_event mid-sleep in
            # small slices so hitting Cancel during a cooldown still
            # stops immediately rather than waiting out the full pause.
            # Tune with PIXELFORGE_UPSCALE_COOLDOWN=<seconds> (e.g. "0"
            # to disable, "1.0" for an even longer breather); default
            # chosen to be clearly noticeable rather than marginal.
            if cooldown_seconds > 0 and done < total_tiles:
                remaining = cooldown_seconds
                while remaining > 0:
                    if cancel_event is not None and cancel_event.is_set():
                        raise UpscaleCancelled("Upscale cancelled.")
                    slice_s = min(0.1, remaining)
                    time.sleep(slice_s)
                    remaining -= slice_s

    weight = np.maximum(weight, 1e-6)
    result = (output / weight).round().clip(0, 255).astype(np.uint8)
    return result
# ===================== DENOISE / ARTIFACT REDUCTION (classical post-pass) =====================

def _apply_denoise(image_rgb: np.ndarray, strength: float) -> np.ndarray:
    """
    strength: 0-100. See the DENOISE header note above -- this is a
    classical (non-AI) OpenCV denoise blended over the AI output, not a
    second trained model.
    """
    if strength <= 0:
        return image_rgb
    h_luma = max(1, round(strength / 100 * 10))  # fastNlMeans "h" param, small range keeps detail
    denoised = cv2.fastNlMeansDenoisingColored(image_rgb, None, h_luma, h_luma, 7, 21)
    alpha = min(1.0, strength / 100)
    return cv2.addWeighted(denoised, alpha, image_rgb, 1 - alpha, 0)


def _apply_artifact_reduction(image_rgb: np.ndarray, strength: float) -> np.ndarray:
    """
    strength: 0-100. Edge-preserving bilateral smoothing to soften
    ringing/compression-artifact halos the AI pass can amplify on a
    low-quality source, without a full denoise pass's detail loss.
    """
    if strength <= 0:
        return image_rgb
    d = 5
    sigma = max(5, round(strength / 100 * 60))
    smoothed = cv2.bilateralFilter(image_rgb, d, sigma, sigma)
    alpha = min(1.0, strength / 100)
    return cv2.addWeighted(smoothed, alpha, image_rgb, 1 - alpha, 0)


# ===================== FACE-AWARE ENHANCEMENT =====================

def _apply_face_aware_boost(source_rgb: np.ndarray, upscaled_rgb: np.ndarray) -> np.ndarray:
    """
    Reuses ai/face_detector.py's real trained detector (no second face
    model) to find faces in the ORIGINAL (pre-upscale) image, maps the
    boxes into the upscaled image's coordinate space, and blends in a
    mild unsharp-mask pass confined to those regions with a feathered
    edge -- concentrating recovered detail on faces the way a portrait-
    aware upscaler should, without a hard seam around the box.
    """
    try:
        faces = detect_faces(Image.fromarray(source_rgb))
    except Exception:  # noqa: BLE001 -- face-aware is a bonus, must never fail the whole upscale
        return upscaled_rgb

    if not faces:
        return upscaled_rgb

    out_h, out_w = upscaled_rgb.shape[:2]
    result = upscaled_rgb.copy().astype(np.float32)

    # Mild unsharp mask, applied once to the whole frame -- cheap, and
    # then only its MASKED contribution (faces + feathered margin) gets
    # blended back in below, so non-face regions are untouched (spec:
    # "Don't alter non-face regions").
    blurred = cv2.GaussianBlur(upscaled_rgb, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(upscaled_rgb, 1.5, blurred, -0.5, 0).astype(np.float32)

    mask = np.zeros((out_h, out_w), dtype=np.float32)
    for face in faces:
        fx, fy, fw, fh = face["x"], face["y"], face["w"], face["h"]
        pad = 0.15  # small margin around the detected box so hairline/ears aren't cut off
        x0 = max(0, int((fx - fw * pad) * out_w))
        y0 = max(0, int((fy - fh * pad) * out_h))
        x1 = min(out_w, int((fx + fw * (1 + pad)) * out_w))
        y1 = min(out_h, int((fy + fh * (1 + pad)) * out_h))
        if x1 <= x0 or y1 <= y0:
            continue
        face_mask = np.zeros((out_h, out_w), dtype=np.float32)
        face_mask[y0:y1, x0:x1] = 1.0
        feather_px = max(4, int(min(x1 - x0, y1 - y0) * 0.2))
        k = feather_px * 2 + 1
        face_mask = cv2.GaussianBlur(face_mask, (k, k), 0)
        mask = np.maximum(mask, face_mask)

    mask3 = mask[..., None]
    result = sharpened * mask3 + result * (1 - mask3)
    return result.round().clip(0, 255).astype(np.uint8)


# ===================== ORCHESTRATOR =====================

def upscale_image(
    image: Image.Image,
    scale: str = "4x",
    custom_width=None,
    custom_height=None,
    denoise_strength: float = 0,
    artifact_reduction: float = 0,
    face_aware: bool = False,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    preserve_aspect: bool = True,
    on_progress=None,
    cancel_event: threading.Event = None,
) -> Image.Image:
    """
    Main entry point. Returns a new PIL Image -- never mutates `image`.

    scale: "original" (no AI pass, returns image unchanged -- lets the
      UI's Original/2x/4x/Custom selector share one code path), "2x",
      "4x", or "custom" (custom_width/custom_height required, at least
      one of them).

    preserve_aspect: when scale == "custom" and only one of
      custom_width/custom_height is given, computes the other from the
      source aspect ratio.

    on_progress(percent, label): percent 0-100, or -1 for "in progress,
      unknown length" (model download/load). Called from whatever thread
      this function runs on -- callers on a background thread (see
      ui/bridge.py's upscale Slots) already forward this straight into
      Signal.emit, which is thread-safe.

    Raises UpscaleError for any explainable failure, UpscaleCancelled if
    cancel_event fires mid-run.
    """
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    had_alpha = image.mode == "RGBA"
    alpha_channel = image.split()[-1] if had_alpha else None
    rgb_image = image.convert("RGB") if had_alpha else image

    src_w, src_h = rgb_image.width, rgb_image.height

    if scale == "original":
        return image.copy()

    if scale not in SCALE_CHOICES:
        raise UpscaleError(f"Unknown scale option: {scale!r}")

    if scale == "custom":
        if not custom_width and not custom_height:
            raise UpscaleError("Custom resolution needs at least a width or a height.")
        if preserve_aspect:
            if custom_width and not custom_height:
                custom_height = round(custom_width * src_h / src_w)
            elif custom_height and not custom_width:
                custom_width = round(custom_height * src_w / src_h)
        target_w = max(1, int(custom_width or src_w))
        target_h = max(1, int(custom_height or src_h))
    elif scale == "2x":
        target_w, target_h = src_w * 2, src_h * 2
    else:  # "4x"
        target_w, target_h = src_w * MODEL_NATIVE_SCALE, src_h * MODEL_NATIVE_SCALE

    check = safety_check(src_w, src_h, "custom", custom_width=target_w, custom_height=target_h)
    if check["blocking"]:
        raise UpscaleError(" ".join(check["blocking"]))

    session, provider = _get_session(on_progress=on_progress, cancel_event=cancel_event)
    if cancel_event is not None and cancel_event.is_set():
        raise UpscaleCancelled("Upscale cancelled.")

    if provider == "cpu" and on_progress:
        on_progress(-1, "CPU processing may be slow.")

    source_np = np.array(rgb_image)

    ai_result = _upscale_ai(
        session, source_np, tile_size, tile_overlap,
        on_progress=on_progress, cancel_event=cancel_event,
        progress_range=(5, 80),
    )

    if cancel_event is not None and cancel_event.is_set():
        raise UpscaleCancelled("Upscale cancelled.")

    if face_aware:
        if on_progress:
            on_progress(85, "Enhancing detected faces...")
        ai_result = _apply_face_aware_boost(source_np, ai_result)

    if denoise_strength:
        if on_progress:
            on_progress(90, "Reducing noise...")
        ai_result = _apply_denoise(ai_result, denoise_strength)

    if artifact_reduction:
        if on_progress:
            on_progress(93, "Reducing artifacts...")
        ai_result = _apply_artifact_reduction(ai_result, artifact_reduction)

    if on_progress:
        on_progress(96, "Finishing...")

    result_img = Image.fromarray(ai_result, mode="RGB")

    # The AI pass is always native 4x; resample to the EXACT requested
    # size (2x, or a custom size that isn't a clean multiple) now that
    # the AI detail recovery has already happened -- see the "ONLY ONE
    # CHECKPOINT" header note.
    if (result_img.width, result_img.height) != (target_w, target_h):
        result_img = result_img.resize((target_w, target_h), Image.LANCZOS)

    if had_alpha:
        alpha_resized = alpha_channel.resize((target_w, target_h), Image.LANCZOS)
        result_img = result_img.convert("RGBA")
        result_img.putalpha(alpha_resized)

    del source_np, ai_result
    gc.collect()  # release RAM before returning, same golden rule as core/batch_processor.py

    if on_progress:
        on_progress(100, "Done.")

    return result_img


# ===================== OUTPUT WORDING HELPERS =====================
#
# "Important wording rule (reconfirmed this session): never claim a poor
# source becomes genuine native 4K quality. Correct wording is 'AI
# Upscaled 4K', not 'true native 4K.'" -- kept as one small helper so
# every call site (bridge.py, batch_processor.py, frontend) uses the
# same honest phrasing instead of composing it ad hoc.

def output_label(width: int, height: int) -> str:
    long_side = max(width, height)
    if long_side >= 3800:
        return f"AI Upscaled 4K ({width}x{height})"
    if long_side >= 1900:
        return f"AI Upscaled ({width}x{height})"
    return f"AI Upscaled ({width}x{height})"