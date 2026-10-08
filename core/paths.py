# core/paths.py
#
# Ek hi jagah jahan se saari heavy cheezein (AI models, caches, temp files,
# QtWebEngine cache) ka location tay hota hai -- taake C: drive par kuch
# na jaye aur models dobara download na hon.
#
# DATA ROOT (default): project folder ke BAGHAL mein "PixelForge_Data".
#   D:\Projects\PixelForge        <- code (isay zip se replace karo to bhi theek)
#   D:\Projects\PixelForge_Data   <- models/cache/temp (code update se safe)
# Kisi aur jagah rakhna ho to environment variable set karo:
#   PIXELFORGE_DATA=E:\PixelForgeData
#
# apply() sab se pehle (main.py ki top par) call hota hai, kisi bhi heavy
# import (torch, rembg, diffusers, tempfile users) se pehle.

import os
import tempfile
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    env = os.environ.get("PIXELFORGE_DATA", "").strip()
    root = Path(env) if env else _PROJECT_DIR.parent / "PixelForge_Data"
    return root


DATA_ROOT = data_root()
MODELS_DIR = DATA_ROOT / "models"
CACHE_DIR = DATA_ROOT / "cache"
TEMP_DIR = DATA_ROOT / "temp"
WEBENGINE_DIR = DATA_ROOT / "webengine"


def model_dir(name: str) -> Path:
    """Models ka folder, e.g. model_dir("upscale"). Banata bhi hai."""
    p = MODELS_DIR / name
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p


def apply() -> None:
    for d in (MODELS_DIR, CACHE_DIR, TEMP_DIR, WEBENGINE_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    env = os.environ
    forced = {
        # Temp files (C:\Users\..\AppData\Local\Temp ki jagah)
        "TEMP": str(TEMP_DIR),
        "TMP": str(TEMP_DIR),
        "TMPDIR": str(TEMP_DIR),
        # rembg (u2net) -- pehle ~/.u2net (C:) mein jata tha
        "U2NET_HOME": str(MODELS_DIR / "u2net"),
        # Hugging Face / diffusers / transformers -- pehle ~/.cache/huggingface
        "HF_HOME": str(MODELS_DIR / "huggingface"),
        "HF_HUB_CACHE": str(MODELS_DIR / "huggingface" / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(MODELS_DIR / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(MODELS_DIR / "huggingface" / "hub"),
        "DIFFUSERS_CACHE": str(MODELS_DIR / "huggingface" / "hub"),
        # torch hub
        "TORCH_HOME": str(MODELS_DIR / "torch"),
        # pooch / generic ~/.cache users
        "XDG_CACHE_HOME": str(CACHE_DIR),
        "PIP_CACHE_DIR": str(CACHE_DIR / "pip"),
        "NUMBA_CACHE_DIR": str(CACHE_DIR / "numba"),
        "MPLCONFIGDIR": str(CACHE_DIR / "matplotlib"),
    }
    for k, v in forced.items():
        env[k] = v  # jaan-boojh kar override: user ke C: wale defaults nahi chahiye

    tempfile.tempdir = str(TEMP_DIR)

    # QtWebEngine (Chromium) apna cache/profile C:\Users\..\AppData\Local mein
    # banata hai -- usay bhi D: par bhejna (Qt ke import se pehle env se).
    flags = env.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    if "--disk-cache-dir" not in flags:
        env["QTWEBENGINE_CHROMIUM_FLAGS"] = (
            flags + f' --disk-cache-dir="{WEBENGINE_DIR / "http_cache"}"'
        ).strip()