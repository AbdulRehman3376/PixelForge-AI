# ai/scene_classifier.py
#
# PHASE 5 -- Optional scene classifier (indoor / outdoor).
#
# WHY THIS FILE EXISTS
# --------------------
# core/analyzer.py's indoor/outdoor read used to be a single heuristic:
# "is there blue at the top of the frame?". That's wrong often enough to
# matter, and it matters more than it looks, because indoor/outdoor feeds
# Phase 6's Smart Pipeline (which look wins) AND Phase 4's Auto Filter.
# Of every possible AI upgrade in this project, a small scene classifier
# is the highest-value one: it makes two existing features smarter
# without adding a model to either of them.
#
# WHAT THIS DOES / DOESN'T DO
# ---------------------------
# This is a LOADER, not a model. It looks for a user-supplied ONNX
# classifier on disk and uses it if present. It NEVER downloads anything
# -- the project's privacy rules are explicit that nothing is required to
# leave the machine, and silently pulling a model on first launch would
# break that. With no model present, classify() returns None and
# core/analyzer.py falls back to its (improved, multi-cue) heuristic. So:
#
#   model installed  -> real classification, scene_source "classifier"
#   no model         -> heuristic,          scene_source "heuristic"
#
# Either way the rest of the app behaves identically; only the confidence
# number changes. Nothing in Phase 6 needs to know which one ran.
#
# TO ENABLE IT
# ------------
#   1. Drop a MobileNet-class ONNX classifier at:
#        models/scene_classifier.onnx
#      (~5-10 MB, CPU-only inference -- fine on the target i7-7600U.
#       onnxruntime is already a dependency via rembg, so there is
#       nothing new to install.)
#   2. Drop its label map at:
#        models/scene_classifier_labels.json
#      shaped as either
#        {"labels": ["indoor", "outdoor", ...]}
#      or, for a Places-style model with many classes,
#        {"labels": [...], "indoor_classes": [...], "outdoor_classes": [...]}
#      where the *_classes lists hold label names to bucket into each side.
#
# CPU-ONLY: the session is created with the CPU execution provider
# explicitly -- no CUDA anywhere in this project (target hardware has
# Intel HD 620 graphics).

import json
from pathlib import Path

import numpy as np

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_PATH = _MODELS_DIR / "scene_classifier.onnx"
_LABELS_PATH = _MODELS_DIR / "scene_classifier_labels.json"

# ImageNet normalization -- what almost every MobileNet/ResNet-family
# classifier expects. Overridable via the label file's "preprocess" key
# for a model trained differently.
_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)
_DEFAULT_SIZE = 224

# Cached across calls -- loading an ONNX session per photo would dominate
# the runtime. `_load_failed` makes the "no model installed" path free
# after the first check instead of hitting the filesystem every analyze.
_session = None
_meta = None
_load_failed = False


def is_available() -> bool:
    """True if a usable classifier is installed (cheap after the first call)."""
    return _ensure_loaded() is not None


def _ensure_loaded():
    global _session, _meta, _load_failed
    if _session is not None:
        return _session
    if _load_failed:
        return None

    if not _MODEL_PATH.exists():
        _load_failed = True
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        _load_failed = True
        return None

    try:
        meta = _load_labels()
        options = ort.SessionOptions()
        # Keep the analyzer responsive on a 2-core laptop: this runs
        # alongside face detection and k-means on the same photo.
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(_MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception:
        # A corrupt/incompatible model must degrade to the heuristic, not
        # take the Analyzer down with it.
        _load_failed = True
        return None

    _session = session
    _meta = meta
    return _session


def _load_labels() -> dict:
    labels = []
    indoor_classes = []
    outdoor_classes = []
    size = _DEFAULT_SIZE
    mean = _DEFAULT_MEAN
    std = _DEFAULT_STD

    if _LABELS_PATH.exists():
        try:
            with open(_LABELS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            labels = [str(x) for x in (data.get("labels") or [])]
            indoor_classes = [str(x).lower() for x in (data.get("indoor_classes") or [])]
            outdoor_classes = [str(x).lower() for x in (data.get("outdoor_classes") or [])]
            pre = data.get("preprocess") or {}
            size = int(pre.get("size", size))
            mean = tuple(pre.get("mean", mean))
            std = tuple(pre.get("std", std))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    if not labels:
        # Minimal sane default: a 2-class indoor/outdoor head.
        labels = ["indoor", "outdoor"]
    if not indoor_classes:
        indoor_classes = ["indoor"]
    if not outdoor_classes:
        outdoor_classes = ["outdoor"]

    return {
        "labels": labels,
        "indoor_classes": set(indoor_classes),
        "outdoor_classes": set(outdoor_classes),
        "size": size,
        "mean": mean,
        "std": std,
    }


def _preprocess(rgb_np: np.ndarray, meta: dict) -> np.ndarray:
    import cv2

    size = meta["size"]
    resized = cv2.resize(rgb_np, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
    # HWC -> NCHW
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)


def classify(rgb_np: np.ndarray):
    """
    Returns {"label": "indoor"|"outdoor", "confidence": 0-100,
             "top_class": str} or None when no classifier is installed.

    `rgb_np` is a plain HxWx3 uint8 RGB array (same thing
    core/analyzer.py already builds), so the caller does no extra work.
    """
    session = _ensure_loaded()
    if session is None:
        return None

    try:
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: _preprocess(rgb_np, _meta)})[0]
        probs = _softmax(np.asarray(logits).reshape(-1).astype(np.float32))
    except Exception:
        return None

    labels = _meta["labels"]
    if len(probs) != len(labels):
        # Label map doesn't match the model head -- don't guess which
        # class means what; hand back None so the heuristic runs instead.
        return None

    indoor_p = 0.0
    outdoor_p = 0.0
    for label, p in zip(labels, probs):
        key = label.lower()
        if key in _meta["indoor_classes"]:
            indoor_p += float(p)
        elif key in _meta["outdoor_classes"]:
            outdoor_p += float(p)

    if indoor_p <= 0.0 and outdoor_p <= 0.0:
        return None

    top_index = int(np.argmax(probs))
    if outdoor_p >= indoor_p:
        label, confidence = "outdoor", outdoor_p
    else:
        label, confidence = "indoor", indoor_p

    return {
        "label": label,
        "confidence": int(round(max(0.0, min(1.0, confidence)) * 100)),
        "top_class": labels[top_index],
    }
