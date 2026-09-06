# ai/face_detector.py
#
# PHASE 5 -- Analyzer: face detection.
#
# This is the one piece of Phase 5 that's a genuine trained model, per
# the project's module split (ai/ = trained-model features, core/ =
# orchestration + classic deterministic processing -- see
# core/enhancer.py's header note, and the "WHERE AI IS ACTUALLY USED"
# section of the spec doc). core/analyzer.py calls into this module the
# same way ui/bridge.py's _get_cutout() calls into ai/bg_remover.py.
#
# MODEL CHOICE: OpenCV's bundled Haar Cascade classifiers (Viola-Jones),
# not a DNN/ONNX model like rembg. Deliberate, for three reasons that
# all trace back to this spec's own rules:
#   1. "No mandatory cloud" / fully local -- the cascade XML files ship
#      INSIDE the opencv-python wheel (cv2.data.haarcascades) and are
#      already on disk the moment opencv-python is installed. Unlike
#      rembg's u2net (~176MB downloaded on first use), there is no
#      first-run download at all.
#   2. "CPU-first, no CUDA, target hardware = i7-7600U / HD 620" -- Haar
#      cascades are the cheapest real trained-model face detector that
#      exists; a modern DNN face detector (e.g. a Caffe/ONNX SSD model)
#      would be more accurate but needs its weights fetched from
#      somewhere first, which this project has no cloud step for outside
#      of rembg's one already-accepted exception.
#   3. It IS still a genuinely trained model (trained via boosted cascades
#      of Haar-like features on labeled face datasets) -- not a
#      hand-written heuristic -- so it correctly satisfies "this is the
#      phase that needs a real trained model" without overclaiming.
#
# Trade-off, stated honestly (matches this project's existing pattern of
# calling out algorithm limits, e.g. object_remover.py's inpainting
# note): Haar cascades are less accurate than a modern DNN detector,
# most reliable on a roughly front-facing, reasonably well-lit face, and
# can miss faces at extreme angles/low light or occasionally false-
# positive on face-like textures. That's an acceptable first version for
# "face detected: yes/no + roughly where" (what Analyzer needs to decide
# portrait vs. landscape and feed Phase 6), not a claim of state-of-the-
# art recognition-grade accuracy.
#
# VERSION PIN REQUIRED: this module needs `opencv-python` PINNED to a
# 4.x release (e.g. `opencv-python==4.13.0.92` in requirements.txt) --
# OpenCV 5.0 moved cv2.CascadeClassifier out of the core package into
# opencv_contrib's xobjdetect module, which plain `opencv-python` does
# NOT ship. An unpinned `pip install opencv-python` can silently resolve
# to 5.x and break this module with "module 'cv2' has no attribute
# 'CascadeClassifier'" -- see the compatibility check in _get_cascades()
# below, which raises a clear, actionable error instead of a bare
# AttributeError if that happens.

from PIL import Image

import cv2
import numpy as np

# Lazily created, cached at module level -- loading a cascade from XML
# is cheap (a few ms) compared to rembg's model load, but there's still
# no reason to re-parse the XML on every single analyzeImage() call.
_frontal_cascade = None
_profile_cascade = None


def _get_cascades():
    global _frontal_cascade, _profile_cascade
    if _frontal_cascade is None:
        # COMPATIBILITY CHECK: OpenCV 5.0 (released after this module was
        # first written) removed cv2.CascadeClassifier from the core
        # objdetect module -- it moved to opencv_contrib's xobjdetect
        # module, which the plain `opencv-python` PyPI package does NOT
        # include. `pip install opencv-python` with no version pin can
        # silently resolve to 5.x and break face detection with
        # "module 'cv2' has no attribute 'CascadeClassifier'". Caught
        # here and re-raised with an actionable message instead of a
        # bare AttributeError, since this is an environment/dependency
        # issue, not a bug in this module -- the real fix is pinning
        # requirements.txt to a 4.x release (e.g. opencv-python==4.13.0.92)
        # rather than anything changing here.
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "This install of opencv-python doesn't have "
                "cv2.CascadeClassifier (OpenCV 5.x moved Haar cascades out "
                "of the core package). Fix: `pip uninstall opencv-python -y` "
                "then `pip install opencv-python==4.13.0.92`, and pin that "
                "exact version in requirements.txt."
            )
        _frontal_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        _profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        if _frontal_cascade.empty() or _profile_cascade.empty():
            # Extremely unlikely (the XMLs ship with the opencv-python
            # wheel itself) but fail loudly rather than silently
            # returning "no faces" for every photo if the install is
            # somehow broken.
            raise RuntimeError(
                "Could not load bundled Haar cascade files from "
                "cv2.data.haarcascades -- check the opencv-python install."
            )
    return _frontal_cascade, _profile_cascade


# Faces smaller than this fraction of the analysis image's shorter side
# are ignored -- cuts down on false positives from tiny face-like
# texture patches (bark, fabric patterns, etc.) that a strict min-size
# alone doesn't catch on a downscaled photo.
_MIN_FACE_FRACTION = 0.035

# The photo is downscaled to at most this size before detection -- Haar
# cascades scan at many scales already; running that on a full 24MP
# photo is wasted CPU on the target hardware (i7-7600U, no GPU) for no
# accuracy benefit, since a face big enough to matter for "portrait vs.
# landscape" is still easily resolved at this size.
_MAX_DETECT_DIM = 900


def detect_faces(image: Image.Image) -> list:
    """
    Detects faces in `image` (any PIL mode/size) using OpenCV's bundled
    Haar Cascade face detectors -- frontal, plus a profile pass mirrored
    horizontally to catch faces looking left (the bundled profile
    cascade is trained facing one direction only).

    Returns a list of dicts, one per detected face, each shaped:
        {"x": float, "y": float, "w": float, "h": float}
    where x/y/w/h are all FRACTIONS of the image's width/height (0-1),
    not pixel coordinates -- so callers (core/analyzer.py, and the
    frontend if it ever draws face boxes) don't need to know what size
    the detector actually ran at. Overlapping frontal/profile/mirrored-
    profile detections of the same face are merged into one box.

    Returns an empty list (not an error) if no faces are found, or if
    the image has zero area.
    """
    if image.width == 0 or image.height == 0:
        return []

    frontal_cascade, profile_cascade = _get_cascades()

    rgb = image.convert("RGB") if image.mode != "RGB" else image
    scale = min(1.0, _MAX_DETECT_DIM / max(rgb.width, rgb.height))
    detect_img = rgb.resize(
        (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
        Image.LANCZOS,
    ) if scale < 1.0 else rgb

    gray = cv2.cvtColor(np.array(detect_img), cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)  # normalizes contrast/lighting before detection

    min_side = min(gray.shape[0], gray.shape[1])
    min_size = max(20, round(min_side * _MIN_FACE_FRACTION))

    def _run(cascade, img):
        boxes = cascade.detectMultiScale(
            img,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_size, min_size),
        )
        return [tuple(int(v) for v in b) for b in boxes]

    raw_boxes = _run(frontal_cascade, gray)
    raw_boxes += _run(profile_cascade, gray)
    mirrored = cv2.flip(gray, 1)
    for (x, y, w, h) in _run(profile_cascade, mirrored):
        # Un-mirror the x coordinate back to the original orientation.
        raw_boxes.append((gray.shape[1] - x - w, y, w, h))

    merged = _merge_overlapping_boxes(raw_boxes)

    dw, dh = detect_img.width, detect_img.height
    return [
        {"x": x / dw, "y": y / dh, "w": w / dw, "h": h / dh}
        for (x, y, w, h) in merged
    ]


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _merge_overlapping_boxes(boxes, iou_threshold=0.3):
    """
    Greedy merge: any two boxes overlapping above `iou_threshold` are
    treated as the same detected face (frontal + profile + mirrored-
    profile passes above frequently all fire on one real face) and
    collapsed to their average box, so a single face isn't reported
    2-3 times.
    """
    remaining = list(boxes)
    merged = []
    while remaining:
        base = remaining.pop(0)
        group = [base]
        still_remaining = []
        for other in remaining:
            if _iou(base, other) >= iou_threshold:
                group.append(other)
            else:
                still_remaining.append(other)
        remaining = still_remaining
        xs = [g[0] for g in group]
        ys = [g[1] for g in group]
        ws = [g[2] for g in group]
        hs = [g[3] for g in group]
        merged.append(
            (round(sum(xs) / len(xs)), round(sum(ys) / len(ys)),
             round(sum(ws) / len(ws)), round(sum(hs) / len(hs)))
        )
    return merged