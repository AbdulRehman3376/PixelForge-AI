# core/video_studio.py
#
# PHASE 11 -- Video Studio.
#
# 🧮 NOT AI. Deterministic video assembly via FFmpeg (motion/transition
# math, filter-graph construction), no trained model -- same "🧮 not AI"
# labeling the spec uses for Batch/Filters/Crop. Same layering as every
# other phase: this module owns the real logic (project state, undo/
# redo, FFmpeg command construction, threaded export with progress +
# cooperative cancellation, project save/load), ui/bridge.py only wires
# it to Slots.
#
# WHY A CONTROLLER OBJECT (same reasoning as core/batch_processor.py's
# BatchController / core/session.py's EditSession): a Video Studio
# project has real state that must survive across many Slot calls --
# the clip timeline, per-clip motion/transition settings, audio track,
# text overlays, undo/redo history -- so it's a stateful object held by
# ui/bridge.py (one shared instance, get_video_controller()), not a
# one-shot function.
#
# FFMPEG FILTER-GRAPH DESIGN (documented here since none of this is
# visible from the Slot layer):
#   1. Each clip's source image is scaled to a slightly oversized
#      "working" canvas (SUPERSAMPLE_FACTOR x the output resolution) so
#      the Ken Burns/zoom/pan filters have room to crop and move within
#      it without ever upscaling past the source's own detail more than
#      necessary.
#   2. Getting from the source's native aspect ratio to that working
#      canvas is either:
#        - "fill"  (cover-crop): scale to cover + center-crop -- the
#          default, no bars.
#        - "fit"   (contain): scale to fit inside + pad, with the pad
#          filled either by a heavily blurred cover-scaled copy of the
#          same image ("Background Fill: blur") or solid black.
#   3. Motion presets (Ken Burns / Zoom In / Zoom Out / Pan L/R/
#      Vertical) are built as FFmpeg `zoompan` filters. Because the
#      working canvas size is a literal Python-known integer (we chose
#      it), every zoompan z/x/y expression below is written entirely in
#      terms of the `on` (output frame number) variable and literal
#      numbers -- no reliance on zoompan's own `zoom`/`d` expression
#      variables, which keeps the expressions simple and unambiguous.
#   4. Clips are then chained with `xfade` for transitions (fade,
#      crossfade, slide, zoom, blur, dip-to-black, light), using the
#      standard progressive-offset pattern for chaining N `xfade`s
#      end-to-end (each new xfade's offset = the running combined
#      length so far minus that transition's duration).
#   5. Background music (volume/fade in/out, auto-trimmed/looped to the
#      final video length) is mixed in with `afade`/`volume`/`atrim`.
#   6. Text/title overlays are `drawtext` filters with start time,
#      duration and a linear fade in/out driven by the `alpha` option.
#   7. Export runs FFmpeg with `-progress pipe:1` so stdout can be
#      parsed for live percent-complete, and is cooperatively
#      cancellable (checked between progress lines; the process is then
#      terminated).
#
# TARGET-HARDWARE NOTE: video export is genuinely heavy even on
# "modern" hardware, let alone the i7-7600U/no-GPU target -- there is no
# way around that for real H.264 encoding. What we control for: default
# to 720p/1080p (not 4K) per the spec's own hardware guidance, use
# `-preset` to trade encode speed for size deliberately (see
# QUALITY_PRESETS), and never hold more than the current clip's PIL
# thumbnail in memory (project state stores paths, not pixel data).

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

PROJECT_EXTENSION = ".pfvproj"

# ----- Enumerations the frontend's dropdowns are built from -----
MOTION_PRESETS = [
    "none", "ken_burns", "zoom_in", "zoom_out",
    "pan_left", "pan_right", "pan_vertical",
    "slow_drift", "tilt_3d",
]
TRANSITIONS = [
    "none", "fade", "crossfade", "dip_to_black", "light",
    "slide_left", "slide_right", "wipe_left", "wipe_right",
    "zoom", "circle_open",
]
RESOLUTIONS = ["720p", "1080p", "4k"]
ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
FPS_CHOICES = [24, 30, 60]
DURATION_MODES = ["auto", "15s", "30s", "60s", "custom"]
QUALITY_LEVELS = ["low", "medium", "high", "custom"]
FIT_MODES = ["fill", "fit"]
BACKGROUND_FILLS = ["blur", "black"]
# ----- color grade presets -----
# Each is an FFmpeg `eq`/`colorbalance` fragment (no leading comma,
# callers add that). "none" skips the grading link entirely so a
# fresh/ungraded look is possible without wasting a filter pass.
COLOR_GRADES = ["none", "cinematic", "warm", "cool", "vivid", "bw"]
_COLOR_GRADE_FILTERS = {
    "cinematic": "vignette=PI/4,eq=contrast=1.08:saturation=0.92:gamma=0.97,"
                 "colorbalance=rs=-0.03:gs=0.0:bs=0.06:rm=-0.02:bm=0.04",
    "warm": "eq=contrast=1.05:saturation=1.1,colorbalance=rs=0.08:gs=0.02:bs=-0.08",
    "cool": "eq=contrast=1.05:saturation=1.05,colorbalance=rs=-0.06:bs=0.09",
    "vivid": "eq=contrast=1.12:saturation=1.35:brightness=0.01",
    "bw": "hue=s=0,eq=contrast=1.15",
}
TEXT_POSITIONS = [
    "top-left", "top-center", "top-right",
    "center-left", "center", "center-right",
    "bottom-left", "bottom-center", "bottom-right",
]

# ----- PHASE 12: Audio / Text -----
# "none"  -- leave the track as-is, atrim in build_ffmpeg_command() still
#            clips it to the video's length if it runs long, but a short
#            track just plays once and then silence for the rest.
# "loop"  -- `-stream_loop -1` on the audio input so a short track
#            repeats seamlessly until the video ends, then gets trimmed
#            to the exact final frame (no half-loop click at the tail).
# "trim"  -- explicit synonym for "none" in the UI (a long track is
#            already trimmed to video length either way) -- kept as a
#            separate label because "Trim to fit" reads clearer to a
#            user than "None" when their music is the LONGER file.
AUDIO_DURATION_MODES = ["none", "loop", "trim"]

TEXT_ANIMATIONS = ["fade", "slide_up", "slide_down", "slide_left", "slide_right", "typewriter"]

WATERMARK_TYPES = ["text", "image"]
WATERMARK_POSITIONS = TEXT_POSITIONS

# Shown by the frontend the moment a music file is attached (Phase 12
# spec requirement: "must warn user about music rights/permission").
# Deliberately generic/non-legal-advice phrasing -- PixelForge has no
# way to know the actual license of a file the user picked off disk.
MUSIC_RIGHTS_NOTICE = (
    "Make sure you have the rights to use this audio (your own "
    "recording, a royalty-free track, or a licensed purchase) before "
    "sharing or publishing this video -- PixelForge can't verify "
    "music licensing for you."
)

# "long edge" pixel count per resolution preset -- see the width/height
# derivation in _output_dimensions() below for how this combines with
# aspect ratio.
_RESOLUTION_LONG_EDGE = {"720p": 1280, "1080p": 1920, "4k": 3840}

_ASPECT_RATIOS_WH = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1)}

_DURATION_PRESET_SECONDS = {"15s": 15.0, "30s": 30.0, "60s": 60.0}

# How strongly each Motion Intensity preset moves the frame -- fraction
# of the frame the zoom/pan travels across the clip's full duration.
_MOTION_INTENSITY = {"slow": 0.06, "medium": 0.12, "strong": 0.20}

# CRF (quality) + x264 preset (encode speed) per Export Quality level.
# Lower CRF = higher quality/bigger file. "custom" is handled separately
# via an explicit bitrate the user provides.
_QUALITY_CRF = {"low": 30, "medium": 24, "high": 18}
_QUALITY_SPEED_PRESET = {"low": "veryfast", "medium": "medium", "high": "slow"}

# How much bigger than the final output the per-clip working canvas is,
# so zoompan has real room to crop/move instead of digitally upscaling
# past the source's detail.
_SUPERSAMPLE_FACTOR = 1.18

_DEFAULT_MIN_CLIP_DURATION = 1.0


class VideoStudioError(Exception):
    """Base error for anything Video Studio specific (bad state, FFmpeg
    missing, etc.) -- mirrors UpscaleError/FaceRestoreError's role in
    the AI phases so ui/bridge.py can catch it the same way."""


class VideoExportCancelled(VideoStudioError):
    """Raised internally when the cooperative cancel flag is seen mid-
    export -- caught by ui/bridge.py exactly like UpscaleCancelled /
    FaceRestoreCancelled in Phase 9/10."""


# ===================== FFMPEG DETECTION =====================

def ffmpeg_status() -> dict:
    """
    Settings screen's "FFmpeg: Detected" badge (frontend/index.html's
    Video group) currently hardcodes "Detected" -- this is the real
    check behind it. FFmpeg is expected on PATH (Phase 0 confirmed it
    installed and verified with `ffmpeg -version`).

    Returns {"available": bool, "version": str, "path": str|None}.
    """
    path = shutil.which("ffmpeg")
    if not path:
        return {"available": False, "version": "", "path": None}
    try:
        proc = subprocess.run(
            [path, "-version"], capture_output=True, text=True, timeout=5,
        )
        first_line = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
        # First line looks like "ffmpeg version 6.0 Copyright (c) ..."
        version = first_line.split("Copyright")[0].strip()
        return {"available": True, "version": version or "unknown", "path": path}
    except Exception:  # noqa: BLE001 - a broken ffmpeg install must not crash Settings
        return {"available": True, "version": "unknown", "path": path}


# ===================== SMALL HELPERS =====================

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _even(n: int) -> int:
    """H.264 needs even width/height."""
    n = int(round(n))
    return n if n % 2 == 0 else n + 1


def _output_dimensions(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    """
    Derives (width, height) from a resolution preset ("720p"/"1080p"/
    "4k" -- the LONG edge pixel count) and an aspect ratio -- landscape/
    square aspects use the long edge as width, portrait (9:16) uses it
    as height, so "1080p" + "9:16" gives a real 1080x1920 vertical
    export instead of a tiny sideways one.
    """
    long_edge = _RESOLUTION_LONG_EDGE.get(resolution, _RESOLUTION_LONG_EDGE["1080p"])
    rw, rh = _ASPECT_RATIOS_WH.get(aspect_ratio, _ASPECT_RATIOS_WH["16:9"])
    if rw >= rh:
        width = long_edge
        height = _even(long_edge * rh / rw)
    else:
        height = long_edge
        width = _even(long_edge * rw / rh)
    return width, height


def _probe_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.width, img.height


def _make_thumbnail(source_path: str, cache_dir: Path, max_side: int = 220) -> str:
    """
    Cheap CPU-only thumbnail for the timeline strip -- kept separate
    from FFmpeg entirely (no reason to shell out for a still image).
    Cached by mtime+size in the filename so re-adding the same file
    doesn't regenerate it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        stat = os.stat(source_path)
        key = f"{abs(hash((source_path, stat.st_mtime, stat.st_size)))}"
    except OSError:
        key = uuid.uuid4().hex
    thumb_path = cache_dir / f"thumb_{key}.jpg"
    if not thumb_path.exists():
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            img.save(thumb_path, quality=85)
    return str(thumb_path)


_FONT_FILE_CACHE: dict[str, str | None] = {}


def _resolve_font_file() -> str | None:
    """
    Finds an absolute path to a usable TrueType/OpenType font for
    FFmpeg's `drawtext` filter (see the BUGFIX note in
    build_text_filter() for why this exists at all). Cached after the
    first call since it never changes for the life of the process.
    """
    if "path" in _FONT_FILE_CACHE:
        return _FONT_FILE_CACHE["path"]

    candidates: list[str] = []
    system = platform.system()
    if system == "Windows":
        win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        candidates += [
            os.path.join(win_fonts, name)
            for name in ("arial.ttf", "arialbd.ttf", "segoeui.ttf", "calibri.ttf", "tahoma.ttf")
        ]
    elif system == "Darwin":
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]

    for path in candidates:
        if os.path.isfile(path):
            _FONT_FILE_CACHE["path"] = path
            return path

    # Last resort: ask fontconfig directly, in case it IS present but
    # just doesn't have a plain "Sans" family under that exact name.
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}"], capture_output=True, text=True, timeout=2)
        found = out.stdout.strip()
        if out.returncode == 0 and found and os.path.isfile(found):
            _FONT_FILE_CACHE["path"] = found
            return found
    except Exception:
        pass

    _FONT_FILE_CACHE["path"] = None
    return None


def _escape_ffmpeg_filter_path(path: str) -> str:
    """Forward slashes sidestep the backslash-escaping mess a Windows
    path (``C:\\Users\\...``) would otherwise need inside an already-
    quoted filtergraph string; FFmpeg accepts forward slashes on
    Windows too. The drive letter's colon still needs escaping since
    ':' is the filtergraph's own option separator."""
    return path.replace("\\", "/").replace(":", "\\:")


# ===================== MOTION EXPRESSIONS (zoompan) =====================

def _motion_zoompan_filter(motion: str, intensity_key: str, canvas_w: int,
                            canvas_h: int, out_w: int, out_h: int,
                            duration: float, fps: int) -> str:
    """
    Builds the `zoompan` filter fragment (without the leading input
    label) for one clip. `canvas_w`/`canvas_h` is the oversampled
    working canvas (see _SUPERSAMPLE_FACTOR); `out_w`/`out_h` is the
    final crop/zoom window size fed to `s=` (which zoompan then holds
    for the whole clip). See the module header for why every expression
    below is written purely in literal numbers + `on`.
    """
    d_frames = max(1, int(round(duration * fps)))
    intensity = _MOTION_INTENSITY.get(intensity_key, _MOTION_INTENSITY["medium"])

    if motion == "none":
        # Static frame: crop a fixed centered out_w x out_h window, no
        # animation at all.
        z, x, y = "1", f"(({canvas_w}-{out_w})/2)", f"(({canvas_h}-{out_h})/2)"
        return f"zoompan=z='{z}':x='{x}':y='{y}':d={d_frames}:s={out_w}x{out_h}:fps={fps}"

    prog = f"(on/{d_frames})"  # 0 -> ~1 progress through the clip

    if motion == "zoom_in":
        z = f"(1+{intensity}*{prog})"
    elif motion == "zoom_out":
        z = f"(1+{intensity}*(1-{prog}))"
    elif motion == "ken_burns":
        z = f"(1+{intensity}*{prog})"
    elif motion == "slow_drift":
        # Deliberately slower and smaller-amplitude than ken_burns --
        # eased with a smoothstep curve (3t^2-2t^3) instead of linear
        # progress so it starts and ends at near-zero velocity, which
        # reads as a graceful, "cinematic slow motion" camera move
        # rather than a constant-speed pan.
        eased = f"(3*{prog}*{prog}-2*{prog}*{prog}*{prog})"
        z = f"(1+{intensity}*0.6*{eased})"
    elif motion == "tilt_3d":
        # Same eased zoom as slow_drift; the actual tilt/rotation is
        # layered on afterwards (see _motion_extra_filters) since
        # zoompan itself has no rotation parameter.
        eased = f"(3*{prog}*{prog}-2*{prog}*{prog}*{prog})"
        z = f"(1+{intensity}*0.7*{eased})"
    else:
        # pan_left / pan_right / pan_vertical: gentle constant zoom just
        # large enough to give the pan somewhere to travel.
        z = f"(1+{intensity*0.5})"

    crop_w = f"({canvas_w}/{z})"
    crop_h = f"({canvas_h}/{z})"

    if motion == "pan_left":
        x = f"(({canvas_w}-{crop_w})*(1-{prog}))"
        y = f"(({canvas_h}-{crop_h})/2)"
    elif motion == "pan_right":
        x = f"(({canvas_w}-{crop_w})*{prog})"
        y = f"(({canvas_h}-{crop_h})/2)"
    elif motion == "pan_vertical":
        x = f"(({canvas_w}-{crop_w})/2)"
        y = f"(({canvas_h}-{crop_h})*{prog})"
    elif motion in ("ken_burns", "slow_drift", "tilt_3d"):
        # Gentle diagonal drift (half-amplitude so it reads as subtle
        # motion, not a full pan) combined with the zoom above.
        x = f"(({canvas_w}-{crop_w})*{prog}*0.5)"
        y = f"(({canvas_h}-{crop_h})*{prog}*0.5)"
    else:  # zoom_in / zoom_out: stay centered
        x = f"(({canvas_w}-{crop_w})/2)"
        y = f"(({canvas_h}-{crop_h})/2)"

    return f"zoompan=z='{z}':x='{x}':y='{y}':d={d_frames}:s={out_w}x{out_h}:fps={fps}"


def _motion_extra_filters(motion: str, duration: float) -> str:
    """
    Filters layered on AFTER zoompan for presets that need something
    zoompan itself can't express. Returns "" when there's nothing extra
    to add -- callers should skip the comma entirely in that case so the
    filter chain doesn't pick up a stray empty link.
    """
    if motion == "tilt_3d":
        # +/-1.2 degree oscillating tilt across the clip, eased with a
        # sine so it settles at 0 at both ends -- combined with the zoom
        # drift above this reads as a subtle handheld/3D-parallax move
        # rather than a flat slideshow pan.
        max_deg = 1.2
        return f"rotate=a='({max_deg}*PI/180)*sin(2*PI*t/{max(duration,0.01):.3f})':c=black@0:ow=iw:oh=ih"
    return ""


# ===================== TRANSITIONS (xfade) =====================

# Maps our spec-facing transition names onto FFmpeg's actual `xfade`
# transition identifiers. Each entry below is intentionally a visually
# distinct xfade effect (no two spec names collapse onto the same
# built-in anymore) so the 10-transition set actually reads as 10
# different looks in the exported video, not repeats of the same 2-3
# effects under different labels.
_XFADE_MAP = {
    "fade": "fadeblack",       # classic dip-through-black
    "crossfade": "fade",       # plain dissolve
    "dip_to_black": "fadegrays",  # slower, "cinematic" dip through gray
    "light": "fadewhite",      # dip-through-white / flash cut
    "slide_left": "slideleft",
    "slide_right": "slideright",
    "wipe_left": "wipeleft",
    "wipe_right": "wiperight",
    "zoom": "zoomin",
    "circle_open": "circleopen",
}


# ===================== CLIP / PROJECT MODEL =====================

def _default_clip(path: str, thumbnail: str) -> dict:
    return {
        "id": _new_id("clip"),
        "path": path,
        "thumbnail": thumbnail,
        "duration": 4.0,             # seconds this clip is on screen
        "motion": "ken_burns",
        "motion_intensity": "medium",
        "transition": "crossfade",   # transition INTO the next clip (ignored on the last clip)
        "transition_duration": 0.8,
        "cover": False,              # this clip is the project's cover/thumbnail frame
    }


def _default_settings() -> dict:
    return {
        "resolution": "1080p",
        "aspect_ratio": "16:9",
        "fps": 30,
        "fit_mode": "fill",          # "fill" (cover-crop) or "fit" (letterbox/pillarbox)
        "background_fill": "blur",   # "blur" or "black" -- only used when fit_mode == "fit"
        "duration_mode": "auto",     # "auto" | "15s" | "30s" | "60s" | "custom"
        "custom_duration": 30.0,
        "quality": "high",           # "low" | "medium" | "high" | "custom"
        "custom_bitrate_kbps": 8000,
        "color_grade": "cinematic",  # "none" | "cinematic" | "warm" | "cool" | "vivid" | "bw"
        "sharpen": True,             # unsharp-mask clarity boost -- big win on soft phone photos
        "cinematic_bars": False,     # 2.35:1 letterbox bars over the top/bottom of the frame
        "film_grain": False,         # subtle noise overlay -- breaks up the "too clean/digital" look
    }


def _default_audio() -> dict:
    return {
        "path": None, "volume": 80, "fade_in": 1.0, "fade_out": 1.5,
        "duration_match": "none",  # "none" | "loop" | "trim" -- see AUDIO_DURATION_MODES
    }


def _default_audio2() -> dict:
    """PHASE 12: second, limited-support audio track -- meant for a
    voice-over/narration clip mixed underneath the primary music track
    (no fades/duration-matching of its own, kept intentionally simple
    per the spec's own "Multiple audio tracks (limited support)"
    wording -- one full-featured primary track + one plain second
    track, not an arbitrary N-track mixer)."""
    return {"path": None, "volume": 100}


def _default_watermark() -> dict:
    """PHASE 12: a persistent overlay shown for the ENTIRE video --
    either a short text string (e.g. a handle/brand name) or a small
    logo image with alpha transparency, low-opacity in a corner by
    default so it reads as a watermark rather than a title."""
    return {
        "enabled": False,
        "type": "text",          # "text" | "image"
        "text": "",
        "image_path": None,
        "opacity": 55,           # 0-100
        "position": "bottom-right",
        "scale": 16,             # image: % of video width. text: font size (px)
    }


class VideoProject:
    """
    Holds one Video Studio project's full state: the clip timeline,
    global settings, background music, text overlays, and an undo/redo
    history of full-state snapshots (simplest possible correct
    implementation -- the state is small JSON, not pixel data, so
    snapshotting the whole thing per edit is cheap, unlike Phase 6's
    Edit Session which snapshots actual images and therefore needs a
    smarter history).
    """

    MAX_HISTORY = 40

    def __init__(self):
        self._lock = threading.RLock()
        self._cache_dir = Path(tempfile.gettempdir()) / "pixelforge_video_thumbs"
        self.reset()

    # ----- lifecycle -----

    def reset(self) -> dict:
        with self._lock:
            self.clips: list[dict] = []
            self.settings: dict = _default_settings()
            self.audio: dict = _default_audio()
            self.audio2: dict = _default_audio2()
            self.watermark: dict = _default_watermark()
            self.text_overlays: list[dict] = []
            self.project_path: str | None = None
            self._undo_stack: list[dict] = []
            self._redo_stack: list[dict] = []
            self._last_export_path: str | None = None
            return self.state()

    # ----- snapshot / undo / redo -----

    def _snapshot(self) -> dict:
        return {
            "clips": json.loads(json.dumps(self.clips)),
            "settings": dict(self.settings),
            "audio": dict(self.audio),
            "audio2": dict(self.audio2),
            "watermark": dict(self.watermark),
            "text_overlays": json.loads(json.dumps(self.text_overlays)),
        }

    def _restore(self, snap: dict) -> None:
        self.clips = json.loads(json.dumps(snap["clips"]))
        self.settings = dict(snap["settings"])
        self.audio = dict(snap["audio"])
        self.audio2 = dict(snap.get("audio2") or _default_audio2())
        self.watermark = dict(snap.get("watermark") or _default_watermark())
        self.text_overlays = json.loads(json.dumps(snap["text_overlays"]))

    def _push_undo(self) -> None:
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.MAX_HISTORY:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> dict:
        with self._lock:
            if not self._undo_stack:
                return self.state()
            self._redo_stack.append(self._snapshot())
            snap = self._undo_stack.pop()
            self._restore(snap)
            return self.state()

    def redo(self) -> dict:
        with self._lock:
            if not self._redo_stack:
                return self.state()
            self._undo_stack.append(self._snapshot())
            snap = self._redo_stack.pop()
            self._restore(snap)
            return self.state()

    # ----- clip management -----

    def add_images(self, paths: list[str]) -> dict:
        with self._lock:
            self._push_undo()
            for path in paths or []:
                if not path or not os.path.isfile(path):
                    continue
                try:
                    thumb = _make_thumbnail(path, self._cache_dir)
                except Exception:  # noqa: BLE001 - a bad thumbnail must not block adding the clip
                    thumb = ""
                clip = _default_clip(path, thumb)
                if not self.clips:
                    clip["cover"] = True  # first clip added is the cover by default
                self.clips.append(clip)
            return self.state()

    def remove_clip(self, clip_id: str) -> dict:
        with self._lock:
            self._push_undo()
            self.clips = [c for c in self.clips if c["id"] != clip_id]
            if self.clips and not any(c["cover"] for c in self.clips):
                self.clips[0]["cover"] = True
            return self.state()

    def reorder(self, clip_ids: list[str]) -> dict:
        with self._lock:
            self._push_undo()
            by_id = {c["id"]: c for c in self.clips}
            ordered = [by_id[cid] for cid in clip_ids if cid in by_id]
            # Any clip not mentioned (shouldn't normally happen) stays,
            # appended at the end, so nothing silently disappears.
            missing = [c for c in self.clips if c["id"] not in clip_ids]
            self.clips = ordered + missing
            return self.state()

    def set_clip_settings(self, clip_id: str, patch: dict) -> dict:
        with self._lock:
            clip = next((c for c in self.clips if c["id"] == clip_id), None)
            if not clip:
                raise VideoStudioError("Clip not found.")
            self._push_undo()
            if "duration" in patch:
                clip["duration"] = max(_DEFAULT_MIN_CLIP_DURATION, float(patch["duration"]))
            if "motion" in patch and patch["motion"] in MOTION_PRESETS:
                clip["motion"] = patch["motion"]
            if "motion_intensity" in patch and patch["motion_intensity"] in _MOTION_INTENSITY:
                clip["motion_intensity"] = patch["motion_intensity"]
            if "transition" in patch and patch["transition"] in TRANSITIONS:
                clip["transition"] = patch["transition"]
            if "transition_duration" in patch:
                clip["transition_duration"] = max(0.1, float(patch["transition_duration"]))
            if patch.get("cover"):
                for c in self.clips:
                    c["cover"] = (c["id"] == clip_id)
            return self.state()

    # ----- project-level settings -----

    def set_project_settings(self, patch: dict) -> dict:
        with self._lock:
            self._push_undo()
            for key in ("resolution", "aspect_ratio", "fit_mode", "background_fill",
                        "duration_mode", "quality"):
                if key in patch and patch[key]:
                    self.settings[key] = patch[key]
            if "cinematic_grade" in patch:
                self.settings["cinematic_grade"] = bool(patch["cinematic_grade"])
            if "color_grade" in patch and patch["color_grade"] in COLOR_GRADES:
                self.settings["color_grade"] = patch["color_grade"]
            if "sharpen" in patch:
                self.settings["sharpen"] = bool(patch["sharpen"])
            if "cinematic_bars" in patch:
                self.settings["cinematic_bars"] = bool(patch["cinematic_bars"])
            if "film_grain" in patch:
                self.settings["film_grain"] = bool(patch["film_grain"])
            if "fps" in patch:
                try:
                    fps = int(patch["fps"])
                    self.settings["fps"] = fps if fps in FPS_CHOICES else self.settings["fps"]
                except (TypeError, ValueError):
                    pass
            if "custom_duration" in patch:
                self.settings["custom_duration"] = max(1.0, float(patch["custom_duration"]))
            if "custom_bitrate_kbps" in patch:
                self.settings["custom_bitrate_kbps"] = max(500, int(patch["custom_bitrate_kbps"]))
            return self.state()

    def set_audio(self, path: str | None, volume: int = 80, fade_in: float = 1.0,
                  fade_out: float = 1.5, duration_match: str = "none") -> dict:
        with self._lock:
            self._push_undo()
            self.audio = {
                "path": path or None,
                "volume": _clamp(int(volume), 0, 100),
                "fade_in": max(0.0, float(fade_in)),
                "fade_out": max(0.0, float(fade_out)),
                "duration_match": duration_match if duration_match in AUDIO_DURATION_MODES else "none",
            }
            return self.state()

    def clear_audio(self) -> dict:
        return self.set_audio(None)

    # ----- PHASE 12: second audio track (limited support -- see _default_audio2) -----

    def set_audio2(self, path: str | None, volume: int = 100) -> dict:
        with self._lock:
            self._push_undo()
            self.audio2 = {"path": path or None, "volume": _clamp(int(volume), 0, 100)}
            return self.state()

    def clear_audio2(self) -> dict:
        return self.set_audio2(None)

    # ----- PHASE 12: watermark / logo -----

    def set_watermark(self, patch: dict) -> dict:
        with self._lock:
            self._push_undo()
            wm = dict(self.watermark)
            if "enabled" in patch:
                wm["enabled"] = bool(patch["enabled"])
            if "type" in patch and patch["type"] in WATERMARK_TYPES:
                wm["type"] = patch["type"]
            if "text" in patch:
                wm["text"] = str(patch["text"])[:80]
            if "image_path" in patch:
                wm["image_path"] = patch["image_path"] or None
            if "opacity" in patch:
                wm["opacity"] = _clamp(int(patch["opacity"]), 5, 100)
            if "position" in patch and patch["position"] in WATERMARK_POSITIONS:
                wm["position"] = patch["position"]
            if "scale" in patch:
                wm["scale"] = _clamp(int(patch["scale"]), 4, 60)
            self.watermark = wm
            return self.state()

    def clear_watermark(self) -> dict:
        with self._lock:
            self._push_undo()
            self.watermark = _default_watermark()
            return self.state()

    # ----- PHASE 12: SRT subtitle import -----

    def import_srt(self, path: str) -> dict:
        """
        Parses a standard .srt file and turns every cue into a caption-
        style text overlay (small font, bottom-center, no fade so
        rapid-fire subtitle cues don't visibly cross-fade into each
        other, `is_caption=True` so the frontend/export can treat these
        distinctly from a single manually-added Title). Malformed
        blocks are skipped rather than aborting the whole import, so
        one bad cue in an otherwise-fine file doesn't lose the rest.
        """
        with self._lock:
            if not path or not os.path.isfile(path):
                raise VideoStudioError("SRT file not found.")
            try:
                raw = Path(path).read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                raw = Path(path).read_text(encoding="latin-1")

            def _srt_time(token: str) -> float | None:
                token = token.strip().replace(",", ".")
                try:
                    h, m, s = token.split(":")
                    return int(h) * 3600 + int(m) * 60 + float(s)
                except (ValueError, IndexError):
                    return None

            blocks = [b.strip() for b in raw.replace("\r\n", "\n").split("\n\n") if b.strip()]
            imported = 0
            self._push_undo()
            for block in blocks:
                lines = block.split("\n")
                # First line is a cue index (optional/ignorable); find
                # the "-->" timing line wherever it actually is instead
                # of assuming a fixed line number -- more tolerant of
                # slightly non-standard exports.
                timing_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
                if timing_idx is None:
                    continue
                parts = lines[timing_idx].split("-->")
                if len(parts) != 2:
                    continue
                start = _srt_time(parts[0])
                end = _srt_time(parts[1])
                if start is None or end is None or end <= start:
                    continue
                text_lines = lines[timing_idx + 1:]
                text = " ".join(t.strip() for t in text_lines if t.strip())
                if not text:
                    continue
                entry = {
                    "id": _new_id("text"),
                    "text": text[:200],
                    "font_size": 34,
                    "color": "#FFFFFF",
                    "position": "bottom-center",
                    "start_time": round(start, 3),
                    "duration": round(max(0.3, end - start), 3),
                    "fade": 0.15,
                    "opacity": 100,
                    "shadow": True,
                    "animation": "fade",
                    "is_caption": True,
                }
                self.text_overlays.append(entry)
                imported += 1
            result = self.state()
            result["imported_count"] = imported
            return result

    # ----- text / title overlays -----

    def add_text_overlay(self, overlay: dict) -> dict:
        with self._lock:
            self._push_undo()
            entry = {
                "id": _new_id("text"),
                "text": str(overlay.get("text", "Title"))[:200],
                "font_size": _clamp(int(overlay.get("font_size", 48)), 10, 200),
                "color": overlay.get("color") or "#FFFFFF",
                "position": overlay.get("position") if overlay.get("position") in TEXT_POSITIONS else "bottom-center",
                "start_time": max(0.0, float(overlay.get("start_time", 0.0))),
                "duration": max(0.2, float(overlay.get("duration", 3.0))),
                "fade": max(0.0, float(overlay.get("fade", 0.4))),
                "opacity": _clamp(int(overlay.get("opacity", 100)), 5, 100),
                "shadow": bool(overlay.get("shadow", True)),
                "animation": overlay.get("animation") if overlay.get("animation") in TEXT_ANIMATIONS else "fade",
                "is_caption": bool(overlay.get("is_caption", False)),
            }
            self.text_overlays.append(entry)
            return self.state()

    def update_text_overlay(self, overlay_id: str, patch: dict) -> dict:
        with self._lock:
            entry = next((t for t in self.text_overlays if t["id"] == overlay_id), None)
            if not entry:
                raise VideoStudioError("Text overlay not found.")
            self._push_undo()
            for key in ("text", "color", "position"):
                if key in patch and patch[key]:
                    entry[key] = patch[key]
            for key, cast in (("font_size", int), ("start_time", float),
                              ("duration", float), ("fade", float)):
                if key in patch:
                    entry[key] = cast(patch[key])
            if "opacity" in patch:
                entry["opacity"] = _clamp(int(patch["opacity"]), 5, 100)
            if "shadow" in patch:
                entry["shadow"] = bool(patch["shadow"])
            if "animation" in patch and patch["animation"] in TEXT_ANIMATIONS:
                entry["animation"] = patch["animation"]
            if "is_caption" in patch:
                entry["is_caption"] = bool(patch["is_caption"])
            return self.state()

    def remove_text_overlay(self, overlay_id: str) -> dict:
        with self._lock:
            self._push_undo()
            self.text_overlays = [t for t in self.text_overlays if t["id"] != overlay_id]
            return self.state()

    # ----- state / timing -----

    def total_duration(self) -> float:
        """Sum of clip screen time minus the overlap each transition eats."""
        if not self.clips:
            return 0.0
        total = sum(c["duration"] for c in self.clips)
        for c in self.clips[:-1]:
            if c["transition"] != "none":
                total -= c["transition_duration"]
        return max(0.1, total)

    def state(self) -> dict:
        width, height = _output_dimensions(self.settings["resolution"], self.settings["aspect_ratio"])
        return {
            "ok": True,
            "clips": self.clips,
            "settings": self.settings,
            "audio": self.audio,
            "audio2": self.audio2,
            "watermark": self.watermark,
            "text_overlays": self.text_overlays,
            "project_path": self.project_path,
            "can_undo": bool(self._undo_stack),
            "can_redo": bool(self._redo_stack),
            "total_duration": round(self.total_duration(), 2),
            "output_width": width,
            "output_height": height,
            "clip_count": len(self.clips),
            # PHASE 12: frontend shows this the moment a music file is
            # attached -- see MUSIC_RIGHTS_NOTICE's own docstring for
            # why it's phrased as a reminder, not a legal guarantee.
            "music_rights_notice": MUSIC_RIGHTS_NOTICE if self.audio.get("path") else None,
        }

    # ----- project save / load -----

    def save_project(self, path: str) -> dict:
        with self._lock:
            if not self.clips:
                raise VideoStudioError("Add at least one image before saving a project.")
            data = {
                "type": "pixelforge_video_project",
                "version": 2,
                "clips": self.clips,
                "settings": self.settings,
                "audio": self.audio,
                "audio2": self.audio2,
                "watermark": self.watermark,
                "text_overlays": self.text_overlays,
            }
            if not path.lower().endswith(PROJECT_EXTENSION):
                path += PROJECT_EXTENSION
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.project_path = path
            return self.state()

    def load_project(self, path: str) -> dict:
        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("type") != "pixelforge_video_project":
                raise VideoStudioError("Not a PixelForge Video Studio project file.")
            self._push_undo()
            self.clips = data.get("clips", [])
            self.settings = {**_default_settings(), **data.get("settings", {})}
            self.audio = {**_default_audio(), **data.get("audio", {})}
            self.audio2 = {**_default_audio2(), **data.get("audio2", {})}
            self.watermark = {**_default_watermark(), **data.get("watermark", {})}
            self.text_overlays = data.get("text_overlays", [])
            self.project_path = path
            return self.state()

    # ===================== FFMPEG COMMAND CONSTRUCTION =====================

    def _effective_durations(self) -> list[float]:
        """
        Applies the project's Duration mode (auto / 15s / 30s / 60s /
        custom) by scaling every clip's duration proportionally so the
        final total matches the target -- "Durations: 15s/30s/60s/
        custom" from the original spec, layered on top of the per-clip
        durations the timeline UI edits directly.
        """
        durations = [max(_DEFAULT_MIN_CLIP_DURATION, float(c["duration"])) for c in self.clips]
        mode = self.settings.get("duration_mode", "auto")
        if mode == "auto" or not durations:
            return durations

        target = _DURATION_PRESET_SECONDS.get(mode)
        if target is None and mode == "custom":
            target = float(self.settings.get("custom_duration", 30.0))
        if not target:
            return durations

        overlap = sum(
            c["transition_duration"] for c in self.clips[:-1] if c["transition"] != "none"
        )
        current_total = sum(durations)
        target_sum = max(len(durations) * _DEFAULT_MIN_CLIP_DURATION, target + overlap)
        scale = target_sum / current_total if current_total else 1.0
        return [max(_DEFAULT_MIN_CLIP_DURATION, d * scale) for d in durations]

    def build_filter_complex(self, out_w: int, out_h: int, fps: int,
                              durations: list[float]) -> tuple[str, str, float]:
        """
        Returns (filter_complex_string, final_video_label, total_video_duration).
        Builds one processed stream per clip (fit/fill + motion), then
        chains them with xfade transitions per the module header.
        """
        canvas_w = _even(out_w * _SUPERSAMPLE_FACTOR)
        canvas_h = _even(out_h * _SUPERSAMPLE_FACTOR)
        fit_mode = self.settings.get("fit_mode", "fill")
        bg_fill = self.settings.get("background_fill", "blur")

        parts: list[str] = []
        clip_labels: list[str] = []

        for i, clip in enumerate(self.clips):
            dur = durations[i]
            src = f"{i}:v"

            if fit_mode == "fit":
                # Letterbox/pillarbox: blurred (or solid) background +
                # a centered, fully-visible (never cropped) copy of the
                # source image on top.
                if bg_fill == "blur":
                    parts.append(
                        f"[{src}]split=2[bg{i}][fg{i}]"
                    )
                    parts.append(
                        f"[bg{i}]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
                        f"crop={canvas_w}:{canvas_h},boxblur=20:2,setsar=1[bgb{i}]"
                    )
                else:
                    parts.append(
                        f"color=c=black:s={canvas_w}x{canvas_h}:d={dur}[bgb{i}]"
                    )
                    parts.append(f"[{src}]copy[fg{i}]")
                parts.append(
                    f"[fg{i}]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease,setsar=1[fgs{i}]"
                )
                parts.append(
                    f"[bgb{i}][fgs{i}]overlay=(W-w)/2:(H-h)/2:shortest=1[canvas{i}]"
                )
            else:
                # Fill (cover-crop): the default, no bars.
                parts.append(
                    f"[{src}]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
                    f"crop={canvas_w}:{canvas_h},setsar=1[canvas{i}]"
                )

            motion_filter = _motion_zoompan_filter(
                clip["motion"], clip["motion_intensity"],
                canvas_w, canvas_h, out_w, out_h, dur, fps,
            )
            extra_motion = _motion_extra_filters(clip["motion"], dur)
            extra_motion_link = f",{extra_motion}" if extra_motion else ""
            # Color grade preset (see COLOR_GRADES / _COLOR_GRADE_FILTERS
            # above) -- the actual "why does this look professionally
            # shot instead of a raw slideshow" lever. "none" skips it.
            grade_key = self.settings.get("color_grade", "cinematic")
            grade_link = ""
            if grade_key in _COLOR_GRADE_FILTERS:
                grade_link = "," + _COLOR_GRADE_FILTERS[grade_key]
            # Sharpen (unsharp mask): meaningfully improves perceived
            # "quality" on soft/compressed phone photos without
            # over-sharpening crisp DSLR shots -- luma-only, moderate
            # amount chosen to avoid halo artifacts.
            sharpen_link = ""
            if self.settings.get("sharpen", True):
                sharpen_link = ",unsharp=5:5:0.8:5:5:0.0"
            # BUGFIX (verified against a real FFmpeg run): zoompan's `d`
            # is how many output frames EACH INCOMING frame is held for,
            # not a total output-frame cap -- with `-loop 1` feeding a
            # continuous stream of identical frames, omitting a hard
            # trim here multiplies frame count (input_frames * d),
            # producing a wildly-too-long clip (e.g. 3s intended -> 225s
            # actual, confirmed empirically). `trim=duration=` right
            # after zoompan is what actually pins the clip to `dur`
            # seconds; the per-input `-t` flag was removed from
            # build_ffmpeg_command() below for the same reason (it only
            # bounds how many SOURCE frames arrive, not zoompan's
            # multiplied output).
            parts.append(
                f"[canvas{i}]{motion_filter}{extra_motion_link},trim=duration={dur:.3f},"
                f"setpts=PTS-STARTPTS{grade_link}{sharpen_link},format=yuv420p[v{i}]"
            )
            clip_labels.append(f"v{i}")

        # ----- chain transitions -----
        if len(clip_labels) == 1:
            final_label = clip_labels[0]
            total = durations[0]
        else:
            running_len = durations[0]
            combined_label = clip_labels[0]
            for i in range(1, len(clip_labels)):
                prev_clip = self.clips[i - 1]
                transition = prev_clip.get("transition", "crossfade")
                td = min(
                    float(prev_clip.get("transition_duration", 0.8)),
                    durations[i - 1] - 0.05, durations[i] - 0.05,
                )
                td = max(0.1, td)
                xfade_name = _XFADE_MAP.get(transition, "fade") if transition != "none" else "fade"
                if transition == "none":
                    td = 0.1  # still need a hard cut; xfade with a tiny duration approximates it closely
                offset = max(0.0, running_len - td)
                out_label = f"x{i}"
                parts.append(
                    f"[{combined_label}][{clip_labels[i]}]xfade=transition={xfade_name}:"
                    f"duration={td:.3f}:offset={offset:.3f}[{out_label}]"
                )
                running_len = running_len + durations[i] - td
                combined_label = out_label
            final_label = combined_label
            total = running_len

        # ----- optional final "filmic" pass (letterbox bars + grain) -----
        # Applied ONCE to the whole assembled/transitioned stream rather
        # than per-clip: cheaper (one pass instead of N), and avoids the
        # bars/grain visibly "cutting" or resetting at every transition
        # seam, which would look worse than not having them at all.
        post_filters: list[str] = []
        if self.settings.get("cinematic_bars"):
            # 2.35:1 letterbox: black bars top+bottom sized so the
            # visible area matches a 2.35:1 crop of the current (out_w,
            # out_h) canvas. drawbox (not crop) so total output
            # dimensions/aspect ratio stay exactly what the project
            # settings say -- this only *looks* letterboxed, it doesn't
            # change the export's actual resolution.
            bar_h = max(0, int(round((out_h - out_w / 2.35) / 2)))
            if bar_h > 0:
                post_filters.append(f"drawbox=x=0:y=0:w={out_w}:h={bar_h}:color=black:t=fill")
                post_filters.append(f"drawbox=x=0:y={out_h - bar_h}:w={out_w}:h={bar_h}:color=black:t=fill")
        if self.settings.get("film_grain"):
            # Subtle luma-only noise -- breaks up the "too clean/digital"
            # look of AI-smoothed/sharpened footage without being an
            # obviously "grungy filter". temporal_noise (bypassed via
            # noise film-grain approximation using `noise` filter,
            # c0 = luma plane only) at a low strength.
            post_filters.append("noise=c0s=6:c0f=t+u")
        if post_filters:
            parts.append(f"[{final_label}]{','.join(post_filters)}[finalv]")
            final_label = "finalv"

        return ";".join(parts), final_label, total

    def build_text_filter(self, video_label: str, out_w: int, out_h: int,
                           fps: int, total_video: float):
        """
        Chains one `drawtext` (or, for "typewriter", a crop-reveal
        overlay) per overlay onto `video_label`. Each overlay entry can
        carry, on top of the original fade-in/out:
          - opacity: a static ceiling multiplied into the fade alpha,
            so a "60% opacity" title never exceeds that even mid-fade
          - shadow: a small drop shadow behind the glyphs
          - animation: "fade" (default -- unchanged from before),
            "slide_up"/"slide_down"/"slide_left"/"slide_right" (the
            same alpha fade PLUS the text physically slides in/out
            along that axis), or "typewriter" (characters reveal
            left-to-right over the fade window)
        Returns video_label unchanged if there are no overlays,
        otherwise (new_final_label, filter_fragment_string).
        """
        if not self.text_overlays:
            return video_label

        # BUGFIX: this used to build drawtext without an explicit
        # `fontfile=`, which makes FFmpeg fall back to fontconfig's
        # "Sans" family lookup. Most prebuilt Windows FFmpeg binaries
        # (gyan.dev/BtbN -- what most users install) ship libfreetype
        # but NOT libfontconfig, so that lookup fails and FFmpeg exits
        # immediately with "Cannot find a valid font for the family
        # Sans" -- the process never gets past frame 0, which is why
        # adding ANY title and generating/exporting used to instantly
        # fail back to 0% while a text-less timeline rendered fine.
        # An explicit fontfile= sidesteps fontconfig entirely.
        font_path = _resolve_font_file()
        if not font_path:
            raise VideoStudioError(
                "Couldn't find a font on this system to draw title text with. "
                "Install any TrueType font (Windows normally has one already at "
                "C:\\Windows\\Fonts\\arial.ttf) and try again, or remove the title."
            )
        font_clause = f"fontfile='{_escape_ffmpeg_filter_path(font_path)}':"

        _POS_EXPR = {
            "top-left": ("40", "40"),
            "top-center": ("(w-text_w)/2", "40"),
            "top-right": ("w-text_w-40", "40"),
            "center-left": ("40", "(h-text_h)/2"),
            "center": ("(w-text_w)/2", "(h-text_h)/2"),
            "center-right": ("w-text_w-40", "(h-text_h)/2"),
            "bottom-left": ("40", "h-text_h-40"),
            "bottom-center": ("(w-text_w)/2", "h-text_h-60"),
            "bottom-right": ("w-text_w-40", "h-text_h-40"),
        }
        SLIDE_PX = 70  # travel distance for slide_* animations, in pixels

        chunks = []
        label = video_label
        for i, ov in enumerate(self.text_overlays):
            x_expr, y_expr = _POS_EXPR.get(ov["position"], _POS_EXPR["bottom-center"])
            start, dur, fade = ov["start_time"], ov["duration"], ov["fade"]
            end = start + dur
            fade = min(fade, dur / 2) if dur > 0 else 0
            safe_text = ov["text"].replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")
            opacity_ceiling = _clamp(ov.get("opacity", 100), 5, 100) / 100.0
            animation = ov.get("animation", "fade")
            shadow_clause = "shadowcolor=black@0.6:shadowx=2:shadowy=2:" if ov.get("shadow", True) else ""

            if fade > 0:
                fade_expr = (
                    f"if(lt(t,{start}),0,"
                    f"if(lt(t,{start + fade}),(t-{start})/{fade},"
                    f"if(lt(t,{end - fade}),1,"
                    f"if(lt(t,{end}),({end}-t)/{fade},0))))"
                )
            else:
                fade_expr = f"between(t,{start},{end})"
            alpha_expr = f"({fade_expr})*{opacity_ceiling:.3f}" if opacity_ceiling < 1.0 else fade_expr

            if animation == "typewriter" and fade > 0:
                # Character-reveal: drawtext has no native per-character
                # reveal option, so the text is drawn once onto its own
                # transparent (yuva420p) layer, that layer's WIDTH is
                # cropped from 0 up to its full text width over `fade`
                # seconds, and the growing sliver is overlaid onto the
                # main stream -- that crop growth is what actually
                # produces the left-to-right "typing" look.
                base_label, drawn_label, cropped_label = f"twbase{i}", f"twdrawn{i}", f"twcrop{i}"
                out_label = f"txt{i}"
                chunks.append(
                    f"color=c=black@0.0:s={out_w}x{out_h}:d={total_video:.3f}:r={fps},"
                    f"format=yuva420p[{base_label}]"
                )
                chunks.append(
                    f"[{base_label}]drawtext={font_clause}text='{safe_text}':fontsize={ov['font_size']}:"
                    f"fontcolor={ov['color']}@{opacity_ceiling:.3f}:{shadow_clause}"
                    f"x={x_expr}:y={y_expr}[{drawn_label}]"
                )
                reveal_w = (
                    f"if(lt(t,{start}),0,"
                    f"if(lt(t,{start + fade}),iw*(t-{start})/{fade},iw))"
                )
                chunks.append(f"[{drawn_label}]crop=w='{reveal_w}':h=ih:x=0:y=0[{cropped_label}]")
                chunks.append(
                    f"[{label}][{cropped_label}]overlay=0:0:enable='between(t,{start},{end})'[{out_label}]"
                )
                label = out_label
                continue

            x_final, y_final = x_expr, y_expr
            if animation in ("slide_up", "slide_down", "slide_left", "slide_right") and fade > 0:
                # Offset shrinks to 0 exactly as the SAME envelope that
                # drives alpha (fade_expr, 0->1->0) rises/falls, so the
                # slide and the fade always finish together.
                sign = -1 if animation in ("slide_up", "slide_left") else 1
                offset_expr = f"({sign})*{SLIDE_PX}*(1-({fade_expr}))"
                if animation in ("slide_up", "slide_down"):
                    y_final = f"(({y_expr})+({offset_expr}))"
                else:
                    x_final = f"(({x_expr})+({offset_expr}))"

            out_label = f"txt{i}"
            chunks.append(
                f"[{label}]drawtext={font_clause}text='{safe_text}':fontsize={ov['font_size']}:"
                f"fontcolor={ov['color']}@1:{shadow_clause}x='{x_final}':y='{y_final}':"
                f"alpha='{alpha_expr}':enable='between(t,{start},{end})':"
                f"box=1:boxcolor=black@0.35:boxborderw=10[{out_label}]"
            )
            label = out_label
        return label, ";".join(chunks)

    # ===================== PHASE 12: WATERMARK / LOGO =====================

    _WATERMARK_OVERLAY_POS = {
        "top-left": ("20", "20"), "top-center": ("(W-w)/2", "20"), "top-right": ("W-w-20", "20"),
        "center-left": ("20", "(H-h)/2"), "center": ("(W-w)/2", "(H-h)/2"), "center-right": ("W-w-20", "(H-h)/2"),
        "bottom-left": ("20", "H-h-20"), "bottom-center": ("(W-w)/2", "H-h-20"), "bottom-right": ("W-w-20", "H-h-20"),
    }
    _WATERMARK_TEXT_POS = {
        "top-left": ("30", "30"), "top-center": ("(w-text_w)/2", "30"), "top-right": ("w-text_w-30", "30"),
        "center-left": ("30", "(h-text_h)/2"), "center": ("(w-text_w)/2", "(h-text_h)/2"), "center-right": ("w-text_w-30", "(h-text_h)/2"),
        "bottom-left": ("30", "h-text_h-30"), "bottom-center": ("(w-text_w)/2", "h-text_h-30"), "bottom-right": ("w-text_w-30", "h-text_h-30"),
    }

    def build_watermark_filter(self, video_label: str, out_w: int,
                                image_input_index: int | None):
        """
        Applies the persistent, whole-video watermark/logo on top of
        everything else (clips/motion/grade/text/captions are already
        baked into `video_label` by the time this runs). Returns None
        when the watermark is off or misconfigured (missing text with
        type=text, missing/unreadable file with type=image) so the
        caller can skip it cleanly rather than fail the whole export
        over a cosmetic extra.
        """
        wm = self.watermark
        if not wm.get("enabled"):
            return None
        opacity = _clamp(wm.get("opacity", 55), 5, 100) / 100.0

        if wm.get("type") == "image":
            if image_input_index is None:
                return None
            wx, wy = self._WATERMARK_OVERLAY_POS.get(wm.get("position"), self._WATERMARK_OVERLAY_POS["bottom-right"])
            target_w = max(20, int(out_w * _clamp(wm.get("scale", 16), 4, 60) / 100.0))
            out_label = "wmimg"
            frag = (
                f"[{image_input_index}:v]scale={target_w}:-1,format=rgba,"
                f"colorchannelmixer=aa={opacity:.3f}[wmsrc];"
                f"[{video_label}][wmsrc]overlay=x={wx}:y={wy}:shortest=1[{out_label}]"
            )
            return out_label, frag

        text = (wm.get("text") or "").strip()
        if not text:
            return None
        font_path = _resolve_font_file()
        if not font_path:
            return None
        font_clause = f"fontfile='{_escape_ffmpeg_filter_path(font_path)}':"
        x_expr, y_expr = self._WATERMARK_TEXT_POS.get(wm.get("position"), self._WATERMARK_TEXT_POS["bottom-right"])
        safe_text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")
        font_size = max(12, int(_clamp(wm.get("scale", 16), 4, 60)) + 8)
        out_label = "wmtext"
        frag = (
            f"[{video_label}]drawtext={font_clause}text='{safe_text}':fontsize={font_size}:"
            f"fontcolor=white@{opacity:.3f}:x='{x_expr}':y='{y_expr}':"
            f"shadowcolor=black@0.5:shadowx=1:shadowy=1[{out_label}]"
        )
        return out_label, frag

    def build_ffmpeg_command(self, output_path: str) -> list[str]:
        """
        Assembles the full argv list for the export. Kept as a pure
        (no side effects, no subprocess) builder so it can be unit-
        tested / logged independently of running it.
        """
        if not self.clips:
            raise VideoStudioError("Add at least 2 images to export a video.")

        out_w, out_h = _output_dimensions(self.settings["resolution"], self.settings["aspect_ratio"])
        fps = int(self.settings.get("fps", 30))
        durations = self._effective_durations()

        filter_complex, video_label, total_video = self.build_filter_complex(out_w, out_h, fps, durations)

        text_result = self.build_text_filter(video_label, out_w, out_h, fps, total_video)
        if isinstance(text_result, tuple):
            video_label, text_filters = text_result
            filter_complex = filter_complex + ";" + text_filters

        cmd = ["ffmpeg", "-y"]

        # ----- inputs: one looped image input per clip -----
        # No per-input `-t` here -- each clip's `trim=duration=` filter
        # (added in build_filter_complex above) is what actually pins
        # its length; see that filter's BUGFIX comment for why relying
        # on input-side `-t` alone produced a wildly-too-long clip with
        # zoompan in the chain.
        for clip in self.clips:
            cmd += ["-loop", "1", "-i", clip["path"]]

        next_input_index = len(self.clips)

        audio_path = self.audio.get("path")
        audio_input_index = None
        if audio_path and os.path.isfile(audio_path):
            audio_input_index = next_input_index
            if self.audio.get("duration_match") == "loop":
                # PHASE 12 duration matching: repeat a short track
                # seamlessly for the whole video instead of it playing
                # once and leaving the rest silent. `-stream_loop -1`
                # must sit on the input itself (before -i); the
                # afterwards atrim=0:total_video below is what then
                # cuts the now-infinite loop down to exactly the
                # video's length instead of running forever.
                cmd += ["-stream_loop", "-1", "-i", audio_path]
            else:
                cmd += ["-i", audio_path]
            next_input_index += 1

        # PHASE 12: second, limited-support audio track (e.g. a voice-
        # over) mixed underneath the primary music track.
        audio2_path = self.audio2.get("path")
        audio2_input_index = None
        if audio2_path and os.path.isfile(audio2_path):
            audio2_input_index = next_input_index
            cmd += ["-i", audio2_path]
            next_input_index += 1

        # PHASE 12: image watermark/logo needs its own looped input,
        # same trick as the clip images -- a single still frame fed
        # continuously so `overlay=...:shortest=1` can hold it on
        # screen for the entire video without it running out early.
        watermark_image_input_index = None
        if self.watermark.get("enabled") and self.watermark.get("type") == "image":
            wm_path = self.watermark.get("image_path")
            if wm_path and os.path.isfile(wm_path):
                watermark_image_input_index = next_input_index
                cmd += ["-loop", "1", "-i", wm_path]
                next_input_index += 1

        # Watermark is composited AFTER text/captions so branding always
        # sits on top, never hidden behind a caption box.
        wm_result = self.build_watermark_filter(video_label, out_w, watermark_image_input_index)
        if wm_result:
            video_label, wm_frag = wm_result
            filter_complex = filter_complex + ";" + wm_frag

        # ----- audio mix (volume + fade in/out, trimmed to video length) -----
        final_audio_label = None
        primary_audio_label = None
        if audio_input_index is not None:
            vol = _clamp(self.audio.get("volume", 80), 0, 100) / 100.0
            fade_in = self.audio.get("fade_in", 1.0)
            fade_out_start = max(0.0, total_video - self.audio.get("fade_out", 1.5))
            audio_chain = (
                f"[{audio_input_index}:a]atrim=0:{total_video:.3f},asetpts=PTS-STARTPTS,"
                f"volume={vol:.3f},afade=t=in:st=0:d={fade_in:.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={self.audio.get('fade_out', 1.5):.3f}[amain]"
            )
            filter_complex = filter_complex + ";" + audio_chain
            primary_audio_label = "amain"

        if audio2_input_index is not None:
            vol2 = _clamp(self.audio2.get("volume", 100), 0, 100) / 100.0
            # No fades/duration-matching on the second track by design
            # (see _default_audio2's docstring: intentionally the
            # "limited support" half of the pair) -- just trimmed to
            # the video length and volume-adjusted.
            a2_chain = (
                f"[{audio2_input_index}:a]atrim=0:{total_video:.3f},"
                f"asetpts=PTS-STARTPTS,volume={vol2:.3f}[avoice]"
            )
            filter_complex = filter_complex + ";" + a2_chain
            if primary_audio_label:
                # amix halves each input's amplitude by default (its
                # anti-clipping default), so `volume=2` after the mix
                # restores the original perceived loudness instead of
                # both tracks sounding quieter than either alone did.
                mix_chain = (
                    f"[{primary_audio_label}][avoice]amix=inputs=2:duration=first:"
                    f"dropout_transition=0,volume=2[aout]"
                )
                filter_complex = filter_complex + ";" + mix_chain
                final_audio_label = "aout"
            else:
                final_audio_label = "avoice"
        else:
            final_audio_label = primary_audio_label

        cmd += ["-filter_complex", filter_complex, "-map", f"[{video_label}]"]
        if final_audio_label:
            cmd += ["-map", f"[{final_audio_label}]"]

        # ----- quality / bitrate -----
        quality = self.settings.get("quality", "high")
        if quality == "custom":
            kbps = int(self.settings.get("custom_bitrate_kbps", 8000))
            cmd += ["-c:v", "libx264", "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k",
                    "-bufsize", f"{kbps * 2}k", "-preset", "medium"]
        else:
            crf = _QUALITY_CRF.get(quality, _QUALITY_CRF["high"])
            speed = _QUALITY_SPEED_PRESET.get(quality, "medium")
            cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", speed]

        cmd += ["-pix_fmt", "yuv420p", "-r", str(fps)]
        if final_audio_label:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", output_path]
        return cmd

    # ===================== EXPORT (threaded by ui/bridge.py) =====================

    def export(self, output_path: str, on_progress=None, cancel_event: threading.Event | None = None) -> dict:
        """
        Blocking call meant to be run on a background thread (same
        convention as ai/upscaler.py::upscale_image / BatchController.
        run()). on_progress(percent:int, label:str) is called
        periodically; cancel_event, if set mid-run, terminates FFmpeg
        and raises VideoExportCancelled.
        """
        status = ffmpeg_status()
        if not status["available"]:
            raise VideoStudioError(
                "FFmpeg wasn't found on PATH. Re-check Settings > Video > FFmpeg."
            )

        cmd = self.build_ffmpeg_command(output_path)
        total_duration = self.total_duration()
        # Audio never extends the video's length -- '-shortest' in
        # build_ffmpeg_command() enforces that, so total_duration (video
        # only) is always the right denominator for progress percent.

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if on_progress:
            on_progress(0, "Starting export...")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
        )

        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if cancel_event is not None and cancel_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise VideoExportCancelled("Export cancelled.")

                line = line.strip()
                if line.startswith("out_time_ms="):
                    try:
                        out_ms = int(line.split("=", 1)[1])
                        percent = 0 if total_duration <= 0 else _clamp(
                            int((out_ms / 1000.0 / 1000.0) / total_duration * 100), 0, 99
                        )
                        if on_progress:
                            on_progress(percent, f"Rendering... {percent}%")
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("progress=") and "end" in line:
                    break
        finally:
            proc.wait()

        if proc.returncode != 0:
            raise VideoStudioError(
                f"FFmpeg exited with an error (code {proc.returncode}). "
                f"Check that every source image is still on disk."
            )

        if on_progress:
            on_progress(100, "Export complete.")

        self._last_export_path = output_path
        return {
            "ok": True,
            "output_path": output_path,
            "duration": round(total_duration, 2),
        }

    # ===================== PREVIEW (low-res proxy) =====================

    def generate_preview(self, on_progress=None, cancel_event: threading.Event | None = None,
                          container: str = "webm") -> dict:
        """
        "Preview playback" -- renders the WHOLE timeline at a reduced
        resolution/short encode preset so it's fast enough to scrub
        through before committing to a full-quality export. Reuses
        build_ffmpeg_command() against a temp file with 720p forced.

        `container` picks the codec/container pair and is decided by
        the FRONTEND (video.js probes `<video>.canPlayType(...)`
        before calling this, see BUGFIX note there) rather than
        guessed here -- exactly which video codecs a given PySide6/Qt
        WebEngine build can decode inline varies by build (some ship
        the royalty-free VP8/VP9/Opus set Chromium always builds with,
        some -- rare, but real -- ship neither that nor H.264/AAC), so
        asking the actual embedded browser what IT supports beats
        hard-coding an assumption in Python that a later Qt/PySide6
        upgrade could silently invalidate again.
          - "webm" (default): VP9 + Opus. Royalty-free, so present on
            effectively every Chromium/QtWebEngine build regardless of
            the proprietary-codecs compile flag.
          - "mp4": H.264 + AAC, same codec choice as the real export.
            Only decodes inline on builds compiled WITH
            `-webengine-proprietary-codecs`.
        """
        if not self.clips:
            raise VideoStudioError("Add at least one image first.")
        if container not in ("webm", "mp4"):
            container = "webm"

        saved_resolution = self.settings["resolution"]
        try:
            # Cheap trick: temporarily force 720p (our smallest labeled
            # preset) rather than adding a whole parallel low-res code
            # path.
            self.settings = dict(self.settings)
            self.settings["resolution"] = "720p"
            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_video_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            preview_path = str(cache_dir / f"preview_{int(time.time() * 1000)}.{container}")

            fps = int(self.settings.get("fps", 30))
            has_audio = (
                (bool(self.audio.get("path")) and os.path.isfile(self.audio.get("path", "")))
                or (bool(self.audio2.get("path")) and os.path.isfile(self.audio2.get("path", "")))
            )
            cmd = self.build_ffmpeg_command(preview_path)
            # build_ffmpeg_command() ends with a fixed
            # "-c:v ... <output_path>" tail (see its own body) -- cut
            # everything from "-c:v" onward and replace it with
            # whichever container's tail we actually want, instead of
            # trying to patch flags in place.
            cut = cmd.index("-c:v") if "-c:v" in cmd else len(cmd) - 1
            cmd = cmd[:cut]
            if container == "webm":
                cmd += [
                    "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "34",
                    "-speed", "8", "-row-mt", "1",
                    "-pix_fmt", "yuv420p", "-r", str(fps),
                ]
                if has_audio:
                    cmd += ["-c:a", "libopus", "-b:a", "96k"]
            else:  # mp4 / h264 -- same codec as the real export, just faster
                cmd += [
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                    "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart",
                ]
                if has_audio:
                    cmd += ["-c:a", "aac", "-b:a", "128k"]
            cmd += ["-progress", "pipe:1", "-nostats", preview_path]

            total_duration = self.total_duration()
            if on_progress:
                on_progress(0, "Building preview...")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
            )
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    if cancel_event is not None and cancel_event.is_set():
                        proc.terminate()
                        raise VideoExportCancelled("Preview cancelled.")
                    line = line.strip()
                    if line.startswith("out_time_ms=") and total_duration > 0:
                        try:
                            out_ms = int(line.split("=", 1)[1])
                            percent = _clamp(int((out_ms / 1000.0 / 1000.0) / total_duration * 100), 0, 99)
                            if on_progress:
                                on_progress(percent, f"Building preview... {percent}%")
                        except (ValueError, IndexError):
                            pass
            finally:
                proc.wait()

            if proc.returncode != 0:
                raise VideoStudioError("Couldn't build a preview from the current timeline.")
            if on_progress:
                on_progress(100, "Preview ready.")
            return {"ok": True, "path": preview_path, "container": container}
        finally:
            self.settings["resolution"] = saved_resolution


# ===================== SHARED SINGLETON =====================
# Same "one shared controller object" convention as core/session.py's
# get_session() and core/batch_processor.py's get_batch_controller().

_video_project: VideoProject | None = None


def get_video_controller() -> VideoProject:
    global _video_project
    if _video_project is None:
        _video_project = VideoProject()
    return _video_project