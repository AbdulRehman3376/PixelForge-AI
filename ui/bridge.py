# Python <-> JavaScript bridge
#
# Every method the frontend needs to call must be decorated with @Slot
# and exposed here. This object is registered on the QWebChannel in
# ui/main_window.py and shows up in JavaScript as `pyBridge`
# (see frontend/js/bridge.js for the JS-side wiring).
#
# Keep this class thin: it should call into core/, ai/, video/, and
# database/ modules (built in later phases) rather than doing real
# work itself.
#
# PHASE 2 ADDITIONS:
#   - getImageInfo(path)        -> JSON string (width/height/format/size)
#   - chooseSaveImagePath(name) -> native "Save As" dialog for export
#   - exportImage(src, dest, adjustmentsJson) -> applies basic manual
#     controls (brightness/contrast/saturation) via Pillow and saves.
#     This is a deliberately thin stand-in for core/exporter.py +
#     core/enhancer.py, which get built properly in Phase 3.

import json
import os
import platform
import threading
import time
import tempfile
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QObject, QStandardPaths, Signal, Slot, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog

from core.analyzer import analyze_image
from ai.face_detector import detect_faces
from core.crop import crop_to_box, rotate_image, transpose_image
from core.enhancer import apply_adjustments, auto_white_balance
from core.filters import (
    apply_preset,
    apply_preset_stack,
    auto_suggest_preset,
    delete_preset,
    duplicate_preset,
    export_preset,
    get_preset,
    import_preset,
    list_presets,
    rename_preset,
    save_custom_preset,
    set_favorite,
)
from core.object_remover import erase_object
from core.session import PROJECT_EXTENSION, get_session, sweep_stale_sessions
from core.settings import get_all as get_all_settings
from core.settings import get_setting, reset_settings, set_setting
from core.smart_pipeline import (
    apply_pipeline,
    customize_pipeline,
    list_pipeline_rules,
    recommend_pipeline,
)

# PHASE 7 -- Prompt Engine (local, rule-based) + AI Image Generation
# (diffusers/local by default, optional cloud API). See
# core/prompt_engine.py and ai/image_generator.py for the real logic;
# this file only wires them to Slots, same layering as every other phase.
from core.prompt_engine import (
    add_history as prompt_add_history,
    clear_history as prompt_clear_history,
    delete_history_entry as prompt_delete_history_entry,
    get_history as prompt_get_history,
    preview_prompt,
)

# PHASE 8 -- Batch Processing (orchestration/queue layer, no model of
# its own -- see core/batch_processor.py's header for the full design).
from core.batch_processor import STATUS_DONE, get_batch_controller
from core import database as db

# PHASE 9 -- AI Upscaling (Real-ESRGAN via ONNX Runtime). See
# ai/upscaler.py's header for the full model-choice/design rationale;
# this file only wires it to Slots, same layering as every other phase.
from ai.upscaler import (
    UpscaleCancelled,
    UpscaleError,
    clear_model_cache as upscaler_clear_model_cache,
    model_status as upscaler_model_status,
    output_label as upscaler_output_label,
    safety_check as upscaler_safety_check,
    upscale_image,
)

# PHASE 10 -- Face Restoration (GFPGAN via ONNX Runtime). See
# ai/face_restorer.py's header for the full model-choice/design
# rationale (verified GFPGANv1.4.onnx download, crop-and-restore
# approach, classical strength/Natural<->Detailed/skin-protection
# post-passes); this file only wires it to Slots, same layering as
# every other phase.
from ai.face_restorer import (
    FaceRestoreCancelled,
    FaceRestoreError,
    clear_model_cache as facerestore_clear_model_cache,
    detect_faces_for_restore,
    model_status as facerestore_model_status,
    output_label as facerestore_output_label,
    restore_faces,
)

# PHASE 11 -- Video Studio (deterministic FFmpeg assembly: motion/
# transition math, no trained model). See core/video_studio.py's header
# for the full filter-graph design; this file only wires it to Slots,
# same layering as every other phase.
from core.video_studio import (
    PROJECT_EXTENSION as VIDEO_PROJECT_EXTENSION,
    VideoExportCancelled,
    VideoStudioError,
    ffmpeg_status as video_ffmpeg_status,
    get_video_controller,
)

# PHASE 14 -- Quality: logging + friendly error handling + system
# monitoring/health check. See each module's own header for the full
# design rationale. Deliberately minimal wiring here: _run_async below
# (the single choke point every AI/heavy Slot already runs through) now
# logs the full traceback and converts unexpected exceptions to a
# friendly message instead of leaking str(exc) straight to the UI --
# the {"ok": False, "error": ...} shape returned to JS is unchanged, so
# no frontend code needs to change.
from core.logger import get_logger
from core.error_handler import log_and_convert, log_exception, to_friendly_message
from core.health_check import run_startup_health_check
from core.system_monitor import get_system_snapshot
from core.resource_guard import ProcessingTimeout, run_with_timeout

_log = get_logger(__name__)


class Bridge(QObject):
    # ----- Signals: Python pushes updates to JavaScript -----
    statusChanged = Signal(str)          # e.g. "Ready", "Processing..."
    progressChanged = Signal(int, str)   # percent, label -- used by Phase 8 batch queue

    # BUGFIX (Remove BG / Object Erase): removeBackgroundPreview,
    # exportRemoveBackground, eraseObjectPreview and exportEraseObject all
    # used to run directly inside the @Slot call, on the Qt GUI thread.
    # rembg's first call in a session has to download+load its ONNX model
    # (~176MB, several seconds to tens of seconds depending on disk/CPU),
    # and cv2.inpaint on a full-res photo isn't instant either -- both
    # block that thread completely, so the whole window (repaints, the
    # view you just switched to, everything) freezes and Windows/macOS
    # flags it "Not Responding" until the call returns. That freeze is
    # also *why* it looked like the previous view (Enhance) was still
    # showing inside Remove BG: the last frame painted before the freeze
    # was still mid-transition, and a frozen window just displays whatever
    # was last drawn.
    #
    # Fix: the four heavy operations now run on a background thread
    # (_run_async below) and report back through this single signal
    # instead of a direct return value. taskResult is thread-safe to
    # emit from a worker thread -- Qt automatically queues the delivery
    # onto the GUI thread/JS side, so the UI stays responsive and the
    # spinner (already wired in removebg.js) is what the user sees
    # instead of a frozen window.
    taskResult = Signal(str, str)  # request_id, result_json

    # PHASE 8 -- Batch Processing. batchItemChanged fires once per
    # queue-row update (a status/progress change on a single item) so
    # the frontend can re-render just that row instead of re-polling
    # the whole queue after every image; batchProgressChanged carries
    # the overall bar + ETA (kept separate from the shared
    # progressChanged/statusChanged pair above, which the global status
    # bar already listens to for a simple "a batch is running" readout).
    batchItemChanged = Signal(str)          # item_json
    batchProgressChanged = Signal(int, str, int)  # percent, label, eta_seconds
    batchFinished = Signal(str)             # summary_json

    # Internal-only: lets the batch worker thread ask the GUI thread to
    # create/show the system tray notification (see _show_system_notification's
    # docstring below for why this indirection is required).
    _notifyRequested = Signal(str, str)

    def __init__(self, main_window):
        super().__init__()
        self._main_window = main_window
        # Caches the raw (unfeathered) rembg cutout for whichever image was
        # last background-removed, keyed by source path. rembg's model pass
        # is the slow part (~1-3s on CPU) -- edge-feather and background-
        # mode changes (color/blur/image) are re-cheap composites over this
        # same cutout, so they shouldn't re-run the model every time a
        # slider moves. Cleared automatically whenever a different source
        # image comes in.
        self._bg_cutout_cache = {"key": None, "path": None, "cutout": None}

        # PHASE 6: the one shared Edit Session every view reads from and
        # writes to (core/session.py). Held here because ui/bridge.py is
        # the only object the frontend can reach, so every mutation
        # arrives through a Slot and is therefore already serialized by
        # the request that triggered it.
        self._session = get_session()
        # Clear out session folders left behind by earlier runs (crash,
        # force-quit). Safe here and only here: nothing has started
        # editing yet, so the live session's folder can't be touched.
        try:
            sweep_stale_sessions()
        except Exception:  # noqa: BLE001 - temp cleanup must never block startup
            pass

        # 🩹 FIX (this session): every preview/export call across every
        # tool (Enhance, Filters, Remove BG, AI Upscale, Face
        # Restoration, etc.) writes a new, uniquely-timestamped temp
        # file into %TEMP%\pixelforge_preview or %TEMP%\pixelforge_working
        # -- but no call site ever deleted the PREVIOUS one. Left
        # running across many sessions/days, this silently filled up
        # the SYSTEM TEMP drive (C: on Windows by default, even when
        # the project itself lives on D:\Projects\PixelForge) -- this
        # is what was actually behind "C: drive filling up" reports.
        # Safe to wipe both folders here and only here: called once at
        # startup, before any preview/session has started, same
        # reasoning as sweep_stale_sessions() directly above.
        try:
            for _folder_name in ("pixelforge_preview", "pixelforge_working"):
                _folder = Path(tempfile.gettempdir()) / _folder_name
                if _folder.exists():
                    shutil.rmtree(_folder, ignore_errors=True)
        except Exception:  # noqa: BLE001 - temp cleanup must never block startup
            pass

        # PHASE 13 CONNECTOR: the database (core/database.py) shipped in
        # Phase 13 with a full Projects/Media/Edits/Video schema and every
        # Bridge Slot it needs (see the "Projects"/"Media"/"Edits"/
        # "Video projects" sections near the bottom of this file) -- but
        # nothing ever called those Slots from an actual editing tool, so
        # the Projects screen was always empty. This cache is the missing
        # link: which database project a given SOURCE FILE's exports
        # belong to. One project per distinct source file -- exporting a
        # photo/video you haven't touched before creates its own new
        # project (so exporting one thing never gets bundled together
        # with something unrelated you exported earlier); exporting the
        # SAME source again re-uses that file's project so repeated edits
        # build up one edit history instead of piling up duplicates. See
        # _db_project_for_source/_db_current_media/_record_edit/
        # _record_video_export below, and their call sites at the end of
        # every export*Async Slot and inside the batch worker's on_item
        # callback.
        self._db_project_by_source = {}  # source_path -> project_id
        self._db_media_cache = {}  # (project_id, input_path) -> media_id

        # PHASE 8: the one shared Batch queue for the app, same
        # single-shared-object convention as self._session above.
        self._batch = get_batch_controller()
        self._notifyRequested.connect(self._show_system_notification)

        # PHASE 9: cooperative cancellation for the single (non-batch)
        # AI Upscale run -- ai/upscaler.py checks this Event between
        # tiles so Cancel actually stops mid-run, same idea as Phase 8's
        # BatchController._cancel_flag but scoped to one image instead
        # of a whole queue.
        self._upscale_cancel_event = threading.Event()

        # PHASE 10: same cooperative-cancellation idea as Phase 9 above,
        # scoped to the single (non-batch) Face Restoration run --
        # ai/face_restorer.py checks this Event between faces and during
        # the one-time model download.
        self._facerestore_cancel_event = threading.Event()

        # PHASE 11: same cooperative-cancellation idea, scoped to the
        # current Video Studio export/preview render -- core/
        # video_studio.py checks this Event between FFmpeg progress
        # lines so Cancel actually kills the running FFmpeg process
        # instead of just hiding the progress bar client-side.
        self._video_cancel_event = threading.Event()
        self._video = get_video_controller()

    # ===================== ASYNC HELPER =====================

    def _run_async(self, request_id, fn):
        """
        Runs fn() on a background thread and emits taskResult(request_id,
        json_string) when it's done -- fn should return a plain dict
        (it gets json.dumps'd here) and may raise; exceptions are turned
        into the usual {"ok": False, "error": ...} shape so callers don't
        need their own try/except.
        """

        def _worker():
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message in JS
                # PHASE 14: log the full traceback and convert to a
                # user-friendly message instead of leaking str(exc)
                # raw -- see core/error_handler.py::log_and_convert.
                # Shape returned to JS is unchanged.
                result = log_and_convert(exc, context=getattr(fn, "__name__", request_id))
            self.taskResult.emit(request_id, json.dumps(result))

        threading.Thread(target=_worker, daemon=True).start()

    # ===================== APP INFO =====================

    @Slot(result=str)
    def getAppVersion(self):
        return "0.1.0-dev"

    @Slot(result=str)
    def getSystemInfo(self):
        return f"{platform.system()} {platform.release()} | Python {platform.python_version()}"

    @Slot(result=bool)
    def ping(self):
        """Lets the frontend confirm the bridge is actually connected."""
        return True

    # ===================== PHASE 14: SYSTEM HEALTH / MONITORING =====================
    # New, additive Slots -- a Settings-page "System" panel can call
    # these any time (not just at startup, unlike main.py's one-off
    # health check on launch) to show live CPU/RAM/disk figures and
    # re-run the same checks main.py already ran on launch. See
    # core/health_check.py and core/system_monitor.py for the real logic.

    @Slot(str)
    def getSystemHealth(self, request_id):
        """Re-runs the full startup-style health check on demand.
        Threaded (like every other heavy/slow Slot here via
        _run_async) -- this check imports onnxruntime/rembg to read
        AI model status, which can take several seconds on some
        machines, so it must never run on the GUI thread or it would
        freeze the whole window (this is exactly what happened when
        main.py used to run it synchronously at startup -- see that
        file's own comment for the full story).
        Emits taskResult(request_id, json) with:
        {"overall", "checks": [...], "summary"}."""
        def work():
            return run_startup_health_check()

        self._run_async(request_id, work)

    @Slot(str, result=str)
    def checkOutputWritable(self, folder_path):
        """PHASE 14: 'Disk write permission check before export' -- the
        frontend can call this right before an export dialog/action to
        catch an unwritable destination (network share dropped, folder
        moved, permissions) with a clear message BEFORE the export
        actually runs, instead of failing mid-export with a cryptic
        error. Returns JSON: {"ok", "checked_path", "error"}. Purely
        additive -- existing export Slots are unchanged and don't call
        this automatically, so nothing here alters current export
        behavior."""
        try:
            from core.system_monitor import check_disk_writable
            return json.dumps(check_disk_writable(folder_path))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="checkOutputWritable"))

    @Slot(result=str)
    def getSystemSnapshot(self):
        """Lightweight, fast CPU/RAM/disk snapshot for a live-updating
        Settings widget (unlike getSystemHealth, this doesn't touch
        FFmpeg/models/database -- just current resource usage).
        Returns JSON: {"cpu_percent", "ram": {...}, "disk": {...},
        "psutil_available"}.
        The "disk" dict is get_system_snapshot()'s usual {"ok",
        "free_mb", "required_mb", "checked_path"} PLUS "total_gb" /
        "used_percent", added here (not in core/system_monitor.py) so
        check_disk_space's existing return shape -- relied on elsewhere
        for the pre-export writable/space check -- stays untouched.
        Purely additive for the new Settings > System Health gauges."""
        try:
            snapshot = get_system_snapshot(live=True)
            disk = snapshot.get("disk") or {}
            if disk.get("ok"):
                try:
                    total_bytes = shutil.disk_usage(disk["checked_path"]).total
                    total_gb = total_bytes / (1024 ** 3)
                    free_gb = (disk.get("free_mb") or 0) / 1024
                    disk["total_gb"] = round(total_gb, 1)
                    disk["used_percent"] = round(
                        max(0.0, min(100.0, (1 - (free_gb / total_gb)) * 100)), 1
                    ) if total_gb > 0 else 0.0
                except OSError:
                    pass  # snapshot still returned below with just free_mb, no crash
            return json.dumps(snapshot)
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="getSystemSnapshot"))

    # ===================== FILE DIALOGS =====================

    @Slot(result=str)
    def openImageDialog(self):
        """Opens a native file picker for a single image. Returns '' if cancelled."""
        default_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        path, _ = QFileDialog.getOpenFileName(
            self._main_window,
            "Open Image",
            default_dir,
            "Images (*.jpg *.jpeg *.jfif *.png *.webp *.bmp *.tiff)",
        )
        return path

    @Slot(result=list)
    def openImagesDialog(self):
        """Opens a native file picker for multiple images (used by Batch, Video Studio)."""
        default_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        paths, _ = QFileDialog.getOpenFileNames(
            self._main_window,
            "Select Images",
            default_dir,
            "Images (*.jpg *.jpeg *.jfif *.png *.webp *.bmp *.tiff)",
        )
        return paths

    @Slot(str, result=bool)
    def openPathExternally(self, path):
        """
        Opens any file with the OS's own default app for it -- added
        alongside Phase 11's Video Studio because QWebEngineView's
        embedded Chromium often ships WITHOUT proprietary H.264/AAC
        decoders (Qt only bundles them if built with
        `-webengine-proprietary-codecs`, which most prebuilt PySide6
        wheels are not), so the in-page <video> preview player can
        silently fail to decode a real FFmpeg-exported MP4 even though
        the file itself is perfectly valid. This routes playback
        through Windows' own registered video player instead, which
        has no such codec gap. Generic on purpose (not video-specific)
        since "open this file with whatever the OS thinks is right"
        is broadly useful, not just for Video Studio.
        """
        if not path or not os.path.isfile(path):
            return False
        return QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    @Slot(result=str)
    def chooseOutputFolder(self):
        default_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        return QFileDialog.getExistingDirectory(
            self._main_window, "Choose Output Folder", default_dir
        )

    # ===================== SESSION (Shared Working Image) =====================
    #
    # PHASE 6 addition: exposes core/session.py's EditSession to the
    # frontend. This is what makes "Filter it -> open Enhance -> same
    # photo" true, gives Ctrl+Z/Ctrl+Y a single global history across
    # every tool, and backs Ctrl+S (Save Project) / Ctrl+Shift+S
    # (Export As). Every Slot here returns the same JSON shape as
    # core/session.py::EditSession.state() (plus an "error" key on
    # failure) so the frontend can always just re-render off the latest
    # returned state rather than tracking its own copy.
    #
    # IMPORTANT: this section only manages STATE (which file is the
    # current working image, undo/redo position, save/load). The actual
    # pixel-changing commits (sessionApplyAdjustments / sessionApplyPreset
    # / sessionCommitFile below) are what each tool calls once the user
    # accepts a result -- previews (previewAdjust, previewFilter, etc.)
    # intentionally do NOT touch the session, so dragging a slider around
    # doesn't spam the undo history.

    def _session_error(self, exc) -> str:
        # PHASE 14: shared by ~13 session Slots below -- fixing here logs
        # the full traceback and friendly-ifies the message for every one
        # of them at once, same idea as the _run_async choke point fix.
        log_exception(exc, context="session")
        return json.dumps({
            "ok": False,
            "error": to_friendly_message(exc),
            "has_session": bool(self._session.history),
        })

    @Slot(result=str)
    def sessionState(self):
        """Current session state (working image, history, undo/redo flags, dirty, etc.)."""
        try:
            return json.dumps(self._session.state())
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    @Slot(str, result=str)
    def sessionOpen(self, original_path):
        """
        Starts a brand-new edit session from `original_path` (the file the
        user just opened/dropped). Every view should call this right
        after it learns about a newly opened image, then read
        `working_path` from the returned state instead of the raw path
        it was handed -- see frontend/js/bridge.js::sessionOpen.
        """
        try:
            return json.dumps(self._session.start(original_path))
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    @Slot(result=str)
    def sessionClose(self):
        """Ends the current session (New Project / closing the image) and wipes its temp files."""
        try:
            return json.dumps(self._session.close())
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    @Slot(result=str)
    def sessionUndo(self):
        try:
            return json.dumps(self._session.undo())
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    @Slot(result=str)
    def sessionRedo(self):
        try:
            return json.dumps(self._session.redo())
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    @Slot(str, result=str)
    def sessionJumpTo(self, entry_id):
        """History-panel click: jump straight to a recorded step."""
        try:
            return json.dumps(self._session.jump_to(entry_id))
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    @Slot(result=str)
    def sessionRevertToOriginal(self):
        try:
            return json.dumps(self._session.revert_to_original())
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    @Slot(str, str, result=str)
    def sessionSetToolState(self, tool, state_json):
        """Remembers a tool's last-used UI values (sliders, selected preset, etc.) on the session."""
        try:
            state = json.loads(state_json) if state_json else {}
            return json.dumps(self._session.set_tool_state(tool, state))
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    @Slot(str, result=str)
    def sessionGetToolState(self, tool):
        try:
            return json.dumps({"ok": True, "tool": tool, "state": self._session.get_tool_state(tool)})
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    # ----- committing a tool's result as the new Working Image -----

    # BUGFIX: sessionApplyAdjustments / sessionApplyPreset used to be plain
    # Slots that ran full-resolution image processing (core/enhancer.py's
    # manual controls, or a Phase 4 preset) directly on the Qt GUI thread --
    # exactly the same class of bug documented on taskResult above for
    # Remove BG / Object Erase. A large photo with noise_reduction/clarity
    # (Enhance) or a stylized preset (Filters) can take several seconds at
    # full resolution, and for that whole time the window doesn't repaint,
    # so it's reported "Not Responding" by the OS even though it's still
    # working. Both now run on a background thread via _run_async and
    # report back through taskResult, same as the rest of this file's
    # heavy operations.
    @Slot(str, str, str)
    def sessionApplyAdjustmentsAsync(self, request_id, adjustments_json, label):
        """
        Enhance's "Apply" button: renders core/enhancer.py's manual
        controls onto the CURRENT working image (not the original) at
        full resolution, and commits the result as the new working image.
        """

        def work():
            adjustments = json.loads(adjustments_json) if adjustments_json else {}
            return self._session.commit_adjustments(adjustments, label or "Enhance")

        self._run_async(request_id, work)

    @Slot(str, str, float, str)
    def sessionApplyPresetAsync(self, request_id, preset_id, intensity, label):
        """Filters' "Apply" button: applies a Phase 4 preset to the working image and commits it."""

        def work():
            return self._session.commit_preset(preset_id, intensity, label or "")

        self._run_async(request_id, work)

    @Slot(str, str, str, result=str)
    def sessionCommitFile(self, result_path, tool, label):
        """
        Generic commit for tools that already render their own
        full-resolution output file (Remove BG's ai/bg_remover.py,
        Straighten/Crop's core/crop.py, Object Erase's
        core/object_remover.py) instead of handing back a raw PIL image.
        The file is copied into the session's own folder -- see
        core/session.py::EditSession.commit_file.
        """
        try:
            return json.dumps(self._session.commit_file(result_path, tool, label or "Edit"))
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    # ----- Ctrl+S (Save Project) / Ctrl+Shift+S is chooseSaveImagePath+exportImage above -----

    @Slot(result=str)
    def sessionChooseProjectSavePath(self):
        """Native 'Save Project' dialog, defaulting to the current project name if one exists."""
        default_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        suggested = self._session.project_path or str(
            Path(default_dir) / f"{Path(self._session.original_name or 'Untitled').stem}{PROJECT_EXTENSION}"
        )
        path, _ = QFileDialog.getSaveFileName(
            self._main_window, "Save Project", suggested, f"PixelForge Project (*{PROJECT_EXTENSION})"
        )
        return path

    @Slot(result=str)
    def sessionChooseProjectOpenPath(self):
        default_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        path, _ = QFileDialog.getOpenFileName(
            self._main_window, "Open Project", default_dir, f"PixelForge Project (*{PROJECT_EXTENSION})"
        )
        return path

    @Slot(str, result=str)
    def sessionSaveProject(self, dest_path):
        """
        Ctrl+S. Empty dest_path re-saves over the project's current
        location if it has one; otherwise the frontend should call
        sessionChooseProjectSavePath first and pass that path in.
        """
        try:
            return json.dumps(self._session.save_project(dest_path))
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    @Slot(str, result=str)
    def sessionLoadProject(self, path):
        try:
            return json.dumps(self._session.load_project(path))
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    # ===================== IMAGE EDITOR (Phase 2) =====================

    # BUGFIX: the file pickers below (openImageDialog, openImagesDialog,
    # Batch/Video Studio's picker) all accept TIFF and BMP alongside
    # jpg/png/webp -- but TIFF is not a format ANY browser engine can
    # decode in an <img>/<canvas> element, and several common BMP
    # compression variants aren't either. editor.js used to point
    # imgBefore/imgAfter straight at the original file path for
    # whatever format was opened; for a TIFF (or an unsupported BMP)
    # that fails deep in Chromium's Skia decoder with "Unknown or
    # unsupported skia image format" in the console and a blank/broken
    # image on screen -- it's not a corrupt file, the format itself
    # just isn't web-safe. Fix: convert anything outside the web-safe
    # set to a cached JPEG proxy once and hand the frontend THAT path
    # to display, while the original file stays what every real edit/
    # export operation (core/enhancer.py etc.) reads and writes.
    _WEB_SAFE_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP"}

    def _ensure_web_safe_preview(self, path: Path, fmt: str) -> str:
        if (fmt or "").upper() in self._WEB_SAFE_IMAGE_FORMATS:
            return str(path)
        try:
            stat = path.stat()
            key = f"{abs(hash((str(path), stat.st_mtime, stat.st_size)))}"
        except OSError:
            key = uuid.uuid4().hex
        cache_dir = Path(tempfile.gettempdir()) / "pixelforge_display_proxy"
        cache_dir.mkdir(parents=True, exist_ok=True)
        proxy_path = cache_dir / f"proxy_{key}.jpg"
        if not proxy_path.exists():
            # Write-then-rename so a slower disk or antivirus scan can
            # never leave editor.js's <img> reading a half-written
            # file (a second, rarer way to trigger the same Skia
            # decode error).
            tmp_path = proxy_path.with_suffix(".tmp")
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                img.convert("RGB").save(tmp_path, "JPEG", quality=95)
            os.replace(tmp_path, proxy_path)
        return str(proxy_path)

    @Slot(str, result=str)
    def getImageInfo(self, path):
        """
        Returns basic metadata for the currently opened image as a JSON
        string: {"ok": true, "width": ..., "height": ..., "format": ...,
        "size_bytes": ...} or {"ok": false, "error": "..."} on failure.

        JSON-string-over-the-wire is used instead of a QVariantMap slot
        return type to keep the bridge signature simple and avoid PySide
        dict-marshalling edge cases; the frontend just JSON.parses it.
        """
        try:
            p = Path(path)
            if not p.exists():
                return json.dumps({"ok": False, "error": "File not found."})

            with Image.open(p) as img:
                width, height = img.size
                fmt = img.format or p.suffix.replace(".", "").upper()

            display_path = self._ensure_web_safe_preview(p, fmt)

            return json.dumps(
                {
                    "ok": True,
                    "path": str(p),
                    "display_path": display_path,
                    "name": p.name,
                    "width": width,
                    "height": height,
                    "format": fmt,
                    "size_bytes": p.stat().st_size,
                }
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message in JS
            log_exception(exc, context="getImageInfo")
            return json.dumps({"ok": False, "error": f"Couldn't read image: {exc}"})

    @Slot(str, result=str)
    def chooseSaveImagePath(self, suggested_name):
        """Opens a native 'Save As' dialog for exporting the edited image."""
        default_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        default_path = str(Path(default_dir) / (suggested_name or "PixelForge_Export.jpg"))
        path, _ = QFileDialog.getSaveFileName(
            self._main_window,
            "Export Image",
            default_path,
            "JPEG (*.jpg *.jpeg);;PNG (*.png);;WEBP (*.webp);;TIFF (*.tiff)",
        )
        return path

    @Slot(str, str, result=str)
    def previewAdjust(self, source_path, adjustments_json):
        """
        Renders a downsized preview (max 1400px on the long edge, for
        speed) with the current manual-control values applied, saves it
        to a temp cache file, and returns that file's path. The editor
        swaps its "After" <img> src to this path on every debounced
        slider change, so what the user sees matches the real
        core/enhancer.py pipeline exactly -- not a CSS approximation.

        A fresh filename is used each call (timestamp-suffixed) so the
        browser doesn't serve a stale cached image for a reused path.

        Returns JSON: {"ok": true, "path": "..."} or {"ok": false, "error": "..."}
        """
        try:
            adjustments = json.loads(adjustments_json) if adjustments_json else {}

            with Image.open(source_path) as img:
                img = img.convert("RGB")
                img.thumbnail((1400, 1400), Image.LANCZOS)
                result = apply_adjustments(img, adjustments)

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            preview_path = cache_dir / f"preview_{int(time.time() * 1000)}.jpg"
            result.save(preview_path, quality=88)

            return json.dumps({"ok": True, "path": str(preview_path)})
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="previewAdjust")
            return json.dumps({"ok": False, "error": f"Preview failed: {exc}"})

    # BUGFIX: same freeze bug as sessionApplyAdjustments/sessionApplyPreset
    # above -- full-resolution export used to run directly on the GUI
    # thread. Now threaded via _run_async/taskResult.
    @Slot(str, str, str, str)
    def exportImageAsync(self, request_id, source_path, dest_path, adjustments_json):
        """
        Applies the full manual-adjustment pipeline (core/enhancer.py) at
        full resolution and saves to dest_path.

        This never overwrites the source file (non-destructive editing
        rule from the spec) -- dest_path always comes from the Save As
        dialog above, which cannot silently equal the source unless the
        user explicitly picks the same filename.

        Reports back through taskResult as {"ok": true, "path": "..."}
        or {"ok": false, "error": "..."}.
        """

        def work():
            if not source_path:
                return {"ok": False, "error": "No image loaded."}
            if not dest_path:
                return {"ok": False, "error": "Export cancelled."}

            adjustments = json.loads(adjustments_json) if adjustments_json else {}

            with Image.open(source_path) as img:
                result = apply_adjustments(img, adjustments)

                dest = Path(dest_path)
                dest.parent.mkdir(parents=True, exist_ok=True)

                save_kwargs = {}
                suffix = dest.suffix.lower()
                if suffix in (".jpg", ".jpeg"):
                    if result.mode == "RGBA":
                        result = result.convert("RGB")
                    save_kwargs["quality"] = 95
                # Missing-feature #5 (this session): re-attach the
                # original file's EXIF (captured once in
                # core/session.py::start(), since it's long gone from
                # this PNG-normalized working-image copy by now).
                exif_bytes = self._session.get_original_exif() if self._session else None
                if exif_bytes:
                    try:
                        # Best-effort -- some formats/Pillow builds
                        # reject the exif kwarg; fall back to a plain
                        # save rather than failing the whole export.
                        result.save(dest, exif=exif_bytes, **save_kwargs)
                    except Exception:  # noqa: BLE001
                        result.save(dest, **save_kwargs)
                else:
                    result.save(dest, **save_kwargs)

            self._record_edit(source_path, str(dest), "enhance", "Enhance",
                               adjustments)
            return {"ok": True, "path": str(dest)}

        self._run_async(request_id, work)

    @Slot(str, result=str)
    def autoWhiteBalance(self, path):
        """
        PHASE 3 -- Auto White Balance (dedicated button). Analyzes the
        source image's neutral midtones and returns a suggested
        temperature/tint pair (core/enhancer.py::auto_white_balance) --
        the frontend drops these straight into the Temperature/Tint
        sliders and re-previews, same pattern as autoEnhanceAnalyze
        below but scoped to just those two controls.

        Returns JSON: {"ok": true, "temperature": ..., "tint": ...,
        "summary": "..."} or {"ok": false, "error": "..."}.
        """
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((800, 800), Image.LANCZOS)
                return json.dumps(auto_white_balance(img))
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="autoWhiteBalance")
            return json.dumps({"ok": False, "error": f"Auto White Balance failed: {exc}"})

    @Slot(str, result=str)
    def autoEnhanceAnalyze(self, path):
        """
        PHASE 2/3 (now subject-aware, closing Phase 3's long-standing
        gap): Looks at the image's histogram/color/sharpness and
        suggests a value for EVERY manual control (brightness, contrast,
        saturation, exposure, highlights, shadows, whites, blacks,
        temperature, tint, vibrance, sharpness, clarity, noise_reduction)
        that nudges a dull/flat/dark/soft photo toward a more
        "professional" look -- not just the first three sliders.

        UPGRADED: works in LAB colorspace (like real photo editors) rather
        than plain RGB, uses percentile-based (not mean-based) tone
        analysis, applies a midtone-weighted "S-curve" style contrast
        push instead of a flat one, and now runs Phase 5's real face
        detector (ai/face_detector.py) to tell a portrait subject from a
        landscape/general scene -- a detected face gets gentler contrast/
        sharpness/clarity and stronger saturation/vibrance protection
        than an empty landscape, which can take a more assertive push.
        This closes the "true subject/scene-aware enhancement" gap this
        function's docstring used to flag as still open (it previously
        had no face/subject detection at all). Still a lighter-weight
        cousin of the full Smart Pipeline (core/analyzer.py +
        core/smart_pipeline.py, Phase 5/6) -- no scene classification,
        no recipe/step-skip reasoning -- but the tone/color reasoning
        below now mirrors what Lightroom/Photoshop's "Auto" does, tuned
        to be a bit more assertive ("impressive") on a genuinely weak
        photo while staying a light touch on a photo that's already
        close to ideal. Every value is a continuous, data-driven reading
        off this specific photo, clamped well inside each slider's safe
        range so it can never "destroy" the image.

        Returns JSON with one key per slider (brightness, contrast,
        saturation, exposure, highlights, shadows, whites, blacks,
        highlight_recovery, shadow_recovery, temperature, tint, vibrance,
        sharpness, clarity, noise_reduction) plus "summary",
        "subject_type" ("portrait" or "general"), and "face_count", or
        {"ok": false, "error": ".."} on failure.
        """
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                rgb_np = np.array(rgb)
                total_pixels = rgb_np.shape[0] * rgb_np.shape[1]
                if total_pixels == 0:
                    return json.dumps({"ok": False, "error": "Empty image."})

                # ----- LAB analysis -----
                # L (perceptual lightness), a (green<->red), b (blue<->
                # yellow). Real editors read color cast off a/b rather
                # than raw RGB averages, since RGB averaging gets thrown
                # off by a photo that's legitimately full of one color
                # (e.g. a green forest or a red sunset) -- LAB separates
                # "how light" from "what color" much more cleanly.
                lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float64)
                L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
                # OpenCV's L channel is 0-255 (scaled from the true 0-100),
                # a/b are stored with a +128 offset -- undo the offset so
                # 0 means "neutral" like everywhere else in this function.
                A = A - 128.0
                B = B - 128.0

                l_flat = L.flatten()
                mean_l = float(l_flat.mean())

                # Percentile-based black/white points instead of a simple
                # min/max histogram walk -- np.percentile is the same idea
                # (1st/99th percentile skips outlier pixels) but reads
                # more precisely off the real distribution.
                low, high = np.percentile(l_flat, [1, 99])
                spread = max(1.0, float(high - low))

                shadow_clip_pct = float((l_flat <= 6).mean()) * 100
                highlight_clip_pct = float((l_flat >= 249).mean()) * 100

                # ----- Subject/scene awareness (Phase 3's long-standing
                # gap, closed here) -----
                # Auto Enhance previously only read histogram/color/blur
                # statistics -- it never knew WHAT it was looking at, so
                # it treated a close-up portrait and an empty landscape
                # identically. Reuses Phase 5's real face detector
                # (ai/face_detector.py, the same Haar Cascade model
                # already used by the Analyzer and by Phase 9's
                # face-aware upscale) -- fast enough to run inline here,
                # same as this function's existing Laplacian/color-
                # clustering work already does synchronously. A face
                # large enough to be the clear subject (same >=3%-of-
                # frame-area rule core/analyzer.py::_classify_subject
                # uses) means "portrait": protect skin more, keep
                # contrast/clarity/sharpening gentler so skin doesn't
                # turn plastic. No such face means "landscape/general":
                # the photo can take a more assertive contrast/vibrance/
                # clarity push since there's no skin to protect.
                try:
                    faces = detect_faces(rgb)
                except Exception:  # noqa: BLE001
                    faces = []  # detection failure must never break Auto Enhance
                face_count = len(faces)
                largest_face_area = max((f["w"] * f["h"] for f in faces), default=0.0)
                is_portrait_subject = largest_face_area >= 0.03 or face_count >= 2
                # 1.0 = full push allowed (no clear human subject);
                # 0.55 = the strongest dial-back, once a real detected
                # face confirms this is a portrait -- a stronger, more
                # trustworthy signal than the HSV skin-color guess
                # below, which stays as a fallback for skin visible
                # outside/around a face Haar Cascade didn't catch.
                subject_guard = 0.55 if is_portrait_subject else 1.0

                # ----- Brightness suggestion -----
                # Target a mid-gray around 128 on the 0-255 L scale.
                brightness_pct = 100 + ((128 - mean_l) / 128) * 25
                brightness_pct = max(80, min(130, brightness_pct))

                # ----- Exposure suggestion -----
                exposure = max(-15, min(15, ((128 - mean_l) / 128) * 12))

                # ----- Contrast suggestion (midtone-weighted "S-curve") -----
                # A flat/hazy photo has a narrow L spread; scale the boost
                # continuously with how narrow it is. Weighted a touch
                # more aggressively than a flat linear boost so a genuinely
                # low-contrast photo gets real, visible punch (what makes
                # an auto-enhance actually look "professional" rather than
                # just technically correct) while a photo that already has
                # a full tonal range stays close to untouched.
                # Subject-aware: a detected portrait subject gets a gentler
                # contrast push (harsh contrast reads as "crunchy" on skin)
                # than a landscape/general photo, which can take more punch.
                contrast_pct = 100 + max(0, (160 - min(spread, 160))) * 0.19 * (0.7 if is_portrait_subject else 1.0)
                contrast_pct = max(100, min(124, contrast_pct))

                # ----- Skin-tone check (protects saturation/vibrance) -----
                # Cheap HSV-range heuristic: what fraction of the photo
                # falls in typical skin-tone hue/sat/value bands. Real
                # editors dial back color-boosting on portraits so faces
                # don't turn orange/plastic -- kept as a fallback signal
                # for skin visible outside/around a face (shoulders, arms)
                # that Haar Cascade doesn't detect as a "face" box.
                hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
                h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
                skin_mask = (
                    (h >= 3) & (h <= 28)
                    & (s >= 40) & (s <= 180)
                    & (v >= 60)
                )
                skin_fraction = float(skin_mask.mean())
                # 1.0 = no skin detected -> full boost allowed;
                # down to ~0.55 when the frame is mostly skin (a close
                # portrait) -> boost is dialed back substantially.
                # Combined (not just averaged) with subject_guard above --
                # a real detected face is the more trustworthy signal, so
                # whichever guard is stricter wins rather than the two
                # partially cancelling each other out.
                skin_guard = min(1.0 - min(0.45, skin_fraction * 0.9), subject_guard)

                # ----- Color sampling (saturation/vibrance/temperature/tint) -----
                small_hsv = cv2.cvtColor(np.array(rgb.resize((96, 96))), cv2.COLOR_RGB2HSV)
                avg_sat = float(small_hsv[:, :, 1].mean())  # 0-255 scale

                saturation_pct = 100 + max(0, (110 - min(avg_sat, 110))) * 0.22 * skin_guard
                saturation_pct = max(100, min(122, saturation_pct))
                vibrance = max(0, min(22, (110 - min(avg_sat, 110)) * 0.16 * skin_guard))

                # ----- Temperature / tint (gentle auto white balance) -----
                # Read straight off LAB's a/b means -- this is exactly
                # what a "cast" is: the average pixel isn't neutral gray.
                mean_a = float(A.mean())
                mean_b = float(B.mean())
                temperature = max(-15, min(15, -mean_b * 0.55))
                tint = max(-15, min(15, -mean_a * 0.55))

                # ----- Highlights / shadows / whites / blacks -----
                highlights = max(-22, min(0, -highlight_clip_pct * 6.5))
                shadows = max(0, min(22, shadow_clip_pct * 6.5 + max(0, 120 - spread) * 0.1))
                whites = max(0, min(16, (255 - high) * 0.16))
                blacks = max(0, min(16, low * 0.16))

                # ----- Highlight / shadow recovery -----
                # These pull detail back into genuinely clipped areas
                # (0-100 "how much recovery" controls, separate from the
                # highlights/shadows tone sliders above) -- the other
                # "professional" finishing touch a real Auto in
                # Lightroom/Photoshop applies on a blown-out sky or a
                # crushed-black shadow. Only kicks in once clipping is
                # actually present; a clean photo gets 0 on both.
                highlight_recovery = max(0, min(60, highlight_clip_pct * 9))
                shadow_recovery = max(0, min(60, shadow_clip_pct * 9))

                # ----- Sharpness / clarity (multi-scale blur detection) -----
                # Variance of the Laplacian at two scales (full-detail 512
                # and a softer 256 pass) is a more robust crispness read
                # than a single scale -- catches both fine-detail softness
                # and broader out-of-focus blur. High variance = crisp
                # photo that needs little help; low variance = soft photo
                # that benefits from a real sharpening push.
                gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
                lap_512 = cv2.Laplacian(cv2.resize(gray, (512, 512)), cv2.CV_64F).var()
                lap_256 = cv2.Laplacian(cv2.resize(gray, (256, 256)), cv2.CV_64F).var()
                laplacian_var = (lap_512 * 0.7) + (lap_256 * 0.3)
                sharpness = max(8, min(48, 48 - (min(laplacian_var, 400) / 400) * 40))
                clarity = max(8, min(22, 22 - min(spread, 240) * 0.055))
                # Subject-aware: a detected portrait subject gets its
                # sharpening/clarity push capped lower -- real editors go
                # easy on skin here too, since Clarity in particular
                # exaggerates pores/texture and Sharpness can produce
                # visible halos around a face's edges at full strength.
                if is_portrait_subject:
                    sharpness = min(sharpness, 30)
                    clarity = min(clarity, 14)

                # ----- Noise reduction -----
                noise_reduction = max(
                    0,
                    min(20, max(0, 60 - laplacian_var) * 0.2 + max(0, spread - 180) * 0.05),
                )

                summary_bits = []
                if brightness_pct >= 108:
                    summary_bits.append("brightened a dark shot")
                elif brightness_pct <= 92:
                    summary_bits.append("pulled back an overexposed shot")
                if contrast_pct >= 112:
                    summary_bits.append("added punch to a flat image")
                if saturation_pct >= 112:
                    summary_bits.append("lifted muted colors" if skin_fraction < 0.2 else "lifted color gently to protect skin tones")
                if highlights <= -5 or shadows >= 5 or highlight_recovery >= 8 or shadow_recovery >= 8:
                    summary_bits.append("recovered clipped highlights/shadows")
                if abs(temperature) >= 2 or abs(tint) >= 2:
                    summary_bits.append("balanced the color cast")
                if sharpness >= 24:
                    summary_bits.append("sharpened a soft image")
                if is_portrait_subject:
                    summary_bits.append(
                        f"protected {'faces' if face_count >= 2 else 'the face'} from over-processing"
                    )
                summary = (
                    "Looks good as-is -- only a light, professional touch-up applied."
                    if not summary_bits
                    else "Auto Enhance " + ", ".join(summary_bits) + "."
                )

                return json.dumps(
                    {
                        "ok": True,
                        "brightness": round(brightness_pct),
                        "contrast": round(contrast_pct),
                        "saturation": round(saturation_pct),
                        "exposure": round(exposure),
                        "highlights": round(highlights),
                        "shadows": round(shadows),
                        "whites": round(whites),
                        "blacks": round(blacks),
                        "highlight_recovery": round(highlight_recovery),
                        "shadow_recovery": round(shadow_recovery),
                        "temperature": round(temperature),
                        "tint": round(tint),
                        "vibrance": round(vibrance),
                        "sharpness": round(sharpness),
                        "clarity": round(clarity),
                        "noise_reduction": round(noise_reduction),
                        "summary": summary,
                        "subject_type": "portrait" if is_portrait_subject else "general",
                        "face_count": face_count,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="autoEnhanceAnalyze")
            return json.dumps({"ok": False, "error": f"Auto Enhance failed: {exc}"})

    # ===================== PHASE 5: ANALYZER =====================
    #
    # Async like removeBackgroundPreviewAsync/eraseObjectPreviewAsync
    # above (see _run_async / taskResult) rather than a plain @Slot
    # return -- face detection (ai/face_detector.py) plus k-means color
    # clustering (core/analyzer.py) is real, if modest, CPU work, and
    # this project's own pattern is "anything that runs a model goes
    # through the background thread + taskResult signal so the UI never
    # blocks" (see the taskResult Signal docstring above).

    @Slot(str, str)
    def analyzeImageAsync(self, request_id, path):
        """
        Runs the full Phase 5 analysis (core/analyzer.py::analyze_image)
        on the image at `path` and reports back through the taskResult
        signal as {"ok": true, ...analysis fields...} or
        {"ok": false, "error": "..."}. See core/analyzer.py's
        analyze_image() docstring for the full result shape -- this now
        includes a dedicated `sky` read, an expanded `quality` block
        (exposure_score/dynamic_range/overall_score/grade), and a
        `light_quality` read (harsh/soft/uneven/even), on top of the
        original Phase 5 fields.
        """

        def work():
            with Image.open(path) as img:
                result = analyze_image(img)
            return {"ok": True, **result}

        self._run_async(request_id, work)

    # ===================== PHASE 6: SMART PIPELINE =====================
    #
    # 🧮 Not AI itself (see core/smart_pipeline.py's header note) -- but
    # still async/threaded like analyzeImageAsync above, for a simple
    # reason: getting a recommendation means running the SAME Phase 5
    # analysis first (face detection + color clustering included), then
    # feeding it through the rule engine. The rule engine step itself is
    # instant; the analysis step is the real work, so this whole call is
    # threaded exactly like Analyzer's own Slot rather than trying to
    # thread just half of it. Deliberately does NOT reuse a cached
    # analyzeImageAsync result from the frontend -- keeping this one
    # call self-contained (photo path in, recommendation out) means the
    # frontend doesn't have to track "did the user already Analyze this
    # exact photo" bookkeeping just to get a pipeline suggestion.

    @Slot(str, str)
    def smartPipelineAsync(self, request_id, path):
        """
        PHASE 6 (v2). Runs core/analyzer.py::analyze_image() on the photo,
        feeds it through core/smart_pipeline.py::recommend_pipeline(), and
        reports back through taskResult as:

            {"ok": true, "analysis": {...}, "recommendation": {...},
             "session": {...}}

        `path` may be an empty string, in which case the CURRENT WORKING
        IMAGE is analyzed -- that's the important change from v1. The
        pipeline now reasons about what the user is actually looking at
        after their filter/enhance/background edits, not about the
        untouched original.

        The analysis and the recommendation are stashed on the shared
        session so the Preview / Customize / Apply calls below don't have
        to re-analyze (a 1-3s job) on every slider nudge, and so a saved
        project can reopen with its pipeline card intact.

        Still purely advisory: this touches no pixels. Applying is a
        separate, explicit call (smartPipelineApplyAsync) -- the spec's
        "user can always override" rule.
        """

        def work():
            source = path or self._session.working_path
            if not source:
                raise ValueError("No image is open yet. Import an image first.")
            with Image.open(source) as img:
                analysis = analyze_image(img)
            recommendation = recommend_pipeline(analysis)
            if self._session.working_path:
                self._session.set_analysis(analysis)
                self._session.set_pipeline(recommendation)
            return {
                "ok": True,
                "analysis": analysis,
                "recommendation": recommendation,
                "session": self._session.state(),
            }

        self._run_async(request_id, work)

    def _pipeline_analysis(self):
        """
        The analysis the pipeline calls below reason about: the cached one
        if it's still valid, otherwise a fresh read of the working image.
        Cached rather than recomputed because Customize can be clicked
        many times per second (a slider), and re-running face detection +
        k-means per keystroke would make it feel broken.
        """
        self._session.require_active()
        if self._session.analysis:
            return self._session.analysis
        with Image.open(self._session.working_path) as img:
            analysis = analyze_image(img)
        self._session.set_analysis(analysis)
        return analysis

    def _pipeline_from_request(self, look_id, overrides_json):
        analysis = self._pipeline_analysis()
        overrides = json.loads(overrides_json) if overrides_json else {}
        target_look = look_id or (self._session.pipeline or {}).get("id", "")
        recommendation = customize_pipeline(analysis, target_look, overrides)
        self._session.set_pipeline(recommendation)
        return recommendation

    @Slot(str, str, str)
    def smartPipelineCustomizeAsync(self, request_id, look_id, overrides_json):
        """
        Recomputes the recipe under the user's own choices -- the
        Customize panel, the Intensity slider, and "pick one of the Other
        Recommendations" all land here (same code path, no special
        cases).

        overrides_json:
            {"intensity": 0-100,
             "steps": {"sharpening": "skip", ...},
             "step_values": {"exposure": {"exposure": 12}}}

        Returns {"ok": true, "recommendation": {...}} -- including a
        freshly re-run safety check, since toggling a step on is exactly
        the kind of change that can introduce a clipping risk.
        """

        def work():
            return {"ok": True, "recommendation": self._pipeline_from_request(look_id, overrides_json)}

        self._run_async(request_id, work)

    @Slot(str, str, str)
    def smartPipelinePreviewAsync(self, request_id, look_id, overrides_json):
        """
        Renders the recipe onto a DOWNSIZED copy of the working image and
        returns the temp file path -- the "Preview" button, and the live
        preview behind the Intensity slider.

        Same 1400px preview / full-res export split as previewAdjust and
        previewFilter, so this is fast enough to drive interactively while
        still being the real core/enhancer.py pipeline rather than a CSS
        approximation.
        """

        def work():
            recommendation = self._pipeline_from_request(look_id, overrides_json)
            with Image.open(self._session.working_path) as img:
                preview_src = img.convert("RGB")
                preview_src.thumbnail((1400, 1400), Image.LANCZOS)
                result = apply_pipeline(preview_src, recommendation)

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            preview_path = cache_dir / f"pipeline_{int(time.time() * 1000)}.jpg"
            result.convert("RGB").save(preview_path, quality=88)

            return {"ok": True, "path": str(preview_path), "recommendation": recommendation}

        self._run_async(request_id, work)

    @Slot(str, str, str)
    def smartPipelineApplyAsync(self, request_id, look_id, overrides_json):
        """
        The "Apply" button. Renders the recipe at FULL RESOLUTION onto the
        current working image and commits the result as the new working
        image, so every other view picks it up on entry.

        The user's original file is not touched (core/session.py's rule 3)
        -- the result lands in the session's temp folder and only reaches
        disk when the user exports.
        """

        def work():
            recommendation = self._pipeline_from_request(look_id, overrides_json)
            source = self._session.open_working()
            result = apply_pipeline(source, recommendation)
            recipe = recommendation.get("recipe") or {}
            label = f"{recommendation.get('name', 'Smart Pipeline')} {recipe.get('intensity', '')}%".strip()
            state = self._session.commit_image(
                result, "pipeline", label,
                settings={
                    "look_id": recommendation.get("id"),
                    "preset_id": recommendation.get("preset_id"),
                    "intensity": recipe.get("intensity"),
                    "adjustments": recommendation.get("adjustments"),
                },
                op={"kind": "pipeline", "look_id": recommendation.get("id"),
                    "overrides": json.loads(overrides_json) if overrides_json else {}},
            )
            # The pixels changed, so the cached analysis describes the
            # PREVIOUS state -- drop it so the next Smart Pipeline run
            # reads the image the user is now looking at.
            self._session.set_analysis(None)
            return {"ok": True, "session": state, "recommendation": recommendation}

        self._run_async(request_id, work)

    @Slot(result=str)
    def listPipelineLooks(self):
        """
        The look catalogue behind the recommendations, so the decision
        logic is inspectable instead of a black box (Settings/docs view).
        """
        try:
            return json.dumps({"ok": True, "looks": list_pipeline_rules()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="listPipelineLooks"))

    # ===================== STRAIGHTEN / CROP (Phase 3, on-request addition) =====================
    #
    # Two independent, user-confirmed steps (straighten, then crop) --
    # see core/crop.py's module docstring for why they're kept separate
    # rather than one combined call. Each one saves a NEW temp working
    # file and hands its path back to the frontend, which reloads it as
    # the editor's new base image (loadImage() in editor.js) -- same
    # non-destructive hand-off pattern already used by "Continue in
    # Remove BG" after an export. Saved as PNG (lossless) since these
    # are intermediate working files, not the final export.

    @Slot(str, float, result=str)
    def straightenImage(self, source_path, angle):
        """
        Rotates the full-resolution image at source_path by `angle`
        degrees (straighten a slightly tilted photo -- see
        core/crop.py::rotate_image) and saves the result as a new temp
        working file.

        Returns JSON: {"ok": true, "path": "...", "width": ..., "height": ...}
        or {"ok": false, "error": "..."}.
        """
        try:
            if not source_path:
                return json.dumps({"ok": False, "error": "No image loaded."})
            if not angle:
                return json.dumps({"ok": False, "error": "Nothing to straighten -- angle is 0."})

            with Image.open(source_path) as img:
                result = rotate_image(img, angle)

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_working"
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_path = cache_dir / f"straighten_{int(time.time() * 1000)}.png"
            result.save(out_path)

            return json.dumps(
                {"ok": True, "path": str(out_path), "width": result.width, "height": result.height}
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="straightenImage")
            return json.dumps({"ok": False, "error": f"Straighten failed: {exc}"})

    @Slot(str, str, result=str)
    def cropImage(self, source_path, crop_box_json):
        """
        Crops the full-resolution image at source_path to crop_box
        (JSON array [left, top, right, bottom] in that image's own
        pixel coordinates -- see core/crop.py::crop_to_box) and saves
        the result as a new temp working file, same hand-off pattern as
        straightenImage() above.

        Returns JSON: {"ok": true, "path": "...", "width": ..., "height": ...}
        or {"ok": false, "error": "..."}.
        """
        try:
            if not source_path:
                return json.dumps({"ok": False, "error": "No image loaded."})
            box = json.loads(crop_box_json) if crop_box_json else None
            if not box or len(box) != 4:
                return json.dumps({"ok": False, "error": "No crop area selected."})

            with Image.open(source_path) as img:
                result = crop_to_box(img, box)

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_working"
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_path = cache_dir / f"crop_{int(time.time() * 1000)}.png"
            result.save(out_path)

            return json.dumps(
                {"ok": True, "path": str(out_path), "width": result.width, "height": result.height}
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="cropImage")
            return json.dumps({"ok": False, "error": f"Crop failed: {exc}"})

    # Missing-feature #1 (this session): Rotate 90 / Flip H / Flip V.
    # Same hand-off pattern as straightenImage/cropImage above (full-res
    # PIL op -> temp PNG -> frontend commits it via session.commitFile).
    # Kept as a plain fast Slot like straighten/crop, not threaded like
    # the Apply/Export fix elsewhere in this file -- transpose() is a
    # single memory-layout operation with no resampling, so even a large
    # photo finishes in milliseconds; it's not the kind of operation
    # that risks freezing the GUI thread.
    @Slot(str, str, result=str)
    def transposeImage(self, source_path, op):
        """
        Rotates/flips the full-resolution image at source_path
        losslessly (see core/crop.py::transpose_image) and saves the
        result as a new temp working file.

        Returns JSON: {"ok": true, "path": "...", "width": ..., "height": ...}
        or {"ok": false, "error": "..."}.
        """
        try:
            if not source_path:
                return json.dumps({"ok": False, "error": "No image loaded."})

            with Image.open(source_path) as img:
                result = transpose_image(img, op)

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_working"
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_path = cache_dir / f"transpose_{int(time.time() * 1000)}.png"
            result.save(out_path)

            return json.dumps(
                {"ok": True, "path": str(out_path), "width": result.width, "height": result.height}
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="transposeImage")
            return json.dumps({"ok": False, "error": f"Rotate/Flip failed: {exc}"})

    # ===================== BACKGROUND / OBJECT REMOVER (Phase 2A) =====================
    #
    # Two independent tools sharing the "Remove BG" screen:
    #   - Background removal/replacement, backed by ai/bg_remover.py (rembg)
    #   - Object erase (mask-brush + inpaint), backed by core/object_remover.py
    #
    # Both follow the same preview/export split as Phase 2's enhancer:
    # *Preview* slots work on a downsized copy and write to a temp cache
    # file for a fast, responsive UI; *export* slots run at full
    # resolution and save to a user-chosen destination, never overwriting
    # the source (same non-destructive rule as exportImage above).

    def _get_cutout(self, source_path, hair_refine=False):
        """
        Returns the cached raw rembg cutout for source_path, computing it
        if needed. Cached per (source_path, hair_refine) -- toggling Hair
        Refine needs a real re-run of the model (it changes rembg's
        alpha-matting pass), so it's part of the cache key rather than a
        cheap post-process like feather/expand-contract.
        """
        from ai.bg_remover import remove_background

        cache_key = (source_path, bool(hair_refine))
        if self._bg_cutout_cache.get("key") == cache_key and self._bg_cutout_cache["cutout"] is not None:
            return self._bg_cutout_cache["cutout"]

        with Image.open(source_path) as img:
            cutout = remove_background(img.convert("RGB"), edge_feather=0, hair_refine=bool(hair_refine))

        self._bg_cutout_cache = {"key": cache_key, "path": source_path, "cutout": cutout}
        return cutout

    def _composite_bg(self, source_path, cutout, options):
        """
        Applies the full touch-up pipeline to a cutout, then the
        background mode, then crop. `options` keys: mode, edge_feather,
        color, blur_radius, background_path, expand_contract (-30..30),
        touchup_mask (data URL of keep/remove brush strokes, or None),
        shadow_mode ("none"|"preserve"|"remove"), shadow_strength
        (0-100), crop_box ([left, top, right, bottom] in the ORIGINAL
        image's pixel coordinates), crop_source_width/crop_source_height
        (the original image's dimensions, needed to scale crop_box down
        to whatever size `cutout` actually is -- preview vs export).

        Order: expand/contract -> touch-up mask -> feather -> shadow ->
        background composite -> crop. Expand/contract and the touch-up
        mask both edit the cutout's own silhouette, so they run before
        feather (which softens whatever edge results) and before shadow
        recovery (which needs the cutout's final alpha to know what's
        "outside" it). Crop runs last so it applies to the final
        composited image. Returns a PIL Image (RGBA for transparent,
        RGB otherwise).
        """
        from ai.bg_remover import (
            apply_manual_mask,
            expand_contract_alpha,
            feather_alpha,
            manual_crop,
            replace_background,
            shadow_preserve,
            shadow_remove,
        )

        working = cutout

        expand_amount = int(options.get("expand_contract", 0) or 0)
        if expand_amount:
            working = expand_contract_alpha(working, expand_amount)

        touchup_mask = options.get("touchup_mask")
        if touchup_mask:
            working = apply_manual_mask(working, touchup_mask)

        # Missing-feature #6 (this session): Edge Decontamination. Runs
        # after the silhouette-editing steps above (expand/contract,
        # touch-up mask) but before Feather Edge below, since it reads
        # the RAW edge-alpha geometry to know which pixels are actually
        # "edge" pixels -- feathering first would soften that signal.
        if options.get("decontaminate"):
            from ai.bg_remover import decontaminate_edges
            working = decontaminate_edges(working, strength=int(options.get("decontaminate_strength", 100) or 0))

        feather = int(options.get("edge_feather", 0) or 0)
        if feather > 0:
            working = feather_alpha(working, feather)

        shadow_mode = options.get("shadow_mode", "none")
        if shadow_mode == "preserve":
            strength = int(options.get("shadow_strength", 50) or 0)
            with Image.open(source_path) as original:
                original_rgb = original.convert("RGB").copy()
            # shadow_preserve reads shadow pixels off the ORIGINAL photo,
            # but `working` may be a downsized preview cutout -- resize
            # the original to match so alpha/pixel coords line up.
            if original_rgb.size != working.size:
                original_rgb = original_rgb.resize(working.size, Image.LANCZOS)
            working = shadow_preserve(original_rgb, working, strength)
        elif shadow_mode == "remove":
            working = shadow_remove(working)

        mode = options.get("mode", "transparent")
        kwargs = {}
        if mode == "color":
            color = options.get("color", [255, 255, 255])
            kwargs["color"] = tuple(int(c) for c in color)
        elif mode == "blur":
            with Image.open(source_path) as original:
                kwargs["original"] = original.convert("RGB").copy()
            kwargs["blur_radius"] = int(options.get("blur_radius", 18) or 18)
        elif mode == "gradient":
            # Missing-feature #8 (this session): Background Gradient.
            kwargs["color1"] = tuple(int(c) for c in options.get("color1", (30, 30, 40)))
            kwargs["color2"] = tuple(int(c) for c in options.get("color2", (200, 200, 220)))
            kwargs["angle"] = float(options.get("angle", 90) or 90)
        elif mode == "image":
            bg_path = options.get("background_path")
            if not bg_path:
                raise ValueError("Background image mode selected but no background_path provided.")
            kwargs["background_path"] = bg_path
            # Missing-feature #9 (this session): Background Position/Scale.
            kwargs["scale"] = float(options.get("scale", 1.0) or 1.0)
            kwargs["offset_x"] = float(options.get("offset_x", 50) or 50)
            kwargs["offset_y"] = float(options.get("offset_y", 50) or 50)

        result = replace_background(working, mode, **kwargs)

        crop_box = options.get("crop_box")
        if crop_box and len(crop_box) == 4:
            src_w = options.get("crop_source_width") or result.width
            src_h = options.get("crop_source_height") or result.height
            scale_x = result.width / src_w if src_w else 1.0
            scale_y = result.height / src_h if src_h else 1.0
            scaled_box = (
                crop_box[0] * scale_x,
                crop_box[1] * scale_y,
                crop_box[2] * scale_x,
                crop_box[3] * scale_y,
            )
            result = manual_crop(result, scaled_box)

        return result

    @Slot(str, str, str)
    def removeBackgroundPreviewAsync(self, request_id, source_path, options_json):
        """
        Async version of background-removal preview (see _run_async /
        taskResult above for why this isn't a direct-return @Slot
        anymore). Runs on a worker thread; result comes back through
        the taskResult signal as {"ok": true, "path": "..."} or
        {"ok": false, "error": "..."}.

        options_json keys: mode ("transparent"|"color"|"blur"|"image"),
        edge_feather (0-100), color ([r,g,b], mode="color"), blur_radius
        (int, mode="blur"), background_path (str, mode="image"),
        hair_refine (bool), expand_contract (-30..30), touchup_mask
        (data URL or None), shadow_mode ("none"|"preserve"|"remove"),
        shadow_strength (0-100), crop_box ([l,t,r,b] or None),
        crop_source_width/crop_source_height (ints, needed with crop_box).
        """

        def work():
            options = json.loads(options_json) if options_json else {}

            cutout = self._get_cutout(source_path, hair_refine=bool(options.get("hair_refine")))
            # Downsize the cutout itself for a fast preview -- full-res
            # compositing happens only in exportRemoveBackgroundAsync below.
            preview_cutout = cutout.copy()
            preview_cutout.thumbnail((1000, 1000), Image.LANCZOS)

            result = self._composite_bg(source_path, preview_cutout, options)

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            preview_path = cache_dir / f"removebg_{int(time.time() * 1000)}.png"
            result.save(preview_path)

            return {"ok": True, "path": str(preview_path)}

        self._run_async(request_id, work)

    @Slot(str, str, str, str)
    def exportRemoveBackgroundAsync(self, request_id, source_path, dest_path, options_json):
        """
        Async, full-resolution version of removeBackgroundPreviewAsync,
        saved to dest_path (from chooseSaveImagePath). Transparent mode
        requires a PNG/WEBP destination -- a JPEG destination gets
        flattened onto white with a friendly note in the response
        rather than silently losing transparency.
        """

        def work():
            if not source_path:
                return {"ok": False, "error": "No image loaded."}
            if not dest_path:
                return {"ok": False, "error": "Export cancelled."}

            options = json.loads(options_json) if options_json else {}
            cutout = self._get_cutout(source_path, hair_refine=bool(options.get("hair_refine")))
            result = self._composite_bg(source_path, cutout, options)

            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            suffix = dest.suffix.lower()

            note = None
            if result.mode == "RGBA" and suffix in (".jpg", ".jpeg"):
                flattened = Image.new("RGB", result.size, (255, 255, 255))
                flattened.paste(result, (0, 0), mask=result.split()[3])
                result = flattened
                note = "JPEG doesn't support transparency -- exported on a white background instead."

            save_kwargs = {"quality": 95} if suffix in (".jpg", ".jpeg") else {}
            result.save(dest, **save_kwargs)

            self._record_edit(source_path, str(dest), "removebg",
                               "Remove Background", options)

            response = {"ok": True, "path": str(dest)}
            if note:
                response["note"] = note
            return response

        self._run_async(request_id, work)

    @Slot(result=str)
    def chooseBackgroundImage(self):
        """Native file picker for a custom replacement background image. Returns '' if cancelled."""
        default_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        path, _ = QFileDialog.getOpenFileName(
            self._main_window,
            "Choose Background Image",
            default_dir,
            "Images (*.jpg *.jpeg *.jfif *.png *.webp *.bmp *.tiff)",
        )
        return path

    @Slot(str, str, str, str)
    def eraseObjectPreviewAsync(self, request_id, source_path, mask_data_url, method):
        """
        Async version of object-erase preview. Inpaints the masked
        region on a downsized (max 1000px) copy of source_path and
        writes the result to a temp JPG for preview.

        mask_data_url: base64 PNG data URL from the mask-brush canvas
        (see core/object_remover.py for the exact format expected --
        white = erase, black = keep). method: "telea" or "navier_stokes".
        """

        def work():
            with Image.open(source_path) as img:
                rgb = img.convert("RGB")
                rgb.thumbnail((1000, 1000), Image.LANCZOS)
                result = erase_object(rgb, mask_data_url, method or "telea")

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            preview_path = cache_dir / f"eraseobj_{int(time.time() * 1000)}.jpg"
            result.save(preview_path, quality=90)

            return {"ok": True, "path": str(preview_path)}

        self._run_async(request_id, work)

    @Slot(str, str, str, str, str)
    def exportEraseObjectAsync(self, request_id, source_path, dest_path, mask_data_url, method):
        """
        Async, full-resolution object erase, saved to dest_path. The
        mask is drawn at preview resolution but core/object_remover.py
        resizes it to match the full-res image, so the same brush
        strokes apply cleanly regardless of what size the user was
        looking at.
        """

        def work():
            if not source_path:
                return {"ok": False, "error": "No image loaded."}
            if not dest_path:
                return {"ok": False, "error": "Export cancelled."}

            with Image.open(source_path) as img:
                result = erase_object(img, mask_data_url, method or "telea")

            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = {"quality": 95} if dest.suffix.lower() in (".jpg", ".jpeg") else {}
            result.save(dest, **save_kwargs)

            self._record_edit(source_path, str(dest), "object_remover",
                               "Erase Object", {"method": method or "telea"})
            return {"ok": True, "path": str(dest)}

        self._run_async(request_id, work)

    # ===================== PRESETS / FILTERS (Phase 4) =====================
    #
    # Classic deterministic processing (core/filters.py wraps
    # core/enhancer.py's adjustment pipeline + monochrome/grain) -- no
    # model load, so these stay synchronous @Slot methods, same pattern
    # as previewAdjust/exportImage/straightenImage/cropImage above,
    # rather than the threaded _run_async pattern used by rembg/inpaint.

    @Slot(result=str)
    def listFilterPresets(self):
        """
        Returns every preset (9 builtins + any custom ones), each
        annotated with is_builtin/is_favorite.

        Returns JSON: {"ok": true, "presets": [...]} or
        {"ok": false, "error": "..."}.
        """
        try:
            return json.dumps({"ok": True, "presets": list_presets()})
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="listFilterPresets")
            return json.dumps({"ok": False, "error": f"Couldn't load presets: {exc}"})

    @Slot(str, str, float, result=str)
    def previewFilter(self, source_path, preset_id, intensity):
        """
        Renders a downsized preview (max 1400px, same cap as
        previewAdjust) with `preset_id` applied at `intensity` (0-100),
        saves it to a temp cache file, and returns that file's path.

        Returns JSON: {"ok": true, "path": "..."} or {"ok": false, "error": "..."}
        """
        try:
            preset = get_preset(preset_id)
            if not preset:
                return json.dumps({"ok": False, "error": "Preset not found."})

            with Image.open(source_path) as img:
                img = img.convert("RGB")
                img.thumbnail((1400, 1400), Image.LANCZOS)
                result = apply_preset(img, preset, intensity)

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            preview_path = cache_dir / f"filter_{int(time.time() * 1000)}.jpg"
            result.save(preview_path, quality=88)

            return json.dumps({"ok": True, "path": str(preview_path)})
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="previewFilter")
            return json.dumps({"ok": False, "error": f"Preview failed: {exc}"})

    # BUGFIX: same freeze bug as exportImage/sessionApplyAdjustments/
    # sessionApplyPreset above -- full-resolution preset export used to
    # run directly on the GUI thread. Now threaded via _run_async/taskResult.
    @Slot(str, str, str, str, float)
    def exportFilterImageAsync(self, request_id, source_path, dest_path, preset_id, intensity):
        """
        Applies `preset_id` at `intensity` (0-100) at full resolution
        and saves to dest_path -- same non-destructive rule as
        exportImage (dest_path always comes from a Save As dialog, and
        this never overwrites the source).

        Reports back through taskResult as {"ok": true, "path": "..."}
        or {"ok": false, "error": "..."}.
        """

        def work():
            if not source_path:
                return {"ok": False, "error": "No image loaded."}
            if not dest_path:
                return {"ok": False, "error": "Export cancelled."}

            preset = get_preset(preset_id)
            if not preset:
                return {"ok": False, "error": "Preset not found."}

            with Image.open(source_path) as img:
                result = apply_preset(img, preset, intensity)

                dest = Path(dest_path)
                dest.parent.mkdir(parents=True, exist_ok=True)

                save_kwargs = {}
                if dest.suffix.lower() in (".jpg", ".jpeg"):
                    if result.mode == "RGBA":
                        result = result.convert("RGB")
                    save_kwargs["quality"] = 95
                # Missing-feature #5 (this session): same EXIF re-attach
                # as exportImageAsync above.
                exif_bytes = self._session.get_original_exif() if self._session else None
                if exif_bytes:
                    try:
                        result.save(dest, exif=exif_bytes, **save_kwargs)
                    except Exception:  # noqa: BLE001
                        result.save(dest, **save_kwargs)
                else:
                    result.save(dest, **save_kwargs)

            self._record_edit(
                source_path, str(dest), "filters",
                preset.get("name", "Filter"),
                {"preset_id": preset_id, "intensity": intensity},
            )
            return {"ok": True, "path": str(dest)}

        self._run_async(request_id, work)

    # ===================== PHASE 4: Auto Filter =====================

    @Slot(str, result=str)
    def autoFilterSuggest(self, path):
        """
        Analyzes the photo and suggests the best-fit builtin preset +
        intensity (core/filters.py::auto_suggest_preset). The frontend
        selects that preset card and sets Intensity to the suggestion,
        same hand-off pattern as autoWhiteBalance/autoEnhanceAnalyze.

        Returns JSON: {"ok": true, "preset_id": ..., "intensity": ...,
        "summary": "..."} or {"ok": false, "error": "..."}.
        """
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((800, 800), Image.LANCZOS)
                return json.dumps(auto_suggest_preset(img))
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="autoFilterSuggest")
            return json.dumps({"ok": False, "error": f"Auto Filter failed: {exc}"})

    # ===================== PHASE 4: Filter Stacking =====================

    @Slot(str, str, result=str)
    def previewPresetStack(self, source_path, layers_json):
        """
        Renders a downsized preview with 2+ presets combined (see
        core/filters.py::apply_preset_stack). layers_json:
        '[{"preset_id": "...", "intensity": 0-100}, ...]'.

        Returns JSON: {"ok": true, "path": "..."} or {"ok": false, "error": "..."}
        """
        try:
            layers = json.loads(layers_json) if layers_json else []
            if len(layers) < 2:
                return json.dumps({"ok": False, "error": "Add at least 2 filters to the stack."})

            with Image.open(source_path) as img:
                img = img.convert("RGB")
                img.thumbnail((1400, 1400), Image.LANCZOS)
                result = apply_preset_stack(img, layers)

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            preview_path = cache_dir / f"stack_{int(time.time() * 1000)}.jpg"
            result.save(preview_path, quality=88)
            return json.dumps({"ok": True, "path": str(preview_path)})
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="previewPresetStack")
            return json.dumps({"ok": False, "error": f"Stack preview failed: {exc}"})

    @Slot(str, str, str, str)
    def exportPresetStackAsync(self, request_id, source_path, dest_path, layers_json):
        """
        Full-resolution export of a filter stack -- same threaded
        pattern and non-destructive dest_path rule as
        exportFilterImageAsync above. Reports back through taskResult.
        """

        def work():
            if not source_path:
                return {"ok": False, "error": "No image loaded."}
            if not dest_path:
                return {"ok": False, "error": "Export cancelled."}
            layers = json.loads(layers_json) if layers_json else []
            if len(layers) < 2:
                return {"ok": False, "error": "Add at least 2 filters to the stack."}

            with Image.open(source_path) as img:
                result = apply_preset_stack(img, layers)

                dest = Path(dest_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                save_kwargs = {}
                if dest.suffix.lower() in (".jpg", ".jpeg"):
                    if result.mode == "RGBA":
                        result = result.convert("RGB")
                    save_kwargs["quality"] = 95
                exif_bytes = self._session.get_original_exif() if self._session else None
                if exif_bytes:
                    try:
                        result.save(dest, exif=exif_bytes, **save_kwargs)
                    except Exception:  # noqa: BLE001
                        result.save(dest, **save_kwargs)
                else:
                    result.save(dest, **save_kwargs)

            self._record_edit(source_path, str(dest), "filters_stack",
                               "Filter Stack", {"layers": layers})
            return {"ok": True, "path": str(dest)}

        self._run_async(request_id, work)

    # ===================== PHASE 4: Filter Comparison =====================

    @Slot(str, str, float, result=str)
    def previewFilterBatch(self, source_path, preset_ids_json, intensity):
        """
        Renders one small preview per preset id, all at the same
        `intensity`, for side-by-side Filter Comparison -- a single
        round trip instead of one previewFilter() call per preset.
        Previews are capped at 500px (comparison thumbnails, not the
        full editing canvas) so N of them stay fast.

        Returns JSON: {"ok": true, "results": [{"preset_id", "path"}]}
        or {"ok": false, "error": "..."}.
        """
        try:
            preset_ids = json.loads(preset_ids_json) if preset_ids_json else []
            if len(preset_ids) < 2:
                return json.dumps({"ok": False, "error": "Select at least 2 filters to compare."})

            with Image.open(source_path) as base_img:
                base_img = base_img.convert("RGB")
                base_img.thumbnail((500, 500), Image.LANCZOS)

                cache_dir = Path(tempfile.gettempdir()) / "pixelforge_preview"
                cache_dir.mkdir(parents=True, exist_ok=True)

                results = []
                for preset_id in preset_ids:
                    preset = get_preset(preset_id)
                    if not preset:
                        continue
                    rendered = apply_preset(base_img.copy(), preset, intensity)
                    out_path = cache_dir / f"compare_{preset_id}_{int(time.time() * 1000)}.jpg"
                    rendered.save(out_path, quality=85)
                    results.append({"preset_id": preset_id, "path": str(out_path)})

            return json.dumps({"ok": True, "results": results})
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="previewFilterBatch")
            return json.dumps({"ok": False, "error": f"Comparison failed: {exc}"})

    # ===================== PHASE 4: Custom Filter live preview =====================

    @Slot(str, str, result=str)
    def previewCustomFilter(self, source_path, filter_json):
        """
        Live preview while BUILDING a brand-new custom filter from
        scratch (not from an existing preset) -- filter_json:
        '{"adjustments": {...}, "grain": 0-100, "monochrome": bool}'.
        Renders the same way apply_preset does, just without a saved
        preset_id, so the frontend's "Create Custom Filter" sliders can
        show a live result before Save.

        Returns JSON: {"ok": true, "path": "..."} or {"ok": false, "error": "..."}
        """
        try:
            data = json.loads(filter_json) if filter_json else {}
            preset_like = {
                "adjustments": data.get("adjustments", {}),
                "grain": data.get("grain", 0),
                "monochrome": bool(data.get("monochrome", False)),
            }
            with Image.open(source_path) as img:
                img = img.convert("RGB")
                img.thumbnail((1400, 1400), Image.LANCZOS)
                result = apply_preset(img, preset_like, intensity=100)

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            preview_path = cache_dir / f"customfilter_{int(time.time() * 1000)}.jpg"
            result.save(preview_path, quality=88)
            return json.dumps({"ok": True, "path": str(preview_path)})
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="previewCustomFilter")
            return json.dumps({"ok": False, "error": f"Preview failed: {exc}"})

    @Slot(str, str, result=str)
    def saveCustomPreset(self, name, preset_json):
        """
        Creates a new custom preset. preset_json keys: adjustments
        (dict, only keys differing from the enhancer defaults need to
        be included), grain (0-100), monochrome (bool) -- same shape
        core/filters.py's BUILTIN_PRESETS use.

        Returns JSON: {"ok": true, "preset": {...}} or {"ok": false, "error": "..."}
        """
        try:
            data = json.loads(preset_json) if preset_json else {}
            preset = save_custom_preset(
                name=name,
                adjustments=data.get("adjustments", {}),
                grain=data.get("grain", 0),
                monochrome=data.get("monochrome", False),
            )
            return json.dumps({"ok": True, "preset": preset})
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="saveCustomPreset")
            return json.dumps({"ok": False, "error": f"Couldn't save preset: {exc}"})

    @Slot(str, str, result=str)
    def renameCustomPreset(self, preset_id, new_name):
        try:
            preset = rename_preset(preset_id, new_name)
            return json.dumps({"ok": True, "preset": preset})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="renameCustomPreset"))

    @Slot(str, result=str)
    def duplicateCustomPreset(self, preset_id):
        try:
            preset = duplicate_preset(preset_id)
            return json.dumps({"ok": True, "preset": preset})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="duplicateCustomPreset"))

    @Slot(str, result=str)
    def deleteCustomPreset(self, preset_id):
        try:
            delete_preset(preset_id)
            return json.dumps({"ok": True})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="deleteCustomPreset"))

    @Slot(str, bool, result=str)
    def setPresetFavorite(self, preset_id, favorite):
        try:
            set_favorite(preset_id, favorite)
            return json.dumps({"ok": True})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="setPresetFavorite"))

    @Slot(str, str, result=str)
    def exportPresetToFile(self, preset_id, suggested_name):
        """
        Opens a native "Save As" dialog and writes `preset_id` out as a
        standalone, shareable .json file. Returns '' (folded into the
        ok/path JSON below) if the dialog is cancelled.
        """
        try:
            default_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            default_path = str(Path(default_dir) / (suggested_name or "preset.json"))
            dest_path, _ = QFileDialog.getSaveFileName(
                self._main_window, "Export Preset", default_path, "Preset JSON (*.json)"
            )
            if not dest_path:
                return json.dumps({"ok": False, "error": "Export cancelled."})

            export_preset(preset_id, dest_path)
            return json.dumps({"ok": True, "path": dest_path})
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="exportPresetToFile")
            return json.dumps({"ok": False, "error": f"Couldn't export preset: {exc}"})

    @Slot(result=str)
    def importPresetFromFile(self):
        """
        Opens a native "Open" dialog for a .json preset file (as written
        by exportPresetToFile) and imports it as a new custom preset.
        """
        try:
            default_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            source_path, _ = QFileDialog.getOpenFileName(
                self._main_window, "Import Preset", default_dir, "Preset JSON (*.json)"
            )
            if not source_path:
                return json.dumps({"ok": False, "error": "Import cancelled."})

            preset = import_preset(source_path)
            return json.dumps({"ok": True, "preset": preset})
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="importPresetFromFile")
            return json.dumps({"ok": False, "error": f"Couldn't import preset: {exc}"})

    # ===================== PHASE 7: PROMPT ENGINE =====================
    #
    # Local, rule-based prompt understanding (core/prompt_engine.py --
    # no LLM, no mandatory paid API). Fast/synchronous like Smart
    # Pipeline's listPipelineLooks above -- pure string/dict work, no
    # model load, so no need for the _run_async pattern here.

    @Slot(str, str, result=str)
    def promptEnginePreview(self, prompt, negative_prompt):
        """
        Parses `prompt` (+ optional `negative_prompt`) into the
        structured preview the spec calls "Prompt preview": detected
        style/mood/lighting/color/background/composition/quality,
        contradiction warnings, and any leftover words that matched no
        known category ("Unknown instruction handling" -- reported
        back to the user, never silently dropped or crashed on).

        Returns JSON: {"ok": true, "summary": "...", "summary_lines": [...],
        "categories": {...}, "conflicts": [...], "unrecognized": [...],
        "has_warnings": bool, "has_unrecognized": bool, "is_empty": bool}
        """
        try:
            return json.dumps(preview_prompt(prompt or "", negative_prompt or ""))
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="promptEnginePreview")
            return json.dumps({"ok": False, "error": f"Couldn't parse prompt: {exc}"})

    @Slot(result=str)
    def promptHistoryList(self):
        """Returns JSON: {"ok": true, "history": [{"id","prompt",...}, ...]}"""
        try:
            limit = int(get_setting("history_limit", "30") or 30)
            return json.dumps({"ok": True, "history": prompt_get_history(limit)})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="promptHistoryList"))

    @Slot(str, result=str)
    def promptHistoryDelete(self, entry_id):
        try:
            removed = prompt_delete_history_entry(entry_id)
            return json.dumps({"ok": True, "removed": removed})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="promptHistoryDelete"))

    @Slot(result=str)
    def promptHistoryClear(self):
        try:
            prompt_clear_history()
            return json.dumps({"ok": True})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="promptHistoryClear"))

    # ===================== PHASE 7: AI IMAGE GENERATION =====================
    #
    # Text-to-image, backed by ai/image_generator.py (local diffusers
    # pipeline by default, optional opt-in cloud API). Model load +
    # inference is genuinely heavy (seconds to minutes on CPU), so this
    # follows the SAME background-thread + taskResult pattern as Remove
    # BG / Object Erase above (see the BUGFIX note near the top of this
    # file) -- generating must never freeze the GUI thread. Progress is
    # additionally streamed through the existing global
    # statusChanged/progressChanged signals (same bar Batch/Upscale will
    # use later) so the user sees "loading model" / "step 2/4" instead
    # of a dead spinner.

    @Slot(result=str)
    def aiImageGenAvailability(self):
        """
        Reports whether local and/or cloud generation are currently
        usable, so the frontend can show an honest state ("Local
        generation ready" / "Install required packages" / "No cloud
        provider configured") instead of only discovering it after a
        failed generate click.

        Returns JSON: {"ok": true, "local": {...}, "cloud": {...}}
        """
        try:
            from ai.image_generator import ASPECT_RATIOS, cloud_provider_status, local_model_status

            return json.dumps(
                {
                    "ok": True,
                    "local": local_model_status(),
                    "cloud": cloud_provider_status(),
                    "aspect_ratios": list(ASPECT_RATIOS.keys()),
                }
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="aiImageGenAvailability"))

    def _generate_image_progress(self, percent, label):
        # Reuses the app's single global status bar (see
        # progressChanged's docstring at the top of this class) rather
        # than a bespoke per-view progress UI.
        self.progressChanged.emit(percent, label)
        if percent < 0 or percent >= 100:
            self.statusChanged.emit(label)

    @Slot(str, str)
    def generateImageAsync(self, request_id, params_json):
        """
        params_json keys: prompt, negative_prompt, aspect_ratio,
        custom_width, custom_height, num_images (1-4), steps,
        guidance_scale, seed, provider ("local"|"cloud"), model_id,
        save_to_history (bool, default true).

        Emits taskResult(request_id, json) with:
        {"ok": true, "provider": "...", "width": ..., "height": ...,
         "images": [{"path": "...", "seed": ...}, ...]}
        or {"ok": false, "error": "..."} on any failure -- including a
        missing local dependency or an unconfigured cloud provider,
        both surfaced as a readable message rather than a crash
        ("Generation progress/error handling" requirement).
        """
        def _worker():
            from ai.image_generator import GenerationError, generate

            try:
                params = json.loads(params_json) if params_json else {}
                result = generate(
                    prompt=params.get("prompt", ""),
                    negative_prompt=params.get("negative_prompt", ""),
                    aspect_ratio=params.get("aspect_ratio", "square"),
                    custom_width=params.get("custom_width"),
                    custom_height=params.get("custom_height"),
                    num_images=params.get("num_images", 1),
                    steps=params.get("steps"),
                    seed=params.get("seed"),
                    provider=params.get("provider", "local"),
                    model_id=params.get("model_id"),
                    on_progress=self._generate_image_progress,
                )
                if params.get("save_to_history", True):
                    try:
                        prompt_add_history(
                            result["prompt"],
                            result["negative_prompt"],
                            {
                                "aspect_ratio": params.get("aspect_ratio", "square"),
                                "provider": result["provider"],
                                "model_id": result.get("model_id"),
                                "num_images": params.get("num_images", 1),
                            },
                        )
                    except Exception:  # noqa: BLE001 -- history is a convenience, must never fail the generation
                        pass
            except GenerationError as exc:
                result = {"ok": False, "error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": f"Generation failed: {exc}"}

            self.taskResult.emit(request_id, json.dumps(result))

        threading.Thread(target=_worker, daemon=True).start()

    @Slot(str, str)
    def regenerateImageAsync(self, request_id, previous_spec_json):
        """
        Re-runs the exact settings of a previous result with a fresh
        seed ("Regenerate" requirement) -- see
        ai/image_generator.py::regenerate for the seed behaviour.
        Same taskResult contract as generateImageAsync above.
        """
        def _worker():
            from ai.image_generator import GenerationError, regenerate

            try:
                previous_spec = json.loads(previous_spec_json) if previous_spec_json else {}
                result = regenerate(previous_spec, on_progress=self._generate_image_progress)
            except GenerationError as exc:
                result = {"ok": False, "error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": f"Regeneration failed: {exc}"}

            self.taskResult.emit(request_id, json.dumps(result))

        threading.Thread(target=_worker, daemon=True).start()

    @Slot(str, result=str)
    def sendGeneratedImageToEditor(self, generated_path):
        """
        "Generated image -> PixelForge editor" hand-off. A generated
        PNG under cache/generated/ isn't yet part of any Edit Session,
        so this starts a brand-new one anchored to it -- identical to
        opening any other photo via Import (see core/session.py::
        EditSession and sessionOpen below), which is what gives the
        generated image full access to Enhance/Filters/Remove BG and
        the existing Save/Export workflow for free, per the spec's
        "Save/export integration" requirement.

        Returns the same JSON shape as sessionOpen().
        """
        try:
            if not generated_path or not Path(generated_path).exists():
                return json.dumps({"ok": False, "error": "That generated image is no longer available."})
            return json.dumps(self._session.start(generated_path))
        except Exception as exc:  # noqa: BLE001
            return self._session_error(exc)

    # ===================== PHASE 8: BATCH PROCESSING =====================
    #
    # 🧮 Not AI -- orchestrates Phase 5 (Analyzer) / Phase 6 (Smart
    # Pipeline) / Phase 4 (Presets) per image, see core/batch_processor.py
    # for the real queue/threading logic. This section stays a thin
    # Slot/Signal wrapper around the single shared BatchController
    # (self._batch), same layering as every other phase.
    #
    # batchStartAsync follows the same request_id/taskResult contract as
    # generateImageAsync etc (see _run_async's docstring above) for the
    # FINAL result, but also streams live per-item and overall-progress
    # updates via the dedicated batchItemChanged/batchProgressChanged
    # signals while it runs -- a single taskResult at the very end
    # wouldn't let the queue table update per-image.

    @Slot(result=str)
    def batchState(self):
        try:
            return json.dumps({"ok": True, **self._batch.state()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="batchState"))

    @Slot(result=str)
    def chooseBatchInputFolder(self):
        """Native folder picker for Batch's 'Folder import' control."""
        default_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        return QFileDialog.getExistingDirectory(self._main_window, "Import Folder", default_dir)

    @Slot(str, result=str)
    def batchAddFiles(self, paths_json):
        try:
            paths = json.loads(paths_json) if paths_json else []
            result = self._batch.add_files(paths)
            return json.dumps({"ok": True, **result, **self._batch.state()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="batchAddFiles"))

    @Slot(str, bool, result=str)
    def batchAddFolder(self, folder, recursive):
        try:
            result = self._batch.add_folder(folder, recursive=bool(recursive))
            return json.dumps({"ok": True, **result, **self._batch.state()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="batchAddFolder"))

    @Slot(result=str)
    def batchClear(self):
        try:
            self._batch.clear()
            return json.dumps({"ok": True, **self._batch.state()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="batchClear"))

    # PHASE 14: crash recovery -- see core/batch_processor.py's
    # get_recovery_state()/discard_recovery_state() docstrings. Purely
    # additive: nothing calls these automatically on the Python side,
    # the frontend decides when to check (see frontend/batch.js).
    @Slot(result=str)
    def getBatchRecoveryState(self):
        try:
            from core.batch_processor import get_recovery_state
            return json.dumps(get_recovery_state())
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="getBatchRecoveryState"))

    @Slot()
    def discardBatchRecoveryState(self):
        from core.batch_processor import discard_recovery_state
        discard_recovery_state()

    @Slot(str, result=str)
    def batchRemoveItem(self, item_id):
        ok = self._batch.remove_item(item_id)
        return json.dumps({"ok": ok, **self._batch.state()})

    @Slot(str, bool, result=str)
    def batchSetIncluded(self, item_id, included):
        ok = self._batch.set_included(item_id, included)
        return json.dumps({"ok": ok, **self._batch.state()})

    @Slot(str, result=str)
    def batchReorder(self, item_ids_json):
        """Drag-to-reorder -- item_ids_json is the full new order."""
        try:
            item_ids = json.loads(item_ids_json) if item_ids_json else []
        except Exception:  # noqa: BLE001
            item_ids = []
        ok = self._batch.reorder(item_ids)
        return json.dumps({"ok": ok, **self._batch.state()})

    @Slot(str, str, result=str)
    def batchBumpPriority(self, item_id, direction):
        """direction: 'top' | 'up' | 'down' | 'bottom' -- the priority-queue control."""
        ok = self._batch.bump_priority(item_id, direction)
        return json.dumps({"ok": ok, **self._batch.state()})

    @Slot(str, result=str)
    def batchSetOptions(self, options_json):
        """
        options keys (all optional, unspecified ones keep their current
        value -- see core/batch_processor.py::DEFAULT_OPTIONS): mode
        ("smart"|"preset"|"none"), preset_id, intensity, output_folder,
        export_format ("keep"|"jpg"|"png"|"webp"), export_quality (1-100),
        resize_mode ("none"|"max_dimension"|"percent"), resize_value,
        naming_template (tokens: {name}{ext}{index}{date}{time}{preset}),
        preserve_structure, overwrite, strip_exif, copyright_author,
        cpu_throttle (0-100), notify_on_complete.
        """
        try:
            options = json.loads(options_json) if options_json else {}
            merged = self._batch.set_options(options)
            return json.dumps({"ok": True, "options": merged})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="batchSetOptions"))

    @Slot(result=str)
    def batchDetectDuplicates(self):
        """Selective duplicate/near-duplicate detection -- flags similar
        photos before processing (see BatchController.detect_duplicates)."""
        try:
            groups = self._batch.detect_duplicates()
            return json.dumps({"ok": True, "groups": groups, **self._batch.state()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="batchDetectDuplicates"))

    @Slot(result=str)
    def batchCheckDiskSpace(self):
        try:
            return json.dumps({"ok": True, **self._batch.check_disk_space()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="batchCheckDiskSpace"))

    def _batch_run_worker(self, request_id):
        def on_item(item):
            # PHASE 13 CONNECTOR: mirror every successfully-finished
            # queue item into the database too, same as every other
            # export path in this file -- otherwise a Batch run (often
            # the largest source of edits in a single sitting) would be
            # the one thing the Projects screen never sees.
            if item.status == STATUS_DONE and item.output_path:
                self._record_edit(
                    item.source_path, item.output_path, "batch",
                    item.applied_look or "Batch",
                    {"category": item.category, "warnings": item.warnings},
                )
            self.batchItemChanged.emit(json.dumps(item.to_dict()))

        def on_progress(pct, label, eta):
            self.batchProgressChanged.emit(pct, label, eta)
            # Also feeds the app-wide global status bar (see
            # progressChanged's docstring at the top of this class) so
            # a batch running in the background is visible from any view.
            self.progressChanged.emit(pct, label)

        try:
            summary = self._batch.run(on_progress=on_progress, on_item=on_item)
            result = {"ok": True, "summary": summary, **self._batch.state()}
            if self._batch.options.get("notify_on_complete", True):
                self._notifyRequested.emit(
                    "PixelForge — Batch complete",
                    f"{summary.get('processed', 0)} processed, {summary.get('failed', 0)} failed.",
                )
        except Exception as exc:  # noqa: BLE001
            result = log_and_convert(exc, context="runBatch")

        self.statusChanged.emit("Ready")
        self.batchFinished.emit(json.dumps(result))
        self.taskResult.emit(request_id, json.dumps(result))

    @Slot(str, str)
    def batchStartAsync(self, request_id, options_json):
        """
        Starts the queue on a background thread (see core/batch_processor.py's
        threading-model note -- pause/cancel/reorder/etc remain callable
        from the GUI thread while this runs). `options_json`, if non-empty,
        is applied via batchSetOptions before starting, so the frontend can
        send "start with these options" in one call.
        """
        try:
            if options_json:
                self._batch.set_options(json.loads(options_json))
        except Exception:  # noqa: BLE001 -- bad options JSON shouldn't block a start
            pass
        threading.Thread(target=self._batch_run_worker, args=(request_id,), daemon=True).start()

    @Slot(str, int)
    def batchScheduleStartAsync(self, request_id, delay_seconds):
        """Scheduled/delayed start -- e.g. 'run overnight'. Fires the
        same worker as batchStartAsync after `delay_seconds`."""
        timer = threading.Timer(max(0, delay_seconds), self._batch_run_worker, args=(request_id,))
        timer.daemon = True
        timer.start()

    @Slot()
    def batchPause(self):
        self._batch.pause()
        self.statusChanged.emit("Batch paused")

    @Slot()
    def batchResume(self):
        self._batch.resume()
        self.statusChanged.emit("Batch resumed")

    @Slot()
    def batchCancel(self):
        self._batch.cancel()
        self.statusChanged.emit("Batch cancelling...")

    @Slot(str, result=str)
    def batchSkipItem(self, item_id):
        ok = self._batch.skip_item(item_id)
        return json.dumps({"ok": ok})

    @Slot(str, result=str)
    def batchRetryItem(self, item_id):
        ok = self._batch.retry_item(item_id)
        return json.dumps({"ok": ok, **self._batch.state()})

    @Slot(result=str)
    def batchRetryFailed(self):
        count = self._batch.retry_failed()
        return json.dumps({"ok": True, "count": count, **self._batch.state()})

    @Slot(result=str)
    def batchRequeueAll(self):
        """Reprocess All -- puts every finished item back into the queue
        (see core/batch_processor.py::requeue_all) so changing the Look/
        options and running again doesn't require removing and re-adding
        every image."""
        try:
            count = self._batch.requeue_all()
            return json.dumps({"ok": True, "count": count, **self._batch.state()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="batchRequeueAll"))

    @Slot(str, result=str)
    def batchCommitPreviewed(self, item_ids_json):
        """
        "Preview before commit" -- finalizes staged dry-run results (see
        core/batch_processor.py::commit_previewed). Pass "" / "null" to
        commit every previewed item, or a JSON array of ids to commit
        only a chosen subset.
        """
        try:
            item_ids = json.loads(item_ids_json) if item_ids_json and item_ids_json != "null" else None
            result = self._batch.commit_previewed(item_ids)
            return json.dumps({"ok": True, **result, **self._batch.state()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="batchCommitPreviewed"))

    @Slot(str, result=str)
    def batchDiscardPreviewed(self, item_ids_json):
        """Discards staged dry-run results and returns those items to the
        queue (see core/batch_processor.py::discard_previewed)."""
        try:
            item_ids = json.loads(item_ids_json) if item_ids_json and item_ids_json != "null" else None
            result = self._batch.discard_previewed(item_ids)
            return json.dumps({"ok": True, **result, **self._batch.state()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="batchDiscardPreviewed"))

    @Slot(result=str)
    def batchExportLogDialog(self):
        """Batch processing log (exportable) -- native Save As, then
        writes the per-image log + summary as JSON."""
        try:
            default_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            default_path = str(Path(default_dir) / "pixelforge_batch_log.json")
            dest_path, _ = QFileDialog.getSaveFileName(
                self._main_window, "Export Batch Log", default_path, "JSON (*.json)"
            )
            if not dest_path:
                return json.dumps({"ok": False, "error": "Export cancelled."})
            saved_path = self._batch.export_log(dest_path)
            return json.dumps({"ok": True, "path": saved_path})
        except Exception as exc:  # noqa: BLE001
            log_exception(exc, context="batchExportLogDialog")
            return json.dumps({"ok": False, "error": f"Couldn't export log: {exc}"})

    @Slot(str, str)
    def _show_system_notification(self, title, message):
        """
        Completion notification (system notification/sound). Best-effort:
        uses Qt's system tray if the platform/desktop supports it; quietly
        does nothing otherwise (e.g. no tray available) rather than
        failing the batch over a non-essential nicety.

        IMPORTANT: this is a @Slot connected to self._notifyRequested
        (a Signal), not called directly from _batch_run_worker -- that
        worker runs on a background thread, and QWidget/QObject children
        (QSystemTrayIcon parented to self._main_window) can only be
        created on the thread that owns their parent (the GUI thread).
        Calling this directly from the worker thread produced exactly
        the "QObject: Cannot create children for a parent that is in a
        different thread" warning. Emitting a Signal instead and letting
        Qt's queued cross-thread connection deliver it to this Slot
        (which runs on the GUI thread, since that's where this Bridge
        object lives) is the correct fix.
        """
        try:
            from PySide6.QtGui import QIcon
            from PySide6.QtWidgets import QApplication, QStyle, QSystemTrayIcon

            if not QSystemTrayIcon.isSystemTrayAvailable():
                return
            tray = getattr(self, "_tray_icon", None)
            if tray is None:
                # QSystemTrayIcon logs "No Icon set" and silently no-ops
                # showMessage() when given an empty QIcon() -- the app
                # has no dedicated tray/app icon file yet, so fall back
                # to a built-in Qt standard icon rather than adding a
                # new asset just for this.
                app = QApplication.instance()
                icon = (
                    app.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
                    if app else QIcon()
                )
                tray = QSystemTrayIcon(icon, self._main_window)
                self._tray_icon = tray
            tray.show()
            tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 5000)
        except Exception:  # noqa: BLE001
            pass

    # ===================== PHASE 9: AI UPSCALING =====================
    #
    # 🤖 Genuinely AI-powered -- see ai/upscaler.py's header for the full
    # model-choice/design rationale (Real-ESRGAN via ONNX Runtime, tiled
    # CPU-safe inference, GPU/CPU auto-fallback). This section stays a
    # thin Slot wrapper around that module, same layering as every other
    # phase (Remove BG -> ai/bg_remover.py, Analyzer -> core/analyzer.py
    # + ai/face_detector.py, etc).
    #
    # Two heavy operations (preview + export), both threaded via
    # _run_async/taskResult for the same reason as Remove BG/Object
    # Erase/Filters above: full-resolution tiled inference easily takes
    # several seconds to tens of seconds on the CPU-only target
    # hardware, and must never block the GUI thread.

    def _upscale_progress(self, percent, label):
        # Reuses the app's single global status bar, same convention as
        # _generate_image_progress above.
        self.progressChanged.emit(percent, label)
        if percent < 0 or percent >= 100:
            self.statusChanged.emit(label)

    def _parse_upscale_options(self, options_json):
        opts = json.loads(options_json) if options_json else {}
        return {
            "scale": opts.get("scale", "4x"),
            "custom_width": opts.get("custom_width") or None,
            "custom_height": opts.get("custom_height") or None,
            "preserve_aspect": opts.get("preserve_aspect", True),
            "denoise_strength": float(opts.get("denoise_strength", 0) or 0),
            "artifact_reduction": float(opts.get("artifact_reduction", 0) or 0),
            "face_aware": bool(opts.get("face_aware", False)),
        }

    @Slot(result=str)
    def getUpscaleModelStatus(self):
        """
        Reports whether the Real-ESRGAN weights are cached yet and which
        onnxruntime execution provider (GPU/CPU) would run inference --
        "Model download progress + caching" + honest CPU-warning
        requirements. The frontend shows this before the user ever clicks
        Upscale, rather than only discovering a ~65MB first-run download
        after the fact.

        Returns JSON: {"ok": true, ...ai.upscaler.model_status() fields}
        """
        try:
            return json.dumps({"ok": True, **upscaler_model_status()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="getUpscaleModelStatus"))

    @Slot(result=str)
    def clearUpscaleModelCache(self):
        """Settings > Cache > Clear -- deletes the downloaded weights from disk."""
        try:
            return json.dumps(upscaler_clear_model_cache())
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="clearUpscaleModelCache"))

    @Slot(str, str, result=str)
    def upscaleSafetyCheck(self, source_path, options_json):
        """
        Runs ai/upscaler.py::safety_check BEFORE any pixel work starts --
        megapixel cap / RAM headroom / disk space -- so the UI can warn
        (or block) before the user waits through a whole tiled pass only
        to hit the cap at the end. Fast, not threaded (no model load).

        Returns JSON: {"ok": true, "warnings": [...], "blocking": [...],
        "estimated_output": {"width", "height", "megapixels"}}
        """
        try:
            options = self._parse_upscale_options(options_json)
            with Image.open(source_path) as img:
                width, height = img.width, img.height
            result = upscaler_safety_check(
                width, height, options["scale"],
                output_folder=get_setting("output_directory", ""),
                custom_width=options["custom_width"],
                custom_height=options["custom_height"],
            )
            return json.dumps({"ok": True, **result})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="upscaleSafetyCheck"))

    def _do_upscale(self, source_path, options, downscale_source_to=None):
        """
        Shared worker body for preview/export below. downscale_source_to
        (int px, longest side) shrinks the SOURCE before the AI pass --
        used only by the preview path, so a live preview on a huge photo
        doesn't run a full tiled 4x pass just to show a small thumbnail.
        """
        if not source_path:
            raise UpscaleError("No image loaded.")

        self._upscale_cancel_event.clear()

        with Image.open(source_path) as img:
            source = img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB")
            exif_bytes = img.info.get("exif")
            icc_profile = img.info.get("icc_profile")

        if downscale_source_to:
            source = source.copy()
            source.thumbnail((downscale_source_to, downscale_source_to), Image.LANCZOS)

        # Phase 14 checklist -- "Processing timeout": core/resource_guard.py's
        # run_with_timeout() existed but wasn't called by any AI module.
        # Wired in here as a genuine safety net against an indefinite
        # hang (e.g. a stalled model download on first use), NOT as a
        # tight bound on legitimate slow CPU-only tiled 4x upscaling --
        # the preview path (small, downscaled source) gets a short
        # ceiling; the full-resolution export path gets a generous one.
        # cancel_event is also set on timeout so upscale_image's own
        # cooperative cancellation gets a chance to actually stop the
        # background thread, on top of run_with_timeout's own
        # documented "can't force-kill a thread" limitation.
        timeout_seconds = 120 if downscale_source_to else 1800

        def _run():
            return upscale_image(
                source,
                scale=options["scale"],
                custom_width=options["custom_width"],
                custom_height=options["custom_height"],
                preserve_aspect=options["preserve_aspect"],
                denoise_strength=options["denoise_strength"],
                artifact_reduction=options["artifact_reduction"],
                face_aware=options["face_aware"],
                on_progress=self._upscale_progress,
                cancel_event=self._upscale_cancel_event,
            )

        try:
            result = run_with_timeout(_run, timeout_seconds)
        except ProcessingTimeout:
            self._upscale_cancel_event.set()
            raise UpscaleError(
                "AI upscaling is taking much longer than expected and was "
                "stopped. This can happen on a slow/first-run model "
                "download -- try again once it's cached, or use a smaller "
                "scale/custom size."
            )
        return result, exif_bytes, icc_profile

    @Slot(str, str, str)
    def upscalePreviewAsync(self, request_id, source_path, options_json):
        """
        Live preview: downsizes the SOURCE to at most 500px (longest
        side) before running the AI pass, so even a 4x request previews
        in a reasonable time on CPU -- the full-resolution pass only
        happens on Export/Apply. Renders to a temp JPG, same convention
        as every other view's preview path (previewFilter, etc).

        Emits taskResult(request_id, json) with:
        {"ok": true, "path": "...", "width": ..., "height": ...,
         "label": "AI Upscaled ..."} or {"ok": false, "error": "...",
         "cancelled": true|false}
        """

        def work():
            options = self._parse_upscale_options(options_json)
            try:
                result, _exif, _icc = self._do_upscale(source_path, options, downscale_source_to=500)
            except UpscaleCancelled:
                return {"ok": False, "error": "Cancelled.", "cancelled": True}
            except UpscaleError as exc:
                return {"ok": False, "error": str(exc)}

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            preview_path = cache_dir / f"upscale_{int(time.time() * 1000)}.jpg"
            save_img = result.convert("RGB") if result.mode == "RGBA" else result
            save_img.save(preview_path, quality=90)

            return {
                "ok": True,
                "path": str(preview_path),
                "width": result.width,
                "height": result.height,
                "label": upscaler_output_label(result.width, result.height),
            }

        self._run_async(request_id, work)

    @Slot(str, str, str, str)
    def exportUpscaleImageAsync(self, request_id, source_path, dest_path, options_json):
        """
        Full-resolution AI Upscale export -- tiled Real-ESRGAN pass at
        the requested scale/options, saved to dest_path. Same
        non-destructive rule as every other export in this app:
        dest_path always comes from a Save As dialog (or a cache-folder
        temp path when called from "Apply", see below), and this never
        overwrites the source.

        EXIF + ICC color profile preserve ("Missing-feature #5" pattern,
        same as exportImageAsync/exportFilterImageAsync -- PIL carries
        icc_profile natively via save(icc_profile=...), no extra
        dependency needed for that half; EXIF re-attach reuses the same
        try/except-fallback shape those two Slots already use).

        Emits taskResult(request_id, json) with:
        {"ok": true, "path": "...", "width": ..., "height": ...,
         "label": "AI Upscaled ..."} or {"ok": false, "error": "...",
         "cancelled": true|false}
        """

        def work():
            if not dest_path:
                return {"ok": False, "error": "Export cancelled."}

            options = self._parse_upscale_options(options_json)
            try:
                result, exif_bytes, icc_profile = self._do_upscale(source_path, options)
            except UpscaleCancelled:
                return {"ok": False, "error": "Cancelled.", "cancelled": True}
            except UpscaleError as exc:
                return {"ok": False, "error": str(exc)}

            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = {}
            save_img = result
            if dest.suffix.lower() in (".jpg", ".jpeg"):
                if save_img.mode == "RGBA":
                    save_img = save_img.convert("RGB")
                save_kwargs["quality"] = 95
            elif dest.suffix.lower() == ".webp":
                save_kwargs["quality"] = 95
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile

            # Prefer the session's already-captured original EXIF (the
            # same source exportImageAsync/exportFilterImageAsync use)
            # so upscaling something several edits deep still carries
            # the ORIGINAL camera EXIF, not None just because this
            # particular file passed through PNG intermediates that
            # dropped it. Falls back to whatever this source file itself
            # had (e.g. upscaling a fresh import with no active session).
            session_exif = self._session.get_original_exif() if self._session else None
            final_exif = session_exif or exif_bytes
            if final_exif:
                try:
                    save_img.save(dest, exif=final_exif, **save_kwargs)
                except Exception:  # noqa: BLE001
                    save_img.save(dest, **save_kwargs)
            else:
                save_img.save(dest, **save_kwargs)

            self._record_edit(source_path, str(dest), "upscale",
                               upscaler_output_label(result.width, result.height),
                               options)

            return {
                "ok": True,
                "path": str(dest),
                "width": result.width,
                "height": result.height,
                "label": upscaler_output_label(result.width, result.height),
            }

        self._run_async(request_id, work)

    @Slot()
    def cancelUpscale(self):
        """Cancel button -- ai/upscaler.py checks this Event between tiles/
        during model download, so this actually stops a running pass
        instead of just hiding the progress bar client-side."""
        self._upscale_cancel_event.set()
        self.statusChanged.emit("Cancelling upscale...")

    @Slot(str, str, str)
    def sessionApplyUpscaleAsync(self, request_id, options_json, label):
        """
        Upscale's "Apply" button -- same shape as sessionApplyAdjustmentsAsync
        / sessionApplyPresetAsync above: runs the AI pass on the CURRENT
        session working image (not the original, and not a re-fetch of
        source_path -- so upscaling several edits deep upscales what's
        actually on screen) at full resolution, and commits the result
        as the new working image. This is the "Session/Undo integration
        ... so Ctrl+Z undoes an upscale too" requirement -- reuses
        core/session.py::commit_image exactly like Enhance/Filters,
        rather than a separate temp-file-then-commitFile round trip.

        Emits taskResult(request_id, json) with core/session.py's
        state() shape, or {"ok": false, "error": "...", "cancelled": bool}.
        """

        def work():
            options = self._parse_upscale_options(options_json)
            try:
                self._session.require_active()
                self._upscale_cancel_event.clear()
                source = self._session.open_working()
                result = upscale_image(
                    source,
                    scale=options["scale"],
                    custom_width=options["custom_width"],
                    custom_height=options["custom_height"],
                    preserve_aspect=options["preserve_aspect"],
                    denoise_strength=options["denoise_strength"],
                    artifact_reduction=options["artifact_reduction"],
                    face_aware=options["face_aware"],
                    on_progress=self._upscale_progress,
                    cancel_event=self._upscale_cancel_event,
                )
            except UpscaleCancelled:
                return {"ok": False, "error": "Cancelled.", "cancelled": True}
            except UpscaleError as exc:
                return {"ok": False, "error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                friendly = log_and_convert(exc, context="upscaleImage")
                friendly["has_session"] = bool(self._session.history)
                return friendly

            pretty = label or upscaler_output_label(result.width, result.height)
            return self._session.commit_image(
                result, "upscale", pretty,
                settings=options,
                op={"kind": "upscale", **options},
            )

        self._run_async(request_id, work)

    # ===================== PHASE 10: FACE RESTORATION =====================
    #
    # 🤖 Genuinely AI-powered -- see ai/face_restorer.py's header for the
    # full model-choice/design rationale (GFPGAN via ONNX Runtime,
    # crop-and-restore per detected face, feathered seamless blend,
    # classical strength/Natural<->Detailed/skin-protection post-passes
    # on top of the AI output). This section stays a thin Slot wrapper
    # around that module, same layering as every other phase -- and the
    # same shape as the "PHASE 9: AI UPSCALING" section directly above
    # (detect -> preview -> export -> cancel -> session-apply).
    #
    # Three heavy-ish operations (detect / preview / export), all
    # threaded via _run_async/taskResult for the same reason as every
    # other AI pass in this app: even a fast Haar Cascade detection pass
    # on a full-res photo, and definitely a GFPGAN forward pass per
    # face, must never block the GUI thread.

    def _facerestore_progress(self, percent, label):
        # Reuses the app's single global status bar, same convention as
        # _upscale_progress above.
        self.progressChanged.emit(percent, label)
        if percent < 0 or percent >= 100:
            self.statusChanged.emit(label)

    def _parse_facerestore_options(self, options_json):
        opts = json.loads(options_json) if options_json else {}
        return {
            "strength": float(opts.get("strength", 80) or 0),
            "natural_detailed": float(opts.get("natural_detailed", 50) or 50),
            "skin_protection": float(opts.get("skin_protection", 60) or 0),
        }

    @Slot(result=str)
    def getFaceRestoreModelStatus(self):
        """
        Reports whether the GFPGAN weights are cached yet and which
        onnxruntime execution provider (GPU/CPU) would run inference --
        same "show the ~340MB first-run download honestly, before the
        user ever clicks Restore" requirement as Phase 9's
        getUpscaleModelStatus.

        Returns JSON: {"ok": true, ...ai.face_restorer.model_status() fields}
        """
        try:
            return json.dumps({"ok": True, **facerestore_model_status()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="getFaceRestoreModelStatus"))

    @Slot(result=str)
    def clearFaceRestoreModelCache(self):
        """Settings > Cache > Clear -- deletes the downloaded weights from disk."""
        try:
            return json.dumps(facerestore_clear_model_cache())
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="clearFaceRestoreModelCache"))

    @Slot(str, str)
    def detectFacesAsync(self, request_id, source_path):
        """
        "Automatic face detection" -- reuses ai/face_detector.py (via
        ai/face_restorer.py::detect_faces_for_restore) to find faces in
        the full-resolution source image, so the frontend can render
        the "Detected Faces: N / ☑ Face 1 ☑ Face 2 ..." checklist the
        spec's example UI shape calls for. Threaded: Haar Cascade on a
        large photo is fast but not instant, and this must never block
        the GUI thread opening the view.

        Emits taskResult(request_id, json) with:
        {"ok": true, "faces": [{"id", "x", "y", "w", "h"}, ...]} or
        {"ok": false, "error": "..."}
        """

        def work():
            if not source_path:
                return {"ok": False, "error": "No image loaded."}
            try:
                with Image.open(source_path) as img:
                    rgb = img.convert("RGB")
                    faces = detect_faces_for_restore(rgb)
                return {"ok": True, "faces": faces}
            except Exception as exc:  # noqa: BLE001
                return log_and_convert(exc, context="detectFacesAsync")

        self._run_async(request_id, work)

    def _do_facerestore(self, source_path, faces, options, downscale_source_to=None):
        """
        Shared worker body for preview/export below -- same shape as
        Phase 9's _do_upscale. downscale_source_to (int px, longest
        side) shrinks the SOURCE before restoring -- used only by the
        preview path so a live preview doesn't re-run full-resolution
        GFPGAN passes on every slider tweak. Face boxes are fractional
        (0-1), so they stay correct against the downscaled preview
        source without any extra coordinate math.
        """
        if not source_path:
            raise FaceRestoreError("No image loaded.")

        self._facerestore_cancel_event.clear()

        with Image.open(source_path) as img:
            source = img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB")
            exif_bytes = img.info.get("exif")
            icc_profile = img.info.get("icc_profile")

        if downscale_source_to:
            source = source.copy()
            source.thumbnail((downscale_source_to, downscale_source_to), Image.LANCZOS)

        # Phase 14 checklist -- "Processing timeout", same reasoning and
        # pattern as Phase 9's _do_upscale above: a generous safety net
        # against an indefinite hang (e.g. a stalled GFPGAN model
        # download), not a tight bound on legitimate per-face CPU work.
        timeout_seconds = 90 if downscale_source_to else 900

        def _run():
            return restore_faces(
                source,
                faces=faces,
                strength=options["strength"],
                natural_detailed=options["natural_detailed"],
                skin_protection=options["skin_protection"],
                on_progress=self._facerestore_progress,
                cancel_event=self._facerestore_cancel_event,
            )

        try:
            result = run_with_timeout(_run, timeout_seconds)
        except ProcessingTimeout:
            self._facerestore_cancel_event.set()
            raise FaceRestoreError(
                "Face restoration is taking much longer than expected and "
                "was stopped. This can happen on a slow/first-run model "
                "download -- try again once it's cached, or restore fewer "
                "faces at once."
            )
        return result, exif_bytes, icc_profile

    @Slot(str, str, str, str)
    def restorePreviewAsync(self, request_id, source_path, faces_json, options_json):
        """
        Live preview: downsizes the SOURCE to at most 700px (longest
        side) before restoring -- large enough that a face crop still
        has real detail to work with (unlike Upscale's 500px preview
        cap, a face here is only a fraction of the frame), small enough
        that a CPU-only GFPGAN pass stays responsive. Renders to a temp
        JPG, same convention as every other view's preview path.

        faces_json: JSON array from the frontend's checklist --
        [{"id","x","y","w","h","selected","strength"?}, ...] (as
        returned by detectFacesAsync, with "selected"/"strength" added
        by the UI).

        Emits taskResult(request_id, json) with:
        {"ok": true, "path": "...", "width": ..., "height": ...,
         "label": "Face Restored (N faces)", "restored_count": N} or
        {"ok": false, "error": "...", "cancelled": true|false}
        """

        def work():
            options = self._parse_facerestore_options(options_json)
            faces = json.loads(faces_json) if faces_json else []
            restored_count = len([f for f in faces if f.get("selected", True)])
            try:
                result, _exif, _icc = self._do_facerestore(source_path, faces, options, downscale_source_to=700)
            except FaceRestoreCancelled:
                return {"ok": False, "error": "Cancelled.", "cancelled": True}
            except FaceRestoreError as exc:
                return {"ok": False, "error": str(exc)}

            cache_dir = Path(tempfile.gettempdir()) / "pixelforge_preview"
            cache_dir.mkdir(parents=True, exist_ok=True)
            preview_path = cache_dir / f"facerestore_{int(time.time() * 1000)}.jpg"
            save_img = result.convert("RGB") if result.mode == "RGBA" else result
            save_img.save(preview_path, quality=90)

            return {
                "ok": True,
                "path": str(preview_path),
                "width": result.width,
                "height": result.height,
                "label": facerestore_output_label(restored_count),
                "restored_count": restored_count,
            }

        self._run_async(request_id, work)

    @Slot(str, str, str, str, str)
    def exportFaceRestoreImageAsync(self, request_id, source_path, dest_path, faces_json, options_json):
        """
        Full-resolution Face Restoration export -- GFPGAN pass on every
        selected face at full source resolution, saved to dest_path.
        Same non-destructive rule as every other export in this app:
        dest_path always comes from a Save As dialog (or a cache-folder
        temp path when called from "Apply", see below).

        EXIF + ICC color profile preserve -- same "Missing-feature #5"
        pattern as exportUpscaleImageAsync.

        Emits taskResult(request_id, json) with:
        {"ok": true, "path": "...", "width": ..., "height": ...,
         "label": "Face Restored (N faces)", "restored_count": N} or
        {"ok": false, "error": "...", "cancelled": true|false}
        """

        def work():
            if not dest_path:
                return {"ok": False, "error": "Export cancelled."}

            options = self._parse_facerestore_options(options_json)
            faces = json.loads(faces_json) if faces_json else []
            restored_count = len([f for f in faces if f.get("selected", True)])
            try:
                result, exif_bytes, icc_profile = self._do_facerestore(source_path, faces, options)
            except FaceRestoreCancelled:
                return {"ok": False, "error": "Cancelled.", "cancelled": True}
            except FaceRestoreError as exc:
                return {"ok": False, "error": str(exc)}

            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = {}
            save_img = result
            if dest.suffix.lower() in (".jpg", ".jpeg"):
                if save_img.mode == "RGBA":
                    save_img = save_img.convert("RGB")
                save_kwargs["quality"] = 95
            elif dest.suffix.lower() == ".webp":
                save_kwargs["quality"] = 95
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile

            # Same "prefer the session's original EXIF" reasoning as
            # exportUpscaleImageAsync -- restoring a face several edits
            # deep should still carry the ORIGINAL camera EXIF.
            session_exif = self._session.get_original_exif() if self._session else None
            final_exif = session_exif or exif_bytes
            if final_exif:
                try:
                    save_img.save(dest, exif=final_exif, **save_kwargs)
                except Exception:  # noqa: BLE001
                    save_img.save(dest, **save_kwargs)
            else:
                save_img.save(dest, **save_kwargs)

            self._record_edit(source_path, str(dest), "restore",
                               facerestore_output_label(restored_count),
                               {**options, "restored_count": restored_count})

            return {
                "ok": True,
                "path": str(dest),
                "width": result.width,
                "height": result.height,
                "label": facerestore_output_label(restored_count),
                "restored_count": restored_count,
            }

        self._run_async(request_id, work)

    @Slot()
    def cancelFaceRestore(self):
        """Cancel button -- ai/face_restorer.py checks this Event between
        faces/during model download, so this actually stops a running
        pass instead of just hiding the progress bar client-side."""
        self._facerestore_cancel_event.set()
        self.statusChanged.emit("Cancelling face restoration...")

    @Slot(str, str, str, str)
    def sessionApplyFaceRestoreAsync(self, request_id, faces_json, options_json, label):
        """
        Face Restoration's "Apply" button -- same shape as Phase 9's
        sessionApplyUpscaleAsync: runs on the CURRENT session working
        image (not the original), and commits the result as the new
        working image, so Ctrl+Z undoes a face restoration too.

        Emits taskResult(request_id, json) with core/session.py's
        state() shape, or {"ok": false, "error": "...", "cancelled": bool}.
        """

        def work():
            options = self._parse_facerestore_options(options_json)
            faces = json.loads(faces_json) if faces_json else []
            restored_count = len([f for f in faces if f.get("selected", True)])
            try:
                self._session.require_active()
                self._facerestore_cancel_event.clear()
                source = self._session.open_working()
                result = restore_faces(
                    source,
                    faces=faces,
                    strength=options["strength"],
                    natural_detailed=options["natural_detailed"],
                    skin_protection=options["skin_protection"],
                    on_progress=self._facerestore_progress,
                    cancel_event=self._facerestore_cancel_event,
                )
            except FaceRestoreCancelled:
                return {"ok": False, "error": "Cancelled.", "cancelled": True}
            except FaceRestoreError as exc:
                return {"ok": False, "error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                friendly = log_and_convert(exc, context="restoreFaces")
                friendly["has_session"] = bool(self._session.history)
                return friendly

            pretty = label or facerestore_output_label(restored_count)
            return self._session.commit_image(
                result, "facerestore", pretty,
                settings=options,
                op={"kind": "facerestore", "faces": faces, **options},
            )

        self._run_async(request_id, work)

    # ===================== PHASE 11: VIDEO STUDIO =====================
    #
    # 🧮 No AI of its own -- pure FFmpeg assembly (see core/
    # video_studio.py's header for the full filter-graph design: motion
    # presets via zoompan, transitions via chained xfade, smart fit/
    # fill with blurred/black letterbox background, background music
    # mix, drawtext title overlays). One shared VideoProject
    # (self._video) holds the whole timeline/settings/undo-redo, same
    # "one shared controller" convention as self._session (Phase 6) and
    # self._batch (Phase 8) above.
    #
    # Every state-mutating call below returns the controller's full
    # state() dict (same shape sessionOpen/sessionUndo/etc. already use)
    # so the frontend can just re-render off the result instead of a
    # separate follow-up fetch. Export/Preview are threaded via
    # _run_async/taskResult -- real H.264 encoding is unavoidably heavy,
    # same class of operation as AI Upscale/Face Restoration above.

    def _video_progress(self, percent, label):
        self.progressChanged.emit(percent, label)
        if percent <= 0 or percent >= 100:
            self.statusChanged.emit(label)

    @Slot(result=str)
    def getFfmpegStatus(self):
        """Settings > Video > FFmpeg badge, and the Video Studio view's
        own startup check before letting the user try to export."""
        try:
            return json.dumps({"ok": True, **video_ffmpeg_status()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="getFfmpegStatus"))

    @Slot(result=str)
    def videoState(self):
        try:
            return json.dumps(self._video.state())
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoState"))

    @Slot(result=str)
    def videoNewProject(self):
        """"New Project" -- clears the timeline, settings, audio and
        text overlays back to defaults (with its own undo history
        reset, since resuming an old undo stack across an unrelated
        new project would be confusing, not helpful)."""
        try:
            return json.dumps(self._video.reset())
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoNewProject"))

    @Slot(str, result=str)
    def videoAddImages(self, paths_json):
        """"Input: 2+ images" -- appends to the timeline. Reuses the
        same file-picker (openImagesDialog) every other multi-file view
        already has; the frontend collects the chosen paths and hands
        them here in one call."""
        try:
            paths = json.loads(paths_json) if paths_json else []
            return json.dumps(self._video.add_images(paths))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoAddImages"))

    @Slot(str, result=str)
    def videoRemoveClip(self, clip_id):
        try:
            return json.dumps(self._video.remove_clip(clip_id))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoRemoveClip"))

    @Slot(str, result=str)
    def videoReorderClips(self, clip_ids_json):
        """"Drag to reorder" -- the frontend's timeline strip sends the
        full new ordering of clip ids after a drop, same "send the
        final order, not individual moves" shape as
        ui/bridge.py::batchReorder in Phase 8."""
        try:
            clip_ids = json.loads(clip_ids_json) if clip_ids_json else []
            return json.dumps(self._video.reorder(clip_ids))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoReorderClips"))

    @Slot(str, str, result=str)
    def videoSetClipSettings(self, clip_id, settings_json):
        """"Per-image duration", "Per-image motion", "Transition
        duration", "Cover frame selection" -- one call covers all of
        these, patch-style (only the keys present are changed)."""
        try:
            patch = json.loads(settings_json) if settings_json else {}
            return json.dumps(self._video.set_clip_settings(clip_id, patch))
        except VideoStudioError as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoSetClipSettings"))

    @Slot(str, result=str)
    def videoSetProjectSettings(self, settings_json):
        """Resolutions / aspect ratios / frame rates / duration mode /
        smart fit-fill + background fill / export quality -- the
        project-wide (not per-clip) settings panel."""
        try:
            patch = json.loads(settings_json) if settings_json else {}
            return json.dumps(self._video.set_project_settings(patch))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoSetProjectSettings"))

    @Slot(result=str)
    def chooseVideoAudioDialog(self):
        """"Background Music / Audio" -- native picker for MP3/WAV."""
        path, _ = QFileDialog.getOpenFileName(
            self._main_window, "Choose Background Music", "",
            "Audio Files (*.mp3 *.wav);;All Files (*)",
        )
        return path or ""

    @Slot(str, int, float, float, str, result=str)
    def videoSetAudio(self, path, volume, fade_in, fade_out, duration_match):
        try:
            return json.dumps(self._video.set_audio(
                path or None, volume, fade_in, fade_out, duration_match or "none"))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoSetAudio"))

    @Slot(result=str)
    def videoClearAudio(self):
        try:
            return json.dumps(self._video.clear_audio())
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoClearAudio"))

    # ----- PHASE 12: second audio track (voice-over, limited support) -----

    @Slot(result=str)
    def chooseVideoAudio2Dialog(self):
        """"Second Audio Track (voice-over)" -- native picker for MP3/WAV."""
        path, _ = QFileDialog.getOpenFileName(
            self._main_window, "Choose Voice-Over / Second Audio Track", "",
            "Audio Files (*.mp3 *.wav);;All Files (*)",
        )
        return path or ""

    @Slot(str, int, result=str)
    def videoSetAudio2(self, path, volume):
        try:
            return json.dumps(self._video.set_audio2(path or None, volume))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoSetAudio2"))

    @Slot(result=str)
    def videoClearAudio2(self):
        try:
            return json.dumps(self._video.clear_audio2())
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoClearAudio2"))

    # ----- PHASE 12: watermark / logo -----

    @Slot(result=str)
    def chooseVideoWatermarkImageDialog(self):
        """"Watermark / Logo" (image mode) -- native picker for PNG/JPG,
        PNG strongly recommended so alpha transparency carries through."""
        path, _ = QFileDialog.getOpenFileName(
            self._main_window, "Choose Watermark / Logo Image", "",
            "Image Files (*.png *.jpg *.jpeg *.webp);;All Files (*)",
        )
        return path or ""

    @Slot(str, result=str)
    def videoSetWatermark(self, patch_json):
        try:
            patch = json.loads(patch_json) if patch_json else {}
            return json.dumps(self._video.set_watermark(patch))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoSetWatermark"))

    @Slot(result=str)
    def videoClearWatermark(self):
        try:
            return json.dumps(self._video.clear_watermark())
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoClearWatermark"))

    @Slot(str, result=str)
    def videoAddTextOverlay(self, overlay_json):
        """"Text / Title Overlay" -- font/size/position/duration/fade,
        add one entry to the project's overlay list."""
        try:
            overlay = json.loads(overlay_json) if overlay_json else {}
            return json.dumps(self._video.add_text_overlay(overlay))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoAddTextOverlay"))

    @Slot(str, str, result=str)
    def videoUpdateTextOverlay(self, overlay_id, patch_json):
        try:
            patch = json.loads(patch_json) if patch_json else {}
            return json.dumps(self._video.update_text_overlay(overlay_id, patch))
        except VideoStudioError as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoUpdateTextOverlay"))

    @Slot(str, result=str)
    def videoRemoveTextOverlay(self, overlay_id):
        try:
            return json.dumps(self._video.remove_text_overlay(overlay_id))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoRemoveTextOverlay"))

    # ----- PHASE 12: SRT subtitle import -----

    @Slot(result=str)
    def chooseSrtFileDialog(self):
        """"Import SRT" -- native picker for a standard subtitle file."""
        path, _ = QFileDialog.getOpenFileName(
            self._main_window, "Import Subtitles (SRT)", "",
            "SubRip Subtitles (*.srt);;All Files (*)",
        )
        return path or ""

    @Slot(str, result=str)
    def videoImportSrt(self, path):
        try:
            return json.dumps(self._video.import_srt(path))
        except VideoStudioError as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoImportSrt"))

    @Slot(result=str)
    def videoUndo(self):
        """"Undo / Redo" -- for timeline changes (reorder, duration,
        motion, transition edits). Whole-state snapshots -- see
        core/video_studio.py::VideoProject's class docstring for why
        that's the right tradeoff here (small JSON, not pixel data)."""
        try:
            return json.dumps(self._video.undo())
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoUndo"))

    @Slot(result=str)
    def videoRedo(self):
        try:
            return json.dumps(self._video.redo())
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoRedo"))

    # ----- Project Save / Load -----

    @Slot(result=str)
    def chooseVideoProjectSavePath(self):
        default_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        path, _ = QFileDialog.getSaveFileName(
            self._main_window, "Save Video Studio Project",
            os.path.join(default_dir, "Untitled" + VIDEO_PROJECT_EXTENSION),
            "PixelForge Video Project (*.pfvproj)",
        )
        return path or ""

    @Slot(result=str)
    def chooseVideoProjectOpenPath(self):
        default_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        path, _ = QFileDialog.getOpenFileName(
            self._main_window, "Open Video Studio Project", default_dir,
            "PixelForge Video Project (*.pfvproj);;All Files (*)",
        )
        return path or ""

    @Slot(str, result=str)
    def videoSaveProject(self, path):
        try:
            return json.dumps(self._video.save_project(path))
        except VideoStudioError as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoSaveProject"))

    @Slot(str, result=str)
    def videoLoadProject(self, path):
        try:
            return json.dumps(self._video.load_project(path))
        except VideoStudioError as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return json.dumps(log_and_convert(exc, context="videoLoadProject"))

    # ----- Export / Preview (threaded, cancellable) -----

    @Slot(result=str)
    def chooseVideoExportPath(self):
        default_dir = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation) or \
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        path, _ = QFileDialog.getSaveFileName(
            self._main_window, "Export Video", os.path.join(default_dir, "pixelforge_video.mp4"),
            "MP4 Video (*.mp4)",
        )
        return path or ""

    @Slot(str, str)
    def videoExportAsync(self, request_id, dest_path):
        """
        "Output: MP4" + "Export Quality/Bitrate Control" + "Export
        Progress + Cancel" -- the final render. Progress arrives through
        the shared progressChanged/statusChanged signals (same global
        status bar every other heavy operation uses); the final
        ok/error/cancelled result arrives through taskResult like every
        other _run_async call.
        """

        def work():
            try:
                self._video_cancel_event.clear()
                if not dest_path:
                    return {"ok": False, "error": "No export destination chosen."}
                result = self._video.export(
                    dest_path, on_progress=self._video_progress,
                    cancel_event=self._video_cancel_event,
                )
                if result and result.get("ok"):
                    self._record_video_export(dest_path, result.get("duration"))
                return result
            except VideoExportCancelled:
                return {"ok": False, "error": "Cancelled.", "cancelled": True}
            except VideoStudioError as exc:
                return {"ok": False, "error": str(exc)}

        self._run_async(request_id, work)

    @Slot(str, str)
    def videoGeneratePreviewAsync(self, request_id, container="webm"):
        """"Preview playback" -- renders the whole current timeline at
        a fast, reduced-resolution proxy so the user can scrub it in a
        <video> element before spending minutes on a full export.

        `container` ("webm" or "mp4") is chosen by video.js after
        probing this exact QWebEngineView's own
        `<video>.canPlayType(...)` -- see the BUGFIX note on
        core/video_studio.py::generate_preview() for why that's the
        Python side's responsibility to accept, not decide."""

        def work():
            try:
                self._video_cancel_event.clear()
                return self._video.generate_preview(
                    on_progress=self._video_progress,
                    cancel_event=self._video_cancel_event,
                    container=container if container in ("webm", "mp4") else "webm",
                )
            except VideoExportCancelled:
                return {"ok": False, "error": "Cancelled.", "cancelled": True}
            except VideoStudioError as exc:
                return {"ok": False, "error": str(exc)}

        self._run_async(request_id, work)

    @Slot()
    def cancelVideoExport(self):
        """Cancel button -- core/video_studio.py checks this Event
        between FFmpeg progress lines and actually terminates the
        running FFmpeg process, same cooperative-cancellation pattern
        as Phase 9/10's Cancel buttons."""
        self._video_cancel_event.set()
        self.statusChanged.emit("Cancelling export...")

    # ===================== SETTINGS =====================
    # PHASE 13 FIX: these two Slots were stubs (getSetting always
    # returned "", setSetting only printed) even though core/settings.py
    # (real config.json persistence) already existed and was already
    # imported/used elsewhere in this file (see get_setting() calls
    # above for history_limit / output_directory). Nothing ever wired
    # the frontend's actual getSetting/setSetting calls -- including
    # "tutorial_seen" -- through to it. Now they do.

    @Slot(str, result=str)
    def getSetting(self, key):
        return get_setting(key, "")

    @Slot(str, str)
    def setSetting(self, key, value):
        set_setting(key, value)

    @Slot(result=str)
    def getAllSettings(self):
        return json.dumps({"ok": True, "settings": get_all_settings()})

    @Slot()
    def resetAllSettings(self):
        reset_settings()

    # BUGFIX: Settings > Storage > "Clear cache" had no backend method
    # and no click handler at all -- clicking it did nothing. This
    # wipes the on-disk temp folders every preview/thumbnail/proxy
    # generator in the app writes to (see the "pixelforge_*" dirs
    # under tempfile.gettempdir() throughout this file, video_studio.py
    # and session.py). It deliberately leaves pixelforge_sessions
    # alone -- that's the active edit session's working copy, not
    # disposable cache, and clearing it mid-edit would lose unsaved
    # work.
    _CACHE_DIR_NAMES = (
        "pixelforge_preview",
        "pixelforge_working",
        "pixelforge_display_proxy",
        "pixelforge_video_thumbs",
        "pixelforge_video_preview",
    )

    @Slot(result=str)
    def clearCache(self):
        base = Path(tempfile.gettempdir())
        cleared = []
        errors = []
        freed_bytes = 0
        for name in self._CACHE_DIR_NAMES:
            d = base / name
            if not d.exists():
                continue
            try:
                for child in d.iterdir():
                    try:
                        if child.is_dir():
                            size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            size = child.stat().st_size
                            child.unlink()
                        freed_bytes += size
                    except Exception:  # noqa: BLE001 - best-effort per-file
                        continue
                cleared.append(name)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
        return json.dumps({
            "ok": not errors,
            "cleared": cleared,
            "errors": errors,
            "freed_bytes": freed_bytes,
        })

    # ===================== PHASE 13: PROJECTS / HISTORY (SQLite) =====================
    # database/database.py owns the schema + all SQL; every Slot here is
    # a thin pass-through that (a) JSON-encodes the result the same way
    # every other Slot in this file does, and (b) turns a raised
    # ValueError/sqlite3 error into the usual {"ok": False, "error": ...}
    # shape instead of letting it cross the Qt/JS boundary as an
    # exception. List/search queries run inline (they're fast, indexed,
    # local reads); anything that touches files (backup export/import)
    # runs through _run_async like the rest of the app's I/O-heavy Slots.

    def _db_result(self, fn):
        try:
            return json.dumps({"ok": True, "data": fn()})
        except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message in JS
            return json.dumps(log_and_convert(exc, context="_db_result"))

    # ---- PHASE 13 CONNECTOR helpers ----
    #
    # Every export Slot in this file (Enhance, Filters, Filter Stack,
    # Remove BG, Object Erase, AI Upscale, Face Restoration, Video
    # Studio, Batch) calls _record_edit()/_record_video_export() at the
    # very end of its own work(), success path only. These are strictly
    # additive on top of what each tool already does -- they never
    # change what gets written to the user's chosen dest_path, and any
    # database problem is swallowed rather than turning a real,
    # successful export into a reported failure.

    _TEMP_CACHE_DIR_NAMES = (
        "pixelforge_preview",
        "pixelforge_working",
        "pixelforge_display_proxy",
        "pixelforge_video_thumbs",
        "pixelforge_video_preview",
        "pixelforge_sessions",
    )

    def _stable_source_path(self, source_path):
        """BUGFIX (duplicate "container" project, part 3): every
        export Slot passes along whatever path the frontend currently
        considers "the working image", but several tools' own preview
        renderers (e.g. previewAdjust's live slider preview, saved as
        .../pixelforge_preview/preview_<timestamp>.jpg) write into the
        same temp cache dirs this class already knows about (see
        _CACHE_DIR_NAMES / clearCache above). If one of those short-
        lived, timestamp-named cache files ever ends up as the
        source_path an export is recorded against, _db_project_for_source
        has nothing stable to key or name a project on -- a new preview
        render next time means a new filename means a new project, and
        the project even ends up visibly named after a temp file (e.g.
        "preview_1788328724983"). Since a temp cache file can never be
        the real identity of "the photo being edited", resolve back to
        the active edit session's actual original file whenever
        source_path points into one of these cache dirs.
        """
        path = source_path or ""
        try:
            resolved = Path(path).resolve()
        except Exception:  # noqa: BLE001
            return path
        base = Path(tempfile.gettempdir()).resolve()
        for name in self._TEMP_CACHE_DIR_NAMES:
            try:
                if resolved.is_relative_to(base / name):
                    original = getattr(self._session, "original_path", "") if self._session else ""
                    return original or path
            except (OSError, ValueError):
                continue
        return path

    def _db_project_for_source(self, source_path):
        """One database project per distinct source file (see the
        comment on self._db_project_by_source in __init__ for why: each
        export you make should show up as its OWN saved thing, not get
        silently merged into whatever you exported earlier). Exporting
        the same source_path again reuses that file's existing project
        so its edit history accumulates in one place instead of
        spawning a duplicate project every time.

        BUGFIX (duplicate "container" project): this used to look
        *only* at self._db_project_by_source, keyed on the exact
        source_path string. "+ New Project" (createProject) and
        opening a project back up (setActiveProject) never populated
        that cache -- they only set the *active* project via
        db.set_active_project(). So the very first export after
        starting/opening a project always missed the cache and minted
        a second, brand-new project (visible in the Projects grid as
        an extra "Untitled Project"/auto-named entry next to the one
        the user actually started), instead of recording the edit
        against the project the user was already in. Now, if there's
        an active project and it isn't already claimed by a different
        source_path, this export is filed under it, and that mapping
        is cached going forward for this same file.
        """
        source_path = self._stable_source_path(source_path)
        cache_key = source_path or ""
        project_id = self._db_project_by_source.get(cache_key)
        if project_id and db.get_project(project_id):
            return project_id

        active = db.get_active_project()
        if active and active.get("id"):
            active_id = active["id"]
            already_claimed = any(
                pid == active_id and key != cache_key
                for key, pid in self._db_project_by_source.items()
            )
            if not already_claimed and db.get_project(active_id):
                self._db_project_by_source[cache_key] = active_id
                return active_id

        name = (Path(source_path).stem if source_path else "") or "Untitled Project"
        project = db.create_project(name)
        self._db_project_by_source[cache_key] = project["id"]
        db.set_active_project(project["id"])
        return project["id"]

    def _db_current_media(self, project_id, input_path, media_type="image"):
        """Finds or creates the media row for input_path under
        project_id. Checked against the project's existing media first
        (not just this instance's in-memory cache) so a project reopened
        via crash recovery still recognizes a file it already knows
        about instead of adding a duplicate row."""
        if not input_path:
            return None
        input_path = self._stable_source_path(input_path)
        cache_key = (project_id, input_path)
        cached = self._db_media_cache.get(cache_key)
        if cached:
            return cached
        try:
            for m in db.list_media(project_id):
                if m.get("input_path") == input_path:
                    self._db_media_cache[cache_key] = m["id"]
                    return m["id"]
        except Exception:  # noqa: BLE001
            pass
        width = height = None
        try:
            with Image.open(input_path) as probe:
                width, height = probe.size
        except Exception:  # noqa: BLE001
            pass
        media = db.add_media(project_id, input_path, media_type=media_type,
                              width=width, height=height)
        self._db_media_cache[cache_key] = media["id"]
        return media["id"]

    def _record_edit(self, source_path, dest_path, tool, label=None,
                      parameters=None, media_type="image"):
        """Records one export as a durable project/media/edit row --
        the thing that makes the Projects screen's Media tab and Edit
        History tab (frontend/projects.js) show real data instead of
        'No media added to this project yet.' Best-effort: wrapped so a
        database hiccup can never fail an export that already succeeded
        and already wrote the user's file to disk."""
        try:
            project_id = self._db_project_for_source(source_path or dest_path)
            media_id = self._db_current_media(project_id, source_path or dest_path, media_type)
            if not media_id:
                return
            db.update_media(media_id, output_path=dest_path)
            db.add_edit(media_id, preset=tool, prompt=label, parameters=parameters)
            if dest_path:
                db.update_project(project_id, thumbnail_path=dest_path)
        except Exception:  # noqa: BLE001 - never let bookkeeping fail a real export
            pass

    def _record_video_export(self, dest_path, duration=None):
        """Video Studio equivalent of _record_edit -- video_projects is
        its own table (core/database.py), separate from media/edits
        since a video export has no single 'source_path' the way an
        image export does (it's assembled from a whole timeline)."""
        try:
            project_id = self._db_project_for_source(dest_path)
            resolution = fps = None
            try:
                resolution = self._video.settings.get("resolution")
                fps = self._video.settings.get("fps")
            except Exception:  # noqa: BLE001
                pass
            db.add_video_project(project_id, duration=duration,
                                  resolution=resolution, fps=fps,
                                  output_path=dest_path)
        except Exception:  # noqa: BLE001 - never let bookkeeping fail a real export
            pass

    # ---- Projects ----

    @Slot(str, result=str)
    def createProject(self, name):
        return self._db_result(lambda: db.create_project(name))

    @Slot(str, result=str)
    def getProject(self, project_id):
        return self._db_result(lambda: db.get_project(project_id))

    @Slot(str, str, result=str)
    def renameProject(self, project_id, new_name):
        return self._db_result(lambda: db.update_project(project_id, name=new_name))

    @Slot(str, result=str)
    def duplicateProject(self, project_id):
        return self._db_result(lambda: db.duplicate_project(project_id))

    @Slot(str, result=str)
    def deleteProject(self, project_id):
        """Permanent delete -- frontend should confirm with the user
        first; trashProject() below is the reversible alternative."""
        return self._db_result(lambda: db.delete_project(project_id))

    @Slot(str, result=str)
    def trashProject(self, project_id):
        return self._db_result(lambda: db.trash_project(project_id))

    @Slot(str, result=str)
    def restoreProject(self, project_id):
        return self._db_result(lambda: db.restore_project(project_id))

    # BUGFIX: this was declared as @Slot(str, result=str) -- only ONE
    # string parameter -- even though the method itself takes TWO
    # (project_id, thumbnail_path). Qt's QWebChannel matches an
    # incoming JS call against the exact registered Slot signature, so
    # every call from the frontend (which correctly sends both
    # project_id and thumbnail_path) failed to match any candidate and
    # was silently dropped with "No candidates found for
    # setProjectThumbnail with 2 arguments" in the console -- the
    # thumbnail never got saved. This sat unnoticed because nothing in
    # the frontend called setProjectThumbnail before now.
    @Slot(str, str, result=str)
    def setProjectThumbnail(self, project_id, thumbnail_path):
        return self._db_result(
            lambda: db.update_project(project_id, thumbnail_path=thumbnail_path)
        )

    @Slot(str, str, str, str, int, result=str)
    def listProjects(self, status, search, sort_by, sort_dir, limit):
        return self._db_result(
            lambda: db.list_projects(
                status=status or None,
                search=search or None,
                sort_by=sort_by or "updated_at",
                sort_dir=sort_dir or "desc",
                limit=limit or None,
            )
        )

    @Slot(int, result=str)
    def recentProjects(self, limit):
        return self._db_result(lambda: db.recent_projects(limit=limit or 8))

    # ---- Media ----

    @Slot(str, str, str, str, int, int, result=str)
    def addMedia(self, project_id, input_path, output_path, media_type, width, height):
        return self._db_result(
            lambda: db.add_media(
                project_id, input_path, output_path or None,
                media_type or "image", width or None, height or None,
            )
        )

    @Slot(str, result=str)
    def listMedia(self, project_id):
        return self._db_result(lambda: db.list_media(project_id))

    @Slot(str, result=str)
    def deleteMedia(self, media_id):
        return self._db_result(lambda: db.delete_media(media_id))

    # ---- Edits (per-project/per-media history) ----

    @Slot(str, str, str, str, result=str)
    def addEdit(self, media_id, preset, prompt, parameters_json):
        params = json.loads(parameters_json) if parameters_json else None
        return self._db_result(
            lambda: db.add_edit(media_id, preset or None, prompt or None, params)
        )

    @Slot(str, result=str)
    def listEdits(self, media_id):
        return self._db_result(lambda: db.list_edits(media_id))

    @Slot(str, result=str)
    def projectEditHistory(self, project_id):
        return self._db_result(lambda: db.project_edit_history(project_id))

    # ---- Video projects ----

    @Slot(str, float, str, float, str, result=str)
    def addVideoProject(self, project_id, duration, resolution, fps, output_path):
        return self._db_result(
            lambda: db.add_video_project(
                project_id, duration or None, resolution or None,
                fps or None, output_path or None,
            )
        )

    @Slot(str, result=str)
    def listVideoProjects(self, project_id):
        return self._db_result(lambda: db.list_video_projects(project_id))

    # ---- Presets (cross-tool saved parameter sets) ----

    @Slot(str, str, str, result=str)
    def savePreset(self, name, category, parameters_json):
        params = json.loads(parameters_json) if parameters_json else None
        return self._db_result(lambda: db.save_preset(name, category or None, params))

    @Slot(str, result=str)
    def listPresets(self, category):
        return self._db_result(lambda: db.list_presets(category or None))

    @Slot(str, result=str)
    def deletePreset(self, preset_id):
        return self._db_result(lambda: db.delete_preset(preset_id))

    # ---- Backup / restore ----

    @Slot(str, str)
    def exportProjectBackup(self, request_id, project_id):
        """Opens a native Save dialog for a .pfbackup.json file, then
        writes the backup on a background thread (same _run_async
        pattern used for exportImage/exportFilterImage) since a large
        project's media history can make this a non-trivial write."""
        default_name = f"{project_id}.pfbackup.json"
        path, _ = QFileDialog.getSaveFileName(
            None, "Export Project Backup", default_name,
            "PixelForge Backup (*.pfbackup.json)",
        )
        if not path:
            self.taskResult.emit(request_id, json.dumps(
                {"ok": False, "error": "Cancelled."}))
            return

        def work():
            backup = db.export_project_backup(project_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(backup, f, indent=2)
            return {"ok": True, "path": path}

        self._run_async(request_id, work)

    @Slot(str)
    def importProjectBackup(self, request_id):
        """Opens a native Open dialog for a .pfbackup.json file and
        restores it as a brand-new project (see
        core/database.py::import_project_backup for why it's always
        'restore as new' rather than overwriting by id)."""
        path, _ = QFileDialog.getOpenFileName(
            None, "Restore Project Backup", "",
            "PixelForge Backup (*.pfbackup.json);;All Files (*)",
        )
        if not path:
            self.taskResult.emit(request_id, json.dumps(
                {"ok": False, "error": "Cancelled."}))
            return

        def work():
            with open(path, "r", encoding="utf-8") as f:
                backup = json.load(f)
            project = db.import_project_backup(backup)
            return {"ok": True, "project": project}

        self._run_async(request_id, work)

    # ---- Crash recovery ----

    @Slot(str)
    def setActiveProject(self, project_id):
        db.set_active_project(project_id)

    @Slot(result=str)
    def getActiveProject(self):
        return self._db_result(lambda: db.get_active_project())

    @Slot()
    def clearActiveProject(self):
        db.clear_active_project()