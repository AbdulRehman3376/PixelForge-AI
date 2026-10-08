# Application entry point
# Responsibilities: startup, logging setup (later phase), main window init

import os

# D: drive redirect -- MUST be first (models/cache/temp ko C: se hataata hai).
from core import paths as _pf_paths
_pf_paths.apply()

# ---------------------------------------------------------------------
# CONFIRMED root cause (seen directly in this machine's startup log):
# "Failed to create GLES3 context, fallback to GLES2" / "Failed to
# create shared context for virtualization" -- Chromium's GPU process
# can't get the specific 3D context it wants for accelerated VIDEO
# DECODE on this machine's Intel HD 620 iGPU (confirmed via DxDiag:
# driver itself is fine, WHQL-signed, "No problems found" -- this is a
# narrower Chromium/ANGLE-vs-this-iGPU quirk, not a driver issue).
#
# IMPORTANT: PixelForge's whole UI renders inside QtWebEngine, not just
# the video preview -- a blanket --disable-gpu (tried earlier) forces
# the ENTIRE app to software-render, which is why the whole UI felt
# sluggish, not just the video panel. The flags below are narrower:
# they turn off only hardware-accelerated video decode/encode (the
# actual broken piece) and leave GPU compositing on for everything
# else, so the rest of the UI stays at full speed.
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-accelerated-video-decode --disable-accelerated-video-encode",
)

import sys
from PySide6.QtWidgets import QApplication

# PHASE 14 -- Quality: logging system + global exception hook + startup
# health check.
#
# IMPORTANT (fixed after a real hang on Windows): the startup health
# check imports ai/upscaler.py + ai/face_restorer.py + rembg to read
# their model status, which pulls in onnxruntime -- a native library
# whose provider probing (DirectML on Windows) does its own internal
# threading/GPU-context setup. Running that BEFORE QApplication/
# QtWebEngine exist was observed to freeze the main window ("Not
# Responding", blank gray canvas) on at least one machine -- almost
# certainly a conflict between onnxruntime's native init and Chromium/
# ANGLE's own GPU context negotiation during QtWebEngine startup (this
# app's UI renders inside QtWebEngine -- see the GPU-flag workaround
# above, confirming this machine's GPU/Chromium interaction is already
# fragile). It also added a real, user-visible 10-15s delay before the
# window even appeared.
#
# Fix: QApplication + the window are created and shown FIRST, with zero
# heavy imports in the way. The health check then runs on a plain
# background thread (not the GUI thread) and only ever logs its result
# -- it never blocks window creation and never touches the GUI thread,
# so it can't interfere with QtWebEngine's own startup regardless of
# how slow onnxruntime's probing is on a given machine. The frontend
# can also re-run the same checks any time via bridge.py's
# getSystemHealth Slot (also backgrounded -- see ui/bridge.py).
import threading

from core.logger import setup_logging, get_logger
from core.error_handler import install_global_exception_hook

log = get_logger(__name__)


def _run_startup_health_check_in_background():
    try:
        from core.health_check import run_startup_health_check
        health = run_startup_health_check()
        if health["overall"] != "ok":
            log.warning("Startup health check found issues: %s", health["summary"])
    except Exception:  # noqa: BLE001 -- health check must never affect the app
        log.exception("Startup health check itself failed unexpectedly (continuing anyway).")


def main():
    setup_logging()
    install_global_exception_hook()
    log.info("PixelForge AI starting up...")

    app = QApplication(sys.argv)
    app.setApplicationName("PixelForge AI")
    app.setOrganizationName("PixelForge")

    # BUGFIX (black-screen hang at startup): this import used to sit at
    # the top of the file, which meant it ran BEFORE QApplication above
    # even existed. `ui.main_window` imports `ui.bridge`, which in turn
    # imports ai/upscaler.py + ai/face_restorer.py -- both of which do a
    # module-level `import onnxruntime`. onnxruntime's native provider
    # probing (DirectML on Windows) does its own GPU-context/threading
    # setup, and doing that before QApplication/QtWebEngine exist is the
    # exact same conflict the health-check comment above already
    # identified: it can freeze the window ("Not Responding", blank
    # black canvas) and adds a real multi-second delay before anything
    # even appears. Importing MainWindow here, after QApplication is
    # already constructed, avoids the conflict.
    from ui.main_window import MainWindow

    try:
        window = MainWindow()
    except Exception:
        log.exception("Fatal error creating the main window.")
        raise
    window.show()
    log.info("Main window shown -- entering event loop.")

    # Fire-and-forget, off the GUI thread -- see the long comment above.
    threading.Thread(
        target=_run_startup_health_check_in_background,
        name="pixelforge-startup-health-check",
        daemon=True,
    ).start()

    exit_code = app.exec()
    log.info("PixelForge AI exiting (code %s).", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()