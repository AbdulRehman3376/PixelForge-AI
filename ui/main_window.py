# Main application window
# Responsibilities:
#   - Host the QWebEngineView that renders the HTML/CSS/JS frontend
#   - Wire up the Python <-> JavaScript bridge (QWebChannel)

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMainWindow

from ui.bridge import Bridge
from core import database as db
from core.logger import get_logger  # PHASE 14: log clean-shutdown for crash-recovery visibility

_log = get_logger(__name__)

# frontend/ lives one level up from ui/
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixelForge AI")
        self.resize(1280, 820)
        self.setMinimumSize(1000, 650)

        # ----- Web view (renders frontend/index.html) -----
        self.web_view = QWebEngineView()
        self.setCentralWidget(self.web_view)

        # BUGFIX (Phase 2): by default, QtWebEngine (Chromium) blocks a
        # file:// page from loading OTHER file:// resources outside its
        # own directory -- this is what made every opened photo show as
        # a broken image in the editor preview (the frontend loads
        # frontend/index.html, but the user's photo lives elsewhere on
        # disk, e.g. D:\Pictures\photo.jpg). Explicitly allow local
        # content to access other local file URLs.
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        # BUGFIX (Phase 2): the editor's Fullscreen button called the
        # browser Fullscreen API (canvasWrap.requestFullscreen()) from
        # JavaScript, but QtWebEngine blocks that by default -- both the
        # attribute AND an explicit accept from the Qt side are required,
        # otherwise the JS promise just silently rejects and nothing
        # happens.
        settings.setAttribute(QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True)
        self.web_view.page().fullScreenRequested.connect(self._handle_fullscreen_request)

        # ----- Python <-> JavaScript bridge -----
        # The Bridge object is exposed to JS as `pyBridge` (see frontend/js/bridge.js)
        self.bridge = Bridge(self)
        self.channel = QWebChannel()
        self.channel.registerObject("pyBridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        # ----- Load the frontend -----
        html_path = FRONTEND_DIR / "index.html"
        if not html_path.exists():
            raise FileNotFoundError(
                f"Frontend not found at {html_path}. "
                f"Expected frontend/index.html next to main.py."
            )
        self.web_view.load(QUrl.fromLocalFile(str(html_path)))

    def _handle_fullscreen_request(self, request):
        """
        Called when JS inside the web view calls element.requestFullscreen()
        (used by the editor's Fullscreen button on the image canvas).
        Must be explicitly accepted, or Chromium silently refuses the
        request. Since the QWebEngineView is docked inside QMainWindow
        (not a standalone window), we toggle the whole app window
        fullscreen -- CSS on the .editor-canvas-wrap:fullscreen selector
        then makes the canvas visually fill that space.
        """
        request.accept()
        if request.toggleOn():
            self.showFullScreen()
        else:
            self.showNormal()

    def closeEvent(self, event):
        """PHASE 13: clean-shutdown signal for crash recovery. If this
        never runs (crash, power loss, force-kill), the 'active_project'
        pointer set by Bridge.setActiveProject() is left behind in the
        database, and the frontend's startup check
        (getActiveProject) can offer to reopen it. A normal close
        clears that pointer so next launch doesn't ask unnecessarily,
        then closes the shared SQLite connection so WAL data is
        flushed to the main .db file rather than left in -wal/-shm
        sidecar files."""
        try:
            db.clear_active_project()
        finally:
            db.close()
            _log.info("Application closing cleanly (active_project cleared, DB connection closed).")
        super().closeEvent(event)