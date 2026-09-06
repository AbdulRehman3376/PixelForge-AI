# core/logger.py
#
# PHASE 14 -- Logging System.
#
# WHY THIS EXISTS: before this, the app had no persistent record of
# what happened -- a crash, a failed AI model load, or a batch item
# that silently failed left nothing behind to debug from except
# whatever the user could describe from memory. This module gives
# every other module (core/, ai/, ui/) one shared, pre-configured
# logger that writes to a real file on disk (`logs/pixelforge.log`),
# rotates so it never grows unbounded, and never crashes the app if
# the log file itself can't be written (e.g. a read-only install
# folder) -- logging failures degrade to console-only, they never take
# the app down.
#
# USAGE (from any module):
#     from core.logger import get_logger
#     log = get_logger(__name__)
#     log.info("Something happened")
#     log.exception("Something failed")   # inside an except block
#
# DESIGN NOTES:
#   - One rotating file handler shared by the whole app (root logger
#     "pixelforge"), so every module's messages land in the same file
#     in the same chronological order -- much easier to read than N
#     separate log files.
#   - Rotates at 2MB x 5 backups (~10MB ceiling) so a runaway loop
#     logging every frame of a video export can't silently eat the
#     user's disk over weeks of use.
#   - setup_logging() is idempotent -- safe to call more than once
#     (e.g. once from main.py, and again defensively from a test) --
#     it only attaches handlers the first time.
#   - Console output is INFO+ by default so a developer running
#     `python main.py` from a terminal still sees activity, while the
#     file captures DEBUG+ for real post-mortem debugging.

import logging
import logging.handlers
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_PATH = _LOG_DIR / "pixelforge.log"

_ROOT_LOGGER_NAME = "pixelforge"
_configured = False


def setup_logging(level: int = logging.DEBUG, console_level: int = logging.INFO) -> logging.Logger:
    """
    Configures the shared "pixelforge" logger once per process. Safe to
    call multiple times (e.g. main.py at startup, and again in tests) --
    only the first call actually attaches handlers.

    Returns the configured root "pixelforge" Logger.
    """
    global _configured
    root = logging.getLogger(_ROOT_LOGGER_NAME)

    if _configured:
        return root

    root.setLevel(level)
    root.propagate = False  # don't also spam Python's real root logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ----- File handler (best-effort -- logging must never crash the app) -----
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            str(_LOG_PATH),
            maxBytes=2 * 1024 * 1024,  # 2MB per file
            backupCount=5,             # ~10MB total ceiling
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Read-only install folder, out of disk space, permissions, etc.
        # -- the app keeps running, it just logs to console only. This
        # itself gets recorded via the console handler below.
        root_console_only = True
    else:
        root_console_only = False

    # ----- Console handler (always attached) -----
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    _configured = True

    if root_console_only:
        root.warning(
            "Couldn't open %s for writing -- logging to console only this session.",
            _LOG_PATH,
        )

    root.info("=" * 60)
    root.info("PixelForge AI -- logging started (%s)", _LOG_PATH)
    root.info("=" * 60)
    return root


def get_logger(name: str = None) -> logging.Logger:
    """
    Returns a child logger of the shared "pixelforge" logger, e.g.
    get_logger(__name__) -> "pixelforge.core.batch_processor". Calling
    this before setup_logging() still works (Python's logging module
    queues nothing, it just has no handlers yet -- messages are dropped
    silently until setup_logging() runs), but every real entry point
    (main.py) calls setup_logging() first, so this is only a concern
    for standalone module testing.
    """
    if name and not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name or _ROOT_LOGGER_NAME)


def log_path() -> str:
    return str(_LOG_PATH)


def is_configured() -> bool:
    return _configured