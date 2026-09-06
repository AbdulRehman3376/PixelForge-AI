# core/error_handler.py
#
# PHASE 14 -- Friendly Error Handling.
#
# WHY THIS EXISTS: ui/bridge.py's `_run_async` (the single choke point
# every AI/heavy Slot already runs through -- see its own docstring)
# turned any unexpected exception into `{"ok": False, "error": str(exc)}`
# and threw the technical detail away afterwards -- nothing was logged,
# and the user saw whatever Python's exception message happened to be
# (e.g. a bare `[Errno 13]` for a permissions problem, or a cryptic
# `'NoneType' object has no attribute 'shape'` for a bug). This module
# is the fix: every unexpected exception is now (1) logged in full,
# with the real traceback, to logs/pixelforge.log, and (2) converted to
# a short, plain-language message before it ever reaches the frontend.
#
# This is deliberately layered UNDER the app's existing, deliberately-
# raised error classes (UpscaleError, FaceRestoreError, VideoStudioError,
# etc.) -- those already carry a hand-written, user-facing message from
# the module that raised them, so this layer passes their `str(exc)`
# through as-is (still logged, just not re-worded). Only *unexpected*
# exception types (KeyError, AttributeError, MemoryError, OSError,
# sqlite3 errors, ...) get the friendly-message treatment below.

import sys
import traceback

from core.logger import get_logger

log = get_logger(__name__)


class PixelForgeError(Exception):
    """
    Optional common base for future app-raised errors. Existing modules
    (UpscaleError, FaceRestoreError, VideoStudioError, UpscaleCancelled,
    etc.) don't need to inherit from this to benefit from this module --
    to_friendly_error() below treats any exception whose message was
    clearly hand-written (i.e. everything that isn't in the KNOWN_TYPES
    map) as already friendly and passes it through unchanged.
    """


# Exception types this module recognizes as "unexpected/technical" --
# these get REPLACED with a friendly message. Anything not in this map
# is assumed to already carry a human-readable message (the app's own
# *Error classes all do) and is passed through as str(exc).
_FRIENDLY_MESSAGES = {
    "MemoryError": "PixelForge ran out of memory processing this. Try a "
                    "smaller image, close other apps, or process fewer "
                    "images at once.",
    "FileNotFoundError": "That file couldn't be found. It may have been "
                          "moved, renamed, or deleted.",
    "PermissionError": "PixelForge doesn't have permission to read or write "
                        "there. Try a different folder, or check the "
                        "file/folder isn't open in another program.",
    "IsADirectoryError": "That's a folder, not a file -- pick a specific file.",
    "NotADirectoryError": "That folder path isn't valid.",
    "TimeoutError": "That took too long and was stopped. Try again, or "
                     "with a smaller image/shorter clip.",
    "ConnectionError": "Couldn't connect to the internet. Check your "
                        "connection and try again.",
    "JSONDecodeError": "A saved file appears to be corrupted and couldn't "
                        "be read. It may need to be recreated.",
    "OperationalError": "The local database couldn't be reached right now. "
                         "Try again, or restart PixelForge if this keeps happening.",
    "DatabaseError": "The local database hit an unexpected problem. Try "
                      "again, or restart PixelForge if this keeps happening.",
    "UnidentifiedImageError": "That file doesn't look like a valid, "
                               "readable image.",
    "KeyError": "PixelForge hit an unexpected internal error (missing "
                "data). This has been logged -- please try again.",
    "AttributeError": "PixelForge hit an unexpected internal error. This "
                       "has been logged -- please try again.",
    "TypeError": "PixelForge hit an unexpected internal error (wrong data "
                 "type). This has been logged -- please try again.",
    "IndexError": "PixelForge hit an unexpected internal error (out of "
                  "range). This has been logged -- please try again.",
    "ZeroDivisionError": "PixelForge hit an unexpected internal calculation "
                          "error. This has been logged -- please try again.",
}

# A generic catch-all for anything not explicitly listed above (custom
# app exceptions with clear messages are NOT routed through this --
# see is_known_app_error below).
_GENERIC_FALLBACK = (
    "Something unexpected went wrong. This has been logged -- please "
    "try again, and restart PixelForge if it keeps happening."
)

# Low-level OSError subclasses (disk full, too many open files, etc.)
# that aren't already covered by name above but share OSError's
# .errno/.strerror shape -- handled generically by class rather than
# listed one-by-one.
_OS_ERROR_FALLBACK = (
    "PixelForge couldn't complete a file/disk operation ({detail}). "
    "Check available disk space and that the file isn't open elsewhere."
)


def is_known_app_error(exc: BaseException) -> bool:
    """
    True for the app's own deliberately-raised error classes (anything
    ending in "Error" that is NOT one of Python's built-in technical
    exception types listed in _FRIENDLY_MESSAGES, and not a raw OSError/
    sqlite3 error). Those already carry a hand-written, user-facing
    message from the raising module -- e.g. UpscaleError,
    FaceRestoreError, VideoStudioError, ValueError raised explicitly
    with a clear message by core/*.py.

    Cancellation exceptions (name ending in "Cancelled") are treated
    the same way -- their message is already the right one to show.
    """
    type_name = type(exc).__name__
    if type_name in _FRIENDLY_MESSAGES:
        return False
    if isinstance(exc, OSError) and type_name not in _FRIENDLY_MESSAGES:
        return False
    return True


def _safe_str(exc: BaseException) -> str:
    """
    str(exc) can itself raise (a custom __str__/__repr__ that blows up,
    or a C-extension exception with a broken message) -- this must
    NEVER be allowed to crash error handling itself, so every caller
    below goes through this instead of a bare str(exc).
    """
    try:
        return str(exc)
    except Exception:  # noqa: BLE001
        return ""


def to_friendly_message(exc: BaseException) -> str:
    """
    Converts any exception into a single, human-readable sentence safe
    to show directly in the UI. Never raises, even if the exception's
    own __str__ is broken.
    """
    type_name = type(exc).__name__

    if type_name in _FRIENDLY_MESSAGES:
        return _FRIENDLY_MESSAGES[type_name]

    if isinstance(exc, OSError):
        detail = getattr(exc, "strerror", None) or _safe_str(exc) or "unknown error"
        return _OS_ERROR_FALLBACK.format(detail=detail)

    if is_known_app_error(exc):
        # The app's own *Error/*Cancelled classes -- message is already
        # written for a human by the module that raised it.
        text = _safe_str(exc).strip()
        return text if text else _GENERIC_FALLBACK

    return _GENERIC_FALLBACK


def log_exception(exc: BaseException, context: str = "") -> None:
    """
    Logs the FULL technical detail (type, message, traceback) at ERROR
    level, tagged with `context` (e.g. the Slot/function name) so a
    later grep through logs/pixelforge.log can find exactly what failed
    and where. Never raises, even if logging itself fails.
    """
    try:
        where = f" [{context}]" if context else ""
        log.error(
            "Unhandled exception%s: %s: %s\n%s",
            where,
            type(exc).__name__,
            _safe_str(exc),
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
    except Exception:  # noqa: BLE001 -- logging must never itself crash the app
        pass


def log_and_convert(exc: BaseException, context: str = "") -> dict:
    """
    The single function ui/bridge.py's _run_async now calls in its
    except block. Logs the full traceback, then returns the same
    {"ok": False, "error": ...} shape the frontend already expects --
    zero contract change, just a friendlier/logged message inside it.
    """
    log_exception(exc, context=context)
    return {"ok": False, "error": to_friendly_message(exc)}


def install_global_exception_hook() -> None:
    """
    Catches anything that would otherwise crash the app to the desktop
    with no record at all (a bug outside any try/except, e.g. during
    Qt event handling or app startup). Logs the full traceback, then
    calls the previous excepthook (Python's default, which prints to
    stderr) so behavior during development is unchanged -- this only
    ADDS a durable log entry, it doesn't hide/suppress crashes.
    """
    previous_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc_value, exc_tb)
            return
        try:
            log.critical(
                "UNCAUGHT EXCEPTION (app-level): %s: %s\n%s",
                exc_type.__name__,
                exc_value,
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            )
        except Exception:  # noqa: BLE001
            pass
        previous_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook