# core/resource_guard.py
#
# PHASE 14 -- Performance/Safety helpers: processing timeout, a
# standardized cancellation token, automatic large-image resize before
# AI processing, and a shared memory-cleanup helper.
#
# WHY THIS EXISTS: Phase 9/10/11 each already invented their own
# cancellation flag (a bare `threading.Event`) and their own ad hoc
# "resize before processing" logic where they needed it. That's fine
# and untouched here -- this module doesn't replace any of it. It's the
# reusable version for everything NEW in Phase 14+ (and any future
# phase) so the pattern doesn't get reinvented a fourth and fifth time.
#
# GOLDEN RULE (spec, reconfirmed in Phase 14): never load/process a full
# 20x20MP image into RAM at once if it can be avoided -- resize large
# images down to a sane working size BEFORE handing them to a heavy AI
# pass, same spirit as core/batch_processor.py's strictly-sequential
# per-image loop.

import gc
import queue
import threading

from PIL import Image

from core.logger import get_logger

log = get_logger(__name__)

try:
    import psutil  # optional -- used only for a best-effort RSS log line
except ImportError:  # pragma: no cover
    psutil = None


# ===================== CANCELLATION =====================

class CancelToken:
    """
    Thin, standardized wrapper around threading.Event for cooperative
    cancellation -- the same pattern Phase 9/10/11 already use directly
    via their own `cancel_event`, formalized here so new long-running
    operations don't have to reinvent the "check between steps, raise a
    custom Cancelled exception" boilerplate each time.

    Usage:
        token = CancelToken()
        ...
        for item in items:
            token.raise_if_cancelled(MyCancelledError)
            do_work(item)
        ...
        token.cancel()   # called from another thread/the GUI thread
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self, exc_cls=None) -> None:
        """Raises exc_cls(message) (default: a plain Exception) if cancelled."""
        if self._event.is_set():
            exc_cls = exc_cls or Exception
            raise exc_cls("Operation cancelled.")

    def reset(self) -> None:
        self._event.clear()

    @property
    def event(self) -> threading.Event:
        """Escape hatch for code that wants the raw Event (e.g. to pass
        straight into an existing function's `cancel_event=` parameter,
        like ai.upscaler.upscale_image)."""
        return self._event


# ===================== TIMEOUT =====================

class ProcessingTimeout(Exception):
    """Raised by run_with_timeout() when `fn` doesn't finish in time."""


def run_with_timeout(fn, timeout_seconds: float, *args, **kwargs):
    """
    Runs fn(*args, **kwargs) on a helper thread and waits up to
    `timeout_seconds`. If it finishes in time, returns its result (or
    re-raises whatever exception it raised). If it doesn't, raises
    ProcessingTimeout -- but note the helper thread itself is NOT
    force-killed (Python can't safely kill a thread); a well-behaved
    long-running function should accept/check a CancelToken so it can
    actually stop promptly instead of running on in the background
    after the caller gives up on it.

    Meant for wrapping a single bounded step (e.g. one image's AI pass)
    where a value is genuinely useful, not for wrapping an entire batch
    run (core/batch_processor.py already has its own per-item timing
    and doesn't need this).
    """
    result_q = queue.Queue(maxsize=1)

    def _worker():
        try:
            result_q.put(("ok", fn(*args, **kwargs)))
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller's thread below
            result_q.put(("error", exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        log.warning("run_with_timeout: %r exceeded %ss, still running in background.",
                    getattr(fn, "__name__", fn), timeout_seconds)
        raise ProcessingTimeout(
            f"This is taking longer than expected ({timeout_seconds:.0f}s) and was stopped."
        )

    status, payload = result_q.get()
    if status == "error":
        raise payload
    return payload


# ===================== AUTO-RESIZE BEFORE AI PROCESSING =====================

def resize_if_too_large(image: Image.Image, max_dimension: int = 4000) -> Image.Image:
    """
    Downscales `image` (keeping aspect ratio) if its longest side
    exceeds `max_dimension`, otherwise returns it unchanged. Meant to
    run BEFORE handing an image to a heavy AI pass, per the spec's
    "Automatic large-image resize before AI processing" requirement --
    on the 16GB/no-GPU target hardware, a 40MP source photo run through
    a neural net tiled pass is needlessly slow and RAM-heavy when the
    feature's own output cap (see e.g. ai/upscaler.py's
    MAX_OUTPUT_MEGAPIXELS) will scale it back down anyway.

    Uses LANCZOS (high-quality downscale) since this feeds further
    processing, not a final export -- correctness/quality matters more
    than raw speed here (the AI pass itself is the slow part).
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_dimension:
        return image
    scale = max_dimension / float(longest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    log.info("resize_if_too_large: %sx%s -> %sx%s (cap %spx)",
              width, height, new_size[0], new_size[1], max_dimension)
    return image.resize(new_size, Image.LANCZOS)


# ===================== MEMORY CLEANUP =====================

def cleanup_memory(context: str = "") -> None:
    """
    Forces a GC pass and logs current RSS (if psutil is available) so
    memory behavior is visible in logs/pixelforge.log during long
    sessions/batches -- same `gc.collect()` core/batch_processor.py
    already calls after every batch item, exposed here as a shared,
    loggable helper for any other feature that wants it.
    """
    gc.collect()
    if psutil is not None:
        try:
            rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
            log.debug("cleanup_memory%s: RSS now %.1fMB", f" [{context}]" if context else "", rss_mb)
        except Exception:  # noqa: BLE001
            pass