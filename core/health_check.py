# core/health_check.py
#
# PHASE 14 -- Startup Health Check.
#
# WHY THIS EXISTS: on launch, PixelForge should tell the user (and the
# log) what's actually working before they hit a confusing failure
# three clicks deep -- e.g. FFmpeg missing (Video Studio export will
# fail), the AI Upscale/Face Restore models not yet downloaded (first
# use will pause to download instead of running immediately), or the
# local database being unreachable (Projects/History won't save).
#
# This is intentionally READ-ONLY and non-blocking: every check here
# only *reports* status, it never downloads a model, never blocks
# startup waiting on a network call, and a single failing check never
# stops the others from running or crashes the app (graceful
# degradation -- exactly the Phase 14 requirement of this name). If a
# check itself raises unexpectedly, that check is reported as an error
# and every other check still runs.
#
# Each module already owns its own status function (ai.upscaler.model_status,
# ai.face_restorer.model_status, core.video_studio.ffmpeg_status) -- this
# module is just the aggregator that calls all of them once, in one
# place, and turns the results into one overall verdict.

from core.logger import get_logger
from core.system_monitor import check_disk_space, check_disk_writable, get_ram_status

log = get_logger(__name__)

STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_ERROR = "error"


def _check(name: str, fn, *, warn_message: str = None) -> dict:
    """Runs one health check, isolating its failure from the rest."""
    try:
        result = fn()
        return {"name": name, "status": STATUS_OK, "detail": result}
    except Exception as exc:  # noqa: BLE001 -- one broken check must not stop the others
        log.warning("Health check %r raised: %s", name, exc)
        return {
            "name": name,
            "status": STATUS_WARNING,
            "detail": warn_message or f"Couldn't check ({exc}).",
        }


def _check_ffmpeg() -> dict:
    from core.video_studio import ffmpeg_status
    status = ffmpeg_status()
    if not status.get("available"):
        return {"status": STATUS_WARNING, "detail":
                 "FFmpeg not found on PATH -- Video Studio export/preview won't work "
                 "until it's installed. See Phase 0 setup notes.", "raw": status}
    return {"status": STATUS_OK, "detail": f"FFmpeg {status.get('version', 'detected')}", "raw": status}


def _check_upscale_model() -> dict:
    from ai.upscaler import model_status
    status = model_status()
    if not status.get("onnxruntime_installed"):
        return {"status": STATUS_WARNING, "detail":
                 "onnxruntime isn't installed -- AI Upscale is unavailable "
                 "(the rest of the app is unaffected).", "raw": status}
    if not status.get("cached"):
        return {"status": STATUS_OK, "detail":
                 f"AI Upscale model not downloaded yet -- first use downloads "
                 f"~{status.get('approx_download_mb', '?')}MB.", "raw": status}
    return {"status": STATUS_OK, "detail":
             f"AI Upscale model ready ({status.get('provider_name', 'CPU')}).", "raw": status}


def _check_face_restore_model() -> dict:
    from ai.face_restorer import model_status
    status = model_status()
    if not status.get("onnxruntime_installed"):
        return {"status": STATUS_WARNING, "detail":
                 "onnxruntime isn't installed -- Face Restoration is unavailable "
                 "(the rest of the app is unaffected).", "raw": status}
    if not status.get("cached"):
        return {"status": STATUS_OK, "detail":
                 f"Face Restoration model not downloaded yet -- first use downloads "
                 f"~{status.get('approx_download_mb', '?')}MB.", "raw": status}
    return {"status": STATUS_OK, "detail":
             f"Face Restoration model ready ({status.get('provider_name', 'CPU')}).", "raw": status}


def _check_bg_remover() -> dict:
    try:
        import rembg  # noqa: F401
    except ImportError:
        return {"status": STATUS_WARNING, "detail":
                 "rembg isn't installed -- Remove Background is unavailable "
                 "(the rest of the app is unaffected)."}
    return {"status": STATUS_OK, "detail": "Remove Background (rembg) ready."}


def _check_database() -> dict:
    from core import database as db
    path = db.db_path()
    # A cheap real read (not just "does the file exist") -- confirms
    # the connection/schema actually works, same call the Projects view
    # makes on load.
    db.recent_projects(limit=1)
    return {"status": STATUS_OK, "detail": f"Database reachable ({path}).", "raw": {"path": path}}


def _check_config() -> dict:
    from core import settings
    data = settings.get_all()
    if not isinstance(data, dict) or not data:
        return {"status": STATUS_WARNING, "detail":
                 "Settings file looked empty/invalid -- using defaults.", "raw": data}
    return {"status": STATUS_OK, "detail": f"Settings loaded ({len(data)} keys).", "raw": {"count": len(data)}}


def _check_output_folder() -> dict:
    from core import settings
    folder = settings.get_setting("output_folder", "")
    if not folder:
        return {"status": STATUS_OK, "detail":
                 "No default output folder set yet -- exports will prompt each time."}
    writable = check_disk_writable(folder)
    if not writable["ok"]:
        return {"status": STATUS_WARNING, "detail":
                 f"Configured output folder isn't writable: {writable.get('error')}",
                 "raw": writable}
    space = check_disk_space(folder, required_mb=200)
    if not space["ok"]:
        return {"status": STATUS_WARNING, "detail":
                 f"Low disk space on output folder ({space['free_mb']}MB free).",
                 "raw": space}
    return {"status": STATUS_OK, "detail": f"Output folder OK ({folder}).", "raw": {"folder": folder}}


def _check_ram() -> dict:
    ram = get_ram_status()
    if not ram["available"]:
        return {"status": STATUS_WARNING, "detail": "Couldn't read RAM usage on this system."}
    if ram["percent_used"] >= 90:
        return {"status": STATUS_WARNING, "detail":
                 f"RAM usage is high ({ram['percent_used']}%). Heavy AI features "
                 "(Upscale, Face Restoration, Video export) may run slowly.",
                 "raw": ram}
    return {"status": STATUS_OK, "detail":
             f"{ram['available_gb']}GB RAM available of {ram['total_gb']}GB.", "raw": ram}


_CHECK_NAMES = [
    ("ffmpeg", "_check_ffmpeg"),
    ("upscale_model", "_check_upscale_model"),
    ("face_restore_model", "_check_face_restore_model"),
    ("background_remover", "_check_bg_remover"),
    ("database", "_check_database"),
    ("config", "_check_config"),
    ("output_folder", "_check_output_folder"),
    ("ram", "_check_ram"),
]


def run_startup_health_check() -> dict:
    """
    Runs every check above, isolating each one's failure, and returns:
        {
            "overall": "ok"|"warning"|"error",
            "checks": [{"name", "status", "detail", "raw"?}, ...],
            "summary": "7/8 OK, 1 warning",
        }
    Always logs the outcome (INFO for a clean pass, WARNING if anything
    needs attention) so a post-mortem "why didn't X work" question can
    be answered by reading logs/pixelforge.log's very first lines.

    Looks up each check function by name on this module at call time
    (rather than capturing the function objects once in a module-level
    list) so tests can monkeypatch e.g. `core.health_check._check_ffmpeg`
    and have it actually take effect.
    """
    import sys
    this_module = sys.modules[__name__]

    results = []
    for name, fn_name in _CHECK_NAMES:
        fn = getattr(this_module, fn_name)
        outcome = _check(name, fn)
        # _check already isolates raise-vs-not; flatten the inner
        # {"status", "detail", "raw"} shape fn() itself returns.
        if isinstance(outcome.get("detail"), dict) and "status" in outcome["detail"]:
            inner = outcome["detail"]
            outcome["status"] = inner.get("status", outcome["status"])
            outcome["raw"] = inner.get("raw")
            outcome["detail"] = inner.get("detail", "")
        results.append(outcome)

    ok_count = sum(1 for r in results if r["status"] == STATUS_OK)
    warn_count = sum(1 for r in results if r["status"] == STATUS_WARNING)
    err_count = sum(1 for r in results if r["status"] == STATUS_ERROR)

    if err_count:
        overall = STATUS_ERROR
    elif warn_count:
        overall = STATUS_WARNING
    else:
        overall = STATUS_OK

    summary = f"{ok_count}/{len(results)} OK"
    if warn_count:
        summary += f", {warn_count} warning{'s' if warn_count != 1 else ''}"
    if err_count:
        summary += f", {err_count} error{'s' if err_count != 1 else ''}"

    report = {"overall": overall, "checks": results, "summary": summary}

    log_fn = log.info if overall == STATUS_OK else log.warning
    log_fn("Startup health check: %s", summary)
    for r in results:
        if r["status"] != STATUS_OK:
            log.warning("  [%s] %s: %s", r["status"].upper(), r["name"], r["detail"])

    return report