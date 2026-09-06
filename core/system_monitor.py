# core/system_monitor.py
#
# PHASE 14 -- Performance: CPU/RAM/disk monitoring.
#
# WHY THIS EXISTS: pieces of this already existed scattered across
# individual features (core/batch_processor.py's `_ram_pressure_relief`,
# ai/upscaler.py's `safety_check`'s RAM/disk checks) -- each reimplementing
# the same `try: import psutil / except ImportError: psutil = None`
# pattern on its own. This module is the single, reusable, testable
# home for those checks, so Phase 14's Settings-page "system health"
# panel (and any future feature) has one place to call instead of
# copy-pasting the pattern again. It deliberately does NOT replace the
# existing per-feature checks (those already work and are tuned to
# their own feature's needs) -- it's additive infrastructure for
# anything new.
#
# GRACEFUL DEGRADATION: psutil is an optional, pinned dependency
# (requirements.txt) but this module must not hard-crash if it's
# somehow missing/broken on a given install -- every function below
# returns a best-effort/partial result instead of raising, same
# philosophy as the rest of Phase 14.

import os
import shutil
from pathlib import Path

try:
    import psutil  # optional -- real CPU/RAM figures when available
except ImportError:  # pragma: no cover - psutil is a pinned dependency
    psutil = None

from core.logger import get_logger

log = get_logger(__name__)


def psutil_available() -> bool:
    return psutil is not None


def get_cpu_percent(interval: float = 0.1) -> float:
    """
    Returns current CPU usage as 0-100. `interval` is a short blocking
    sample window (psutil's own recommended pattern for an accurate
    one-shot reading) -- keep this small since it's meant for a quick
    Settings-page snapshot, not a tight polling loop. Returns -1.0 if
    psutil isn't available (caller should treat that as "unknown", not
    "0%").
    NOTE: for a widget that polls repeatedly (e.g. every 2s), prefer
    get_cpu_percent_live() below instead -- a 0.1s blocking sample is
    too short a window and reads as noisy, quantized jumps (0%, then
    25%, then 50%...) on a multi-core machine, since it only "sees"
    whichever core(s) happened to spike during that 100ms slice rather
    than a representative average.
    """
    if psutil is None:
        return -1.0
    try:
        return float(psutil.cpu_percent(interval=interval))
    except Exception:  # noqa: BLE001
        return -1.0


# Primed once at import time (see get_cpu_percent_live's docstring for
# why priming matters) so the very first live poll already returns a
# meaningful number instead of the guaranteed-0.0 "first call" reading.
if psutil is not None:
    try:
        psutil.cpu_percent(interval=None)
    except Exception:  # noqa: BLE001
        pass


def get_cpu_percent_live() -> float:
    """
    Non-blocking CPU reading meant for a widget that polls repeatedly
    (e.g. Settings > System Health, every ~2s) -- unlike get_cpu_percent's
    0.1s blocking sample, this returns instantly and measures the
    average usage since the PREVIOUS call, which is exactly the ~2s
    window the caller actually cares about. That's what makes it
    smooth/accurate instead of a quantized one-core-spike reading;
    it's specifically wrong for a single one-shot read (the very first
    call before anything has "warmed up" the counter would read 0.0 --
    handled above by priming it once at import time), so
    get_cpu_percent() above stays the right choice for one-off checks.
    Returns -1.0 if psutil isn't available.
    """
    if psutil is None:
        return -1.0
    try:
        return float(psutil.cpu_percent(interval=None))
    except Exception:  # noqa: BLE001
        return -1.0


def get_ram_status() -> dict:
    """
    Returns {"available": bool, "total_gb", "used_gb", "available_gb",
    "percent_used"}. "available": False means psutil couldn't be read --
    every numeric field is then 0.0 and callers should show "Unknown"
    rather than treating it as "0GB used".
    """
    if psutil is None:
        return {
            "available": False, "total_gb": 0.0, "used_gb": 0.0,
            "available_gb": 0.0, "percent_used": 0.0,
        }
    try:
        vm = psutil.virtual_memory()
        gb = 1024 ** 3
        return {
            "available": True,
            "total_gb": round(vm.total / gb, 2),
            "used_gb": round((vm.total - vm.available) / gb, 2),
            "available_gb": round(vm.available / gb, 2),
            "percent_used": round(vm.percent, 1),
        }
    except Exception:  # noqa: BLE001
        return {
            "available": False, "total_gb": 0.0, "used_gb": 0.0,
            "available_gb": 0.0, "percent_used": 0.0,
        }


def check_disk_space(path: str, required_mb: float = 200) -> dict:
    """
    Checks free disk space on the drive containing `path` (the folder
    itself doesn't need to exist yet -- checks the nearest existing
    parent, same pattern as core/batch_processor.py::check_disk_space).
    Returns {"ok": bool, "free_mb": float, "required_mb": float,
    "checked_path": str}. Never raises -- an unreadable/invalid path
    returns ok=False with free_mb=0.0 so callers fail safe (treat it as
    a blocking warning, not silently proceed).
    """
    target = Path(path) if path else Path.cwd()
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        free_bytes = shutil.disk_usage(str(target)).free
        free_mb = free_bytes / (1024 ** 2)
        return {
            "ok": free_mb >= required_mb,
            "free_mb": round(free_mb, 1),
            "required_mb": required_mb,
            "checked_path": str(target),
        }
    except OSError as exc:
        log.warning("check_disk_space couldn't read %s: %s", target, exc)
        return {
            "ok": False, "free_mb": 0.0, "required_mb": required_mb,
            "checked_path": str(target),
        }


def check_disk_writable(path: str) -> dict:
    """
    Confirms the destination folder is actually WRITABLE, not just has
    space -- so export doesn't fail late with a cryptic "Access Denied"
    (spec's explicit Phase 14 "Disk write permission check" item).
    Writes and immediately deletes a small temp probe file rather than
    trusting os.access() alone, since os.access() can be wrong on some
    Windows network-share/permission setups. Returns {"ok": bool,
    "checked_path": str, "error": str|None}.
    """
    target = Path(path) if path else Path.cwd()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"ok": False, "checked_path": str(target), "error": str(exc)}

    probe = target / f".pixelforge_write_test_{os.getpid()}.tmp"
    try:
        with open(probe, "wb") as f:
            f.write(b"ok")
        probe.unlink()
        return {"ok": True, "checked_path": str(target), "error": None}
    except OSError as exc:
        log.warning("check_disk_writable: %s is not writable: %s", target, exc)
        return {"ok": False, "checked_path": str(target), "error": str(exc)}
    finally:
        try:
            if probe.exists():
                probe.unlink()
        except OSError:
            pass


def get_system_snapshot(check_path: str = "", live: bool = False) -> dict:
    """
    One combined read for a Settings-page "System" panel: CPU, RAM,
    disk space at `check_path` (defaults to the app's own folder).
    Best-effort/never raises.
    `live=True` (used by the Settings > System Health widget, which
    polls this every ~2s) swaps in get_cpu_percent_live()'s non-
    blocking, since-last-call reading instead of the default 0.1s
    blocking sample -- see that function's docstring for why a short
    blocking sample reads as noisy/quantized under repeated polling.
    Default (live=False) is unchanged for existing one-shot callers.
    """
    from pathlib import Path as _Path
    default_path = str(_Path(__file__).resolve().parent.parent)
    disk = check_disk_space(check_path or default_path, required_mb=200)
    return {
        "cpu_percent": get_cpu_percent_live() if live else get_cpu_percent(),
        "ram": get_ram_status(),
        "disk": disk,
        "psutil_available": psutil_available(),
    }