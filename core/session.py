# core/session.py
#
# PHASE 6 -- Shared Working Image / Edit Session State  (+ global
# undo/redo, + non-destructive editing, + Save Project).
#
# THE PROBLEM THIS FIXES
# ----------------------
# Every tool used to be an island. Filters re-read its own copy of the
# image, Enhance re-read its own copy, Remove BG re-read its own copy --
# so the reported bug "Filter lagaya -> Enhance khola -> original image
# wapas aa gayi" was structural, not a UI glitch: there was no single
# place that knew what "the image I am currently editing" is.
#
# This module is that single place. One authoritative WORKING IMAGE:
#
#   Original  ->  [ Filters | Enhance | Remove BG | Crop | Smart Pipeline ]
#                            |
#                            v
#                     WORKING IMAGE (latest result)
#                            |
#                            v
#                   every view reads THIS on entry
#
# Rules enforced here:
#   1. Every view reads the LATEST working image when it opens.
#   2. After any successful operation, working image = that result.
#   3. The ORIGINAL FILE IS NEVER WRITTEN TO. Ever. State 0 of the
#      history points at the user's original path and is only ever
#      opened read-only; every derived state is a new PNG inside a
#      throwaway session folder under the OS temp dir. The user's photo
#      only changes when they explicitly Export (Ctrl+Shift+S) to a
#      destination they picked themselves.
#   4. One GLOBAL history, so Ctrl+Z walks back across tools --
#      Pipeline -> Remove BG -> Crop -> Enhance -> Filter -> Original --
#      instead of each tool having a private undo stack that the others
#      can't see.
#
# WHY PNG FOR INTERMEDIATE STATES: lossless. A chain of five edits saved
# as JPEG each time would visibly degrade the photo through
# recompression; the whole point of a working image is that the chain
# costs nothing in quality.
#
# PROJECT FILES (.pfproj): a plain ZIP holding project.json + a copy of
# the original + one PNG per history state, so a saved project reopens
# exactly as it was left even after a restart (temp folders are gone by
# then). ZIP + JSON only -- stdlib, no new dependency, and readable by
# hand if anything ever goes wrong.
#
# MODULE SPLIT: same rule as every other core/ module -- state and
# orchestration over already-built engines (core/enhancer.py,
# core/filters.py, core/smart_pipeline.py), no trained model, no Qt.
# ui/bridge.py stays a thin relay on top of this.

import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from PIL import Image

from core.enhancer import DEFAULT_ADJUSTMENTS, apply_adjustments
from core.filters import apply_preset, get_preset
from core.settings import get_setting

PROJECT_EXTENSION = ".pfproj"
PROJECT_FORMAT_VERSION = 1

_SESSIONS_ROOT = Path(tempfile.gettempdir()) / "pixelforge_sessions"

# How long an abandoned session folder is kept before the next launch
# sweeps it up. Long enough that a crash + restart still finds the temp
# states; short enough that temp doesn't grow forever.
_STALE_SESSION_SECONDS = 2 * 24 * 60 * 60

_DEFAULT_HISTORY_LIMIT = 30

# Tools whose UI state travels with the session (so re-opening a view --
# or re-opening a saved project -- restores the sliders the user left
# there, not the factory defaults).
TOOL_KEYS = ("enhance", "filters", "removebg", "crop", "pipeline")


def _history_limit() -> int:
    try:
        value = int(get_setting("history_limit", str(_DEFAULT_HISTORY_LIMIT)))
    except (TypeError, ValueError):
        return _DEFAULT_HISTORY_LIMIT
    # Below 2 there'd be no undo at all; above 200 the temp folder gets
    # silly for full-resolution PNGs.
    return max(2, min(200, value))


def sweep_stale_sessions() -> None:
    """
    Deletes session folders left behind by earlier runs. Called once at
    startup (ui/bridge.py) -- never mid-edit, so it can't remove the
    folder the live session is using.
    """
    if not _SESSIONS_ROOT.exists():
        return
    cutoff = time.time() - _STALE_SESSION_SECONDS
    for child in _SESSIONS_ROOT.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


class EditSession:
    """
    The live edit session: one original, one ordered history of states,
    one cursor into that history (= the working image), plus each tool's
    last-used settings.
    """

    def __init__(self):
        self.session_id = uuid.uuid4().hex[:12]
        self.dir = _SESSIONS_ROOT / self.session_id
        self.original_path = ""
        self.original_name = ""
        self.original_exif = None  # Missing-feature #5 (this session): see start()
        self.history = []       # list of state dicts, oldest first
        self.index = -1         # cursor: which state is the working image
        self.tool_states = {}   # tool -> arbitrary JSON-able UI state
        self.analysis = None    # last Analyzer result (Phase 5)
        self.pipeline = None    # last Smart Pipeline result (Phase 6)
        self.project_path = ""  # where Ctrl+S last saved (if anywhere)
        self.dirty = False
        self._seq = 0

    # ---------------- lifecycle ----------------

    def _ensure_dir(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def start(self, original_path: str) -> dict:
        """
        Begins a brand-new session from a source file on disk. State 0
        POINTS AT the user's file -- it is never copied, never written,
        only ever opened read-only (see rule 3 in the header).
        """
        path = str(original_path or "").strip()
        if not path or not os.path.isfile(path):
            raise ValueError("Image not found.")

        # BUGFIX (container/session duplication): start() used to
        # unconditionally mint a new session_id + folder on every call,
        # even when the caller was simply re-opening the SAME photo
        # that was already the active session (e.g. re-selecting it
        # from a tool's own "Open" button after "Start Project" already
        # created one, or switching tools in a way that re-triggers
        # pixelforgeStartSession). That silently abandoned the current
        # session's history/container and started a brand-new one, so
        # anything exported afterward landed in that new, disconnected
        # container instead of the one the project actually started
        # with. If the resolved path matches the currently active
        # session's original file, just resume it -- same session_id,
        # same dir, same history -- instead of restarting from scratch.
        resolved = os.path.abspath(path)
        if self.history and self.original_path and os.path.abspath(self.original_path) == resolved:
            return self.state()

        # Validate it really is a readable image before we build a
        # session around it, so a bad path fails here instead of
        # halfway through the user's first edit.
        with Image.open(path) as probe:
            probe.verify()

        # Missing-feature #5 (this session): EXIF preservation on
        # export. Every state after this one gets re-saved as PNG by
        # commit_image()/commit_file() without carrying EXIF forward
        # (PNG re-encoding is what makes the shared Working Image
        # lossless/alpha-safe -- see commit_file()'s docstring -- but
        # it also drops metadata), so the ONLY point EXIF is still
        # readable is here, from the original file, before any edit
        # happens. Captured once and carried on the session so
        # exportImage/exportFilterImage in ui/bridge.py can re-attach
        # it to the final saved file however many edits deep the
        # working image now is. Best-effort: a photo with no/corrupt
        # EXIF (e.g. a screenshot, or most PNGs) just exports without
        # it, same as today.
        self.original_exif = None
        try:
            with Image.open(path) as exif_probe:
                self.original_exif = exif_probe.info.get("exif")
        except Exception:
            self.original_exif = None

        self.session_id = uuid.uuid4().hex[:12]
        self.dir = _SESSIONS_ROOT / self.session_id
        self.original_path = os.path.abspath(path)
        self.original_name = os.path.basename(self.original_path)
        self.history = []
        self.index = -1
        self.tool_states = {}
        self.analysis = None
        self.pipeline = None
        self.project_path = ""
        self.dirty = False
        self._seq = 0

        self._push({
            "id": self._next_id(),
            "tool": "original",
            "label": "Original",
            "path": self.original_path,
            "settings": {},
            "op": {"kind": "original"},
            "timestamp": int(time.time()),
            "is_original": True,
        })
        self.dirty = False  # a freshly opened image isn't an unsaved edit
        return self.state()

    def close(self, delete_files: bool = True) -> dict:
        """Ends the session (New Project / closing an image)."""
        if delete_files and self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        self.original_path = ""
        self.original_name = ""
        self.original_exif = None
        self.history = []
        self.index = -1
        self.tool_states = {}
        self.analysis = None
        self.pipeline = None
        self.project_path = ""
        self.dirty = False
        return self.state()

    def get_original_exif(self):
        """
        Raw EXIF bytes captured from the original file at start(), or
        None if there is no active session / the source had none. See
        the long comment in start() above for why this has to be
        captured up front rather than read from the current working
        file. Used by ui/bridge.py's exportImageAsync/
        exportFilterImageAsync to re-attach EXIF on export.
        """
        return self.original_exif

    # ---------------- history plumbing ----------------

    def _next_id(self) -> str:
        self._seq += 1
        return f"s{self._seq:04d}"

    def _push(self, entry: dict) -> None:
        # A new edit after an undo starts a new branch: drop the redo
        # states it would otherwise silently clobber, and delete their
        # files (they're unreachable now). Same branch-truncating rule
        # as the per-tool histories in frontend/removebg.js, applied
        # once, globally.
        for dead in self.history[self.index + 1:]:
            self._discard_file(dead)
        self.history = self.history[: self.index + 1]

        self.history.append(entry)

        limit = _history_limit()
        # Never drop index 0 -- that's the Original, and "reset to
        # original" and the non-destructive guarantee both depend on it
        # still being there. Trim the OLDEST DERIVED state instead.
        while len(self.history) > limit and len(self.history) > 2:
            dropped = self.history.pop(1)
            self._discard_file(dropped)

        self.index = len(self.history) - 1
        self.dirty = True

    def _discard_file(self, entry: dict) -> None:
        """Deletes a state's PNG -- but NEVER the user's original."""
        if not entry or entry.get("is_original"):
            return
        path = entry.get("path")
        if not path:
            return
        try:
            # Only ever delete inside our own session folder. Belt and
            # braces against a remapped/loaded-project path pointing
            # somewhere real on the user's disk.
            if Path(path).resolve().is_relative_to(_SESSIONS_ROOT.resolve()):
                os.unlink(path)
        except (OSError, ValueError):
            pass

    def _state_file(self, entry_id: str) -> Path:
        return self._ensure_dir() / f"state_{entry_id}.png"

    def require_active(self) -> None:
        if not self.history or self.index < 0:
            raise ValueError("No image is open yet. Import an image first.")

    # ---------------- the working image ----------------

    @property
    def working_path(self) -> str:
        if not self.history or self.index < 0:
            return ""
        return self.history[self.index]["path"]

    @property
    def current(self) -> dict:
        if not self.history or self.index < 0:
            return {}
        return self.history[self.index]

    def open_working(self) -> Image.Image:
        """
        Opens the working image for processing. Returns a loaded copy so
        the caller can't hold a lazy handle on the original file.
        """
        self.require_active()
        with Image.open(self.working_path) as img:
            return img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB")

    # ---------------- committing results ----------------

    def commit_image(self, image: Image.Image, tool: str, label: str,
                     settings=None, op=None) -> dict:
        """
        Records `image` as the new working image (rule 2 in the header).
        Written as PNG -- lossless, so a long chain of edits costs
        nothing in quality.
        """
        self.require_active()
        entry_id = self._next_id()
        dest = self._state_file(entry_id)

        save_img = image
        if image.mode not in ("RGB", "RGBA"):
            save_img = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        save_img.save(dest, format="PNG")

        self._push({
            "id": entry_id,
            "tool": tool or "edit",
            "label": label or "Edit",
            "path": str(dest),
            "settings": settings or {},
            "op": op or {"kind": "image"},
            "timestamp": int(time.time()),
            "is_original": False,
        })
        return self.state()

    def commit_file(self, source_path: str, tool: str, label: str,
                    settings=None, op=None, move: bool = False) -> dict:
        """
        Records an ALREADY-RENDERED full-resolution file as the new
        working image -- used by tools that produce their result through
        their own engine (Crop/Straighten via core/crop.py, Remove BG
        via ai/bg_remover.py) instead of handing back a PIL image.

        The file is copied (or moved, for throwaway temp output) into the
        session folder so the session owns its own states and nothing
        outside can invalidate them.
        """
        self.require_active()
        src = str(source_path or "")
        if not os.path.isfile(src):
            raise ValueError("Result file not found.")

        entry_id = self._next_id()
        dest = self._state_file(entry_id)

        # Normalize to PNG regardless of what the tool produced, so
        # every state in the history is lossless and alpha-safe.
        if src.lower().endswith(".png"):
            if move:
                shutil.move(src, dest)
            else:
                shutil.copy2(src, dest)
        else:
            with Image.open(src) as img:
                out = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
                out.save(dest, format="PNG")
            if move:
                try:
                    os.unlink(src)
                except OSError:
                    pass

        self._push({
            "id": entry_id,
            "tool": tool or "edit",
            "label": label or "Edit",
            "path": str(dest),
            "settings": settings or {},
            "op": op or {"kind": "file"},
            "timestamp": int(time.time()),
            "is_original": False,
        })
        return self.state()

    def commit_adjustments(self, adjustments: dict, label: str = "Adjustments",
                           tool: str = "enhance") -> dict:
        """
        Renders core/enhancer.py's manual controls onto the CURRENT
        WORKING IMAGE at full resolution and commits the result. This is
        what makes "Enhance sees the filtered image" true: the input is
        always self.working_path, never the original.
        """
        clean = {k: v for k, v in (adjustments or {}).items() if k in DEFAULT_ADJUSTMENTS}
        source = self.open_working()
        result = apply_adjustments(source, clean)
        return self.commit_image(
            result, tool, label,
            settings={"adjustments": clean},
            op={"kind": "adjustments", "adjustments": clean},
        )

    def commit_preset(self, preset_id: str, intensity: float = 100,
                       label: str = "", tool: str = "filters") -> dict:
        """Applies a Phase 4 preset to the working image and commits it."""
        preset = get_preset(preset_id)
        if not preset:
            raise ValueError("Preset not found.")
        source = self.open_working()
        result = apply_preset(source, preset, intensity)
        pretty = label or f"{preset['name']} {round(intensity)}%"
        return self.commit_image(
            result, tool, pretty,
            settings={"preset_id": preset_id, "intensity": round(float(intensity), 1)},
            op={"kind": "preset", "preset_id": preset_id, "intensity": round(float(intensity), 1)},
        )

    # ---------------- global undo / redo ----------------

    def undo(self) -> dict:
        self.require_active()
        if self.index > 0:
            self.index -= 1
            self.dirty = True
        return self.state()

    def redo(self) -> dict:
        self.require_active()
        if self.index < len(self.history) - 1:
            self.index += 1
            self.dirty = True
        return self.state()

    def jump_to(self, entry_id: str) -> dict:
        """History-panel click: jump straight to any recorded state."""
        self.require_active()
        for i, entry in enumerate(self.history):
            if entry["id"] == entry_id:
                self.index = i
                self.dirty = True
                return self.state()
        raise ValueError("History step not found.")

    def revert_to_original(self) -> dict:
        """
        Moves the cursor back to state 0. Deliberately does NOT delete
        the later states -- Redo still works, so this is undoable like
        everything else.
        """
        self.require_active()
        self.index = 0
        self.dirty = True
        return self.state()

    @property
    def can_undo(self) -> bool:
        return bool(self.history) and self.index > 0

    @property
    def can_redo(self) -> bool:
        return bool(self.history) and self.index < len(self.history) - 1

    # ---------------- per-tool UI state ----------------

    def set_tool_state(self, tool: str, state) -> dict:
        key = (tool or "").strip()
        if not key:
            raise ValueError("Tool name can't be empty.")
        self.tool_states[key] = state
        return self.state()

    def get_tool_state(self, tool: str):
        return self.tool_states.get((tool or "").strip())

    def set_analysis(self, analysis) -> None:
        self.analysis = analysis

    def set_pipeline(self, pipeline) -> None:
        self.pipeline = pipeline

    # ---------------- serialization for the frontend ----------------

    def state(self) -> dict:
        entries = []
        for i, entry in enumerate(self.history):
            entries.append({
                "id": entry["id"],
                "tool": entry.get("tool", "edit"),
                "label": entry.get("label", "Edit"),
                "timestamp": entry.get("timestamp", 0),
                "is_original": bool(entry.get("is_original")),
                "is_current": i == self.index,
                "step": i,
            })
        return {
            "ok": True,
            "has_session": bool(self.history),
            "session_id": self.session_id,
            "original_path": self.original_path,
            "original_name": self.original_name,
            "working_path": self.working_path,
            "current_step": self.index,
            "step_count": len(self.history),
            "can_undo": self.can_undo,
            "can_redo": self.can_redo,
            "current_label": self.current.get("label", "") if self.current else "",
            "current_tool": self.current.get("tool", "") if self.current else "",
            "current_settings": self.current.get("settings", {}) if self.current else {},
            "history": entries,
            "tool_states": self.tool_states,
            "analysis": self.analysis,
            "pipeline": self.pipeline,
            "project_path": self.project_path,
            "project_name": os.path.basename(self.project_path) if self.project_path else "",
            "dirty": self.dirty,
            "is_original_view": bool(self.current.get("is_original")) if self.current else True,
        }

    # ---------------- Save / Load project (Ctrl+S) ----------------

    def save_project(self, dest_path: str = "") -> dict:
        """
        Ctrl+S. Writes a .pfproj ZIP containing:
            project.json      -- everything below, as metadata
            original/<name>   -- a COPY of the original (never the file
                                 itself; the user's photo stays untouched)
            states/<id>.png   -- one image per history step
        so reopening restores the working image, the whole edit history,
        the cursor position, every tool's settings, and the last
        Analyzer/Smart-Pipeline results.

        Passing an empty dest_path re-saves over the current project
        (that's the plain Ctrl+S path once a project has a home).
        """
        self.require_active()
        target = str(dest_path or "").strip() or self.project_path
        if not target:
            raise ValueError("No project path chosen.")
        if not target.lower().endswith(PROJECT_EXTENSION):
            target += PROJECT_EXTENSION

        original_arcname = ""
        if self.original_path and os.path.isfile(self.original_path):
            original_arcname = f"original/{self.original_name or 'original'}"

        entries = []
        for entry in self.history:
            arcname = ""
            if os.path.isfile(entry["path"]):
                if entry.get("is_original") and original_arcname:
                    arcname = original_arcname
                else:
                    arcname = f"states/{entry['id']}.png"
            entries.append({
                "id": entry["id"],
                "tool": entry.get("tool", "edit"),
                "label": entry.get("label", "Edit"),
                "settings": entry.get("settings", {}),
                "op": entry.get("op", {}),
                "timestamp": entry.get("timestamp", 0),
                "is_original": bool(entry.get("is_original")),
                "arcname": arcname,
            })

        manifest = {
            "format": "pixelforge-project",
            "version": PROJECT_FORMAT_VERSION,
            "saved_at": int(time.time()),
            "original_path": self.original_path,
            "original_name": self.original_name,
            "original_arcname": original_arcname,
            "current_step": self.index,
            "tool_states": self.tool_states,
            "analysis": self.analysis,
            "pipeline": self.pipeline,
            "history": entries,
        }

        # Write to a temp file next to the target, then replace -- so an
        # interrupted save can't destroy the previous good project file.
        dest = Path(target)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".pfproj_", suffix=".part", dir=str(dest.parent))
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                zf.writestr("project.json", json.dumps(manifest, indent=2))
                written = set()
                for entry, meta in zip(self.history, entries):
                    arcname = meta["arcname"]
                    if not arcname or arcname in written:
                        continue
                    zf.write(entry["path"], arcname)
                    written.add(arcname)
                if original_arcname and original_arcname not in written:
                    zf.write(self.original_path, original_arcname)
            os.replace(tmp_path, dest)
        except (OSError, zipfile.BadZipFile):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self.project_path = str(dest)
        self.dirty = False
        return {
            **self.state(),
            "saved_to": str(dest),
            "steps_saved": len(entries),
        }

    def load_project(self, path: str) -> dict:
        """
        Opens a .pfproj back into a live session: state images are
        extracted into a FRESH temp session folder (the folder the
        project was created in is long gone), and every history path is
        remapped onto the extracted copies.
        """
        src = str(path or "").strip()
        if not src or not os.path.isfile(src):
            raise ValueError("Project file not found.")

        with zipfile.ZipFile(src, "r") as zf:
            try:
                manifest = json.loads(zf.read("project.json").decode("utf-8"))
            except KeyError:
                raise ValueError("Not a valid PixelForge project file.")
            if manifest.get("format") != "pixelforge-project":
                raise ValueError("Not a valid PixelForge project file.")

            # Fresh session identity before extracting anything.
            self.session_id = uuid.uuid4().hex[:12]
            self.dir = _SESSIONS_ROOT / self.session_id
            self._ensure_dir()

            extracted = {}
            for member in manifest.get("history", []):
                arcname = member.get("arcname")
                if not arcname or arcname in extracted:
                    continue
                try:
                    data = zf.read(arcname)
                except KeyError:
                    continue
                out_path = self.dir / f"state_{member['id']}{Path(arcname).suffix or '.png'}"
                with open(out_path, "wb") as f:
                    f.write(data)
                extracted[arcname] = str(out_path)

            original_arcname = manifest.get("original_arcname") or ""
            original_local = extracted.get(original_arcname, "")
            if original_arcname and not original_local:
                try:
                    data = zf.read(original_arcname)
                    out_path = self.dir / f"original_{Path(original_arcname).name}"
                    with open(out_path, "wb") as f:
                        f.write(data)
                    original_local = str(out_path)
                    extracted[original_arcname] = original_local
                except KeyError:
                    original_local = ""

        history = []
        max_seq = 0
        for member in manifest.get("history", []):
            local = extracted.get(member.get("arcname") or "", "")
            if not local:
                # A state whose image didn't make it into the archive
                # can't be restored as a step -- skip it rather than
                # leaving a history entry that points nowhere.
                continue
            history.append({
                "id": member["id"],
                "tool": member.get("tool", "edit"),
                "label": member.get("label", "Edit"),
                "path": local,
                "settings": member.get("settings", {}),
                "op": member.get("op", {}),
                "timestamp": member.get("timestamp", 0),
                # Deliberately False: this is the EXTRACTED copy inside
                # our temp folder, not the user's file on disk, so it's
                # ours to clean up. original_path below still records
                # where the photo came from.
                "is_original": False,
            })
            try:
                max_seq = max(max_seq, int(str(member["id"]).lstrip("s")))
            except (ValueError, KeyError):
                pass

        if not history:
            raise ValueError("This project has no restorable image states.")

        self.original_path = manifest.get("original_path", "") or original_local
        self.original_name = manifest.get("original_name", "") or os.path.basename(self.original_path)
        self.history = history
        self._seq = max_seq
        saved_index = manifest.get("current_step", len(history) - 1)
        self.index = max(0, min(len(history) - 1, int(saved_index)))
        self.tool_states = manifest.get("tool_states", {}) or {}
        self.analysis = manifest.get("analysis")
        self.pipeline = manifest.get("pipeline")
        self.project_path = os.path.abspath(src)
        self.dirty = False

        # The first restored step is the project's "original" for
        # display purposes even though it lives in temp now.
        self.history[0]["is_original"] = False
        return {**self.state(), "loaded_from": self.project_path}


# ===================== MODULE-LEVEL SINGLETON =====================
#
# One session per running app -- the whole point is that every view
# shares it. ui/bridge.py holds the only reference the frontend can
# reach, and every mutation goes through a Slot, so all access is
# already serialized by the request that triggered it.

_session = EditSession()


def get_session() -> EditSession:
    return _session