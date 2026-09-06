# core/batch_processor.py
#
# PHASE 8 -- Batch Processing (orchestration/queue layer).
#
# 🧮 NOT AI. This module owns no model of its own -- it runs the
# already-built tools (Phase 5 Analyzer, Phase 6 Smart Pipeline, Phase 4
# Presets/Filters) over many files in sequence:
#
#     20 images -> Analyze -> Enhance/Preset -> Export
#
# WHY A DEDICATED MODULE (not just a loop in ui/bridge.py)
# ----------------------------------------------------------
# Batch has real state that has to survive across many Slot calls from
# the frontend (queue contents, pause/cancel flags, per-item progress,
# a running log) -- that's a controller object, not a one-shot function.
# Keeping it here (not in ui/bridge.py) matches every earlier phase's
# layering: ui/bridge.py stays a thin Slot/Signal wrapper, the real
# logic lives in core/.
#
# THREADING MODEL
# ----------------
# BatchController.run() is a BLOCKING call meant to be started on a
# background thread by ui/bridge.py (same pattern as every other heavy
# operation -- see _run_async/taskResult in ui/bridge.py). While it
# runs, the GUI thread can still call pause()/resume()/cancel()/
# retry_item()/set_included()/reorder() etc; those only flip flags or
# touch the queue list, guarded by self._lock, so they're safe to call
# from another thread while run() is mid-flight.
#
# TARGET-HARDWARE RULE (Phase 14's "golden rule", enforced here already
# since Phase 8 IS the batch queue that rule was written for):
# strictly sequential, one image fully in memory at a time, released
# before the next one loads:
#
#     Image 1 -> Analyze -> Process -> Export -> Release RAM
#     Image 2 -> Analyze -> Process -> Export -> Release RAM
#     ...
#
# No threading/multiprocessing fan-out across images -- that would load
# several full-res images (and, once Phase 9/10 land, several heavy
# models) into RAM at once on a 16GB no-GPU laptop.

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

try:
    import psutil  # optional -- real CPU/RAM figures when available
except ImportError:  # pragma: no cover - not a hard dependency
    psutil = None

from core.analyzer import analyze_image
from core.filters import apply_preset, get_preset
from core.smart_pipeline import apply_pipeline, recommend_pipeline
from core.logger import get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Statuses an item can be in.
STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_PREVIEWED = "previewed"  # dry-run result staged, waiting for commit/discard
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_CANCELLED = "cancelled"

_TERMINAL = {STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED, STATUS_CANCELLED}

DEFAULT_OPTIONS = {
    # "smart"  -> Phase 6 Smart Pipeline recommendation per image (default)
    # "preset" -> one chosen Phase 4 preset applied to the whole batch
    # "none"   -> no look change; useful for pure resize/format-convert/rename runs
    "mode": "smart",
    "preset_id": "",
    "intensity": 80,
    "output_folder": "",
    "export_format": "keep",   # "keep" | "jpg" | "png" | "webp"
    "export_quality": 92,      # JPEG/WebP quality, 1-100
    "resize_mode": "none",     # "none" | "max_dimension" | "percent"
    "resize_value": 2048,
    "naming_template": "{name}",
    "preserve_structure": True,
    "overwrite": False,
    "strip_exif": False,
    "copyright_author": "",
    "cpu_throttle": 0,         # 0-100 -- % of each item's processing time added as an idle pause after it
    "notify_on_complete": True,
    "dry_run": False,          # "Preview before commit" -- stage results, don't write to output_folder yet

    # PHASE 9 -- AI Upscale batch integration ("an 'AI Upscale' step
    # selectable inside Phase 8's Batch queue (alongside Look/Preset), so
    # a whole queue can be upscaled in one run, not just single images").
    # Deliberately independent of `mode` above -- a batch can apply a
    # Look/Preset AND upscale on top, or upscale on its own with
    # mode="none". See ai/upscaler.py for what each option means.
    "upscale_enabled": False,
    "upscale_scale": "2x",     # "2x" | "4x" | "custom"
    "upscale_custom_width": None,
    "upscale_custom_height": None,
    "upscale_denoise_strength": 0,
    "upscale_artifact_reduction": 0,
    "upscale_face_aware": False,

    # PHASE 10 -- Face Restoration batch integration, same independent-
    # of-`mode` convention as Phase 9's upscale_* options directly
    # above: a batch can apply a Look/Preset AND restore faces on top,
    # restore on its own with mode="none", and/or combine with
    # upscale_enabled. See ai/face_restorer.py for what each option
    # means. There's no per-image face-selection UI in a batch run (20
    # images can't each get their own checklist), so batch mode always
    # restores EVERY face auto-detected in each image -- honestly
    # documented here rather than silently only doing "face 1".
    "facerestore_enabled": False,
    "facerestore_strength": 80,
    "facerestore_natural_detailed": 50,
    "facerestore_skin_protection": 60,
}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _ahash(path: Path):
    """
    Cheap 8x8 average-hash for near-duplicate detection. Deliberately
    NOT a new dependency (no imagehash package) -- Pillow + stdlib only,
    same "stay free/local, minimal deps" rule the whole app follows.
    Returns a 64-bit int, or None if the file can't be read as an image.
    """
    try:
        with Image.open(path) as img:
            small = img.convert("L").resize((8, 8), Image.LANCZOS)
            pixels = list(small.getdata())
    except Exception:  # noqa: BLE001 -- a bad/corrupt file just can't be hashed
        return None
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return int(bits, 2)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _categorize(analysis: dict) -> str:
    """Batch Analyzer summary bucket -- mirrors the spec's example report
    (Portrait / Landscape / Night / Other)."""
    if analysis.get("face_detected"):
        return "Portrait"
    if analysis.get("lighting") == "dark":
        return "Night"
    if analysis.get("subject_type") == "landscape_scene":
        return "Landscape"
    return "Other"


class BatchItem:
    """One queue entry. Plain object with a to_dict() for the JSON the
    frontend renders -- same convention as core/session.py's state()."""

    def __init__(self, source_path: str, relative_dir: str = ""):
        self.id = uuid.uuid4().hex[:10]
        self.source_path = source_path
        # Sub-folder path relative to the imported root, used by
        # "preserve folder structure" -- "" for flat file/drag-drop imports.
        self.relative_dir = relative_dir
        self.name = Path(source_path).name
        self.included = True
        self.status = STATUS_QUEUED
        self.progress = 0  # 0-100, this item only
        self.error = ""
        self.output_path = ""
        self.category = ""       # filled in after analysis (Portrait/Landscape/...)
        self.applied_look = ""   # preset/pipeline name actually used
        self.warnings = []
        self.duplicate_of = ""   # id of an earlier item this looks like a near-duplicate of
        self.staged_path = ""    # dry-run output, waiting in a temp staging dir for commit/discard
        self.started_at = None
        self.finished_at = None
        self.duration_seconds = None
        try:
            self.size_bytes = os.path.getsize(source_path)
        except OSError:
            self.size_bytes = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "relative_dir": self.relative_dir,
            "name": self.name,
            "included": self.included,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "output_path": self.output_path,
            "category": self.category,
            "applied_look": self.applied_look,
            "warnings": self.warnings,
            "duplicate_of": self.duplicate_of,
            "staged_path": self.staged_path,
            "duration_seconds": self.duration_seconds,
            "size_bytes": self.size_bytes,
        }


class BatchController:
    """
    Owns the queue + options for one Batch session. ui/bridge.py holds a
    single instance of this (like it holds one core/session.py
    EditSession) and exposes its methods as Slots.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.items: list[BatchItem] = []
        self.options: dict = dict(DEFAULT_OPTIONS)
        self._pause_event = threading.Event()
        self._pause_event.set()  # set == NOT paused (Event semantics: wait() blocks only while cleared)
        self._cancel_flag = False
        self._running = False
        self._skip_requested_id = None
        self.log: list[dict] = []
        self._run_started_at = None
        self._per_item_durations: list[float] = []
        self._staging_dir = None  # created lazily, only if a dry-run is actually used

    # ===================== QUEUE BUILDING =====================

    def clear(self):
        with self._lock:
            if self._running:
                raise RuntimeError("Cannot clear the queue while a batch is running.")
            self.items = []
            self.log = []
            self._cleanup_staging()
        discard_recovery_state()

    def add_files(self, paths: list) -> dict:
        added, skipped = [], []
        with self._lock:
            existing = {i.source_path for i in self.items}
            for p in paths:
                path = Path(p)
                if not path.exists() or not _is_image(path):
                    skipped.append(p)
                    continue
                if str(path) in existing:
                    skipped.append(p)
                    continue
                item = BatchItem(str(path))
                self.items.append(item)
                added.append(item.id)
        return {"added": len(added), "skipped": len(skipped)}

    def add_folder(self, folder: str, recursive: bool = True) -> dict:
        root = Path(folder)
        if not root.is_dir():
            return {"added": 0, "skipped": 0, "error": "Folder not found."}
        pattern_iter = root.rglob("*") if recursive else root.glob("*")
        added, skipped = 0, 0
        with self._lock:
            existing = {i.source_path for i in self.items}
            for path in sorted(pattern_iter):
                if not path.is_file() or not _is_image(path):
                    continue
                if str(path) in existing:
                    skipped += 1
                    continue
                rel_dir = str(path.parent.relative_to(root)) if path.parent != root else ""
                if rel_dir == ".":
                    rel_dir = ""
                item = BatchItem(str(path), relative_dir=rel_dir)
                self.items.append(item)
                added += 1
        return {"added": added, "skipped": skipped}

    def remove_item(self, item_id: str) -> bool:
        with self._lock:
            before = len(self.items)
            self.items = [i for i in self.items if i.id != item_id]
            return len(self.items) != before

    def set_included(self, item_id: str, included: bool) -> bool:
        with self._lock:
            item = self._find(item_id)
            if not item:
                return False
            item.included = bool(included)
            return True

    def reorder(self, item_ids: list) -> bool:
        """Explicit full reorder -- drives drag-to-reorder in the UI."""
        with self._lock:
            by_id = {i.id: i for i in self.items}
            if set(item_ids) != set(by_id.keys()):
                return False
            self.items = [by_id[iid] for iid in item_ids]
            return True

    def bump_priority(self, item_id: str, direction: str) -> bool:
        """direction: 'top' | 'up' | 'down' | 'bottom' -- the priority-queue control."""
        with self._lock:
            idx = next((i for i, it in enumerate(self.items) if it.id == item_id), None)
            if idx is None:
                return False
            item = self.items.pop(idx)
            if direction == "top":
                self.items.insert(0, item)
            elif direction == "bottom":
                self.items.append(item)
            elif direction == "up":
                self.items.insert(max(0, idx - 1), item)
            elif direction == "down":
                self.items.insert(min(len(self.items), idx + 1), item)
            else:
                self.items.insert(idx, item)
                return False
            return True

    def set_options(self, options: dict) -> dict:
        with self._lock:
            merged = dict(self.options)
            merged.update({k: v for k, v in (options or {}).items() if k in DEFAULT_OPTIONS})
            self.options = merged
            return dict(self.options)

    def _find(self, item_id: str):
        return next((i for i in self.items if i.id == item_id), None)

    # ===================== DUPLICATE DETECTION =====================

    def detect_duplicates(self, threshold: int = 6) -> list:
        """
        Flags near-duplicate photos before processing (burst shots,
        near-identical exports). O(n^2) hamming compare on a 64-bit
        hash -- fine for realistic batch sizes (hundreds of images).
        Returns groups: [{ "items": [id, ...] }, ...]. Mutates
        item.duplicate_of so the queue UI can badge them.
        """
        with self._lock:
            hashes = {}
            for item in self.items:
                item.duplicate_of = ""
                h = _ahash(Path(item.source_path))
                if h is not None:
                    hashes[item.id] = h

            groups = []
            seen = set()
            ids = list(hashes.keys())
            for i, id_a in enumerate(ids):
                if id_a in seen:
                    continue
                group = [id_a]
                for id_b in ids[i + 1:]:
                    if id_b in seen:
                        continue
                    if _hamming(hashes[id_a], hashes[id_b]) <= threshold:
                        group.append(id_b)
                        seen.add(id_b)
                if len(group) > 1:
                    seen.add(id_a)
                    groups.append(group)
                    for later_id in group[1:]:
                        self._find(later_id).duplicate_of = id_a

            return [{"items": g} for g in groups]

    # ===================== DISK SPACE CHECK =====================

    def check_disk_space(self) -> dict:
        """
        Estimates total output size (sum of source file sizes -- a
        reasonable proxy since export re-encodes but rarely balloons
        far past the source) and compares to free space on the output
        drive. Relevant given the 512GB SSD target hardware.
        """
        with self._lock:
            included = [i for i in self.items if i.included]
            estimated_bytes = sum(i.size_bytes for i in included)
        out_folder = self.options.get("output_folder") or str(Path(tempfile.gettempdir()))
        try:
            usage = shutil.disk_usage(out_folder)
            free_bytes = usage.free
        except OSError:
            free_bytes = None
        warning = None
        if free_bytes is not None and estimated_bytes > 0:
            if free_bytes < estimated_bytes * 1.2:
                warning = "Low disk space: estimated output may not fit the free space on this drive."
        return {
            "ok": warning is None,
            "estimated_bytes": estimated_bytes,
            "free_bytes": free_bytes,
            "warning": warning,
        }

    # ===================== CONTROL (pause/resume/cancel/skip/retry) =====================

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def cancel(self):
        self._cancel_flag = True
        self._pause_event.set()  # unblock a paused run so it can see the cancel flag

    def skip_item(self, item_id: str) -> bool:
        with self._lock:
            item = self._find(item_id)
            if not item:
                return False
            if item.status == STATUS_QUEUED:
                item.status = STATUS_SKIPPED
                return True
            if item.status == STATUS_PROCESSING:
                # Checked between the analyze/process/export sub-steps
                # of _process_one() for the currently-running item.
                self._skip_requested_id = item_id
                return True
            return False

    def retry_item(self, item_id: str) -> bool:
        with self._lock:
            item = self._find(item_id)
            if not item or item.status not in (STATUS_FAILED, STATUS_SKIPPED, STATUS_CANCELLED):
                return False
            item.status = STATUS_QUEUED
            item.error = ""
            item.progress = 0
            return True

    def retry_failed(self) -> int:
        with self._lock:
            count = 0
            for item in self.items:
                if item.status == STATUS_FAILED:
                    item.status = STATUS_QUEUED
                    item.error = ""
                    item.progress = 0
                    count += 1
            return count

    def requeue_all(self) -> int:
        """
        Puts every finished item (done/failed/skipped/cancelled) back
        into the queue so the whole batch can be run again -- e.g. after
        changing the Look/preset/intensity and wanting to re-export
        everything, rather than only being able to Retry Failed ones.
        Any still-staged dry-run previews are discarded first (their
        temp files no longer match the about-to-change options).
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Cannot requeue while a batch is running.")
            self.discard_previewed()
            count = 0
            for item in self.items:
                if item.status in (STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED, STATUS_CANCELLED):
                    item.status = STATUS_QUEUED
                    item.error = ""
                    item.progress = 0
                    item.output_path = ""
                    count += 1
            return count

    # ===================== NAMING / RESIZE / EXPORT HELPERS =====================

    def _output_extension(self, source: Path) -> str:
        fmt = self.options.get("export_format", "keep")
        return {"jpg": ".jpg", "png": ".png", "webp": ".webp"}.get(fmt, source.suffix.lower())

    def _apply_naming_template(self, item: BatchItem, index: int, look_name: str) -> str:
        template = self.options.get("naming_template") or "{name}"
        source = Path(item.source_path)
        tokens = {
            "name": source.stem,
            "ext": self._output_extension(source).lstrip("."),
            "index": f"{index:04d}",
            "date": datetime.now().strftime("%Y%m%d"),
            "time": datetime.now().strftime("%H%M%S"),
            "preset": look_name or "original",
        }
        try:
            return template.format(**tokens)
        except (KeyError, IndexError):
            return tokens["name"]

    def _build_output_path(self, item: BatchItem, index: int, look_name: str) -> Path:
        out_folder = Path(self.options.get("output_folder") or ".")
        if self.options.get("preserve_structure") and item.relative_dir:
            out_folder = out_folder / item.relative_dir
        out_folder.mkdir(parents=True, exist_ok=True)
        stem = self._apply_naming_template(item, index, look_name)
        ext = self._output_extension(Path(item.source_path))
        dest = out_folder / f"{stem}{ext}"
        if not self.options.get("overwrite") and dest.exists():
            n = 2
            while (out_folder / f"{stem} ({n}){ext}").exists():
                n += 1
            dest = out_folder / f"{stem} ({n}){ext}"
        return dest

    def _staging_path(self, item: BatchItem) -> Path:
        if self._staging_dir is None:
            self._staging_dir = Path(tempfile.mkdtemp(prefix="pixelforge_batch_preview_"))
        ext = self._output_extension(Path(item.source_path))
        return self._staging_dir / f"{item.id}{ext}"

    def _cleanup_staging(self):
        if self._staging_dir and Path(self._staging_dir).exists():
            shutil.rmtree(self._staging_dir, ignore_errors=True)
        self._staging_dir = None

    def commit_previewed(self, item_ids: list = None) -> dict:
        """
        Finalizes dry-run results: moves each previewed item's staged
        file to its real output path (computed fresh here, so the
        naming template's {index}/{date}/{time} tokens reflect commit
        time, not preview time) and marks it done. `item_ids=None`
        commits every previewed item.
        """
        with self._lock:
            targets = [
                i for i in self.items
                if i.status == STATUS_PREVIEWED and (item_ids is None or i.id in item_ids)
            ]
            committed, failed = 0, 0
            for index, item in enumerate(targets, start=1):
                try:
                    dest = self._build_output_path(item, index, item.applied_look)
                    shutil.move(item.staged_path, dest)
                    item.output_path = str(dest)
                    item.staged_path = ""
                    item.status = STATUS_DONE
                    committed += 1
                except Exception as exc:  # noqa: BLE001
                    item.status = STATUS_FAILED
                    item.error = f"Couldn't commit preview: {exc}"
                    failed += 1
            return {"committed": committed, "failed": failed}

    def discard_previewed(self, item_ids: list = None) -> dict:
        """Deletes staged dry-run results and returns those items to the
        queue so they can be reprocessed (different options) or removed."""
        with self._lock:
            targets = [
                i for i in self.items
                if i.status == STATUS_PREVIEWED and (item_ids is None or i.id in item_ids)
            ]
            for item in targets:
                if item.staged_path:
                    Path(item.staged_path).unlink(missing_ok=True)
                item.staged_path = ""
                item.status = STATUS_QUEUED
                item.progress = 0
            return {"discarded": len(targets)}

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        mode = self.options.get("resize_mode", "none")
        value = self.options.get("resize_value")
        if mode == "none" or not value:
            return image
        w, h = image.size
        if mode == "percent":
            scale = max(0.01, min(4.0, float(value) / 100.0))
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        elif mode == "max_dimension":
            longest = max(w, h)
            if longest <= value:
                return image
            scale = float(value) / longest
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        else:
            return image
        return image.resize(new_size, Image.LANCZOS)

    def _save(self, image: Image.Image, dest: Path, original_exif):
        ext = dest.suffix.lower()
        save_kwargs = {}
        if ext in (".jpg", ".jpeg"):
            image = image.convert("RGB")
            save_kwargs["quality"] = int(self.options.get("export_quality", 92))
        elif ext == ".webp":
            save_kwargs["quality"] = int(self.options.get("export_quality", 92))

        if not self.options.get("strip_exif") and original_exif:
            save_kwargs["exif"] = original_exif

        author = (self.options.get("copyright_author") or "").strip()
        if author and not self.options.get("strip_exif"):
            # Best-effort copyright/author injection. Pillow's plain
            # save(exif=...) path only reliably carries the raw EXIF
            # blob forward -- rewriting a specific tag inside it needs
            # piexif, which isn't a project dependency, so this stays a
            # graceful no-op (silently skipped) on formats/EXIF blobs
            # where Pillow can't attach the tag rather than failing the
            # whole export.
            try:
                exif_obj = image.getexif()
                exif_obj[0x8298] = author  # Copyright tag
                exif_obj[0x013B] = author  # Artist tag
                save_kwargs["exif"] = exif_obj.tobytes()
            except Exception:  # noqa: BLE001
                pass

        try:
            image.save(dest, **save_kwargs)
        except Exception:
            # e.g. an EXIF blob the target format rejects -- retry clean.
            save_kwargs.pop("exif", None)
            image.save(dest, **save_kwargs)

    # ===================== PROCESS ONE ITEM =====================

    def _process_one(self, item: BatchItem, index: int, item_progress_cb=None) -> None:
        item.started_at = time.time()
        item.status = STATUS_PROCESSING
        item.progress = 0

        def sub_progress(pct):
            item.progress = pct
            if item_progress_cb:
                item_progress_cb(item, pct)

        try:
            with Image.open(item.source_path) as src:
                image = src.convert("RGB") if src.mode not in ("RGB", "RGBA") else src.copy()
                original_exif = src.info.get("exif")

            sub_progress(20)

            look_name = ""
            mode = self.options.get("mode", "smart")
            if mode == "smart":
                analysis = analyze_image(image)
                item.category = _categorize(analysis)
                recommendation = recommend_pipeline(analysis)
                look_name = recommendation.get("name", "")
                intensity = self.options.get("intensity")
                image = apply_pipeline(image, recommendation, intensity=intensity)
            elif mode == "preset":
                preset = get_preset(self.options.get("preset_id", ""))
                if preset:
                    look_name = preset.get("name", self.options.get("preset_id", ""))
                    image = apply_preset(image, preset, intensity=self.options.get("intensity", 100))
                else:
                    item.warnings.append("Selected preset not found -- exported without a look applied.")
            # mode == "none": no look change, resize/format/rename only.

            item.applied_look = look_name
            sub_progress(60)

            # PHASE 10 -- Face Restoration batch step. Independent of
            # `mode` above, same convention as Phase 9's upscale step
            # below (see DEFAULT_OPTIONS's comment). Deliberately placed
            # BEFORE the Upscale step: restoring face detail first and
            # then (optionally) upscaling on top gives Upscale's AI
            # detail-recovery pass better source detail to work with on
            # the face than upscaling a not-yet-restored face would.
            if self.options.get("facerestore_enabled"):
                from ai.face_restorer import FaceRestoreError, restore_faces

                def facerestore_sub_progress(pct, label=""):
                    # Maps the AI pass's own 0-100 onto this item's 60-67 slice.
                    if pct is not None and pct >= 0:
                        sub_progress(60 + round(pct * 0.07))

                try:
                    # faces=None -> ai/face_restorer.py auto-detects and
                    # restores EVERY face in this image (see the
                    # DEFAULT_OPTIONS comment above on why batch has no
                    # per-face selection UI). No cancel_event passed --
                    # same "Batch's own cancel/pause is only checked
                    # BETWEEN items, not partway through one image"
                    # reasoning as every other per-image step here.
                    image = restore_faces(
                        image,
                        faces=None,
                        strength=self.options.get("facerestore_strength", 80),
                        natural_detailed=self.options.get("facerestore_natural_detailed", 50),
                        skin_protection=self.options.get("facerestore_skin_protection", 60),
                        on_progress=facerestore_sub_progress,
                    )
                except FaceRestoreError as exc:
                    # Same "one bad file must not kill the batch item if
                    # the Look itself already succeeded" philosophy as
                    # the Upscale step's own except below.
                    item.warnings.append(f"Face Restoration skipped: {exc}")
            sub_progress(67)

            # PHASE 9 -- AI Upscale batch step. Independent of `mode`
            # above (see DEFAULT_OPTIONS's comment): runs on top of
            # whatever Look/Preset (and Face Restoration) was just
            # applied, or on its own with mode="none". Deliberately
            # placed BEFORE _resize_if_needed -- resize_mode is meant to
            # cap large exports back down, so combining "Upscale 2x"
            # with "resize_mode: max_dimension" is the user's own
            # explicit choice to upscale-then-cap, not a bug to work
            # around here.
            if self.options.get("upscale_enabled"):
                from ai.upscaler import UpscaleError, upscale_image

                def upscale_sub_progress(pct, label=""):
                    # Maps the AI pass's own 0-100 onto this item's 67-75 slice.
                    if pct is not None and pct >= 0:
                        sub_progress(67 + round(pct * 0.08))

                try:
                    # No cancel_event passed here -- like every other
                    # per-image step in this method, Batch's own
                    # cancel/pause is only checked BETWEEN items (see
                    # run() below), not partway through one image's
                    # processing. A cancel mid-upscale finishes this one
                    # tile pass and then stops before the next item,
                    # same as it would mid-preset or mid-analyze.
                    image = upscale_image(
                        image,
                        scale=self.options.get("upscale_scale", "2x"),
                        custom_width=self.options.get("upscale_custom_width"),
                        custom_height=self.options.get("upscale_custom_height"),
                        denoise_strength=self.options.get("upscale_denoise_strength", 0),
                        artifact_reduction=self.options.get("upscale_artifact_reduction", 0),
                        face_aware=self.options.get("upscale_face_aware", False),
                        on_progress=upscale_sub_progress,
                    )
                except UpscaleError as exc:
                    # One item's upscale failing shouldn't necessarily
                    # kill the whole batch item if the Look itself
                    # already succeeded -- record it as a warning and
                    # continue with the pre-upscale image, same
                    # "one bad file must not kill the batch" philosophy
                    # as this method's outer except below.
                    item.warnings.append(f"AI Upscale skipped: {exc}")
            sub_progress(75)

            image = self._resize_if_needed(image)
            sub_progress(80)

            if self.options.get("dry_run"):
                # "Preview before commit" -- write the fully-processed
                # result to a private staging area instead of the real
                # output folder, so the user can spot-check it (and the
                # naming template's {index}/{date}/etc still resolve
                # correctly once committed, since the final path is
                # computed fresh at commit time).
                staged = self._staging_path(item)
                self._save(image, staged, original_exif)
                item.staged_path = str(staged)
                item.status = STATUS_PREVIEWED
            else:
                dest = self._build_output_path(item, index, look_name)
                self._save(image, dest, original_exif)
                item.output_path = str(dest)
                item.status = STATUS_DONE

            del image
            gc.collect()  # Phase 14 golden rule: release RAM before the next image

            item.progress = 100
        except Exception as exc:  # noqa: BLE001 -- one bad file must not kill the batch
            item.status = STATUS_FAILED
            item.error = str(exc)
        finally:
            item.finished_at = time.time()
            item.duration_seconds = round(item.finished_at - item.started_at, 2)
            self.log.append(
                {
                    "id": item.id,
                    "name": item.name,
                    "status": item.status,
                    "applied_look": item.applied_look,
                    "output_path": item.output_path or item.staged_path,
                    "error": item.error,
                    "warnings": item.warnings,
                    "duration_seconds": item.duration_seconds,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )

    # ===================== RUN =====================

    def _ram_pressure_relief(self):
        """
        The 'RAM' half of "CPU/RAM throttle control". Best-effort: if
        psutil is available and the machine is genuinely under memory
        pressure (relevant on the 16GB, no-GPU target hardware once
        Phase 9/10's heavier models are also running), force a GC pass
        and a short pause before starting the next image, regardless of
        the user's cpu_throttle slider. Silently does nothing without
        psutil rather than adding it as a hard dependency.
        """
        if psutil is None:
            return
        try:
            percent = psutil.virtual_memory().percent
        except Exception:  # noqa: BLE001
            return
        if percent >= 85:
            gc.collect()
            time.sleep(0.5)

    def run(self, on_progress=None, on_item=None) -> dict:
        """
        Blocking. Runs every included, queued item in order.

        on_progress(overall_percent, label, eta_seconds) -- overall bar.
        on_item(item) -- called after every status/progress change on an
        item, so the frontend queue table can re-render that one row.

        Returns the Batch Analyzer summary dict (spec's example shape):
        {"total", "processed", "failed", "skipped", "categories": {...}}
        """
        with self._lock:
            if self._running:
                raise RuntimeError("A batch is already running.")
            self._running = True
            self._cancel_flag = False
            self._pause_event.set()
            self._run_started_at = time.time()
            self._per_item_durations = []
            queue = [i for i in self.items if i.included and i.status == STATUS_QUEUED]
            total = len(queue)
            _write_recovery_snapshot(self.items, self.options)

        throttle_pct = max(0, min(100, int(self.options.get("cpu_throttle", 0) or 0)))

        def item_progress_cb(item, pct):
            if on_item:
                on_item(item)

        try:
            for idx, item in enumerate(queue, start=1):
                # ----- pause -----
                self._pause_event.wait()
                if self._cancel_flag:
                    item.status = STATUS_CANCELLED
                    if on_item:
                        on_item(item)
                    continue

                self._skip_requested_id = None
                self._ram_pressure_relief()
                self._process_one(item, idx, item_progress_cb=item_progress_cb)

                if self._skip_requested_id == item.id and item.status == STATUS_PROCESSING:
                    item.status = STATUS_SKIPPED

                if item.duration_seconds:
                    self._per_item_durations.append(item.duration_seconds)

                if on_item:
                    on_item(item)

                _write_recovery_snapshot(self.items, self.options)

                # ----- overall progress + ETA -----
                done_count = idx
                overall_pct = round((done_count / total) * 100) if total else 100
                eta = self._estimate_eta(done_count, total)
                if on_progress:
                    on_progress(overall_pct, f"Batch: {done_count}/{total}", eta)

                # ----- CPU/RAM throttle: idle pause between items so the
                # target hardware (i7-7600U, no GPU) stays responsive
                # during a long batch -----
                if throttle_pct and item.duration_seconds:
                    time.sleep(item.duration_seconds * (throttle_pct / 100.0))

                if self._cancel_flag:
                    # Mark any still-queued remainder as cancelled too.
                    for remaining in queue[idx:]:
                        remaining.status = STATUS_CANCELLED
                        if on_item:
                            on_item(remaining)
                    break
        finally:
            self._running = False
            with self._lock:
                still_queued = any(i.status == STATUS_QUEUED for i in self.items)
            if not still_queued:
                discard_recovery_state()

        return self.summary()

    def _estimate_eta(self, done_count: int, total: int):
        if not self._per_item_durations or done_count >= total:
            return 0
        avg = sum(self._per_item_durations) / len(self._per_item_durations)
        return round(avg * (total - done_count))

    # ===================== SUMMARY / REPORTING =====================

    def summary(self) -> dict:
        with self._lock:
            items = list(self.items)
        categories = {}
        for i in items:
            if i.status == STATUS_DONE and i.category:
                categories[i.category] = categories.get(i.category, 0) + 1
        return {
            "total": len(items),
            "included": sum(1 for i in items if i.included),
            "processed": sum(1 for i in items if i.status == STATUS_DONE),
            "previewed": sum(1 for i in items if i.status == STATUS_PREVIEWED),
            "failed": sum(1 for i in items if i.status == STATUS_FAILED),
            "skipped": sum(1 for i in items if i.status == STATUS_SKIPPED),
            "cancelled": sum(1 for i in items if i.status == STATUS_CANCELLED),
            "categories": categories,
            "failed_items": [
                {"id": i.id, "name": i.name, "error": i.error} for i in items if i.status == STATUS_FAILED
            ],
        }

    def state(self) -> dict:
        with self._lock:
            return {
                "items": [i.to_dict() for i in self.items],
                "options": dict(self.options),
                "running": self._running,
                "paused": not self._pause_event.is_set(),
                "summary": self.summary(),
            }

    def export_log(self, dest_path: str) -> str:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "options": self.options,
            "summary": self.summary(),
            "entries": self.log,
        }
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return str(dest)


# One shared queue for the whole app session, same convention as
# core/session.py::get_session() -- the frontend has exactly one Batch
# view, so it needs exactly one controller behind it.
_controller = None


def get_batch_controller() -> BatchController:
    global _controller
    if _controller is None:
        _controller = BatchController()
    return _controller


# ===================== PHASE 14: CRASH RECOVERY =====================
# Added this session -- the queue itself was always purely in-memory
# (by design, same as core/session.py's working-image state), so an
# app crash or forced-close mid-batch silently lost the whole queue
# with no way to pick it back up. This is a small, best-effort JSON
# snapshot written to the OS temp dir (same convention as
# ui/bridge.py's clearCache dirs) -- NOT a live database, just enough
# breadcrumb (source paths + statuses + options) for the frontend to
# ask "resume the N items that hadn't finished yet?" on next launch.
# Never raises -- a recovery feature that can itself crash the app
# defeats its own purpose.

_RECOVERY_FILE = Path(tempfile.gettempdir()) / "pixelforge_batch_recovery.json"


def _write_recovery_snapshot(items: list, options: dict) -> None:
    try:
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "options": options,
            "items": [
                {
                    "source_path": i.source_path,
                    "relative_dir": i.relative_dir,
                    "included": i.included,
                    "status": i.status,
                }
                for i in items
            ],
        }
        with open(_RECOVERY_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        log.warning("Batch crash-recovery snapshot couldn't be written (non-fatal).", exc_info=True)


def get_recovery_state() -> dict:
    """
    Returns the last saved snapshot (only if it still has unfinished
    items in it -- a snapshot left over from a batch that actually
    completed normally is discarded automatically, see
    BatchController.run()'s finally block) or {} if there's nothing to
    recover. Never raises -- a corrupt/unreadable file is treated the
    same as "nothing to recover", not an error.
    """
    if not _RECOVERY_FILE.exists():
        return {}
    try:
        with open(_RECOVERY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        pending = [
            i for i in data.get("items", [])
            if i.get("status") not in (STATUS_DONE, STATUS_SKIPPED, STATUS_CANCELLED)
        ]
        if not pending:
            return {}
        return {"saved_at": data.get("saved_at", ""), "options": data.get("options", {}), "items": pending}
    except (OSError, ValueError):
        return {}


def discard_recovery_state() -> None:
    """Deletes the recovery snapshot -- called once a batch finishes
    with nothing left queued, when the user clears the queue, or when
    the frontend's "resume?" prompt is declined."""
    try:
        if _RECOVERY_FILE.exists():
            _RECOVERY_FILE.unlink()
    except OSError:
        pass