# core/database.py
#
# PHASE 13 -- Projects / History (SQLite)
#
# WHY THIS EXISTS: every earlier phase (Enhance, Filters, Remove BG,
# Video Studio, etc.) only knows about ONE image/video at a time via
# core/session.py's in-memory working session. There has never been a
# durable, browsable record of "what has this user made" -- no way to
# reopen yesterday's project, no recents list, no per-project edit
# history distinct from the app-wide undo/redo stack. This module is
# that durable layer.
#
# DESIGN NOTES (matches the rest of the codebase's conventions):
#   - Plain local SQLite file at database/pixelforge.db (next to the
#     existing empty `database/` folder from the original scaffold),
#     same "no cloud, no mandatory service" principle as everything
#     else in this project.
#   - One shared, lazily-created connection per process, guarded by a
#     threading.Lock -- Bridge Slots and _run_async background workers
#     both call into this module, and sqlite3 connections are not
#     safe to share across threads without serializing access
#     ourselves (check_same_thread=False + our own lock).
#   - Every public function returns plain dict / list[dict] (already
#     JSON-serializable) so ui/bridge.py Slots can hand results
#     straight to `json.dumps` or to `_run_async`'s taskResult
#     signal, exactly like core/settings.py and core/filters.py do.
#   - Writes are wrapped in explicit transactions; reads never raise
#     on a missing row -- callers get None / [] and decide what that
#     means, same "reads never raise" spirit as core/settings.py.
#   - Foreign keys are ON (PRAGMA foreign_keys = ON) so deleting a
#     project cascades to its media/edits/video_projects rows instead
#     of leaving orphans behind.
#   - `parameters_json` columns store arbitrary preset/edit parameters
#     as TEXT (json.dumps'd) rather than a rigid schema, since
#     different tools (Enhance sliders, Filters, prompt-based tools)
#     have very different parameter shapes.
#
# PRIVACY: this database holds only local file paths, dimensions,
# timestamps, and parameter values the user themselves set inside the
# app -- no image pixel data, no telemetry, nothing that leaves the
# machine. It lives entirely under the project's own `database/`
# folder.

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

_DB_DIR = Path(__file__).resolve().parent.parent / "database"
_DB_PATH = _DB_DIR / "pixelforge.db"

_lock = threading.Lock()
_conn = None  # lazily created, see _get_conn()

# Valid `status` values for a project row. Kept as a tuple (not an
# enum) so it round-trips through JSON/JS without extra glue, same
# pattern as core/video_studio.py's string-based state fields.
PROJECT_STATUSES = ("active", "archived", "trashed")
MEDIA_TYPES = ("image", "video")


# ------------------------------------------------------------------
# Connection / schema
# ------------------------------------------------------------------

def _get_conn():
    global _conn
    if _conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA foreign_keys = ON")
        _conn.execute("PRAGMA journal_mode = WAL")  # safer under app crashes
        _init_schema(_conn)
    return _conn


def _init_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            thumbnail_path  TEXT,
            status          TEXT NOT NULL DEFAULT 'active'
        );

        CREATE TABLE IF NOT EXISTS media (
            id           TEXT PRIMARY KEY,
            project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            input_path   TEXT NOT NULL,
            output_path  TEXT,
            media_type   TEXT NOT NULL DEFAULT 'image',
            width        INTEGER,
            height       INTEGER,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS edits (
            id               TEXT PRIMARY KEY,
            media_id         TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
            preset           TEXT,
            prompt           TEXT,
            parameters_json  TEXT,
            created_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS video_projects (
            id           TEXT PRIMARY KEY,
            project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            duration     REAL,
            resolution   TEXT,
            fps          REAL,
            output_path  TEXT,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS presets (
            id               TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            category         TEXT,
            parameters_json  TEXT,
            created_at       TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_media_project      ON media(project_id);
        CREATE INDEX IF NOT EXISTS idx_edits_media         ON edits(media_id);
        CREATE INDEX IF NOT EXISTS idx_video_projects_proj ON video_projects(project_id);
        CREATE INDEX IF NOT EXISTS idx_projects_status     ON projects(status);
        CREATE INDEX IF NOT EXISTS idx_projects_updated    ON projects(updated_at);
        """
    )
    conn.commit()


def _now() -> str:
    # ISO-8601 UTC, sortable as plain text -- matches how the rest of
    # the app already timestamps things (see core/session.py).
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_id() -> str:
    return uuid.uuid4().hex


def _row_to_dict(row) -> dict:
    return dict(row) if row is not None else None


def db_path() -> str:
    return str(_DB_PATH)


# ------------------------------------------------------------------
# PROJECTS
# ------------------------------------------------------------------

def create_project(name: str, thumbnail_path: str = None) -> dict:
    name = (name or "").strip() or "Untitled Project"
    conn = _get_conn()
    project_id = _new_id()
    ts = _now()
    with _lock:
        conn.execute(
            "INSERT INTO projects (id, name, created_at, updated_at, "
            "thumbnail_path, status) VALUES (?, ?, ?, ?, ?, 'active')",
            (project_id, name, ts, ts, thumbnail_path),
        )
        conn.commit()
    return get_project(project_id)


def get_project(project_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    return _row_to_dict(row)


def update_project(project_id: str, **fields) -> dict:
    """Updates any subset of name/thumbnail_path/status; always bumps
    updated_at. Silently ignores unknown field names so callers can
    pass a loosely-shaped dict from the frontend without extra
    filtering on the JS side."""
    allowed = {"name", "thumbnail_path", "status"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if "status" in updates and updates["status"] not in PROJECT_STATUSES:
        raise ValueError(f"Invalid status: {updates['status']!r}")
    if not updates:
        return get_project(project_id)
    updates["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    conn = _get_conn()
    with _lock:
        conn.execute(
            f"UPDATE projects SET {set_clause} WHERE id = ?",
            (*updates.values(), project_id),
        )
        conn.commit()
    return get_project(project_id)


def touch_project(project_id: str) -> None:
    """Bumps updated_at only -- called whenever media/edits are added
    under a project, so 'recent projects' reflects real activity, not
    just renames."""
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (_now(), project_id),
        )
        conn.commit()


def delete_project(project_id: str) -> bool:
    """Hard delete -- cascades to media/edits/video_projects via the
    FK ON DELETE CASCADE declared in the schema. Use
    update_project(id, status='trashed') instead for a soft/reversible
    delete (see restore_project)."""
    conn = _get_conn()
    with _lock:
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
    return cur.rowcount > 0


def trash_project(project_id: str) -> dict:
    """Soft delete -- moves the project to the 'trashed' status instead
    of removing rows, so restore_project() can bring it back."""
    return update_project(project_id, status="trashed")


def restore_project(project_id: str) -> dict:
    """Un-deletes a trashed project back to 'active'."""
    return update_project(project_id, status="active")


def duplicate_project(project_id: str, new_name: str = None) -> dict:
    """Deep-copies a project row plus its media, edits, and
    video_projects rows under fresh ids. Does NOT copy files on disk
    (input_path/output_path values are duplicated as references to the
    same files) -- that keeps duplication instant and disk-cheap;
    Export/Save Project already exists for producing an independent
    copy of the actual media."""
    original = get_project(project_id)
    if original is None:
        raise ValueError(f"Project not found: {project_id}")

    conn = _get_conn()
    new_id = _new_id()
    ts = _now()
    name = new_name or f"{original['name']} (Copy)"

    with _lock:
        conn.execute(
            "INSERT INTO projects (id, name, created_at, updated_at, "
            "thumbnail_path, status) VALUES (?, ?, ?, ?, ?, 'active')",
            (new_id, name, ts, ts, original["thumbnail_path"]),
        )

        media_rows = conn.execute(
            "SELECT * FROM media WHERE project_id = ?", (project_id,)
        ).fetchall()
        media_id_map = {}
        for m in media_rows:
            new_media_id = _new_id()
            media_id_map[m["id"]] = new_media_id
            conn.execute(
                "INSERT INTO media (id, project_id, input_path, output_path, "
                "media_type, width, height, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_media_id, new_id, m["input_path"], m["output_path"],
                    m["media_type"], m["width"], m["height"], ts,
                ),
            )

        edit_rows = conn.execute(
            "SELECT e.* FROM edits e JOIN media m ON e.media_id = m.id "
            "WHERE m.project_id = ?",
            (project_id,),
        ).fetchall()
        for e in edit_rows:
            new_media_id = media_id_map.get(e["media_id"])
            if new_media_id is None:
                continue  # orphaned edit row, skip rather than fail the whole copy
            conn.execute(
                "INSERT INTO edits (id, media_id, preset, prompt, "
                "parameters_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (_new_id(), new_media_id, e["preset"], e["prompt"],
                 e["parameters_json"], ts),
            )

        vp_rows = conn.execute(
            "SELECT * FROM video_projects WHERE project_id = ?", (project_id,)
        ).fetchall()
        for vp in vp_rows:
            conn.execute(
                "INSERT INTO video_projects (id, project_id, duration, "
                "resolution, fps, output_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_new_id(), new_id, vp["duration"], vp["resolution"],
                 vp["fps"], vp["output_path"], ts),
            )

        conn.commit()

    return get_project(new_id)


def list_projects(
    status: str = "active",
    search: str = None,
    sort_by: str = "updated_at",
    sort_dir: str = "desc",
    limit: int = None,
) -> list:
    """Main Projects-screen query: filter by status (None = all
    statuses), optional case-insensitive name search, sortable by
    name/created_at/updated_at, optional limit (e.g. limit=10 for a
    'Recent Projects' rail on Home)."""
    sort_columns = {"name", "created_at", "updated_at"}
    if sort_by not in sort_columns:
        sort_by = "updated_at"
    sort_dir = "ASC" if str(sort_dir).lower() == "asc" else "DESC"

    clauses, params = [], []
    if status:
        if status not in PROJECT_STATUSES:
            raise ValueError(f"Invalid status: {status!r}")
        clauses.append("status = ?")
        params.append(status)
    if search:
        clauses.append("name LIKE ? COLLATE NOCASE")
        params.append(f"%{search.strip()}%")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = ""
    if limit:
        limit_sql = "LIMIT ?"
        params.append(int(limit))

    conn = _get_conn()
    rows = conn.execute(
        f"SELECT * FROM projects {where} ORDER BY {sort_by} {sort_dir} {limit_sql}",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def recent_projects(limit: int = 8) -> list:
    return list_projects(status="active", sort_by="updated_at",
                          sort_dir="desc", limit=limit)


# ------------------------------------------------------------------
# MEDIA
# ------------------------------------------------------------------

def add_media(
    project_id: str,
    input_path: str,
    output_path: str = None,
    media_type: str = "image",
    width: int = None,
    height: int = None,
) -> dict:
    if media_type not in MEDIA_TYPES:
        raise ValueError(f"Invalid media_type: {media_type!r}")
    if get_project(project_id) is None:
        raise ValueError(f"Project not found: {project_id}")

    conn = _get_conn()
    media_id = _new_id()
    ts = _now()
    with _lock:
        conn.execute(
            "INSERT INTO media (id, project_id, input_path, output_path, "
            "media_type, width, height, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (media_id, project_id, input_path, output_path, media_type,
             width, height, ts),
        )
        conn.commit()
    touch_project(project_id)
    return get_media(media_id)


def get_media(media_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
    return _row_to_dict(row)


def update_media(media_id: str, **fields) -> dict:
    allowed = {"output_path", "width", "height"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_media(media_id)
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    conn = _get_conn()
    with _lock:
        conn.execute(
            f"UPDATE media SET {set_clause} WHERE id = ?",
            (*updates.values(), media_id),
        )
        conn.commit()
    media = get_media(media_id)
    if media:
        touch_project(media["project_id"])
    return media


def list_media(project_id: str) -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM media WHERE project_id = ? ORDER BY created_at ASC",
        (project_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_media(media_id: str) -> bool:
    conn = _get_conn()
    with _lock:
        cur = conn.execute("DELETE FROM media WHERE id = ?", (media_id,))
        conn.commit()
    return cur.rowcount > 0


# ------------------------------------------------------------------
# EDITS  (per-media edit history -- distinct from the app-wide
# undo/redo stack in core/session.py; this is a durable, browsable
# log of "what was done to this file", not an in-memory stack)
# ------------------------------------------------------------------

def add_edit(media_id: str, preset: str = None, prompt: str = None,
             parameters: dict = None) -> dict:
    if get_media(media_id) is None:
        raise ValueError(f"Media not found: {media_id}")
    conn = _get_conn()
    edit_id = _new_id()
    ts = _now()
    params_json = json.dumps(parameters) if parameters is not None else None
    with _lock:
        conn.execute(
            "INSERT INTO edits (id, media_id, preset, prompt, "
            "parameters_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (edit_id, media_id, preset, prompt, params_json, ts),
        )
        conn.commit()
    # BUGFIX: add_media()/add_video_project() both call touch_project()
    # so "recent projects" reflects real activity, but add_edit() -- the
    # single most common write once every editing tool is actually wired
    # to the database (see ui/bridge.py's _record_edit) -- never did.
    # Without this, exporting the same photo through Enhance five times
    # in a row would leave the project's updated_at frozen at whenever
    # its first media row was added, so it would silently drop out of a
    # "sort by recently updated" view despite being worked on seconds ago.
    media = get_media(media_id)
    if media:
        touch_project(media["project_id"])
    return get_edit(edit_id)


def get_edit(edit_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM edits WHERE id = ?", (edit_id,)).fetchone()
    d = _row_to_dict(row)
    if d and d.get("parameters_json"):
        d["parameters"] = json.loads(d["parameters_json"])
    return d


def list_edits(media_id: str) -> list:
    """Full per-media edit history, oldest first -- what the Projects
    screen shows as 'History' for a given file."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM edits WHERE media_id = ? ORDER BY created_at ASC",
        (media_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("parameters_json"):
            d["parameters"] = json.loads(d["parameters_json"])
        out.append(d)
    return out


def project_edit_history(project_id: str) -> list:
    """All edits across every media item in a project, newest first --
    the 'per-project edit history' feature from the Missing
    Professional Features list, joined with the media row so the
    frontend has enough context (input_path) to display each entry
    without a second round trip."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT e.*, m.input_path, m.media_type FROM edits e "
        "JOIN media m ON e.media_id = m.id "
        "WHERE m.project_id = ? ORDER BY e.created_at DESC",
        (project_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("parameters_json"):
            d["parameters"] = json.loads(d["parameters_json"])
        out.append(d)
    return out


# ------------------------------------------------------------------
# VIDEO PROJECTS
# ------------------------------------------------------------------

def add_video_project(project_id: str, duration: float = None,
                       resolution: str = None, fps: float = None,
                       output_path: str = None) -> dict:
    if get_project(project_id) is None:
        raise ValueError(f"Project not found: {project_id}")
    conn = _get_conn()
    vp_id = _new_id()
    ts = _now()
    with _lock:
        conn.execute(
            "INSERT INTO video_projects (id, project_id, duration, "
            "resolution, fps, output_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (vp_id, project_id, duration, resolution, fps, output_path, ts),
        )
        conn.commit()
    touch_project(project_id)
    return get_video_project(vp_id)


def get_video_project(vp_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM video_projects WHERE id = ?", (vp_id,)
    ).fetchone()
    return _row_to_dict(row)


def list_video_projects(project_id: str) -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM video_projects WHERE project_id = ? "
        "ORDER BY created_at DESC",
        (project_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# PRESETS  (user-saved parameter sets -- separate from the built-in
# Filters presets already shipped by core/filters.py; this table is
# for cross-tool saved presets, e.g. a named Enhance slider snapshot)
# ------------------------------------------------------------------

def save_preset(name: str, category: str = None, parameters: dict = None) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("Preset name can't be empty.")
    conn = _get_conn()
    preset_id = _new_id()
    ts = _now()
    params_json = json.dumps(parameters) if parameters is not None else None
    with _lock:
        conn.execute(
            "INSERT INTO presets (id, name, category, parameters_json, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (preset_id, name, category, params_json, ts),
        )
        conn.commit()
    return get_preset(preset_id)


def get_preset(preset_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM presets WHERE id = ?", (preset_id,)).fetchone()
    d = _row_to_dict(row)
    if d and d.get("parameters_json"):
        d["parameters"] = json.loads(d["parameters_json"])
    return d


def list_presets(category: str = None) -> list:
    conn = _get_conn()
    if category:
        rows = conn.execute(
            "SELECT * FROM presets WHERE category = ? ORDER BY name ASC",
            (category,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM presets ORDER BY name ASC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("parameters_json"):
            d["parameters"] = json.loads(d["parameters_json"])
        out.append(d)
    return out


def delete_preset(preset_id: str) -> bool:
    conn = _get_conn()
    with _lock:
        cur = conn.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
        conn.commit()
    return cur.rowcount > 0


# ------------------------------------------------------------------
# BACKUP / RESTORE  (whole-database export/import -- the
# "Export/backup project" + "Restore project" features. Backs up the
# *database records*, i.e. project/media/edit/preset metadata; actual
# media files on disk are backed up separately via each tool's own
# Export, same division of responsibility as core/session.py's
# .pfproj files vs exported images.)
# ------------------------------------------------------------------

def export_project_backup(project_id: str) -> dict:
    """Returns a single JSON-serializable dict containing the project
    row plus every media/edits/video_projects row beneath it -- the
    caller (ui/bridge.py) writes this to a user-chosen .json file via
    a native Save dialog, mirroring how core/session.py hands off
    Save Project to the bridge rather than touching QFileDialog
    itself."""
    project = get_project(project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")

    media_list = list_media(project_id)
    for m in media_list:
        m["edits"] = list_edits(m["id"])
    video_projects = list_video_projects(project_id)

    return {
        "pixelforge_backup_version": 1,
        "exported_at": _now(),
        "project": project,
        "media": media_list,
        "video_projects": video_projects,
    }


def import_project_backup(backup: dict, as_new: bool = True) -> dict:
    """Restores a backup dict produced by export_project_backup().
    as_new=True (default, and the only mode that's safe against id
    collisions with an existing local database) always creates a
    fresh project + fresh media/edit/video_project ids, preserving
    the original name with a '(Restored)' suffix, same spirit as
    duplicate_project(). Does not restore actual media files -- only
    the database records; the referenced input_path/output_path
    values are carried over as-is and may or may not still exist on
    this machine."""
    if not isinstance(backup, dict) or "project" not in backup:
        raise ValueError("Not a valid PixelForge project backup.")

    src_project = backup["project"]
    conn = _get_conn()
    new_project_id = _new_id()
    ts = _now()
    name = f"{src_project.get('name', 'Untitled Project')} (Restored)"

    with _lock:
        conn.execute(
            "INSERT INTO projects (id, name, created_at, updated_at, "
            "thumbnail_path, status) VALUES (?, ?, ?, ?, ?, 'active')",
            (new_project_id, name, ts, ts, src_project.get("thumbnail_path")),
        )

        for m in backup.get("media", []):
            new_media_id = _new_id()
            conn.execute(
                "INSERT INTO media (id, project_id, input_path, output_path, "
                "media_type, width, height, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_media_id, new_project_id, m.get("input_path"),
                    m.get("output_path"), m.get("media_type", "image"),
                    m.get("width"), m.get("height"), ts,
                ),
            )
            for e in m.get("edits", []):
                params = e.get("parameters")
                params_json = json.dumps(params) if params is not None else e.get("parameters_json")
                conn.execute(
                    "INSERT INTO edits (id, media_id, preset, prompt, "
                    "parameters_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (_new_id(), new_media_id, e.get("preset"), e.get("prompt"),
                     params_json, ts),
                )

        for vp in backup.get("video_projects", []):
            conn.execute(
                "INSERT INTO video_projects (id, project_id, duration, "
                "resolution, fps, output_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_new_id(), new_project_id, vp.get("duration"),
                 vp.get("resolution"), vp.get("fps"), vp.get("output_path"), ts),
            )

        conn.commit()

    return get_project(new_project_id)


# ------------------------------------------------------------------
# AUTOSAVE / CRASH RECOVERY
# ------------------------------------------------------------------
#
# Deliberately thin: core/session.py already owns the *content*
# autosave (the working image + adjustment state on disk for the tool
# currently open). What's missing at the database layer is simply
# knowing "was a project actively being worked on when the app last
# closed, so we can offer to reopen it" -- a single key/value pointer,
# not a duplicate copy of session.py's job.

def set_active_project(project_id: str) -> None:
    """Records the project currently open, for crash-recovery on next
    launch. Stored as a normal preset row under a reserved category
    ('__system__') so no schema migration is needed for one pointer
    value."""
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM presets WHERE category = '__system__' "
                     "AND name = 'active_project'")
        conn.execute(
            "INSERT INTO presets (id, name, category, parameters_json, "
            "created_at) VALUES (?, 'active_project', '__system__', ?, ?)",
            (_new_id(), json.dumps({"project_id": project_id}), _now()),
        )
        conn.commit()


def get_active_project() -> dict:
    """Returns the last-active project (for a 'Recover unsaved
    project?' prompt on launch), or None if there isn't one / it was
    since deleted."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM presets WHERE category = '__system__' "
        "AND name = 'active_project' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(row["parameters_json"] or "{}")
    project_id = payload.get("project_id")
    if not project_id:
        return None
    return get_project(project_id)


def clear_active_project() -> None:
    """Called on clean app shutdown -- an unclaimed pointer left behind
    on next launch is exactly the crash-recovery signal."""
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM presets WHERE category = '__system__' "
                     "AND name = 'active_project'")
        conn.commit()


def close() -> None:
    """Closes the shared connection -- call on clean app shutdown from
    main.py/main_window.py, mirroring how core/session.py cleans up
    stale sessions on exit."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None