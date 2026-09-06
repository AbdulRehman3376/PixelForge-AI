// PHASE 13: Projects / History (SQLite)
//
// Drives the Projects screen (#view-projects in index.html): grid of
// project cards, search/sort/status filter, a detail overlay with
// Media + Edit History tabs, and the project-level actions (new,
// rename, duplicate, trash/restore, delete forever, export/restore
// backup). All data comes from window.pixelforge.projects.* (see
// frontend/bridge.js), which is a thin pass-through to
// core/database.py via Bridge Slots in ui/bridge.py.
//
// CRASH RECOVERY: on app start, checks getActive() -- if the app was
// closed uncleanly while a project was open (MainWindow.closeEvent
// didn't run to clear the pointer), offers to reopen it.

document.addEventListener("DOMContentLoaded", () => {
    const grid = document.getElementById("projects-grid");
    const emptyState = document.getElementById("projects-empty");
    const emptyText = document.getElementById("projects-empty-text");
    const searchInput = document.getElementById("projects-search");
    const statusFilter = document.getElementById("projects-status-filter");
    const sortSelect = document.getElementById("projects-sort");
    const btnNew = document.getElementById("btn-projects-new");
    const btnImportBackup = document.getElementById("btn-projects-import-backup");

    const detailOverlay = document.getElementById("projects-detail-overlay");
    const detailName = document.getElementById("projects-detail-name");
    const detailMeta = document.getElementById("projects-detail-meta");
    const detailClose = document.getElementById("btn-projects-detail-close");
    const detailOpen = document.getElementById("btn-projects-detail-open");
    const detailDuplicate = document.getElementById("btn-projects-detail-duplicate");
    const detailBackup = document.getElementById("btn-projects-detail-backup");
    const detailTrash = document.getElementById("btn-projects-detail-trash");
    const detailRestore = document.getElementById("btn-projects-detail-restore");
    const detailDelete = document.getElementById("btn-projects-detail-delete");
    const detailHint = document.getElementById("projects-detail-hint");
    const detailTabs = document.querySelectorAll(".projects-detail-tab");
    const detailMedia = document.getElementById("projects-detail-media");
    const detailHistory = document.getElementById("projects-detail-history");

    if (!grid) return; // Projects markup not present on this build -- no-op.

    let currentProjectId = null;
    let currentProjectStatus = null;
    let searchDebounce = null;
    // BUGFIX: pressing Enter in the rename field used to only blur the
    // input (which saves the name via the "change" listener below) --
    // the detail card itself never closed, so it looked like Enter did
    // nothing. This flag, set only on the Enter path (not on a plain
    // click-away blur), tells the rename handler to also close the
    // card once the save actually completes.
    let closeAfterRename = false;

    // ----- Open a project's picture back up in the editor -----
    //
    // This was the actual missing piece: the detail card only ever
    // showed the project's name/media list as text -- there was no way
    // to get a project's picture back into Enhance to keep working on
    // it, and "+ New Project" only created an empty database row with
    // nothing inside it. openProjectInEditor() is the fix for both:
    // called after creating a fresh project (no media yet -- goes
    // straight to the file picker) and from "Open in Editor" / a media
    // row on an existing project (loads its most recent picture).
    function openProjectInEditor(projectId, mediaPathOverride) {
        window.pixelforge.projects.setActive(projectId);
        if (window.pixelforgeSetActiveDbProject) window.pixelforgeSetActiveDbProject(projectId);

        const launch = (path) => {
            closeDetail();
            if (window.pixelforgeShowView) window.pixelforgeShowView("enhance");
            if (path && window.pixelforgeOpenImage) window.pixelforgeOpenImage(path);
        };

        if (mediaPathOverride) {
            launch(mediaPathOverride);
            return;
        }

        window.pixelforge.projects.listMedia(projectId, (res) => {
            const items = res.ok ? res.data : [];
            if (items.length) {
                // Most recently added picture in this project.
                const latest = items[items.length - 1];
                launch(latest.output_path || latest.input_path);
            } else {
                // Nothing added to this project yet -- let the user pick
                // a picture now, same as Import/Open Image elsewhere.
                closeDetail();
                if (!window.pixelforge.openImageDialog) return;
                window.pixelforge.openImageDialog((path) => {
                    if (path && window.pixelforgeOpenImage) window.pixelforgeOpenImage(path);
                });
            }
        });
    }

    // BUGFIX: project card thumbnails weren't loading (broken-image
    // icon shown instead of the picture) because this file built its
    // file:// URL differently from every other view in the app --
    // editor.js/removebg.js/filters.js/upscale.js/restore.js/video.js/
    // batch.js all normalize backslashes to forward slashes first
    // (Windows paths like "C:\Users\...\photo.jpg" aren't valid inside
    // a file:// URL as-is). This file skipped that step, so thumbnails
    // only ever worked by accident on paths that happened to already
    // use forward slashes. Reuses the same convention as those files.
    function toFileUrl(path) {
        const normalized = String(path || "").replace(/\\/g, "/").replace(/^\/+/, "");
        return "file:///" + encodeURI(normalized);
    }

    function fmtDate(iso) {
        if (!iso) return "";
        try {
            const d = new Date(iso);
            return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
        } catch (e) {
            return iso;
        }
    }

    function escapeHtml(s) {
        return String(s ?? "").replace(/[&<>"']/g, (c) => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
        }[c]));
    }

    // ----- Grid rendering -----

    function renderGrid(projects) {
        grid.innerHTML = "";
        if (!projects || projects.length === 0) {
            emptyState.classList.remove("view--hidden");
            emptyText.textContent = statusFilter.value === "trashed"
                ? "Trash is empty"
                : "No projects yet";
            return;
        }
        emptyState.classList.add("view--hidden");

        projects.forEach((p) => {
            const isTrashed = p.status === "trashed";
            const card = document.createElement("div");
            card.className = "project-card";
            card.dataset.id = p.id;
            // Focusable so the keyboard Delete/Backspace handler below
            // has something to act on without opening the detail card
            // first -- previously the only way to delete anything was
            // to click into the overlay.
            card.tabIndex = 0;
            card.innerHTML = `
                <div class="project-card-quick-actions">
                    ${isTrashed ? "" : `<button class="btn btn-icon project-card-quick-open" title="Open in Editor" aria-label="Open in Editor">▶</button>`}
                    <button class="btn btn-icon project-card-quick-delete" title="${isTrashed ? "Delete Forever" : "Move to Trash"}" aria-label="${isTrashed ? "Delete Forever" : "Move to Trash"}">🗑</button>
                </div>
                <div class="project-card-thumb">
                    ${p.thumbnail_path
                        ? `<img src="${toFileUrl(p.thumbnail_path)}" alt="">`
                        : "🖼"}
                </div>
                <div class="project-card-body">
                    <div class="project-card-name">${escapeHtml(p.name)}</div>
                    <div class="project-card-meta">Updated ${fmtDate(p.updated_at)}</div>
                </div>
            `;
            card.addEventListener("click", () => openDetail(p.id));
            card.addEventListener("keydown", (e) => {
                if (e.key === "Delete" || e.key === "Backspace") {
                    e.preventDefault();
                    quickDeleteProject(p.id, isTrashed);
                } else if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    openDetail(p.id);
                }
            });
            const quickOpen = card.querySelector(".project-card-quick-open");
            if (quickOpen) {
                quickOpen.addEventListener("click", (e) => {
                    e.stopPropagation();
                    openProjectInEditor(p.id);
                });
            }
            card.querySelector(".project-card-quick-delete").addEventListener("click", (e) => {
                e.stopPropagation();
                quickDeleteProject(p.id, isTrashed);
            });
            grid.appendChild(card);
        });
    }

    // Shared by the card's trash icon and its keyboard Delete/Backspace
    // handler -- trash is reversible so it doesn't need a confirm;
    // permanent delete (only reachable once a project is already in
    // Trash) still does.
    function quickDeleteProject(projectId, isTrashed) {
        if (isTrashed) {
            if (!window.confirm("Delete this project forever? This cannot be undone.")) return;
            window.pixelforge.projects.deleteForever(projectId, () => refreshGrid());
        } else {
            window.pixelforge.projects.trash(projectId, () => refreshGrid());
        }
    }

    function refreshGrid() {
        window.pixelforge.projects.list(
            statusFilter.value,
            searchInput.value,
            sortSelect.value,
            "desc",
            0,
            (res) => renderGrid(res.ok ? res.data : [])
        );
    }

    // ----- Detail overlay -----

    function openDetail(projectId) {
        currentProjectId = projectId;
        window.pixelforge.projects.get(projectId, (res) => {
            if (!res.ok || !res.data) return;
            const p = res.data;
            detailName.value = p.name;
            detailMeta.textContent = `Created ${fmtDate(p.created_at)} · Updated ${fmtDate(p.updated_at)} · Status: ${p.status}`;
            const isTrashed = p.status === "trashed";
            currentProjectStatus = p.status;
            detailOpen.classList.toggle("view--hidden", isTrashed);
            detailTrash.classList.toggle("view--hidden", isTrashed);
            detailRestore.classList.toggle("view--hidden", !isTrashed);
            detailDelete.classList.toggle("view--hidden", !isTrashed);
            if (detailHint) {
                detailHint.textContent = isTrashed
                    ? "Restore this project before opening it in the editor."
                    : "Tip: the Delete key trashes this project (or deletes it forever once already in Trash).";
            }
            detailOverlay.classList.remove("view--hidden");
            loadMediaTab(projectId);
            loadHistoryTab(projectId);
        });
    }

    function closeDetail() {
        detailOverlay.classList.add("view--hidden");
        currentProjectId = null;
        currentProjectStatus = null;
        closeAfterRename = false;
    }

    function loadMediaTab(projectId) {
        window.pixelforge.projects.listMedia(projectId, (res) => {
            const items = res.ok ? res.data : [];
            if (!items.length) {
                detailMedia.innerHTML = `<div class="projects-empty-inline">No media added to this project yet. Use "Open in Editor" to add one.</div>`;
                return;
            }
            detailMedia.innerHTML = items.map((m) => `
                <div class="projects-media-row" data-media-id="${m.id}" title="Click to continue editing this picture">
                    <span class="projects-media-row-path" title="${escapeHtml(m.input_path)}">${escapeHtml(m.input_path)}</span>
                    <span class="badge badge--muted">${escapeHtml(m.media_type)}</span>
                </div>
            `).join("");
            // Clicking a specific picture continues editing THAT one,
            // instead of always jumping to whichever was added last.
            detailMedia.querySelectorAll(".projects-media-row").forEach((row) => {
                row.addEventListener("click", () => {
                    const m = items.find((it) => it.id === row.dataset.mediaId);
                    if (m && currentProjectStatus !== "trashed") {
                        openProjectInEditor(projectId, m.output_path || m.input_path);
                    }
                });
            });
        });
    }

    function loadHistoryTab(projectId) {
        window.pixelforge.projects.editHistory(projectId, (res) => {
            const items = res.ok ? res.data : [];
            if (!items.length) {
                detailHistory.innerHTML = `<div class="projects-empty-inline">No edit history recorded yet.</div>`;
                return;
            }
            detailHistory.innerHTML = items.map((e) => {
                const desc = e.preset ? `Preset: ${e.preset}` : (e.prompt ? `Prompt: ${e.prompt}` : "Edit");
                return `
                    <div class="projects-history-row">
                        <span class="projects-history-row-desc" title="${escapeHtml(e.input_path || "")}">${escapeHtml(desc)}</span>
                        <span class="projects-history-row-time">${fmtDate(e.created_at)}</span>
                    </div>
                `;
            }).join("");
        });
    }

    detailTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            detailTabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            const which = tab.dataset.tab;
            detailMedia.classList.toggle("view--hidden", which !== "media");
            detailHistory.classList.toggle("view--hidden", which !== "history");
        });
    });

    detailClose.addEventListener("click", closeDetail);

    detailName.addEventListener("change", () => {
        if (!currentProjectId) return;
        const shouldClose = closeAfterRename;
        closeAfterRename = false;
        window.pixelforge.projects.rename(currentProjectId, detailName.value, () => {
            refreshGrid();
            // Only Enter closes the card -- a plain click-away blur
            // should just save the name and leave the card open, since
            // the user may still want to look at Media/Edit History.
            if (shouldClose) closeDetail();
        });
    });

    // Enter key: pressing Enter didn't do anything before, because the
    // rename above only fires on the input's "change" event (which only
    // fires on blur, i.e. clicking away) -- not on Enter by itself.
    // This blurs the field on Enter, which then triggers that same
    // "change" -> rename flow, so Enter now saves the name immediately
    // instead of requiring a click elsewhere first. It also now closes
    // the card once that save completes (see closeAfterRename above) --
    // previously Enter looked like it did nothing at all, because the
    // card just sat there open with no visible change.
    detailName.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            closeAfterRename = true;
            detailName.blur();
        }
    });

    detailOpen.addEventListener("click", () => {
        if (!currentProjectId) return;
        openProjectInEditor(currentProjectId);
    });

    // Keyboard Delete/Backspace on the open detail card -- previously
    // the physical Delete key did nothing at all here; only clicking
    // "Move to Trash" / "Delete Forever" worked. Skipped while the
    // rename field (or anything else typeable) has focus so it doesn't
    // hijack normal text editing.
    document.addEventListener("keydown", (e) => {
        if (detailOverlay.classList.contains("view--hidden")) return;
        if (!currentProjectId) return;
        const tag = document.activeElement && document.activeElement.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        if (e.key !== "Delete" && e.key !== "Backspace") return;
        e.preventDefault();
        if (currentProjectStatus === "trashed") {
            detailDelete.click();
        } else {
            detailTrash.click();
        }
    });

    detailDuplicate.addEventListener("click", () => {
        if (!currentProjectId) return;
        window.pixelforge.projects.duplicate(currentProjectId, (res) => {
            closeDetail();
            refreshGrid();
        });
    });

    detailBackup.addEventListener("click", () => {
        if (!currentProjectId) return;
        window.pixelforge.projects.exportBackup(currentProjectId, (res) => {
            if (window.pixelforge && window.pixelforge.bridge) {
                // No dedicated toast system in this build -- reuse the
                // existing global status text the same way other views do.
            }
        });
    });

    detailTrash.addEventListener("click", () => {
        if (!currentProjectId) return;
        window.pixelforge.projects.trash(currentProjectId, () => {
            closeDetail();
            refreshGrid();
        });
    });

    detailRestore.addEventListener("click", () => {
        if (!currentProjectId) return;
        window.pixelforge.projects.restore(currentProjectId, () => {
            closeDetail();
            refreshGrid();
        });
    });

    detailDelete.addEventListener("click", () => {
        if (!currentProjectId) return;
        if (!window.confirm("Delete this project forever? This cannot be undone.")) return;
        window.pixelforge.projects.deleteForever(currentProjectId, () => {
            closeDetail();
            refreshGrid();
        });
    });

    // ----- Toolbar -----

    btnNew.addEventListener("click", () => {
        const name = window.prompt("Project name:", "Untitled Project");
        if (name === null) return; // cancelled
        // BUGFIX: this used to just create an empty database row and
        // refresh the grid -- there was no way to actually get a
        // picture into the new project, so it sat there forever as
        // just a name with nothing inside it. Now it immediately hands
        // off to openProjectInEditor(), which (for a brand-new,
        // media-less project) opens the file picker so the user can
        // pick the project's first picture right away.
        window.pixelforge.projects.create(name, (res) => {
            refreshGrid();
            if (res.ok && res.data) openProjectInEditor(res.data.id);
        });
    });

    btnImportBackup.addEventListener("click", () => {
        window.pixelforge.projects.importBackup((res) => refreshGrid());
    });

    searchInput.addEventListener("input", () => {
        clearTimeout(searchDebounce);
        searchDebounce = setTimeout(refreshGrid, 250);
    });
    statusFilter.addEventListener("change", refreshGrid);
    sortSelect.addEventListener("change", refreshGrid);

    // ----- View-shown hook (see ui.js's showView) -----
    document.addEventListener("pixelforge:viewshown", (e) => {
        if (e.detail && e.detail.view === "projects") refreshGrid();
    });

    // ----- Crash recovery prompt -----
    window.onPixelforgeReady(() => {
        window.pixelforge.projects.getActive((res) => {
            if (!res.ok || !res.data) return;
            const p = res.data;
            const reopen = window.confirm(
                `PixelForge didn't close cleanly last time. Reopen "${p.name}"?`
            );
            if (reopen && typeof window.pixelforgeShowView === "function") {
                window.pixelforgeShowView("projects");
                openDetail(p.id);
            } else {
                window.pixelforge.projects.clearActive();
            }
        });
    });
});