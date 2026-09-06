// frontend/session.js
//
// PHASE 6 -- Shared Working Image / Edit Session (frontend half).
//
// THE BUG THIS FIXES
// ------------------
// "Filter lagaya -> Enhance khola -> original image wapas aa gayi."
//
// Every view used to keep its own private idea of "the current image",
// and ui.js only handed an image across when the target view was still
// EMPTY (its dropzone visible). So a view that already had a photo in it
// kept showing that stale photo forever, even after another tool had
// produced a newer result.
//
// This file replaces that guesswork with one rule:
//
//     THE SESSION OWNS THE IMAGE. VIEWS ONLY DISPLAY IT.
//
// Concretely:
//   - core/session.py holds one WORKING IMAGE + one global history.
//   - Each view registers itself here (registerView) with a load()
//     function and a getPath() so we can tell what it's showing.
//   - Whenever the working image changes -- from ANY tool -- the visible
//     view is re-loaded from it; hidden views are re-loaded lazily when
//     they're switched to (syncView), so we never pay to render a view
//     nobody is looking at.
//   - So: Filters -> Enhance -> Remove BG -> Filters all show the same,
//     latest result, in any order, with no per-view hand-off code.
//
// ALSO OWNED HERE
// ---------------
//   - Global Undo / Redo (Ctrl+Z / Ctrl+Y) across tool boundaries, plus
//     the history panel that lists every step and lets the user jump.
//   - Ctrl+S       = Save Project (.pfproj -- working image + history +
//                    every tool's settings). NOT a screenshot.
//   - Ctrl+Shift+S = Export As (flatten the working image to a normal
//                    image file at a path the user picks).
//   - The unsaved-changes dot, and the "your original is never touched"
//     guarantee being visible in the UI rather than just true internally.
//
// Load order note: this file must run AFTER bridge.js (it uses
// window.onPixelforgeReady) and can load before or after the view
// scripts -- registerView is safe to call at any time, and each
// registration immediately syncs if a session is already open.

(function () {
    "use strict";

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    // Same conversion as editor.js/filters.js/removebg.js: Windows paths
    // need forward slashes + a leading slash, and encodeURI keeps ":" and
    // "/" intact while escaping spaces.
    function toFileUrl(path) {
        if (!path) return "";
        const normalized = String(path).replace(/\\/g, "/").replace(/^\/+/, "");
        return "file:///" + encodeURI(normalized);
    }

    let state = null;             // last known session state from Python
    const changeListeners = [];   // cb(state)
    const views = {};             // viewName -> {load, getPath}
    let activeView = null;        // whichever view is on screen
    let adopting = false;         // guards view-load -> start() recursion
    let busyCount = 0;

    function bridge() {
        return window.pixelforge || null;
    }

    function hasSession() {
        return !!(state && state.has_session);
    }

    function workingPath() {
        return state && state.has_session ? state.working_path : "";
    }

    // ---------------------------------------------------------------
    // Toasts
    // ---------------------------------------------------------------
    // The codebase used alert() for "Exported to: ..." which is a modal
    // that blocks the whole window. A toast says the same thing without
    // stopping the user mid-edit. Exposed globally so the other views can
    // drop their alert()s onto it too.

    let toastHost = null;

    function toast(message, kind) {
        if (!message) return;
        if (!toastHost) {
            toastHost = document.getElementById("toast-host");
            if (!toastHost) {
                toastHost = document.createElement("div");
                toastHost.id = "toast-host";
                toastHost.className = "toast-host";
                document.body.appendChild(toastHost);
            }
        }
        const el = document.createElement("div");
        el.className = "toast" + (kind ? ` toast--${kind}` : "");
        el.textContent = message;
        toastHost.appendChild(el);
        // Force a frame before adding the class so the CSS transition runs.
        requestAnimationFrame(() => el.classList.add("is-visible"));
        setTimeout(() => {
            el.classList.remove("is-visible");
            setTimeout(() => el.remove(), 260);
        }, kind === "error" ? 5200 : 3200);
    }

    window.pixelforgeToast = toast;

    // ---------------------------------------------------------------
    // Session bar DOM
    // ---------------------------------------------------------------

    const el = {};

    function cacheDom() {
        el.bar = document.getElementById("session-bar");
        el.undo = document.getElementById("btn-session-undo");
        el.redo = document.getElementById("btn-session-redo");
        el.historyToggle = document.getElementById("btn-session-history");
        el.stepLabel = document.getElementById("session-step-label");
        el.stepCount = document.getElementById("session-step-count");
        el.dirty = document.getElementById("session-dirty");
        el.history = document.getElementById("session-history");
        el.historyList = document.getElementById("session-history-list");
        el.historySub = document.getElementById("session-history-sub");
        el.revert = document.getElementById("btn-session-revert");
        el.save = document.getElementById("btn-session-save");
        el.export = document.getElementById("btn-session-export");
        el.open = document.getElementById("btn-session-open-project");
    }

    const TOOL_LABELS = {
        original: "Original",
        enhance: "Enhance",
        filters: "Filter",
        removebg: "Background",
        crop: "Crop",
        pipeline: "Smart Pipeline",
        edit: "Edit",
    };

    function setBusy(on) {
        busyCount = Math.max(0, busyCount + (on ? 1 : -1));
        if (el.bar) el.bar.classList.toggle("is-busy", busyCount > 0);
    }

    function render() {
        if (!el.bar) return;

        el.bar.classList.toggle("view--hidden", !hasSession());
        if (!hasSession()) {
            closeHistory();
            return;
        }

        if (el.undo) el.undo.disabled = !state.can_undo;
        if (el.redo) el.redo.disabled = !state.can_redo;

        if (el.stepLabel) el.stepLabel.textContent = state.current_label || "Original";
        if (el.stepCount) {
            el.stepCount.textContent = `${(state.current_step || 0) + 1}/${state.step_count || 1}`;
        }
        if (el.dirty) el.dirty.classList.toggle("view--hidden", !state.dirty);
        if (el.historyToggle) {
            el.historyToggle.title = state.project_name
                ? `${state.project_name}${state.dirty ? " (unsaved changes)" : ""}`
                : "Edit history -- your original file is never modified";
        }
        if (el.save) {
            el.save.textContent = state.project_path ? "Save" : "Save Project";
        }
        if (el.revert) el.revert.disabled = !state.can_undo;

        renderHistory();
    }

    function renderHistory() {
        if (!el.historyList) return;

        if (el.historySub) {
            el.historySub.textContent = state.original_name
                ? `from ${state.original_name}`
                : "";
        }

        el.historyList.innerHTML = "";
        (state.history || []).forEach((entry) => {
            const row = document.createElement("button");
            row.type = "button";
            row.className = "history-row";
            if (entry.is_current) row.classList.add("is-current");
            // Steps after the cursor are the redo branch -- still
            // reachable, but shown dimmed so "where am I" is obvious.
            if (entry.step > (state.current_step || 0)) row.classList.add("is-future");
            row.dataset.entryId = entry.id;

            const tool = document.createElement("span");
            tool.className = "history-row-tool";
            tool.textContent = TOOL_LABELS[entry.tool] || entry.tool;

            const label = document.createElement("span");
            label.className = "history-row-label";
            label.textContent = entry.label || "Edit";

            const step = document.createElement("span");
            step.className = "history-row-step";
            step.textContent = entry.step === 0 ? "base" : `#${entry.step}`;

            row.append(tool, label, step);
            row.addEventListener("click", () => {
                closeHistory();
                jumpTo(entry.id);
            });
            el.historyList.appendChild(row);
        });
    }

    function openHistory() {
        if (el.history) el.history.classList.remove("view--hidden");
        if (el.historyToggle) el.historyToggle.classList.add("is-open");
    }

    function closeHistory() {
        if (el.history) el.history.classList.add("view--hidden");
        if (el.historyToggle) el.historyToggle.classList.remove("is-open");
    }

    function toggleHistory() {
        if (!el.history) return;
        if (el.history.classList.contains("view--hidden")) openHistory();
        else closeHistory();
    }

    // ---------------------------------------------------------------
    // View registration + adoption -- the actual fix for the stale-image bug
    // ---------------------------------------------------------------

    function registerView(name, handlers) {
        if (!name || !handlers || typeof handlers.load !== "function") return;
        views[name] = handlers;
        // A view can register after a session already exists (script load
        // order, or a view built later) -- sync it right away if it's the
        // one on screen.
        if (name === activeView) syncView(name);
    }

    function syncView(name) {
        const view = views[name];
        if (!view || !hasSession()) return;
        const path = workingPath();
        if (!path) return;

        const shown = typeof view.getPath === "function" ? view.getPath() : null;
        if (shown === path) return; // already showing the latest -- nothing to do

        adopting = true;
        try {
            view.load(path);
        } finally {
            // Cleared on the next tick, not synchronously: view loaders
            // are async (they call getImageInfo first), so the guard has
            // to outlive this call stack.
            setTimeout(() => { adopting = false; }, 0);
        }
    }

    function setActiveView(name) {
        activeView = name;
        closeHistory();
        syncView(name);
    }

    function notify() {
        // Keep the legacy global in step: removebg.js/editor.js/filters.js
        // still read window.pixelforgeCurrentImage in places, and any code
        // that hasn't been migrated should see the same truth.
        if (hasSession()) window.pixelforgeCurrentImage = workingPath();
        render();
        if (activeView) syncView(activeView);
        changeListeners.forEach((cb) => {
            try {
                cb(state);
            } catch (err) {
                console.error("PixelForge: session listener failed", err);
            }
        });
    }

    function absorb(result, cb, errorPrefix) {
        setBusy(false);
        if (!result || !result.ok) {
            const message = (result && result.error) || `${errorPrefix || "Operation"} failed.`;
            toast(message, "error");
            if (cb) cb(result || { ok: false, error: message });
            return;
        }
        state = result;
        notify();
        if (cb) cb(result);
    }

    // ---------------------------------------------------------------
    // Public operations
    // ---------------------------------------------------------------

    function refresh(cb) {
        const api = bridge();
        if (!api) return;
        api.sessionState((result) => {
            if (result && result.ok) {
                state = result;
                notify();
            }
            if (cb) cb(result);
        });
    }

    // For call sites that already hold a fresh session state from their own
    // bridge call (smartPipelineApply, exportRemoveBackground, ... all
    // return one) -- adopt it directly instead of paying for a second
    // round trip to read back what we were just handed.
    function adopt(newState) {
        if (!newState || !newState.ok) return;
        state = newState;
        notify();
    }

    function start(path, cb) {
        const api = bridge();
        if (!api || !path) return;
        setBusy(true);
        api.sessionStart(path, (result) => absorb(result, cb, "Opening the image"));
    }

    // kind: "adjustments" | "preset" | "straighten" | "crop" |
    //       "removebg" | "erase" | "file"   (see ui/bridge.py)
    function commit(kind, payload, cb) {
        const api = bridge();
        if (!api) return;
        if (!hasSession()) {
            toast("Open an image first.", "error");
            return;
        }
        setBusy(true);
        api.sessionCommit(kind, payload || {}, (result) => absorb(result, cb, "Applying the edit"));
    }

    function undo(cb) {
        const api = bridge();
        if (!api || !hasSession() || !state.can_undo) return;
        setBusy(true);
        api.sessionUndo((result) => absorb(result, cb, "Undo"));
    }

    function redo(cb) {
        const api = bridge();
        if (!api || !hasSession() || !state.can_redo) return;
        setBusy(true);
        api.sessionRedo((result) => absorb(result, cb, "Redo"));
    }

    function jumpTo(entryId, cb) {
        const api = bridge();
        if (!api || !entryId) return;
        setBusy(true);
        api.sessionJumpTo(entryId, (result) => absorb(result, cb, "Jumping to that step"));
    }

    function revertToOriginal(cb) {
        const api = bridge();
        if (!api || !hasSession()) return;
        setBusy(true);
        api.sessionRevertToOriginal((result) => {
            absorb(result, cb, "Reset");
            if (result && result.ok) {
                toast("Back to the original. Redo (Ctrl+Y) brings your edits back.");
            }
        });
    }

    function close(cb) {
        const api = bridge();
        if (!api) return;
        api.sessionClose((result) => {
            if (result && result.ok) {
                state = result;
                window.pixelforgeCurrentImage = null;
                notify();
            }
            if (cb) cb(result);
        });
    }

    function setToolState(tool, toolState) {
        const api = bridge();
        if (!api || !hasSession()) return;
        // Fire-and-forget: this is UI bookkeeping, and refreshing the
        // whole session (and therefore re-rendering the history panel)
        // every time a slider moves would be wasteful.
        api.sessionSetToolState(tool, toolState, () => {});
    }

    function getToolState(tool, cb) {
        const api = bridge();
        if (!api) {
            cb(null);
            return;
        }
        api.sessionGetToolState(tool, (result) => cb(result && result.ok ? result.state : null));
    }

    // ---------------------------------------------------------------
    // Ctrl+S -- Save Project
    // ---------------------------------------------------------------
    // A PROJECT, not a screenshot: working image, every history step, the
    // cursor position, each tool's settings, and the last Analyzer/Smart
    // Pipeline result. Reopening lands the user exactly where they were,
    // with undo still working.

    function saveProject(forceDialog, cb) {
        const api = bridge();
        if (!api) return;
        if (!hasSession()) {
            toast("Nothing to save yet -- open an image first.", "error");
            return;
        }

        const writeTo = (destPath) => {
            setBusy(true);
            api.saveProject(destPath || "", (result) => {
                absorb(result, cb, "Saving the project");
                if (result && result.ok) {
                    toast(`Project saved: ${result.saved_to} (${result.steps_saved} steps)`);
                }
            });
        };

        if (!forceDialog && state.project_path) {
            writeTo("");   // re-save over the existing .pfproj
            return;
        }
        api.chooseProjectSavePath((destPath) => {
            if (!destPath) return; // cancelled
            writeTo(destPath);
        });
    }

    function openProject(cb) {
        const api = bridge();
        if (!api) return;
        api.chooseProjectOpenPath((path) => {
            if (!path) return;
            setBusy(true);
            api.loadProject(path, (result) => {
                absorb(result, cb, "Opening the project");
                if (result && result.ok) {
                    toast(`Opened ${result.project_name || "project"} -- ${result.step_count} steps restored`);
                    if (window.pixelforgeShowView) window.pixelforgeShowView("enhance");
                }
            });
        });
    }

    // ---------------------------------------------------------------
    // Ctrl+Shift+S -- Export As
    // ---------------------------------------------------------------
    // The only path by which an edit reaches the user's disk as a normal
    // image, and it always goes to a destination they picked. Python
    // refuses outright if that destination IS the original file.

    function exportAs(cb) {
        const api = bridge();
        if (!api) return;
        if (!hasSession()) {
            toast("Nothing to export yet -- open an image first.", "error");
            return;
        }

        const name = state.original_name || "image.png";
        const dot = name.lastIndexOf(".");
        const base = dot > 0 ? name.slice(0, dot) : name;
        const suggested = `PixelForge_${base}.png`;

        api.chooseSaveImagePath(suggested, (destPath) => {
            if (!destPath) return; // cancelled
            setBusy(true);
            api.exportWorkingImage(destPath, {}, (result) => {
                setBusy(false);
                if (!result.ok) {
                    toast(result.error || "Export failed.", "error");
                    if (cb) cb(result);
                    return;
                }
                toast(result.note ? `Exported to ${result.path} -- ${result.note}` : `Exported to ${result.path}`);
                if (cb) cb(result);
            });
        });
    }

    // ---------------------------------------------------------------
    // Wiring
    // ---------------------------------------------------------------

    function wireBar() {
        cacheDom();
        if (el.undo) el.undo.addEventListener("click", () => undo());
        if (el.redo) el.redo.addEventListener("click", () => redo());
        if (el.historyToggle) {
            el.historyToggle.addEventListener("click", (e) => {
                e.stopPropagation();
                toggleHistory();
            });
        }
        if (el.revert) el.revert.addEventListener("click", () => { closeHistory(); revertToOriginal(); });
        if (el.save) el.save.addEventListener("click", () => saveProject(false));
        if (el.export) el.export.addEventListener("click", () => exportAs());
        if (el.open) el.open.addEventListener("click", () => openProject());

        // Click-away closes the history dropdown.
        document.addEventListener("click", (e) => {
            if (!el.history || el.history.classList.contains("view--hidden")) return;
            if (el.history.contains(e.target)) return;
            if (el.historyToggle && el.historyToggle.contains(e.target)) return;
            closeHistory();
        });
    }

    // ----- Global keyboard shortcuts -----
    // Deliberately global (document level) rather than per-view: the
    // whole point of one shared history is that Ctrl+Z means the same
    // thing everywhere. removebg.js still has its own view-scoped
    // Ctrl+Z for its brush strokes; that handler runs first and calls
    // preventDefault when it consumes the key, which is why this one
    // checks defaultPrevented before acting.
    function wireShortcuts() {
        document.addEventListener("keydown", (e) => {
            if (!(e.ctrlKey || e.metaKey)) return;
            if (e.defaultPrevented) return;

            // Don't hijack typing in a text field (preset names, search).
            const tag = (e.target && e.target.tagName) || "";
            const isTextField =
                tag === "INPUT" && !["range", "checkbox", "radio", "color"].includes((e.target.type || "").toLowerCase());
            if (isTextField || tag === "TEXTAREA" || (e.target && e.target.isContentEditable)) {
                // Ctrl+S should still save even from a text field --
                // that's what every other editor does.
                if (e.key.toLowerCase() !== "s") return;
            }

            const key = e.key.toLowerCase();

            if (key === "s") {
                e.preventDefault();
                if (e.shiftKey) exportAs();
                else saveProject(false);
                return;
            }
            if (key === "z") {
                e.preventDefault();
                if (e.shiftKey) redo();
                else undo();
                return;
            }
            if (key === "y") {
                e.preventDefault();
                redo();
                return;
            }
            if (key === "o") {
                e.preventDefault();
                if (e.shiftKey) {
                    openProject();
                } else if (bridge()) {
                    bridge().openImageDialog((path) => { if (path) start(path); });
                }
                return;
            }
            if (key === "h") {
                e.preventDefault();
                if (hasSession()) toggleHistory();
            }
        });
    }

    // ---------------------------------------------------------------
    // Public surface
    // ---------------------------------------------------------------

    window.pixelforgeSession = {
        // reads
        get state() { return state; },
        hasSession,
        workingPath,
        originalName: () => (state ? state.original_name : ""),
        isAdopting: () => adopting,
        toFileUrl,

        // views
        registerView,
        syncView,
        setActiveView,
        onChange: (cb) => { if (typeof cb === "function") changeListeners.push(cb); },

        // operations
        refresh,
        adopt,
        start,
        commit,
        undo,
        redo,
        jumpTo,
        revertToOriginal,
        close,
        setToolState,
        getToolState,

        // project / export
        saveProject,
        openProject,
        exportAs,

        // misc
        toast,
        openHistory,
        closeHistory,
    };

    document.addEventListener("DOMContentLoaded", () => {
        wireBar();
        wireShortcuts();
        render();
        if (window.onPixelforgeReady) {
            // Python may already have a session (e.g. after a reload of
            // the page inside a running app), so read it rather than
            // assuming an empty one.
            window.onPixelforgeReady(() => refresh());
        }
    });
})();
