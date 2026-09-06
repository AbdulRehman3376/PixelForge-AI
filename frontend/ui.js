// UI interaction logic
// Handles sidebar navigation, view switching, and wiring buttons to the
// Python bridge (see bridge.js / ui/bridge.py).

document.addEventListener("DOMContentLoaded", () => {
    const navItems = document.querySelectorAll(".nav-item[data-view]");
    const sectionTitle = document.getElementById("current-section");

    // Views that have real, built content. Everything else falls back to
    // #view-placeholder so unbuilt tools don't 404 or show a blank screen.
    const BUILT_VIEWS = ["home", "settings", "enhance", "removebg", "filters", "aigen", "batch", "upscale", "portrait", "video", "projects"]; // "enhance" added in Phase 2, "removebg" in Phase 2A, "filters" in Phase 4, "aigen" in Phase 7, "batch" in Phase 8, "upscale" in Phase 9, "portrait" in Phase 10, "video" in Phase 11, "projects" in Phase 13

    const VIEW_LABELS = {
        home: "Home",
        enhance: "Enhance",
        upscale: "AI Upscale",
        filters: "Filters",
        removebg: "Remove BG",
        aigen: "AI Generate",
        portrait: "Portrait",
        batch: "Batch",
        video: "Video Studio",
        projects: "Projects",
        settings: "Settings",
    };

    // ----- "Coming Soon" badges -----
    // Any sidebar item whose view isn't in BUILT_VIEWS gets a badge
    // automatically -- add a view to BUILT_VIEWS above and its badge
    // disappears on its own, no manual bookkeeping per feature.
    navItems.forEach((item) => {
        if (!BUILT_VIEWS.includes(item.dataset.view)) {
            const label = item.querySelector(".nav-item-label");
            if (label && !item.querySelector(".nav-item-badge")) {
                label.insertAdjacentHTML(
                    "afterend",
                    '<span class="badge badge--muted nav-item-badge">Soon</span>'
                );
            }
        }
    });

    // Set by editor.js/removebg.js whenever an image is loaded there, so
    // switching tools can hand the same photo across automatically
    // instead of forcing the user to browse for it again in every view.
    window.pixelforgeCurrentImage = window.pixelforgeCurrentImage || null;

    // viewName -> [dropzone id, loader fn name] for views that can pick
    // up window.pixelforgeCurrentImage on first switch-in. Only wired for
    // views that actually have their own image workspace.
    const AUTO_LOAD_TARGETS = {
        enhance: { dropzoneId: "editor-dropzone", loaderName: "pixelforgeLoadImageIntoEditor" },
        removebg: { dropzoneId: "removebg-dropzone", loaderName: "pixelforgeLoadImageIntoRemoveBG" },
        filters: { dropzoneId: "filters-dropzone", loaderName: "pixelforgeLoadImageIntoFilters" },
        upscale: { dropzoneId: "upscale-dropzone", loaderName: "pixelforgeLoadImageIntoUpscale" }, // PHASE 9
        portrait: { dropzoneId: "portrait-dropzone", loaderName: "pixelforgeLoadImageIntoPortrait" }, // PHASE 10
    };

    // PHASE 6: viewName -> the global var each view's loadImage() sets
    // with the path it currently has open (editor.js/filters.js/
    // removebg.js all set their own copy on load -- see the matching
    // `window.pixelforge<View>CurrentPath = path;` line in each file).
    // Used below so switching back into a view whose dropzone is
    // already hidden (an image was loaded there before) still reloads
    // when the shared Working Image has moved on since -- this is the
    // actual fix for "Filter it -> open Enhance -> old photo is back".
    const VIEW_CURRENT_PATH_VARS = {
        enhance: "pixelforgeEnhanceCurrentPath",
        removebg: "pixelforgeRemoveBGCurrentPath",
        filters: "pixelforgeFiltersCurrentPath",
        upscale: "pixelforgeUpscaleCurrentPath", // PHASE 9
        portrait: "pixelforgePortraitCurrentPath", // PHASE 10
    };

    function maybeAutoLoadCurrentImage(viewName) {
        const target = AUTO_LOAD_TARGETS[viewName];
        if (!target || !window.pixelforgeCurrentImage) return;

        const dropzone = document.getElementById(target.dropzoneId);
        const loader = window[target.loaderName];
        if (viewName === "removebg" && typeof window.pixelforgeGetEditorTransferPath === "function") {
            const transferPath = window.pixelforgeGetEditorTransferPath();
            if (transferPath) window.pixelforgeCurrentImage = transferPath;
        }
        // Load when the target has no image loaded yet, OR when this
        // view's last-loaded path no longer matches the shared Working
        // Image (i.e. some other tool committed a newer edit since).
        const pathVar = VIEW_CURRENT_PATH_VARS[viewName];
        const staleAgainstSession = pathVar ? window[pathVar] !== window.pixelforgeCurrentImage : false;
        const shouldLoad = staleAgainstSession || (dropzone && !dropzone.classList.contains("view--hidden"));
        if (loader && typeof loader === "function" && shouldLoad) {
            loader(window.pixelforgeCurrentImage);
        }
    }

    // BUGFIX: the Settings page's "AI Models" badges (Upscaling model /
    // Face restoration) were hardcoded "Not installed" directly in
    // index.html and never wired to anything -- unlike the FFmpeg badge
    // right below them, nothing ever called back into the bridge to
    // check the real state, so they stayed wrong forever even after the
    // models were actually downloaded/cached. ai/upscaler.py::model_status
    // and ai/face_restorer.py::model_status already existed (used by the
    // Upscale/Portrait tools' own in-panel status line) -- this just
    // reuses them here too.
    function maybeRefreshSettingsModelBadges() {
        const upscaleBadge = document.getElementById("settings-upscale-model-status");
        const faceBadge = document.getElementById("settings-facerestore-model-status");
        const ffmpegBadge = document.getElementById("ffmpeg-status");

        function applyBadge(el, ok, okText, notOkText) {
            if (!el) return;
            el.textContent = ok ? okText : notOkText;
            el.classList.toggle("badge--ok", !!ok);
            el.classList.toggle("badge--muted", !ok);
        }

        if (upscaleBadge && window.pixelforge && window.pixelforge.upscale) {
            window.pixelforge.upscale.modelStatus((result) => {
                const ready = !!(result && result.ok && result.cached);
                applyBadge(upscaleBadge, ready, "Installed", "Not installed");
            });
        }
        if (faceBadge && window.pixelforge && window.pixelforge.faceRestore) {
            window.pixelforge.faceRestore.modelStatus((result) => {
                const ready = !!(result && result.ok && result.cached);
                applyBadge(faceBadge, ready, "Installed", "Not installed");
            });
        }
        if (ffmpegBadge && window.pixelforge && window.pixelforge.video) {
            window.pixelforge.video.ffmpegStatus((status) => {
                const available = !!(status && status.available);
                applyBadge(ffmpegBadge, available, status && status.version ? `Detected (${status.version})` : "Detected", "Not detected");
            });
        }
    }
    window.onPixelforgeReady(() => {
        if (!document.getElementById("view-settings").classList.contains("view--hidden")) {
            maybeRefreshSettingsModelBadges();
        }
    });

    // BUGFIX: "Clear cache" button had no id and no listener at all --
    // clicking it was a no-op. Now wired to the new clearCache bridge
    // method (see ui/bridge.py).
    const btnClearCache = document.getElementById("btn-clear-cache");
    if (btnClearCache) {
        btnClearCache.addEventListener("click", () => {
            if (!window.pixelforge || !window.pixelforge.clearCache) return;
            const original = btnClearCache.textContent;
            btnClearCache.disabled = true;
            btnClearCache.textContent = "Clearing...";
            window.pixelforge.clearCache((res) => {
                btnClearCache.disabled = false;
                btnClearCache.textContent = original;
                if (res && res.ok) {
                    const mb = res.freed_bytes ? (res.freed_bytes / (1024 * 1024)).toFixed(1) : "0";
                    btnClearCache.textContent = `Cleared (${mb} MB)`;
                    setTimeout(() => { btnClearCache.textContent = original; }, 2000);
                } else {
                    window.alert("Couldn't fully clear the cache. Some files may be in use.");
                }
            });
        });
    }

    // PHASE 14: Settings > System Health -- live CPU/RAM/Disk gauges.
    // Polls the lightweight getSystemSnapshot bridge call (plain psutil
    // read, not the heavier getSystemHealth model/FFmpeg check) every
    // 2s, but ONLY while the Settings view is actually on screen --
    // started in showView("settings") below and stopped the instant the
    // user navigates away, so this never runs a background timer for
    // the whole app session.
    const GAUGE_CIRCUMFERENCE = 263.9; // 2 * pi * 42, matches the SVG r=42 in index.html
    const HEALTH_POLL_MS = 2000;
    const CPU_WARN_PCT = 75;
    const CPU_DANGER_PCT = 90;
    const RAM_WARN_PCT = 80;
    const RAM_DANGER_PCT = 92;
    let healthPollTimer = null;
    let healthHotStreak = 0; // consecutive hot readings, so one brief spike doesn't trigger the banner
    let smoothedCpuPct = null; // EMA state, reset each time polling (re)starts -- see startHealthPolling()

    function setGauge(gaugeId, valueElId, percent) {
        const gaugeEl = document.getElementById(gaugeId);
        if (!gaugeEl) return;
        const pct = Math.max(0, Math.min(100, percent || 0));
        const circle = gaugeEl.querySelector(".health-gauge-progress");
        if (circle) {
            circle.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE * (1 - pct / 100));
        }
        const valueEl = document.getElementById(valueElId);
        if (valueEl) valueEl.textContent = `${Math.round(pct)}%`;

        gaugeEl.classList.remove("health-gauge--warn", "health-gauge--danger");
        if (pct >= CPU_DANGER_PCT) gaugeEl.classList.add("health-gauge--danger");
        else if (pct >= CPU_WARN_PCT) gaugeEl.classList.add("health-gauge--warn");
    }

    function formatGB(n) {
        return typeof n === "number" ? `${n.toFixed(1)} GB` : "--";
    }

    function applyHealthSnapshot(snap) {
        if (!snap) return;
        const rawCpuPct = snap.cpu_percent || 0;
        const ram = snap.ram || {};
        const disk = snap.disk || {};
        const ramPct = ram.percent_used || 0;

        // Smooth the CPU ring with an EMA (30% weight on the newest
        // sample) so a single brief 100% blip -- real background-process
        // activity, not a bug, now that get_cpu_percent_live() reads
        // accurately -- doesn't make the gauge itself look erratic. A
        // SUSTAINED high load still climbs the smoothed value within a
        // couple of polls and correctly reaches the danger zone; a true
        // one-off spike is visibly damped instead of causing a jarring
        // full-ring jump and instant snap-back.
        smoothedCpuPct = smoothedCpuPct === null ? rawCpuPct : (smoothedCpuPct * 0.7 + rawCpuPct * 0.3);
        const cpuPct = smoothedCpuPct;

        let diskPct = 0;
        if (disk.ok && typeof disk.used_percent === "number") {
            diskPct = disk.used_percent;
        }

        setGauge("health-gauge-cpu", "health-cpu-value", cpuPct);
        setGauge("health-gauge-ram", "health-ram-value", ramPct);
        setGauge("health-gauge-disk", "health-disk-value", diskPct);

        const ramDetail = document.getElementById("health-ram-detail");
        if (ramDetail) {
            ramDetail.textContent = ram.available
                ? `${formatGB(ram.used_gb)} / ${formatGB(ram.total_gb)}`
                : "Unavailable";
        }
        const diskDetail = document.getElementById("health-disk-detail");
        if (diskDetail) {
            if (disk.ok && typeof disk.total_gb === "number") {
                diskDetail.textContent = `${(disk.free_mb / 1024).toFixed(1)} GB free / ${disk.total_gb.toFixed(1)} GB`;
            } else if (disk.ok) {
                diskDetail.textContent = `${(disk.free_mb / 1024).toFixed(1)} GB free`;
            } else {
                diskDetail.textContent = "Unavailable";
            }
        }

        // Overheat / cooldown notice: checked against the RAW (unsmoothed)
        // CPU reading, not the damped gauge value -- a real sustained
        // spike must still trigger the warning promptly and can't be
        // hidden by the EMA above. Requires 2 consecutive hot polls
        // (~4s sustained, not a one-frame blip) before warning, and
        // drops it the instant things cool back down.
        const isHot = rawCpuPct >= CPU_DANGER_PCT || ramPct >= RAM_DANGER_PCT;
        const isWarm = rawCpuPct >= CPU_WARN_PCT || ramPct >= RAM_WARN_PCT;
        healthHotStreak = isHot ? healthHotStreak + 1 : 0;

        const banner = document.getElementById("health-warning-banner");
        const bannerText = document.getElementById("health-warning-text");
        const statusBadge = document.getElementById("health-status-badge");

        if (healthHotStreak >= 2) {
            if (banner) banner.classList.remove("health-warning-banner--hidden");
            if (bannerText) {
                bannerText.textContent = cpuPct >= CPU_DANGER_PCT
                    ? "CPU running hot -- close a few heavy tools or pause processing to let it cool down."
                    : "RAM nearly full -- close unused tabs/tools to free up memory.";
            }
            if (statusBadge) {
                statusBadge.textContent = "Cool down recommended";
                statusBadge.className = "badge badge--error";
            }
        } else {
            if (banner) banner.classList.add("health-warning-banner--hidden");
            if (statusBadge) {
                if (isWarm) {
                    statusBadge.textContent = "Elevated";
                    statusBadge.className = "badge badge--warn";
                } else {
                    statusBadge.textContent = "Nominal";
                    statusBadge.className = "badge badge--ok";
                }
            }
        }
    }

    function pollHealthOnce() {
        if (!window.pixelforge || !window.pixelforge.getSystemSnapshot) return;
        window.pixelforge.getSystemSnapshot((snap) => applyHealthSnapshot(snap));
    }

    function startHealthPolling() {
        stopHealthPolling();
        smoothedCpuPct = null;
        pollHealthOnce();
        healthPollTimer = setInterval(pollHealthOnce, HEALTH_POLL_MS);
    }

    function stopHealthPolling() {
        if (healthPollTimer) {
            clearInterval(healthPollTimer);
            healthPollTimer = null;
        }
        healthHotStreak = 0;
    }

    function showView(viewName) {
        // Hide all real views
        document.querySelectorAll(".view").forEach((el) => el.classList.add("view--hidden"));

        if (BUILT_VIEWS.includes(viewName)) {
            const target = document.getElementById(`view-${viewName}`);
            if (target) target.classList.remove("view--hidden");
            maybeAutoLoadCurrentImage(viewName);
            if (viewName === "settings") {
                maybeRefreshSettingsModelBadges();
                startHealthPolling();
            } else {
                stopHealthPolling();
            }
        } else {
            stopHealthPolling();
            const placeholder = document.getElementById("view-placeholder");
            document.getElementById("placeholder-text").textContent =
                `${VIEW_LABELS[viewName] || "This tool"} isn't built yet`;
            placeholder.classList.remove("view--hidden");
        }

        // Sync sidebar active state + top bar title
        navItems.forEach((item) => {
            item.classList.toggle("active", item.dataset.view === viewName);
        });
        sectionTitle.textContent = VIEW_LABELS[viewName] || viewName;

        // PHASE 13: lets projects.js (and any future view) refresh its
        // own data only when actually switched into, instead of every
        // view polling the database on a timer or on every keystroke
        // elsewhere in the app.
        document.dispatchEvent(new CustomEvent("pixelforge:viewshown", { detail: { view: viewName } }));
    }

    // Exposed so other screens (editor.js, removebg.js) can switch views
    // themselves -- used by the "Continue in Remove BG" / "Continue in
    // Enhance" handoff buttons after an export.
    window.pixelforgeShowView = showView;

    navItems.forEach((item) => {
        item.addEventListener("click", () => showView(item.dataset.view));
    });

    // ----- Theme switching (Settings > General > Theme) -----
    const themeSelect = document.getElementById("setting-theme");
    if (themeSelect) {
        themeSelect.addEventListener("change", () => {
            const theme = themeSelect.value; // "dark" | "light"
            document.documentElement.setAttribute("data-theme", theme);
            if (window.pixelforge) {
                window.pixelforge.setSetting("theme", theme);
            }
        });
    }

    // Home screen quick-action cards
    document.querySelectorAll(".quick-card").forEach((card) => {
        card.addEventListener("click", () => {
            const viewTarget = card.dataset.viewTarget;
            const action = card.dataset.action;

            if (viewTarget) {
                showView(viewTarget);
                return;
            }

            if (action === "open-image" && window.pixelforge) {
                // Plain "Open Image" from Home isn't tied to any
                // project -- clear whatever project was last active so
                // this picture doesn't get silently filed under it.
                if (window.pixelforgeSetActiveDbProject) window.pixelforgeSetActiveDbProject(null);
                window.pixelforge.openImageDialog((path) => {
                    if (path) pixelforgeOpenImage(path);
                });
            } else if (action === "open-batch" && window.pixelforge) {
                // PHASE 8: hand picked files straight to the Batch queue
                // and switch there, instead of just logging them.
                window.pixelforge.openImagesDialog((paths) => {
                    if (paths && paths.length) {
                        showView("batch");
                        if (window.pixelforgeBatchAddFiles) window.pixelforgeBatchAddFiles(paths);
                    }
                });
            }
        });
    });

    // Settings: "Browse" output directory
    const btnChooseOutput = document.getElementById("btn-choose-output");
    if (btnChooseOutput) {
        btnChooseOutput.addEventListener("click", () => {
            if (!window.pixelforge) return;
            window.pixelforge.chooseOutputFolder((path) => {
                if (path) {
                    document.getElementById("setting-output-dir").value = path;
                    window.pixelforge.setSetting("output_dir", path);
                }
            });
        });
    }

    // Top bar "Import" button behaves like the Home "Open Image" card
    const btnImport = document.getElementById("btn-import");
    if (btnImport) {
        btnImport.addEventListener("click", () => {
            if (!window.pixelforge) return;
            if (window.pixelforgeSetActiveDbProject) window.pixelforgeSetActiveDbProject(null);
            window.pixelforge.openImageDialog((path) => {
                if (path) pixelforgeOpenImage(path);
            });
        });
    }

    // Top bar "New Project" button.
    // BUGFIX: this button had NO click listener anywhere in the app --
    // clicking it silently did nothing, which is also why "Save
    // Project" right next to it looked broken: Save Project only
    // enables once a session is open (see pixelforgeOnSessionUpdated
    // below), and with New Project doing nothing there was no way to
    // get a fresh session going without already having one.
    //
    // Behavior: if there's an unsaved session open, confirm before
    // discarding it; then closes the current session -- exactly what
    // core/session.py::EditSession.close()'s own docstring says it's
    // for ("New Project / closing the image") -- and immediately opens
    // the picker so the user can start editing the new project's first
    // photo, same as Import/the Home "Open Image" card.
    const btnNewProject = document.getElementById("btn-new-project");
    if (btnNewProject) {
        btnNewProject.addEventListener("click", () => {
            if (!window.pixelforge) return;
            const current = window.pixelforgeSessionState;
            if (current && current.dirty) {
                const proceed = window.confirm(
                    "This project has unsaved changes. Start a new project anyway?"
                );
                if (!proceed) return;
            }

            const openFresh = () => {
                if (window.pixelforgeSetActiveDbProject) window.pixelforgeSetActiveDbProject(null);
                window.pixelforge.openImageDialog((path) => {
                    if (path) pixelforgeOpenImage(path);
                });
            };

            if (current && current.has_session) {
                window.pixelforge.session.close((state) => {
                    pixelforgeOnSessionUpdated(state);
                    window.pixelforgeCurrentImage = null;
                    openFresh();
                });
            } else {
                openFresh();
            }
        });
    }

    // ===================== PHASE 6: Shared Working Image / Session =====================
    //
    // Single choke point for "the user just opened a photo": starts a
    // fresh core/session.py EditSession anchored to that file, then
    // hands the WORKING IMAGE path (== the original, for a brand-new
    // session) to Enhance. Every other view picks it up automatically
    // via maybeAutoLoadCurrentImage() above once it becomes the active
    // view, and stays in sync afterward through pixelforgeOnSessionUpdated
    // (called by editor.js/filters.js/removebg.js after each Apply, and
    // by the Undo/Redo/Save Project buttons below).
    //
    // View-agnostic version, used by every view's own drag-and-drop
    // handler (editor.js/filters.js/removebg.js) -- dropping a NEW photo
    // directly into Filters or Remove BG is just as much "opening an
    // image" as using the Import button, and must start a session too,
    // or Ctrl+S / Undo-Redo / Apply / Smart Pipeline all silently no-op
    // because the backend has no active session to act on. Does NOT
    // switch views -- the caller is already the right view.
    function pixelforgeStartSession(path, onDone) {
        if (!window.pixelforge || !window.pixelforge.session) {
            if (onDone) onDone(null);
            return;
        }
        window.pixelforge.session.open(path, (state) => {
            if (!state || !state.ok) {
                console.warn("Session open failed:", state && state.error);
                window.pixelforgeCurrentImage = path;
                if (onDone) onDone(null);
                return;
            }
            pixelforgeOnSessionUpdated(state);
            if (onDone) onDone(state);
        });
    }
    window.pixelforgeStartSession = pixelforgeStartSession;

    // PHASE 13 FOLLOW-UP: the SQLite Projects grid (frontend/projects.js)
    // and this Shared Working Image/Session mechanism used to know
    // nothing about each other -- opening a picture never told the
    // Projects database about it, so a project's Media tab stayed
    // empty forever and its card never got a thumbnail no matter how
    // much editing happened. window.pixelforgeActiveDbProjectId is set
    // by projects.js right before it hands off to pixelforgeOpenImage()
    // (via "+ New Project" or a project's "Open in Editor"/media-row
    // click); when it's set, opening a picture here also records it
    // against that project and gives the project card a real preview
    // image instead of the placeholder icon.
    window.pixelforgeActiveDbProjectId = window.pixelforgeActiveDbProjectId || null;
    window.pixelforgeActiveDbMediaId = window.pixelforgeActiveDbMediaId || null;

    function pixelforgeSetActiveDbProject(projectId) {
        window.pixelforgeActiveDbProjectId = projectId || null;
        window.pixelforgeActiveDbMediaId = null;
        // BUGFIX (duplicate "container" project, part 2): clearing the
        // active project to null here only ever cleared the client-side
        // pointer -- the backend (core/database.py's active-project row,
        // which _db_project_for_source now also consults on export) was
        // never told, so it kept reporting the *previous* project as
        // active. That stale backend "active project" then caused the
        // next image's edits to be misfiled into the old project instead
        // of starting clean. "New Project" (topbar) is the caller that
        // passes null -- mirror that over to the backend too.
        if (!projectId && window.pixelforge && window.pixelforge.projects && window.pixelforge.projects.clearActive) {
            window.pixelforge.projects.clearActive();
        }
    }
    window.pixelforgeSetActiveDbProject = pixelforgeSetActiveDbProject;

    function pixelforgeAttachToActiveDbProject(path) {
        if (!window.pixelforgeActiveDbProjectId || !window.pixelforge || !window.pixelforge.projects) return;
        const projectId = window.pixelforgeActiveDbProjectId;
        window.pixelforge.projects.addMedia(projectId, path, "", "image", 0, 0, (res) => {
            if (res && res.ok && res.data) {
                window.pixelforgeActiveDbMediaId = res.data.id;
            }
            // Gives the project card a real preview instead of the "no
            // picture" placeholder icon -- this is what "the project
            // is useless, there's just a name in it" was missing.
            window.pixelforge.projects.setThumbnail(projectId, path, () => {});
        });
    }
    window.pixelforgeAttachToActiveDbProject = pixelforgeAttachToActiveDbProject;

    function pixelforgeOpenImage(path) {
        pixelforgeStartSession(path, (state) => {
            showView("enhance");
            const target = (state && state.working_path) || path;
            if (window.pixelforgeLoadImageIntoEditor) window.pixelforgeLoadImageIntoEditor(target);
            pixelforgeAttachToActiveDbProject(path);
        });
    }
    window.pixelforgeOpenImage = pixelforgeOpenImage;

    const sessionIndicator = document.getElementById("session-indicator");
    const sessionProjectName = document.getElementById("session-project-name");
    const btnSessionUndo = document.getElementById("btn-session-undo");
    const btnSessionRedo = document.getElementById("btn-session-redo");
    const btnSessionSave = document.getElementById("btn-session-save");

    // Re-invokes whichever view's own loader for the CURRENT working
    // image, forcing a reload even if that view already has an image
    // showing -- used after Undo/Redo/Save so the on-screen photo
    // always matches the session's cursor position.
    function pixelforgeForceReloadActiveView() {
        const activeNav = document.querySelector(".nav-item.active[data-view]");
        const viewName = activeNav && activeNav.dataset.view;
        const target = AUTO_LOAD_TARGETS[viewName];
        if (!target) return;
        const loader = window[target.loaderName];
        if (loader && typeof loader === "function" && window.pixelforgeCurrentImage) {
            loader(window.pixelforgeCurrentImage);
        }
    }

    // Called after ANY successful session mutation (open/apply/undo/
    // redo/jump/revert/load-project) so the top bar and every view's
    // notion of "the current photo" stay correct.
    function pixelforgeOnSessionUpdated(state) {
        if (!state) return;
        window.pixelforgeSessionState = state;
        window.pixelforgeCurrentImage = state.working_path || window.pixelforgeCurrentImage;

        if (sessionIndicator) {
            sessionIndicator.classList.toggle("view--hidden", !state.has_session);
            sessionIndicator.classList.toggle("is-dirty", !!state.dirty);
            sessionIndicator.title = state.dirty ? "Unsaved changes" : "All changes saved";
        }
        if (sessionProjectName) {
            sessionProjectName.textContent = state.project_name || state.original_name || "Untitled";
        }
        if (btnSessionUndo) btnSessionUndo.disabled = !state.can_undo;
        if (btnSessionRedo) btnSessionRedo.disabled = !state.can_redo;
        if (btnSessionSave) btnSessionSave.disabled = !state.has_session;
    }
    window.pixelforgeOnSessionUpdated = pixelforgeOnSessionUpdated;

    if (btnSessionUndo) {
        btnSessionUndo.addEventListener("click", () => {
            if (!window.pixelforge) return;
            window.pixelforge.session.undo((state) => {
                pixelforgeOnSessionUpdated(state);
                pixelforgeForceReloadActiveView();
            });
        });
    }

    if (btnSessionRedo) {
        btnSessionRedo.addEventListener("click", () => {
            if (!window.pixelforge) return;
            window.pixelforge.session.redo((state) => {
                pixelforgeOnSessionUpdated(state);
                pixelforgeForceReloadActiveView();
            });
        });
    }

    function pixelforgeSaveProject() {
        if (!window.pixelforge || !window.pixelforgeSessionState || !window.pixelforgeSessionState.has_session) return;
        const finishSave = (destPath) => {
            window.pixelforge.session.saveProject(destPath || "", (state) => {
                pixelforgeOnSessionUpdated(state);
            });
        };
        if (window.pixelforgeSessionState.project_path) {
            // Already has a home on disk -- plain Ctrl+S re-saves there.
            finishSave("");
        } else {
            window.pixelforge.session.chooseProjectSavePath((path) => {
                if (path) finishSave(path);
            });
        }
    }
    window.pixelforgeSaveProject = pixelforgeSaveProject;

    if (btnSessionSave) {
        btnSessionSave.addEventListener("click", pixelforgeSaveProject);
    }

    // ----- Global keyboard shortcuts -----
    // Ctrl+S = Save Project, Ctrl+Shift+S = Export As (delegates to
    // whichever view is active, since each tool's Export dialog needs
    // that tool's own adjustments/preset/mask state -- Session doesn't
    // know those). Ctrl+Z / Ctrl+Y (and Ctrl+Shift+Z) = global undo/redo.
    document.addEventListener("keydown", (e) => {
        const ctrlOrCmd = e.ctrlKey || e.metaKey;
        if (!ctrlOrCmd) return;
        const key = e.key.toLowerCase();

        if (key === "s" && !e.shiftKey) {
            e.preventDefault();
            pixelforgeSaveProject();
        } else if (key === "s" && e.shiftKey) {
            e.preventDefault();
            const activeNav = document.querySelector(".nav-item.active[data-view]");
            const viewName = activeNav && activeNav.dataset.view;
            const exportBtnId = {
                enhance: "btn-editor-export",
                filters: "btn-filters-export",
                removebg: "btn-removebg-export",
                portrait: "btn-portrait-export", // PHASE 10
            }[viewName];
            const btn = exportBtnId && document.getElementById(exportBtnId);
            if (btn && !btn.disabled) btn.click();
        } else if (key === "z" && !e.shiftKey) {
            if (btnSessionUndo && !btnSessionUndo.disabled) {
                e.preventDefault();
                btnSessionUndo.click();
            }
        } else if ((key === "y") || (key === "z" && e.shiftKey)) {
            if (btnSessionRedo && !btnSessionRedo.disabled) {
                e.preventDefault();
                btnSessionRedo.click();
            }
        }
    });

    // Populate version once the Python bridge finishes connecting.
    // window.onPixelforgeReady is defined synchronously in bridge.js, so
    // this works no matter which script's DOMContentLoaded fires first.
    if (window.onPixelforgeReady) {
        window.onPixelforgeReady(() => {
            window.pixelforge.getAppVersion((v) => {
                document.getElementById("app-version").textContent = `v${v}`;
            });
        });
    }

    // ----- Global status / progress bar -----
    // Shared by every long-running operation (batch, upscale, video
    // export, ...) instead of each view building its own. Driven by
    // pyBridge.statusChanged / progressChanged (see bridge.py / bridge.js).
    const statusBar = document.getElementById("global-status-bar");
    const statusText = document.getElementById("global-status-text");
    const statusFill = document.getElementById("global-status-fill");
    const statusPercent = document.getElementById("global-status-percent");

    function setProgress(percent, label) {
        if (label) statusText.textContent = label;
        statusBar.classList.add("is-visible");

        if (percent < 0) {
            // Unknown length (e.g. "loading model...") -- show motion,
            // not a fake percentage.
            statusBar.classList.add("is-indeterminate");
            statusPercent.textContent = "";
        } else {
            statusBar.classList.remove("is-indeterminate");
            statusFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
            statusPercent.textContent = `${Math.round(percent)}%`;
        }

        if (percent >= 100) {
            // Let the full bar register with the user, then hide.
            setTimeout(() => statusBar.classList.remove("is-visible"), 900);
        }
    }

    function setStatus(label) {
        statusText.textContent = label;
        statusBar.classList.add("is-visible");
        if (label === "Ready" || label === "Idle") {
            statusBar.classList.remove("is-visible");
        }
    }

    if (window.onPixelforgeReady) {
        window.onPixelforgeReady(() => {
            window.pixelforge.onProgress(setProgress);
            window.pixelforge.onStatus(setStatus);
        });
    }

    // ----- First-run tutorial -----
    const tutorialOverlay = document.getElementById("tutorial-overlay");
    const tutorialSteps = Array.from(document.querySelectorAll(".tutorial-step"));
    const tutorialDots = Array.from(document.querySelectorAll(".tutorial-dot"));
    const tutorialPrev = document.getElementById("tutorial-prev");
    const tutorialNext = document.getElementById("tutorial-next");
    const tutorialDontShow = document.getElementById("tutorial-dont-show");
    const btnShowTutorial = document.getElementById("btn-show-tutorial");
    let tutorialStepIndex = 0;

    function renderTutorialStep() {
        tutorialSteps.forEach((el, i) => el.classList.toggle("active", i === tutorialStepIndex));
        tutorialDots.forEach((el, i) => el.classList.toggle("active", i === tutorialStepIndex));
        tutorialPrev.style.visibility = tutorialStepIndex === 0 ? "hidden" : "visible";
        tutorialNext.textContent = tutorialStepIndex === tutorialSteps.length - 1 ? "Done" : "Next";
    }

    function openTutorial() {
        tutorialStepIndex = 0;
        renderTutorialStep();
        tutorialOverlay.classList.remove("view--hidden");
    }

    function closeTutorial() {
        tutorialOverlay.classList.add("view--hidden");
        if (tutorialDontShow.checked && window.pixelforge) {
            window.pixelforge.setSetting("tutorial_seen", "true");
        }
    }

    if (tutorialNext) {
        tutorialNext.addEventListener("click", () => {
            if (tutorialStepIndex === tutorialSteps.length - 1) {
                closeTutorial();
            } else {
                tutorialStepIndex += 1;
                renderTutorialStep();
            }
        });
    }

    if (tutorialPrev) {
        tutorialPrev.addEventListener("click", () => {
            if (tutorialStepIndex > 0) {
                tutorialStepIndex -= 1;
                renderTutorialStep();
            }
        });
    }

    if (btnShowTutorial) {
        btnShowTutorial.addEventListener("click", openTutorial);
    }
});