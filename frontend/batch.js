// frontend/batch.js
//
// PHASE 8 -- Batch Processing (frontend).
//
// 🧮 Not AI -- this view is a thin renderer over the single shared
// queue/orchestration object that lives in core/batch_processor.py
// (exposed here as window.pixelforge.batch, see frontend/bridge.js).
// Every mutation (add/remove/reorder/options/start/pause/...) is sent
// to Python and the response is the FULL current state, which is what
// this file re-renders from -- same "server owns the truth, client
// just displays it" rule Phase 6's session.js established, so this view
// can never drift out of sync with what the backend is actually doing.
//
// Live updates while a batch is running arrive two ways:
//   - onItem(item)         -> one row changed (status/progress) -- patched
//                              in place instead of a full re-render, so a
//                              50-image queue doesn't repaint every frame.
//   - onProgress(pct, label, eta) -> the overall bar + ETA text.
// A full state refresh still happens whenever the user changes the
// queue itself (add/remove/reorder) or the run finishes.

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("batch-dropzone");
    const workspace = document.getElementById("batch-workspace");
    if (!dropzone || !workspace) return; // view not present in this build

    const batchError = document.getElementById("batch-error");

    const btnAddFiles = document.getElementById("btn-batch-add-files");
    const btnAddFolder = document.getElementById("btn-batch-add-folder");
    const btnAddFiles2 = document.getElementById("btn-batch-add-files-2");
    const btnAddFolder2 = document.getElementById("btn-batch-add-folder-2");
    const btnDetectDupes = document.getElementById("btn-batch-detect-dupes");
    const btnClear = document.getElementById("btn-batch-clear");

    const countQueued = document.getElementById("batch-count-queued");
    const countDone = document.getElementById("batch-count-done");
    const countFailed = document.getElementById("batch-count-failed");

    const diskWarning = document.getElementById("batch-disk-warning");
    const dupWarning = document.getElementById("batch-dup-warning");

    const queueList = document.getElementById("batch-queue-list");
    const queueCount = document.getElementById("batch-queue-count");
    const selectAll = document.getElementById("batch-select-all");

    // ----- Options panel controls -----
    const optMode = document.getElementById("batch-opt-mode");
    const presetRow = document.getElementById("batch-preset-row");
    const optPreset = document.getElementById("batch-opt-preset");
    const optIntensity = document.getElementById("batch-opt-intensity");
    const valIntensity = document.getElementById("val-batch-intensity");
    const optOutputFolder = document.getElementById("batch-opt-output-folder");
    const btnOutputFolder = document.getElementById("btn-batch-output-folder");
    const optFormat = document.getElementById("batch-opt-format");
    const qualityRow = document.getElementById("batch-quality-row");
    const optQuality = document.getElementById("batch-opt-quality");
    const valQuality = document.getElementById("val-batch-quality");
    const optResizeMode = document.getElementById("batch-opt-resize-mode");
    const resizeValueRow = document.getElementById("batch-resize-value-row");
    const optResizeValue = document.getElementById("batch-opt-resize-value");
    const optNaming = document.getElementById("batch-opt-naming");
    const optPreserve = document.getElementById("batch-opt-preserve");
    const optOverwrite = document.getElementById("batch-opt-overwrite");
    const optStripExif = document.getElementById("batch-opt-strip-exif");
    const optCopyright = document.getElementById("batch-opt-copyright");
    const optThrottle = document.getElementById("batch-opt-throttle");
    const valThrottle = document.getElementById("val-batch-throttle");
    const optNotify = document.getElementById("batch-opt-notify");
    const optDryRun = document.getElementById("batch-opt-dry-run");
    const optScheduleMinutes = document.getElementById("batch-opt-schedule-minutes");
    const btnSchedule = document.getElementById("btn-batch-schedule");

    // ----- Run bar -----
    const runLabel = document.getElementById("batch-run-label");
    const runEta = document.getElementById("batch-run-eta");
    const runFill = document.getElementById("batch-run-fill");
    const btnStart = document.getElementById("btn-batch-start");
    const btnPause = document.getElementById("btn-batch-pause");
    const btnResume = document.getElementById("btn-batch-resume");
    const btnCancel = document.getElementById("btn-batch-cancel");
    const btnRetryFailed = document.getElementById("btn-batch-retry-failed");
    const btnExportLog = document.getElementById("btn-batch-export-log");
    const btnCommit = document.getElementById("btn-batch-commit");
    const btnDiscard = document.getElementById("btn-batch-discard");
    const countPreviewed = document.getElementById("batch-count-previewed");

    // ----- Summary -----
    const summaryPanel = document.getElementById("batch-summary");
    const summaryStats = document.getElementById("batch-summary-stats");
    const summaryCategories = document.getElementById("batch-summary-categories");
    const summaryFailedWrap = document.getElementById("batch-summary-failed");
    const failedList = document.getElementById("batch-failed-list");

    // ----- Preview modal -----
    const previewOverlay = document.getElementById("batch-preview-overlay");
    const previewName = document.getElementById("batch-preview-name");
    const previewBefore = document.getElementById("batch-preview-before");
    const previewAfter = document.getElementById("batch-preview-after");
    const btnPreviewClose = document.getElementById("btn-batch-preview-close");

    let lastState = { items: [], options: {}, running: false, paused: false, summary: {} };
    let dragSrcId = null; // true drag-to-reorder state (see renderRow's dragstart/drop handlers)

    function showError(msg) {
        if (!batchError) return;
        batchError.textContent = msg;
        batchError.classList.remove("view--hidden");
        setTimeout(() => batchError.classList.add("view--hidden"), 6000);
    }

    function fileUrl(path) {
        // Same convention as editor.js/filters.js/removebg.js's toFileUrl:
        // Windows paths need forward slashes + a leading slash for a
        // valid file:// URL, and encodeURI handles spaces/special chars.
        if (!path) return "";
        const normalized = path.replace(/\\/g, "/").replace(/^\/+/, "");
        return "file:///" + encodeURI(normalized);
    }

    function formatBytes(n) {
        if (!n) return "0 B";
        const units = ["B", "KB", "MB", "GB"];
        let i = 0;
        let v = n;
        while (v >= 1024 && i < units.length - 1) {
            v /= 1024;
            i += 1;
        }
        return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
    }

    function formatEta(seconds) {
        if (!seconds || seconds <= 0) return "";
        const m = Math.floor(seconds / 60);
        const s = seconds % 60;
        return m > 0 ? `~${m}m ${s}s left` : `~${s}s left`;
    }

    const STATUS_LABEL = {
        queued: "Queued",
        processing: "Processing",
        previewed: "Previewed",
        done: "Done",
        failed: "Failed",
        skipped: "Skipped",
        cancelled: "Cancelled",
    };
    const STATUS_BADGE_CLASS = {
        queued: "badge--muted",
        processing: "badge--muted",
        previewed: "badge--muted",
        done: "badge--ok",
        failed: "badge--error",
        skipped: "badge--muted",
        cancelled: "badge--muted",
    };

    // ===================== RENDER =====================

    function renderRow(item) {
        const row = document.createElement("div");
        row.className = "batch-row";
        row.dataset.itemId = item.id;
        row.draggable = true;
        row.innerHTML = `
            <span class="batch-row-drag-handle" title="Drag to reorder">⠿</span>
            <label class="batch-row-include">
                <input type="checkbox" class="batch-row-checkbox" ${item.included ? "checked" : ""}>
            </label>
            <div class="batch-row-main">
                <div class="batch-row-name" title="${item.source_path}">
                    ${item.name}
                    ${item.duplicate_of ? '<span class="badge badge--muted batch-row-dupe" title="Looks similar to an earlier item">Possible duplicate</span>' : ""}
                </div>
                <div class="batch-row-meta">
                    <span>${formatBytes(item.size_bytes)}</span>
                    ${item.category ? `<span>· ${item.category}</span>` : ""}
                    ${item.applied_look ? `<span>· ${item.applied_look}</span>` : ""}
                    ${item.error ? `<span class="batch-row-error">· ${item.error}</span>` : ""}
                </div>
                <div class="batch-row-progress-track ${item.status === "processing" ? "" : "view--hidden"}">
                    <div class="batch-row-progress-fill" style="width:${item.progress}%;"></div>
                </div>
            </div>
            <span class="badge ${STATUS_BADGE_CLASS[item.status] || "badge--muted"} batch-row-status">${STATUS_LABEL[item.status] || item.status}</span>
            <div class="batch-row-actions">
                ${item.status === "done" || item.status === "previewed" ? '<button class="btn-icon batch-row-preview" title="Preview before/after">👁</button>' : ""}
                ${item.status === "failed" || item.status === "skipped" ? '<button class="btn-icon batch-row-retry" title="Retry">↺</button>' : ""}
                ${item.status === "queued" ? '<button class="btn-icon batch-row-skip" title="Skip">⤼</button>' : ""}
                <button class="btn-icon batch-row-remove" title="Remove from queue">✕</button>
            </div>
        `;

        row.querySelector(".batch-row-checkbox").addEventListener("change", (e) => {
            window.pixelforge.batch.setIncluded(item.id, e.target.checked, (state) => applyState(state));
        });
        row.querySelector(".batch-row-remove").addEventListener("click", () => {
            window.pixelforge.batch.removeItem(item.id, (state) => applyState(state));
        });
        const retryBtn = row.querySelector(".batch-row-retry");
        if (retryBtn) {
            retryBtn.addEventListener("click", () => {
                window.pixelforge.batch.retryItem(item.id, (state) => applyState(state));
            });
        }
        const skipBtn = row.querySelector(".batch-row-skip");
        if (skipBtn) {
            skipBtn.addEventListener("click", () => {
                window.pixelforge.batch.skipItem(item.id, () => window.pixelforge.batch.state((s) => applyState(s)));
            });
        }
        const previewBtn = row.querySelector(".batch-row-preview");
        if (previewBtn) {
            previewBtn.addEventListener("click", () => openPreview(item));
        }

        // ----- True drag-to-reorder (mouse drag, not just buttons) -----
        row.addEventListener("dragstart", (e) => {
            dragSrcId = item.id;
            row.classList.add("is-dragging");
            e.dataTransfer.effectAllowed = "move";
            e.dataTransfer.setData("text/plain", item.id); // Firefox needs data set to allow the drag
        });
        row.addEventListener("dragend", () => {
            dragSrcId = null;
            row.classList.remove("is-dragging");
            queueList.querySelectorAll(".batch-row").forEach((r) => r.classList.remove("is-drop-target"));
        });
        row.addEventListener("dragover", (e) => {
            if (!dragSrcId || dragSrcId === item.id) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
            row.classList.add("is-drop-target");
        });
        row.addEventListener("dragleave", () => row.classList.remove("is-drop-target"));
        row.addEventListener("drop", (e) => {
            if (!dragSrcId || dragSrcId === item.id) return;
            e.preventDefault();
            e.stopPropagation();
            row.classList.remove("is-drop-target");
            const before = e.clientY < row.getBoundingClientRect().top + row.offsetHeight / 2;
            const ids = lastState.items.map((i) => i.id).filter((id) => id !== dragSrcId);
            const targetIdx = ids.indexOf(item.id);
            ids.splice(before ? targetIdx : targetIdx + 1, 0, dragSrcId);
            window.pixelforge.batch.reorder(ids, (state) => applyState(state));
        });
        return row;
    }

    function renderQueue(items) {
        queueList.innerHTML = "";
        items.forEach((item) => queueList.appendChild(renderRow(item)));
        queueCount.textContent = String(items.length);
    }

    function patchRow(item) {
        const row = queueList.querySelector(`.batch-row[data-item-id="${item.id}"]`);
        if (!row) return;
        const fresh = renderRow(item);
        row.replaceWith(fresh);
    }

    function renderCounts(summary, items) {
        const queuedCount = items.filter((i) => i.status === "queued").length;
        const previewedCount = items.filter((i) => i.status === "previewed").length;
        countQueued.textContent = `${queuedCount} queued`;
        if (countPreviewed) {
            countPreviewed.textContent = `${previewedCount} previewed`;
            countPreviewed.classList.toggle("view--hidden", previewedCount === 0);
        }
        countDone.textContent = `${summary.processed || 0} done`;
        countFailed.textContent = `${summary.failed || 0} failed`;
        btnRetryFailed.disabled = !(summary.failed > 0);
    }

    function renderSummary(summary) {
        if (!summary || !(summary.processed || summary.failed || summary.skipped || summary.cancelled)) {
            summaryPanel.classList.add("view--hidden");
            return;
        }
        summaryPanel.classList.remove("view--hidden");
        summaryStats.innerHTML = `
            <div class="batch-stat"><span class="batch-stat-value">${summary.total || 0}</span><span>Images</span></div>
            <div class="batch-stat"><span class="batch-stat-value">${summary.processed || 0}</span><span>Processed</span></div>
            <div class="batch-stat"><span class="batch-stat-value">${summary.failed || 0}</span><span>Failed</span></div>
            <div class="batch-stat"><span class="batch-stat-value">${summary.skipped || 0}</span><span>Skipped</span></div>
        `;
        const cats = summary.categories || {};
        summaryCategories.innerHTML = Object.keys(cats).length
            ? Object.entries(cats).map(([name, count]) => `<span class="badge badge--muted">${name} · ${count}</span>`).join("")
            : "";
        if (summary.failed_items && summary.failed_items.length) {
            summaryFailedWrap.classList.remove("view--hidden");
            failedList.innerHTML = summary.failed_items
                .map((f) => `<div class="batch-failed-row"><span>${f.name}</span><span class="batch-row-error">${f.error}</span></div>`)
                .join("");
        } else {
            summaryFailedWrap.classList.add("view--hidden");
        }
    }

    function renderOptions(options) {
        if (!options) return;
        optMode.value = options.mode || "smart";
        presetRow.classList.toggle("view--hidden", optMode.value !== "preset");
        optIntensity.value = options.intensity != null ? options.intensity : 80;
        valIntensity.textContent = `${optIntensity.value}%`;
        optOutputFolder.value = options.output_folder || "";
        optFormat.value = options.export_format || "keep";
        optQuality.value = options.export_quality != null ? options.export_quality : 92;
        valQuality.textContent = `${optQuality.value}%`;
        optResizeMode.value = options.resize_mode || "none";
        resizeValueRow.classList.toggle("view--hidden", optResizeMode.value === "none");
        optResizeValue.value = options.resize_value || 2048;
        optNaming.value = options.naming_template || "{name}";
        optPreserve.checked = !!options.preserve_structure;
        optOverwrite.checked = !!options.overwrite;
        optStripExif.checked = !!options.strip_exif;
        optCopyright.value = options.copyright_author || "";
        optThrottle.value = options.cpu_throttle || 0;
        valThrottle.textContent = optThrottle.value == 0 ? "Off" : `${optThrottle.value}%`;
        optNotify.checked = options.notify_on_complete !== false;
        if (optDryRun) optDryRun.checked = !!options.dry_run;
    }

    function renderRunControls(state) {
        const running = !!state.running;
        const paused = !!state.paused;
        const previewedCount = state.items.filter((i) => i.status === "previewed").length;
        const queuedCount = state.items.filter((i) => i.status === "queued").length;
        const finishedCount = state.items.filter((i) => ["done", "failed", "skipped", "cancelled"].includes(i.status)).length;
        // Everything in the queue has already been run at least once and
        // nothing is left queued -- offer a one-click "Reprocess All"
        // instead of a dead "Start Batch" (this is what re-running the
        // same images with a changed Look/preset uses).
        const allFinished = !running && queuedCount === 0 && previewedCount === 0 && finishedCount > 0;

        btnStart.classList.toggle("view--hidden", running || previewedCount > 0);
        btnPause.classList.toggle("view--hidden", !running || paused);
        btnResume.classList.toggle("view--hidden", !running || !paused);
        btnCancel.classList.toggle("view--hidden", !running);
        btnCommit.classList.toggle("view--hidden", running || previewedCount === 0);
        btnDiscard.classList.toggle("view--hidden", running || previewedCount === 0);

        if (allFinished) {
            btnStart.textContent = "Reprocess All";
            btnStart.disabled = false;
        } else {
            btnStart.textContent = state.options && state.options.dry_run ? "Preview Batch" : "Start Batch";
            btnStart.disabled = queuedCount === 0;
        }
        btnStart.dataset.mode = allFinished ? "reprocess" : "start";

        if (!running) {
            if (finishedCount === 0 && previewedCount === 0) {
                runLabel.textContent = "Ready";
                runFill.style.width = "0%";
            } else if (previewedCount > 0 && finishedCount === 0) {
                runLabel.textContent = `${previewedCount} staged for review -- commit or discard below`;
            } else if (allFinished) {
                runLabel.textContent = `Done -- change options above and click "Reprocess All" to run again`;
            }
        }
    }

    function applyState(state) {
        if (!state || state.ok === false) {
            if (state && state.error) showError(state.error);
            return;
        }
        lastState = state;
        const hasItems = (state.items || []).length > 0;
        dropzone.classList.toggle("view--hidden", hasItems);
        workspace.classList.toggle("view--hidden", !hasItems);
        if (!hasItems) return;

        renderQueue(state.items);
        renderCounts(state.summary || {}, state.items);
        renderOptions(state.options);
        renderRunControls(state);
        renderSummary(state.summary);
        selectAll.checked = state.items.every((i) => i.included);
    }

    function refresh() {
        window.pixelforge.batch.state((state) => applyState(state));
    }

    // ===================== ADD IMAGES / FOLDER =====================

    function addFiles(paths) {
        if (!paths || !paths.length) return;
        window.pixelforge.batch.addFiles(paths, (state) => applyState(state));
    }
    window.pixelforgeBatchAddFiles = addFiles;

    function openFilesDialog() {
        window.pixelforge.openImagesDialog((paths) => addFiles(paths));
    }
    function openFolderDialog() {
        window.pixelforge.batch.chooseInputFolder((folder) => {
            if (!folder) return;
            window.pixelforge.batch.addFolder(folder, true, (state) => applyState(state));
        });
    }
    [btnAddFiles, btnAddFiles2].forEach((btn) => btn && btn.addEventListener("click", openFilesDialog));
    [btnAddFolder, btnAddFolder2].forEach((btn) => btn && btn.addEventListener("click", openFolderDialog));

    // ----- Drag & drop: individual files AND whole folders. Folders are
    // detected via the HTML5 DataTransferItem entry API (webkitGetAsEntry)
    // -- when the dropped item is a directory, its real OS path (exposed
    // by this app's Chromium/QtWebEngine shell the same way File.path is,
    // see editor.js/filters.js) is routed to addFolder instead of addFiles. -----
    [dropzone, workspace].forEach((zone) => {
        ["dragenter", "dragover"].forEach((evt) => zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("is-dragover"); }));
        ["dragleave", "drop"].forEach((evt) => zone.addEventListener(evt, () => zone.classList.remove("is-dragover")));
        zone.addEventListener("drop", (e) => {
            e.preventDefault();
            const dt = e.dataTransfer;
            if (!dt) return;

            const folderPaths = [];
            const filePaths = [];

            if (dt.items && dt.items.length) {
                Array.from(dt.items).forEach((item, idx) => {
                    const entry = item.webkitGetAsEntry && item.webkitGetAsEntry();
                    const file = item.getAsFile && item.getAsFile();
                    if (entry && entry.isDirectory && file && file.path) {
                        folderPaths.push(file.path);
                    } else if (file && file.path) {
                        filePaths.push(file.path);
                    }
                });
            } else if (dt.files && dt.files.length) {
                Array.from(dt.files).forEach((f) => { if (f.path) filePaths.push(f.path); });
            }

            if (folderPaths.length) {
                folderPaths.forEach((folder) => window.pixelforge.batch.addFolder(folder, true, (state) => applyState(state)));
            }
            if (filePaths.length) {
                addFiles(filePaths);
            }
            if (!folderPaths.length && !filePaths.length) {
                showError("Couldn't read the dropped items' location -- use Add Images/Add Folder instead.");
            }
        });
    });

    if (btnClear) {
        btnClear.addEventListener("click", () => {
            window.pixelforge.batch.clear((state) => applyState(state));
        });
    }

    if (selectAll) {
        selectAll.addEventListener("change", () => {
            const checked = selectAll.checked;
            const ids = lastState.items.map((i) => i.id);
            let remaining = ids.length;
            if (!remaining) return;
            ids.forEach((id) => {
                window.pixelforge.batch.setIncluded(id, checked, () => {
                    remaining -= 1;
                    if (remaining === 0) refresh();
                });
            });
        });
    }

    if (btnDetectDupes) {
        btnDetectDupes.addEventListener("click", () => {
            window.pixelforge.batch.detectDuplicates((state) => {
                applyState(state);
                const groups = (state && state.groups) || [];
                if (groups.length) {
                    dupWarning.textContent = `${groups.length} possible duplicate group${groups.length > 1 ? "s" : ""} found -- flagged items are badged in the queue below.`;
                    dupWarning.classList.remove("view--hidden");
                } else {
                    dupWarning.textContent = "No near-duplicates found.";
                    dupWarning.classList.remove("view--hidden");
                    setTimeout(() => dupWarning.classList.add("view--hidden"), 4000);
                }
            });
        });
    }

    // ===================== OPTIONS PANEL =====================

    function setOption(key, value) {
        window.pixelforge.batch.setOptions({ [key]: value }, (res) => {
            if (res && res.options) renderOptions(res.options);
        });
    }

    optMode.addEventListener("change", () => {
        presetRow.classList.toggle("view--hidden", optMode.value !== "preset");
        setOption("mode", optMode.value);
    });
    optPreset.addEventListener("change", () => setOption("preset_id", optPreset.value));
    optIntensity.addEventListener("input", () => { valIntensity.textContent = `${optIntensity.value}%`; });
    optIntensity.addEventListener("change", () => setOption("intensity", Number(optIntensity.value)));

    btnOutputFolder.addEventListener("click", () => {
        window.pixelforge.chooseOutputFolder((path) => {
            if (!path) return;
            optOutputFolder.value = path;
            setOption("output_folder", path);
        });
    });

    optFormat.addEventListener("change", () => setOption("export_format", optFormat.value));
    optQuality.addEventListener("input", () => { valQuality.textContent = `${optQuality.value}%`; });
    optQuality.addEventListener("change", () => setOption("export_quality", Number(optQuality.value)));

    optResizeMode.addEventListener("change", () => {
        resizeValueRow.classList.toggle("view--hidden", optResizeMode.value === "none");
        setOption("resize_mode", optResizeMode.value);
    });
    optResizeValue.addEventListener("change", () => setOption("resize_value", Number(optResizeValue.value)));

    optNaming.addEventListener("change", () => setOption("naming_template", optNaming.value || "{name}"));
    optPreserve.addEventListener("change", () => setOption("preserve_structure", optPreserve.checked));
    optOverwrite.addEventListener("change", () => setOption("overwrite", optOverwrite.checked));
    optStripExif.addEventListener("change", () => setOption("strip_exif", optStripExif.checked));
    optCopyright.addEventListener("change", () => setOption("copyright_author", optCopyright.value));

    optThrottle.addEventListener("input", () => {
        valThrottle.textContent = optThrottle.value == 0 ? "Off" : `${optThrottle.value}%`;
    });
    optThrottle.addEventListener("change", () => setOption("cpu_throttle", Number(optThrottle.value)));
    optNotify.addEventListener("change", () => setOption("notify_on_complete", optNotify.checked));
    if (optDryRun) {
        optDryRun.addEventListener("change", () => {
            setOption("dry_run", optDryRun.checked);
            btnStart.textContent = optDryRun.checked ? "Preview Batch" : "Start Batch";
        });
    }

    // ===================== DISK SPACE CHECK =====================

    function checkDiskSpace() {
        window.pixelforge.batch.checkDiskSpace((res) => {
            if (!res) return;
            if (res.warning) {
                diskWarning.textContent = `⚠ ${res.warning} (estimated ${formatBytes(res.estimated_bytes)}, ${formatBytes(res.free_bytes)} free).`;
                diskWarning.classList.remove("view--hidden");
            } else {
                diskWarning.classList.add("view--hidden");
            }
        });
    }

    // ===================== START / PAUSE / RESUME / CANCEL =====================

    function startBatch(delayMinutes) {
        checkDiskSpace();
        summaryPanel.classList.add("view--hidden");
        runLabel.textContent = delayMinutes ? `Starting in ${delayMinutes} min...` : "Starting...";
        const onDone = (result) => {
            if (result && result.ok === false) {
                showError(result.error || "Batch failed to run.");
            }
            applyState(result);
        };
        if (delayMinutes && delayMinutes > 0) {
            window.pixelforge.batch.scheduleStart(delayMinutes * 60, onDone);
        } else {
            window.pixelforge.batch.start({}, onDone);
        }
        refresh();
    }

    btnStart.addEventListener("click", () => {
        if (btnStart.dataset.mode === "reprocess") {
            window.pixelforge.batch.requeueAll(() => startBatch(0));
        } else {
            startBatch(0);
        }
    });
    btnSchedule.addEventListener("click", () => startBatch(Number(optScheduleMinutes.value) || 0));

    btnPause.addEventListener("click", () => {
        window.pixelforge.batch.pause();
        refresh();
    });
    btnResume.addEventListener("click", () => {
        window.pixelforge.batch.resume();
        refresh();
    });
    btnCancel.addEventListener("click", () => {
        window.pixelforge.batch.cancel();
    });

    btnRetryFailed.addEventListener("click", () => {
        window.pixelforge.batch.retryFailed((state) => applyState(state));
    });

    // "Preview before commit" -- finalize or throw away everything
    // currently staged (see core/batch_processor.py::commit_previewed).
    btnCommit.addEventListener("click", () => {
        window.pixelforge.batch.commitPreviewed(null, (state) => applyState(state));
    });
    btnDiscard.addEventListener("click", () => {
        window.pixelforge.batch.discardPreviewed(null, (state) => applyState(state));
    });

    btnExportLog.addEventListener("click", () => {
        window.pixelforge.batch.exportLog((res) => {
            if (!res || !res.ok) {
                if (res && res.error && res.error !== "Export cancelled.") showError(res.error);
                return;
            }
        });
    });

    // ===================== PREVIEW BEFORE/AFTER (spot-check) =====================

    function openPreview(item) {
        previewName.textContent = item.name;
        // Cache-busting timestamp: QtWebEngine caches file:// images by
        // URL, so re-previewing the same output path after a retry (or
        // committing a fresh preview to the same filename) would
        // otherwise keep showing the stale/previous bytes -- same fix
        // editor.js/filters.js/removebg.js already use everywhere they
        // load a just-written result.
        const stamp = `?t=${Date.now()}`;
        previewBefore.src = fileUrl(item.source_path) + stamp;
        previewAfter.src = fileUrl(item.output_path || item.staged_path || item.source_path) + stamp;
        previewOverlay.classList.remove("view--hidden");
    }
    if (btnPreviewClose) {
        btnPreviewClose.addEventListener("click", () => previewOverlay.classList.add("view--hidden"));
    }
    if (previewOverlay) {
        previewOverlay.addEventListener("click", (e) => {
            if (e.target === previewOverlay) previewOverlay.classList.add("view--hidden");
        });
    }

    // ===================== LIVE UPDATES WHILE RUNNING =====================

    if (window.onPixelforgeReady) {
        window.onPixelforgeReady(() => {
            // Preset dropdown, same source as Filters view.
            window.pixelforge.listFilterPresets((res) => {
                if (!res || !res.presets) return;
                optPreset.innerHTML = res.presets
                    .map((p) => `<option value="${p.id}">${p.name}</option>`)
                    .join("");
            });

            window.pixelforge.batch.onItem((item) => {
                patchRow(item);
                const items = lastState.items.map((i) => (i.id === item.id ? item : i));
                lastState.items = items;
                renderCounts(lastState.summary || {}, items);
            });

            window.pixelforge.batch.onProgress((pct, label, eta) => {
                runLabel.textContent = label || "Processing...";
                runEta.textContent = formatEta(eta);
                runFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
            });

            window.pixelforge.batch.onFinished((result) => {
                applyState(result);
            });

            refresh();

            // PHASE 14: crash recovery. Only relevant right after a
            // fresh launch that follows an unclean shutdown mid-batch
            // (core/batch_processor.py only ever writes this snapshot
            // while a run is actually in flight, and clears it itself
            // on any normal finish/clear) -- so an empty {} response
            // here is the overwhelmingly common case and nothing shows.
            if (window.pixelforge.batch.getRecoveryState) {
                window.pixelforge.batch.getRecoveryState((state) => {
                    const items = state && state.items;
                    if (!items || !items.length) return;
                    const resume = window.confirm(
                        `PixelForge closed unexpectedly during a batch with ${items.length} item(s) still pending. Add them back to the queue to continue?`
                    );
                    if (resume) {
                        const paths = items.map((i) => i.source_path).filter(Boolean);
                        window.pixelforge.batch.addFiles(paths, (res) => {
                            if (res) applyState(res);
                            window.pixelforge.batch.discardRecoveryState();
                        });
                    } else {
                        window.pixelforge.batch.discardRecoveryState();
                    }
                });
            }
        });
    }
});