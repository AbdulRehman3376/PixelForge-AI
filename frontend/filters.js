// PHASE 4 -- Presets / Smart Filters view logic.
//
// Same dropzone -> workspace shell and before/after compare pattern as
// editor.js/removebg.js (see those files for the original comments on
// why each piece works the way it does -- kept consistent here rather
// than re-explaining). What's new for this view:
//   - a preset grid (9 builtins + custom) fetched from
//     pyBridge.listFilterPresets() and rendered into #filter-preset-grid
//   - an Intensity slider that blends a preset toward "no effect"
//     (see core/filters.py::_scaled_adjustments for why that's always
//     safe)
//   - Save/Rename/Duplicate/Delete/Export/Import for custom presets
//
// Exposes window.pixelforgeLoadImageIntoFilters(path), same handoff
// pattern ui.js already uses for Enhance/Remove BG. Also exposes
// window.pixelforgeApplySmartPipeline(presetId, intensity) for Phase 6
// -- see the "Phase 6: Smart Pipeline hand-off" section below.

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("filters-dropzone");
    const workspace = document.getElementById("filters-workspace");
    if (!dropzone || !workspace) return; // view not present in this build

    const btnOpenEmpty = document.getElementById("btn-filters-open-empty");
    const btnReplace = document.getElementById("btn-filters-replace");
    const canvasWrap = document.getElementById("filters-canvas-wrap");
    const canvas = document.getElementById("filters-canvas");
    const filtersWorkspace = document.getElementById("filters-workspace");
    const btnFullscreenExit = document.getElementById("btn-filters-fullscreen-exit");
    const zoomLevelEl = document.getElementById("filters-zoom-level");
    const btnZoomIn = document.getElementById("btn-filters-zoom-in");
    const btnZoomOut = document.getElementById("btn-filters-zoom-out");
    const btnZoomFit = document.getElementById("btn-filters-zoom-fit");
    const btnZoomReset = document.getElementById("btn-filters-zoom-reset");
    const btnFullscreen = document.getElementById("btn-filters-fullscreen");
    const imgBefore = document.getElementById("filters-image-before");
    const imgAfter = document.getElementById("filters-image-after");
    const afterWrap = document.getElementById("filters-image-after-wrap");
    const compareHandle = document.getElementById("filters-compare-handle");
    const compareSlider = document.getElementById("filters-compare-slider");
    const fileNameEl = document.getElementById("filters-filename");
    const dimensionsEl = document.getElementById("filters-dimensions");
    const previewSpinner = document.getElementById("filters-preview-spinner");
    const filtersError = document.getElementById("filters-error");

    const categoryTabsWrap = document.getElementById("filter-category-tabs");
    const categoryTabs = categoryTabsWrap ? Array.from(categoryTabsWrap.querySelectorAll(".bg-mode-tab")) : [];
    const presetGrid = document.getElementById("filter-preset-grid");
    const emptyNote = document.getElementById("filters-empty-note");
    const searchInput = document.getElementById("filter-search-input");

    const ctrlIntensity = document.getElementById("ctrl-filter-intensity");
    const valIntensity = document.getElementById("val-filter-intensity");

    const inputPresetName = document.getElementById("input-preset-name");
    const btnSavePreset = document.getElementById("btn-save-preset");
    const customPresetActions = document.getElementById("custom-preset-actions");
    const btnPresetRename = document.getElementById("btn-preset-rename");
    const btnPresetDuplicate = document.getElementById("btn-preset-duplicate");
    const btnPresetDelete = document.getElementById("btn-preset-delete");
    const btnPresetExport = document.getElementById("btn-preset-export");
    const btnPresetImport = document.getElementById("btn-preset-import");

    const btnExport = document.getElementById("btn-filters-export");
    const btnApply = document.getElementById("btn-filters-apply"); // PHASE 6

    // ----- PHASE 4: Auto Filter -----
    const btnAutoFilter = document.getElementById("btn-auto-filter");
    const autoFilterSummary = document.getElementById("auto-filter-summary");

    // ----- PHASE 4: Filter Stacking -----
    const filterStackPanel = document.getElementById("filter-stack-panel");
    const filterStackChips = document.getElementById("filter-stack-chips");
    const btnStackClear = document.getElementById("btn-stack-clear");
    let stackedPresetIds = [];

    // ----- PHASE 4: Filter Comparison -----
    const filterComparePanel = document.getElementById("filter-compare-panel");
    const compareNote = document.getElementById("filter-compare-note");
    const btnCompareSelected = document.getElementById("btn-compare-selected");
    const compareOverlay = document.getElementById("filter-compare-overlay");
    const compareStrip = document.getElementById("filter-compare-strip");
    const btnCompareClose = document.getElementById("btn-compare-close");
    let compareSelectedIds = [];

    // ----- PHASE 4: Custom filter live preview (banate waqt) -----
    const btnToggleCustomBuilder = document.getElementById("btn-toggle-custom-builder");
    const customFilterBuilder = document.getElementById("custom-filter-builder");
    const cfSliderIds = ["contrast", "saturation", "temperature", "vibrance", "clarity", "vignette", "smoke", "grain"];
    const cfSliders = {};
    cfSliderIds.forEach((id) => { cfSliders[id] = document.getElementById(`ctrl-cf-${id}`); });
    const btnCfMonochrome = document.getElementById("btn-cf-monochrome");
    const inputCfName = document.getElementById("input-cf-name");
    const btnCfSave = document.getElementById("btn-cf-save");
    let cfMonochrome = false;
    let customBuilderOpen = false;

    // Mirrors core/enhancer.py::DEFAULT_ADJUSTMENTS -- needed client-side
    // only for "Save as Preset" (blending the selected preset's values
    // toward these defaults by the current Intensity, same formula as
    // core/filters.py::_scaled_adjustments, so the saved preset bakes in
    // whatever strength was on screen when Save was clicked).
    const DEFAULT_ADJUSTMENTS = {
        brightness: 100, contrast: 100, saturation: 100,
        exposure: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
        temperature: 0, tint: 0, vibrance: 0,
        sharpness: 0, clarity: 0, noise_reduction: 0, vignette: 0, smoke: 0,
    };

    let currentPath = null;
    let naturalWidth = 0;
    let naturalHeight = 0;
    let zoom = 1;

    let allPresets = [];       // full list from listFilterPresets()
    let activeCategory = "all";
    let selectedPresetId = null;
    let searchQuery = "";

    // ----- Helpers (same shape as editor.js) -----

    function toFileUrl(path) {
        const normalized = path.replace(/\\/g, "/").replace(/^\/+/, "");
        return "file:///" + encodeURI(normalized);
    }

    function showError(message) {
        if (!filtersError) return;
        filtersError.textContent = message;
        filtersError.classList.toggle("view--hidden", !message);
    }

    function sizeStageToImage() {
        if (!naturalWidth || !naturalHeight) return;
        canvas.style.width = `${naturalWidth}px`;
        canvas.style.height = `${naturalHeight}px`;
    }

    function applyZoom() {
        canvas.style.transform = `translate(-50%, -50%) scale(${zoom})`;
        if (zoomLevelEl) zoomLevelEl.textContent = `${Math.round(zoom * 100)}%`;
    }

    function setZoom(next) {
        zoom = Math.min(4, Math.max(0.1, next));
        applyZoom();
    }

    function fitToContainer() {
        if (!naturalWidth || !naturalHeight) return;
        const rect = canvasWrap.getBoundingClientRect();
        const padding = 32;
        const scaleX = (rect.width - padding) / naturalWidth;
        const scaleY = (rect.height - padding) / naturalHeight;
        setZoom(Math.min(scaleX, scaleY, 1));
    }

    function setComparePosition(percent) {
        const clamped = Math.min(100, Math.max(0, percent));
        afterWrap.style.clipPath = `inset(0 ${100 - clamped}% 0 0)`;
        compareHandle.style.left = `${clamped}%`;
        compareSlider.value = clamped;
    }

    function getPresetById(id) {
        return allPresets.find((p) => p.id === id) || null;
    }

    // ----- Preset grid rendering -----

    function categoryMatches(preset) {
        if (activeCategory === "all") return true;
        if (activeCategory === "favorites") return preset.is_favorite;
        if (activeCategory === "custom") return !preset.is_builtin;
        // Any other tab value is a specific builtin preset id (Cinematic,
        // Portrait, Vintage, Moody, Luxury, Film, Travel, Night,
        // Black & White) -- a custom preset only shows there if it was
        // duplicated from that builtin (same id prefix isn't tracked, so
        // this only ever matches the single builtin itself, which is the
        // expected "jump straight to this look" behavior for that tab).
        return preset.id === activeCategory;
    }

    function searchMatches(preset) {
        if (!searchQuery) return true;
        const q = searchQuery.toLowerCase();
        return (
            preset.name.toLowerCase().includes(q) ||
            (preset.description || "").toLowerCase().includes(q)
        );
    }

    function renderPresetGrid() {
        if (!presetGrid) return;
        presetGrid.innerHTML = "";
        const visible = allPresets.filter((p) => categoryMatches(p) && searchMatches(p));

        if (emptyNote) {
            emptyNote.classList.toggle("view--hidden", visible.length > 0);
            if (visible.length === 0) {
                if (searchQuery) {
                    emptyNote.textContent = `No filters match "${searchQuery}".`;
                } else if (activeCategory === "custom") {
                    emptyNote.textContent = "No custom presets yet -- select a preset, adjust Intensity, then Save it below.";
                } else if (activeCategory === "favorites") {
                    emptyNote.textContent = "No favorites yet -- tap the ☆ on any preset to add it here.";
                } else {
                    emptyNote.textContent = "No presets in this category yet.";
                }
            }
        }

        visible.forEach((preset) => {
            const card = document.createElement("button");
            card.type = "button";
            card.className = "filter-preset-card" + (preset.id === selectedPresetId ? " active" : "");
            card.dataset.presetId = preset.id;

            const thumb = document.createElement("div");
            thumb.className = "filter-preset-thumb";
            thumb.dataset.thumb = preset.id;
            // Once a photo is loaded, swap the flat gradient swatch for the
            // actual photo with a CSS filter approximating that preset (see
            // main.css [data-thumb-live="1"] rules) -- a real live thumbnail
            // instead of a generic color hint, with no per-card backend
            // render needed.
            if (currentPath) {
                thumb.dataset.thumbLive = "1";
                thumb.style.backgroundImage = `url("${toFileUrl(currentPath)}")`;
            }

            const nameRow = document.createElement("div");
            nameRow.className = "filter-preset-name";
            const nameSpan = document.createElement("span");
            nameSpan.textContent = preset.name;
            nameRow.appendChild(nameSpan);
            if (!preset.is_builtin) {
                const badge = document.createElement("span");
                badge.className = "filter-preset-badge";
                badge.textContent = "Custom";
                nameRow.appendChild(badge);
            }
            const favBtn = document.createElement("button");
            favBtn.type = "button";
            favBtn.className = "filter-preset-favorite" + (preset.is_favorite ? " is-favorite" : "");
            favBtn.textContent = preset.is_favorite ? "★" : "☆";
            favBtn.title = preset.is_favorite ? "Remove from favorites" : "Add to favorites";
            favBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                if (!window.pixelforge) return;
                const nextFavorite = !preset.is_favorite;
                window.pixelforge.setPresetFavorite(preset.id, nextFavorite, (result) => {
                    if (!result.ok) {
                        showError(result.error || "Couldn't update favorite.");
                        return;
                    }
                    preset.is_favorite = nextFavorite;
                    renderPresetGrid();
                });
            });
            nameRow.appendChild(favBtn);

            // PHASE 4: Filter Stacking -- "＋Stack" toggles this preset's
            // membership in stackedPresetIds. PHASE 4: Filter Comparison
            // -- "⇄" toggles membership in compareSelectedIds (max 4).
            // Both are independent of the normal card click (which still
            // does the original single-select-to-preview behavior).
            const stackBtn = document.createElement("button");
            stackBtn.type = "button";
            stackBtn.className = "filter-preset-stack-btn" + (stackedPresetIds.includes(preset.id) ? " is-active" : "");
            stackBtn.textContent = stackedPresetIds.includes(preset.id) ? "✓ Stacked" : "＋ Stack";
            stackBtn.title = "Add to / remove from the Filter Stack";
            stackBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                toggleStackMembership(preset.id);
            });

            const compareBtn = document.createElement("button");
            compareBtn.type = "button";
            compareBtn.className = "filter-preset-compare-btn" + (compareSelectedIds.includes(preset.id) ? " is-active" : "");
            compareBtn.textContent = "⇄";
            compareBtn.title = "Select for side-by-side comparison";
            compareBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                toggleCompareMembership(preset.id);
            });

            const miniActions = document.createElement("div");
            miniActions.className = "filter-preset-mini-actions";
            miniActions.style.cssText = "display:flex; gap:6px; margin-top:4px;";
            miniActions.appendChild(stackBtn);
            miniActions.appendChild(compareBtn);

            const desc = document.createElement("div");
            desc.className = "filter-preset-desc";
            desc.textContent = preset.description || "";

            card.appendChild(thumb);
            card.appendChild(nameRow);
            card.appendChild(desc);
            card.appendChild(miniActions);
            card.addEventListener("click", () => selectPreset(preset.id));

            presetGrid.appendChild(card);
        });
    }

    // ----- PHASE 4: Filter Stacking -----

    function renderStackChips() {
        if (!filterStackChips) return;
        filterStackChips.innerHTML = "";
        stackedPresetIds.forEach((id) => {
            const preset = getPresetById(id);
            const chip = document.createElement("span");
            chip.className = "filter-stack-chip";
            const label = document.createElement("span");
            label.textContent = preset ? preset.name : id;
            const removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.textContent = "✕";
            removeBtn.addEventListener("click", () => toggleStackMembership(id));
            chip.appendChild(label);
            chip.appendChild(removeBtn);
            filterStackChips.appendChild(chip);
        });
        if (filterStackPanel) filterStackPanel.classList.toggle("view--hidden", stackedPresetIds.length === 0);
    }

    function toggleStackMembership(presetId) {
        const idx = stackedPresetIds.indexOf(presetId);
        if (idx === -1) {
            stackedPresetIds.push(presetId);
        } else {
            stackedPresetIds.splice(idx, 1);
        }
        renderStackChips();
        renderPresetGrid();
        // Stacking (2+) takes over the main preview and Export; Apply
        // (Phase 6 session-commit) only supports a single preset, so
        // it's disabled while a 2+ stack is active.
        if (stackedPresetIds.length >= 2) {
            if (btnExport) btnExport.disabled = false;
            if (btnApply) btnApply.disabled = true;
        } else {
            if (btnExport) btnExport.disabled = !selectedPresetId;
            if (btnApply) btnApply.disabled = !selectedPresetId;
        }
        requestPreviewUpdate();
    }

    if (btnStackClear) {
        btnStackClear.addEventListener("click", () => {
            stackedPresetIds = [];
            renderStackChips();
            renderPresetGrid();
            requestPreviewUpdate();
        });
    }

    // ----- PHASE 4: Filter Comparison -----

    function toggleCompareMembership(presetId) {
        const idx = compareSelectedIds.indexOf(presetId);
        if (idx === -1) {
            if (compareSelectedIds.length >= 4) {
                showError("You can compare up to 4 filters at a time.");
                return;
            }
            compareSelectedIds.push(presetId);
        } else {
            compareSelectedIds.splice(idx, 1);
        }
        renderPresetGrid();
        if (compareNote) {
            compareNote.textContent = compareSelectedIds.length >= 2
                ? `${compareSelectedIds.length} filters selected -- ready to compare.`
                : "Select 2-4 filters (⇄ on each card) to compare.";
        }
        if (btnCompareSelected) btnCompareSelected.disabled = compareSelectedIds.length < 2;
        if (filterComparePanel) filterComparePanel.classList.toggle("view--hidden", compareSelectedIds.length === 0);
    }

    if (btnCompareSelected) {
        btnCompareSelected.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge || compareSelectedIds.length < 2) return;
            const intensity = Number(ctrlIntensity.value) || 100;
            btnCompareSelected.disabled = true;
            btnCompareSelected.textContent = "Rendering...";
            window.pixelforge.previewFilterBatch(currentPath, compareSelectedIds, intensity, (result) => {
                btnCompareSelected.disabled = false;
                btnCompareSelected.textContent = "Compare Selected";
                if (!result || !result.ok) {
                    showError((result && result.error) || "Comparison failed.");
                    return;
                }
                if (compareStrip) {
                    compareStrip.innerHTML = "";
                    (result.results || []).forEach((r) => {
                        const preset = getPresetById(r.preset_id);
                        const item = document.createElement("div");
                        item.className = "filter-compare-item";
                        const img = document.createElement("img");
                        img.src = toFileUrl(r.path) + `?t=${Date.now()}`;
                        const label = document.createElement("span");
                        label.textContent = preset ? preset.name : r.preset_id;
                        item.appendChild(img);
                        item.appendChild(label);
                        item.addEventListener("click", () => {
                            selectPreset(r.preset_id);
                            if (compareOverlay) compareOverlay.classList.add("view--hidden");
                        });
                        compareStrip.appendChild(item);
                    });
                }
                if (compareOverlay) compareOverlay.classList.remove("view--hidden");
            });
        });
    }

    if (btnCompareClose) {
        btnCompareClose.addEventListener("click", () => {
            if (compareOverlay) compareOverlay.classList.add("view--hidden");
        });
    }

    function loadPresetList(cb) {
        if (!window.pixelforge) return;
        window.pixelforge.listFilterPresets((result) => {
            if (!result.ok) {
                showError(result.error || "Couldn't load presets.");
                return;
            }
            allPresets = result.presets || [];
            renderPresetGrid();
            if (cb) cb();
        });
    }

    // ----- Selecting a preset -----

    function selectPreset(id) {
        selectedPresetId = id;
        const preset = getPresetById(id);

        if (ctrlIntensity) ctrlIntensity.disabled = !preset;
        if (btnExport) btnExport.disabled = !preset;
        if (btnApply) btnApply.disabled = !preset;
        if (btnSavePreset) btnSavePreset.disabled = !preset;
        if (customPresetActions) {
            customPresetActions.classList.toggle("view--hidden", !preset || preset.is_builtin);
        }

        renderPresetGrid();
        requestPreviewUpdate();
    }

    // ----- Live preview -----

    let previewDebounceTimer = null;
    let previewRequestSeq = 0;

    function requestPreviewUpdate() {
        if (!currentPath || !window.pixelforge) return;

        // PHASE 4: Filter Stacking -- 2+ stacked presets override the
        // normal single-preset preview entirely.
        if (stackedPresetIds.length >= 2) {
            clearTimeout(previewDebounceTimer);
            if (previewSpinner) previewSpinner.classList.add("is-visible");
            const intensity = Number(ctrlIntensity.value);
            const layers = stackedPresetIds.map((id) => ({ preset_id: id, intensity }));
            previewDebounceTimer = setTimeout(() => {
                const seq = ++previewRequestSeq;
                window.pixelforge.previewPresetStack(currentPath, layers, (result) => {
                    if (seq !== previewRequestSeq) return;
                    if (previewSpinner) previewSpinner.classList.remove("is-visible");
                    if (!result.ok) { showError(result.error || "Stack preview failed."); return; }
                    imgAfter.src = toFileUrl(result.path) + `?t=${Date.now()}`;
                });
            }, 180);
            return;
        }

        if (!selectedPresetId) {
            // No preset selected -- "after" is just the original photo.
            imgAfter.src = toFileUrl(currentPath) + `?t=${Date.now()}`;
            return;
        }

        clearTimeout(previewDebounceTimer);
        if (previewSpinner) previewSpinner.classList.add("is-visible");

        const intensity = Number(ctrlIntensity.value);
        previewDebounceTimer = setTimeout(() => {
            const seq = ++previewRequestSeq;
            window.pixelforge.previewFilter(currentPath, selectedPresetId, intensity, (result) => {
                if (seq !== previewRequestSeq) return; // superseded by a newer request
                if (previewSpinner) previewSpinner.classList.remove("is-visible");
                if (!result.ok) {
                    showError(result.error || "Preview failed.");
                    return;
                }
                imgAfter.src = toFileUrl(result.path) + `?t=${Date.now()}`;
            });
        }, 180);
    }

    // ----- Loading an image -----

    function loadImage(path, onLoaded) {
        window.pixelforgeCurrentImage = path;
        // PHASE 6: see VIEW_CURRENT_PATH_VARS in ui.js.
        window.pixelforgeFiltersCurrentPath = path;
        if (!path) return;
        showError("");

        if (!window.pixelforge) return;
        window.pixelforge.getImageInfo(path, (info) => {
            if (!info.ok) {
                showError(info.error || "Couldn't open that image.");
                return;
            }
            currentPath = info.path;
            naturalWidth = info.width;
            naturalHeight = info.height;
            sizeStageToImage();

            const url = toFileUrl(info.path);
            imgBefore.onerror = () => showError("Image failed to load in the preview. Check the file isn't moved/renamed and try again.");
            imgBefore.onload = () => showError("");
            imgBefore.src = url;
            imgAfter.src = url;

            fileNameEl.textContent = info.name;
            dimensionsEl.textContent = `${info.width} × ${info.height} · ${info.format}`;

            dropzone.classList.add("view--hidden");
            workspace.classList.remove("view--hidden");

            selectedPresetId = null;
            if (ctrlIntensity) { ctrlIntensity.value = 100; ctrlIntensity.disabled = true; }
            if (valIntensity) valIntensity.textContent = "100%";
            if (btnExport) btnExport.disabled = true;
            if (btnApply) btnApply.disabled = true;
            if (btnSavePreset) btnSavePreset.disabled = true;
            if (customPresetActions) customPresetActions.classList.add("view--hidden");
            setComparePosition(50);

            if (allPresets.length) {
                renderPresetGrid();
            } else {
                loadPresetList();
            }

            requestAnimationFrame(fitToContainer);
            if (typeof onLoaded === "function") onLoaded();
        });
    }
    window.pixelforgeLoadImageIntoFilters = loadImage;

    // ----- Phase 6: Smart Pipeline hand-off -----
    // Called by editor.js's "Apply in Filters" button after a Smart
    // Pipeline recommendation is accepted (see core/smart_pipeline.py /
    // ui/bridge.py::smartPipelineAsync). Selects the recommended preset
    // at the recommended intensity and kicks off the normal live
    // preview -- same selectPreset()/requestPreviewUpdate() path a user
    // clicking a preset card themselves would trigger, so nothing here
    // needs its own image-processing logic. If this view's image isn't
    // loaded yet (or is a different photo than the one currently
    // showing), the photo is loaded first and the preset applied once
    // that finishes -- same hand-off pattern ui.js's
    // maybeAutoLoadCurrentImage() already uses elsewhere.
    window.pixelforgeApplySmartPipeline = function (presetId, intensity) {
        const targetPath = window.pixelforgeCurrentImage;
        if (!targetPath || !presetId) return;

        function applyNow() {
            const clampedIntensity = Math.max(0, Math.min(100, Math.round(intensity)));
            if (ctrlIntensity) {
                ctrlIntensity.value = clampedIntensity;
                ctrlIntensity.disabled = false;
            }
            if (valIntensity) valIntensity.textContent = `${clampedIntensity}%`;
            selectPreset(presetId);
        }

        if (allPresets.length) {
            if (currentPath === targetPath) {
                applyNow();
            } else {
                loadImage(targetPath, applyNow);
            }
        } else {
            // Preset list hasn't been fetched in this view yet -- load
            // it first so getPresetById()/selectPreset() inside applyNow
            // has something to select from.
            loadPresetList(() => {
                if (currentPath === targetPath) {
                    applyNow();
                } else {
                    loadImage(targetPath, applyNow);
                }
            });
        }
    };

    // ----- Open via dialog -----

    function openViaDialog() {
        if (!window.pixelforge) return;
        window.pixelforge.openImageDialog((path) => {
            if (!path) return;
            if (window.pixelforgeStartSession) {
                window.pixelforgeStartSession(path, (state) => loadImage((state && state.working_path) || path));
            } else {
                loadImage(path);
            }
        });
    }

    if (btnOpenEmpty) btnOpenEmpty.addEventListener("click", openViaDialog);
    if (btnReplace) btnReplace.addEventListener("click", openViaDialog);

    // ----- Drag & drop (same file.path assumption as editor.js) -----

    ["dragenter", "dragover"].forEach((evt) => {
        dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("is-dragover"); });
        workspace.addEventListener(evt, (e) => { e.preventDefault(); workspace.classList.add("is-dragover"); });
    });
    ["dragleave", "drop"].forEach((evt) => {
        dropzone.addEventListener(evt, () => dropzone.classList.remove("is-dragover"));
        workspace.addEventListener(evt, () => workspace.classList.remove("is-dragover"));
    });
    function handleDrop(e) {
        e.preventDefault();
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (!file) return;
        if (file.path) {
            // PHASE 6: a drop here is a new original image too -- must
            // start a session, same reasoning as editor.js.
            if (window.pixelforgeStartSession) {
                window.pixelforgeStartSession(file.path, (state) => loadImage((state && state.working_path) || file.path));
            } else {
                loadImage(file.path);
            }
        } else {
            showError("Couldn't read the dropped file's location -- use Open Image instead.");
        }
    }
    dropzone.addEventListener("drop", handleDrop);
    workspace.addEventListener("drop", handleDrop);

    // ----- Category tabs -----

    categoryTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            activeCategory = tab.dataset.filterCategory;
            categoryTabs.forEach((t) => t.classList.toggle("active", t === tab));
            renderPresetGrid();
        });
    });

    // ----- Search -----

    if (searchInput) {
        searchInput.addEventListener("input", () => {
            searchQuery = searchInput.value.trim();
            renderPresetGrid();
        });
    }

    // ----- Intensity slider -----

    if (ctrlIntensity) {
        ctrlIntensity.addEventListener("input", () => {
            if (valIntensity) valIntensity.textContent = `${ctrlIntensity.value}%`;
            requestPreviewUpdate();
        });
    }

    // ----- Before/After compare -----

    let draggingHandle = false;
    function positionFromEvent(clientX) {
        const rect = canvasWrap.getBoundingClientRect();
        const percent = ((clientX - rect.left) / rect.width) * 100;
        setComparePosition(percent);
    }
    if (compareHandle) {
        compareHandle.addEventListener("mousedown", () => (draggingHandle = true));
        window.addEventListener("mouseup", () => (draggingHandle = false));
        window.addEventListener("mousemove", (e) => {
            if (draggingHandle) positionFromEvent(e.clientX);
        });
    }
    if (compareSlider) {
        compareSlider.addEventListener("input", () => setComparePosition(Number(compareSlider.value)));
    }

    window.addEventListener("resize", () => {
        if (!workspace.classList.contains("view--hidden")) fitToContainer();
    });

    // ----- Zoom / fit / reset / fullscreen -----

    if (btnZoomIn) btnZoomIn.addEventListener("click", () => setZoom(zoom + 0.1));
    if (btnZoomOut) btnZoomOut.addEventListener("click", () => setZoom(zoom - 0.1));
    if (btnZoomFit) btnZoomFit.addEventListener("click", fitToContainer);
    if (btnZoomReset) btnZoomReset.addEventListener("click", () => setZoom(1));

    if (btnFullscreen) {
        btnFullscreen.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                canvasWrap.requestFullscreen().catch(() => {
                    /* Fullscreen not permitted in this context -- ignore. */
                });
            } else {
                document.exitFullscreen();
            }
        });
    }

    // Same bugfix as editor.js: mark the rest of the UI hidden ourselves
    // on fullscreen change, since QtWebEngine doesn't fully suppress
    // painting of elements outside the fullscreened node.
    document.addEventListener("fullscreenchange", () => {
        if (filtersWorkspace) {
            filtersWorkspace.classList.toggle("is-fullscreen-active", !!document.fullscreenElement);
        }
    });

    if (btnFullscreenExit) {
        btnFullscreenExit.addEventListener("click", () => {
            if (document.fullscreenElement) document.exitFullscreen();
        });
    }

    // ----- Save current preset+intensity as a new custom preset -----

    if (btnSavePreset) {
        btnSavePreset.addEventListener("click", () => {
            const preset = getPresetById(selectedPresetId);
            if (!preset || !window.pixelforge) return;
            const name = (inputPresetName.value || "").trim();
            if (!name) {
                showError("Give the preset a name before saving.");
                return;
            }

            const t = Math.max(0, Math.min(100, Number(ctrlIntensity.value))) / 100;
            const scaledAdjustments = {};
            Object.entries(preset.adjustments || {}).forEach(([key, presetVal]) => {
                const defaultVal = DEFAULT_ADJUSTMENTS[key];
                scaledAdjustments[key] = defaultVal + (presetVal - defaultVal) * t;
            });
            const scaledGrain = Math.round((preset.grain || 0) * t);

            window.pixelforge.saveCustomPreset(
                name,
                { adjustments: scaledAdjustments, grain: scaledGrain, monochrome: !!preset.monochrome },
                (result) => {
                    if (!result.ok) {
                        showError(result.error || "Couldn't save preset.");
                        return;
                    }
                    inputPresetName.value = "";
                    showError("");
                    loadPresetList(() => selectPreset(result.preset.id));
                }
            );
        });
    }

    // ----- Custom preset actions -----

    if (btnPresetRename) {
        btnPresetRename.addEventListener("click", () => {
            const preset = getPresetById(selectedPresetId);
            if (!preset || !window.pixelforge) return;
            const newName = window.prompt("Rename preset", preset.name);
            if (!newName) return;
            window.pixelforge.renameCustomPreset(preset.id, newName, (result) => {
                if (!result.ok) { showError(result.error || "Rename failed."); return; }
                loadPresetList(() => selectPreset(preset.id));
            });
        });
    }

    if (btnPresetDuplicate) {
        btnPresetDuplicate.addEventListener("click", () => {
            if (!selectedPresetId || !window.pixelforge) return;
            window.pixelforge.duplicateCustomPreset(selectedPresetId, (result) => {
                if (!result.ok) { showError(result.error || "Duplicate failed."); return; }
                loadPresetList(() => selectPreset(result.preset.id));
            });
        });
    }

    if (btnPresetDelete) {
        btnPresetDelete.addEventListener("click", () => {
            const preset = getPresetById(selectedPresetId);
            if (!preset || !window.pixelforge) return;
            if (!window.confirm(`Delete "${preset.name}"? This can't be undone.`)) return;
            window.pixelforge.deleteCustomPreset(preset.id, (result) => {
                if (!result.ok) { showError(result.error || "Delete failed."); return; }
                selectedPresetId = null;
                loadPresetList(() => {
                    if (ctrlIntensity) ctrlIntensity.disabled = true;
                    if (btnExport) btnExport.disabled = true;
                    if (btnSavePreset) btnSavePreset.disabled = true;
                    if (customPresetActions) customPresetActions.classList.add("view--hidden");
                    requestPreviewUpdate();
                });
            });
        });
    }

    if (btnPresetExport) {
        btnPresetExport.addEventListener("click", () => {
            const preset = getPresetById(selectedPresetId);
            if (!preset || !window.pixelforge) return;
            const suggested = `${preset.name.replace(/[^a-z0-9]+/gi, "_").toLowerCase()}.json`;
            window.pixelforge.exportPresetToFile(preset.id, suggested, (result) => {
                if (!result.ok) showError(result.error || "Export failed.");
            });
        });
    }

    if (btnPresetImport) {
        btnPresetImport.addEventListener("click", () => {
            if (!window.pixelforge) return;
            window.pixelforge.importPresetFromFile((result) => {
                if (!result.ok) { showError(result.error || "Import failed."); return; }
                loadPresetList(() => {
                    activeCategory = "custom";
                    categoryTabs.forEach((t) => t.classList.toggle("active", t.dataset.filterCategory === "custom"));
                    selectPreset(result.preset.id);
                });
            });
        });
    }

    // ----- PHASE 6: Apply (commits the selected preset onto the shared Working Image) -----

    if (btnApply) {
        btnApply.addEventListener("click", () => {
            if (!currentPath || !selectedPresetId || !window.pixelforge || !window.pixelforge.session) return;
            const preset = allPresets.find((p) => p.id === selectedPresetId);
            const intensity = Number(ctrlIntensity.value);
            btnApply.disabled = true;
            btnApply.textContent = "Applying...";
            const label = preset ? `${preset.name} ${Math.round(intensity)}%` : "";
            window.pixelforge.session.applyPreset(selectedPresetId, intensity, label, (state) => {
                btnApply.disabled = false;
                btnApply.textContent = "Apply";
                if (!state || !state.ok) {
                    showError((state && state.error) || "Couldn't apply preset.");
                    return;
                }
                if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
                // Reload as the new baseline -- Intensity resets to 100%
                // since the preset's effect is now baked into the photo
                // itself, same reasoning as Enhance's Apply.
                loadImage(state.working_path);
            });
        });
    }

    // ----- Export final image -----

    if (btnExport) {
        btnExport.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge) return;
            // PHASE 4: Filter Stacking -- export the combined stack when
            // 2+ filters are stacked, otherwise the normal single-preset
            // export path (unchanged).
            if (stackedPresetIds.length >= 2) {
                window.pixelforge.chooseSaveImagePath("PixelForge_Stacked.jpg", (destPath) => {
                    if (!destPath) return;
                    const intensity = Number(ctrlIntensity.value);
                    const layers = stackedPresetIds.map((id) => ({ preset_id: id, intensity }));
                    window.pixelforge.exportPresetStack(currentPath, destPath, layers, (result) => {
                        if (!result.ok) { showError(result.error || "Export failed."); return; }
                        showError("");
                    });
                });
                return;
            }
            if (!selectedPresetId) return;
            window.pixelforge.chooseSaveImagePath("PixelForge_Filtered.jpg", (destPath) => {
                if (!destPath) return;
                const intensity = Number(ctrlIntensity.value);
                window.pixelforge.exportFilterImage(currentPath, destPath, selectedPresetId, intensity, (result) => {
                    if (!result.ok) {
                        showError(result.error || "Export failed.");
                        return;
                    }
                    showError("");
                });
            });
        });
    }

    // Presets don't depend on an image being loaded -- fetch the list as
    // soon as the bridge is ready so the grid isn't empty on first paint
    // once a photo is opened.
    if (window.onPixelforgeReady) {
        window.onPixelforgeReady(() => loadPresetList());
    }

    // ----- PHASE 4: Auto Filter -----

    if (btnAutoFilter) {
        btnAutoFilter.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge) return;
            btnAutoFilter.disabled = true;
            window.pixelforge.autoFilterSuggest(currentPath, (result) => {
                btnAutoFilter.disabled = false;
                if (!result || !result.ok) {
                    showError((result && result.error) || "Auto Filter failed.");
                    return;
                }
                if (allPresets.length) {
                    applyAutoFilterSuggestion(result);
                } else {
                    loadPresetList(() => applyAutoFilterSuggestion(result));
                }
            });
        });
    }

    function applyAutoFilterSuggestion(result) {
        stackedPresetIds = [];
        renderStackChips();
        if (ctrlIntensity) { ctrlIntensity.disabled = false; ctrlIntensity.value = result.intensity; }
        if (valIntensity) valIntensity.textContent = `${result.intensity}%`;
        selectPreset(result.preset_id);
        if (autoFilterSummary) {
            autoFilterSummary.textContent = result.summary || "";
            autoFilterSummary.classList.toggle("view--hidden", !result.summary);
        }
    }

    // ----- PHASE 4: Custom filter live preview (banate waqt) -----
    // Builds a brand-new custom filter from scratch (distinct from
    // "Save Preset" above, which scales an EXISTING preset's values).
    // Every slider drag re-renders the After image live via
    // previewCustomFilter, same debounce pattern as the main preview.

    function customFilterSpec() {
        const adjustments = {};
        cfSliderIds.forEach((id) => {
            if (id === "grain") return;
            adjustments[id] = Number(cfSliders[id].value);
        });
        return {
            adjustments,
            grain: Number(cfSliders.grain.value),
            monochrome: cfMonochrome,
        };
    }

    let cfPreviewTimer = null;
    let cfPreviewSeq = 0;
    function requestCustomFilterPreview() {
        if (!customBuilderOpen || !currentPath || !window.pixelforge) return;
        clearTimeout(cfPreviewTimer);
        if (previewSpinner) previewSpinner.classList.add("is-visible");
        cfPreviewTimer = setTimeout(() => {
            const seq = ++cfPreviewSeq;
            window.pixelforge.previewCustomFilter(currentPath, customFilterSpec(), (result) => {
                if (seq !== cfPreviewSeq) return;
                if (previewSpinner) previewSpinner.classList.remove("is-visible");
                if (!result.ok) { showError(result.error || "Preview failed."); return; }
                imgAfter.src = toFileUrl(result.path) + `?t=${Date.now()}`;
            });
        }, 180);
    }

    if (btnToggleCustomBuilder) {
        btnToggleCustomBuilder.addEventListener("click", () => {
            customBuilderOpen = !customBuilderOpen;
            if (customFilterBuilder) customFilterBuilder.classList.toggle("view--hidden", !customBuilderOpen);
            btnToggleCustomBuilder.textContent = customBuilderOpen ? "− Hide Custom Filter Builder" : "+ Create Custom Filter";
            if (customBuilderOpen) {
                // Building a new filter takes over the After preview,
                // same "one active preview source" rule as Filter Stacking.
                selectedPresetId = null;
                stackedPresetIds = [];
                renderStackChips();
                renderPresetGrid();
                requestCustomFilterPreview();
            } else {
                requestPreviewUpdate();
            }
        });
    }

    cfSliderIds.forEach((id) => {
        if (!cfSliders[id]) return;
        cfSliders[id].addEventListener("input", () => {
            const valEl = document.getElementById(`val-cf-${id}`);
            if (valEl) valEl.textContent = cfSliders[id].value;
            requestCustomFilterPreview();
        });
    });

    if (btnCfMonochrome) {
        btnCfMonochrome.addEventListener("click", () => {
            cfMonochrome = !cfMonochrome;
            btnCfMonochrome.classList.toggle("is-active", cfMonochrome);
            requestCustomFilterPreview();
        });
    }

    if (btnCfSave) {
        btnCfSave.addEventListener("click", () => {
            if (!window.pixelforge) return;
            const name = (inputCfName.value || "").trim();
            if (!name) {
                showError("Give the custom filter a name before saving.");
                return;
            }
            window.pixelforge.saveCustomPreset(name, customFilterSpec(), (result) => {
                if (!result.ok) {
                    showError(result.error || "Couldn't save filter.");
                    return;
                }
                inputCfName.value = "";
                showError("");
                loadPresetList(() => {
                    customBuilderOpen = false;
                    if (customFilterBuilder) customFilterBuilder.classList.add("view--hidden");
                    if (btnToggleCustomBuilder) btnToggleCustomBuilder.textContent = "+ Create Custom Filter";
                    activeCategory = "custom";
                    categoryTabs.forEach((t) => t.classList.toggle("active", t.dataset.filterCategory === "custom"));
                    selectPreset(result.preset.id);
                });
            });
        });
    }
});