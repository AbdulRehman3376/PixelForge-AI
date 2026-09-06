// PHASE 2 -- Image Editor
// Handles: open image (dialog + drag/drop), center preview (zoom, fit,
// reset, fullscreen), before/after comparison slider, basic manual
// controls (live CSS-filter preview), and export via the Python bridge.
//
// Scope note: brightness/contrast/saturation here are a *preview-only*
// stand-in for the real enhancement engine (core/enhancer.py, Phase 3).
// The same numbers are sent to bridge.exportImage() so what you see is
// what gets saved, but this is not the final Auto Enhance pipeline.
//
// Also hosts Phase 5 (Analyzer) and Phase 6 (Smart Pipeline) -- both
// read-only sections below that describe a photo / recommend a named
// preset+intensity pipeline, but never touch the sliders or preview by
// themselves. See each section's own header comment.

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("editor-dropzone");
    const workspace = document.getElementById("editor-workspace");
    if (!dropzone || !workspace) return; // Enhance view not present -- nothing to wire up.

    const btnOpenEmpty = document.getElementById("btn-editor-open-empty");
    const btnReplace = document.getElementById("btn-editor-replace");
    const canvasWrap = document.getElementById("editor-canvas-wrap");
    const canvas = document.getElementById("editor-canvas");
    const editorWorkspace = document.getElementById("editor-workspace");
    const btnFullscreenExit = document.getElementById("btn-fullscreen-exit");
    const imgBefore = document.getElementById("editor-image-before");
    const imgAfter = document.getElementById("editor-image-after");
    const afterWrap = document.getElementById("editor-image-after-wrap");
    const compareHandle = document.getElementById("editor-compare-handle");
    const compareSlider = document.getElementById("editor-compare-slider");
    const fileNameEl = document.getElementById("editor-filename");
    const dimensionsEl = document.getElementById("editor-dimensions");
    const zoomLevelEl = document.getElementById("editor-zoom-level");
    const btnZoomIn = document.getElementById("btn-zoom-in");
    const btnZoomOut = document.getElementById("btn-zoom-out");
    const btnZoomFit = document.getElementById("btn-zoom-fit");
    const btnZoomReset = document.getElementById("btn-zoom-reset");
    const btnFullscreen = document.getElementById("btn-fullscreen");
    const sliderIds = [
        "brightness", "contrast", "saturation",
        "exposure", "highlights", "shadows", "whites", "blacks",
        "highlight_recovery", "shadow_recovery",
        "temperature", "tint", "vibrance",
        "sharpness", "clarity", "noise_reduction", "vignette", "smoke",
    ];
    const sliders = {};
    const valueLabels = {};
    sliderIds.forEach((id) => {
        sliders[id] = document.getElementById(`ctrl-${id}`);
        valueLabels[id] = document.getElementById(`val-${id}`);
    });

    // ----- PHASE 3: Straighten & Crop -----
    const ctrlStraighten = document.getElementById("ctrl-straighten");
    const valStraighten = document.getElementById("val-straighten");
    const btnStraightenApply = document.getElementById("btn-straighten-apply");
    const btnCropStart = document.getElementById("btn-editor-crop-start");
    const btnCropApply = document.getElementById("btn-editor-crop-apply");
    const btnCropCancel = document.getElementById("btn-editor-crop-cancel");
    const cropOverlay = document.getElementById("editor-crop-overlay");
    const cropBoxEl = document.getElementById("editor-crop-box");
    const cropHandles = cropOverlay ? Array.from(cropOverlay.querySelectorAll(".crop-handle")) : [];

    const btnResetControls = document.getElementById("btn-reset-controls");
    const btnExport = document.getElementById("btn-editor-export");
    const btnSendToRemoveBg = document.getElementById("btn-editor-send-to-removebg");
    const btnAutoEnhance = document.getElementById("btn-auto-enhance");
    const autoEnhanceSummary = document.getElementById("auto-enhance-summary");
    const editorError = document.getElementById("editor-error");
    const previewSpinner = document.getElementById("editor-preview-spinner");

    // ----- PHASE 3: Auto White Balance -----
    const btnAutoWB = document.getElementById("btn-auto-wb");
    const autoWbSummary = document.getElementById("auto-wb-summary");

    // ----- PHASE 3: Skin-tone Protection / Smart Sharpen Protection (toggles) -----
    const btnToggleSkinProtect = document.getElementById("btn-toggle-skin-protect");
    const btnToggleSharpenProtect = document.getElementById("btn-toggle-sharpen-protect");
    let skinProtect = false;
    let sharpenProtect = false;

    // ----- PHASE 3: Highlight/Shadow Clipping Warning (zebra stripes) -----
    const btnToggleHighlightClip = document.getElementById("btn-toggle-highlight-clip");
    const btnToggleShadowClip = document.getElementById("btn-toggle-shadow-clip");
    const clipCanvas = document.getElementById("editor-clip-canvas");
    let showHighlightClip = false;
    let showShadowClip = false;

    // ----- PHASE 3: Selective/Local Adjustments (single radial region) -----
    const btnLocalAdd = document.getElementById("btn-local-add");
    const btnLocalRemove = document.getElementById("btn-local-remove");
    const btnLocalInvert = document.getElementById("btn-local-invert");
    const localPanel = document.getElementById("local-adjust-panel");
    const localRegionEl = document.getElementById("editor-local-region");
    const localResizeHandle = localRegionEl ? localRegionEl.querySelector(".local-adjust-resize") : null;
    const ctrlLocalExposure = document.getElementById("ctrl-local-exposure");
    const ctrlLocalSaturation = document.getElementById("ctrl-local-saturation");
    const ctrlLocalSharpness = document.getElementById("ctrl-local-sharpness");
    const ctrlLocalFeather = document.getElementById("ctrl-local-feather");
    const valLocalExposure = document.getElementById("val-local-exposure");
    const valLocalSaturation = document.getElementById("val-local-saturation");
    const valLocalSharpness = document.getElementById("val-local-sharpness");
    const valLocalFeather = document.getElementById("val-local-feather");
    // Position/size as fractions of the natural image size (0-1), the
    // same units core/enhancer.py::_ellipse_mask expects -- resolution
    // independent between the downsized preview and full export.
    let localRegion = null; // {cx, cy, rx, ry, feather, invert} | null

    // ----- Phase 5: Analyzer -----
    const btnAnalyze = document.getElementById("btn-analyze-photo");
    const analyzePanel = document.getElementById("analyze-results");
    const analyzeSummary = document.getElementById("analyze-summary");
    const analyzeBadges = document.getElementById("analyze-badges");
    const analyzeStats = document.getElementById("analyze-stats");
    const analyzeSwatches = document.getElementById("analyze-swatches");
    const analyzeScoreRing = document.getElementById("analyze-score-ring");
    const analyzeScoreValue = document.getElementById("analyze-score-value");
    const analyzeScoreGrade = document.getElementById("analyze-score-grade");
    const analyzeQualityStats = document.getElementById("analyze-quality-stats");
    const analyzeLightingStats = document.getElementById("analyze-lighting-stats");
    const analyzeSkyStats = document.getElementById("analyze-sky-stats");

    // ----- Phase 6: Smart Pipeline -----
    const btnSmartPipeline = document.getElementById("btn-smart-pipeline");
    const pipelinePanel = document.getElementById("pipeline-results");
    const pipelineName = document.getElementById("pipeline-name");
    const pipelineMatchScore = document.getElementById("pipeline-match-score");
    const pipelineSignals = document.getElementById("pipeline-signals");
    const pipelineReason = document.getElementById("pipeline-reason");
    const pipelineRecipeEl = document.getElementById("pipeline-recipe");
    const ctrlPipelineIntensity = document.getElementById("ctrl-pipeline-intensity");
    const valPipelineIntensity = document.getElementById("val-pipeline-intensity");
    const pipelineSafetyEl = document.getElementById("pipeline-safety");
    const pipelineAlternativesEl = document.getElementById("pipeline-alternatives");
    const btnPipelinePreview = document.getElementById("btn-pipeline-preview");
    const btnPipelineApply = document.getElementById("btn-pipeline-apply");
    const btnPipelineDismiss = document.getElementById("btn-pipeline-dismiss");
    const btnPipelineRerun = document.getElementById("btn-pipeline-rerun");
    const pipelineAppliedPanel = document.getElementById("pipeline-applied-panel");
    const pipelineAppliedSummary = document.getElementById("pipeline-applied-summary");
    const btnPipelineUndoApplied = document.getElementById("btn-pipeline-undo-applied");
    let currentRecommendation = null; // full v2 recommendation payload (see core/smart_pipeline.py::recommend_pipeline)
    let pipelineStepOverrides = {};   // {stepId: "apply"|"skip"} -- user toggles, sent back as overrides.steps
    let pipelineStepDebounce = null;

    let currentPath = null;
    let currentName = null;
    let naturalWidth = 0;
    let naturalHeight = 0;
    let zoom = 1;
    const ZOOM_MIN = 0.1;
    const ZOOM_MAX = 4;

    // ----- Helpers -----

    function toFileUrl(path) {
        // Windows paths ("D:\foo\bar.jpg") need forward slashes and a
        // leading slash for a valid file:// URL. encodeURI handles
        // spaces and other special characters in the path/filename
        // (keeps ":" and "/" intact, which a plain encodeURIComponent
        // would otherwise mangle).
        const normalized = path.replace(/\\/g, "/").replace(/^\/+/, "");
        return "file:///" + encodeURI(normalized);
    }

    function showError(message) {
        if (!editorError) return;
        editorError.textContent = message;
        editorError.classList.toggle("view--hidden", !message);
    }

    function currentAdjustments() {
        const out = {};
        sliderIds.forEach((id) => {
            out[id] = Number(sliders[id].value);
        });
        out.skin_protect = skinProtect ? 1 : 0;
        out.sharpen_protect = sharpenProtect ? 1 : 0;
        out.local_adjustments = localRegion
            ? [{
                cx: localRegion.cx, cy: localRegion.cy,
                rx: localRegion.rx, ry: localRegion.ry,
                feather: localRegion.feather, invert: localRegion.invert,
                adjust: {
                    exposure: Number(ctrlLocalExposure ? ctrlLocalExposure.value : 0),
                    saturation: Number(ctrlLocalSaturation ? ctrlLocalSaturation.value : 100),
                    sharpness: Number(ctrlLocalSharpness ? ctrlLocalSharpness.value : 0),
                },
            }]
            : [];
        return out;
    }

    let previewDebounceTimer = null;
    let previewRequestSeq = 0;
    let latestRenderedPath = null;

    function updateValueLabels() {
        sliderIds.forEach((id) => {
            const isPercent = id === "brightness" || id === "contrast" || id === "saturation";
            valueLabels[id].textContent = isPercent ? `${sliders[id].value}%` : sliders[id].value;
        });
    }

    // The old CSS-filter live preview only approximated brightness/
    // contrast/saturation and couldn't represent highlights, shadows,
    // temperature, clarity, noise reduction, etc. at all. Instead, every
    // slider change now asks the Python backend (core/enhancer.py) to
    // render a real downsized preview and we swap the "After" image to
    // it -- debounced so dragging a slider doesn't spam the backend.
    function requestPreviewUpdate() {
        if (!currentPath || !window.pixelforge) return;
        clearTimeout(previewDebounceTimer);
        if (previewSpinner) previewSpinner.classList.add("is-visible");

        previewDebounceTimer = setTimeout(() => {
            const seq = ++previewRequestSeq;
            window.pixelforge.previewAdjust(currentPath, currentAdjustments(), (result) => {
                if (seq !== previewRequestSeq) return; // a newer request superseded this one
                if (previewSpinner) previewSpinner.classList.remove("is-visible");
                if (!result.ok) {
                    showError(result.error || "Preview failed.");
                    return;
                }
                latestRenderedPath = result.path || null;
                imgAfter.src = toFileUrl(result.path) + `?t=${Date.now()}`;
                imgAfter.onload = () => { redrawClipWarning(); };
            });
        }, 180);
    }

    // ----- PHASE 3: Highlight/Shadow Clipping Warning -----
    // Client-side only -- no backend round trip. Reads the ALREADY
    // rendered "after" preview's own pixels (so it reflects every
    // slider currently applied, not just the raw photo) via an
    // offscreen canvas, and draws a diagonal zebra-stripe pattern over
    // any pixel that's essentially blown to white (highlight warning)
    // or crushed to black (shadow warning).
    let clipWarningRaf = null;
    function redrawClipWarning() {
        if (!clipCanvas) return;
        if (!showHighlightClip && !showShadowClip) {
            clipCanvas.classList.add("view--hidden");
            return;
        }
        if (!imgAfter || !imgAfter.complete || !imgAfter.naturalWidth) return;

        cancelAnimationFrame(clipWarningRaf);
        clipWarningRaf = requestAnimationFrame(() => {
            const w = imgAfter.naturalWidth;
            const h = imgAfter.naturalHeight;

            const sample = document.createElement("canvas");
            sample.width = w;
            sample.height = h;
            const sctx = sample.getContext("2d", { willReadFrequently: true });
            sctx.drawImage(imgAfter, 0, 0, w, h);
            let pixels;
            try {
                pixels = sctx.getImageData(0, 0, w, h).data;
            } catch (e) {
                return; // e.g. tainted canvas on some platforms -- fail silently, warning is a nice-to-have
            }

            clipCanvas.width = w;
            clipCanvas.height = h;
            clipCanvas.classList.remove("view--hidden");
            const ctx = clipCanvas.getContext("2d");
            ctx.clearRect(0, 0, w, h);
            const out = ctx.createImageData(w, h);

            const HI_THRESH = 250;  // near-255 on all channels = blown highlight
            const LO_THRESH = 5;    // near-0 on all channels = crushed shadow
            for (let i = 0; i < pixels.length; i += 4) {
                const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
                const px = (i / 4) % w;
                const py = Math.floor((i / 4) / w);
                const stripe = ((px + py) % 16) < 8; // diagonal zebra pattern

                let hit = false, color = null;
                if (showHighlightClip && r >= HI_THRESH && g >= HI_THRESH && b >= HI_THRESH) {
                    hit = true; color = stripe ? [255, 30, 30] : [0, 0, 0];
                } else if (showShadowClip && r <= LO_THRESH && g <= LO_THRESH && b <= LO_THRESH) {
                    hit = true; color = stripe ? [30, 120, 255] : [255, 255, 255];
                }

                if (hit) {
                    out.data[i] = color[0];
                    out.data[i + 1] = color[1];
                    out.data[i + 2] = color[2];
                    out.data[i + 3] = 255;
                }
            }
            ctx.putImageData(out, 0, 0);
        });
    }

    function applyZoom() {
        // Zoom is applied once, to the shared canvas "stage" -- not to
        // the before/after layers individually -- so both layers (and
        // the compare handle drawn on top of them) always scale together
        // and stay perfectly aligned, regardless of each <img>'s own
        // intrinsic resolution (the "after" preview is downsized by the
        // backend for speed; see the .editor-canvas comment in main.css).
        canvas.style.transform = `translate(-50%, -50%) scale(${zoom})`;
        zoomLevelEl.textContent = `${Math.round(zoom * 100)}%`;
    }

    // Sizes the shared stage to the image's real aspect ratio so the
    // before layer, the (possibly-downsized) after layer, and the
    // compare handle all cover the exact same rectangle before zoom is
    // applied on top.
    function sizeStageToImage() {
        if (!naturalWidth || !naturalHeight) return;
        canvas.style.width = `${naturalWidth}px`;
        canvas.style.height = `${naturalHeight}px`;
    }

    function setZoom(next) {
        zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, next));
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

    // ----- Loading an image -----

    function loadImage(path) {
        // Remembered so other tools (Remove BG, etc.) can pick up the
        // same photo without the user having to browse for it again --
        // see window.pixelforgeCurrentImage / showView() in ui.js.
        window.pixelforgeCurrentImage = path;
        // PHASE 6: lets ui.js know Enhance is now showing THIS path, so
        // switching away and back only reloads if the shared Working
        // Image has moved on since (see VIEW_CURRENT_PATH_VARS in ui.js).
        window.pixelforgeEnhanceCurrentPath = path;
        latestRenderedPath = null;
        if (!path) return;
        showError("");
        if (btnSendToRemoveBg) btnSendToRemoveBg.classList.add("view--hidden");

        if (window.pixelforge) {
            window.pixelforge.getImageInfo(path, (info) => {
                if (!info.ok) {
                    showError(info.error || "Couldn't open that image.");
                    return;
                }
                currentPath = info.path;
                currentName = info.name;
                naturalWidth = info.width;
                naturalHeight = info.height;
                sizeStageToImage();

                // BUGFIX: TIFF (and some BMP variants) are allowed by
                // the file picker but can't be decoded by Chromium's
                // <img> at all -- that's the "Unknown or unsupported
                // skia image format" console error. getImageInfo now
                // returns display_path, a cached web-safe JPEG proxy
                // for exactly those formats (identical to info.path
                // for anything already web-safe) -- use that for the
                // on-screen <img>s, while info.path (the real file)
                // stays what every actual edit/export operates on.
                const url = toFileUrl(info.display_path || info.path);
                imgBefore.onerror = () => showError("Image failed to load in the preview. Check the file isn't moved/renamed and try again.");
                imgBefore.onload = () => showError("");
                imgBefore.src = url;
                imgAfter.src = url;

                fileNameEl.textContent = info.name;
                dimensionsEl.textContent = `${info.width} × ${info.height} · ${info.format}`;

                dropzone.classList.add("view--hidden");
                workspace.classList.remove("view--hidden");

                // BUGFIX: restore this tool's last-known slider positions
                // (persisted via persistToolState() above) instead of
                // always snapping back to defaults -- see the comment by
                // sliderIds.forEach()/setToolState above. Falls back to
                // resetControls() when there's no session, no saved
                // state yet, or the state came back empty/malformed.
                const applySavedState = (result) => {
                    // session.getToolState resolves with {ok, tool, state} --
                    // unwrap to the actual saved slider values.
                    const saved = result && result.ok ? result.state : null;
                    if (saved && typeof saved === "object" && Object.keys(saved).length) {
                        sliderIds.forEach((id) => {
                            if (sliders[id] && saved[id] !== undefined) {
                                sliders[id].value = saved[id];
                            }
                        });
                        updateValueLabels();
                        skinProtect = !!saved.skin_protect;
                        sharpenProtect = !!saved.sharpen_protect;
                        if (btnToggleSkinProtect) btnToggleSkinProtect.classList.toggle("is-active", skinProtect);
                        if (btnToggleSharpenProtect) btnToggleSharpenProtect.classList.toggle("is-active", sharpenProtect);
                        if (Array.isArray(saved.local_adjustments) && saved.local_adjustments.length) {
                            const r = saved.local_adjustments[0];
                            localRegion = { cx: r.cx, cy: r.cy, rx: r.rx, ry: r.ry, feather: r.feather, invert: !!r.invert };
                            if (localPanel) localPanel.classList.remove("view--hidden");
                            if (btnLocalInvert) btnLocalInvert.classList.toggle("is-active", localRegion.invert);
                            if (r.adjust) {
                                if (ctrlLocalExposure) ctrlLocalExposure.value = r.adjust.exposure || 0;
                                if (ctrlLocalSaturation) ctrlLocalSaturation.value = r.adjust.saturation || 100;
                                if (ctrlLocalSharpness) ctrlLocalSharpness.value = r.adjust.sharpness || 0;
                                if (valLocalExposure) valLocalExposure.textContent = ctrlLocalExposure.value;
                                if (valLocalSaturation) valLocalSaturation.textContent = `${ctrlLocalSaturation.value}%`;
                                if (valLocalSharpness) valLocalSharpness.textContent = ctrlLocalSharpness.value;
                            }
                            if (ctrlLocalFeather) {
                                ctrlLocalFeather.value = r.feather || 50;
                                if (valLocalFeather) valLocalFeather.textContent = ctrlLocalFeather.value;
                            }
                            syncLocalRegionBox();
                        }
                        requestPreviewUpdate();
                    } else {
                        resetControls();
                    }
                };
                if (window.pixelforge.session && window.pixelforge.session.getToolState) {
                    window.pixelforge.session.getToolState("enhance", applySavedState);
                } else {
                    resetControls();
                }
                setComparePosition(50);
                resetStraightenPreview();
                cancelCropMode();
                if (btnCropStart) btnCropStart.disabled = false;
                if (btnRotate90) btnRotate90.disabled = false;
                if (btnFlipH) btnFlipH.disabled = false;
                if (btnFlipV) btnFlipV.disabled = false;

                // A new photo invalidates any analysis run on the
                // previous one -- hide the results panel until the user
                // re-runs Analyze Photo on this one.
                if (analyzePanel) analyzePanel.classList.add("view--hidden");
                // Same reasoning for a Smart Pipeline recommendation --
                // it was computed for the previous photo's Analyzer
                // output and no longer applies.
                if (pipelinePanel) pipelinePanel.classList.add("view--hidden");
                currentRecommendation = null;

                // Fit once the image has actually painted so we know its
                // rendered container size.
                requestAnimationFrame(fitToContainer);
            });
        }
    }

    function resetControls() {
        sliderIds.forEach((id) => {
            sliders[id].value = defaultValueFor(id);
        });
        updateValueLabels();
        skinProtect = false;
        sharpenProtect = false;
        if (btnToggleSkinProtect) btnToggleSkinProtect.classList.remove("is-active");
        if (btnToggleSharpenProtect) btnToggleSharpenProtect.classList.remove("is-active");
        showHighlightClip = false;
        showShadowClip = false;
        if (btnToggleHighlightClip) btnToggleHighlightClip.classList.remove("is-active");
        if (btnToggleShadowClip) btnToggleShadowClip.classList.remove("is-active");
        if (clipCanvas) clipCanvas.classList.add("view--hidden");
        localRegion = null;
        if (localPanel) localPanel.classList.add("view--hidden");
        if (localRegionEl) localRegionEl.classList.add("view--hidden");
        if (autoWbSummary) autoWbSummary.classList.add("view--hidden");
        requestPreviewUpdate();
        if (autoEnhanceSummary) autoEnhanceSummary.textContent = "";
    }

    // Missing-feature #4 (this session): Per-slider Reset. Shared by
    // resetControls() above (unchanged behaviour) and the new
    // per-slider "↺" buttons below, so there's exactly one place that
    // defines what "default" means for each control.
    function defaultValueFor(id) {
        return (id === "brightness" || id === "contrast" || id === "saturation") ? 100 : 0;
    }

    // ----- Open via dialog -----

    function openViaDialog() {
        if (!window.pixelforge) return;
        window.pixelforge.openImageDialog((path) => {
            if (!path) return;
            if (window.pixelforgeStartSession) {
                window.pixelforgeStartSession(path, (state) => {
                    loadImage((state && state.working_path) || path);
                });
            } else {
                loadImage(path);
            }
        });
    }

    if (btnOpenEmpty) btnOpenEmpty.addEventListener("click", openViaDialog);
    if (btnReplace) btnReplace.addEventListener("click", openViaDialog);

    // ----- Drag & drop -----
    // Note: in a QWebEngineView, a dropped file's File object exposes a
    // real filesystem `.path` (Chromium/Qt extension), which is what we
    // need to hand to the Python bridge (Pillow needs a real path, not
    // file bytes). If `.path` is ever empty (e.g. testing in a plain
    // browser), we fall back to opening the picker dialog instead.
    ["dragenter", "dragover"].forEach((evt) => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.add("is-dragover");
        });
        workspace.addEventListener(evt, (e) => {
            e.preventDefault();
            workspace.classList.add("is-dragover");
        });
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
            // PHASE 6: a drop is a NEW original image (same as Import),
            // so it must start a session too -- otherwise Ctrl+S,
            // Undo/Redo, Apply, and Smart Pipeline all silently do
            // nothing because the backend has no active session.
            if (window.pixelforgeStartSession) {
                window.pixelforgeStartSession(file.path, (state) => {
                    loadImage((state && state.working_path) || file.path);
                    // Same Projects-database hand-off as the Import/Open
                    // Image path (see pixelforgeOpenImage in ui.js) --
                    // otherwise a picture dropped directly onto the
                    // canvas while a project is open never gets recorded
                    // against it.
                    if (window.pixelforgeAttachToActiveDbProject) {
                        window.pixelforgeAttachToActiveDbProject(file.path);
                    }
                });
            } else {
                loadImage(file.path);
            }
        } else {
            showError("Couldn't read the dropped file's location -- use Open Image instead.");
        }
    }

    dropzone.addEventListener("drop", handleDrop);
    workspace.addEventListener("drop", handleDrop);

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

    // The toolbar's Fullscreen button disappears once fullscreen kicks in
    // (it's outside editor-canvas-wrap, which is the only element the
    // Fullscreen API keeps visible), so this dedicated on-canvas button
    // is the visible way out.
    // BUGFIX: QtWebEngine's Fullscreen API support doesn't fully suppress
    // painting of elements outside the fullscreened node the way a real
    // browser does (see main_window.py's fullScreenRequested comment for
    // the related Qt-side workaround). Without this, .editor-panel's text
    // (e.g. the Auto Enhance summary) kept rendering on top of the
    // fullscreened canvas, overlapping the Before/After labels. Explicitly
    // mark the rest of the editor UI as hidden ourselves whenever
    // fullscreen state changes, instead of trusting the browser to do it.
    document.addEventListener("fullscreenchange", () => {
        if (editorWorkspace) {
            editorWorkspace.classList.toggle("is-fullscreen-active", !!document.fullscreenElement);
        }
    });

    if (btnFullscreenExit) {
        btnFullscreenExit.addEventListener("click", () => {
            if (document.fullscreenElement) document.exitFullscreen();
        });
    }

    window.addEventListener("resize", () => {
        if (!workspace.classList.contains("view--hidden")) fitToContainer();
    });

    // ----- Before / after comparison slider -----

    let draggingHandle = false;

    function positionFromEvent(clientX) {
        const rect = canvasWrap.getBoundingClientRect();
        const percent = ((clientX - rect.left) / rect.width) * 100;
        setComparePosition(percent);
    }

    compareHandle.addEventListener("mousedown", () => (draggingHandle = true));
    window.addEventListener("mouseup", () => (draggingHandle = false));
    window.addEventListener("mousemove", (e) => {
        if (draggingHandle) positionFromEvent(e.clientX);
    });
    if (compareSlider) {
        compareSlider.addEventListener("input", () => setComparePosition(Number(compareSlider.value)));
    }

    // ----- Manual controls (live preview) -----

    // BUGFIX: the session already exposes setToolState/getToolState
    // (see core/session.py / ui/bridge.py) specifically so a tool's
    // in-progress control values survive switching to another view and
    // back -- but nothing here was actually calling them, so the sliders
    // silently reset every time loadImage() ran (see resetControls()
    // in loadImage below), even though the shared Working Image itself
    // was untouched. Persist on every change (debounced with the
    // preview so it doesn't spam the backend while dragging).
    let toolStateSaveTimer = null;
    function persistToolState() {
        if (!window.pixelforge || !window.pixelforge.session) return;
        clearTimeout(toolStateSaveTimer);
        toolStateSaveTimer = setTimeout(() => {
            window.pixelforge.session.setToolState("enhance", currentAdjustments(), () => {});
        }, 200);
    }

    sliderIds.forEach((id) => {
        if (sliders[id]) {
            sliders[id].addEventListener("input", () => {
                updateValueLabels();
                requestPreviewUpdate();
                persistToolState();
            });
        }
    });

    // ----- PHASE 3: Auto White Balance -----
    if (btnAutoWB) {
        btnAutoWB.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge) return;
            btnAutoWB.disabled = true;
            window.pixelforge.autoWhiteBalance(currentPath, (result) => {
                btnAutoWB.disabled = false;
                if (!result || !result.ok) {
                    showError((result && result.error) || "Auto White Balance failed.");
                    return;
                }
                if (sliders.temperature) sliders.temperature.value = result.temperature;
                if (sliders.tint) sliders.tint.value = result.tint;
                updateValueLabels();
                requestPreviewUpdate();
                persistToolState();
                if (autoWbSummary) {
                    autoWbSummary.textContent = result.summary || "";
                    autoWbSummary.classList.toggle("view--hidden", !result.summary);
                }
            });
        });
    }

    // ----- PHASE 3: Skin-tone Protection / Smart Sharpen Protection -----
    if (btnToggleSkinProtect) {
        btnToggleSkinProtect.addEventListener("click", () => {
            skinProtect = !skinProtect;
            btnToggleSkinProtect.classList.toggle("is-active", skinProtect);
            requestPreviewUpdate();
            persistToolState();
        });
    }
    if (btnToggleSharpenProtect) {
        btnToggleSharpenProtect.addEventListener("click", () => {
            sharpenProtect = !sharpenProtect;
            btnToggleSharpenProtect.classList.toggle("is-active", sharpenProtect);
            requestPreviewUpdate();
            persistToolState();
        });
    }

    // ----- PHASE 3: Highlight/Shadow Clipping Warning -----
    if (btnToggleHighlightClip) {
        btnToggleHighlightClip.addEventListener("click", () => {
            showHighlightClip = !showHighlightClip;
            btnToggleHighlightClip.classList.toggle("is-active", showHighlightClip);
            redrawClipWarning();
        });
    }
    if (btnToggleShadowClip) {
        btnToggleShadowClip.addEventListener("click", () => {
            showShadowClip = !showShadowClip;
            btnToggleShadowClip.classList.toggle("is-active", showShadowClip);
            redrawClipWarning();
        });
    }

    // ----- PHASE 3: Selective/Local Adjustments (single radial region) -----
    // The region's on-screen box is kept in sync with localRegion's
    // 0-1 fractions any time the stage is sized/zoomed (sizeStageToImage/
    // applyZoom use CSS transform on a fixed-px stage, so px math here
    // stays valid at any zoom level -- same trick .crop-box relies on).
    function syncLocalRegionBox() {
        if (!localRegionEl || !naturalWidth || !naturalHeight) return;
        if (!localRegion) {
            localRegionEl.classList.add("view--hidden");
            return;
        }
        const wPx = localRegion.rx * 2 * naturalWidth;
        const hPx = localRegion.ry * 2 * naturalHeight;
        const leftPx = localRegion.cx * naturalWidth - wPx / 2;
        const topPx = localRegion.cy * naturalHeight - hPx / 2;
        localRegionEl.style.left = `${leftPx}px`;
        localRegionEl.style.top = `${topPx}px`;
        localRegionEl.style.width = `${wPx}px`;
        localRegionEl.style.height = `${hPx}px`;
        localRegionEl.classList.remove("view--hidden");
        localRegionEl.classList.toggle("is-inverted", !!localRegion.invert);
    }

    function addLocalRegion() {
        if (!naturalWidth || !naturalHeight) return;
        localRegion = { cx: 0.5, cy: 0.5, rx: 0.2, ry: 0.2, feather: 50, invert: false };
        if (localPanel) localPanel.classList.remove("view--hidden");
        syncLocalRegionBox();
        requestPreviewUpdate();
        persistToolState();
    }

    function removeLocalRegion() {
        localRegion = null;
        if (localPanel) localPanel.classList.add("view--hidden");
        if (btnLocalInvert) btnLocalInvert.classList.remove("is-active");
        syncLocalRegionBox();
        requestPreviewUpdate();
        persistToolState();
    }

    if (btnLocalAdd) btnLocalAdd.addEventListener("click", addLocalRegion);
    if (btnLocalRemove) btnLocalRemove.addEventListener("click", removeLocalRegion);
    if (btnLocalInvert) {
        btnLocalInvert.addEventListener("click", () => {
            if (!localRegion) return;
            localRegion.invert = !localRegion.invert;
            btnLocalInvert.classList.toggle("is-active", localRegion.invert);
            syncLocalRegionBox();
            requestPreviewUpdate();
            persistToolState();
        });
    }
    [ctrlLocalExposure, ctrlLocalSaturation, ctrlLocalSharpness].forEach((ctrl) => {
        if (!ctrl) return;
        ctrl.addEventListener("input", () => {
            if (valLocalExposure) valLocalExposure.textContent = ctrlLocalExposure.value;
            if (valLocalSaturation) valLocalSaturation.textContent = `${ctrlLocalSaturation.value}%`;
            if (valLocalSharpness) valLocalSharpness.textContent = ctrlLocalSharpness.value;
            requestPreviewUpdate();
            persistToolState();
        });
    });
    if (ctrlLocalFeather) {
        ctrlLocalFeather.addEventListener("input", () => {
            if (valLocalFeather) valLocalFeather.textContent = ctrlLocalFeather.value;
            if (localRegion) localRegion.feather = Number(ctrlLocalFeather.value);
            requestPreviewUpdate();
            persistToolState();
        });
    }

    // Drag-to-move / drag-handle-to-resize, same pointer-event pattern
    // used elsewhere in this file for the crop box / compare handle.
    let localDrag = null; // {mode: "move"|"resize", startX, startY, orig}
    if (localRegionEl) {
        localRegionEl.addEventListener("pointerdown", (e) => {
            if (!localRegion) return;
            const isResize = e.target === localResizeHandle;
            localDrag = {
                mode: isResize ? "resize" : "move",
                startX: e.clientX,
                startY: e.clientY,
                orig: { ...localRegion },
            };
            e.preventDefault();
            e.stopPropagation();
            localRegionEl.setPointerCapture(e.pointerId);
        });
        localRegionEl.addEventListener("pointermove", (e) => {
            if (!localDrag || !localRegion || !naturalWidth || !naturalHeight) return;
            const dxPx = (e.clientX - localDrag.startX) / zoom;
            const dyPx = (e.clientY - localDrag.startY) / zoom;

            if (localDrag.mode === "move") {
                localRegion.cx = Math.min(1, Math.max(0, localDrag.orig.cx + dxPx / naturalWidth));
                localRegion.cy = Math.min(1, Math.max(0, localDrag.orig.cy + dyPx / naturalHeight));
            } else {
                localRegion.rx = Math.min(0.9, Math.max(0.03, localDrag.orig.rx + dxPx / naturalWidth));
                localRegion.ry = Math.min(0.9, Math.max(0.03, localDrag.orig.ry + dyPx / naturalHeight));
            }
            syncLocalRegionBox();
        });
        const endDrag = () => {
            if (!localDrag) return;
            localDrag = null;
            requestPreviewUpdate();
            persistToolState();
        };
        localRegionEl.addEventListener("pointerup", endDrag);
        localRegionEl.addEventListener("pointercancel", endDrag);
    }

    if (btnResetControls) {
        btnResetControls.addEventListener("click", () => {
            resetControls();
            persistToolState();
        });
    }

    // Missing-feature #4 (this session): Per-slider Reset -- a small
    // "↺" button next to each control's value label (see index.html),
    // resetting just that one slider instead of Reset All. One
    // delegated listener rather than 15 individual ones since the
    // buttons all live under the same panel and share a data-ctrl
    // attribute naming the slider id.
    const manualControlsPanel = document.getElementById("editor-panel");
    if (manualControlsPanel) {
        manualControlsPanel.addEventListener("click", (e) => {
            const btn = e.target.closest(".ctrl-reset-btn");
            if (!btn) return;
            const id = btn.dataset.ctrl;
            if (!sliders[id]) return;
            sliders[id].value = defaultValueFor(id);
            updateValueLabels();
            requestPreviewUpdate();
            persistToolState();
        });
    }

    // Missing-feature #3 (this session): Copy/Paste Adjustments -- copy
    // the current slider values in-memory (this browser tab only, no
    // disk/session storage needed) and paste them onto whatever photo
    // is open when Paste is clicked, e.g. after switching to a
    // different photo via Replace/Open.
    let copiedAdjustments = null;
    const btnCopyAdjustments = document.getElementById("btn-copy-adjustments");
    const btnPasteAdjustments = document.getElementById("btn-paste-adjustments");
    if (btnCopyAdjustments) {
        btnCopyAdjustments.addEventListener("click", () => {
            copiedAdjustments = currentAdjustments();
            if (btnPasteAdjustments) btnPasteAdjustments.disabled = false;
            showError("");
        });
    }
    if (btnPasteAdjustments) {
        btnPasteAdjustments.disabled = true;
        btnPasteAdjustments.addEventListener("click", () => {
            if (!copiedAdjustments) return;
            sliderIds.forEach((id) => {
                if (sliders[id] && copiedAdjustments[id] !== undefined) {
                    sliders[id].value = copiedAdjustments[id];
                }
            });
            updateValueLabels();
            skinProtect = !!copiedAdjustments.skin_protect;
            sharpenProtect = !!copiedAdjustments.sharpen_protect;
            if (btnToggleSkinProtect) btnToggleSkinProtect.classList.toggle("is-active", skinProtect);
            if (btnToggleSharpenProtect) btnToggleSharpenProtect.classList.toggle("is-active", sharpenProtect);
            // Local Adjustments are region/photo-specific (drawn at a
            // particular spot on THIS photo) -- deliberately not copied
            // across photos, same reasoning as Straighten/Crop staying
            // out of Copy/Paste.
            requestPreviewUpdate();
            persistToolState();
        });
    }

    // ----- PHASE 3: Straighten -----
    // Two independent, user-confirmed steps (straighten, then crop) --
    // NOT wired into the live previewAdjust() pipeline above. Applying
    // either one calls the Python bridge for a real pixel-accurate
    // result and reloads it as the new working image via loadImage()
    // (same hand-off pattern as "Continue in Remove BG" after an
    // export) -- see ui/bridge.py::straightenImage/cropImage and
    // core/crop.py for why they're kept separate rather than one
    // combined rotate+crop call.

    function updateStraightenLabel() {
        if (valStraighten && ctrlStraighten) valStraighten.textContent = `${ctrlStraighten.value}°`;
    }

    function resetStraightenPreview() {
        if (ctrlStraighten) ctrlStraighten.value = 0;
        updateStraightenLabel();
        // Clears the cheap CSS-only preview rotation below.
        imgBefore.style.transform = "";
        imgAfter.style.transform = "";
        if (btnStraightenApply) btnStraightenApply.disabled = true;
    }

    if (ctrlStraighten) {
        ctrlStraighten.addEventListener("input", () => {
            updateStraightenLabel();
            // Cheap CSS-only live approximation while dragging -- same
            // "preview-only stand-in" idea as the old brightness/
            // contrast CSS filter (see the Phase 2 scope note at the
            // top of this file); the real pixel-accurate rotation only
            // happens once "Apply Straighten" below is clicked.
            const angle = Number(ctrlStraighten.value);
            const cssRotate = `rotate(${angle}deg)`;
            imgBefore.style.transform = cssRotate;
            imgAfter.style.transform = cssRotate;
            if (btnStraightenApply) btnStraightenApply.disabled = angle === 0;
        });
    }

    if (btnStraightenApply) {
        btnStraightenApply.addEventListener("click", () => {
            const angle = Number(ctrlStraighten.value);
            if (!currentPath || !window.pixelforge || !angle) return;

            btnStraightenApply.disabled = true;
            btnStraightenApply.textContent = "Straightening...";

            window.pixelforge.straightenImage(currentPath, angle, (result) => {
                btnStraightenApply.textContent = "Apply Straighten";
                if (!result.ok) {
                    showError(result.error || "Straighten failed.");
                    btnStraightenApply.disabled = false;
                    return;
                }
                showError("");
                // PHASE 6: record this as a real Working Image step (so
                // global Undo/Redo and other tools see it), THEN reload
                // it as the new base image -- resets sliders, crop, and
                // the straighten preview too, since they all apply to
                // the *previous* base image.
                const afterCommit = (path) => loadImage(path);
                if (window.pixelforge.session) {
                    window.pixelforge.session.commitFile(result.path, "crop", "Straighten", (state) => {
                        if (state && state.ok) {
                            if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
                            afterCommit(state.working_path);
                            return;
                        }
                        afterCommit(result.path);
                    });
                } else {
                    afterCommit(result.path);
                }
            });
        });
    }

    // Missing-feature #1 (this session): Rotate 90 / Flip H / Flip V.
    // Same commit-then-reload pattern as "Apply Straighten" above, but
    // each button commits immediately on click -- there's no in-between
    // preview state to adjust for an exact 90/flip the way there is for
    // the fine-angle Straighten slider.
    const btnRotate90 = document.getElementById("btn-rotate-90");
    const btnFlipH = document.getElementById("btn-flip-h");
    const btnFlipV = document.getElementById("btn-flip-v");

    function runTranspose(op, btn, label) {
        if (!currentPath || !window.pixelforge || !window.pixelforge.transposeImage) return;
        const allBtns = [btnRotate90, btnFlipH, btnFlipV].filter(Boolean);
        allBtns.forEach((b) => { b.disabled = true; });
        const originalText = btn.textContent;
        btn.textContent = "...";

        window.pixelforge.transposeImage(currentPath, op, (result) => {
            btn.textContent = originalText;
            allBtns.forEach((b) => { b.disabled = false; });
            if (!result.ok) {
                showError(result.error || `${label} failed.`);
                return;
            }
            showError("");
            const afterCommit = (path) => loadImage(path);
            if (window.pixelforge.session) {
                window.pixelforge.session.commitFile(result.path, "crop", label, (state) => {
                    if (state && state.ok) {
                        if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
                        afterCommit(state.working_path);
                        return;
                    }
                    afterCommit(result.path);
                });
            } else {
                afterCommit(result.path);
            }
        });
    }

    if (btnRotate90) btnRotate90.addEventListener("click", () => runTranspose("rotate90", btnRotate90, "Rotate"));
    if (btnFlipH) btnFlipH.addEventListener("click", () => runTranspose("flip_h", btnFlipH, "Flip"));
    if (btnFlipV) btnFlipV.addEventListener("click", () => runTranspose("flip_v", btnFlipV, "Flip"));

    // ----- PHASE 3: Manual Crop -----
    // Same crop-box drag/resize pattern as Remove BG's Manual Crop
    // (frontend/js/removebg.js) -- draggable box with 4 corner handles,
    // converted to the loaded image's own pixel coordinates on Apply.

    function positionCropBoxFromNaturalBox(box) {
        if (!cropBoxEl || !naturalWidth || !naturalHeight) return;
        const leftPct = (box[0] / naturalWidth) * 100;
        const topPct = (box[1] / naturalHeight) * 100;
        const widthPct = ((box[2] - box[0]) / naturalWidth) * 100;
        const heightPct = ((box[3] - box[1]) / naturalHeight) * 100;
        cropBoxEl.style.left = `${leftPct}%`;
        cropBoxEl.style.top = `${topPct}%`;
        cropBoxEl.style.width = `${widthPct}%`;
        cropBoxEl.style.height = `${heightPct}%`;
    }

    function naturalBoxFromCropBoxEl() {
        if (!cropBoxEl || !canvas) return null;
        const stageRect = canvas.getBoundingClientRect();
        const boxRect = cropBoxEl.getBoundingClientRect();
        if (stageRect.width === 0 || stageRect.height === 0) return null;
        const scaleX = naturalWidth / stageRect.width;
        const scaleY = naturalHeight / stageRect.height;
        const left = (boxRect.left - stageRect.left) * scaleX;
        const top = (boxRect.top - stageRect.top) * scaleY;
        const right = (boxRect.right - stageRect.left) * scaleX;
        const bottom = (boxRect.bottom - stageRect.top) * scaleY;
        return [
            Math.max(0, Math.round(left)),
            Math.max(0, Math.round(top)),
            Math.min(naturalWidth, Math.round(right)),
            Math.min(naturalHeight, Math.round(bottom)),
        ];
    }

    // Missing-feature #2 (this session): crop aspect-ratio presets.
    // Percent-space (left/top/width/height, all 0-100) is proportional
    // to real image pixels here because the stage is always sized to
    // the loaded image's exact aspect ratio (see the "Sizes the shared
    // stage..." comment near fitToContainer above) -- so converting a
    // target real-world ratio (e.g. 16:9) into a percent-space
    // width/height pair just needs naturalWidth/naturalHeight as the
    // percent<->pixel conversion factor.
    const cropRatioRow = document.getElementById("editor-crop-ratio-row");
    const cropRatioBtns = cropRatioRow ? Array.from(cropRatioRow.querySelectorAll(".crop-ratio-btn")) : [];
    const CROP_RATIO_MAP = { "1:1": 1, "4:5": 4 / 5, "16:9": 16 / 9, "9:16": 9 / 16 };
    let currentCropRatio = null; // real-world w/h, or null = "Free" (no lock)

    function heightPctForWidthPct(widthPct, ratio) {
        if (!ratio || !naturalWidth || !naturalHeight) return null;
        return widthPct * (naturalWidth / naturalHeight) / ratio;
    }

    function applyCurrentRatioToBox() {
        if (!currentCropRatio || !cropBoxEl || !cropOverlay) return;
        const overlayRect = cropOverlay.getBoundingClientRect();
        const boxRect = cropBoxEl.getBoundingClientRect();
        if (!overlayRect.width || !overlayRect.height) return;
        const left = ((boxRect.left - overlayRect.left) / overlayRect.width) * 100;
        const top = ((boxRect.top - overlayRect.top) / overlayRect.height) * 100;
        const width = (boxRect.width / overlayRect.width) * 100;
        const height = (boxRect.height / overlayRect.height) * 100;
        const centerY = top + height / 2;
        let newHeight = heightPctForWidthPct(width, currentCropRatio);
        if (newHeight === null) return;
        let newTop = centerY - newHeight / 2;
        if (newTop < 0) newTop = 0;
        if (newTop + newHeight > 100) newHeight = 100 - newTop;
        cropBoxEl.style.left = `${left}%`;
        cropBoxEl.style.top = `${newTop}%`;
        cropBoxEl.style.width = `${width}%`;
        cropBoxEl.style.height = `${newHeight}%`;
    }

    cropRatioBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            cropRatioBtns.forEach((b) => b.classList.remove("is-active"));
            btn.classList.add("is-active");
            currentCropRatio = CROP_RATIO_MAP[btn.dataset.ratio] || null; // "free" -> null
            applyCurrentRatioToBox();
        });
    });

    function resetCropRatioSelection() {
        currentCropRatio = null;
        cropRatioBtns.forEach((b) => b.classList.toggle("is-active", b.dataset.ratio === "free"));
    }

    function enterCropMode() {
        if (!cropOverlay || !currentPath || !naturalWidth || !naturalHeight) return;
        cropOverlay.classList.remove("view--hidden");
        if (btnCropStart) btnCropStart.classList.add("view--hidden");
        if (btnCropApply) btnCropApply.classList.remove("view--hidden");
        if (btnCropCancel) btnCropCancel.classList.remove("view--hidden");
        if (cropRatioRow) cropRatioRow.classList.remove("view--hidden");
        resetCropRatioSelection();
        const startBox = [
            Math.round(naturalWidth * 0.1),
            Math.round(naturalHeight * 0.1),
            Math.round(naturalWidth * 0.9),
            Math.round(naturalHeight * 0.9),
        ];
        positionCropBoxFromNaturalBox(startBox);
    }

    function cancelCropMode() {
        if (cropOverlay) cropOverlay.classList.add("view--hidden");
        if (btnCropStart) btnCropStart.classList.remove("view--hidden");
        if (btnCropApply) btnCropApply.classList.add("view--hidden");
        if (btnCropCancel) btnCropCancel.classList.add("view--hidden");
        if (cropRatioRow) cropRatioRow.classList.add("view--hidden");
    }

    if (btnCropStart) btnCropStart.addEventListener("click", enterCropMode);
    if (btnCropCancel) btnCropCancel.addEventListener("click", cancelCropMode);

    if (btnCropApply) {
        btnCropApply.addEventListener("click", () => {
            const box = naturalBoxFromCropBoxEl();
            if (!box || box[2] <= box[0] || box[3] <= box[1] || !currentPath || !window.pixelforge) {
                cancelCropMode();
                return;
            }

            btnCropApply.disabled = true;
            btnCropApply.textContent = "Cropping...";

            window.pixelforge.cropImage(currentPath, box, (result) => {
                btnCropApply.disabled = false;
                btnCropApply.textContent = "Apply";
                if (!result.ok) {
                    showError(result.error || "Crop failed.");
                    return;
                }
                showError("");
                cancelCropMode();
                // PHASE 6: same commit-then-reload pattern as Apply
                // Straighten above.
                const afterCommit = (path) => loadImage(path);
                if (window.pixelforge.session) {
                    window.pixelforge.session.commitFile(result.path, "crop", "Crop", (state) => {
                        if (state && state.ok) {
                            if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
                            afterCommit(state.working_path);
                            return;
                        }
                        afterCommit(result.path);
                    });
                } else {
                    afterCommit(result.path);
                }
            });
        });
    }

    // ----- Dragging the crop box itself (move) and its 4 corner handles (resize) -----
    if (cropBoxEl && cropOverlay) {
        let dragMode = null; // null | "move" | "nw" | "ne" | "sw" | "se"
        let dragStart = { x: 0, y: 0 };
        let boxStart = { left: 0, top: 0, width: 0, height: 0, overlayRect: null };

        function overlayRectPercent() {
            const overlayRect = cropOverlay.getBoundingClientRect();
            const boxRect = cropBoxEl.getBoundingClientRect();
            return {
                overlayRect,
                left: ((boxRect.left - overlayRect.left) / overlayRect.width) * 100,
                top: ((boxRect.top - overlayRect.top) / overlayRect.height) * 100,
                width: (boxRect.width / overlayRect.width) * 100,
                height: (boxRect.height / overlayRect.height) * 100,
            };
        }

        cropBoxEl.addEventListener("mousedown", (e) => {
            if (e.target.classList.contains("crop-handle")) return; // handled below
            dragMode = "move";
            dragStart = { x: e.clientX, y: e.clientY };
            boxStart = overlayRectPercent();
            e.preventDefault();
        });

        cropHandles.forEach((handle) => {
            handle.addEventListener("mousedown", (e) => {
                dragMode = handle.dataset.handle;
                dragStart = { x: e.clientX, y: e.clientY };
                boxStart = overlayRectPercent();
                e.stopPropagation();
                e.preventDefault();
            });
        });

        window.addEventListener("mousemove", (e) => {
            if (!dragMode) return;
            const overlayRect = boxStart.overlayRect;
            if (!overlayRect || overlayRect.width === 0 || overlayRect.height === 0) return;
            const dxPct = ((e.clientX - dragStart.x) / overlayRect.width) * 100;
            const dyPct = ((e.clientY - dragStart.y) / overlayRect.height) * 100;

            let { left, top, width, height } = boxStart;

            if (dragMode === "move") {
                left = Math.max(0, Math.min(100 - width, left + dxPct));
                top = Math.max(0, Math.min(100 - height, top + dyPct));
            } else {
                // Corner resize -- clamp so the box never inverts (min 4%
                // on each axis) and never drags outside the stage.
                if (dragMode.includes("w")) {
                    const right = left + width;
                    left = Math.max(0, Math.min(right - 4, left + dxPct));
                    width = right - left;
                }
                if (dragMode.includes("e")) {
                    width = Math.max(4, Math.min(100 - left, width + dxPct));
                }
                if (dragMode.includes("n")) {
                    const bottom = top + height;
                    top = Math.max(0, Math.min(bottom - 4, top + dyPct));
                    height = bottom - top;
                }
                if (dragMode.includes("s")) {
                    height = Math.max(4, Math.min(100 - top, height + dyPct));
                }

                // Missing-feature #2 (this session): if a ratio preset
                // is selected, override the free-form height above so
                // the box always resizes at that ratio -- driven off
                // width, anchored to whichever corner ISN'T being
                // dragged so that corner doesn't visibly jump.
                if (currentCropRatio) {
                    const fixedLeft = dragMode.includes("w") ? (boxStart.left + boxStart.width) : boxStart.left;
                    const fixedTop = dragMode.includes("n") ? (boxStart.top + boxStart.height) : boxStart.top;

                    if (dragMode.includes("w")) {
                        left = Math.max(0, Math.min(fixedLeft - 4, left));
                        width = fixedLeft - left;
                    } else {
                        width = Math.min(width, 100 - fixedLeft);
                        left = fixedLeft;
                    }

                    let lockedHeight = heightPctForWidthPct(width, currentCropRatio);
                    if (lockedHeight !== null) {
                        if (dragMode.includes("n")) {
                            top = Math.max(0, fixedTop - lockedHeight);
                            height = fixedTop - top;
                        } else {
                            height = Math.min(lockedHeight, 100 - fixedTop);
                            top = fixedTop;
                        }
                    }
                }
            }

            cropBoxEl.style.left = `${left}%`;
            cropBoxEl.style.top = `${top}%`;
            cropBoxEl.style.width = `${width}%`;
            cropBoxEl.style.height = `${height}%`;
        });

        window.addEventListener("mouseup", () => { dragMode = null; });
    }

    // ----- Auto Enhance -----
    // Analyzes the loaded photo (brightness/contrast/color spread) and
    // sets the manual-control sliders to a suggested "professional"
    // look automatically. Values stay within safe ranges server-side
    // (see bridge.py::autoEnhanceAnalyze) so it never overcooks the
    // image -- the user can still nudge the sliders afterward, or hit
    // Reset to go back to the original.
    if (btnAutoEnhance) {
        btnAutoEnhance.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge) {
                showError("Open an image first.");
                return;
            }

            btnAutoEnhance.classList.add("is-loading");
            btnAutoEnhance.textContent = "Analyzing...";

            window.pixelforge.autoEnhanceAnalyze(currentPath, (result) => {
                btnAutoEnhance.classList.remove("is-loading");
                btnAutoEnhance.innerHTML =
                    '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8"/></svg> Auto Enhance';

                if (!result.ok) {
                    showError(result.error || "Auto Enhance failed.");
                    return;
                }

                showError("");
                resetControls(); // start Auto Enhance from a clean slate

                // Apply every control the backend suggested a value for
                // (brightness/contrast/saturation plus light, color, and
                // detail controls) instead of only the first three --
                // whichever of the 14 sliders autoEnhanceAnalyze() didn't
                // return a value for is simply left at its reset default.
                sliderIds.forEach((id) => {
                    if (result[id] !== undefined && sliders[id]) {
                        sliders[id].value = result[id];
                    }
                });
                updateValueLabels();
                requestPreviewUpdate();

                // Nudge the before/after handle so the improvement is
                // obvious at a glance right after running it.
                setComparePosition(50);

                if (autoEnhanceSummary) {
                    autoEnhanceSummary.textContent = result.summary || "";
                    // BUGFIX: after this text mutation grows the paragraph
                    // from empty/short to several lines, QtWebEngine doesn't
                    // always re-run layout on the sibling <h3> ("Before /
                    // After") before its next paint -- it kept getting
                    // drawn at its old (pre-growth) position, overlapping
                    // the last line of the summary. Reading offsetHeight
                    // forces a synchronous layout right now, so the browser
                    // has correct, up-to-date box positions before it paints.
                    void autoEnhanceSummary.offsetHeight;
                }
            });
        });
    }

    // ----- Phase 5: Analyzer -----
    // Runs core/analyzer.py (via ui/bridge.py::analyzeImageAsync) on the
    // ORIGINAL opened photo (currentPath, not the live-adjusted preview)
    // -- Analyzer describes what the photo IS (subject, lighting, faces,
    // color, quality), which shouldn't shift just because the user is
    // mid-way through dragging a slider. Purely read-only: unlike Auto
    // Enhance, nothing here touches the sliders or the preview image.
    const LIGHTING_LABELS = { bright: "Bright", normal: "Normal light", dark: "Low light", backlit: "Backlit" };
    const SUBJECT_LABELS = { portrait: "Portrait", landscape_scene: "Landscape / scene", general: "General photo" };
    const INDOOR_OUTDOOR_LABELS = { indoor: "Indoor", outdoor: "Outdoor", uncertain: "Indoor/outdoor unclear" };
    const QUALITY_LABELS = { sharp: "Sharp", soft: "Slightly soft", blurry: "Blurry" };
    const GRADE_LABELS = { excellent: "Excellent", good: "Good", fair: "Fair", poor: "Poor" };
    const LIGHT_QUALITY_LABELS = { harsh: "Harsh light", soft: "Soft, diffused light", uneven: "Uneven light", even: "Even light" };

    // Small helper shared by every stats block below -- builds one
    // label/value row in the same .analyze-stat-row shape the panel
    // already used pre-Phase-5-upgrade, optionally flagged as a
    // warning row (amber, see .analyze-stat-row--warn in main.css).
    function _analyzeRow(container, label, value, warn) {
        const row = document.createElement("div");
        row.className = warn ? "analyze-stat-row analyze-stat-row--warn" : "analyze-stat-row";
        row.innerHTML = `<span>${label}</span><span>${value}</span>`;
        container.appendChild(row);
    }

    function renderAnalysis(result) {
        if (!analyzePanel) return;

        const quality = result.quality || {};
        const lightQuality = result.light_quality || {};
        const sky = result.sky || {};

        // ----- Header: overall score ring + grade + summary line -----
        if (analyzeScoreRing && analyzeScoreValue) {
            analyzeScoreRing.dataset.grade = quality.grade || "";
            analyzeScoreValue.textContent = quality.overall_score != null ? quality.overall_score : "--";
        }
        if (analyzeScoreGrade) {
            analyzeScoreGrade.textContent = quality.grade
                ? `${GRADE_LABELS[quality.grade] || quality.grade} quality`
                : "";
        }
        if (analyzeSummary) analyzeSummary.textContent = result.summary || "";

        // ----- Badges: quick-scan tags -----
        if (analyzeBadges) {
            const badges = [
                { text: SUBJECT_LABELS[result.subject_type] || result.subject_type },
                { text: LIGHTING_LABELS[result.lighting] || result.lighting },
            ];
            if (INDOOR_OUTDOOR_LABELS[result.indoor_outdoor] && result.indoor_outdoor !== "uncertain") {
                badges.push({ text: INDOOR_OUTDOOR_LABELS[result.indoor_outdoor] });
            }
            badges.push({
                text: result.face_count > 0
                    ? `${result.face_count} face${result.face_count > 1 ? "s" : ""} detected`
                    : "No faces detected",
            });
            badges.push({ text: QUALITY_LABELS[quality.rating] || quality.rating });

            analyzeBadges.innerHTML = "";
            badges.forEach((b) => {
                const span = document.createElement("span");
                span.className = "badge badge--muted analyze-badge";
                span.textContent = b.text;
                analyzeBadges.appendChild(span);
            });
        }

        // ----- Quality section: sharpness/noise/exposure/dynamic range -----
        // Falls back to "--" instead of "undefined" for analysis dicts
        // saved before this Phase 5 upgrade (old projects can still
        // reopen with a cached pre-upgrade analysis -- see core/session.py).
        const _q = (v) => (v == null ? "--" : v);
        if (analyzeQualityStats) {
            analyzeQualityStats.innerHTML = "";
            _analyzeRow(analyzeQualityStats, "Sharpness", `${_q(quality.sharpness_score)}/100 (${QUALITY_LABELS[quality.rating] || quality.rating || "--"})`);
            _analyzeRow(analyzeQualityStats, "Noise", `${_q(quality.noise_score)}/100`);
            _analyzeRow(analyzeQualityStats, "Exposure", `${_q(quality.exposure_score)}/100`);
            _analyzeRow(analyzeQualityStats, "Dynamic range", `${_q(quality.dynamic_range)}/100`);
        }

        // ----- Lighting section: exposure read + light character -----
        if (analyzeLightingStats) {
            analyzeLightingStats.innerHTML = "";
            _analyzeRow(analyzeLightingStats, "Lighting", LIGHTING_LABELS[result.lighting] || result.lighting);
            _analyzeRow(analyzeLightingStats, "Light quality", LIGHT_QUALITY_LABELS[lightQuality.label] || lightQuality.label || "--");
            _analyzeRow(analyzeLightingStats, "Brightness", `${result.brightness}/100`);
            _analyzeRow(analyzeLightingStats, "Contrast", `${result.contrast}/100`);
        }

        // ----- Sky section: dedicated read, for highlight-risk awareness -----
        if (analyzeSkyStats) {
            analyzeSkyStats.innerHTML = "";
            if (sky.present) {
                _analyzeRow(analyzeSkyStats, "Sky detected", `${sky.coverage_percent}% of frame`);
                _analyzeRow(analyzeSkyStats, "Confidence", `${sky.confidence}/100`);
                _analyzeRow(
                    analyzeSkyStats,
                    "Highlight risk",
                    sky.highlight_risk ? `Blown -- ${sky.clipped_percent}% clipped` : "Looks fine",
                    !!sky.highlight_risk
                );
            } else {
                _analyzeRow(analyzeSkyStats, "Sky detected", "No sky in frame");
            }
        }

        // ----- Scene & dimensions -----
        if (analyzeStats) {
            analyzeStats.innerHTML = "";
            const rows = [
                ["Dimensions", `${result.width} × ${result.height} (${result.megapixels} MP)`],
                ["Orientation", result.orientation.charAt(0).toUpperCase() + result.orientation.slice(1)],
                ["Scene", INDOOR_OUTDOOR_LABELS[result.indoor_outdoor] || result.indoor_outdoor],
            ];
            rows.forEach(([label, value]) => _analyzeRow(analyzeStats, label, value));
        }

        // ----- Dominant colors -----
        if (analyzeSwatches) {
            analyzeSwatches.innerHTML = "";
            (result.dominant_colors || []).forEach((c) => {
                const chip = document.createElement("div");
                chip.className = "analyze-swatch";
                chip.style.background = c.hex;
                chip.title = `${c.hex} (${c.percent}%)`;
                analyzeSwatches.appendChild(chip);
            });
        }

        analyzePanel.classList.remove("view--hidden");
    }

    if (btnAnalyze) {
        btnAnalyze.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge) {
                showError("Open an image first.");
                return;
            }

            btnAnalyze.classList.add("is-loading");
            btnAnalyze.textContent = "Analyzing...";

            window.pixelforge.analyzeImage(currentPath, (result) => {
                btnAnalyze.classList.remove("is-loading");
                btnAnalyze.textContent = "Analyze Photo";

                if (!result.ok) {
                    showError(result.error || "Analysis failed.");
                    return;
                }

                showError("");
                renderAnalysis(result);
            });
        });
    }

    // ----- Phase 6: Smart Pipeline -----
    // Runs core/analyzer.py + core/smart_pipeline.py (via
    // ui/bridge.py::smartPipelineAsync) on the ORIGINAL opened photo --
    // same reasoning as Analyzer above: a recommendation describes what
    // the photo IS, so it shouldn't be computed off a half-adjusted
    // preview. Purely advisory: this never touches the sliders/preview
    // by itself. The only way anything actually changes is the user
    // clicking "Apply in Filters" below, which hands off to the Filters
    // view (core/filters.py::apply_preset) rather than duplicating that
    // processing here -- Smart Pipeline's job is the recommendation,
    // not a second rendering path.
    // ----- Phase 6 v2: Smart Pipeline -----
    // Runs core/analyzer.py + core/smart_pipeline.py (via
    // ui/bridge.py::smartPipelineAsync) on the ORIGINAL opened photo --
    // a recommendation describes what the photo IS, so it's computed
    // off the unmodified working image, not a half-adjusted preview.
    // Everything below this point (toggling a step, dragging Intensity,
    // Preview, Apply) talks to the SAME session-backed engine
    // (smartPipelineCustomizeAsync / PreviewAsync / ApplyAsync) rather
    // than duplicating any processing here -- this file only renders
    // what the backend recipe/safety/alternatives data already contains.

    function currentOverrides() {
        return {
            intensity: ctrlPipelineIntensity ? Number(ctrlPipelineIntensity.value) : undefined,
            steps: pipelineStepOverrides,
        };
    }

    function renderPipelineRecipe(recipe) {
        if (!pipelineRecipeEl) return;
        pipelineRecipeEl.innerHTML = "";
        (recipe.steps || []).forEach((step) => {
            const row = document.createElement("div");
            row.className = "pipeline-step" + (step.status === "skip" ? " is-skipped" : "");

            // "suggested" (currently only the Background step) is
            // advisory, not something this recipe applies itself -- no
            // on/off toggle, just a button that jumps to the real tool
            // with the shared Working Image already loaded there (see
            // ui.js's session-aware view switching).
            if (step.status === "suggested") {
                const goBtn = document.createElement("button");
                goBtn.type = "button";
                goBtn.className = "pipeline-step-toggle pipeline-step-go";
                goBtn.title = "Open Remove BG";
                goBtn.textContent = "→";
                goBtn.addEventListener("click", () => {
                    if (window.pixelforgeShowView) window.pixelforgeShowView("removebg");
                });
                row.appendChild(goBtn);
            } else {
                const toggle = document.createElement("input");
                toggle.type = "checkbox";
                toggle.className = "pipeline-step-toggle";
                toggle.checked = step.status === "apply";
                toggle.title = step.status === "apply" ? "Turn this step off" : "Turn this step on";
                toggle.addEventListener("change", () => {
                    pipelineStepOverrides[step.id] = toggle.checked ? "apply" : "skip";
                    runPipelineCustomize();
                });
                row.appendChild(toggle);
            }

            const body = document.createElement("div");
            body.className = "pipeline-step-body";

            const top = document.createElement("div");
            top.className = "pipeline-step-top";
            const label = document.createElement("span");
            label.className = "pipeline-step-label";
            label.textContent = step.label;
            const statusBadge = document.createElement("span");
            statusBadge.className = "pipeline-step-status " + step.status;
            statusBadge.textContent =
                step.status === "apply" ? "Apply" : step.status === "suggested" ? "Suggested" : "Skip";
            top.appendChild(label);
            top.appendChild(statusBadge);
            body.appendChild(top);

            const action = document.createElement("div");
            action.className = "pipeline-step-action";
            action.textContent = step.action || "";
            body.appendChild(action);

            const reason = document.createElement("div");
            reason.className = "pipeline-step-reason";
            reason.textContent = step.reason || "";
            body.appendChild(reason);

            row.appendChild(body);
            pipelineRecipeEl.appendChild(row);
        });

    }

    function renderPipelineSafety(safety) {
        if (!pipelineSafetyEl) return;
        pipelineSafetyEl.innerHTML = "";
        if (!safety || safety.safe) {
            const ok = document.createElement("div");
            ok.className = "pipeline-safety-ok";
            ok.textContent = "✓ " + ((safety && safety.label) || "Pipeline safe to apply");
            pipelineSafetyEl.appendChild(ok);
            return;
        }
        (safety.warnings || []).forEach((w) => {
            const box = document.createElement("div");
            box.className = "pipeline-safety-warning";
            const title = document.createElement("div");
            title.className = "pipeline-safety-warning-title";
            title.textContent = "⚠️ " + w.title;
            const detail = document.createElement("div");
            detail.className = "pipeline-safety-warning-detail";
            detail.textContent = w.detail;
            const fix = document.createElement("div");
            fix.className = "pipeline-safety-warning-fix";
            fix.textContent = "Fix: " + w.fix;
            box.appendChild(title);
            box.appendChild(detail);
            box.appendChild(fix);
            pipelineSafetyEl.appendChild(box);
        });
    }

    function renderPipelineAlternatives(alternatives) {
        if (!pipelineAlternativesEl) return;
        pipelineAlternativesEl.innerHTML = "";
        (alternatives || []).forEach((alt) => {
            const row = document.createElement("button");
            row.type = "button";
            row.className = "pipeline-alt";
            const name = document.createElement("span");
            name.className = "pipeline-alt-name";
            name.textContent = alt.name;
            const score = document.createElement("span");
            score.className = "pipeline-alt-score";
            score.textContent = `${alt.match_score}% Match`;
            row.appendChild(name);
            row.appendChild(score);
            row.title = (alt.reasons || []).concat(alt.why_not || []).join(" · ");
            row.addEventListener("click", () => {
                pipelineStepOverrides = {}; // switching looks starts from that look's own recipe
                runPipelineCustomize(alt.id);
            });
            pipelineAlternativesEl.appendChild(row);
        });
    }

    function renderRecommendation(recommendation) {
        if (!pipelinePanel) return;
        currentRecommendation = recommendation;

        if (pipelineName) pipelineName.textContent = recommendation.name || "";
        if (pipelineMatchScore) {
            const score = recommendation.match_score != null ? recommendation.match_score : recommendation.confidence;
            pipelineMatchScore.textContent = `${score}% Match`;
            pipelineMatchScore.className = "badge " + (score >= 80 ? "badge--ok" : score >= 55 ? "badge--muted" : "badge--error");
        }
        if (pipelineSignals) {
            pipelineSignals.innerHTML = "";
            (recommendation.signals || []).forEach((text) => {
                const span = document.createElement("span");
                span.className = "badge badge--muted analyze-badge";
                span.textContent = text;
                pipelineSignals.appendChild(span);
            });
        }
        if (pipelineReason) pipelineReason.textContent = recommendation.explanation || recommendation.reason || "";

        const recipe = recommendation.recipe || {};
        renderPipelineRecipe(recipe);
        if (ctrlPipelineIntensity) ctrlPipelineIntensity.value = recipe.intensity != null ? recipe.intensity : 70;
        if (valPipelineIntensity) valPipelineIntensity.textContent = `${ctrlPipelineIntensity ? ctrlPipelineIntensity.value : 70}%`;

        renderPipelineSafety(recommendation.safety);
        renderPipelineAlternatives(recommendation.alternatives);

        pipelinePanel.classList.remove("view--hidden");
        if (pipelineAppliedPanel) pipelineAppliedPanel.classList.add("view--hidden");
    }

    // Debounced re-customize -- fires when a step toggle or the
    // Intensity slider changes, so dragging the slider doesn't spam the
    // backend with a call per pixel of drag.
    function runPipelineCustomize(lookId) {
        if (!window.pixelforge || !currentRecommendation) return;
        const targetLook = lookId || currentRecommendation.id;
        if (pipelineStepDebounce) clearTimeout(pipelineStepDebounce);
        pipelineStepDebounce = setTimeout(() => {
            window.pixelforge.smartPipelineCustomize(targetLook, currentOverrides(), (result) => {
                if (!result || !result.ok) {
                    showError((result && result.error) || "Couldn't update the pipeline.");
                    return;
                }
                renderRecommendation(result.recommendation);
            });
        }, 150);
    }

    if (ctrlPipelineIntensity) {
        ctrlPipelineIntensity.addEventListener("input", () => {
            if (valPipelineIntensity) valPipelineIntensity.textContent = `${ctrlPipelineIntensity.value}%`;
            runPipelineCustomize();
        });
    }

    if (btnSmartPipeline) {
        btnSmartPipeline.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge) {
                showError("Open an image first.");
                return;
            }

            pipelineStepOverrides = {};
            btnSmartPipeline.classList.add("is-loading");
            btnSmartPipeline.textContent = "Thinking...";

            window.pixelforge.smartPipeline(currentPath, (result) => {
                btnSmartPipeline.classList.remove("is-loading");
                btnSmartPipeline.innerHTML =
                    '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M13 2 3 14h7l-1 8 10-12h-7l1-8Z"/></svg> Smart Pipeline';

                if (!result.ok) {
                    showError(result.error || "Smart Pipeline failed.");
                    return;
                }

                showError("");
                renderRecommendation(result.recommendation);
                // A Smart Pipeline run also computes a full Analyzer
                // pass -- show that too, same as clicking "Analyze
                // Photo" separately, so the user sees the evidence
                // behind the recommendation without an extra click.
                if (result.analysis) renderAnalysis(result.analysis);
            });
        });
    }

    if (btnPipelineRerun) {
        btnPipelineRerun.addEventListener("click", () => {
            if (btnSmartPipeline) btnSmartPipeline.click();
        });
    }

    if (btnPipelineDismiss) {
        btnPipelineDismiss.addEventListener("click", () => {
            if (pipelinePanel) pipelinePanel.classList.add("view--hidden");
        });
    }

    if (btnPipelinePreview) {
        btnPipelinePreview.addEventListener("click", () => {
            if (!currentRecommendation || !window.pixelforge) return;
            btnPipelinePreview.disabled = true;
            btnPipelinePreview.textContent = "Rendering...";
            window.pixelforge.smartPipelinePreview(currentRecommendation.id, currentOverrides(), (result) => {
                btnPipelinePreview.disabled = false;
                btnPipelinePreview.textContent = "Preview";
                if (!result || !result.ok) {
                    showError((result && result.error) || "Preview failed.");
                    return;
                }
                if (result.recommendation) renderRecommendation(result.recommendation);
                // Reuse the existing before/after comparison layer --
                // this is the real, full recipe render, not a CSS
                // approximation, same as previewAdjust for the manual
                // sliders.
                imgAfter.src = toFileUrl(result.path) + "?t=" + Date.now();
                setComparePosition(50);
            });
        });
    }

    if (btnPipelineApply) {
        btnPipelineApply.addEventListener("click", () => {
            if (!currentRecommendation || !window.pixelforge) return;
            btnPipelineApply.disabled = true;
            btnPipelineApply.textContent = "Applying...";
            window.pixelforge.smartPipelineApply(currentRecommendation.id, currentOverrides(), (result) => {
                btnPipelineApply.disabled = false;
                btnPipelineApply.textContent = "Apply Pipeline";
                if (!result || !result.ok) {
                    showError((result && result.error) || "Couldn't apply the pipeline.");
                    return;
                }
                showError("");
                if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(result.session);

                const recipe = (result.recommendation && result.recommendation.recipe) || {};
                if (pipelineAppliedSummary) {
                    pipelineAppliedSummary.textContent =
                        `Applied "${result.recommendation.name}" -- ${recipe.applied_count} of ` +
                        `${(recipe.steps || []).length} steps ran at ${recipe.intensity}% strength.`;
                }
                if (pipelineAppliedPanel) pipelineAppliedPanel.classList.remove("view--hidden");
                if (pipelinePanel) pipelinePanel.classList.add("view--hidden");

                // Reload the working image the pipeline just committed --
                // same non-destructive hand-off pattern as Straighten/Crop
                // Apply above (resets sliders since the effect is baked in).
                loadImage(result.session.working_path);
            });
        });
    }

    if (btnPipelineUndoApplied) {
        btnPipelineUndoApplied.addEventListener("click", () => {
            if (!window.pixelforge || !window.pixelforge.session) return;
            window.pixelforge.session.undo((state) => {
                if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
                if (pipelineAppliedPanel) pipelineAppliedPanel.classList.add("view--hidden");
                loadImage(state.working_path);
            });
        });
    }

    // Path that Remove BG should receive. Manual Enhance/Auto Enhance previews are
    // real backend-rendered images, so hand the latest rendered preview across.
    window.pixelforgeGetEditorTransferPath = () => latestRenderedPath || currentPath;

    // ----- PHASE 6: Apply (commits the 14 sliders onto the shared Working Image) -----

    const btnApply = document.getElementById("btn-editor-apply");
    if (btnApply) {
        btnApply.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge || !window.pixelforge.session) return;
            btnApply.disabled = true;
            btnApply.textContent = "Applying...";
            window.pixelforge.session.applyAdjustments(currentAdjustments(), "Enhance", (state) => {
                btnApply.disabled = false;
                btnApply.textContent = "Apply";
                if (!state || !state.ok) {
                    showError((state && state.error) || "Couldn't apply adjustments.");
                    return;
                }
                if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
                // The adjustments are now baked into the Working Image, so
                // the saved slider positions from persistToolState() no
                // longer describe "unapplied changes" -- clear them before
                // reloading, or loadImage()'s getToolState() restore above
                // would re-apply the same look on top of the new baseline.
                clearTimeout(toolStateSaveTimer);
                if (window.pixelforge.session.setToolState) {
                    window.pixelforge.session.setToolState("enhance", {}, () => {
                        // The working image just changed underneath this view --
                        // reload it as the new baseline (sliders reset to 0/100
                        // since their effect is now baked in) so a second Apply
                        // doesn't double-apply the same adjustments.
                        loadImage(state.working_path);
                    });
                } else {
                    loadImage(state.working_path);
                }
            });
        });
    }

    // ----- Export -----

    if (btnExport) {
        btnExport.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge) return;

            const dotIndex = currentName.lastIndexOf(".");
            const base = dotIndex > -1 ? currentName.slice(0, dotIndex) : currentName;
            const suggested = `PixelForge_${base}.jpg`;

            window.pixelforge.chooseSaveImagePath(suggested, (destPath) => {
                if (!destPath) return; // cancelled

                btnExport.disabled = true;
                btnExport.textContent = "Exporting...";

                window.pixelforge.exportImage(currentPath, destPath, currentAdjustments(), (result) => {
                    btnExport.disabled = false;
                    btnExport.textContent = "Export";

                    if (result.ok) {
                        showError("");
                        if (window.pixelforge.setSetting) {
                            // Cheap toast substitute for now -- Phase 15 can
                            // replace this with a proper notification system.
                        }
                        alert(`Exported to:\n${result.path}`);

                        // Record this export in the active project's Edit
                        // History and refresh its thumbnail to the edited
                        // result -- otherwise a project's history/preview
                        // stayed empty forever no matter how much editing
                        // actually happened in it.
                        if (window.pixelforgeActiveDbProjectId && window.pixelforge.projects) {
                            window.pixelforge.projects.setThumbnail(
                                window.pixelforgeActiveDbProjectId, result.path, () => {}
                            );
                            if (window.pixelforgeActiveDbMediaId) {
                                window.pixelforge.projects.addEdit(
                                    window.pixelforgeActiveDbMediaId, "", "", currentAdjustments(), () => {}
                                );
                            }
                        }

                        // Let the user pick this exported (enhanced) file
                        // back up in Remove BG, so "enhance first, then
                        // remove/replace background" is a real one-click
                        // workflow instead of manually re-opening the file.
                        if (btnSendToRemoveBg) {
                            btnSendToRemoveBg.classList.remove("view--hidden");
                            btnSendToRemoveBg.onclick = () => {
                                if (window.pixelforgeShowView) window.pixelforgeShowView("removebg");
                                if (window.pixelforgeLoadImageIntoRemoveBG) {
                                    window.pixelforgeLoadImageIntoRemoveBG(result.path);
                                }
                            };
                        }
                    } else {
                        showError(result.error || "Export failed.");
                    }
                });
            });
        });
    }

    // Import button (top bar) and Home "Open Image" card already call
    // window.pixelforge.openImageDialog() directly (see ui.js). Route
    // their result into the editor too, so opening an image from
    // anywhere in the app lands you in the Enhance view with it loaded.
    window.pixelforgeLoadImageIntoEditor = loadImage;
});