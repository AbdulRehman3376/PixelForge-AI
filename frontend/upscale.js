// PHASE 9 -- AI Upscaling view logic.
//
// Same dropzone -> workspace shell, zoom/fit/reset/fullscreen toolbar,
// and Before/After compare-slider pattern as editor.js/removebg.js/
// filters.js (see those files for the original comments on why each
// piece works the way it does -- kept consistent here rather than
// re-explaining). What's new for this view:
//   - a Resolution selector (Original / 2x / 4x / Custom) instead of a
//     preset grid, backed by ai/upscaler.py via ui/bridge.py's
//     "PHASE 9: AI UPSCALING" section
//   - a live-updating model status line (cached/ready vs. "downloads
//     ~65MB on first use" vs. CPU-warning)
//   - a safety check (megapixel cap / RAM / disk) that runs before each
//     preview, same idea as Phase 8's Batch checkDiskSpace
//   - Denoise / Artifact Reduction sliders + a Face-aware toggle
//   - a Cancel button wired to ai/upscaler.py's cooperative cancellation
//   - Apply commits straight onto the shared session Working Image
//     (ui/bridge.py::sessionApplyUpscaleAsync), Export goes through the
//     normal Save As + full-resolution export path
//
// Exposes window.pixelforgeLoadImageIntoUpscale(path), same handoff
// pattern ui.js already uses for Enhance/Remove BG/Filters.

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("upscale-dropzone");
    const workspace = document.getElementById("upscale-workspace");
    if (!dropzone || !workspace) return; // view not present in this build

    const btnOpenEmpty = document.getElementById("btn-upscale-open-empty");
    const btnReplace = document.getElementById("btn-upscale-replace");
    const canvasWrap = document.getElementById("upscale-canvas-wrap");
    const canvas = document.getElementById("upscale-canvas");
    const upscaleWorkspace = document.getElementById("upscale-workspace");
    const btnFullscreenExit = document.getElementById("btn-upscale-fullscreen-exit");
    const zoomLevelEl = document.getElementById("upscale-zoom-level");
    const btnZoomIn = document.getElementById("btn-upscale-zoom-in");
    const btnZoomOut = document.getElementById("btn-upscale-zoom-out");
    const btnZoomFit = document.getElementById("btn-upscale-zoom-fit");
    const btnZoomReset = document.getElementById("btn-upscale-zoom-reset");
    const btnFullscreen = document.getElementById("btn-upscale-fullscreen");
    const imgBefore = document.getElementById("upscale-image-before");
    const imgAfter = document.getElementById("upscale-image-after");
    const afterWrap = document.getElementById("upscale-image-after-wrap");
    const compareHandle = document.getElementById("upscale-compare-handle");
    const compareSlider = document.getElementById("upscale-compare-slider");
    const fileNameEl = document.getElementById("upscale-filename");
    const dimensionsEl = document.getElementById("upscale-dimensions");
    const previewSpinner = document.getElementById("upscale-preview-spinner");
    const upscaleError = document.getElementById("upscale-error");

    const modelStatusEl = document.getElementById("upscale-model-status");
    const scaleTabsWrap = document.getElementById("upscale-scale-tabs");
    const scaleTabs = scaleTabsWrap ? Array.from(scaleTabsWrap.querySelectorAll(".bg-mode-tab")) : [];
    const customSizeRow = document.getElementById("upscale-custom-size-row");
    const inputWidth = document.getElementById("input-upscale-width");
    const inputHeight = document.getElementById("input-upscale-height");
    const chkPreserveAspect = document.getElementById("chk-upscale-preserve-aspect");
    const outputResolutionEl = document.getElementById("upscale-output-resolution");
    const safetyWarningEl = document.getElementById("upscale-safety-warning");

    const ctrlDenoise = document.getElementById("ctrl-upscale-denoise");
    const valDenoise = document.getElementById("val-upscale-denoise");
    const ctrlArtifact = document.getElementById("ctrl-upscale-artifact");
    const valArtifact = document.getElementById("val-upscale-artifact");
    const chkFaceAware = document.getElementById("chk-upscale-face-aware");

    const btnStart = document.getElementById("btn-upscale-start");
    const btnCancel = document.getElementById("btn-upscale-cancel");
    const btnApply = document.getElementById("btn-upscale-apply");
    const btnExport = document.getElementById("btn-upscale-export");

    let currentPath = null;
    let naturalWidth = 0;
    let naturalHeight = 0;
    let zoom = 1;

    let scaleChoice = "2x";
    let taskRunning = false;
    let lastPreviewOk = false;

    // ----- Helpers (same shape as editor.js/filters.js) -----

    function toFileUrl(path) {
        const normalized = path.replace(/\\/g, "/").replace(/^\/+/, "");
        return "file:///" + encodeURI(normalized);
    }

    function showError(message) {
        if (!upscaleError) return;
        upscaleError.textContent = message;
        upscaleError.classList.toggle("view--hidden", !message);
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

    // ----- Build the options object every bridge call needs -----

    function buildOptions() {
        return {
            scale: scaleChoice,
            custom_width: scaleChoice === "custom" ? (Number(inputWidth.value) || null) : null,
            custom_height: scaleChoice === "custom" ? (Number(inputHeight.value) || null) : null,
            preserve_aspect: !!(chkPreserveAspect && chkPreserveAspect.checked),
            denoise_strength: Number(ctrlDenoise.value) || 0,
            artifact_reduction: Number(ctrlArtifact.value) || 0,
            face_aware: !!(chkFaceAware && chkFaceAware.checked),
        };
    }

    function setTaskRunning(running) {
        taskRunning = running;
        if (btnCancel) btnCancel.disabled = !running;
        if (btnApply) btnApply.disabled = running || !lastPreviewOk;
        if (btnExport) btnExport.disabled = running || !lastPreviewOk;
        if (btnStart) {
            btnStart.disabled = running || !currentPath;
            btnStart.textContent = running ? "Processing..." : "Start AI Upscale";
        }
        scaleTabs.forEach((t) => { t.disabled = running; });
    }

    // ----- Model status (cached / downloads ~65MB / CPU warning) -----

    function refreshModelStatus() {
        if (!modelStatusEl || !window.pixelforge || !window.pixelforge.upscale) return;
        window.pixelforge.upscale.modelStatus((result) => {
            if (!result || !result.ok) {
                modelStatusEl.textContent = "Couldn't check AI model status.";
                return;
            }
            const parts = [];
            parts.push(
                result.cached
                    ? "AI model ready."
                    : `First upscale downloads the AI model (~${result.approx_download_mb}MB, one-time only).`
            );
            parts.push(result.cpu_only ? "Running on CPU -- large images may be slow." : "GPU acceleration available.");
            modelStatusEl.textContent = parts.join(" ");
        });
    }

    // ----- Safety check (megapixel cap / RAM / disk) -----

    function refreshSafetyCheck() {
        if (!currentPath || !window.pixelforge || !window.pixelforge.upscale) return;
        window.pixelforge.upscale.safetyCheck(currentPath, buildOptions(), (result) => {
            if (!result || !result.ok) {
                if (outputResolutionEl) outputResolutionEl.textContent = "";
                if (safetyWarningEl) {
                    safetyWarningEl.textContent = (result && (result.blocking || []).join(" ")) || result.error || "";
                    safetyWarningEl.classList.toggle("view--hidden", !safetyWarningEl.textContent);
                }
                return;
            }
            const out = result.estimated_output || {};
            if (outputResolutionEl && out.width) {
                outputResolutionEl.textContent = `Output: ${out.width} × ${out.height} (${out.megapixels}MP)`;
            }
            const messages = [...(result.blocking || []), ...(result.warnings || [])];
            if (safetyWarningEl) {
                safetyWarningEl.textContent = messages.join(" ");
                safetyWarningEl.classList.toggle("view--hidden", messages.length === 0);
            }
        });
    }

    // ----- Live preview (debounced, downsized source -- see ai/upscaler.py) -----

    let previewDebounceTimer = null;
    let previewRequestSeq = 0;

    // MANUAL START: this used to fire automatically on every option
    // change (scale tab, sliders, face-aware toggle, even right after
    // opening an image) which meant the heavy AI pass kicked off before
    // the user had finished choosing settings -- especially painful on
    // CPU-only machines where each run pegs the CPU and makes the whole
    // PC feel like it's hanging. Now requestPreviewUpdate() only runs
    // when the user explicitly clicks "Start AI Upscale" (btnStart
    // below); changing options just marks the preview stale so Start
    // is clearly still needed.
    let previewStale = true;

    function markPreviewStale() {
        previewStale = true;
        lastPreviewOk = false;
        if (btnApply) btnApply.disabled = true;
        if (btnExport) btnExport.disabled = true;
        if (btnStart) btnStart.disabled = !currentPath || taskRunning;
    }

    function requestPreviewUpdate() {
        if (!currentPath || !window.pixelforge || !window.pixelforge.upscale) return;
        showError("");
        refreshSafetyCheck();
        previewStale = false;

        if (scaleChoice === "original") {
            imgAfter.src = toFileUrl(currentPath) + `?t=${Date.now()}`;
            lastPreviewOk = true;
            setTaskRunning(false);
            return;
        }

        clearTimeout(previewDebounceTimer);
        if (previewSpinner) previewSpinner.classList.add("is-visible");
        setTaskRunning(true);

        previewDebounceTimer = setTimeout(() => {
            const seq = ++previewRequestSeq;
            window.pixelforge.upscale.preview(currentPath, buildOptions(), (result) => {
                if (seq !== previewRequestSeq) return; // superseded by a newer request
                if (previewSpinner) previewSpinner.classList.remove("is-visible");
                lastPreviewOk = !!(result && result.ok);
                setTaskRunning(false);
                if (!result || !result.ok) {
                    if (!(result && result.cancelled)) {
                        showError((result && result.error) || "Preview failed.");
                    }
                    return;
                }
                imgAfter.src = toFileUrl(result.path) + `?t=${Date.now()}`;
                if (dimensionsEl && result.label) {
                    dimensionsEl.textContent = `${naturalWidth} × ${naturalHeight} -> ${result.label}`;
                }
            });
        }, 220);
    }

    // ----- Load image -----

    function loadImage(path, onLoaded) {
        window.pixelforgeCurrentImage = path;
        // PHASE 6: see VIEW_CURRENT_PATH_VARS in ui.js.
        window.pixelforgeUpscaleCurrentPath = path;
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

            if (inputWidth) inputWidth.value = "";
            if (inputHeight) inputHeight.value = "";
            lastPreviewOk = false;
            setTaskRunning(false);
            setComparePosition(50);

            refreshModelStatus();
            imgAfter.src = url; // show the plain photo until Start is clicked -- no auto-run
            markPreviewStale();
            refreshSafetyCheck();

            requestAnimationFrame(fitToContainer);
            if (typeof onLoaded === "function") onLoaded();
        });
    }
    window.pixelforgeLoadImageIntoUpscale = loadImage;

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
            // start a session, same reasoning as editor.js/filters.js.
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

    // ----- Resolution tabs -----

    scaleTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            scaleChoice = tab.dataset.upscaleScale;
            scaleTabs.forEach((t) => t.classList.toggle("active", t === tab));
            if (customSizeRow) customSizeRow.classList.toggle("view--hidden", scaleChoice !== "custom");
            markPreviewStale();
            refreshSafetyCheck();
        });
    });

    if (inputWidth) inputWidth.addEventListener("input", () => { if (scaleChoice === "custom") { markPreviewStale(); refreshSafetyCheck(); } });
    if (inputHeight) inputHeight.addEventListener("input", () => { if (scaleChoice === "custom") { markPreviewStale(); refreshSafetyCheck(); } });
    if (chkPreserveAspect) chkPreserveAspect.addEventListener("change", () => { if (scaleChoice === "custom") { markPreviewStale(); refreshSafetyCheck(); } });

    // ----- AI Quality controls -----

    if (ctrlDenoise) {
        ctrlDenoise.addEventListener("input", () => {
            if (valDenoise) valDenoise.textContent = `${ctrlDenoise.value}%`;
            markPreviewStale();
        });
    }
    if (ctrlArtifact) {
        ctrlArtifact.addEventListener("input", () => {
            if (valArtifact) valArtifact.textContent = `${ctrlArtifact.value}%`;
            markPreviewStale();
        });
    }
    if (chkFaceAware) chkFaceAware.addEventListener("change", () => markPreviewStale());

    // ----- Start AI Upscale (manual trigger -- see markPreviewStale note above) -----

    if (btnStart) {
        btnStart.addEventListener("click", () => requestPreviewUpdate());
    }

    // ----- Before / After compare slider -----
    //
    // BUGFIX: unlike editor.js/filters.js/removebg.js, this view only ever
    // wired the hidden <input type="range"> (compareSlider) -- there was no
    // mousedown/mousemove handling on the visual handle itself, so dragging
    // the handle directly on the photo did nothing at all (the range input
    // is small/offscreen and not what people actually try to drag). Also,
    // per the same fix in editor.js: measure against the actual image
    // rect (canvas), not the surrounding canvasWrap, since the image is
    // fit-to-container/centered and usually smaller than the wrap -- using
    // the wrap's rect made the handle track only a fraction of the real
    // drag distance.

    let draggingCompareHandle = false;

    function positionFromEvent(clientX) {
        const rect = canvas.getBoundingClientRect();
        if (!rect.width) return;
        const percent = ((clientX - rect.left) / rect.width) * 100;
        setComparePosition(percent);
    }

    if (compareHandle) {
        compareHandle.addEventListener("mousedown", (e) => {
            e.preventDefault();
            draggingCompareHandle = true;
        });
        window.addEventListener("mouseup", () => { draggingCompareHandle = false; });
        window.addEventListener("mousemove", (e) => {
            if (draggingCompareHandle) positionFromEvent(e.clientX);
        });
    }
    if (compareSlider) {
        compareSlider.addEventListener("input", () => setComparePosition(Number(compareSlider.value)));
    }

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

    // Same bugfix as editor.js/filters.js: mark the rest of the UI
    // hidden ourselves on fullscreen change, since QtWebEngine doesn't
    // fully suppress painting of elements outside the fullscreened node.
    document.addEventListener("fullscreenchange", () => {
        if (upscaleWorkspace) {
            upscaleWorkspace.classList.toggle("is-fullscreen-active", !!document.fullscreenElement);
        }
    });

    if (btnFullscreenExit) {
        btnFullscreenExit.addEventListener("click", () => {
            if (document.fullscreenElement) document.exitFullscreen();
        });
    }

    // ----- Cancel (cooperative -- ai/upscaler.py checks this between tiles) -----

    if (btnCancel) {
        btnCancel.addEventListener("click", () => {
            if (!window.pixelforge || !window.pixelforge.upscale) return;
            window.pixelforge.upscale.cancel();
        });
    }

    // ----- PHASE 6: Apply (commits onto the shared Working Image) -----

    if (btnApply) {
        btnApply.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge || !window.pixelforge.session || !window.pixelforge.upscale) return;
            const options = buildOptions();
            btnApply.disabled = true;
            btnApply.textContent = "Applying...";
            setTaskRunning(true);
            window.pixelforge.upscale.apply(options, "", (state) => {
                btnApply.textContent = "Apply";
                setTaskRunning(false);
                if (!state || !state.ok) {
                    if (!(state && state.cancelled)) {
                        showError((state && state.error) || "Couldn't apply AI Upscale.");
                    }
                    return;
                }
                if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
                // Reload as the new baseline -- same reasoning as
                // Enhance/Filters' Apply: the upscale is now baked into
                // the photo itself, so the panel resets against it.
                loadImage(state.working_path);
            });
        });
    }

    // ----- Export final image -----

    if (btnExport) {
        btnExport.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge || !window.pixelforge.upscale) return;
            window.pixelforge.chooseSaveImagePath("PixelForge_Upscaled.jpg", (destPath) => {
                if (!destPath) return;
                const options = buildOptions();
                btnExport.disabled = true;
                btnExport.textContent = "Exporting...";
                setTaskRunning(true);
                window.pixelforge.upscale.export(currentPath, destPath, options, (result) => {
                    btnExport.textContent = "Export";
                    setTaskRunning(false);
                    if (!result || !result.ok) {
                        if (!(result && result.cancelled)) {
                            showError(result.error || "Export failed.");
                        }
                        return;
                    }
                    showError("");
                });
            });
        });
    }

    // Model status doesn't depend on an image being loaded -- check it
    // as soon as the bridge is ready so the "downloads ~65MB" / CPU
    // note is visible before the user ever opens a photo here.
    if (window.onPixelforgeReady) {
        window.onPixelforgeReady(() => refreshModelStatus());
    }
});