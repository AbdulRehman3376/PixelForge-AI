// PHASE 10 -- Face Restoration ("Portrait") view logic.
//
// Same dropzone -> workspace shell, zoom/fit/reset/fullscreen toolbar,
// and Before/After compare-slider pattern as upscale.js/editor.js/
// removebg.js/filters.js (see those files for the original comments on
// why each piece works the way it does -- kept consistent here rather
// than re-explaining). What's new for this view:
//   - a "Detected Faces" checklist (spec's own example UI shape:
//     "Detected Faces: 4 / ☑ Face 1 ☑ Face 2 ☐ Face 3 ☐ Face 4"), each
//     row with its own optional per-face strength override -- backed
//     by ui/bridge.py's "PHASE 10: FACE RESTORATION" section
//   - a live-updating model status line (cached/ready vs. "downloads
//     ~340MB on first use" vs. CPU-warning), same pattern as Upscale
//   - Restoration Strength / Natural<->Detailed / Skin Protection
//     sliders
//   - a Cancel button wired to ai/face_restorer.py's cooperative
//     cancellation
//   - Apply commits straight onto the shared session Working Image
//     (ui/bridge.py::sessionApplyFaceRestoreAsync), Export goes through
//     the normal Save As + full-resolution export path
//   - "Face Before/After zoom": reuses the same zoom/fit/fullscreen +
//     Before/After compare-slider system every other tool view already
//     has (same reuse Phase 9 documented for its own zoom system),
//     rather than a separate per-face zoom mechanic.
//
// Exposes window.pixelforgeLoadImageIntoPortrait(path), same handoff
// pattern ui.js already uses for Enhance/Remove BG/Filters/Upscale.

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("portrait-dropzone");
    const workspace = document.getElementById("portrait-workspace");
    if (!dropzone || !workspace) return; // view not present in this build

    const btnOpenEmpty = document.getElementById("btn-portrait-open-empty");
    const btnReplace = document.getElementById("btn-portrait-replace");
    const canvasWrap = document.getElementById("portrait-canvas-wrap");
    const canvas = document.getElementById("portrait-canvas");
    const portraitWorkspace = document.getElementById("portrait-workspace");
    const btnFullscreenExit = document.getElementById("btn-portrait-fullscreen-exit");
    const zoomLevelEl = document.getElementById("portrait-zoom-level");
    const btnZoomIn = document.getElementById("btn-portrait-zoom-in");
    const btnZoomOut = document.getElementById("btn-portrait-zoom-out");
    const btnZoomFit = document.getElementById("btn-portrait-zoom-fit");
    const btnZoomReset = document.getElementById("btn-portrait-zoom-reset");
    const btnFullscreen = document.getElementById("btn-portrait-fullscreen");
    const imgBefore = document.getElementById("portrait-image-before");
    const imgAfter = document.getElementById("portrait-image-after");
    const afterWrap = document.getElementById("portrait-image-after-wrap");
    const compareHandle = document.getElementById("portrait-compare-handle");
    const compareSlider = document.getElementById("portrait-compare-slider");
    const fileNameEl = document.getElementById("portrait-filename");
    const dimensionsEl = document.getElementById("portrait-dimensions");
    const previewSpinner = document.getElementById("portrait-preview-spinner");
    const portraitError = document.getElementById("portrait-error");

    const modelStatusEl = document.getElementById("portrait-model-status");
    const faceListEl = document.getElementById("portrait-face-list");
    const faceCountEl = document.getElementById("portrait-face-count");
    const faceEmptyEl = document.getElementById("portrait-face-empty");

    const ctrlStrength = document.getElementById("ctrl-portrait-strength");
    const valStrength = document.getElementById("val-portrait-strength");
    const ctrlNaturalDetailed = document.getElementById("ctrl-portrait-natural-detailed");
    const valNaturalDetailed = document.getElementById("val-portrait-natural-detailed");
    const ctrlSkinProtection = document.getElementById("ctrl-portrait-skin-protection");
    const valSkinProtection = document.getElementById("val-portrait-skin-protection");

    const btnStart = document.getElementById("btn-portrait-start");
    const btnCancel = document.getElementById("btn-portrait-cancel");
    const btnApply = document.getElementById("btn-portrait-apply");
    const btnExport = document.getElementById("btn-portrait-export");

    let currentPath = null;
    let naturalWidth = 0;
    let naturalHeight = 0;
    let zoom = 1;

    // Each entry: {id, x, y, w, h, selected, strength}. `strength` is
    // this face's OWN override -- defaults to the global slider value
    // and only travels separately once the user edits that face's row
    // ("Per-face restoration ... different strength per selected face,
    // not one global setting").
    let detectedFaces = [];
    let taskRunning = false;
    let lastPreviewOk = false;

    // ----- Helpers (same shape as upscale.js) -----

    function toFileUrl(path) {
        const normalized = path.replace(/\\/g, "/").replace(/^\/+/, "");
        return "file:///" + encodeURI(normalized);
    }

    function showError(message) {
        if (!portraitError) return;
        portraitError.textContent = message;
        portraitError.classList.toggle("view--hidden", !message);
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

    function selectedFaceCount() {
        return detectedFaces.filter((f) => f.selected).length;
    }

    // ----- Build the faces[] + options object every bridge call needs -----

    function buildFacesPayload() {
        return detectedFaces.map((f) => ({
            id: f.id,
            x: f.x,
            y: f.y,
            w: f.w,
            h: f.h,
            selected: !!f.selected,
            strength: f.strength,
        }));
    }

    function buildOptions() {
        return {
            strength: Number(ctrlStrength.value) || 0,
            natural_detailed: Number(ctrlNaturalDetailed.value) || 50,
            skin_protection: Number(ctrlSkinProtection.value) || 0,
        };
    }

    function setTaskRunning(running) {
        taskRunning = running;
        if (btnCancel) btnCancel.disabled = !running;
        if (btnApply) btnApply.disabled = running || !lastPreviewOk || selectedFaceCount() === 0;
        if (btnExport) btnExport.disabled = running || !lastPreviewOk || selectedFaceCount() === 0;
        if (btnStart) {
            btnStart.disabled = running || !currentPath || selectedFaceCount() === 0;
            btnStart.textContent = running ? "Processing..." : "Start Face Restoration";
        }
    }

    // ----- Model status (cached / downloads ~340MB / CPU warning) -----

    function refreshModelStatus() {
        if (!modelStatusEl || !window.pixelforge || !window.pixelforge.faceRestore) return;
        window.pixelforge.faceRestore.modelStatus((result) => {
            if (!result || !result.ok) {
                modelStatusEl.textContent = "Couldn't check AI model status.";
                return;
            }
            const parts = [];
            parts.push(
                result.cached
                    ? "AI model ready."
                    : `First restoration downloads the AI model (~${result.approx_download_mb}MB, one-time only).`
            );
            parts.push(result.cpu_only ? "Running on CPU -- may take a few seconds per face." : "GPU acceleration available.");
            modelStatusEl.textContent = parts.join(" ");
        });
    }

    // ----- Detected Faces checklist -----
    //
    // Spec's own example UI shape:
    //   Detected Faces: 4
    //   ☑ Face 1  ☑ Face 2  ☐ Face 3  ☐ Face 4
    //   Strength: 35%
    // Rendered here as one row per face: a checkbox (selection), the
    // label, and a small per-face strength number input that starts
    // blank (meaning "use the global Strength slider") and only takes
    // over for that one face once the user actually edits it.

    function renderFaceList() {
        if (!faceListEl) return;
        faceListEl.innerHTML = "";

        if (faceCountEl) faceCountEl.textContent = String(detectedFaces.length);
        if (faceEmptyEl) faceEmptyEl.classList.toggle("view--hidden", detectedFaces.length > 0);
        faceListEl.classList.toggle("view--hidden", detectedFaces.length === 0);

        detectedFaces.forEach((face, idx) => {
            const row = document.createElement("div");
            row.className = "face-select-row";

            const label = document.createElement("label");
            label.className = "face-select-checkbox";

            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.checked = !!face.selected;
            checkbox.addEventListener("change", () => {
                face.selected = checkbox.checked;
                markPreviewStale();
                setTaskRunning(taskRunning);
                // MANUAL START: picking/unpicking a face just marks the
                // preview stale (Apply/Export gray out, Start re-enables)
                // -- it must NOT kick off the AI pass itself. Only the
                // "Start Face Restoration" button below does that.
            });

            const text = document.createElement("span");
            text.textContent = `Face ${idx + 1}`;

            label.appendChild(checkbox);
            label.appendChild(text);

            const strengthInput = document.createElement("input");
            strengthInput.type = "number";
            strengthInput.className = "input face-select-strength";
            strengthInput.min = "0";
            strengthInput.max = "100";
            strengthInput.placeholder = "auto";
            strengthInput.title = "Per-face strength override (blank = use the global Strength slider)";
            if (face.strength !== undefined && face.strength !== null) {
                strengthInput.value = String(face.strength);
            }
            strengthInput.addEventListener("input", () => {
                face.strength = strengthInput.value === "" ? undefined : Number(strengthInput.value);
                markPreviewStale();
            });
            // MANUAL START: committing a per-face strength override (blur/
            // enter) also just marks the preview stale -- Start Face
            // Restoration is the only thing that runs the AI pass.
            strengthInput.addEventListener("change", () => markPreviewStale());

            row.appendChild(label);
            row.appendChild(strengthInput);
            faceListEl.appendChild(row);
        });
    }

    function requestFaceDetection() {
        if (!currentPath || !window.pixelforge || !window.pixelforge.faceRestore) return;
        detectedFaces = [];
        renderFaceList();
        if (faceCountEl) faceCountEl.textContent = "…";

        window.pixelforge.faceRestore.detectFaces(currentPath, (result) => {
            if (!result || !result.ok) {
                showError((result && result.error) || "Couldn't detect faces in this image.");
                detectedFaces = [];
                renderFaceList();
                setTaskRunning(false);
                return;
            }
            // All faces start selected -- matches "restore what was
            // found" as the sensible default; unchecking is one click.
            detectedFaces = (result.faces || []).map((f) => ({ ...f, selected: true, strength: undefined }));
            renderFaceList();
            markPreviewStale();
            setTaskRunning(false);
        });
    }

    // ----- Live preview (debounced, downsized source) -----

    let previewDebounceTimer = null;
    let previewRequestSeq = 0;

    // MANUAL START: same reasoning as Phase 9's upscale.js -- the AI
    // pass only runs when the user explicitly clicks "Start Face
    // Restoration", not on every slider/checkbox change, so heavy
    // CPU-bound work doesn't kick off repeatedly before the user has
    // finished choosing which faces and how strongly to restore them.
    let previewStale = true;

    function markPreviewStale() {
        previewStale = true;
        lastPreviewOk = false;
        if (btnApply) btnApply.disabled = true;
        if (btnExport) btnExport.disabled = true;
        if (btnStart) btnStart.disabled = !currentPath || taskRunning || selectedFaceCount() === 0;
    }

    function requestPreviewUpdate() {
        if (!currentPath || !window.pixelforge || !window.pixelforge.faceRestore) return;
        showError("");
        previewStale = false;

        if (selectedFaceCount() === 0) {
            showError("Select at least one detected face to restore.");
            return;
        }

        clearTimeout(previewDebounceTimer);
        if (previewSpinner) previewSpinner.classList.add("is-visible");
        setTaskRunning(true);

        previewDebounceTimer = setTimeout(() => {
            const seq = ++previewRequestSeq;
            window.pixelforge.faceRestore.preview(currentPath, buildFacesPayload(), buildOptions(), (result) => {
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
        window.pixelforgePortraitCurrentPath = path;
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

            lastPreviewOk = false;
            setTaskRunning(false);
            setComparePosition(50);

            refreshModelStatus();
            imgAfter.src = url; // show the plain photo until Start is clicked -- no auto-run
            markPreviewStale();
            requestFaceDetection();

            requestAnimationFrame(fitToContainer);
            if (typeof onLoaded === "function") onLoaded();
        });
    }
    window.pixelforgeLoadImageIntoPortrait = loadImage;

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

    // ----- Drag & drop (same file.path assumption as upscale.js) -----

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

    // ----- AI Quality controls -----
    //
    // BUGFIX (was: sliders only updated their own label text and called
    // markPreviewStale() live-updates the label and grays out Apply/
    // Export/Start-until-changed-again, but must NOT itself trigger the
    // AI pass -- same "Manual Start" rule as Upscale's own sliders
    // (see the note above requestPreviewUpdate()). Restoration only
    // ever runs when the user explicitly presses "Start Face
    // Restoration", so settings can be dialed in first without the
    // heavy GFPGAN pass kicking off after every release.

    if (ctrlStrength) {
        ctrlStrength.addEventListener("input", () => {
            if (valStrength) valStrength.textContent = `${ctrlStrength.value}%`;
            markPreviewStale();
        });
    }
    if (ctrlNaturalDetailed) {
        ctrlNaturalDetailed.addEventListener("input", () => {
            if (valNaturalDetailed) {
                const v = Number(ctrlNaturalDetailed.value);
                valNaturalDetailed.textContent = v === 50 ? "Neutral" : (v < 50 ? `Natural ${50 - v}%` : `Detailed ${v - 50}%`);
            }
            markPreviewStale();
        });
    }
    if (ctrlSkinProtection) {
        ctrlSkinProtection.addEventListener("input", () => {
            if (valSkinProtection) valSkinProtection.textContent = `${ctrlSkinProtection.value}%`;
            markPreviewStale();
        });
    }

    // ----- Start Face Restoration (manual trigger, e.g. after picking/unpicking faces) -----

    if (btnStart) {
        btnStart.addEventListener("click", () => requestPreviewUpdate());
    }

    // ----- Before / After compare slider (same bugfix-shape as upscale.js) -----

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

    // ----- Zoom / fit / reset / fullscreen ("Face Before/After zoom") -----

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

    // Same bugfix as upscale.js/editor.js/filters.js: mark the rest of
    // the UI hidden ourselves on fullscreen change, since QtWebEngine
    // doesn't fully suppress painting of elements outside the
    // fullscreened node.
    document.addEventListener("fullscreenchange", () => {
        if (portraitWorkspace) {
            portraitWorkspace.classList.toggle("is-fullscreen-active", !!document.fullscreenElement);
        }
    });

    if (btnFullscreenExit) {
        btnFullscreenExit.addEventListener("click", () => {
            if (document.fullscreenElement) document.exitFullscreen();
        });
    }

    // ----- Cancel (cooperative -- ai/face_restorer.py checks this between faces) -----

    if (btnCancel) {
        btnCancel.addEventListener("click", () => {
            if (!window.pixelforge || !window.pixelforge.faceRestore) return;
            window.pixelforge.faceRestore.cancel();
        });
    }

    // ----- PHASE 6: Apply (commits onto the shared Working Image) -----

    if (btnApply) {
        btnApply.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge || !window.pixelforge.session || !window.pixelforge.faceRestore) return;
            btnApply.disabled = true;
            btnApply.textContent = "Applying...";
            setTaskRunning(true);
            window.pixelforge.faceRestore.apply(buildFacesPayload(), buildOptions(), "", (state) => {
                btnApply.textContent = "Apply";
                setTaskRunning(false);
                if (!state || !state.ok) {
                    if (!(state && state.cancelled)) {
                        showError((state && state.error) || "Couldn't apply Face Restoration.");
                    }
                    return;
                }
                if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
                // Reload as the new baseline -- same reasoning as
                // Upscale/Enhance/Filters' Apply: the restoration is now
                // baked into the photo itself, so the panel resets
                // against it (and re-detects faces against the new
                // pixels).
                loadImage(state.working_path);
            });
        });
    }

    // ----- Export final image -----

    if (btnExport) {
        btnExport.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge || !window.pixelforge.faceRestore) return;
            window.pixelforge.chooseSaveImagePath("PixelForge_FaceRestored.jpg", (destPath) => {
                if (!destPath) return;
                btnExport.disabled = true;
                btnExport.textContent = "Exporting...";
                setTaskRunning(true);
                window.pixelforge.faceRestore.export(currentPath, destPath, buildFacesPayload(), buildOptions(), (result) => {
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
    // as soon as the bridge is ready, same as upscale.js.
    if (window.onPixelforgeReady) {
        window.onPixelforgeReady(() => refreshModelStatus());
    }
});