// PHASE 2A -- Background / Object Remover
// Handles: open image (dialog + drag/drop), the Background-removal tool
// (remove, edge feather, transparent/color/blur/image background,
// export) and the Object-erase tool (mask brush + inpaint, export), and
// the before/after comparison slider shared by both. Talks to the
// Python bridge via window.pixelforge.{removeBackgroundPreview,
// exportRemoveBackground, chooseBackgroundImage, eraseObjectPreview,
// exportEraseObject} -- see bridge.js / ui/bridge.py.
//
// Mirrors editor.js's structure (Phase 2) so the two screens stay easy
// to read side by side; kept in its own file rather than folded into
// editor.js since the two screens' state (adjustments sliders vs.
// bg-mode/mask-brush state) don't overlap.

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("removebg-dropzone");
    const workspace = document.getElementById("removebg-workspace");
    if (!dropzone || !workspace) return; // Remove BG view not present -- nothing to wire up.

    const btnOpenEmpty = document.getElementById("btn-removebg-open-empty");
    const btnReplace = document.getElementById("btn-removebg-replace");
    const canvasWrap = document.getElementById("removebg-canvas-wrap");
    const canvas = document.getElementById("removebg-canvas");
    const removebgWorkspace = document.getElementById("removebg-workspace");
    const btnFullscreenExit = document.getElementById("btn-removebg-fullscreen-exit");
    const zoomLevelEl = document.getElementById("removebg-zoom-level");
    const btnZoomIn = document.getElementById("btn-removebg-zoom-in");
    const btnZoomOut = document.getElementById("btn-removebg-zoom-out");
    const btnZoomFit = document.getElementById("btn-removebg-zoom-fit");
    const btnZoomReset = document.getElementById("btn-removebg-zoom-reset");
    const btnFullscreen = document.getElementById("btn-removebg-fullscreen");
    const imgBefore = document.getElementById("removebg-image-before");
    const imgAfter = document.getElementById("removebg-image-after");
    const afterWrap = document.getElementById("removebg-image-after-wrap");
    const maskCanvas = document.getElementById("removebg-mask-canvas");
    const compareHandle = document.getElementById("removebg-compare-handle");
    const bgCompareSlider = document.getElementById("removebg-compare-slider");
    const eraseCompareSlider = document.getElementById("erase-compare-slider");
    const fileNameEl = document.getElementById("removebg-filename");
    const dimensionsEl = document.getElementById("removebg-dimensions");
    const errorEl = document.getElementById("removebg-error");
    const previewSpinner = document.getElementById("removebg-preview-spinner");

    const toolTabs = Array.from(document.querySelectorAll(".tool-tab"));
    const panelBackground = document.getElementById("panel-tool-background");
    const panelErase = document.getElementById("panel-tool-erase");

    // ----- Background-removal tool elements -----
    const btnRemoveBg = document.getElementById("btn-remove-bg");
    const removeBgSummary = document.getElementById("removebg-summary");
    const edgeFeatherSlider = document.getElementById("ctrl-edge-feather");
    const edgeFeatherValue = document.getElementById("val-edge-feather");
    const bgModeTabs = Array.from(document.querySelectorAll(".bg-mode-tab[data-bg-mode]"));
    const bgOptionPanels = {
        transparent: null,
        color: document.getElementById("bg-options-color"),
        blur: document.getElementById("bg-options-blur"),
        gradient: document.getElementById("bg-options-gradient"),
        image: document.getElementById("bg-options-image"),
    };
    const bgColorInput = document.getElementById("ctrl-bg-color");
    const bgBlurSlider = document.getElementById("ctrl-bg-blur");
    const bgBlurValue = document.getElementById("val-bg-blur");
    // Missing-feature #8 (this session): Background Gradient controls.
    const bgGradientColor1Input = document.getElementById("ctrl-bg-gradient-color1");
    const bgGradientColor2Input = document.getElementById("ctrl-bg-gradient-color2");
    const bgGradientAngleSlider = document.getElementById("ctrl-bg-gradient-angle");
    const bgGradientAngleValue = document.getElementById("val-bg-gradient-angle");
    const btnChooseBgImage = document.getElementById("btn-choose-bg-image");
    const bgImageFilename = document.getElementById("bg-image-filename");
    const btnRemoveBgExport = document.getElementById("btn-removebg-export");
    const btnRemoveBgApply = document.getElementById("btn-removebg-apply"); // PHASE 6
    const btnSendToEnhance = document.getElementById("btn-removebg-send-to-enhance");
    const edgeExpandSlider = document.getElementById("ctrl-edge-expand");
    const edgeExpandValue = document.getElementById("val-edge-expand");
    const hairRefineCheckbox = document.getElementById("ctrl-hair-refine");
    // Missing-feature #6 (this session): Edge Decontamination toggle.
    const decontaminateCheckbox = document.getElementById("ctrl-decontaminate");
    // Missing-feature #9 (this session): Background Position/Scale controls.
    const bgImageScaleSlider = document.getElementById("ctrl-bg-image-scale");
    const bgImageScaleValue = document.getElementById("val-bg-image-scale");
    const bgImageOffsetXSlider = document.getElementById("ctrl-bg-image-offset-x");
    const bgImageOffsetXValue = document.getElementById("val-bg-image-offset-x");
    const bgImageOffsetYSlider = document.getElementById("ctrl-bg-image-offset-y");
    const bgImageOffsetYValue = document.getElementById("val-bg-image-offset-y");
    // Missing-feature #10 (this session): Subject-only preview toggle.
    const subjectOnlyCheckbox = document.getElementById("ctrl-subject-only");

    // ----- Touch-Up brush (Keep/Remove) elements -----
    const touchupCanvas = document.getElementById("touchup-mask-canvas");
    const touchupModeTabs = Array.from(document.querySelectorAll(".bg-mode-tab[data-touchup-mode]"));
    const touchupBrushSlider = document.getElementById("ctrl-touchup-brush-size");
    const touchupBrushValue = document.getElementById("val-touchup-brush-size");
    const btnTouchupClear = document.getElementById("btn-touchup-clear");
    const btnTouchupToggle = document.getElementById("btn-touchup-toggle");
    const btnTouchupUndo = document.getElementById("btn-touchup-undo");
    const btnTouchupRedo = document.getElementById("btn-touchup-redo");
    const btnTouchupApply = document.getElementById("btn-touchup-apply");

    // ----- Shadow elements -----
    const shadowModeTabs = Array.from(document.querySelectorAll(".bg-mode-tab[data-shadow-mode]"));
    const rowShadowStrength = document.getElementById("row-shadow-strength");
    const shadowStrengthSlider = document.getElementById("ctrl-shadow-strength");
    const shadowStrengthValue = document.getElementById("val-shadow-strength");

    // ----- Manual Crop elements -----
    const btnCropStart = document.getElementById("btn-crop-start");
    const btnCropApply = document.getElementById("btn-crop-apply");
    const btnCropCancel = document.getElementById("btn-crop-cancel");
    const btnCropReset = document.getElementById("btn-crop-reset");
    const cropOverlay = document.getElementById("crop-overlay");
    const cropBoxEl = document.getElementById("crop-box");
    const cropHandles = cropOverlay ? Array.from(cropOverlay.querySelectorAll(".crop-handle")) : [];

    // ----- Object-erase tool elements -----
    const brushSizeSlider = document.getElementById("ctrl-brush-size");
    const brushSizeValue = document.getElementById("val-brush-size");
    const btnClearMask = document.getElementById("btn-clear-mask");
    const eraseMethodTabs = Array.from(document.querySelectorAll(".bg-mode-tab[data-erase-method]"));
    const btnEraseObject = document.getElementById("btn-erase-object");
    const eraseSummary = document.getElementById("erase-summary");
    const btnEraseExport = document.getElementById("btn-erase-export");
    const btnEraseApply = document.getElementById("btn-erase-apply"); // PHASE 6

    // ----- Undo/Redo elements (optional -- wired defensively like the
    // rest of this file, so this still works fine if the toolbar buttons
    // haven't been added to index.html yet) -----
    const btnBgUndo = document.getElementById("btn-removebg-undo");
    const btnBgRedo = document.getElementById("btn-removebg-redo");
    const btnMaskUndo = document.getElementById("btn-erase-undo");
    const btnMaskRedo = document.getElementById("btn-erase-redo");

    const bgHistory = createHistory(30); // Background tool: one entry per remove/replace-background result
    const maskHistory = createHistory(50); // Erase tool: one entry per completed brush stroke

    let currentPath = null;
    let currentName = null;
    let activeTool = "background"; // "background" | "erase"
    let bgMode = "transparent";
    let eraseMethod = "telea";
    let hasCutout = false; // true once removeBackgroundPreview has succeeded at least once
    let latestBgResultPath = null;   // PHASE 6: latest Background-tool preview render (for Apply)
    let latestEraseResultPath = null; // PHASE 6: latest Object-Erase preview render (for Apply)
    let backgroundImagePath = null;
    let maskHasStrokes = false;
    let previewToken = 0; // guards against a slow older preview overwriting a newer one
    let naturalWidth = 0;
    let naturalHeight = 0;
    let zoom = 1;
    const ZOOM_MIN = 0.1;
    const ZOOM_MAX = 4;

    // ----- Touch-up brush state -----
    let touchupMode = "keep"; // "keep" | "remove"
    let touchupBrushSize = touchupBrushSlider ? Number(touchupBrushSlider.value) : 28;
    let touchupHasStrokes = false;
    let touchupBrushEnabled = false;
    const touchupHistory = createHistory(50); // one entry per completed brush stroke, same pattern as maskHistory
    // Once "Apply Touch-Up" is clicked, this holds the painted mask's data
    // URL so it keeps getting re-applied to the cutout on every future
    // composite (feather/bg-mode/shadow/crop changes) -- not just the one
    // preview right after Apply. Cleared by "Clear" or a new image load.
    let appliedTouchupMaskDataUrl = null;

    // ----- Shadow state -----
    let shadowMode = "none"; // "none" | "preserve" | "remove"

    // ----- Manual Crop state -----
    let cropActive = false; // true while the crop overlay/handles are being dragged (not yet applied)
    let cropApplied = false; // true once a crop_box has actually been sent to the backend
    // cropBox is always in the ORIGINAL image's pixel coordinates (left, top, right, bottom),
    // regardless of how zoomed/scaled the on-screen canvas currently is.
    let cropBox = null;

    // ----- Undo/Redo -----
    // Shared by both tools: the Background tool undoes/redoes through its
    // successive remove/replace-background RESULTS (each already has a
    // rendered file from removeBackgroundPreview, so undo/redo here is just
    // swapping which cached result + control values are shown -- no extra
    // backend call needed). The Erase tool undoes/redoes individual BRUSH
    // STROKES on the mask canvas, which is the more natural "undo" for a
    // paint tool -- see maskHistory below.
    function createHistory(maxSize) {
        let stack = [];
        let index = -1; // index of the current state within stack

        return {
            push(state) {
                // A push after an undo starts a new branch -- drop the
                // redo states it would otherwise clobber silently.
                stack = stack.slice(0, index + 1);
                stack.push(state);
                if (stack.length > maxSize) stack.shift();
                index = stack.length - 1;
            },
            canUndo: () => index > 0,
            canRedo: () => index < stack.length - 1,
            undo() {
                if (index <= 0) return null;
                index -= 1;
                return stack[index];
            },
            redo() {
                if (index >= stack.length - 1) return null;
                index += 1;
                return stack[index];
            },
            reset() {
                stack = [];
                index = -1;
            },
        };
    }

    // ----- Helpers -----

    function toFileUrl(path) {
        // See editor.js's identical helper for why this normalization
        // is needed (Windows paths + spaces/special chars).
        const normalized = path.replace(/\\/g, "/").replace(/^\/+/, "");
        return "file:///" + encodeURI(normalized);
    }

    function showError(message) {
        if (!errorEl) return;
        errorEl.textContent = message;
        errorEl.classList.toggle("view--hidden", !message);
    }

    function setSpinner(visible) {
        if (previewSpinner) previewSpinner.classList.toggle("is-visible", visible);
    }

    function updateCompare(percent) {
        const clamped = Math.max(0, Math.min(100, percent));
        if (afterWrap) afterWrap.style.clipPath = `inset(0 ${100 - clamped}% 0 0)`;
        if (compareHandle) compareHandle.style.left = `${clamped}%`;
    }

    // Missing-feature #11 (this session): transparent checkerboard
    // preview. The "After" result actually has transparency whenever
    // the effective mode is "transparent" -- either the Background tab
    // is literally set to Transparent, or the Subject-only-preview
    // toggle is forcing that mode for inspection regardless of the
    // real tab selection (see buildBgOptions() above).
    function updateCheckerboardPreview() {
        if (!afterWrap) return;
        const isTransparentPreview = bgMode === "transparent" || !!(subjectOnlyCheckbox && subjectOnlyCheckbox.checked);
        afterWrap.classList.toggle("checkerboard-bg", isTransparentPreview);
    }

    function resizeMaskCanvas() {
        if (!maskCanvas || !imgBefore) return;
        const rect = imgBefore.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        // Redraw-preserving resize would need extra bookkeeping this
        // first version skips -- switching tools or resizing the window
        // mid-paint is rare enough that clearing the mask is an
        // acceptable tradeoff over the complexity of preserving it.
        maskCanvas.width = Math.round(rect.width);
        maskCanvas.height = Math.round(rect.height);
        const ctx = maskCanvas.getContext("2d");
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
        maskHasStrokes = false;
        updateEraseButtonState();

        // A resize/clear invalidates any prior strokes' pixel dimensions,
        // so this blank canvas becomes the new undo baseline.
        maskHistory.reset();
        maskHistory.push(ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
        updateMaskUndoRedoButtons();
    }

    function updateEraseButtonState() {
        if (btnEraseObject) btnEraseObject.disabled = !maskHasStrokes;
    }

    // ----- Touch-Up brush canvas sizing -----
    // Same reasoning as resizeMaskCanvas above, but left TRANSPARENT by
    // default (not filled black) -- core/ai's apply_manual_mask() treats
    // "this canvas pixel's own alpha is 0" as "untouched, leave the auto
    // cutout's alpha alone", so an unpainted area must actually be
    // transparent, not a solid color.
    function resizeTouchupCanvas() {
        if (!touchupCanvas || !imgBefore) return;
        const rect = imgBefore.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        touchupCanvas.width = Math.round(rect.width);
        touchupCanvas.height = Math.round(rect.height);
        touchupHasStrokes = false;
        updateTouchupButtonState();
        touchupHistory.reset();
        const ctx = touchupCanvas.getContext("2d");
        touchupHistory.push(ctx.getImageData(0, 0, touchupCanvas.width, touchupCanvas.height));
        updateTouchupUndoRedoButtons();
    }

    function updateTouchupButtonState() {
        if (btnTouchupApply) btnTouchupApply.disabled = !touchupHasStrokes;
    }

    function updateTouchupControlsEnabled() {
        const enabled = hasCutout;
        if (btnTouchupToggle) {
            btnTouchupToggle.disabled = !enabled;
            btnTouchupToggle.textContent = touchupBrushEnabled ? "Disable Brush" : "Enable Brush";
        }
        if (touchupBrushSlider) touchupBrushSlider.disabled = !enabled;
        if (btnTouchupClear) btnTouchupClear.disabled = !enabled;
        updateTouchupButtonState();
    }

    function updateTouchupUndoRedoButtons() {
        if (btnTouchupUndo) btnTouchupUndo.disabled = !touchupHistory.canUndo();
        if (btnTouchupRedo) btnTouchupRedo.disabled = !touchupHistory.canRedo();
    }

    // ----- Tool tab switching (Background vs Erase Object) -----

    function setActiveTool(tool) {
        activeTool = tool;
        toolTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tool === tool));
        panelBackground.classList.toggle("view--hidden", tool !== "background");
        panelErase.classList.toggle("view--hidden", tool !== "erase");
        maskCanvas.classList.toggle("view--hidden", tool !== "erase");
        if (tool === "erase") {
            resizeMaskCanvas();
            cancelCropMode();
        }
        // Touch-up canvas only paints while the Background tool is active
        // AND the user hasn't got the crop overlay open (the two overlays
        // would otherwise fight for the same mouse drags).
        if (touchupCanvas) {
            touchupCanvas.classList.toggle("view--hidden", tool !== "background" || cropActive || !touchupBrushEnabled);
            if (tool === "background" && touchupBrushEnabled) resizeTouchupCanvas();
        }
    }

    toolTabs.forEach((tab) => {
        tab.addEventListener("click", () => setActiveTool(tab.dataset.tool));
    });

    // BUGFIX: #removebg-canvas (class .editor-canvas) is deliberately given
    // width:0/height:0 in CSS -- it's meant to be sized in real pixels by JS
    // once the loaded image's dimensions are known (see the comment on
    // .editor-canvas in main.css, and editor.js's identical sizeStageToImage/
    // fitToContainer pair). This file never did that, so the canvas box that
    // hosts the before/after <img>s (both width:100%/height:100% of it)
    // stayed 0x0 forever -- the photo never became visible, no matter how
    // many times you loaded or reloaded it.
    function sizeStageToImage() {
        if (!canvas || !naturalWidth || !naturalHeight) return;
        canvas.style.width = `${naturalWidth}px`;
        canvas.style.height = `${naturalHeight}px`;
    }

    function applyZoom() {
        canvas.style.transform = `translate(-50%, -50%) scale(${zoom})`;
        if (zoomLevelEl) zoomLevelEl.textContent = `${Math.round(zoom * 100)}%`;
    }

    function setZoom(next) {
        zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, next));
        applyZoom();
    }

    function fitToContainer() {
        if (!canvas || !naturalWidth || !naturalHeight || !canvasWrap) return;
        const rect = canvasWrap.getBoundingClientRect();
        const padding = 32;
        const scaleX = (rect.width - padding) / naturalWidth;
        const scaleY = (rect.height - padding) / naturalHeight;
        setZoom(Math.min(scaleX, scaleY, 1));
    }

    // ----- Loading an image into the workspace -----

    function loadImage(path) {
        // Remembered so other tools (Enhance, etc.) can pick up the
        // same photo without the user having to browse for it again --
        // see window.pixelforgeCurrentImage / showView() in ui.js.
        window.pixelforgeCurrentImage = path;
        window.pixelforgeRemoveBGCurrentPath = path;
        currentPath = path;
        currentName = path.split(/[\\/]/).pop();
        showError("");
        hasCutout = false;
        maskHasStrokes = false;
        if (btnSendToEnhance) btnSendToEnhance.classList.add("view--hidden");

        // A new photo means neither tool's history applies anymore.
        bgHistory.reset();
        maskHistory.reset();
        updateBgUndoRedoButtons();
        updateMaskUndoRedoButtons();

        if (window.pixelforge) {
            window.pixelforge.getImageInfo(path, (info) => {
                if (!info.ok) {
                    showError(info.error || "Couldn't read image.");
                    return;
                }
                fileNameEl.textContent = info.name;
                dimensionsEl.textContent = `${info.width} × ${info.height}`;

                // BUGFIX: this is what actually makes the photo visible --
                // see sizeStageToImage()/fitToContainer() above.
                naturalWidth = info.width;
                naturalHeight = info.height;
                sizeStageToImage();
                requestAnimationFrame(fitToContainer);
            });
        }

        const url = toFileUrl(path);
        imgBefore.src = url;
        imgAfter.src = url; // shows the original until the first Remove/Erase runs
        if (afterWrap) afterWrap.classList.remove("checkerboard-bg");

        dropzone.classList.add("view--hidden");
        workspace.classList.remove("view--hidden");

        edgeFeatherSlider.disabled = true;
        edgeFeatherSlider.value = 0;
        edgeFeatherValue.textContent = "0";
        if (edgeExpandSlider) {
            edgeExpandSlider.disabled = true;
            edgeExpandSlider.value = 0;
            edgeExpandValue.textContent = "0";
        }
        if (hairRefineCheckbox) {
            hairRefineCheckbox.disabled = true;
            hairRefineCheckbox.checked = false;
        }
        if (decontaminateCheckbox) {
            decontaminateCheckbox.disabled = true;
            decontaminateCheckbox.checked = false;
        }
        if (subjectOnlyCheckbox) {
            subjectOnlyCheckbox.disabled = true;
            subjectOnlyCheckbox.checked = false;
        }
        btnRemoveBgExport.disabled = true;
        btnEraseExport.disabled = true;
        if (btnRemoveBgApply) btnRemoveBgApply.disabled = true;
        if (btnEraseApply) btnEraseApply.disabled = true;
        removeBgSummary.textContent = "Cuts the subject out using a local AI model (rembg) — nothing leaves your computer.";
        eraseSummary.textContent = "";
        updateCompare(50);
        bgCompareSlider.value = 50;
        eraseCompareSlider.value = 50;

        // Reset Touch-Up brush
        appliedTouchupMaskDataUrl = null;
        touchupHasStrokes = false;
        touchupBrushEnabled = false;
        touchupHistory.reset();
        if (touchupCanvas) touchupCanvas.classList.add("view--hidden");
        updateTouchupControlsEnabled();
        updateTouchupUndoRedoButtons();

        // Reset Shadow
        shadowMode = "none";
        shadowModeTabs.forEach((t) => t.classList.toggle("active", t.dataset.shadowMode === "none"));
        if (rowShadowStrength) rowShadowStrength.classList.add("view--hidden");

        // Reset Crop
        cancelCropMode();
        cropApplied = false;
        cropBox = null;
        if (btnCropReset) btnCropReset.disabled = true;

        imgBefore.onload = () => {
            if (activeTool === "erase") resizeMaskCanvas();
            if (activeTool === "background") resizeTouchupCanvas();
        };
    }

    // Exposed the same way editor.js exposes window.pixelforgeLoadImageIntoEditor,
    // so a future "open in Remove BG" entry point (Home quick-action, top-bar
    // Import, etc.) can route here without needing to know this file's internals.
    window.pixelforgeLoadImageIntoRemoveBG = loadImage;

    function openWithSession(path) {
        if (window.pixelforgeStartSession) {
            window.pixelforgeStartSession(path, (state) => loadImage((state && state.working_path) || path));
        } else {
            loadImage(path);
        }
    }

    if (btnOpenEmpty) {
        btnOpenEmpty.addEventListener("click", () => {
            if (!window.pixelforge) return;
            window.pixelforge.openImageDialog((path) => {
                if (path) openWithSession(path);
            });
        });
    }

    if (btnReplace) {
        btnReplace.addEventListener("click", () => {
            if (!window.pixelforge) return;
            window.pixelforge.openImageDialog((path) => {
                if (path) openWithSession(path);
            });
        });
    }

    // ----- Drag & drop (same pattern as editor.js) -----

    ["dragenter", "dragover"].forEach((evt) => {
        [dropzone, workspace].forEach((el) => {
            el.addEventListener(evt, (e) => {
                e.preventDefault();
                el.classList.add("is-dragover");
            });
        });
    });
    ["dragleave", "drop"].forEach((evt) => {
        [dropzone, workspace].forEach((el) => {
            el.addEventListener(evt, (e) => {
                e.preventDefault();
                el.classList.remove("is-dragover");
            });
        });
    });
    [dropzone, workspace].forEach((el) => {
        el.addEventListener("drop", (e) => {
            const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            // PHASE 6: starts a session too -- same reasoning as editor.js/filters.js.
            if (file && file.path) openWithSession(file.path);
        });
    });

    // ----- Before/after compare slider + draggable handle -----

    function wireCompareSlider(slider) {
        if (!slider) return;
        slider.addEventListener("input", () => updateCompare(Number(slider.value)));
    }
    wireCompareSlider(bgCompareSlider);
    wireCompareSlider(eraseCompareSlider);

    if (compareHandle && canvasWrap) {
        let dragging = false;
        compareHandle.addEventListener("mousedown", () => { dragging = true; });
        window.addEventListener("mouseup", () => { dragging = false; });
        window.addEventListener("mousemove", (e) => {
            if (!dragging) return;
            const rect = canvasWrap.getBoundingClientRect();
            const percent = ((e.clientX - rect.left) / rect.width) * 100;
            updateCompare(percent);
            const activeSlider = activeTool === "erase" ? eraseCompareSlider : bgCompareSlider;
            if (activeSlider) activeSlider.value = Math.max(0, Math.min(100, percent));
        });
    }

    // ===================== BACKGROUND REMOVAL TOOL =====================

    function buildBgOptions() {
        const options = {
            mode: subjectOnlyCheckbox && subjectOnlyCheckbox.checked ? "transparent" : bgMode,
            edge_feather: Number(edgeFeatherSlider.value),
            hair_refine: !!(hairRefineCheckbox && hairRefineCheckbox.checked),
            expand_contract: edgeExpandSlider ? Number(edgeExpandSlider.value) : 0,
            shadow_mode: shadowMode,
            // Missing-feature #6 (this session): Edge Decontamination.
            decontaminate: !!(decontaminateCheckbox && decontaminateCheckbox.checked),
        };
        if (shadowMode === "preserve" && shadowStrengthSlider) {
            options.shadow_strength = Number(shadowStrengthSlider.value);
        }
        if (bgMode === "color") {
            const hex = bgColorInput.value || "#ffffff";
            options.color = [
                parseInt(hex.slice(1, 3), 16),
                parseInt(hex.slice(3, 5), 16),
                parseInt(hex.slice(5, 7), 16),
            ];
        } else if (bgMode === "blur") {
            options.blur_radius = Number(bgBlurSlider.value);
        } else if (bgMode === "gradient") {
            // Missing-feature #8 (this session): Background Gradient.
            const hexToRgb = (hex) => [
                parseInt(hex.slice(1, 3), 16),
                parseInt(hex.slice(3, 5), 16),
                parseInt(hex.slice(5, 7), 16),
            ];
            options.color1 = hexToRgb((bgGradientColor1Input && bgGradientColor1Input.value) || "#1e1e28");
            options.color2 = hexToRgb((bgGradientColor2Input && bgGradientColor2Input.value) || "#c8c8dc");
            options.angle = bgGradientAngleSlider ? Number(bgGradientAngleSlider.value) : 90;
        } else if (bgMode === "image") {
            options.background_path = backgroundImagePath;
            // Missing-feature #9 (this session): Background Position/Scale.
            options.scale = bgImageScaleSlider ? Number(bgImageScaleSlider.value) / 100 : 1.0;
            options.offset_x = bgImageOffsetXSlider ? Number(bgImageOffsetXSlider.value) : 50;
            options.offset_y = bgImageOffsetYSlider ? Number(bgImageOffsetYSlider.value) : 50;
        }
        if (appliedTouchupMaskDataUrl) {
            options.touchup_mask = appliedTouchupMaskDataUrl;
        }
        if (cropApplied && cropBox) {
            options.crop_box = cropBox;
            options.crop_source_width = naturalWidth;
            options.crop_source_height = naturalHeight;
        }
        return options;
    }

    function runBackgroundPreview() {
        if (!currentPath || !window.pixelforge) return;
        const subjectOnlyActive = !!(subjectOnlyCheckbox && subjectOnlyCheckbox.checked);
        if (!subjectOnlyActive && bgMode === "image" && !backgroundImagePath) {
            showError("Choose a background image first.");
            return;
        }
        showError("");
        setSpinner(true);
        const token = ++previewToken;
        window.pixelforge.removeBackgroundPreview(currentPath, buildBgOptions(), (result) => {
            if (token !== previewToken) return; // a newer request already superseded this one
            setSpinner(false);
            if (!result.ok) {
                // BUGFIX: this used to be immediately overwritten by the
                // "Background removed..." message below, regardless of
                // whether removeBackgroundPreview actually succeeded --
                // so a failure (e.g. a missing rembg model/dependency)
                // silently looked like a success in the summary text
                // while the real error only showed in the small error
                // banner. The summary now only updates once we know the
                // real outcome.
                removeBgSummary.textContent = "";
                showError(result.error || "Background removal failed.");
                return;
            }
            hasCutout = true;
            latestBgResultPath = result.path;
            edgeFeatherSlider.disabled = false;
            if (edgeExpandSlider) edgeExpandSlider.disabled = false;
            if (hairRefineCheckbox) hairRefineCheckbox.disabled = false;
            if (decontaminateCheckbox) decontaminateCheckbox.disabled = false;
            if (subjectOnlyCheckbox) subjectOnlyCheckbox.disabled = false;
            btnRemoveBgExport.disabled = false;
            if (btnRemoveBgApply) btnRemoveBgApply.disabled = false;
            if (btnCropStart) btnCropStart.disabled = false;
            updateTouchupControlsEnabled();
            imgAfter.src = toFileUrl(result.path) + "?t=" + Date.now();
            updateCheckerboardPreview();
            removeBgSummary.textContent = "Background removed. Adjust edge feather or pick a new background below.";

            bgHistory.push({
                path: result.path,
                edgeFeather: Number(edgeFeatherSlider.value),
                edgeExpand: edgeExpandSlider ? Number(edgeExpandSlider.value) : 0,
                hairRefine: !!(hairRefineCheckbox && hairRefineCheckbox.checked),
                decontaminate: !!(decontaminateCheckbox && decontaminateCheckbox.checked),
                bgMode,
                bgColor: bgColorInput.value,
                bgBlur: Number(bgBlurSlider.value),
                bgGradientColor1: bgGradientColor1Input ? bgGradientColor1Input.value : "#1e1e28",
                bgGradientColor2: bgGradientColor2Input ? bgGradientColor2Input.value : "#c8c8dc",
                bgGradientAngle: bgGradientAngleSlider ? Number(bgGradientAngleSlider.value) : 90,
                backgroundImagePath,
                bgImageScale: bgImageScaleSlider ? Number(bgImageScaleSlider.value) : 100,
                bgImageOffsetX: bgImageOffsetXSlider ? Number(bgImageOffsetXSlider.value) : 50,
                bgImageOffsetY: bgImageOffsetYSlider ? Number(bgImageOffsetYSlider.value) : 50,
                shadowMode,
                shadowStrength: shadowStrengthSlider ? Number(shadowStrengthSlider.value) : 50,
                touchupMask: appliedTouchupMaskDataUrl,
                cropBox: cropApplied ? cropBox : null,
            });
            updateBgUndoRedoButtons();
        });
    }

    if (btnRemoveBg) {
        btnRemoveBg.addEventListener("click", () => {
            removeBgSummary.textContent = "Removing background…";
            runBackgroundPreview();
        });
    }

    // Debounced re-preview on edge-feather change, same pattern as
    // editor.js's manual-control sliders -- only meaningful once a
    // cutout exists (slider stays disabled until then).
    let featherDebounce = null;
    edgeFeatherSlider.addEventListener("input", () => {
        edgeFeatherValue.textContent = edgeFeatherSlider.value;
        if (!hasCutout) return;
        clearTimeout(featherDebounce);
        featherDebounce = setTimeout(runBackgroundPreview, 150);
    });

    let expandDebounce = null;
    if (edgeExpandSlider) {
        edgeExpandSlider.addEventListener("input", () => {
            edgeExpandValue.textContent = edgeExpandSlider.value;
            if (!hasCutout) return;
            clearTimeout(expandDebounce);
            expandDebounce = setTimeout(runBackgroundPreview, 150);
        });
    }

    // Hair Refine re-runs the actual rembg model (not just a cheap
    // composite), so it's noticeably slower than the other sliders --
    // no debounce needed since it's a discrete toggle, not a drag.
    if (hairRefineCheckbox) {
        hairRefineCheckbox.addEventListener("change", () => {
            if (!hasCutout) return;
            removeBgSummary.textContent = hairRefineCheckbox.checked
                ? "Re-running with AI Hair Refine…"
                : "Re-running…";
            runBackgroundPreview();
        });
    }

    bgModeTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            bgMode = tab.dataset.bgMode;
            bgModeTabs.forEach((t) => t.classList.toggle("active", t === tab));
            Object.entries(bgOptionPanels).forEach(([mode, panel]) => {
                if (panel) panel.classList.toggle("view--hidden", mode !== bgMode);
            });
            if (hasCutout) runBackgroundPreview();
        });
    });

    bgColorInput.addEventListener("input", () => {
        if (hasCutout && bgMode === "color") runBackgroundPreview();
    });

    shadowModeTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            shadowMode = tab.dataset.shadowMode;
            shadowModeTabs.forEach((t) => t.classList.toggle("active", t === tab));
            if (rowShadowStrength) rowShadowStrength.classList.toggle("view--hidden", shadowMode !== "preserve");
            if (hasCutout) runBackgroundPreview();
        });
    });

    let shadowDebounce = null;
    if (shadowStrengthSlider) {
        shadowStrengthSlider.addEventListener("input", () => {
            if (shadowStrengthValue) shadowStrengthValue.textContent = shadowStrengthSlider.value;
            if (!hasCutout || shadowMode !== "preserve") return;
            clearTimeout(shadowDebounce);
            shadowDebounce = setTimeout(runBackgroundPreview, 150);
        });
    }

    let blurDebounce = null;
    bgBlurSlider.addEventListener("input", () => {
        bgBlurValue.textContent = bgBlurSlider.value;
        if (!hasCutout || bgMode !== "blur") return;
        clearTimeout(blurDebounce);
        blurDebounce = setTimeout(runBackgroundPreview, 150);
    });

    // Missing-feature #6 (this session): Edge Decontamination toggle --
    // a cheap post-process on the existing cutout, no debounce needed.
    if (decontaminateCheckbox) {
        decontaminateCheckbox.addEventListener("change", () => {
            if (hasCutout) runBackgroundPreview();
        });
    }

    // Missing-feature #8 (this session): Background Gradient controls.
    let gradientDebounce = null;
    const scheduleGradientPreview = () => {
        if (!hasCutout || bgMode !== "gradient") return;
        clearTimeout(gradientDebounce);
        gradientDebounce = setTimeout(runBackgroundPreview, 150);
    };
    if (bgGradientColor1Input) bgGradientColor1Input.addEventListener("input", scheduleGradientPreview);
    if (bgGradientColor2Input) bgGradientColor2Input.addEventListener("input", scheduleGradientPreview);
    if (bgGradientAngleSlider) {
        bgGradientAngleSlider.addEventListener("input", () => {
            if (bgGradientAngleValue) bgGradientAngleValue.textContent = `${bgGradientAngleSlider.value}°`;
            scheduleGradientPreview();
        });
    }

    // Missing-feature #9 (this session): Background Position/Scale --
    // pan/zoom the chosen background image, only meaningful in Image mode.
    let bgImageTransformDebounce = null;
    const scheduleBgImagePreview = () => {
        if (!hasCutout || bgMode !== "image" || !backgroundImagePath) return;
        clearTimeout(bgImageTransformDebounce);
        bgImageTransformDebounce = setTimeout(runBackgroundPreview, 150);
    };
    if (bgImageScaleSlider) {
        bgImageScaleSlider.addEventListener("input", () => {
            if (bgImageScaleValue) bgImageScaleValue.textContent = `${bgImageScaleSlider.value}%`;
            scheduleBgImagePreview();
        });
    }
    if (bgImageOffsetXSlider) {
        bgImageOffsetXSlider.addEventListener("input", () => {
            if (bgImageOffsetXValue) bgImageOffsetXValue.textContent = `${bgImageOffsetXSlider.value}%`;
            scheduleBgImagePreview();
        });
    }
    if (bgImageOffsetYSlider) {
        bgImageOffsetYSlider.addEventListener("input", () => {
            if (bgImageOffsetYValue) bgImageOffsetYValue.textContent = `${bgImageOffsetYSlider.value}%`;
            scheduleBgImagePreview();
        });
    }

    // Missing-feature #10 (this session): Subject-only preview toggle --
    // overrides the effective mode to "transparent" inside
    // buildBgOptions() above without touching the actual bgMode/tab
    // selection, so unchecking just re-runs with whatever was already
    // selected.
    if (subjectOnlyCheckbox) {
        subjectOnlyCheckbox.addEventListener("change", () => {
            if (hasCutout) runBackgroundPreview();
        });
    }

    if (btnChooseBgImage) {
        btnChooseBgImage.addEventListener("click", () => {
            if (!window.pixelforge) return;
            window.pixelforge.chooseBackgroundImage((path) => {
                if (!path) return;
                backgroundImagePath = path;
                bgImageFilename.textContent = path.split(/[\\/]/).pop();
                if (hasCutout && bgMode === "image") runBackgroundPreview();
            });
        });
    }

    // ----- PHASE 6: Apply (commits the Background result onto the shared Working Image) -----

    if (btnRemoveBgApply) {
        btnRemoveBgApply.addEventListener("click", () => {
            if (!window.pixelforge || !window.pixelforge.session || !latestBgResultPath) return;
            btnRemoveBgApply.disabled = true;
            btnRemoveBgApply.textContent = "Applying...";
            window.pixelforge.session.commitFile(latestBgResultPath, "removebg", "Remove/Replace Background", (state) => {
                btnRemoveBgApply.disabled = false;
                btnRemoveBgApply.textContent = "Apply";
                if (!state || !state.ok) {
                    showError((state && state.error) || "Couldn't apply this result.");
                    return;
                }
                if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
                // Reload as the new baseline so the next edit (a new
                // background swap, a crop, etc.) builds on this result
                // instead of re-running against the pre-cutout photo.
                loadImage(state.working_path);
            });
        });
    }

    if (btnRemoveBgExport) {
        btnRemoveBgExport.addEventListener("click", () => {
            if (!window.pixelforge || !currentPath) return;
            const suggested = currentName ? currentName.replace(/\.[^.]+$/, "") + "_nobg.png" : "PixelForge_NoBG.png";
            window.pixelforge.chooseSaveImagePath(suggested, (destPath) => {
                if (!destPath) return;
                setSpinner(true);
                window.pixelforge.exportRemoveBackground(currentPath, destPath, buildBgOptions(), (result) => {
                    setSpinner(false);
                    if (!result.ok) {
                        showError(result.error || "Export failed.");
                        return;
                    }
                    removeBgSummary.textContent = result.note
                        ? `Exported to ${destPath}. ${result.note}`
                        : `Exported to ${destPath}.`;

                    // Let the user pick this exported cutout/composite
                    // back up in Enhance, so "remove/replace background
                    // first, then enhance" is a real one-click workflow.
                    if (btnSendToEnhance) {
                        btnSendToEnhance.classList.remove("view--hidden");
                        btnSendToEnhance.onclick = () => {
                            if (window.pixelforgeShowView) window.pixelforgeShowView("enhance");
                            if (window.pixelforgeLoadImageIntoEditor) {
                                window.pixelforgeLoadImageIntoEditor(result.path);
                            }
                        };
                    }
                });
            });
        });
    }

    // ===================== OBJECT ERASE TOOL =====================

    let painting = false;
    let currentBrushSize = Number(brushSizeSlider.value);

    function canvasPointFromEvent(e) {
        const rect = maskCanvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
        };
    }

    function paintAt(point) {
        const ctx = maskCanvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(point.x, point.y, currentBrushSize / 2, 0, Math.PI * 2);
        ctx.fill();
        maskHasStrokes = true;
        updateEraseButtonState();
    }

    maskCanvas.addEventListener("mousedown", (e) => {
        if (activeTool !== "erase") return;
        painting = true;
        paintAt(canvasPointFromEvent(e));
    });
    maskCanvas.addEventListener("mousemove", (e) => {
        if (!painting || activeTool !== "erase") return;
        paintAt(canvasPointFromEvent(e));
    });
    window.addEventListener("mouseup", () => {
        // A whole stroke (mousedown -> drag -> mouseup) is one undo step,
        // not one step per paintAt() dab -- otherwise Ctrl+Z would only
        // erase a tiny fraction of the stroke the user just drew.
        if (painting && activeTool === "erase") {
            const ctx = maskCanvas.getContext("2d");
            maskHistory.push(ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
            updateMaskUndoRedoButtons();
        }
        painting = false;
    });
    maskCanvas.addEventListener("mouseleave", () => { /* keep painting=true so dragging back in resumes cleanly */ });

    brushSizeSlider.addEventListener("input", () => {
        currentBrushSize = Number(brushSizeSlider.value);
        brushSizeValue.textContent = brushSizeSlider.value;
    });

    if (btnClearMask) {
        btnClearMask.addEventListener("click", () => resizeMaskCanvas());
    }

    eraseMethodTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            eraseMethod = tab.dataset.eraseMethod;
            eraseMethodTabs.forEach((t) => t.classList.toggle("active", t === tab));
        });
    });

    function maskDataUrl() {
        return maskCanvas.toDataURL("image/png");
    }

    if (btnEraseObject) {
        btnEraseObject.addEventListener("click", () => {
            if (!currentPath || !window.pixelforge || !maskHasStrokes) return;
            eraseSummary.textContent = "Erasing…";
            setSpinner(true);
            window.pixelforge.eraseObjectPreview(currentPath, maskDataUrl(), eraseMethod, (result) => {
                setSpinner(false);
                if (!result.ok) {
                    showError(result.error || "Object erase failed.");
                    eraseSummary.textContent = "";
                    return;
                }
                btnEraseExport.disabled = false;
                if (btnEraseApply) btnEraseApply.disabled = false;
                latestEraseResultPath = result.path;
                imgAfter.src = toFileUrl(result.path) + "?t=" + Date.now();
                // Object Erase output is always fully opaque (inpainted),
                // never transparent, regardless of whatever the
                // Background tab's mode/subject-only state is.
                if (afterWrap) afterWrap.classList.remove("checkerboard-bg");
                eraseSummary.textContent = "Object erased. Export below, or paint more and erase again.";
            });
        });
    }

    // ----- PHASE 6: Apply (commits the Object-Erase result onto the shared Working Image) -----

    if (btnEraseApply) {
        btnEraseApply.addEventListener("click", () => {
            if (!window.pixelforge || !window.pixelforge.session || !latestEraseResultPath) return;
            btnEraseApply.disabled = true;
            btnEraseApply.textContent = "Applying...";
            window.pixelforge.session.commitFile(latestEraseResultPath, "removebg", "Erase Object", (state) => {
                btnEraseApply.disabled = false;
                btnEraseApply.textContent = "Apply";
                if (!state || !state.ok) {
                    showError((state && state.error) || "Couldn't apply this result.");
                    return;
                }
                if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
                loadImage(state.working_path);
            });
        });
    }

    if (btnEraseExport) {
        btnEraseExport.addEventListener("click", () => {
            if (!window.pixelforge || !currentPath) return;
            const suggested = currentName ? currentName.replace(/\.[^.]+$/, "") + "_erased" + (currentName.match(/\.[^.]+$/) || [".jpg"])[0] : "PixelForge_Erased.jpg";
            window.pixelforge.chooseSaveImagePath(suggested, (destPath) => {
                if (!destPath) return;
                setSpinner(true);
                window.pixelforge.exportEraseObject(currentPath, destPath, maskDataUrl(), eraseMethod, (result) => {
                    setSpinner(false);
                    if (!result.ok) {
                        showError(result.error || "Export failed.");
                        return;
                    }
                    eraseSummary.textContent = `Exported to ${destPath}.`;
                });
            });
        });
    }

    // ===================== TOUCH-UP BRUSH (Keep/Remove) =====================

    let touchupPainting = false;
    let currentTouchupBrushSize = touchupBrushSlider ? Number(touchupBrushSlider.value) : 28;

    function touchupPointFromEvent(e) {
        const rect = touchupCanvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    function touchupPaintAt(point) {
        const ctx = touchupCanvas.getContext("2d");
        // Keep strokes render white, Remove strokes render black -- this
        // matches core/ai/bg_remover.py's apply_manual_mask() exactly:
        // white-ish (r>128) = force keep, black-ish (r<=128) = force remove.
        ctx.fillStyle = touchupMode === "keep" ? "#ffffff" : "#000000";
        ctx.beginPath();
        ctx.arc(point.x, point.y, currentTouchupBrushSize / 2, 0, Math.PI * 2);
        ctx.fill();
        touchupHasStrokes = true;
        updateTouchupButtonState();
    }

    if (touchupCanvas) {
        touchupCanvas.addEventListener("mousedown", (e) => {
            if (activeTool !== "background" || cropActive || !touchupBrushEnabled) return;
            touchupPainting = true;
            touchupPaintAt(touchupPointFromEvent(e));
        });
        touchupCanvas.addEventListener("mousemove", (e) => {
            if (!touchupPainting || activeTool !== "background" || cropActive || !touchupBrushEnabled) return;
            touchupPaintAt(touchupPointFromEvent(e));
        });
        window.addEventListener("mouseup", () => {
            if (touchupPainting) {
                const ctx = touchupCanvas.getContext("2d");
                touchupHistory.push(ctx.getImageData(0, 0, touchupCanvas.width, touchupCanvas.height));
                updateTouchupUndoRedoButtons();
            }
            touchupPainting = false;
        });
        touchupCanvas.addEventListener("mouseleave", () => { /* keep painting=true so dragging back in resumes cleanly */ });
    }

    if (touchupBrushSlider) {
        touchupBrushSlider.addEventListener("input", () => {
            currentTouchupBrushSize = Number(touchupBrushSlider.value);
            touchupBrushValue.textContent = touchupBrushSlider.value;
        });
    }

    touchupModeTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            touchupMode = tab.dataset.touchupMode;
            touchupModeTabs.forEach((t) => t.classList.toggle("active", t === tab));
        });
    });

    function restoreTouchupState(imageData) {
        if (!imageData || !touchupCanvas) return;
        if (imageData.width !== touchupCanvas.width || imageData.height !== touchupCanvas.height) return;
        touchupCanvas.getContext("2d").putImageData(imageData, 0, 0);
        // Whether strokes remain depends on the restored snapshot itself,
        // not just "we did an undo" -- undoing all the way back to the
        // blank baseline should re-disable Apply.
        const ctx = touchupCanvas.getContext("2d");
        const px = ctx.getImageData(0, 0, touchupCanvas.width, touchupCanvas.height).data;
        touchupHasStrokes = false;
        for (let i = 3; i < px.length; i += 4) {
            if (px[i] > 10) { touchupHasStrokes = true; break; }
        }
        updateTouchupButtonState();
    }

    function undoTouchup() {
        if (!touchupHistory.canUndo()) return;
        restoreTouchupState(touchupHistory.undo());
        updateTouchupUndoRedoButtons();
    }
    function redoTouchup() {
        if (!touchupHistory.canRedo()) return;
        restoreTouchupState(touchupHistory.redo());
        updateTouchupUndoRedoButtons();
    }

    if (btnTouchupUndo) btnTouchupUndo.addEventListener("click", undoTouchup);
    if (btnTouchupRedo) btnTouchupRedo.addEventListener("click", redoTouchup);
    if (btnTouchupToggle) btnTouchupToggle.addEventListener("click", () => {
        if (!hasCutout) return;
        touchupBrushEnabled = !touchupBrushEnabled;
        touchupPainting = false;
        if (touchupCanvas) {
            touchupCanvas.classList.toggle("view--hidden", !touchupBrushEnabled || activeTool !== "background" || cropActive);
            if (touchupBrushEnabled) resizeTouchupCanvas();
        }
        updateTouchupControlsEnabled();
    });
    if (btnTouchupClear) btnTouchupClear.addEventListener("click", () => {
        if (!touchupCanvas) return;
        const ctx = touchupCanvas.getContext("2d");
        ctx.clearRect(0, 0, touchupCanvas.width, touchupCanvas.height);
        touchupHasStrokes = false;
        touchupHistory.reset();
        touchupHistory.push(ctx.getImageData(0, 0, touchupCanvas.width, touchupCanvas.height));
        updateTouchupButtonState();
        updateTouchupUndoRedoButtons();
    });

    if (btnTouchupApply) {
        btnTouchupApply.addEventListener("click", () => {
            if (!touchupCanvas || !touchupHasStrokes || !hasCutout) return;
            // Replaces (not merges with) any previously-applied touch-up --
            // a simplification that keeps this file's state model simple;
            // painting again after Apply starts a fresh set of strokes on
            // a cleared canvas, so it's still additive from the user's
            // point of view as long as they don't need BOTH an old and a
            // brand-new stroke to survive at once.
            appliedTouchupMaskDataUrl = touchupCanvas.toDataURL("image/png");
            removeBgSummary.textContent = "Applying touch-up…";
            runBackgroundPreview();
            resizeTouchupCanvas();
        });
    }

    // ===================== MANUAL CROP =====================

    function positionCropBoxFromNaturalBox(box) {
        // Converts an ORIGINAL-image-pixel box into on-screen percentages
        // of the (possibly scaled) canvas stage, so the crop box tracks
        // correctly regardless of the current fitToContainer() zoom.
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

    function enterCropMode() {
        if (!cropOverlay || !currentPath || !naturalWidth || !naturalHeight) return;
        cropActive = true;
        if (touchupCanvas) touchupCanvas.classList.add("view--hidden");
        cropOverlay.classList.remove("view--hidden");
        if (btnCropStart) btnCropStart.classList.add("view--hidden");
        if (btnCropApply) btnCropApply.classList.remove("view--hidden");
        if (btnCropCancel) btnCropCancel.classList.remove("view--hidden");
        // Start from the existing crop (if any) or a centered 80% box.
        const startBox = cropBox || [
            Math.round(naturalWidth * 0.1),
            Math.round(naturalHeight * 0.1),
            Math.round(naturalWidth * 0.9),
            Math.round(naturalHeight * 0.9),
        ];
        positionCropBoxFromNaturalBox(startBox);
    }

    function cancelCropMode() {
        cropActive = false;
        if (cropOverlay) cropOverlay.classList.add("view--hidden");
        if (btnCropStart) btnCropStart.classList.remove("view--hidden");
        if (btnCropApply) btnCropApply.classList.add("view--hidden");
        if (btnCropCancel) btnCropCancel.classList.add("view--hidden");
        if (touchupCanvas && activeTool === "background" && touchupBrushEnabled) touchupCanvas.classList.remove("view--hidden");
    }

    if (btnCropStart) btnCropStart.addEventListener("click", enterCropMode);
    if (btnCropCancel) btnCropCancel.addEventListener("click", cancelCropMode);

    if (btnCropApply) {
        btnCropApply.addEventListener("click", () => {
            const box = naturalBoxFromCropBoxEl();
            if (!box || box[2] <= box[0] || box[3] <= box[1]) {
                cancelCropMode();
                return;
            }
            cropBox = box;
            cropApplied = true;
            if (btnCropReset) btnCropReset.disabled = false;
            cancelCropMode();
            if (hasCutout) runBackgroundPreview();
        });
    }

    if (btnCropReset) {
        btnCropReset.addEventListener("click", () => {
            cropBox = null;
            cropApplied = false;
            btnCropReset.disabled = true;
            if (hasCutout) runBackgroundPreview();
        });
    }

    // ----- Dragging the crop box itself (move) and its 4 corner handles (resize) -----
    if (cropBoxEl && cropOverlay) {
        let dragMode = null; // null | "move" | "nw" | "ne" | "sw" | "se"
        let dragStart = { x: 0, y: 0 };
        let boxStart = { left: 0, top: 0, width: 0, height: 0 };

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
            }

            cropBoxEl.style.left = `${left}%`;
            cropBoxEl.style.top = `${top}%`;
            cropBoxEl.style.width = `${width}%`;
            cropBoxEl.style.height = `${height}%`;
        });

        window.addEventListener("mouseup", () => { dragMode = null; });
    }

    // ===================== UNDO / REDO =====================

    function updateBgUndoRedoButtons() {
        if (btnBgUndo) btnBgUndo.disabled = !bgHistory.canUndo();
        if (btnBgRedo) btnBgRedo.disabled = !bgHistory.canRedo();
    }

    function updateMaskUndoRedoButtons() {
        if (btnMaskUndo) btnMaskUndo.disabled = !maskHistory.canUndo();
        if (btnMaskRedo) btnMaskRedo.disabled = !maskHistory.canRedo();
    }

    // Restores a previous background-tool result: swaps the shown image
    // AND the control values (edge feather, bg mode + its per-mode
    // options) back to what produced it, so re-running "Remove Background"
    // afterward continues from a state the user actually recognizes.
    function applyBgState(state) {
        if (!state) return;

        edgeFeatherSlider.value = state.edgeFeather;
        edgeFeatherValue.textContent = String(state.edgeFeather);

        if (edgeExpandSlider) {
            edgeExpandSlider.value = state.edgeExpand || 0;
            edgeExpandValue.textContent = String(state.edgeExpand || 0);
        }
        if (hairRefineCheckbox) hairRefineCheckbox.checked = !!state.hairRefine;

        shadowMode = state.shadowMode || "none";
        shadowModeTabs.forEach((t) => t.classList.toggle("active", t.dataset.shadowMode === shadowMode));
        if (rowShadowStrength) rowShadowStrength.classList.toggle("view--hidden", shadowMode !== "preserve");
        if (shadowStrengthSlider && state.shadowStrength != null) shadowStrengthSlider.value = state.shadowStrength;
        if (shadowStrengthValue && state.shadowStrength != null) shadowStrengthValue.textContent = String(state.shadowStrength);

        appliedTouchupMaskDataUrl = state.touchupMask || null;

        cropBox = state.cropBox || null;
        cropApplied = !!cropBox;
        if (btnCropReset) btnCropReset.disabled = !cropApplied;

        bgMode = state.bgMode;
        bgModeTabs.forEach((t) => t.classList.toggle("active", t.dataset.bgMode === bgMode));
        Object.entries(bgOptionPanels).forEach(([mode, panel]) => {
            if (panel) panel.classList.toggle("view--hidden", mode !== bgMode);
        });

        bgColorInput.value = state.bgColor;
        bgBlurSlider.value = state.bgBlur;
        bgBlurValue.textContent = String(state.bgBlur);
        if (bgGradientColor1Input && state.bgGradientColor1) bgGradientColor1Input.value = state.bgGradientColor1;
        if (bgGradientColor2Input && state.bgGradientColor2) bgGradientColor2Input.value = state.bgGradientColor2;
        if (bgGradientAngleSlider && state.bgGradientAngle != null) {
            bgGradientAngleSlider.value = state.bgGradientAngle;
            if (bgGradientAngleValue) bgGradientAngleValue.textContent = `${state.bgGradientAngle}°`;
        }
        backgroundImagePath = state.backgroundImagePath;
        if (backgroundImagePath) bgImageFilename.textContent = backgroundImagePath.split(/[\\/]/).pop();
        if (bgImageScaleSlider && state.bgImageScale != null) {
            bgImageScaleSlider.value = state.bgImageScale;
            if (bgImageScaleValue) bgImageScaleValue.textContent = `${state.bgImageScale}%`;
        }
        if (bgImageOffsetXSlider && state.bgImageOffsetX != null) {
            bgImageOffsetXSlider.value = state.bgImageOffsetX;
            if (bgImageOffsetXValue) bgImageOffsetXValue.textContent = `${state.bgImageOffsetX}%`;
        }
        if (bgImageOffsetYSlider && state.bgImageOffsetY != null) {
            bgImageOffsetYSlider.value = state.bgImageOffsetY;
            if (bgImageOffsetYValue) bgImageOffsetYValue.textContent = `${state.bgImageOffsetY}%`;
        }
        if (decontaminateCheckbox) decontaminateCheckbox.checked = !!state.decontaminate;

        hasCutout = true;
        latestBgResultPath = state.path;
        btnRemoveBgExport.disabled = false;
        if (btnRemoveBgApply) btnRemoveBgApply.disabled = false;
        imgAfter.src = toFileUrl(state.path) + "?t=" + Date.now();
        updateCheckerboardPreview();
        removeBgSummary.textContent = "Restored a previous result.";
    }

    function restoreMaskState(imageData) {
        if (!imageData) return;
        // Guard against a stale snapshot from before a canvas resize --
        // putImageData with mismatched dimensions throws.
        if (imageData.width !== maskCanvas.width || imageData.height !== maskCanvas.height) return;
        maskCanvas.getContext("2d").putImageData(imageData, 0, 0);
        maskHasStrokes = true;
        updateEraseButtonState();
    }

    function undoBg() {
        if (!bgHistory.canUndo()) return;
        applyBgState(bgHistory.undo());
        updateBgUndoRedoButtons();
    }
    function redoBg() {
        if (!bgHistory.canRedo()) return;
        applyBgState(bgHistory.redo());
        updateBgUndoRedoButtons();
    }
    function undoMaskStroke() {
        if (!maskHistory.canUndo()) return;
        restoreMaskState(maskHistory.undo());
        updateMaskUndoRedoButtons();
    }
    function redoMaskStroke() {
        if (!maskHistory.canRedo()) return;
        restoreMaskState(maskHistory.redo());
        updateMaskUndoRedoButtons();
    }

    if (btnBgUndo) btnBgUndo.addEventListener("click", undoBg);
    if (btnBgRedo) btnBgRedo.addEventListener("click", redoBg);
    if (btnMaskUndo) btnMaskUndo.addEventListener("click", undoMaskStroke);
    if (btnMaskRedo) btnMaskRedo.addEventListener("click", redoMaskStroke);

    updateBgUndoRedoButtons();
    updateMaskUndoRedoButtons();

    // Ctrl+Z / Ctrl+Y (and Ctrl+Shift+Z) -- routed to whichever tool is
    // active, and only while this view is actually the one on screen so
    // it doesn't steal undo from Enhance or another view's own shortcuts.
    window.addEventListener("keydown", (e) => {
        if (workspace.classList.contains("view--hidden")) return;
        if (!(e.ctrlKey || e.metaKey)) return;

        const key = e.key.toLowerCase();
        if (key === "z" && !e.shiftKey) {
            e.preventDefault();
            if (activeTool === "erase") undoMaskStroke();
            else undoBg();
        } else if (key === "y" || (key === "z" && e.shiftKey)) {
            e.preventDefault();
            if (activeTool === "erase") redoMaskStroke();
            else redoBg();
        }
    });

    window.addEventListener("resize", () => {
        if (currentPath) fitToContainer();
        if (activeTool === "erase" && currentPath) resizeMaskCanvas();
        if (activeTool === "background" && currentPath && !cropActive) resizeTouchupCanvas();
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
        if (removebgWorkspace) {
            removebgWorkspace.classList.toggle("is-fullscreen-active", !!document.fullscreenElement);
        }
    });

    if (btnFullscreenExit) {
        btnFullscreenExit.addEventListener("click", () => {
            if (document.fullscreenElement) document.exitFullscreen();
        });
    }
});