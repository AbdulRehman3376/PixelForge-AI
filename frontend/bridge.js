// Connects the frontend to the Python backend via QWebChannel.
// Exposes a single global: window.pixelforge
//
// Usage from other JS files:
//   window.pixelforge.ready(() => { ... })
//   window.pixelforge.openImageDialog((path) => { ... })
//
// PHASE 2 ADDITIONS: getImageInfo, chooseSaveImagePath, exportImage
// PHASE 2A ADDITIONS: removeBackgroundPreview, exportRemoveBackground,
// chooseBackgroundImage, eraseObjectPreview, exportEraseObject
// PHASE 3 ADDITIONS: straightenImage, cropImage (Straighten & Crop)
// PHASE 4 ADDITIONS: listFilterPresets, previewFilter, exportFilterImage,
// saveCustomPreset, renameCustomPreset, duplicateCustomPreset,
// deleteCustomPreset, setPresetFavorite, exportPresetToFile,
// importPresetFromFile (Presets / Smart Filters)
// PHASE 5 ADDITIONS: analyzeImage (Analyzer -- subject/scene/face
// detection, lighting, dominant colors, quality; see core/analyzer.py
// and ai/face_detector.py)
// PHASE 6 ADDITIONS: smartPipeline (Smart Pipeline -- rule engine that
// consumes Analyzer output and recommends a named preset+intensity
// pipeline, e.g. "Night Portrait"; see core/smart_pipeline.py)
// PHASE 7 ADDITIONS: promptEnginePreview, promptHistoryList/Delete/Clear
// (local rule-based Prompt Engine, see core/prompt_engine.py);
// aiImageGenAvailability, generateImage, regenerateImage,
// sendGeneratedImageToEditor (AI Image Generation -- local diffusers
// pipeline by default, optional cloud API; see ai/image_generator.py)
// PHASE 8 ADDITIONS: window.pixelforge.batch.* -- queue/orchestration
// only, no model of its own (see core/batch_processor.py, ui/bridge.py's
// "PHASE 8: BATCH PROCESSING" section).
// (see ui/bridge.py for the Python-side implementations).

(function () {
    let bridgeReady = false;
    const readyCallbacks = [];

    function whenReady(cb) {
        if (bridgeReady) {
            cb();
        } else {
            readyCallbacks.push(cb);
        }
    }

    // Exposed immediately (synchronously), independent of script load order,
    // so other JS files can safely call window.onPixelforgeReady(cb) from
    // their own DOMContentLoaded handler without a race condition.
    window.onPixelforgeReady = whenReady;

    function setBridgeStatus(state) {
        // state: 'ok' | 'error'
        const dot = document.getElementById("bridge-status");
        if (!dot) return;
        dot.classList.remove("status-dot--pending", "status-dot--ok", "status-dot--error");
        dot.classList.add(state === "ok" ? "status-dot--ok" : "status-dot--error");
    }

    // Small helper: Python Slots that return JSON strings (getImageInfo,
    // exportImage) get parsed here so callers just deal with plain objects.
    function parseJson(raw, cb) {
        try {
            cb(JSON.parse(raw));
        } catch (err) {
            cb({ ok: false, error: "Unexpected response from backend." });
        }
    }

    function init() {
        if (typeof qt === "undefined" || typeof QWebChannel === "undefined") {
            // Not running inside the PySide6 QWebEngineView (e.g. opened directly
            // in a regular browser for quick CSS/JS iteration). Fail quietly.
            console.warn("PixelForge: no Qt bridge available (running outside the app shell).");
            setBridgeStatus("error");
            return;
        }

        new QWebChannel(qt.webChannelTransport, (channel) => {
            const pyBridge = channel.objects.pyBridge;

            // Listeners registered via window.pixelforge.onProgress/onStatus.
            // Kept here (not per-view) so any tool -- batch, upscale, video
            // export, etc. -- can report into the same global status bar.
            const progressListeners = [];
            const statusListeners = [];

            pyBridge.progressChanged.connect((percent, label) => {
                progressListeners.forEach((cb) => cb(percent, label));
            });
            pyBridge.statusChanged.connect((status) => {
                statusListeners.forEach((cb) => cb(status));
            });

            // BUGFIX: removeBackgroundPreview / exportRemoveBackground /
            // eraseObjectPreview / exportEraseObject used to be plain
            // Slots that ran (and blocked the whole window) directly on
            // the Qt GUI thread -- rembg's model load alone can take
            // long enough to make the app look frozen/"Not Responding",
            // with whatever view was on screen before the freeze stuck
            // there until it finished. Python now runs that work on a
            // background thread and reports back through a single
            // pyBridge.taskResult(requestId, json) signal (see
            // ui/bridge.py) -- this map + helper reconnects each
            // request's callback to that shared signal so every other
            // call site below still just does `fn(args..., cb)`.
            const pendingTasks = {};
            let taskCounter = 0;

            pyBridge.taskResult.connect((requestId, raw) => {
                const cb = pendingTasks[requestId];
                if (!cb) return; // already handled, or belongs to a stale/unknown request
                delete pendingTasks[requestId];
                parseJson(raw, cb);
            });

            function callAsync(pyMethodName, args, cb) {
                const requestId = `t${++taskCounter}`;
                pendingTasks[requestId] = cb;
                pyBridge[pyMethodName](requestId, ...args);
            }

            window.pixelforge = {
                bridge: pyBridge,
                ready: whenReady,
                ping: () => pyBridge.ping(),
                getAppVersion: (cb) => pyBridge.getAppVersion(cb),
                getSystemInfo: (cb) => pyBridge.getSystemInfo(cb),
                openImageDialog: (cb) => pyBridge.openImageDialog(cb),
                openImagesDialog: (cb) => pyBridge.openImagesDialog(cb),
                openPathExternally: (path, cb) => pyBridge.openPathExternally(path, cb || (() => {})),
                chooseOutputFolder: (cb) => pyBridge.chooseOutputFolder(cb),
                getSetting: (key, cb) => pyBridge.getSetting(key, cb),
                setSetting: (key, value) => pyBridge.setSetting(key, value),
                getAllSettings: (cb) => pyBridge.getAllSettings((raw) => parseJson(raw, cb)),
                resetAllSettings: (cb) => { pyBridge.resetAllSettings(); if (cb) cb(); },
                clearCache: (cb) => pyBridge.clearCache((raw) => parseJson(raw, cb)),
                // PHASE 14: lightweight live CPU/RAM/disk snapshot for the
                // Settings > System Health gauges (see ui/bridge.py's
                // getSystemSnapshot Slot). Plain (non-threaded) Slot on the
                // Python side -- psutil's instantaneous readings are cheap,
                // unlike getSystemHealth's model/FFmpeg checks -- so no
                // callAsync/taskResult plumbing needed here.
                getSystemSnapshot: (cb) => pyBridge.getSystemSnapshot((raw) => parseJson(raw, cb)),

                // ----- Phase 2: Image Editor -----
                getImageInfo: (path, cb) => pyBridge.getImageInfo(path, (raw) => parseJson(raw, cb)),
                chooseSaveImagePath: (suggestedName, cb) => pyBridge.chooseSaveImagePath(suggestedName, cb),
                // BUGFIX: exportImage used to be a plain (blocking) Slot --
                // full-resolution processing on the GUI thread froze the
                // whole window for the duration (same class of bug as
                // Remove BG / Object Erase, see taskResult's docstring in
                // ui/bridge.py). Now routed through callAsync like every
                // other heavy operation; the public signature here is
                // unchanged so editor.js doesn't need to know the difference.
                exportImage: (sourcePath, destPath, adjustments, cb) =>
                    callAsync("exportImageAsync", [sourcePath, destPath, JSON.stringify(adjustments)], cb),
                previewAdjust: (sourcePath, adjustments, cb) =>
                    pyBridge.previewAdjust(sourcePath, JSON.stringify(adjustments), (raw) => parseJson(raw, cb)),
                autoEnhanceAnalyze: (path, cb) => pyBridge.autoEnhanceAnalyze(path, (raw) => parseJson(raw, cb)),
                // PHASE 3: Auto White Balance (dedicated button) -- returns
                // {ok, temperature, tint, summary}, see ui/bridge.py::autoWhiteBalance.
                autoWhiteBalance: (path, cb) => pyBridge.autoWhiteBalance(path, (raw) => parseJson(raw, cb)),

                // ----- Phase 3: Straighten / Crop (fast, not threaded --
                // same as previewAdjust/exportImage above; no model load) -----
                straightenImage: (sourcePath, angle, cb) =>
                    pyBridge.straightenImage(sourcePath, angle, (raw) => parseJson(raw, cb)),
                cropImage: (sourcePath, cropBox, cb) =>
                    pyBridge.cropImage(sourcePath, JSON.stringify(cropBox), (raw) => parseJson(raw, cb)),
                // Missing-feature #1 (this session): Rotate 90 / Flip H / Flip V.
                // `op` is one of "rotate90"/"rotate180"/"rotate270"/"flip_h"/"flip_v"
                // -- see core/crop.py::transpose_image.
                transposeImage: (sourcePath, op, cb) =>
                    pyBridge.transposeImage(sourcePath, op, (raw) => parseJson(raw, cb)),

                // ----- Phase 4: Presets / Filters -----
                // (listing/preview are fast, not threaded -- exportFilterImage
                // below is the one that's threaded, see its own comment)
                listFilterPresets: (cb) => pyBridge.listFilterPresets((raw) => parseJson(raw, cb)),
                previewFilter: (sourcePath, presetId, intensity, cb) =>
                    pyBridge.previewFilter(sourcePath, presetId, intensity, (raw) => parseJson(raw, cb)),
                // BUGFIX: same freeze bug as exportImage above -- full-res
                // preset export now runs off the GUI thread.
                exportFilterImage: (sourcePath, destPath, presetId, intensity, cb) =>
                    callAsync("exportFilterImageAsync", [sourcePath, destPath, presetId, intensity], cb),
                saveCustomPreset: (name, preset, cb) =>
                    pyBridge.saveCustomPreset(name, JSON.stringify(preset), (raw) => parseJson(raw, cb)),
                renameCustomPreset: (presetId, newName, cb) =>
                    pyBridge.renameCustomPreset(presetId, newName, (raw) => parseJson(raw, cb)),
                duplicateCustomPreset: (presetId, cb) =>
                    pyBridge.duplicateCustomPreset(presetId, (raw) => parseJson(raw, cb)),
                deleteCustomPreset: (presetId, cb) =>
                    pyBridge.deleteCustomPreset(presetId, (raw) => parseJson(raw, cb)),
                setPresetFavorite: (presetId, favorite, cb) =>
                    pyBridge.setPresetFavorite(presetId, favorite, (raw) => parseJson(raw, cb)),
                exportPresetToFile: (presetId, suggestedName, cb) =>
                    pyBridge.exportPresetToFile(presetId, suggestedName, (raw) => parseJson(raw, cb)),
                importPresetFromFile: (cb) => pyBridge.importPresetFromFile((raw) => parseJson(raw, cb)),

                // ----- PHASE 4: Auto Filter -----
                autoFilterSuggest: (sourcePath, cb) =>
                    pyBridge.autoFilterSuggest(sourcePath, (raw) => parseJson(raw, cb)),

                // ----- PHASE 4: Filter Stacking -----
                previewPresetStack: (sourcePath, layers, cb) =>
                    pyBridge.previewPresetStack(sourcePath, JSON.stringify(layers), (raw) => parseJson(raw, cb)),
                exportPresetStack: (sourcePath, destPath, layers, cb) =>
                    callAsync("exportPresetStackAsync", [sourcePath, destPath, JSON.stringify(layers)], cb),

                // ----- PHASE 4: Filter Comparison -----
                previewFilterBatch: (sourcePath, presetIds, intensity, cb) =>
                    pyBridge.previewFilterBatch(sourcePath, JSON.stringify(presetIds), intensity, (raw) => parseJson(raw, cb)),

                // ----- PHASE 4: Custom filter live preview (banate waqt) -----
                previewCustomFilter: (sourcePath, filterSpec, cb) =>
                    pyBridge.previewCustomFilter(sourcePath, JSON.stringify(filterSpec), (raw) => parseJson(raw, cb)),

                // ----- Phase 2A: Remove BG / Object Remover -----
                // (async/threaded on the Python side now -- see callAsync above)
                removeBackgroundPreview: (sourcePath, options, cb) =>
                    callAsync("removeBackgroundPreviewAsync", [sourcePath, JSON.stringify(options)], cb),
                exportRemoveBackground: (sourcePath, destPath, options, cb) =>
                    callAsync(
                        "exportRemoveBackgroundAsync",
                        [sourcePath, destPath, JSON.stringify(options)],
                        cb
                    ),
                chooseBackgroundImage: (cb) => pyBridge.chooseBackgroundImage(cb),

                // ----- Phase 5: Analyzer -----
                // (async/threaded on the Python side -- see callAsync above;
                // face detection + color clustering is real, if modest, work)
                analyzeImage: (sourcePath, cb) => callAsync("analyzeImageAsync", [sourcePath], cb),

                // ----- Phase 6: Smart Pipeline -----
                // (async/threaded -- runs Analyzer internally, see
                // ui/bridge.py::smartPipelineAsync and core/smart_pipeline.py)
                smartPipeline: (sourcePath, cb) => callAsync("smartPipelineAsync", [sourcePath], cb),
                // v2: per-step on/off + intensity overrides, live preview,
                // and a direct Apply that commits onto the shared session
                // (see ui/bridge.py's "PHASE 6: SMART PIPELINE" section).
                smartPipelineCustomize: (lookId, overrides, cb) =>
                    callAsync("smartPipelineCustomizeAsync", [lookId || "", JSON.stringify(overrides || {})], cb),
                smartPipelinePreview: (lookId, overrides, cb) =>
                    callAsync("smartPipelinePreviewAsync", [lookId || "", JSON.stringify(overrides || {})], cb),
                smartPipelineApply: (lookId, overrides, cb) =>
                    callAsync("smartPipelineApplyAsync", [lookId || "", JSON.stringify(overrides || {})], cb),
                listPipelineLooks: (cb) => pyBridge.listPipelineLooks((raw) => parseJson(raw, cb)),
                eraseObjectPreview: (sourcePath, maskDataUrl, method, cb) =>
                    callAsync("eraseObjectPreviewAsync", [sourcePath, maskDataUrl, method], cb),
                exportEraseObject: (sourcePath, destPath, maskDataUrl, method, cb) =>
                    callAsync(
                        "exportEraseObjectAsync",
                        [sourcePath, destPath, maskDataUrl, method],
                        cb
                    ),

                // percent: 0-100, or -1 for "in progress, unknown length"
                onProgress: (cb) => progressListeners.push(cb),
                onStatus: (cb) => statusListeners.push(cb),

                // ----- PHASE 6: Shared Working Image / Edit Session -----
                // (fast, not threaded -- these only touch the session's
                // small JSON state + PNG copies, no model load; see
                // ui/bridge.py's "SESSION" section). Every call resolves
                // with the same state shape as session.state(), so
                // callers can just re-render off the result.
                session: {
                    state: (cb) => pyBridge.sessionState((raw) => parseJson(raw, cb)),
                    open: (originalPath, cb) => pyBridge.sessionOpen(originalPath, (raw) => parseJson(raw, cb)),
                    close: (cb) => pyBridge.sessionClose((raw) => parseJson(raw, cb)),
                    undo: (cb) => pyBridge.sessionUndo((raw) => parseJson(raw, cb)),
                    redo: (cb) => pyBridge.sessionRedo((raw) => parseJson(raw, cb)),
                    jumpTo: (entryId, cb) => pyBridge.sessionJumpTo(entryId, (raw) => parseJson(raw, cb)),
                    revertToOriginal: (cb) => pyBridge.sessionRevertToOriginal((raw) => parseJson(raw, cb)),
                    setToolState: (tool, state, cb) =>
                        pyBridge.sessionSetToolState(tool, JSON.stringify(state || {}), (raw) =>
                            parseJson(raw, cb)
                        ),
                    getToolState: (tool, cb) => pyBridge.sessionGetToolState(tool, (raw) => parseJson(raw, cb)),

                    // Commit a tool's result as the new Working Image.
                    // BUGFIX: applyAdjustments/applyPreset (Enhance's and
                    // Filters' "Apply" buttons) used to be plain Slots that
                    // ran full-resolution processing on the GUI thread --
                    // same freeze bug as exportImage/exportFilterImage
                    // above. Now threaded; signature unchanged for callers.
                    applyAdjustments: (adjustments, label, cb) =>
                        callAsync("sessionApplyAdjustmentsAsync", [JSON.stringify(adjustments || {}), label || ""], cb),
                    applyPreset: (presetId, intensity, label, cb) =>
                        callAsync("sessionApplyPresetAsync", [presetId, intensity, label || ""], cb),
                    commitFile: (resultPath, tool, label, cb) =>
                        pyBridge.sessionCommitFile(resultPath, tool, label || "", (raw) => parseJson(raw, cb)),

                    // Ctrl+S / Ctrl+Shift+S support.
                    chooseProjectSavePath: (cb) => pyBridge.sessionChooseProjectSavePath(cb),
                    chooseProjectOpenPath: (cb) => pyBridge.sessionChooseProjectOpenPath(cb),
                    saveProject: (destPath, cb) => pyBridge.sessionSaveProject(destPath || "", (raw) => parseJson(raw, cb)),
                    loadProject: (path, cb) => pyBridge.sessionLoadProject(path, (raw) => parseJson(raw, cb)),
                },

                // ----- PHASE 7: Prompt Engine (local, rule-based) -----
                // Fast/synchronous -- pure string parsing, no model load
                // (see ui/bridge.py's "PHASE 7: PROMPT ENGINE" section).
                promptEnginePreview: (prompt, negativePrompt, cb) =>
                    pyBridge.promptEnginePreview(prompt || "", negativePrompt || "", (raw) => parseJson(raw, cb)),
                promptHistoryList: (cb) => pyBridge.promptHistoryList((raw) => parseJson(raw, cb)),
                promptHistoryDelete: (entryId, cb) => pyBridge.promptHistoryDelete(entryId, (raw) => parseJson(raw, cb)),
                promptHistoryClear: (cb) => pyBridge.promptHistoryClear((raw) => parseJson(raw, cb)),

                // ----- PHASE 7: AI Image Generation -----
                // Threaded on the Python side (model load + inference can
                // take seconds to minutes on CPU) -- routed through
                // callAsync like Remove BG / Analyzer / Smart Pipeline
                // above, see ui/bridge.py's "PHASE 7: AI IMAGE GENERATION"
                // section for the taskResult contract.
                aiImageGenAvailability: (cb) => pyBridge.aiImageGenAvailability((raw) => parseJson(raw, cb)),
                generateImage: (params, cb) => callAsync("generateImageAsync", [JSON.stringify(params || {})], cb),
                regenerateImage: (previousSpec, cb) =>
                    callAsync("regenerateImageAsync", [JSON.stringify(previousSpec || {})], cb),
                sendGeneratedImageToEditor: (generatedPath, cb) =>
                    pyBridge.sendGeneratedImageToEditor(generatedPath, (raw) => parseJson(raw, cb)),

                // ----- PHASE 8: Batch Processing -----
                // 🧮 Not AI -- orchestrates Phase 5/6/4 per image (see
                // core/batch_processor.py). batchStart is routed through
                // callAsync for its FINAL result (same taskResult
                // contract as generateImage/analyzeImage above), but the
                // live per-item + overall-progress updates while it runs
                // arrive through the two dedicated listeners registered
                // below (onBatchItem/onBatchProgress), not through cb.
                batch: {
                    state: (cb) => pyBridge.batchState((raw) => parseJson(raw, cb)),
                    chooseInputFolder: (cb) => pyBridge.chooseBatchInputFolder(cb),
                    addFiles: (paths, cb) => pyBridge.batchAddFiles(JSON.stringify(paths || []), (raw) => parseJson(raw, cb)),
                    // PHASE 14: crash recovery (see core/batch_processor.py).
                    getRecoveryState: (cb) => pyBridge.getBatchRecoveryState((raw) => parseJson(raw, cb)),
                    discardRecoveryState: () => pyBridge.discardBatchRecoveryState(),
                    addFolder: (folder, recursive, cb) =>
                        pyBridge.batchAddFolder(folder, !!recursive, (raw) => parseJson(raw, cb)),
                    clear: (cb) => pyBridge.batchClear((raw) => parseJson(raw, cb)),
                    removeItem: (itemId, cb) => pyBridge.batchRemoveItem(itemId, (raw) => parseJson(raw, cb)),
                    setIncluded: (itemId, included, cb) =>
                        pyBridge.batchSetIncluded(itemId, !!included, (raw) => parseJson(raw, cb)),
                    reorder: (itemIds, cb) => pyBridge.batchReorder(JSON.stringify(itemIds || []), (raw) => parseJson(raw, cb)),
                    bumpPriority: (itemId, direction, cb) =>
                        pyBridge.batchBumpPriority(itemId, direction, (raw) => parseJson(raw, cb)),
                    setOptions: (options, cb) => pyBridge.batchSetOptions(JSON.stringify(options || {}), (raw) => parseJson(raw, cb)),
                    detectDuplicates: (cb) => pyBridge.batchDetectDuplicates((raw) => parseJson(raw, cb)),
                    checkDiskSpace: (cb) => pyBridge.batchCheckDiskSpace((raw) => parseJson(raw, cb)),
                    start: (options, cb) => callAsync("batchStartAsync", [JSON.stringify(options || {})], cb),
                    scheduleStart: (delaySeconds, cb) => callAsync("batchScheduleStartAsync", [delaySeconds | 0], cb),
                    pause: () => pyBridge.batchPause(),
                    resume: () => pyBridge.batchResume(),
                    cancel: () => pyBridge.batchCancel(),
                    skipItem: (itemId, cb) => pyBridge.batchSkipItem(itemId, (raw) => parseJson(raw, cb)),
                    retryItem: (itemId, cb) => pyBridge.batchRetryItem(itemId, (raw) => parseJson(raw, cb)),
                    retryFailed: (cb) => pyBridge.batchRetryFailed((raw) => parseJson(raw, cb)),
                    requeueAll: (cb) => pyBridge.batchRequeueAll((raw) => parseJson(raw, cb)),
                    commitPreviewed: (itemIds, cb) => pyBridge.batchCommitPreviewed(itemIds ? JSON.stringify(itemIds) : "null", (raw) => parseJson(raw, cb)),
                    discardPreviewed: (itemIds, cb) => pyBridge.batchDiscardPreviewed(itemIds ? JSON.stringify(itemIds) : "null", (raw) => parseJson(raw, cb)),
                    exportLog: (cb) => pyBridge.batchExportLogDialog((raw) => parseJson(raw, cb)),
                    onItem: (cb) => pyBridge.batchItemChanged.connect((raw) => parseJson(raw, cb)),
                    onProgress: (cb) => pyBridge.batchProgressChanged.connect((pct, label, eta) => cb(pct, label, eta)),
                    onFinished: (cb) => pyBridge.batchFinished.connect((raw) => parseJson(raw, cb)),
                },

                // ----- PHASE 9: AI Upscaling -----
                // 🤖 Real-ESRGAN via ONNX Runtime (see ai/upscaler.py's
                // header for the full design). Preview/Export are
                // threaded through callAsync like Remove BG/Filters
                // above -- a full tiled AI pass is genuinely heavy, even
                // on the preview's downsized source. Progress/status
                // during a run (including the one-time model download)
                // arrive through the same shared onProgress/onStatus
                // listeners every other heavy operation already uses,
                // not a dedicated per-view channel.
                upscale: {
                    modelStatus: (cb) => pyBridge.getUpscaleModelStatus((raw) => parseJson(raw, cb)),
                    clearModelCache: (cb) => pyBridge.clearUpscaleModelCache((raw) => parseJson(raw, cb)),
                    safetyCheck: (sourcePath, options, cb) =>
                        pyBridge.upscaleSafetyCheck(sourcePath, JSON.stringify(options || {}), (raw) => parseJson(raw, cb)),
                    preview: (sourcePath, options, cb) =>
                        callAsync("upscalePreviewAsync", [sourcePath, JSON.stringify(options || {})], cb),
                    export: (sourcePath, destPath, options, cb) =>
                        callAsync("exportUpscaleImageAsync", [sourcePath, destPath, JSON.stringify(options || {})], cb),
                    cancel: () => pyBridge.cancelUpscale(),
                    // "Apply" -- commits directly onto the shared session
                    // Working Image (ui/bridge.py::sessionApplyUpscaleAsync),
                    // same shape as session.applyPreset/applyAdjustments.
                    apply: (options, label, cb) =>
                        callAsync("sessionApplyUpscaleAsync", [JSON.stringify(options || {}), label || ""], cb),
                },

                // ----- PHASE 10: Face Restoration -----
                // 🤖 GFPGAN via ONNX Runtime (see ai/face_restorer.py's
                // header for the full design). detectFaces/preview/
                // export are threaded through callAsync like Upscale
                // above -- a GFPGAN forward pass per face is genuinely
                // heavy on CPU-only hardware. Progress/status during a
                // run (including the one-time ~340MB model download)
                // arrive through the same shared onProgress/onStatus
                // listeners every other heavy operation already uses.
                faceRestore: {
                    modelStatus: (cb) => pyBridge.getFaceRestoreModelStatus((raw) => parseJson(raw, cb)),
                    clearModelCache: (cb) => pyBridge.clearFaceRestoreModelCache((raw) => parseJson(raw, cb)),
                    // "Automatic face detection" -- returns
                    // [{"id","x","y","w","h"}, ...] for the view's
                    // "Detected Faces" checklist.
                    detectFaces: (sourcePath, cb) =>
                        callAsync("detectFacesAsync", [sourcePath], cb),
                    preview: (sourcePath, faces, options, cb) =>
                        callAsync("restorePreviewAsync", [sourcePath, JSON.stringify(faces || []), JSON.stringify(options || {})], cb),
                    export: (sourcePath, destPath, faces, options, cb) =>
                        callAsync("exportFaceRestoreImageAsync", [sourcePath, destPath, JSON.stringify(faces || []), JSON.stringify(options || {})], cb),
                    cancel: () => pyBridge.cancelFaceRestore(),
                    // "Apply" -- commits directly onto the shared session
                    // Working Image (ui/bridge.py::sessionApplyFaceRestoreAsync),
                    // same shape as upscale.apply above.
                    apply: (faces, options, label, cb) =>
                        callAsync("sessionApplyFaceRestoreAsync", [JSON.stringify(faces || []), JSON.stringify(options || {}), label || ""], cb),
                },

                // ----- PHASE 11: Video Studio -----
                // 🧮 Not AI -- deterministic FFmpeg assembly (motion/
                // transition math), see core/video_studio.py's header
                // for the full filter-graph design. Every mutating call
                // returns the controller's FULL state (same "server
                // owns the truth" convention as batch/session above),
                // so frontend/video.js just re-renders off the result.
                // Export/Preview are threaded through callAsync; live
                // percent/label during either arrive through the same
                // shared onProgress/onStatus listeners every other
                // heavy operation already uses -- real H.264 encoding
                // is unavoidably heavy, same class of work as AI
                // Upscale/Face Restoration above even though this phase
                // has no model of its own.
                video: {
                    ffmpegStatus: (cb) => pyBridge.getFfmpegStatus((raw) => parseJson(raw, cb)),
                    state: (cb) => pyBridge.videoState((raw) => parseJson(raw, cb)),
                    newProject: (cb) => pyBridge.videoNewProject((raw) => parseJson(raw, cb)),
                    addImages: (paths, cb) => pyBridge.videoAddImages(JSON.stringify(paths || []), (raw) => parseJson(raw, cb)),
                    removeClip: (clipId, cb) => pyBridge.videoRemoveClip(clipId, (raw) => parseJson(raw, cb)),
                    reorder: (clipIds, cb) => pyBridge.videoReorderClips(JSON.stringify(clipIds || []), (raw) => parseJson(raw, cb)),
                    setClipSettings: (clipId, patch, cb) =>
                        pyBridge.videoSetClipSettings(clipId, JSON.stringify(patch || {}), (raw) => parseJson(raw, cb)),
                    setProjectSettings: (patch, cb) =>
                        pyBridge.videoSetProjectSettings(JSON.stringify(patch || {}), (raw) => parseJson(raw, cb)),
                    chooseAudio: (cb) => pyBridge.chooseVideoAudioDialog(cb),
                    setAudio: (path, volume, fadeIn, fadeOut, durationMatch, cb) =>
                        pyBridge.videoSetAudio(path || "", volume | 0, +fadeIn || 0, +fadeOut || 0, durationMatch || "none", (raw) => parseJson(raw, cb)),
                    clearAudio: (cb) => pyBridge.videoClearAudio((raw) => parseJson(raw, cb)),
                    // PHASE 12: second, limited-support audio track
                    // (voice-over) mixed underneath the primary music.
                    chooseAudio2: (cb) => pyBridge.chooseVideoAudio2Dialog(cb),
                    setAudio2: (path, volume, cb) =>
                        pyBridge.videoSetAudio2(path || "", volume | 0, (raw) => parseJson(raw, cb)),
                    clearAudio2: (cb) => pyBridge.videoClearAudio2((raw) => parseJson(raw, cb)),
                    // PHASE 12: watermark / logo (persistent, whole-video overlay).
                    chooseWatermarkImage: (cb) => pyBridge.chooseVideoWatermarkImageDialog(cb),
                    setWatermark: (patch, cb) =>
                        pyBridge.videoSetWatermark(JSON.stringify(patch || {}), (raw) => parseJson(raw, cb)),
                    clearWatermark: (cb) => pyBridge.videoClearWatermark((raw) => parseJson(raw, cb)),
                    addTextOverlay: (overlay, cb) => pyBridge.videoAddTextOverlay(JSON.stringify(overlay || {}), (raw) => parseJson(raw, cb)),
                    updateTextOverlay: (overlayId, patch, cb) =>
                        pyBridge.videoUpdateTextOverlay(overlayId, JSON.stringify(patch || {}), (raw) => parseJson(raw, cb)),
                    removeTextOverlay: (overlayId, cb) => pyBridge.videoRemoveTextOverlay(overlayId, (raw) => parseJson(raw, cb)),
                    // PHASE 12: SRT subtitle import -- turns each cue into
                    // a caption-style text overlay in one call.
                    chooseSrtFile: (cb) => pyBridge.chooseSrtFileDialog(cb),
                    importSrt: (path, cb) => pyBridge.videoImportSrt(path, (raw) => parseJson(raw, cb)),
                    undo: (cb) => pyBridge.videoUndo((raw) => parseJson(raw, cb)),
                    redo: (cb) => pyBridge.videoRedo((raw) => parseJson(raw, cb)),
                    chooseProjectSavePath: (cb) => pyBridge.chooseVideoProjectSavePath(cb),
                    chooseProjectOpenPath: (cb) => pyBridge.chooseVideoProjectOpenPath(cb),
                    saveProject: (path, cb) => pyBridge.videoSaveProject(path, (raw) => parseJson(raw, cb)),
                    loadProject: (path, cb) => pyBridge.videoLoadProject(path, (raw) => parseJson(raw, cb)),
                    chooseExportPath: (cb) => pyBridge.chooseVideoExportPath(cb),
                    export: (destPath, cb) => callAsync("videoExportAsync", [destPath], cb),
                    generatePreview: (container, cb) => callAsync("videoGeneratePreviewAsync", [container], cb),
                    cancel: () => pyBridge.cancelVideoExport(),
                },

                // ----- Phase 13: Projects / History (SQLite) -----
                // Every Slot here returns {ok, data|error} (see
                // Bridge._db_result in ui/bridge.py) so all callbacks get
                // that shape directly, already parsed.
                projects: {
                    create: (name, cb) => pyBridge.createProject(name, (raw) => parseJson(raw, cb)),
                    get: (projectId, cb) => pyBridge.getProject(projectId, (raw) => parseJson(raw, cb)),
                    rename: (projectId, newName, cb) =>
                        pyBridge.renameProject(projectId, newName, (raw) => parseJson(raw, cb)),
                    duplicate: (projectId, cb) => pyBridge.duplicateProject(projectId, (raw) => parseJson(raw, cb)),
                    // Permanent -- the frontend must confirm with the user first.
                    deleteForever: (projectId, cb) => pyBridge.deleteProject(projectId, (raw) => parseJson(raw, cb)),
                    trash: (projectId, cb) => pyBridge.trashProject(projectId, (raw) => parseJson(raw, cb)),
                    restore: (projectId, cb) => pyBridge.restoreProject(projectId, (raw) => parseJson(raw, cb)),
                    setThumbnail: (projectId, thumbnailPath, cb) =>
                        pyBridge.setProjectThumbnail(projectId, thumbnailPath, (raw) => parseJson(raw, cb)),
                    // status: "active" | "trashed" | "" (empty string = all statuses)
                    list: (status, search, sortBy, sortDir, limit, cb) =>
                        pyBridge.listProjects(status || "", search || "", sortBy || "updated_at", sortDir || "desc", limit || 0, (raw) => parseJson(raw, cb)),
                    recent: (limit, cb) => pyBridge.recentProjects(limit || 8, (raw) => parseJson(raw, cb)),

                    addMedia: (projectId, inputPath, outputPath, mediaType, width, height, cb) =>
                        pyBridge.addMedia(projectId, inputPath, outputPath || "", mediaType || "image", width || 0, height || 0, (raw) => parseJson(raw, cb)),
                    listMedia: (projectId, cb) => pyBridge.listMedia(projectId, (raw) => parseJson(raw, cb)),
                    deleteMedia: (mediaId, cb) => pyBridge.deleteMedia(mediaId, (raw) => parseJson(raw, cb)),

                    addEdit: (mediaId, preset, prompt, parameters, cb) =>
                        pyBridge.addEdit(mediaId, preset || "", prompt || "", parameters ? JSON.stringify(parameters) : "", (raw) => parseJson(raw, cb)),
                    listEdits: (mediaId, cb) => pyBridge.listEdits(mediaId, (raw) => parseJson(raw, cb)),
                    // Per-project edit history -- distinct from the app-wide
                    // Undo/Redo stack in core/session.py; this is a durable,
                    // browsable log, not an in-memory stack.
                    editHistory: (projectId, cb) => pyBridge.projectEditHistory(projectId, (raw) => parseJson(raw, cb)),

                    addVideoProject: (projectId, duration, resolution, fps, outputPath, cb) =>
                        pyBridge.addVideoProject(projectId, duration || 0, resolution || "", fps || 0, outputPath || "", (raw) => parseJson(raw, cb)),
                    listVideoProjects: (projectId, cb) => pyBridge.listVideoProjects(projectId, (raw) => parseJson(raw, cb)),

                    savePreset: (name, category, parameters, cb) =>
                        pyBridge.savePreset(name, category || "", parameters ? JSON.stringify(parameters) : "", (raw) => parseJson(raw, cb)),
                    listPresets: (category, cb) => pyBridge.listPresets(category || "", (raw) => parseJson(raw, cb)),
                    deletePreset: (presetId, cb) => pyBridge.deletePreset(presetId, (raw) => parseJson(raw, cb)),

                    // Backup export/import are threaded (see ui/bridge.py) --
                    // both open a native file dialog first, then do the
                    // actual read/write off the GUI thread.
                    exportBackup: (projectId, cb) => callAsync("exportProjectBackup", [projectId], cb),
                    importBackup: (cb) => callAsync("importProjectBackup", [], cb),

                    // Crash recovery -- call setActiveProject(id) whenever a
                    // project becomes the one currently open, and
                    // getActiveProject on launch to offer reopening it if the
                    // app didn't shut down cleanly last time (see
                    // MainWindow.closeEvent in ui/main_window.py).
                    setActive: (projectId) => pyBridge.setActiveProject(projectId),
                    getActive: (cb) => pyBridge.getActiveProject((raw) => parseJson(raw, cb)),
                    clearActive: () => pyBridge.clearActiveProject(),
                },
            };

            bridgeReady = true;
            setBridgeStatus("ok");
            readyCallbacks.forEach((cb) => cb());
            readyCallbacks.length = 0;
        });
    }

    document.addEventListener("DOMContentLoaded", init);
})();