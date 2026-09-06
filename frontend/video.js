// frontend/video.js
//
// PHASE 11 -- Video Studio (frontend).
//
// 🧮 Not AI -- this view is a thin renderer over the single shared
// VideoProject controller that lives in core/video_studio.py (exposed
// here as window.pixelforge.video, see frontend/bridge.js). Same
// "server owns the truth, client just re-renders off the returned
// state" rule as batch.js/session.js -- every call below that mutates
// anything (add/remove/reorder/settings/undo/redo/...) gets the FULL
// current state back and re-renders from it, so this view can never
// drift out of sync with what core/video_studio.py actually holds.
//
// Export/Preview progress arrives through the shared global status bar
// (window.pixelforge.onProgress/onStatus, already wired once in ui.js)
// -- no dedicated per-view progress channel needed, same as Upscale/
// Face Restoration.

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("video-dropzone");
    const workspace = document.getElementById("video-workspace");
    if (!dropzone || !workspace) return; // view not present in this build

    const videoError = document.getElementById("video-error");
    const ffmpegMissing = document.getElementById("video-ffmpeg-missing");
    const ffmpegBadge = document.getElementById("video-ffmpeg-badge");

    const btnAddImagesEmpty = document.getElementById("btn-video-add-images-empty");
    const btnAddImages = document.getElementById("btn-video-add-images");
    const btnUndo = document.getElementById("btn-video-undo");
    const btnRedo = document.getElementById("btn-video-redo");
    const btnNew = document.getElementById("btn-video-new");
    const btnLoadProject = document.getElementById("btn-video-load-project");
    const btnSaveProject = document.getElementById("btn-video-save-project");

    const clipCountEl = document.getElementById("video-clip-count");
    const totalDurationEl = document.getElementById("video-total-duration");
    const outputDimsEl = document.getElementById("video-output-dims");
    const timelineStrip = document.getElementById("video-timeline-strip");

    const previewFrame = document.getElementById("video-preview-frame");
    const previewPlayer = document.getElementById("video-preview-player");
    const previewEmpty = document.getElementById("video-preview-empty");
    const previewPlaybackError = document.getElementById("video-preview-playback-error");
    const btnOpenPreviewExternally = document.getElementById("btn-video-open-preview-externally");
    const btnGeneratePreview = document.getElementById("btn-video-generate-preview");
    const btnMixTransitions = document.getElementById("btn-video-mix-transitions");
    const btnAutoPro = document.getElementById("btn-video-auto-pro");

    // ----- Tabs -----
    const tabButtons = document.querySelectorAll(".video-tab");
    const tabPanels = {
        clip: document.getElementById("video-tab-clip"),
        project: document.getElementById("video-tab-project"),
        audio: document.getElementById("video-tab-audio"),
        text: document.getElementById("video-tab-text"),
        watermark: document.getElementById("video-tab-watermark"),
        export: document.getElementById("video-tab-export"),
    };

    // ----- Clip tab -----
    const clipNoneMsg = document.getElementById("video-clip-none");
    const clipFields = document.getElementById("video-clip-fields");
    const clipDuration = document.getElementById("video-clip-duration");
    const valClipDuration = document.getElementById("val-video-clip-duration");
    const clipMotion = document.getElementById("video-clip-motion");
    const clipIntensity = document.getElementById("video-clip-intensity");
    const clipTransition = document.getElementById("video-clip-transition");
    const clipTransitionDuration = document.getElementById("video-clip-transition-duration");
    const valClipTransitionDuration = document.getElementById("val-video-clip-transition-duration");
    const clipCover = document.getElementById("video-clip-cover");
    const btnRemoveClip = document.getElementById("btn-video-remove-clip");

    // ----- Project tab -----
    const projResolution = document.getElementById("video-proj-resolution");
    const projAspect = document.getElementById("video-proj-aspect");
    const projFps = document.getElementById("video-proj-fps");
    const projDurationMode = document.getElementById("video-proj-duration-mode");
    const projCustomDurationRow = document.getElementById("video-proj-custom-duration-row");
    const projCustomDuration = document.getElementById("video-proj-custom-duration");
    const projFitMode = document.getElementById("video-proj-fit-mode");
    const projBgFillRow = document.getElementById("video-proj-bg-fill-row");
    const projBgFill = document.getElementById("video-proj-bg-fill");
    const projColorGrade = document.getElementById("video-proj-color-grade");
    const projSharpen = document.getElementById("video-proj-sharpen");
    const projCinematicBars = document.getElementById("video-proj-cinematic-bars");
    const projFilmGrain = document.getElementById("video-proj-film-grain");

    // ----- Audio tab -----
    const audioFilename = document.getElementById("video-audio-filename");
    const btnChooseAudio = document.getElementById("btn-video-choose-audio");
    const btnClearAudio = document.getElementById("btn-video-clear-audio");
    const audioVolume = document.getElementById("video-audio-volume");
    const valAudioVolume = document.getElementById("val-video-audio-volume");
    const audioFadeIn = document.getElementById("video-audio-fadein");
    const valAudioFadeIn = document.getElementById("val-video-audio-fadein");
    const audioFadeOut = document.getElementById("video-audio-fadeout");
    const valAudioFadeOut = document.getElementById("val-video-audio-fadeout");
    const audioDurationMatch = document.getElementById("video-audio-duration-match");
    const audioRightsNotice = document.getElementById("video-audio-rights-notice");
    // PHASE 12: second, limited-support audio track (voice-over).
    const audio2Filename = document.getElementById("video-audio2-filename");
    const btnChooseAudio2 = document.getElementById("btn-video-choose-audio2");
    const btnClearAudio2 = document.getElementById("btn-video-clear-audio2");
    const audio2Volume = document.getElementById("video-audio2-volume");
    const valAudio2Volume = document.getElementById("val-video-audio2-volume");

    // ----- Text tab -----
    const textList = document.getElementById("video-text-list");
    const textInput = document.getElementById("video-text-input");
    const textPosition = document.getElementById("video-text-position");
    const textAnimation = document.getElementById("video-text-animation");
    const textSize = document.getElementById("video-text-size");
    const valTextSize = document.getElementById("val-video-text-size");
    const textColor = document.getElementById("video-text-color");
    const textOpacity = document.getElementById("video-text-opacity");
    const valTextOpacity = document.getElementById("val-video-text-opacity");
    const textShadow = document.getElementById("video-text-shadow");
    const textStart = document.getElementById("video-text-start");
    const valTextStart = document.getElementById("val-video-text-start");
    const textDuration = document.getElementById("video-text-duration");
    const valTextDuration = document.getElementById("val-video-text-duration");
    const btnAddText = document.getElementById("btn-video-add-text");
    // PHASE 12: SRT subtitle import.
    const srtFilename = document.getElementById("video-srt-filename");
    const btnImportSrt = document.getElementById("btn-video-import-srt");

    // ----- Watermark tab (PHASE 12) -----
    const watermarkEnabled = document.getElementById("video-watermark-enabled");
    const watermarkType = document.getElementById("video-watermark-type");
    const watermarkTextRow = document.getElementById("video-watermark-text-row");
    const watermarkImageRow = document.getElementById("video-watermark-image-row");
    const watermarkText = document.getElementById("video-watermark-text");
    const watermarkImageFilename = document.getElementById("video-watermark-image-filename");
    const btnChooseWatermarkImage = document.getElementById("btn-video-choose-watermark-image");
    const watermarkPosition = document.getElementById("video-watermark-position");
    const watermarkOpacity = document.getElementById("video-watermark-opacity");
    const valWatermarkOpacity = document.getElementById("val-video-watermark-opacity");
    const watermarkScale = document.getElementById("video-watermark-scale");
    const valWatermarkScale = document.getElementById("val-video-watermark-scale");
    const btnApplyWatermark = document.getElementById("btn-video-apply-watermark");
    const btnClearWatermark = document.getElementById("btn-video-clear-watermark");
    let watermarkImagePath = null;

    // ----- Export tab -----
    const exportQuality = document.getElementById("video-export-quality");
    const exportBitrateRow = document.getElementById("video-export-bitrate-row");
    const exportBitrate = document.getElementById("video-export-bitrate");
    const btnExport = document.getElementById("btn-video-export");
    const btnCancelExport = document.getElementById("btn-video-cancel-export");
    const exportResult = document.getElementById("video-export-result");
    const btnOpenExport = document.getElementById("btn-video-open-export");

    let lastState = { clips: [], settings: {}, audio: {}, audio2: {}, watermark: {}, text_overlays: [], can_undo: false, can_redo: false };
    let selectedClipId = null;
    let dragSrcId = null;
    let busy = false; // export or preview currently running
    let lastPreviewPath = null;
    let lastExportPath = null;

    // ===================== HELPERS =====================

    function fileUrl(path) {
        // Same convention as batch.js/restore.js's toFileUrl.
        if (!path) return "";
        const normalized = path.replace(/\\/g, "/").replace(/^\/+/, "");
        return "file:///" + encodeURI(normalized);
    }

    function showError(msg) {
        if (!videoError) return;
        videoError.classList.remove("is-success");
        videoError.textContent = msg;
        videoError.classList.remove("view--hidden");
        setTimeout(() => videoError.classList.add("view--hidden"), 6000);
    }

    function showNotice(msg) {
        // Same banner as showError but styled green -- used for
        // confirmations (e.g. "Mix Transitions" applied) where there's
        // no other visible feedback once a clip is deselected, since
        // the embedded preview can't be trusted to reflect it on every
        // machine (see the preview-playback fallback below).
        if (!videoError) return;
        videoError.classList.add("is-success");
        videoError.textContent = msg;
        videoError.classList.remove("view--hidden");
        setTimeout(() => {
            videoError.classList.add("view--hidden");
            videoError.classList.remove("is-success");
        }, 4000);
    }

    function formatSeconds(s) {
        return `${Number(s || 0).toFixed(1)}s`;
    }

    const MOTION_LABELS = {
        none: "Static", ken_burns: "Ken Burns", zoom_in: "Zoom In", zoom_out: "Zoom Out",
        pan_left: "Pan Left", pan_right: "Pan Right", pan_vertical: "V. Pan",
        slow_drift: "Slow Drift", tilt_3d: "3D Tilt",
    };

    // ===================== TOP-LEVEL RENDER =====================

    function applyState(state) {
        if (!state || state.ok === false) {
            if (state && state.error) showError(state.error);
            return;
        }
        lastState = state;

        const hasClips = state.clips && state.clips.length > 0;
        dropzone.classList.toggle("view--hidden", hasClips);
        workspace.classList.toggle("view--hidden", !hasClips);

        clipCountEl.textContent = String(state.clip_count || 0);
        totalDurationEl.textContent = formatSeconds(state.total_duration);
        outputDimsEl.textContent = state.output_width ? `${state.output_width}×${state.output_height}` : "";

        btnUndo.disabled = !state.can_undo;
        btnRedo.disabled = !state.can_redo;

        renderTimeline(state.clips || []);
        renderProjectTab(state.settings || {});
        renderAudioTab(state.audio || {}, state.audio2 || {}, state.music_rights_notice || null);
        renderTextList(state.text_overlays || []);
        renderWatermarkTab(state.watermark || {});

        // Keep the selected clip's panel in sync (e.g. after undo/redo).
        if (selectedClipId && !state.clips.find((c) => c.id === selectedClipId)) {
            selectedClipId = null;
        }
        renderClipTab();
    }

    // ===================== TIMELINE (drag to reorder) =====================

    function renderClipCard(clip, index) {
        const card = document.createElement("div");
        card.className = "video-clip-card" + (clip.id === selectedClipId ? " is-selected" : "");
        card.dataset.clipId = clip.id;
        card.draggable = true;
        card.innerHTML = `
            <img class="video-clip-thumb" data-motion="${clip.motion}" src="${fileUrl(clip.thumbnail || clip.path)}" alt="">
            <button class="video-clip-remove" title="Remove">✕</button>
            <div class="video-clip-meta">
                <span class="video-clip-index">${index + 1}</span>
                <span>${formatSeconds(clip.duration)}</span>
                ${clip.cover ? '<span class="video-clip-cover-flag" title="Cover frame">★</span>' : ""}
            </div>
        `;
        card.title = `${MOTION_LABELS[clip.motion] || clip.motion} · ${clip.transition !== "none" ? clip.transition : "cut"}`;

        card.addEventListener("click", (e) => {
            if (e.target.closest(".video-clip-remove")) return;
            selectedClipId = clip.id;
            renderTimeline(lastState.clips);
            renderClipTab();
            setActiveTab("clip");
        });
        card.querySelector(".video-clip-remove").addEventListener("click", (e) => {
            e.stopPropagation();
            if (selectedClipId === clip.id) selectedClipId = null;
            window.pixelforge.video.removeClip(clip.id, applyState);
        });

        // ----- True drag-to-reorder, same pattern as batch.js's renderRow -----
        card.addEventListener("dragstart", () => {
            dragSrcId = clip.id;
            card.classList.add("is-dragging");
        });
        card.addEventListener("dragend", () => {
            dragSrcId = null;
            card.classList.remove("is-dragging");
            timelineStrip.querySelectorAll(".video-clip-card").forEach((c) => c.classList.remove("is-drop-target"));
        });
        card.addEventListener("dragover", (e) => {
            if (!dragSrcId || dragSrcId === clip.id) return;
            e.preventDefault();
            card.classList.add("is-drop-target");
        });
        card.addEventListener("dragleave", () => card.classList.remove("is-drop-target"));
        card.addEventListener("drop", (e) => {
            if (!dragSrcId || dragSrcId === clip.id) return;
            e.preventDefault();
            card.classList.remove("is-drop-target");
            const before = e.clientX < card.getBoundingClientRect().left + card.offsetWidth / 2;
            const ids = lastState.clips.map((c) => c.id).filter((id) => id !== dragSrcId);
            const targetIdx = ids.indexOf(clip.id);
            ids.splice(before ? targetIdx : targetIdx + 1, 0, dragSrcId);
            window.pixelforge.video.reorder(ids, applyState);
        });

        return card;
    }

    function renderTimeline(clips) {
        timelineStrip.innerHTML = "";
        clips.forEach((clip, i) => timelineStrip.appendChild(renderClipCard(clip, i)));
    }

    // ===================== CLIP TAB =====================

    function renderClipTab() {
        const clip = lastState.clips.find((c) => c.id === selectedClipId);
        if (!clip) {
            clipNoneMsg.classList.remove("view--hidden");
            clipFields.classList.add("view--hidden");
            return;
        }
        clipNoneMsg.classList.add("view--hidden");
        clipFields.classList.remove("view--hidden");

        clipDuration.value = clip.duration;
        valClipDuration.textContent = formatSeconds(clip.duration);
        clipMotion.value = clip.motion;
        clipIntensity.value = clip.motion_intensity;
        clipTransition.value = clip.transition;
        clipTransitionDuration.value = clip.transition_duration;
        valClipTransitionDuration.textContent = formatSeconds(clip.transition_duration);
        clipCover.checked = !!clip.cover;
    }

    function patchSelectedClip(patch) {
        if (!selectedClipId) return;
        window.pixelforge.video.setClipSettings(selectedClipId, patch, applyState);
    }

    clipDuration.addEventListener("input", () => { valClipDuration.textContent = formatSeconds(clipDuration.value); });
    clipDuration.addEventListener("change", () => patchSelectedClip({ duration: parseFloat(clipDuration.value) }));
    clipMotion.addEventListener("change", () => patchSelectedClip({ motion: clipMotion.value }));
    clipIntensity.addEventListener("change", () => patchSelectedClip({ motion_intensity: clipIntensity.value }));
    clipTransition.addEventListener("change", () => patchSelectedClip({ transition: clipTransition.value }));
    clipTransitionDuration.addEventListener("input", () => { valClipTransitionDuration.textContent = formatSeconds(clipTransitionDuration.value); });
    clipTransitionDuration.addEventListener("change", () => patchSelectedClip({ transition_duration: parseFloat(clipTransitionDuration.value) }));
    clipCover.addEventListener("change", () => { if (clipCover.checked) patchSelectedClip({ cover: true }); });
    btnRemoveClip.addEventListener("click", () => {
        if (!selectedClipId) return;
        const id = selectedClipId;
        selectedClipId = null;
        window.pixelforge.video.removeClip(id, applyState);
    });

    // ===================== MIX TRANSITIONS =====================
    // "Mix Transitions" -- one repeated transition (e.g. every clip set
    // to "Crossfade") is what actually reads as flat/amateur, even
    // though each individual transition looks fine on its own. This
    // picks a fresh, varied rotation of transitions from the palette
    // below and assigns them clip-by-clip (never repeating the same one
    // back-to-back) so the export reads like an edited video rather
    // than one effect stamped on every cut. "none" is deliberately
    // excluded -- this is about varying *how* clips transition, not
    // adding hard cuts.
    const TRANSITION_MIX_POOL = [
        "crossfade", "slide_left", "wipe_right", "zoom",
        "dip_to_black", "circle_open", "slide_right", "wipe_left", "light",
    ];

    function shuffled(arr) {
        const a = arr.slice();
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }

    function applyClipSettingsSequentially(patches, onDone) {
        // Applied one at a time (waiting for each callback) rather than
        // fired all at once -- setClipSettings returns the full project
        // state each call, so overlapping calls could race and clobber
        // each other's result.
        if (!patches.length) { onDone(); return; }
        const [id, patch] = patches[0];
        window.pixelforge.video.setClipSettings(id, patch, (state) => {
            applyState(state);
            applyClipSettingsSequentially(patches.slice(1), onDone);
        });
    }

    btnMixTransitions.addEventListener("click", () => {
        if (busy) return;
        if (lastState.clips.length < 2) {
            showError("Add at least 2 clips before mixing transitions.");
            return;
        }
        // Pick a random set of ~4 distinct transitions per click so
        // repeated use of the button gives a different mix each time,
        // then cycle through them without ever repeating one back-to-back.
        const palette = shuffled(TRANSITION_MIX_POOL).slice(0, Math.min(4, TRANSITION_MIX_POOL.length));
        let lastPicked = null;
        const patches = lastState.clips.map((clip) => {
            let choices = palette.filter((t) => t !== lastPicked);
            if (!choices.length) choices = palette;
            const pick = choices[Math.floor(Math.random() * choices.length)];
            lastPicked = pick;
            return [clip.id, { transition: pick }];
        });
        btnMixTransitions.disabled = true;
        applyClipSettingsSequentially(patches, () => {
            btnMixTransitions.disabled = false;
            // Visible proof this actually ran -- don't rely on the
            // embedded preview to show it, since on some machines it
            // can't play inline at all (unrelated Qt/codec issue).
            // Re-export or open each clip's dropdown to see the values.
            showNotice(`Mixed ${palette.length} transitions across ${patches.length} clips: ${palette.join(", ")}. Re-export (or check each clip's Transition dropdown) to see it.`);
        });
    });

    // ===================== MAKE PROFESSIONAL (one click) =====================
    // Combines everything this session added into a single action:
    // varied transitions (same engine as Mix Transitions), varied
    // motion per clip (so it's not the same Ken Burns on every image),
    // varied pacing (not every clip sitting on screen for the exact
    // same seconds -- constant-duration cuts are what read as
    // "robotic/slideshow" over "edited"), the Cinematic color grade,
    // sharpening, cinematic letterbox bars, and film grain turned on.
    // This is the "I don't want to tune every dropdown per clip, just
    // make it look professional" button.
    const MOTION_MIX_POOL = ["ken_burns", "slow_drift", "tilt_3d", "zoom_in", "pan_left", "pan_right"];
    // Duration variety: most clips sit near the base length, roughly
    // every 3rd clip runs longer (a "hold on this one" beat) -- mimics
    // how real edits aren't metronomic, without needing per-clip manual
    // tuning.
    const DURATION_MIX_FACTORS = [0.85, 1.0, 1.0, 1.4, 0.9, 1.15];

    btnAutoPro.addEventListener("click", () => {
        if (busy) return;
        if (!lastState.clips.length) {
            showError("Add some images first.");
            return;
        }
        btnAutoPro.disabled = true;
        btnMixTransitions.disabled = true;

        // 1) Project-level: cinematic grade + sharpen + letterbox + grain on.
        patchProjectSettings({ color_grade: "cinematic", sharpen: true, cinematic_bars: true, film_grain: true });

        // 2) Per clip: varied transition + varied motion + varied
        // pacing, each never repeating back-to-back, same approach as
        // Mix Transitions.
        const transPalette = shuffled(TRANSITION_MIX_POOL).slice(0, Math.min(4, TRANSITION_MIX_POOL.length));
        const motionPalette = shuffled(MOTION_MIX_POOL).slice(0, Math.min(4, MOTION_MIX_POOL.length));
        const durationFactors = shuffled(DURATION_MIX_FACTORS);
        let lastTrans = null, lastMotion = null;
        const patches = lastState.clips.map((clip, idx) => {
            let tChoices = transPalette.filter((t) => t !== lastTrans);
            if (!tChoices.length) tChoices = transPalette;
            const t = tChoices[Math.floor(Math.random() * tChoices.length)];
            lastTrans = t;

            let mChoices = motionPalette.filter((m) => m !== lastMotion);
            if (!mChoices.length) mChoices = motionPalette;
            const m = mChoices[Math.floor(Math.random() * mChoices.length)];
            lastMotion = m;

            const baseDuration = Number(clip.duration) || 4.0;
            const factor = durationFactors[idx % durationFactors.length];
            const duration = Math.max(1.0, Math.round(baseDuration * factor * 10) / 10);

            return [clip.id, { transition: t, motion: m, duration }];
        });
        applyClipSettingsSequentially(patches, () => {
            btnAutoPro.disabled = false;
            btnMixTransitions.disabled = false;
            showNotice("Applied: cinematic grade, sharpening, letterbox bars, film grain, and a varied transition + motion + pacing mix across every clip. Export to see the result.");
        });
    });

    // ===================== PROJECT TAB =====================

    function renderProjectTab(settings) {
        projResolution.value = settings.resolution || "1080p";
        projAspect.value = settings.aspect_ratio || "16:9";
        projFps.value = String(settings.fps || 30);
        projDurationMode.value = settings.duration_mode || "auto";
        projCustomDuration.value = settings.custom_duration || 30;
        projFitMode.value = settings.fit_mode || "fill";
        projBgFill.value = settings.background_fill || "blur";
        projCustomDurationRow.classList.toggle("view--hidden", projDurationMode.value !== "custom");
        projBgFillRow.classList.toggle("view--hidden", projFitMode.value !== "fit");
        projColorGrade.value = settings.color_grade || "cinematic";
        projSharpen.checked = settings.sharpen !== false;
        projCinematicBars.checked = !!settings.cinematic_bars;
        projFilmGrain.checked = !!settings.film_grain;
        exportQuality.value = settings.quality || "high";
        exportBitrate.value = settings.custom_bitrate_kbps || 8000;
        exportBitrateRow.classList.toggle("view--hidden", exportQuality.value !== "custom");
    }

    function patchProjectSettings(patch) {
        window.pixelforge.video.setProjectSettings(patch, applyState);
    }

    projResolution.addEventListener("change", () => patchProjectSettings({ resolution: projResolution.value }));
    projAspect.addEventListener("change", () => patchProjectSettings({ aspect_ratio: projAspect.value }));
    projFps.addEventListener("change", () => patchProjectSettings({ fps: parseInt(projFps.value, 10) }));
    projColorGrade.addEventListener("change", () => patchProjectSettings({ color_grade: projColorGrade.value }));
    projSharpen.addEventListener("change", () => patchProjectSettings({ sharpen: projSharpen.checked }));
    projCinematicBars.addEventListener("change", () => patchProjectSettings({ cinematic_bars: projCinematicBars.checked }));
    projFilmGrain.addEventListener("change", () => patchProjectSettings({ film_grain: projFilmGrain.checked }));
    projDurationMode.addEventListener("change", () => {
        projCustomDurationRow.classList.toggle("view--hidden", projDurationMode.value !== "custom");
        patchProjectSettings({ duration_mode: projDurationMode.value });
    });
    projCustomDuration.addEventListener("change", () => patchProjectSettings({ custom_duration: parseFloat(projCustomDuration.value) }));
    projFitMode.addEventListener("change", () => {
        projBgFillRow.classList.toggle("view--hidden", projFitMode.value !== "fit");
        patchProjectSettings({ fit_mode: projFitMode.value });
    });
    projBgFill.addEventListener("change", () => patchProjectSettings({ background_fill: projBgFill.value }));

    // ===================== AUDIO TAB =====================

    function renderAudioTab(audio, audio2, rightsNotice) {
        audioFilename.textContent = audio.path ? audio.path.split(/[\\/]/).pop() : "None selected";
        audioVolume.value = audio.volume != null ? audio.volume : 80;
        valAudioVolume.textContent = `${audioVolume.value}%`;
        audioFadeIn.value = audio.fade_in != null ? audio.fade_in : 1.0;
        valAudioFadeIn.textContent = formatSeconds(audioFadeIn.value);
        audioFadeOut.value = audio.fade_out != null ? audio.fade_out : 1.5;
        valAudioFadeOut.textContent = formatSeconds(audioFadeOut.value);
        if (audioDurationMatch) audioDurationMatch.value = audio.duration_match || "none";
        // PHASE 12: music-rights reminder -- only shown while a track
        // is actually attached (rightsNotice is null otherwise, see
        // VideoProject.state()'s music_rights_notice field).
        if (audioRightsNotice) {
            audioRightsNotice.classList.toggle("view--hidden", !rightsNotice);
            if (rightsNotice) audioRightsNotice.textContent = rightsNotice;
        }
        // PHASE 12: second (voice-over) track.
        if (audio2Filename) {
            audio2Filename.textContent = audio2.path ? audio2.path.split(/[\\/]/).pop() : "None selected";
        }
        if (audio2Volume) {
            audio2Volume.value = audio2.volume != null ? audio2.volume : 100;
            valAudio2Volume.textContent = `${audio2Volume.value}%`;
        }
    }

    function pushAudio() {
        window.pixelforge.video.setAudio(
            lastState.audio.path || "", audioVolume.value, audioFadeIn.value, audioFadeOut.value,
            audioDurationMatch ? audioDurationMatch.value : "none", applyState,
        );
    }

    btnChooseAudio.addEventListener("click", () => {
        window.pixelforge.video.chooseAudio((path) => {
            if (!path) return;
            window.pixelforge.video.setAudio(
                path, audioVolume.value, audioFadeIn.value, audioFadeOut.value,
                audioDurationMatch ? audioDurationMatch.value : "none", applyState,
            );
        });
    });
    btnClearAudio.addEventListener("click", () => window.pixelforge.video.clearAudio(applyState));
    audioVolume.addEventListener("input", () => { valAudioVolume.textContent = `${audioVolume.value}%`; });
    audioVolume.addEventListener("change", pushAudio);
    audioFadeIn.addEventListener("input", () => { valAudioFadeIn.textContent = formatSeconds(audioFadeIn.value); });
    audioFadeIn.addEventListener("change", pushAudio);
    audioFadeOut.addEventListener("input", () => { valAudioFadeOut.textContent = formatSeconds(audioFadeOut.value); });
    audioFadeOut.addEventListener("change", pushAudio);
    if (audioDurationMatch) audioDurationMatch.addEventListener("change", pushAudio);

    // PHASE 12: second (voice-over) audio track -- limited support,
    // no fade controls of its own (see _default_audio2's docstring).
    if (btnChooseAudio2) {
        btnChooseAudio2.addEventListener("click", () => {
            window.pixelforge.video.chooseAudio2((path) => {
                if (!path) return;
                window.pixelforge.video.setAudio2(path, audio2Volume.value, applyState);
            });
        });
    }
    if (btnClearAudio2) btnClearAudio2.addEventListener("click", () => window.pixelforge.video.clearAudio2(applyState));
    if (audio2Volume) {
        audio2Volume.addEventListener("input", () => { valAudio2Volume.textContent = `${audio2Volume.value}%`; });
        audio2Volume.addEventListener("change", () => {
            window.pixelforge.video.setAudio2(lastState.audio2.path || "", audio2Volume.value, applyState);
        });
    }

    // ===================== TEXT TAB =====================

    function renderTextList(overlays) {
        textList.innerHTML = "";
        if (!overlays.length) {
            const empty = document.createElement("div");
            empty.className = "placeholder-sub";
            empty.textContent = "No titles added yet.";
            textList.appendChild(empty);
            return;
        }
        overlays.forEach((ov) => {
            const row = document.createElement("div");
            row.className = "video-text-row";
            // PHASE 12: SRT-imported cues carry is_caption=True -- badge
            // them "CC" so the list distinguishes a manual Title from an
            // imported subtitle cue at a glance.
            const badge = ov.is_caption ? `<span class="video-text-row-badge">CC</span>` : "";
            row.innerHTML = `
                ${badge}
                <span class="video-text-row-label">${ov.text}</span>
                <span class="video-text-row-meta">${formatSeconds(ov.start_time)} · ${formatSeconds(ov.duration)}</span>
                <button class="btn-icon" title="Remove">✕</button>
            `;
            row.querySelector(".btn-icon").addEventListener("click", () => {
                window.pixelforge.video.removeTextOverlay(ov.id, applyState);
            });
            textList.appendChild(row);
        });
    }

    textSize.addEventListener("input", () => { valTextSize.textContent = `${textSize.value}px`; });
    textStart.addEventListener("input", () => { valTextStart.textContent = formatSeconds(textStart.value); });
    textDuration.addEventListener("input", () => { valTextDuration.textContent = formatSeconds(textDuration.value); });
    if (textOpacity) textOpacity.addEventListener("input", () => { valTextOpacity.textContent = `${textOpacity.value}%`; });

    btnAddText.addEventListener("click", () => {
        const text = (textInput.value || "").trim();
        if (!text) { showError("Enter some title text first."); return; }
        window.pixelforge.video.addTextOverlay({
            text,
            position: textPosition.value,
            font_size: parseInt(textSize.value, 10),
            color: textColor.value,
            start_time: parseFloat(textStart.value),
            duration: parseFloat(textDuration.value),
            fade: 0.4,
            opacity: textOpacity ? parseInt(textOpacity.value, 10) : 100,
            shadow: textShadow ? textShadow.checked : true,
            animation: textAnimation ? textAnimation.value : "fade",
        }, (state) => {
            applyState(state);
            textInput.value = "";
        });
    });

    // PHASE 12: SRT subtitle import -- one click turns every cue in the
    // chosen .srt file into a caption-style entry in the list above.
    if (btnImportSrt) {
        btnImportSrt.addEventListener("click", () => {
            window.pixelforge.video.chooseSrtFile((path) => {
                if (!path) return;
                if (srtFilename) srtFilename.textContent = path.split(/[\\/]/).pop();
                window.pixelforge.video.importSrt(path, (state) => {
                    applyState(state);
                    if (state && state.ok) {
                        showNotice(`Imported ${state.imported_count} caption${state.imported_count === 1 ? "" : "s"}.`);
                    }
                });
            });
        });
    }

    // ===================== WATERMARK TAB (PHASE 12) =====================

    function renderWatermarkTab(wm) {
        if (!watermarkEnabled) return;
        watermarkEnabled.checked = !!wm.enabled;
        watermarkType.value = wm.type || "text";
        watermarkText.value = wm.text || "";
        watermarkImagePath = wm.image_path || null;
        watermarkImageFilename.textContent = watermarkImagePath ? watermarkImagePath.split(/[\\/]/).pop() : "None selected";
        watermarkPosition.value = wm.position || "bottom-right";
        watermarkOpacity.value = wm.opacity != null ? wm.opacity : 55;
        valWatermarkOpacity.textContent = `${watermarkOpacity.value}%`;
        watermarkScale.value = wm.scale != null ? wm.scale : 16;
        valWatermarkScale.textContent = `${watermarkScale.value}%`;
        const isImage = watermarkType.value === "image";
        watermarkTextRow.classList.toggle("view--hidden", isImage);
        watermarkImageRow.classList.toggle("view--hidden", !isImage);
        btnChooseWatermarkImage.classList.toggle("view--hidden", !isImage);
    }

    if (watermarkType) {
        watermarkType.addEventListener("change", () => {
            const isImage = watermarkType.value === "image";
            watermarkTextRow.classList.toggle("view--hidden", isImage);
            watermarkImageRow.classList.toggle("view--hidden", !isImage);
            btnChooseWatermarkImage.classList.toggle("view--hidden", !isImage);
        });
        watermarkOpacity.addEventListener("input", () => { valWatermarkOpacity.textContent = `${watermarkOpacity.value}%`; });
        watermarkScale.addEventListener("input", () => { valWatermarkScale.textContent = `${watermarkScale.value}%`; });
        btnChooseWatermarkImage.addEventListener("click", () => {
            window.pixelforge.video.chooseWatermarkImage((path) => {
                if (!path) return;
                watermarkImagePath = path;
                watermarkImageFilename.textContent = path.split(/[\\/]/).pop();
            });
        });
        btnApplyWatermark.addEventListener("click", () => {
            const isImage = watermarkType.value === "image";
            if (isImage && !watermarkImagePath) { showError("Choose a logo image first."); return; }
            if (!isImage && !(watermarkText.value || "").trim()) { showError("Enter watermark text first."); return; }
            window.pixelforge.video.setWatermark({
                enabled: true,
                type: watermarkType.value,
                text: watermarkText.value,
                image_path: watermarkImagePath,
                opacity: parseInt(watermarkOpacity.value, 10),
                position: watermarkPosition.value,
                scale: parseInt(watermarkScale.value, 10),
            }, (state) => {
                applyState(state);
                showNotice("Watermark applied.");
            });
        });
        btnClearWatermark.addEventListener("click", () => window.pixelforge.video.clearWatermark(applyState));
    }

    // ===================== TABS =====================

    function setActiveTab(name) {
        tabButtons.forEach((btn) => btn.classList.toggle("is-active", btn.dataset.tab === name));
        Object.keys(tabPanels).forEach((key) => {
            if (tabPanels[key]) tabPanels[key].classList.toggle("view--hidden", key !== name);
        });
    }
    tabButtons.forEach((btn) => btn.addEventListener("click", () => setActiveTab(btn.dataset.tab)));

    // ===================== ADD IMAGES / DROPZONE =====================

    function addImages(paths) {
        if (!paths || !paths.length) return;
        window.pixelforge.video.addImages(paths, applyState);
    }
    btnAddImagesEmpty.addEventListener("click", () => window.pixelforge.openImagesDialog(addImages));
    btnAddImages.addEventListener("click", () => window.pixelforge.openImagesDialog(addImages));

    ["dragover", "dragenter"].forEach((evt) => {
        dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("is-dragover"); });
    });
    ["dragleave", "drop"].forEach((evt) => {
        dropzone.addEventListener(evt, () => dropzone.classList.remove("is-dragover"));
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        const files = Array.from(e.dataTransfer.files || []).map((f) => f.path).filter(Boolean);
        addImages(files);
    });

    // ===================== UNDO / REDO / NEW =====================

    btnUndo.addEventListener("click", () => window.pixelforge.video.undo(applyState));
    btnRedo.addEventListener("click", () => window.pixelforge.video.redo(applyState));
    btnNew.addEventListener("click", () => {
        selectedClipId = null;
        lastPreviewPath = null;
        previewPlayer.src = "";
        previewPlayer.classList.add("view--hidden");
        previewEmpty.classList.remove("view--hidden");
        previewPlaybackError.classList.add("view--hidden");
        btnOpenPreviewExternally.classList.add("view--hidden");
        window.pixelforge.video.newProject(applyState);
    });

    // ===================== PROJECT SAVE / LOAD =====================

    btnSaveProject.addEventListener("click", () => {
        window.pixelforge.video.chooseProjectSavePath((path) => {
            if (!path) return;
            window.pixelforge.video.saveProject(path, (state) => {
                if (state && state.ok === false) showError(state.error || "Couldn't save the project.");
                else applyState(state);
            });
        });
    });
    btnLoadProject.addEventListener("click", () => {
        window.pixelforge.video.chooseProjectOpenPath((path) => {
            if (!path) return;
            selectedClipId = null;
            window.pixelforge.video.loadProject(path, (state) => {
                if (state && state.ok === false) showError(state.error || "Couldn't open that project.");
                else applyState(state);
            });
        });
    });

    // ===================== PREVIEW CODEC DETECTION =====================
    // BUGFIX: switching the preview to WebM/VP9 (see core/video_studio.py's
    // generate_preview() docstring) fixed this on most PySide6 builds, but
    // not all -- a handful of QtWebEngine builds strip inline video decode
    // entirely, or only ship H.264 and not VP9. Rather than betting on one
    // codec, ask THIS EXACT <video> element what it can actually decode
    // before generating anything, and if the chosen one still fails,
    // automatically retry once with the other container before giving up
    // and showing "open externally".
    function canPlay(mime) {
        try {
            return !!previewPlayer.canPlayType(mime);
        } catch (e) {
            return false;
        }
    }
    const CAN_PLAY_WEBM = canPlay('video/webm; codecs="vp9,opus"') || canPlay('video/webm; codecs="vp8,vorbis"');
    const CAN_PLAY_MP4 = canPlay('video/mp4; codecs="avc1.42E01E,mp4a.40.2"');
    // Prefer WebM/VP9 (royalty-free, present on nearly every build) but
    // respect what THIS browser actually reports if it disagrees.
    function preferredContainer() {
        // MP4/H.264 tried first: libx264 encodes noticeably lighter on
        // CPU than libvpx-vp9 (WebM), and it's the same codec the final
        // export uses anyway -- no reason to default to the heavier
        // codec first and make every preview click run hotter/slower
        // than it needs to.
        if (CAN_PLAY_MP4) return "mp4";
        if (CAN_PLAY_WEBM) return "webm";
        return "mp4"; // neither reported support -- still worth trying once, lighter one first
    }
    let previewRetried = false;
    // Once both containers have failed to play inline in this running
    // session, stop wasting a second FFmpeg encode + a confusing retry
    // dance on every click -- generate once and go straight to the
    // external player. Reset on next app launch in case the underlying
    // Qt/codec install gets fixed.
    let inlinePreviewConfirmedBroken = false;

    // ===================== PREVIEW =====================

    function setBusy(isBusy) {
        busy = isBusy;
        btnGeneratePreview.disabled = isBusy;
        btnExport.disabled = isBusy || !lastState.clips.length;
        btnCancelExport.classList.toggle("view--hidden", !isBusy);
    }

    function runGeneratePreview(container) {
        window.pixelforge.video.generatePreview(container, (result) => {
            setBusy(false);
            previewFrame.classList.remove("is-busy");
            btnGeneratePreview.classList.remove("is-loading");
            if (result && result.ok) {
                previewRetried = false;
                lastPreviewPath = result.path;
                btnOpenPreviewExternally.classList.remove("view--hidden");
                if (inlinePreviewConfirmedBroken) {
                    // Already know inline playback can't work this
                    // session -- don't even attempt it, just hand the
                    // file straight to the OS's default player.
                    previewPlaybackError.classList.remove("view--hidden");
                    window.pixelforge.openPathExternally(result.path);
                    return;
                }
                previewPlayer.src = fileUrl(result.path);
                previewPlayer.classList.remove("view--hidden");
                previewEmpty.classList.add("view--hidden");
                previewPlayer.play().catch(() => {});
            } else if (!(result && result.cancelled)) {
                showError((result && result.error) || "Couldn't build a preview.");
            }
        });
    }

    btnGeneratePreview.addEventListener("click", () => {
        if (busy || !lastState.clips.length) return;
        previewRetried = false;
        setBusy(true);
        previewFrame.classList.add("is-busy");
        btnGeneratePreview.classList.add("is-loading");
        previewPlaybackError.classList.add("view--hidden");
        btnOpenPreviewExternally.classList.add("view--hidden");
        runGeneratePreview(preferredContainer());
    });

    // BUGFIX: QWebEngineView's embedded Chromium build varies in which
    // video codecs it can decode inline (see the codec-detection block
    // above) -- canPlayType() is a *hint*, not a guarantee, so the
    // <video> element's own "error" event is still the ground truth.
    // If it fires and we haven't already tried the OTHER container this
    // round, retry once automatically before giving up and falling back
    // to "open in default player". console.warn here so the actual
    // MediaError code/message is visible in devtools if it still fails
    // after both attempts -- 1=ABORTED, 2=NETWORK, 3=DECODE,
    // 4=SRC_NOT_SUPPORTED (that last one means this Qt build has no
    // usable inline video decoder at all, WebM or MP4).
    previewPlayer.addEventListener("error", () => {
        if (!lastPreviewPath) return;
        const mediaError = previewPlayer.error;
        // NOTE: was passing the error as an object literal to console.warn --
        // QtWebEngine's console forwarder stringifies each arg with
        // toString() before it reaches the terminal, so an object always
        // prints as "[object Object]" there (still fine in real devtools,
        // just useless in the "js: ..." terminal log). Build a plain
        // string instead so `python main.py`'s terminal output is useful.
        const codeNames = { 1: "ABORTED", 2: "NETWORK", 3: "DECODE", 4: "SRC_NOT_SUPPORTED" };
        console.warn(
            "[Video Studio] preview playback failed" +
            (mediaError
                ? ` code=${mediaError.code} (${codeNames[mediaError.code] || "UNKNOWN"}) message=${mediaError.message || "(none)"}`
                : " (no MediaError available)"),
        );
        if (!previewRetried) {
            previewRetried = true;
            const other = lastPreviewPath.endsWith(".webm") ? "mp4" : "webm";
            setBusy(true);
            previewFrame.classList.add("is-busy");
            btnGeneratePreview.classList.add("is-loading");
            runGeneratePreview(other);
            return;
        }
        // Both containers failed inline -- this Qt build's embedded
        // browser genuinely has no working video decoder (not a codec
        // choice issue). Remember that for the rest of this session, and
        // open the file in the default OS player automatically instead
        // of leaving the user to notice and click the fallback button.
        inlinePreviewConfirmedBroken = true;
        previewPlaybackError.classList.remove("view--hidden");
        btnOpenPreviewExternally.classList.remove("view--hidden");
        if (lastPreviewPath) window.pixelforge.openPathExternally(lastPreviewPath);
    });

    btnOpenPreviewExternally.addEventListener("click", () => {
        if (lastPreviewPath) window.pixelforge.openPathExternally(lastPreviewPath);
    });

    // ===================== EXPORT =====================

    exportQuality.addEventListener("change", () => {
        exportBitrateRow.classList.toggle("view--hidden", exportQuality.value !== "custom");
        patchProjectSettings({ quality: exportQuality.value });
    });
    exportBitrate.addEventListener("change", () => patchProjectSettings({ custom_bitrate_kbps: parseInt(exportBitrate.value, 10) }));

    btnExport.addEventListener("click", () => {
        if (busy || !lastState.clips.length) return;
        window.pixelforge.video.chooseExportPath((destPath) => {
            if (!destPath) return;
            setBusy(true);
            btnExport.classList.add("is-loading");
            exportResult.classList.add("view--hidden");
            btnOpenExport.classList.add("view--hidden");
            window.pixelforge.video.export(destPath, (result) => {
                setBusy(false);
                btnExport.classList.remove("is-loading");
                if (result && result.ok) {
                    lastExportPath = result.output_path;
                    exportResult.textContent = `Exported to ${result.output_path}`;
                    exportResult.classList.remove("view--hidden");
                    btnOpenExport.classList.remove("view--hidden");
                } else if (!(result && result.cancelled)) {
                    showError((result && result.error) || "Export failed.");
                }
            });
        });
    });

    btnOpenExport.addEventListener("click", () => {
        if (lastExportPath) window.pixelforge.openPathExternally(lastExportPath);
    });

    btnCancelExport.addEventListener("click", () => {
        window.pixelforge.video.cancel();
    });

    // ===================== FFMPEG STATUS =====================

    function checkFfmpeg() {
        window.pixelforge.video.ffmpegStatus((status) => {
            const available = status && status.available;
            ffmpegBadge.textContent = available ? `FFmpeg ${status.version || "detected"}` : "FFmpeg not found";
            ffmpegBadge.className = "badge " + (available ? "badge--ok" : "badge--error");
            ffmpegMissing.classList.toggle("view--hidden", !!available);
            btnExport.disabled = !available || busy || !lastState.clips.length;
            btnGeneratePreview.disabled = !available || busy;
        });
    }

    // ===================== INIT =====================
    // Same "fetch real state once the Qt bridge is ready" pattern as
    // batch.js/upscale.js's refresh()-on-onPixelforgeReady.

    window.onPixelforgeReady(() => {
        checkFfmpeg();
        window.pixelforge.video.state(applyState);
    });
});