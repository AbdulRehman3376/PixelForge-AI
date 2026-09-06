// frontend/promptgen.js
//
// PHASE 7 -- Prompt Engine + AI Image Generation view logic.
//
// Backed by:
//   - core/prompt_engine.py  (local, rule-based -- synonym recognition,
//     prompt preview, conflict detection, unknown-instruction handling,
//     prompt history) via window.pixelforge.promptEngine*
//   - ai/image_generator.py  (local diffusers pipeline by default,
//     optional opt-in cloud API) via window.pixelforge.generateImage /
//     regenerateImage
// (see ui/bridge.py's "PHASE 7" sections and frontend/bridge.js for the
// wiring in between).
//
// Workflow this file implements end-to-end:
//   Prompt -> Prompt Processing (live preview) -> AI Image Generation
//   -> Result Variations -> Select Result -> PixelForge Editor ->
//   Save/Export (the last two steps reuse the existing Enhance view +
//   its Export button, via window.pixelforgeStartSession/showView --
//   no new export code needed, per the spec's "Save/export integration"
//   requirement).

document.addEventListener("DOMContentLoaded", () => {
    const promptInput = document.getElementById("aigen-prompt");
    const negativeInput = document.getElementById("aigen-negative-prompt");
    if (!promptInput || !negativeInput) return; // view not present in this build

    // ----- Preview panel -----
    const previewEmpty = document.getElementById("aigen-preview-empty");
    const previewBody = document.getElementById("aigen-preview-body");
    const previewSummary = document.getElementById("aigen-preview-summary");
    const previewTags = document.getElementById("aigen-preview-tags");
    const previewWarnings = document.getElementById("aigen-preview-warnings");
    const previewUnrecognized = document.getElementById("aigen-preview-unrecognized");

    // ----- Settings controls -----
    const aspectGroup = document.getElementById("aigen-aspect-group");
    const customSizeWrap = document.getElementById("aigen-custom-size");
    const customWidthInput = document.getElementById("aigen-custom-width");
    const customHeightInput = document.getElementById("aigen-custom-height");
    const variationsSelect = document.getElementById("aigen-variations");
    const providerSelect = document.getElementById("aigen-provider");
    const modelSelect = document.getElementById("aigen-model");
    const seedInput = document.getElementById("aigen-seed");
    const availabilityBox = document.getElementById("aigen-availability");

    // ----- Actions / status -----
    const btnGenerate = document.getElementById("btn-aigen-generate");
    const btnRegenerate = document.getElementById("btn-aigen-regenerate");
    const errorBox = document.getElementById("aigen-error");

    // ----- Results -----
    const resultsEmpty = document.getElementById("aigen-results-empty");
    const resultsGrid = document.getElementById("aigen-results-grid");

    // ----- History sidebar -----
    const historyList = document.getElementById("aigen-history-list");
    const historyEmpty = document.getElementById("aigen-history-empty");
    const btnHistoryClear = document.getElementById("btn-aigen-history-clear");

    let selectedAspect = "square";
    let lastResult = null; // full generate() response, used for Regenerate
    let previewDebounceTimer = null;
    let modelManuallySet = false; // becomes true the moment the user touches the dropdown themselves

    // BUGFIX (this session): the prompt engine already detects a
    // "style" keyword in what the user types (anime, cartoon,
    // photorealistic, etc.) and showed it as a preview tag -- but
    // nothing ever fed that detection into the Style/Model dropdown.
    // So generation always used whatever model the dropdown happened
    // to be sitting on (default "Realistic Photo"), completely
    // ignoring a style the user explicitly asked for in the prompt --
    // e.g. typing "anime portrait" still generated a realistic photo.
    // Now the detected style auto-selects the matching model, as long
    // as the user hasn't manually picked one themselves (a manual
    // pick always wins, so this never fights the user's own choice).
    const STYLE_TO_MODEL = {
        "anime": "flux-anime",
        "cartoon": "flux-anime",
        "pixel art": "flux-anime",
        "sketch": "flux-anime",
        "photorealistic": "flux-realism",
    };

    function applyDetectedStyleToModel(categories) {
        if (modelManuallySet || !modelSelect) return;
        const styles = (categories && categories.style) || [];
        for (const style of styles) {
            if (STYLE_TO_MODEL[style]) {
                modelSelect.value = STYLE_TO_MODEL[style];
                return;
            }
        }
        // No recognized style in the prompt -- fall back to the
        // balanced general-purpose model rather than silently keeping
        // whatever was last auto-picked for a different prompt.
        modelSelect.value = "flux";
    }

    if (modelSelect) {
        modelSelect.addEventListener("change", () => { modelManuallySet = true; });
    }

    // ============================================================
    // Small local helper (this view's own copy -- editor.js/filters.js
    // each keep their own too rather than sharing a global, same
    // pattern already used across this project's views).
    // ============================================================
    function debounce(fn, delay) {
        let timer = null;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => fn(...args), delay);
        };
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text == null ? "" : String(text);
        return div.innerHTML;
    }

    // ============================================================
    // PROMPT PREVIEW (Synonym recognition / Prompt preview /
    // Conflict detection / Unknown instruction handling)
    // ============================================================
    function renderPreview(result) {
        if (!result || !result.ok) {
            previewEmpty.classList.remove("view--hidden");
            previewEmpty.textContent = (result && result.error) || "Couldn't parse that prompt.";
            previewBody.classList.add("view--hidden");
            return;
        }

        if (result.is_empty) {
            previewEmpty.classList.remove("view--hidden");
            previewEmpty.textContent = "Start typing a prompt to see what PixelForge understood, before you generate anything.";
            previewBody.classList.add("view--hidden");
            return;
        }

        previewEmpty.classList.add("view--hidden");
        previewBody.classList.remove("view--hidden");

        previewSummary.textContent = result.summary || "";

        // Detected categories as tag chips (Synonym recognition made visible).
        previewTags.innerHTML = "";
        const categories = result.categories || {};
        Object.keys(categories).forEach((category) => {
            categories[category].forEach((value) => {
                const chip = document.createElement("span");
                chip.className = `aigen-tag aigen-tag--${category}`;
                chip.textContent = `${category}: ${value}`;
                previewTags.appendChild(chip);
            });
        });

        applyDetectedStyleToModel(categories);

        // Conflict detection -- shown as a warning, never silently resolved.
        if (result.conflicts && result.conflicts.length) {
            previewWarnings.classList.remove("view--hidden");
            previewWarnings.innerHTML =
                `<strong>⚠ Conflicting instructions:</strong><ul>` +
                result.conflicts.map((c) => `<li>${escapeHtml(c.message)}</li>`).join("") +
                `</ul>`;
        } else {
            previewWarnings.classList.add("view--hidden");
            previewWarnings.innerHTML = "";
        }

        // Unknown instruction handling -- surfaced, not dropped silently.
        if (result.unrecognized && result.unrecognized.length) {
            previewUnrecognized.classList.remove("view--hidden");
            previewUnrecognized.innerHTML =
                `<strong>Not recognized as a style/mood/lighting keyword</strong> (used as free text): ` +
                escapeHtml(result.unrecognized.join(", "));
        } else {
            previewUnrecognized.classList.add("view--hidden");
            previewUnrecognized.innerHTML = "";
        }
    }

    const runPreview = debounce(() => {
        if (!window.pixelforge) return;
        window.pixelforge.promptEnginePreview(promptInput.value, negativeInput.value, renderPreview);
    }, 300);

    promptInput.addEventListener("input", runPreview);
    negativeInput.addEventListener("input", runPreview);

    // ============================================================
    // ASPECT RATIO / CUSTOM SIZE
    // ============================================================
    if (aspectGroup) {
        aspectGroup.querySelectorAll(".aigen-aspect-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                aspectGroup.querySelectorAll(".aigen-aspect-btn").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                selectedAspect = btn.dataset.aspect;
                customSizeWrap.classList.toggle("view--hidden", selectedAspect !== "custom");
            });
        });
    }

    // ============================================================
    // AVAILABILITY (local/cloud provider status)
    // ============================================================
    function renderAvailability(result) {
        if (!result || !result.ok) return;
        const local = result.local || {};
        const cloud = result.cloud || {};
        availabilityBox.classList.remove("view--hidden");

        const localBadge = local.available
            ? `<span class="badge badge--ok">Local: ready</span>`
            : `<span class="badge badge--muted" title="${escapeHtml(local.message || "")}">Local: needs setup</span>`;
        const cloudBadge = cloud.available
            ? `<span class="badge badge--ok">Cloud: configured</span>`
            : `<span class="badge badge--muted" title="${escapeHtml(cloud.message || "")}">Cloud: not configured (optional)</span>`;

        availabilityBox.innerHTML = `${localBadge} ${cloudBadge}`;
        if (!local.available) {
            availabilityBox.title = local.message || "";
        }
    }

    function checkAvailability() {
        if (!window.pixelforge) return;
        window.pixelforge.aiImageGenAvailability(renderAvailability);
    }

    // ============================================================
    // GENERATE / REGENERATE (progress + error handling)
    // ============================================================
    function currentDimensions() {
        if (selectedAspect === "custom") {
            return {
                width: Number(customWidthInput.value) || 512,
                height: Number(customHeightInput.value) || 512,
            };
        }
        return {};
    }

    function buildGenerateParams() {
        const dims = currentDimensions();
        const seedRaw = seedInput.value.trim();
        return {
            prompt: promptInput.value.trim(),
            negative_prompt: negativeInput.value.trim(),
            aspect_ratio: selectedAspect,
            custom_width: dims.width,
            custom_height: dims.height,
            num_images: Number(variationsSelect.value) || 1,
            provider: providerSelect.value || "local",
            model_id: modelSelect ? (modelSelect.value || "flux-realism") : "flux-realism",
            seed: seedRaw === "" ? null : Number(seedRaw),
        };
    }

    function setBusy(isBusy) {
        btnGenerate.disabled = isBusy;
        btnGenerate.textContent = isBusy ? "Generating..." : "";
        if (!isBusy) {
            btnGenerate.innerHTML =
                '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">' +
                '<path d="M12 3l1.6 4.6L18 9l-4.4 1.4L12 15l-1.6-4.6L6 9l4.4-1.4z"/></svg> Generate';
        }
    }

    function showError(message) {
        if (!message) {
            errorBox.classList.add("view--hidden");
            errorBox.textContent = "";
            return;
        }
        errorBox.classList.remove("view--hidden");
        errorBox.textContent = message;
    }

    function renderResults(result) {
        showError(null);
        if (!result || !result.ok) {
            showError((result && result.error) || "Generation failed for an unknown reason.");
            resultsEmpty.classList.remove("view--hidden");
            resultsGrid.classList.add("view--hidden");
            return;
        }

        lastResult = result;
        btnRegenerate.disabled = false;

        resultsEmpty.classList.add("view--hidden");
        resultsGrid.classList.remove("view--hidden");
        resultsGrid.innerHTML = "";

        (result.images || []).forEach((img, index) => {
            const card = document.createElement("div");
            card.className = "aigen-result-card";

            const fileUrl = `file:///${img.path.replace(/\\/g, "/")}`;
            card.innerHTML = `
                <div class="aigen-result-thumb">
                    <img src="${fileUrl}" alt="Generated image ${index + 1}">
                </div>
                <div class="aigen-result-meta">
                    <span>${result.width}&times;${result.height}</span>
                    <span>seed ${escapeHtml(img.seed)}</span>
                </div>
                <div class="aigen-result-actions">
                    <button class="btn btn-primary aigen-btn-edit">Edit in PixelForge</button>
                </div>
            `;

            card.querySelector(".aigen-btn-edit").addEventListener("click", () => {
                sendToEditor(img.path);
            });

            resultsGrid.appendChild(card);
        });
    }

    function generate() {
        if (!window.pixelforge) return;
        const params = buildGenerateParams();
        if (!params.prompt) {
            showError("Please enter a prompt describing the image you want to generate.");
            return;
        }

        showError(null);
        setBusy(true);
        window.pixelforge.generateImage(params, (result) => {
            setBusy(false);
            renderResults(result);
            refreshHistory();
        });
    }

    function regenerate() {
        if (!window.pixelforge || !lastResult) return;
        showError(null);
        setBusy(true);
        const spec = {
            prompt: lastResult.prompt,
            negative_prompt: lastResult.negative_prompt,
            aspect_ratio: "custom",
            width: lastResult.width,
            height: lastResult.height,
            num_images: Number(variationsSelect.value) || 1,
            provider: lastResult.provider,
            model_id: lastResult.model_id,
            // seed intentionally omitted -- see ai/image_generator.py::regenerate,
            // a fresh seed is what makes "Regenerate" produce something new.
        };
        window.pixelforge.regenerateImage(spec, (result) => {
            setBusy(false);
            renderResults(result);
        });
    }

    btnGenerate.addEventListener("click", generate);
    btnRegenerate.addEventListener("click", regenerate);

    // ----- Generated image -> PixelForge editor hand-off -----
    // Starts a brand-new Edit Session anchored to the generated file
    // (ui/bridge.py::sendGeneratedImageToEditor), then switches to
    // Enhance -- from there the existing Export button and full
    // Enhance/Filters/Remove BG toolset all just work, per the spec's
    // "Save/export integration" requirement (no new export code needed).
    function sendToEditor(generatedPath) {
        if (!window.pixelforge) return;
        window.pixelforge.sendGeneratedImageToEditor(generatedPath, (state) => {
            if (!state || !state.ok) {
                showError((state && state.error) || "Couldn't open that image in the editor.");
                return;
            }
            if (window.pixelforgeOnSessionUpdated) window.pixelforgeOnSessionUpdated(state);
            const target = state.working_path || generatedPath;
            if (window.pixelforgeShowView) window.pixelforgeShowView("enhance");
            if (window.pixelforgeLoadImageIntoEditor) window.pixelforgeLoadImageIntoEditor(target);
        });
    }

    // ============================================================
    // PROMPT HISTORY (save/reuse)
    // ============================================================
    function renderHistory(result) {
        if (!result || !result.ok || !result.history || !result.history.length) {
            historyEmpty.classList.remove("view--hidden");
            historyList.querySelectorAll(".aigen-history-item").forEach((el) => el.remove());
            return;
        }

        historyEmpty.classList.add("view--hidden");
        historyList.querySelectorAll(".aigen-history-item").forEach((el) => el.remove());

        result.history.forEach((entry) => {
            const item = document.createElement("div");
            item.className = "aigen-history-item";
            item.innerHTML = `
                <div class="aigen-history-prompt" title="${escapeHtml(entry.prompt)}">${escapeHtml(entry.prompt)}</div>
                <div class="aigen-history-actions">
                    <button class="btn-icon aigen-history-reuse" title="Reuse this prompt">Reuse</button>
                    <button class="btn-icon aigen-history-delete" title="Remove from history">&times;</button>
                </div>
            `;

            item.querySelector(".aigen-history-reuse").addEventListener("click", () => {
                promptInput.value = entry.prompt || "";
                negativeInput.value = entry.negative_prompt || "";
                const settings = entry.settings || {};
                if (settings.aspect_ratio && aspectGroup) {
                    const btn = aspectGroup.querySelector(`[data-aspect="${settings.aspect_ratio}"]`);
                    if (btn) btn.click();
                }
                if (settings.provider) providerSelect.value = settings.provider;
                if (settings.model_id && modelSelect) modelSelect.value = settings.model_id;
                runPreview();
                promptInput.focus();
            });

            item.querySelector(".aigen-history-delete").addEventListener("click", () => {
                if (!window.pixelforge) return;
                window.pixelforge.promptHistoryDelete(entry.id, () => refreshHistory());
            });

            historyList.appendChild(item);
        });
    }

    function refreshHistory() {
        if (!window.pixelforge) return;
        window.pixelforge.promptHistoryList(renderHistory);
    }

    if (btnHistoryClear) {
        btnHistoryClear.addEventListener("click", () => {
            if (!window.pixelforge) return;
            window.pixelforge.promptHistoryClear(() => refreshHistory());
        });
    }

    // ============================================================
    // INIT
    // ============================================================
    if (window.onPixelforgeReady) {
        window.onPixelforgeReady(() => {
            checkAvailability();
            refreshHistory();
        });
    }
});