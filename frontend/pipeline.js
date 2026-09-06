// frontend/pipeline.js
//
// PHASE 6 -- Smart Pipeline (frontend, v2).
//
// WHAT CHANGED FROM v1, AND WHY
// -----------------------------
// v1 showed one line: "Night Portrait -- 82% confidence [Apply in Filters]".
// That made the whole feature a SMART FILTER PICKER: it chose a preset and
// a fixed strength and told you nothing about what it was actually going
// to do to your photo.
//
// v2 shows the DECISION, not just the conclusion:
//
//     Analyze -> Understand -> Diagnose -> Build a recipe -> Preview -> Apply
//
//   - the recipe itself: 12 ordered steps, each with the value it will
//     apply AND the measured reason, in the real processing order
//   - every SKIPPED step, with why ("already sharp 68/100", "no face
//     detected") -- so the skips are visibly deliberate, not omissions
//   - a match score, and the runners-up with their own scores plus a
//     "Why not?" for each
//   - an intensity that comes from THIS photo (a nearly-right photo gets
//     less; a badly-lit one gets more) and is still user-overridable
//   - a safety check before Apply -- clipping, crushed shadows, neon
//     colour, halo sharpening, denoise-vs-sharpen conflict
//   - Customize: flip any step on or off and watch the recipe and the
//     safety verdict recompute
//
// All of the reasoning lives in core/smart_pipeline.py -- this file
// renders it and sends back the user's overrides. No decision logic here,
// deliberately: two implementations of the same rules would drift.
//
// AI PLACEMENT: there is no model in this feature, on purpose. Phase 5's
// Analyzer does the perception (face detection is a real model; the
// optional scene classifier is too), Phase 6 is a transparent rule engine,
// and Phase 2/3's enhancer moves the pixels. Auditable beats mysterious
// for something that edits the user's photos.

(function () {
    "use strict";

    let root = null;          // #pipeline-results
    let el = {};              // cached child nodes
    let built = false;

    let current = null;       // the recommendation object from Python
    let overrides = { steps: {} };
    let busy = false;
    let customizeOpen = false;
    let intensityTimer = null;
    // Which working image the recipe on screen was built from. If the
    // session moves off it (undo, another tool's edit, a project load),
    // the recipe describes pixels that are no longer on screen.
    let builtFrom = "";

    function api() {
        return window.pixelforge || null;
    }

    function session() {
        return window.pixelforgeSession || null;
    }

    function toast(message, kind) {
        if (window.pixelforgeToast) window.pixelforgeToast(message, kind);
        else console.log("PixelForge:", message);
    }

    // ---------------------------------------------------------------
    // Markup
    // ---------------------------------------------------------------
    // Built once in JS rather than written into index.html: the recipe
    // table, the alternatives and the warning list are all data-length
    // driven, so there is no fixed markup to hand-write for them anyway.

    function build() {
        if (built || !root) return;
        root.innerHTML = `
            <div class="pipeline-header">
                <div class="pipeline-title-group">
                    <span class="pipeline-name" id="pipeline-name">Smart Pipeline</span>
                    <span class="pipeline-goal" id="pipeline-goal"></span>
                </div>
                <span class="pipeline-score" id="pipeline-score">--</span>
            </div>

            <button type="button" class="pipeline-disclosure" id="btn-pipeline-why">
                <span>Why this recommendation?</span>
                <span class="pipeline-caret">&#9662;</span>
            </button>
            <div class="pipeline-why view--hidden" id="pipeline-why">
                <p class="pipeline-explanation" id="pipeline-explanation"></p>
                <div class="pipeline-findings" id="pipeline-findings"></div>
                <div class="analyze-badges" id="pipeline-signals"></div>
            </div>

            <div class="pipeline-section-head">
                <h4 class="pipeline-section-title">Recipe</h4>
                <span class="pipeline-section-meta" id="pipeline-recipe-meta"></span>
            </div>
            <div class="pipeline-recipe" id="pipeline-recipe"></div>

            <div class="pipeline-intensity">
                <div class="editor-control-label">
                    <span>Intensity <span class="pipeline-intensity-hint" id="pipeline-intensity-hint"></span></span>
                    <span class="editor-control-value" id="pipeline-intensity-value">70%</span>
                </div>
                <input type="range" class="input-range" id="ctrl-pipeline-intensity" min="0" max="100" value="70">
            </div>

            <div class="pipeline-safety" id="pipeline-safety"></div>

            <div class="pipeline-actions">
                <button class="btn btn-secondary btn-small" id="btn-pipeline-preview">Preview</button>
                <button class="btn btn-secondary btn-small" id="btn-pipeline-customize">Customize</button>
                <button class="btn btn-primary btn-small" id="btn-pipeline-apply">Apply</button>
            </div>

            <div class="pipeline-alternatives" id="pipeline-alternatives"></div>

            <div class="pipeline-footer">
                <button type="button" class="pipeline-link" id="btn-pipeline-reset">Reset my changes</button>
                <button type="button" class="pipeline-link" id="btn-pipeline-dismiss">Dismiss</button>
            </div>
        `;

        el = {
            name: root.querySelector("#pipeline-name"),
            goal: root.querySelector("#pipeline-goal"),
            score: root.querySelector("#pipeline-score"),
            whyToggle: root.querySelector("#btn-pipeline-why"),
            why: root.querySelector("#pipeline-why"),
            explanation: root.querySelector("#pipeline-explanation"),
            findings: root.querySelector("#pipeline-findings"),
            signals: root.querySelector("#pipeline-signals"),
            recipe: root.querySelector("#pipeline-recipe"),
            recipeMeta: root.querySelector("#pipeline-recipe-meta"),
            intensity: root.querySelector("#ctrl-pipeline-intensity"),
            intensityValue: root.querySelector("#pipeline-intensity-value"),
            intensityHint: root.querySelector("#pipeline-intensity-hint"),
            safety: root.querySelector("#pipeline-safety"),
            preview: root.querySelector("#btn-pipeline-preview"),
            customize: root.querySelector("#btn-pipeline-customize"),
            apply: root.querySelector("#btn-pipeline-apply"),
            alternatives: root.querySelector("#pipeline-alternatives"),
            reset: root.querySelector("#btn-pipeline-reset"),
            dismiss: root.querySelector("#btn-pipeline-dismiss"),
        };

        el.whyToggle.addEventListener("click", () => {
            const hidden = el.why.classList.toggle("view--hidden");
            el.whyToggle.classList.toggle("is-open", !hidden);
        });

        // Slider: the label tracks the drag live, but the recipe only
        // recomputes after a short pause -- each recompute is a Python
        // round trip, and intensity feeds the safety check, so firing on
        // every pixel of drag would spam the bridge for no benefit.
        el.intensity.addEventListener("input", () => {
            el.intensityValue.textContent = `${el.intensity.value}%`;
            markCustomized();
            clearTimeout(intensityTimer);
            intensityTimer = setTimeout(() => {
                overrides.intensity = Number(el.intensity.value);
                recompute();
            }, 220);
        });

        el.preview.addEventListener("click", doPreview);
        el.customize.addEventListener("click", toggleCustomize);
        el.apply.addEventListener("click", doApply);
        el.reset.addEventListener("click", () => {
            overrides = { steps: {} };
            recompute();
        });
        el.dismiss.addEventListener("click", hide);

        built = true;
    }

    // ---------------------------------------------------------------
    // Rendering
    // ---------------------------------------------------------------

    const SEVERITY_CLASS = { high: "is-high", medium: "is-medium", low: "is-low" };

    function scoreClass(score) {
        if (score >= 80) return "is-strong";
        if (score >= 60) return "is-fair";
        return "is-weak";
    }

    function render() {
        if (!current || !root) return;
        build();

        const recipe = current.recipe || {};
        const score = current.match_score || current.confidence || 0;

        el.name.textContent = current.name || "Smart Pipeline";
        el.goal.textContent = current.goal || "";
        el.score.textContent = `${score}% Match`;
        el.score.className = `pipeline-score ${scoreClass(score)}`;

        el.explanation.textContent = current.explanation || current.reason || "";
        renderFindings();
        renderSignals();
        renderRecipe(recipe);

        const intensity = recipe.intensity != null ? recipe.intensity : (current.intensity || 70);
        el.intensity.value = intensity;
        el.intensityValue.textContent = `${intensity}%`;
        el.intensityHint.textContent = current.customized
            ? "(your setting)"
            : "(chosen for this photo)";

        renderSafety(current.safety || {});
        renderAlternatives(current.alternatives || []);

        el.reset.classList.toggle("view--hidden", !current.customized);
        root.classList.remove("view--hidden");
    }

    function renderFindings() {
        el.findings.innerHTML = "";
        const problems = current.problems || [];
        const strengths = current.strengths || [];
        if (!problems.length && !strengths.length) return;

        if (problems.length) {
            const group = document.createElement("div");
            group.className = "pipeline-finding-group";
            group.innerHTML = `<span class="pipeline-finding-head">What needs fixing</span>`;
            problems.forEach((p) => {
                const row = document.createElement("div");
                row.className = `pipeline-finding ${SEVERITY_CLASS[p.severity] || "is-low"}`;
                row.innerHTML =
                    `<span class="pipeline-finding-label">${escapeHtml(p.label)}</span>` +
                    `<span class="pipeline-finding-detail">${escapeHtml(p.detail || "")}</span>`;
                group.appendChild(row);
            });
            el.findings.appendChild(group);
        }

        if (strengths.length) {
            // Shown as prominently as the problems on purpose: "this is
            // already good, so I'm leaving it alone" is the part users
            // don't trust an auto-editor to do, and it's exactly what
            // drives the skips below.
            const group = document.createElement("div");
            group.className = "pipeline-finding-group";
            group.innerHTML = `<span class="pipeline-finding-head">Already good (left alone)</span>`;
            strengths.forEach((s) => {
                const row = document.createElement("div");
                row.className = "pipeline-finding is-good";
                row.innerHTML =
                    `<span class="pipeline-finding-label">${escapeHtml(s.label)}</span>` +
                    `<span class="pipeline-finding-detail">${escapeHtml(s.detail || "")}</span>`;
                group.appendChild(row);
            });
            el.findings.appendChild(group);
        }
    }

    function renderSignals() {
        el.signals.innerHTML = "";
        (current.signals || []).forEach((text) => {
            const badge = document.createElement("span");
            badge.className = "badge badge--muted";
            badge.textContent = text;
            el.signals.appendChild(badge);
        });
    }

    function renderRecipe(recipe) {
        const steps = recipe.steps || [];
        el.recipeMeta.textContent = steps.length
            ? `${recipe.applied_count} applied · ${recipe.skipped_count} skipped`
            : "";

        el.recipe.innerHTML = "";
        steps.forEach((step) => {
            const row = document.createElement("div");
            row.className = `recipe-row is-${step.status}`;
            if (customizeOpen) row.classList.add("is-editable");

            const label = document.createElement("span");
            label.className = "recipe-step";
            label.textContent = step.label;

            const action = document.createElement("span");
            action.className = "recipe-action";
            action.textContent = step.action || "No change";

            const status = document.createElement("span");
            status.className = "recipe-status";

            if (customizeOpen) {
                // In Customize the status cell becomes the control -- one
                // click flips the step. Steps the engine has no value for
                // can't be forced on (there'd be nothing to apply), and
                // say so rather than silently doing nothing.
                const hasValue = step.adjustments && Object.keys(step.adjustments).length > 0;
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "recipe-toggle";
                btn.textContent = step.status === "apply" ? "✓ Apply" : "⏭ Skipped";
                if (step.status === "skip" && !hasValue) {
                    btn.disabled = true;
                    btn.title = "Nothing measured to apply for this step on this photo";
                } else {
                    btn.title = step.status === "apply" ? "Click to skip this step" : "Click to apply this step";
                    btn.addEventListener("click", () => {
                        overrides.steps[step.id] = step.status === "apply" ? "skip" : "apply";
                        recompute();
                    });
                }
                status.appendChild(btn);
            } else {
                status.textContent = step.status === "apply" ? "✓ Apply" : "⏭ Skipped";
            }

            const reason = document.createElement("span");
            reason.className = "recipe-reason";
            reason.textContent = step.reason || "";

            row.append(label, action, status, reason);
            el.recipe.appendChild(row);
        });
    }

    function renderSafety(safety) {
        el.safety.innerHTML = "";
        const warnings = safety.warnings || [];

        const head = document.createElement("div");
        head.className = `pipeline-safety-head ${safety.safe ? "is-safe" : "is-warn"}`;
        head.textContent = safety.safe
            ? "✓ Pipeline safe to apply"
            : `⚠ ${safety.label || "Check these before applying"}`;
        el.safety.appendChild(head);

        warnings.forEach((w) => {
            const row = document.createElement("div");
            row.className = "pipeline-warning";
            row.innerHTML =
                `<span class="pipeline-warning-title">${escapeHtml(w.title)}</span>` +
                `<span class="pipeline-warning-detail">${escapeHtml(w.detail || "")}</span>` +
                `<span class="pipeline-warning-fix">Fix: ${escapeHtml(w.fix || "")}</span>`;
            el.safety.appendChild(row);
        });

        // A warning is advice, not a block -- the user can still apply,
        // and can still undo (Ctrl+Z) afterwards. But the button should
        // read differently so the click is an informed one.
        el.apply.textContent = safety.safe ? "Apply" : "Apply anyway";
    }

    function renderAlternatives(alternatives) {
        el.alternatives.innerHTML = "";
        if (!alternatives.length) return;

        const head = document.createElement("h4");
        head.className = "pipeline-section-title";
        head.textContent = "Other recommendations";
        el.alternatives.appendChild(head);

        alternatives.forEach((alt) => {
            const item = document.createElement("div");
            item.className = "pipeline-alt";

            const top = document.createElement("div");
            top.className = "pipeline-alt-top";

            const name = document.createElement("span");
            name.className = "pipeline-alt-name";
            name.textContent = alt.name;

            const score = document.createElement("span");
            score.className = `pipeline-alt-score ${scoreClass(alt.match_score)}`;
            score.textContent = `${alt.match_score}%`;

            const use = document.createElement("button");
            use.type = "button";
            use.className = "btn btn-secondary btn-small";
            use.textContent = "Use";
            use.addEventListener("click", () => {
                // Switching looks resets the overrides: the new look has
                // its own adaptive intensity and its own step decisions,
                // and carrying the old look's manual tweaks across would
                // silently misapply them.
                overrides = { steps: {} };
                recompute(alt.id);
            });

            top.append(name, score, use);
            item.appendChild(top);

            if ((alt.why_not || []).length || (alt.reasons || []).length) {
                const toggle = document.createElement("button");
                toggle.type = "button";
                toggle.className = "pipeline-disclosure pipeline-disclosure--sub";
                toggle.innerHTML = `<span>Why not this one?</span><span class="pipeline-caret">&#9662;</span>`;

                const body = document.createElement("div");
                body.className = "pipeline-alt-why view--hidden";

                (alt.why_not || []).forEach((text) => {
                    const li = document.createElement("p");
                    li.className = "pipeline-alt-con";
                    li.textContent = `– ${text}`;
                    body.appendChild(li);
                });
                (alt.reasons || []).forEach((text) => {
                    const li = document.createElement("p");
                    li.className = "pipeline-alt-pro";
                    li.textContent = `+ ${text}`;
                    body.appendChild(li);
                });

                toggle.addEventListener("click", () => {
                    const hidden = body.classList.toggle("view--hidden");
                    toggle.classList.toggle("is-open", !hidden);
                });

                item.append(toggle, body);
            }

            el.alternatives.appendChild(item);
        });
    }

    function escapeHtml(text) {
        return String(text == null ? "" : text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    // ---------------------------------------------------------------
    // Actions
    // ---------------------------------------------------------------

    function setBusy(on, labelTarget) {
        busy = on;
        if (!built) return;
        [el.preview, el.customize, el.apply].forEach((b) => { b.disabled = on; });
        el.intensity.disabled = on;
        root.classList.toggle("is-busy", on);
        if (labelTarget) labelTarget.classList.toggle("is-loading", on);
    }

    function markCustomized() {
        if (!current) return;
        current.customized = true;
        el.intensityHint.textContent = "(your setting)";
        el.reset.classList.remove("view--hidden");
    }

    // Full run: analyze + recommend. sourcePath "" = the current working
    // image, which is what we always want -- the recommendation should
    // describe the photo the user is looking at right now, after their
    // filter/background/crop edits, not the untouched original.
    function run(sourcePath) {
        const bridge = api();
        if (!bridge) return;
        build();
        overrides = { steps: {} };
        customizeOpen = false;
        if (built) el.customize.textContent = "Customize";

        root.classList.remove("view--hidden");
        root.classList.add("is-busy");
        el.name.textContent = "Analyzing your photo…";
        el.goal.textContent = "";
        el.score.textContent = "…";
        el.recipe.innerHTML = "";
        el.safety.innerHTML = "";
        el.alternatives.innerHTML = "";

        bridge.smartPipeline(sourcePath || "", (result) => {
            root.classList.remove("is-busy");
            if (!result.ok) {
                hide();
                toast(result.error || "Smart Pipeline failed.", "error");
                return;
            }
            current = result.recommendation;
            builtFrom = (result.session && result.session.working_path)
                || (session() && session().workingPath())
                || sourcePath
                || "";
            if (result.analysis && window.pixelforgeRenderAnalysis) {
                // The pipeline already ran the full Phase 5 analysis, so
                // hand it to the Analyzer panel too instead of making the
                // user click Analyze and pay for the same work twice.
                window.pixelforgeRenderAnalysis(result.analysis);
            }
            render();
        });
    }

    // Re-derives the recipe under the current overrides. Used by the
    // intensity slider, the Customize toggles, "Use" on an alternative,
    // and Reset -- one code path, so they can't disagree.
    function recompute(lookId) {
        const bridge = api();
        if (!bridge || !current) return;
        setBusy(true);
        bridge.smartPipelineCustomize(lookId || current.id, overrides, (result) => {
            setBusy(false);
            if (!result.ok) {
                toast(result.error || "Could not update the recipe.", "error");
                return;
            }
            current = result.recommendation;
            render();
        });
    }

    function doPreview() {
        const bridge = api();
        if (!bridge || !current || busy) return;
        setBusy(true);
        el.preview.textContent = "Rendering…";
        bridge.smartPipelinePreview(current.id, overrides, (result) => {
            setBusy(false);
            el.preview.textContent = "Preview";
            if (!result.ok) {
                toast(result.error || "Preview failed.", "error");
                return;
            }
            if (result.recommendation) current = result.recommendation;
            render();
            if (window.pixelforgeShowPipelinePreview) {
                window.pixelforgeShowPipelinePreview(result.path);
                toast("Preview only -- nothing applied yet. Press Apply to keep it.");
            }
        });
    }

    function toggleCustomize() {
        customizeOpen = !customizeOpen;
        el.customize.textContent = customizeOpen ? "Done customizing" : "Customize";
        root.classList.toggle("is-customizing", customizeOpen);
        if (current) renderRecipe(current.recipe || {});
    }

    function doApply() {
        const bridge = api();
        if (!bridge || !current || busy) return;
        setBusy(true);
        el.apply.textContent = "Applying…";
        bridge.smartPipelineApply(current.id, overrides, (result) => {
            setBusy(false);
            if (!result.ok) {
                el.apply.textContent = "Apply";
                toast(result.error || "Could not apply the pipeline.", "error");
                return;
            }
            // Grab the labels BEFORE adopting: adopting moves the session
            // onto the new working image, which fires the onChange handler
            // at the bottom of this file and clears `current`.
            const recipe = current.recipe || {};
            const appliedName = current.name;
            const appliedIntensity = recipe.intensity;

            // The result became the new WORKING IMAGE. Handing the state
            // to session.js is what makes every other view (Enhance,
            // Filters, Remove BG, Export) pick it up -- and Ctrl+Z undo
            // it -- without any per-view hand-off code.
            const sess = session();
            if (sess && result.session) sess.adopt(result.session);

            el.apply.textContent = "Apply";
            // The old recommendation described the photo BEFORE this edit,
            // so leaving it on screen would invite applying the same grade
            // twice. (adopt() above usually clears it already; this is the
            // no-session-manager fallback.)
            hide();
            toast(`${appliedName} applied at ${appliedIntensity}% -- Ctrl+Z to undo`);
        });
    }

    function hide() {
        if (root) root.classList.add("view--hidden");
        current = null;
        builtFrom = "";
        overrides = { steps: {} };
        customizeOpen = false;
        if (built) {
            el.customize.textContent = "Customize";
            root.classList.remove("is-customizing", "is-busy");
        }
    }

    // ---------------------------------------------------------------
    // Public surface
    // ---------------------------------------------------------------

    window.pixelforgePipeline = {
        run,
        hide,
        isOpen: () => !!(root && !root.classList.contains("view--hidden")),
        recommendation: () => current,
    };

    document.addEventListener("DOMContentLoaded", () => {
        root = document.getElementById("pipeline-results");
        if (root) build();

        // The working image changing under us (undo, another tool's edit,
        // a project load) invalidates the recipe on screen -- it was built
        // from different pixels. Drop it rather than show a stale plan.
        if (window.pixelforgeSession) {
            window.pixelforgeSession.onChange((state) => {
                if (!current) return;
                if (!state || !state.has_session || state.working_path !== builtFrom) hide();
            });
        }
    });
})();
