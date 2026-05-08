# Keyhole + Skippy: Remediation plan for NXP-internal review

**Document purpose.** This is a peer-review remediation plan written for execution by Claude Code agents working in the `keyhole` and `personal-ai-framework` (Skippy) repos. It came out of an external Claude session that cross-reviewed the `CLAUDE_REVIEW_BRIEFING.md` (Keyhole), `skippy-claude-briefing.md`, the recipe taxonomy, the Keyhole data bundle (json/md/xlsx), and the Skippy data bundle (xlsx). Audience for the resulting deck is **NXP-internal review**, which means methodology rigor is the binding constraint.

**Out of scope (explicit).** Power, thermal envelope, sustained-utilization throttling. Don't touch these. If a task in this plan tempts you to add power modeling, stop and skip it.

**Out of scope (implicit).** Don't delete historical run data even if a current projection rule would say "this configuration shouldn't deploy." The NPU Mid spec evolved during the campaign — it was originally FP-capable, then INT8 was added, then it was locked to INT8-only with FP moved to NPU High. The historical FP-on-Mid run data stays in the JSON bundles for reproducibility. The fix for that issue is **rendering / projection logic that respects the current dtype gating**, not a backfill of the data layer.

**How to read this document.**

- Tasks are prefixed by repo scope: `KH-` (Keyhole), `SK-` (Skippy), `SHARED-` (changes both repos must coordinate on).
- Priority levels: `P0` = blocks NXP-internal review; `P1` = methodology improvement before next data run; `P2` = experiments worth running; `P3` = documentation / framing.
- Each task has a "Why" section linking to the reviewer concern, a "Do" section with concrete actions, a "Files" section with best-guess paths to touch, and a "Verify" section with acceptance criteria.
- File paths are best-guess based on what was visible in the briefings. Verify paths against the actual repo before editing.
- When in doubt about scope, ask — don't expand. The reviewer concerns are narrow and specific.

---

## Priority 0 — Stop-the-show items

These must be addressed before any version of the deck is shown to NXP. Each one would, on its own, be enough to derail the review.

### KH-P0-001 — Reconcile dual "edge BW-bound" estimates

**Severity:** Blocks NXP review.

**Why.** The same workload appears with two contradictory edge-FPS numbers in the bundle, both nominally at NPU Mid stock LPDDR5X:

- `keyhole_data_bundle.md` table 1 (and `data_bundle.json` → `ncu.workloads[].edge_projection_npu_mid`) reports `yolo_seg_fp16_trt`: **218.65 MB/forward → 2.32 ms BW-bound floor → 430 FPS**.
- `bakeoffs.trt_yolo_edge_projection.projections.720p.fp16` reports the same workload: `bandwidth_limited_ms = 52.87 ms`, `projected_fps_edge = 18.6`.

Same workload, same target tier, **22.7× discrepancy**. The ncu projection's `interpretation` field already says it's a BW floor (best case) and that real edge FPS = `min(bw_bound_fps, compute_bound_fps)` — but the bake-off projection's `bandwidth_limited_ms` doesn't appear to be derived from the same DRAM-bytes-per-forward measurement. A reviewer flipping between the two documents will see the gap and ask which number is real.

**Do.**

1. Read `scripts/profile_ncu.py`, `scripts/export_ncu_for_sizer.py`, and whichever script generates the `*_edge_projection.json` files for the bake-offs (likely `scripts/bakeoff_trt_yolo.py` or a sibling). Document the actual computation behind `bandwidth_limited_ms` in the bake-off projections.
2. Reconcile to one of three explicit framings:
   - **Option A (cleanest):** rename ncu `edge_projection_npu_mid` fields to `bw_floor_*` everywhere and rename the bake-off projection's `bandwidth_limited_ms` to `effective_edge_ms_with_overhead` or similar. Add a methodology-section reconciliation table showing both numbers per workload and explaining the gap is overhead (kernel launches, memory hierarchy effects, NMS dynamic dispatch, etc.).
   - **Option B:** if the bake-off projection's number is just wrong (e.g., it uses 5090 wall-time × something other than 16.19, or it includes overheads it shouldn't), fix it.
   - **Option C:** if the bake-off projection's number is correct and the ncu floor is misleading, remove the ncu BW-floor rows from any user-facing artifact (deck, briefing, streamlit) and keep them in JSON only with a `_internal` flag.
3. The deck and briefing must reference one number per workload per tier. If two are presented, they must be labeled as "floor" and "projected" with the relationship between them explained.

**Files (best guess).**
- `scripts/profile_ncu.py`
- `scripts/export_ncu_for_sizer.py`
- `scripts/bakeoff_trt_yolo.py` (and other `bakeoff_*.py` siblings that emit `_edge_projection.json`)
- `data/output/keyhole_data_bundle.md` § 1 (the ncu table)
- `data/output/keyhole_data_bundle.json` schema
- `docs/CLAUDE_REVIEW_BRIEFING.md` § 8 (the methodology note that says "per-forward DRAM ÷ effective BW = ms/frame" — needs to match whichever option is chosen)
- `scripts/build_deck.py` (any slide that surfaces edge FPS)
- `keyhole-sizer` repo (separate, Streamlit app — it'll consume whichever schema lands)

**Verify.**
- Grep for `bw_bound_fps_max` and `projected_fps_edge` across the repo. Each user-facing artifact should reference one or the other consistently.
- Pick three workloads (yolo_seg_fp16_trt, clip_trt, sam3_bf16_reference) and trace each from raw measurement → projection → deck slide. Numbers should be derivable end-to-end with the methodology note matching the implementation.
- Reconciliation table should land in `CLAUDE_REVIEW_BRIEFING.md` § 8.

---

### KH-P0-002 — Apply dtype gating to projection rendering (don't delete historical data)

**Severity:** Blocks NXP review.

**Why.** The current spec is "NPU Mid is INT8-only at 200 TOPS, no FP path; NPU High is FP-capable." The original spec had FP on both Mid and High, and the historical `*_edge_projection.json` files contain FP16/FP8 projections targeting NPU Mid — which under the current spec shouldn't be there at all. A reviewer who sees `projected_fps_edge: 18.6` for "FP16 on NPU Mid" will ask "wait, I thought Mid had no FP path, what's that number?"

**Do.**

1. Add a `tier_dtype_support` mapping somewhere central (sizer config or shared constants):
   ```python
   TIER_DTYPE_SUPPORT = {
       "NPU Low-LP4":     ["INT8"],
       "NPU Low-LP5X":    ["INT8"],
       "NPU Low-LP5-32b": ["INT8"],
       "NPU Mid":         ["INT8"],
       "NPU High":        ["INT8", "FP16", "BF16", "FP8"],
       "RTX 5090":        ["INT8", "FP16", "BF16", "FP8", "FP32"],
   }
   ```
2. Add a `dtype_mismatch` flag to every projection cell whose recipe dtype isn't in the target tier's supported list. The flag already exists per the briefing; verify it's actually applied to all FP-on-Mid entries.
3. **Do not delete historical FP-on-Mid run data.** Keep it in the JSON for audit trail. Add a top-level `_schema_note` field to each `_edge_projection.json` explaining: "Some entries have `dtype_mismatch=True` reflecting NPU Mid's INT8-only spec finalized 2026-04-XX. Earlier runs assumed FP-capable Mid; data is preserved for reproducibility."
4. Update `build_deck.py` and the streamlit sizer to suppress or visibly tag dtype-mismatched cells. Two acceptable behaviors:
   - **Suppression:** dtype-mismatched cells are hidden from default views, with an "include incompatible" toggle for power users.
   - **Re-projection:** dtype-mismatched cells project to NPU High (same memory subsystem, FP-capable) with a visible "deploys to High" tag. The briefing already implies this for FP8 CLIP.
5. Update the data bundle markdown table in § 1 to add a `dtype_supported_on_mid` column or footnote.

**Files (best guess).**
- `keyhole-sizer/sizer/platform_budget.py` (presumably has the tier definitions)
- `scripts/bakeoff_*.py` (whichever generate the projections)
- `scripts/build_deck.py`
- `data/output/keyhole_data_bundle.md`
- The streamlit app (separate repo `keyhole-sizer`)

**Verify.**
- Pick one historical FP16 entry projecting to NPU Mid. Confirm the JSON now has `dtype_mismatch=True`.
- Run the streamlit sizer with default settings and confirm FP recipes don't appear under NPU Mid.
- Confirm `data_bundle.md` § 1 either flags dtype-mismatch rows or visibly distinguishes Mid-deployable (INT8) workloads from Mid-incompatible (FP) ones.

---

### KH-P0-003 — Re-label recall metrics to reflect engine-self-comparison

**Severity:** Blocks NXP review (semi-major; would catch in Q&A).

**Why.** In `trt_yolo_summary.json` → `quality.720p.fp16`, the FP16 row has `n_ref: 0, n_var: 0, n_matched: 0`. FP16 was the **reference** used to compute INT8/FP8 recall. So "FP8 recall = 1.000" actually means "FP8 boxes match FP16 engine boxes perfectly" — not "FP8 matches ground-truth labels perfectly." The deck/briefing currently presents this as task-quality preservation. Open-vocab segmentation quality on out-of-distribution concepts is the actual SAM 3 capability bar, and engine-self-comparison can't measure it.

**Do.**

1. Rename the `box_recall` and `mean_matched_iou` fields anywhere they appear in user-facing artifacts to `box_recall_vs_fp16_engine` and `mean_matched_iou_vs_fp16_engine`. JSON schema update with a `schema_version` bump.
2. In every deck/briefing/methodology mention of "recall 1.000" / "preserves recall," reword to make explicit it's vs the FP16 engine, not vs ground-truth labels. Example reword:
   - Before: "FP8 preserves recall perfectly (1.000 vs FP16's 1.000)"
   - After: "FP8 produces detection sets that match the FP16 engine at IoU 0.997. This is engine-self-consistency, not ground-truth task accuracy. Open-vocab segmentation quality on novel concepts is not characterized in this study."
3. Add a paragraph to § 8 (methodology) explicitly stating: "Box recall and IoU in this study are computed engine-to-engine (FP8 / INT8 vs FP16 reference). They measure quantization preservation, not absolute task quality."
4. Soften the "Hybrid V2 matches SAM 3 capability" claim in § 4 (shipping recommendation) and TL;DR per question 8 in the briefing — the team already flagged this internally; now make the public framing match.

**Files.**
- `data/output/bakeoff/trt_yolo_summary.json` and other `*_summary.json` with quality blocks (if they have the same pattern)
- `scripts/build_deck.py`
- `docs/CLAUDE_REVIEW_BRIEFING.md` § 3.7, § 4, § 8, TL;DR
- Any deck slide referencing recall numbers

**Verify.**
- Grep for "recall" across docs/ and report sites. Every instance should clarify "vs FP16 engine" or be removed.
- TL;DR no longer asserts "recall 1.000" without the engine-self caveat.

---

### SK-P0-001 — Quarantine the broken `persona` eval category

**Severity:** Blocks NXP review.

**Why.** The `persona` category scores 0/6 for every model in the bundle: stock Qwen 7B, stock Qwen 32B, stock Mistral 7B, stock Qwen3-30B, Skippy 7B v1, Skippy 7B v4, Skippy 32B v4, Skippy Mistral v4. This is a constant -6 contribution to every headline. Either the prompt design is broken, the gold substrings can't be matched by any plausible answer, or the category measures something the substring grader can't capture. Until it's fixed, it's dead weight inflating the denominator without contributing to model differentiation.

**Do.**

1. Open `eval/prompts_v2.json` and inspect the `persona` category prompts and their gold substrings. Determine which of these is the cause:
   - Gold substrings too narrow (e.g., requiring a specific name, format, or phrasing the model has no reason to emit).
   - Prompts are open-ended in a way the substring grader can't capture.
   - The category was added speculatively and never validated.
2. Three acceptable resolutions:
   - **Fix it:** rewrite gold substrings to be matchable by any reasonable persona response. Re-run all baselines and FT candidates against the fixed category. (~1 day if the current prompts/gold are recoverable.)
   - **Drop it:** remove the category from `prompts_v2.json` and reduce eval set to 38 prompts × 3 = 114 samples. Update all headline pass rates accordingly (every score increases by ~4.5pp). Note this in a `EVAL_SET_CHANGELOG.md`.
   - **Mark it broken:** keep the prompts in the file with a `category_status: "BROKEN_CONSTANT_FAIL"` flag. Exclude from headline pass rate. Surface in per-category but not in headline.
3. Whichever option lands, **all v4-vs-baseline deltas need to be recomputed**. The deltas don't change in absolute count of passing prompts, but as percentages they shift slightly when the denominator changes.
4. Same audit for `reasoning` category. It's near-binary across models (Qwen baseline 6/6, Mistral 0/6, Llama 1/6) — that's not a graded category, that's a chain-of-thought presence detector. May not need to be dropped, but should be flagged in methodology as binary-ish.

**Files.**
- `eval/prompts_v2.json`
- `eval/run_accuracy_eval.py` (if it has logic for category status flags)
- `eval/compare_accuracy_runs.py`
- All `eval/results/acc_*.json` (will need a re-run or post-hoc filter; choose one)
- `docs/skippy-white-paper.md`, `docs/recipe-taxonomy.md`, deck slides — every headline number needs to be re-stated against the cleaned eval

**Verify.**
- Run all baselines and v4 candidates against the cleaned eval. Each headline shifts predictably (drops by 6/132 → 0/126 if dropped, recomputes to higher % if kept-and-fixed).
- New `EVAL_SET_CHANGELOG.md` documents the change with date and rationale.
- The `samples_per_prompt: 3, total: 132` notes in methodology update to reflect the new sample count.

---

### SK-P0-002 — Add variance bounds to headline pass rates

**Severity:** Blocks NXP review.

**Why.** Every pass rate in the bundle is a single point estimate from a single eval run. The two consecutive Qwen3-30B-Thinking runs in the data (`candidate-moe-thinking-v2-rag` at 09:34:42 and 09:35:25 same day) show 62.1% vs 63.6% — a 2-pass / 1.5pp swing on the same model from sampling variance alone. The headline cross-family finding is Mistral v4 −3.8pp (5 passes vs base). At 1-2σ measurement noise, a 5-pass delta is ~2-3σ above noise. A reviewer will ask whether the delta is real.

**Do.**

1. Pick five representative runs that should bound the variance well: stock Qwen 7B (passes), stock Mistral 7B (passes), Skippy 7B v4 (FT, passes), Skippy Mistral v4 (FT, fails), one stock at the boundary like Qwen 32B. For each, run the eval **5 times** with different seeds (or different sampling temperature/top-p if the eval is deterministic). Report mean ± σ.
2. Add a `variance_bounds` table or sheet to the data bundle xlsx, columns: `model`, `n_runs`, `mean_pass_rate`, `stddev_pass_rate`, `min`, `max`, `samples_per_run`.
3. Update headline tables in white-paper and briefing to report `pass_rate ± σ` for the 5 anchored runs. Other runs report point estimate with a note: "Single-run estimate. Run-to-run variance bounded by anchored runs at ±X.Xpp."
4. Re-evaluate every "delta" claim in the deck/briefing against the variance bound. Specifically:
   - Qwen 7B v4 +3.1pp — is that ≥2σ above noise?
   - Mistral v4 −3.8pp — same question.
   - 32B v4 −4.6pp — same question.
   - MoE attention-only −9.8pp — likely real (large delta) but verify.
   - MoE +router +6.0pp from MoE attention-only — likely real, verify.
5. Any delta that doesn't survive ≥2σ gets reframed as "directional, within sampling variance." Don't kill the finding — just calibrate the confidence.

**Cost.** ~5 hours of 5090 wall time for 5 runs × 5 models × ~12 min/run RAG eval (per `perf_5090` data: 12.3 sec/RAG × 132 prompts × 3 samples = ~80 min/run; could be wrong — check actual run times). Free local cost.

**Files.**
- `eval/run_accuracy_eval.py` (may need a `--seed` or `--n_repeats` flag)
- `eval/compare_accuracy_runs.py` (variance computation)
- `scripts/build_data_bundle.py` (new sheet)
- `docs/skippy-data-bundle.xlsx` (new `variance_bounds` sheet)
- `docs/skippy-white-paper.md`, `docs/recipe-taxonomy.md`, deck

**Verify.**
- New `variance_bounds` sheet has 5 rows × 5 runs each.
- Every delta claim in the deck has a confidence note ("3.1σ above noise," "within 1σ — directional only," etc.).
- The eval CLI accepts a seed parameter and re-runs are reproducible.

---

### SHARED-P0-001 — Pull or downgrade gotcha #7 / Keyhole § 5.5 (recipe-base-coupling)

**Severity:** Blocks NXP review (over-claim concentrated in one finding, propagated across both projects).

**Why.** Both the Skippy white paper (gotcha #7) and the Keyhole briefing (§ 5.5) assert "recipe transfer is base-family-coupled" based on **one** Mistral 7B v4 data point. The hypothesis is structurally suspect — Mistral required `{% generation %}` chat-template patching to enable `assistant_only_loss`, and the patched template + assistant-only loss combination may have damaged retrieval on Mistral specifically rather than reflecting a general family-coupling phenomenon. Llama-3.1 8B v4 is "predicted but not run." Without Llama or a Mistral falsification experiment, this is N=1 with an alternative explanation the team has identified but not tested.

**Do — choose one:**

**Option A (preferred): downgrade to "preliminary observation" until one of the two follow-ups lands.**

In both repos:
1. Replace "gotcha #7: recipe transfer is base-family-coupled" with: "Preliminary observation (N=1 cross-family): Mistral 7B v4 regressed −3.8pp from its base where Qwen 7B v4 lifted +3.1pp. Alternative hypothesis (Mistral chat-template patching damaged retrieval) is identified but not yet falsified. Treating as a preliminary cross-family signal until either the Mistral full-sequence-loss falsification (SK-P2-001) or Llama 8B v4 (SK-P2-002) lands."
2. In the Keyhole briefing § 5.5, since recipe-base-coupling is referenced for context but doesn't drive any silicon-tier decision, the downgrade is straightforward — just match Skippy's framing.
3. In the recipe taxonomy doc, soften the "gotcha generalizes" framing to "single-cell observation, two known follow-ups in flight."

**Option B: pull the finding entirely until a falsification or replication run lands.**

If the team prefers a clean cut over a softened claim, just remove gotcha #7 from the white paper and § 5.5 from the Keyhole briefing. Re-add when Llama v4 or Mistral full-seq-loss data is available.

**Files.**
- `docs/skippy-white-paper.md` (gotcha #7 section)
- `docs/recipe-taxonomy.md` (Mistral row, "Reading the matrix" section)
- `personal-ai-use-cases.pptx` (any slide naming the finding)
- `keyhole/docs/CLAUDE_REVIEW_BRIEFING.md` § 5.5 and TL;DR
- Any deck slide in either project that names "base-family-coupled"

**Verify.**
- Grep for "base-family-coupled" and "gotcha #7" across both repos. Each instance is either rephrased to "preliminary" or removed.
- The two follow-up tasks (SK-P2-001, SK-P2-002) are filed as P2 priorities with assignees and deadlines.

---

## Priority 1 — Methodology improvements (do before next data run)

### KH-P1-001 — Document and justify the 0.70 BW efficiency value

**Why.** § 8 of the briefing reports "BW efficiency = 0.70 uniform across all 4 NPU tier presets" with a note that earlier deck snapshots used 0.75/0.80 and this was reconciled on 2026-04-21. An NXP reviewer who works on memory controllers will ask: how was 0.70 chosen, and does the same value defensibly apply to LP4, LP5X-64bit, LP5-32bit, and LP5X-128bit? Different bus widths, different controllers, different workload patterns.

**Do.**

1. Reconstruct the 0.70 reconciliation. Was it averaged across measured points? Pulled from a vendor benchmark? Conservative estimate? Document in a new `docs/methodology/bw_efficiency_derivation.md`.
2. If 0.70 is a single-vendor or single-tier number extrapolated, state that. If it's an average, show the per-tier estimates that were averaged and the spread. If it's "we picked the most defensible single value," state that and explain why differentiation across tiers wasn't supportable.
3. State the asymmetry explicitly: 0.85 on 5090 (GDDR7 + huge L2 cache) vs 0.70 on edge LPDDR (smaller on-chip SRAM, more frequent DRAM hits). One paragraph justifying.

**Files.**
- New: `docs/methodology/bw_efficiency_derivation.md`
- Update: `docs/CLAUDE_REVIEW_BRIEFING.md` § 8 (point to the new derivation doc)

**Verify.** A reviewer who asks "where did 0.70 come from" can be pointed at the new doc and get a satisfying answer in <2 minutes.

---

### KH-P1-002 — Surface confidence badges in the streamlit sizer UI

**Why.** The 🟢/🟠/🔴 cross-class confidence badges exist in the JSON (per § 7.5: "the cross_class badge correctly flags lower confidence; measured cells replace them as bake-offs land"), but users of the streamlit sizer at `keyhole-sizer.streamlit.app` see numbers without the confidence context. The known 1.95× over-projection on Llama-8B and 2.3× pessimism on LLM decode are real calibration gaps — surfacing these to the user prevents "I tried your sizer and got X tok/s but vendor says Y" complaints.

**Do.**

1. In the streamlit app, render every cell with a colored badge indicating measurement provenance: 🟢 = vendor-anchored or directly measured, 🟠 = cross-class projection, 🔴 = analytical-fallback only.
2. Hover/click on a badge surfaces: "This cell is projected from 5090 anchor data. The cross-class projection is known to over-estimate by ~1.95× for fp16-dense bases like Llama-8B per vendor anchor at NPU Mid (37.85 tok/s vs projection 16.5 tok/s)."
3. Add a sidebar info panel: "Calibration status: <X> cells anchored to vendor data, <Y> cells projected cross-class with known ~1.95× pessimism, <Z> cells analytical fallback. Confidence reduces in that order."

**Files.**
- `keyhole-sizer/streamlit_app.py` (separate repo)
- `keyhole-sizer/sizer/cell_provenance.py` or similar

**Verify.** Open the streamlit app and pick a Llama-8B cell on NPU Mid. The 🟠 badge displays the known pessimism. A new user understands the confidence level without reading the JSON.

---

### KH-P1-003 — Compute-ceiling clamp Phase 2 for sub-5-TOPS silicon

**Why.** § 7.6 of the briefing acknowledges Phase 2 (per-tier `compute_efficiency` clamp) is deferred. The i.MX 95 ground-truth data point (32 ms / 1080p measured vs 18 ms BW-only projected = 1.7× optimistic) is the *only* edge-silicon calibration in the entire bundle, and the workloads NXP-internal will care most about are exactly the i.MX-class ones. "Mid + High remain BW-bound at the workloads that matter" is true for the workloads measured; it's not necessarily true for an NXP customer asking about i.MX 8M or older silicon.

**Do.**

1. Implement the deferred Phase 2: per-tier `compute_efficiency` field on the Hardware dataclass, defaulting to 1.0 (no clamp) on Mid/High, and 0.19 on Neutron-class per the i.MX 95 measurement. This was the "compute_efficiency = 0.19 for Neutron-class" back-solve in § 2.
2. Add a `GOPs_per_pipeline` annotation to each pipeline so the clamp can compute "compute-bound floor" alongside BW-bound floor.
3. Update edge projections to take `min(bw_bound_ms, compute_bound_ms / compute_efficiency)`.
4. **Do not** apply the clamp to Mid/High projections (it's 1.0 for them anyway, but be explicit so a future reader doesn't think NPU Mid is being clamped).
5. Add an i.MX 95 cell to the streamlit sizer with the ground-truth measurement annotated.

**Files.**
- `keyhole-sizer/sizer/platform_budget.py`
- `keyhole-sizer/sizer/hardware.py` (or similar — wherever the Hardware dataclass lives)
- `scripts/bakeoff_*.py` (for the GOPs annotation, ideally measured at bake-off time)

**Verify.** i.MX 95 yolov8n-seg projection now lands at ~31 ms (matching the 32 ms ground truth), not 18 ms. NPU Mid + High projections are unchanged.

---

### SK-P1-001 — Soften and re-scope the recipe taxonomy

**Why.** The 8-dimensional taxonomy is a useful customer-facing artifact, but as currently framed it has gaps:
- Dim 8 (hardware) isn't a recipe dimension — it's a feasibility constraint. Two cells matching on dims 1–7 should produce identical outcomes regardless of training GPU.
- Missing dim: training data quantity. The headline "32B regresses because corpus is too small" makes corpus *size* a critical variable, but it's frozen at 6,517 across all cells.
- Missing dim: random seed and data ordering. At LoRA r=64 across 6.5K examples × 2 epochs, run-to-run variance from seed alone is non-trivial. The "match on all dims → same outcome" claim needs a seed dim or a "modulo seed variance" caveat.
- Missing dim: tokenizer / chat template. The Mistral failure may live here. If it's relevant enough to be the suspect mechanism for gotcha #7, it's a dimension.

**Do.**

1. Re-scope to **6 functional dimensions**: arch class, base size, LoRA target set, loss masking, corpus shape, hyperparameters. Hardware moves to a "feasibility note" alongside.
2. Add a `corpus_size` field within dim 5 (corpus shape) — current value 6,517, with a note: "Recipe outcomes shift with corpus size; doubling corpus may unlock 32B; halving may make 7B regress like 32B did."
3. Add a `tokenizer_template` field within dim 1 or as a new dim, calling out Mistral's `{% generation %}` patch as a known gotcha.
4. Add a "Reproducibility scope" note: "Two recipes that match on all dimensions should produce equivalent outcomes **modulo seed variance, which has been bounded at ±Xpp by the variance-bounds runs in SK-P0-002**."
5. The customer-template framing in `recipe-taxonomy.md` § "Customer-template decision framework" is fine as-is; just thread the new caveats through.

**Files.**
- `docs/recipe-taxonomy.md`
- `docs/skippy-white-paper.md` (any cross-references)

**Verify.** Recipe taxonomy now has 6 functional dims + 2 feasibility/reproducibility caveats. The customer-template decision framework still works.

---

### SK-P1-002 — Add LLM-as-judge as a tertiary capability gate

**Why.** Substring grading is gameable (the team admits this). Voice + safety as compensating gates work *for the specific failure modes they catch* (verbosity, fabrication on N=3 prompts), but neither catches "wrong-but-confidently-stated answer that hits the gold tokens." For an NXP audience, the team needs at least one independent semantic check.

**Do.**

1. Add an LLM-as-judge eval that scores a held-out 50-sample subset of `prompts_v2.json` against a rubric: factual correctness, instruction-following, faithfulness to RAG context if applicable. Use Claude Sonnet 4.6 or 4.7 via Anthropic API (the user already has API access).
2. Run on **the five anchored variance-bounds models** (per SK-P0-002): Qwen 7B base, Mistral 7B base, Skippy 7B v4, Skippy Mistral v4, one boundary model. Compare LLM-judge ranking to substring-grader ranking.
3. **Acceptance check:** if LLM-judge agrees substring v4 > v3 and v4 > v1 (matching the team's intuition), substring grading is validated for this corpus. If LLM-judge disagrees with the production-ship decision, that's a real problem and the deck framing needs to change.
4. Don't make LLM-judge the primary gate — keep substring as primary for cost and speed reasons. LLM-judge is a sanity check ("triangulation" with teeth).

**Cost.** ~50 prompts × 5 models × Claude Sonnet 4.7 input/output tokens. Probably <$2 per run total.

**Files.**
- New: `eval/run_llm_judge_eval.py`
- New: `eval/judge_rubric_v1.json`
- Update: `docs/skippy-white-paper.md` (add LLM-judge section to verification framework)

**Verify.** LLM-judge results land in the bundle. Either they validate the substring rankings (acknowledge in deck) or they don't (revise framing).

---

### SK-P1-003 — Expand `made_up_peripheral` probe set

**Why.** The 14B v4 fabrication finding rests on 0/3 vs 7B v4's 9/9 on the made_up_peripheral test. Three prompts is too few to ground a "ship-smaller" decision. Different fictional peripheral types (power management vs comms vs sensor) may behave differently.

**Do.**

1. Extend the made_up_peripheral category from 3 to 9 prompts: 3 each across power-management, comms-interface, and sensor-type fictional peripherals. Keep the existing 3 to preserve backward comparison.
2. Re-run 7B v4 and 14B v4 against the expanded probe set.
3. If 14B v4 still fabricates (e.g., 0–6 of 27 vs 7B's >24 of 27), the ship-smaller decision is well-grounded. If 14B v4 holds up (e.g., 24+ of 27), revisit and consider unblock.
4. Document the expanded set as `eval/probes_made_up_peripheral_v2.json` — keep separate from the main eval prompts.

**Files.**
- `eval/probes_made_up_peripheral_v2.json` (new)
- `eval/run_accuracy_eval.py` (if it needs to ingest a separate probe file)
- `docs/skippy-white-paper.md` (the safety-gate section)

**Verify.** New probe data lands in the bundle. Ship-smaller framing is either reaffirmed at N=27 or revisited.

---

## Priority 2 — Experiments worth running

### SK-P2-001 — Mistral 7B v4 with full-sequence loss (gotcha #7 falsification)

**Why.** The team's hypothesis for the Mistral v4 retrieval damage is the `{% generation %}` chat-template patch + assistant_only_loss interaction. The cleanest falsification is to run Mistral 7B with v4 recipe but full-sequence loss (no template patch, no assistant_only). If retrieval damage disappears, the hypothesis is confirmed and gotcha #7 collapses to a Mistral-template-specific gotcha. If retrieval still tanks, family-coupling gets real evidence.

**Cost.** Local 5090, ~50 minutes wall time, $0.

**Do.**

1. Modify `training/train_lora_v4.py` (or wherever Mistral v4 was trained) to disable `assistant_only_loss` and use full-sequence loss instead.
2. Train Mistral 7B v0.3 with this recipe variant. Tag as `kyle-mistral-7b-v4-fullseq`.
3. Run against `prompts_v2.json` (post-SK-P0-001 cleanup) with same RAG config as the original Mistral v4 run.
4. Compare per-category to: (a) Mistral 7B v0.3 stock baseline, (b) original Mistral v4 (assistant-only).

**Outcomes:**
- **If full-seq lifts retrieval close to or above stock baseline:** template-patch hypothesis confirmed. Gotcha #7 narrows to "Mistral chat-template patching is gotcha-worthy; recipe transfer is not family-coupled in general." Update deck and white paper.
- **If full-seq still regresses retrieval:** family-coupling hypothesis strengthens. Run SK-P2-002 (Llama v4) before publishing.

**Files.**
- `training/train_lora_v4.py` or sibling
- `eval/results/acc_candidate-kyle-mistral-7b-v4-fullseq_*.json` (new)
- `docs/skippy-data-bundle.xlsx` (new row)

---

### SK-P2-002 — Llama-3.1 8B v4 fine-tune

**Why.** The pre-registered prediction in the white paper says Llama v4 will land at 60–63%. The Mistral falsification gives one data point on family-coupling; Llama gives a second. With both data points, gotcha #7 is either solidly confirmed (both regress) or broken (one regresses, one lifts → recipe transfer depends on something other than family).

**Cost.** Local 5090, ~50 minutes, $0.

**Do.**

1. Train Llama-3.1 8B Instruct with v4 recipe (attention-only LoRA, assistant-only loss, 6,517 examples + 100 refusal).
2. Note: Llama uses ChatML-like templates similar to Qwen; verify whether `{% generation %}` patching is needed. If yes, that's a confound — run two variants (with and without). If no, single run is fine.
3. Run against cleaned eval. Compare to stock Llama 8B baseline (already in bundle at 56.8%).
4. Update gotcha #7 / § 5.5 framing per the result.

**Outcomes:**
- **Lifts (e.g., +X pp):** family-coupling broken, Mistral was Mistral-specific. Pull gotcha #7.
- **Regresses similar magnitude to Mistral:** family-coupling supported with N=2. Strengthen gotcha #7 framing — but still note N=2 is small.
- **Plateaus near stock:** ambiguous. Run SK-P2-001 if not already, or expand to a third base.

**Files.**
- `training/train_lora_v4.py` (Llama version)
- `eval/results/acc_candidate-kyle-llama-3.1-8b-v4-*.json`
- `docs/skippy-data-bundle.xlsx` (new row)
- `docs/skippy-white-paper.md` (pre-registered prediction now has data)

---

### SK-P2-003 — MoE + (router + experts) re-run with reduced expert LoRA rank

**Why.** The current `MoE v4 + router + experts` cell over-fits at r=8 expert LoRA on 6,517 examples (rag_blog 3/3 → 0/3, rag_datasheet −4/78). The taxonomy hypothesis is "374M trainable params over-fit the corpus." A targeted falsification: reduce expert LoRA rank to r=2 or r=4, halving or quartering trainable params, and see if retrieval recovers without losing reasoning recovery.

**Cost.** RunPod H100, ~$15-25, ~5 hours.

**Do.**

1. Train Qwen3-30B-A3B with attention + router + experts LoRA, expert rank = 2 (vs current 8).
2. Run eval. Compare to existing `MoE v4 + router` (no experts, 67.4%) and `MoE v4 + router + experts r=8` (62.9%).
3. Expected outcome: somewhere between 62.9% and 67.4% if the over-fitting hypothesis is right; possibly above 67.4% if expert FFN LoRA at lower rank actually helps.

**Files.**
- `training/pod/train_moe_lora.py`
- New eval result file
- Recipe taxonomy update (new row)

---

### KH-P2-001 — One real edge-silicon anchor beyond i.MX 95

**Why.** Currently the entire Mid/High projection edifice has zero measured edge silicon. i.MX 95 (sub-5-TOPS) is the only edge anchor in the bundle, and it's the wrong tier for the deployment story. Without at least one Mid-class anchor, the 36 FPS shipping claim is "5090 wall-time × 16.19 with overheads" — defensible but not measured.

**Do.**

1. Identify a candidate Mid-class NPU silicon that ships today and can be borrowed/loaned for 1-2 weeks of testing. Vendor-published benchmarks are an acceptable second-best if hardware is unavailable.
2. Run yolo_seg INT8 TRT (or the closest equivalent that compiles for the target NPU's toolchain) and yolov8n-seg INT8.
3. Compare measured edge ms/frame to the projection. Document the calibration ratio — this is the "5090-projection vs real edge" anchor for Mid that the Llama vendor anchor was for LLM.
4. If measurement diverges from projection by >2× in either direction, that's a methodology finding worth a deck slide.

**Cost.** Hardware-dependent. May be infeasible for this review cycle; mark as "P2 deferred if no silicon access."

**Files.**
- New: `data/output/edge_anchors/<vendor>_<part>_anchor.json`
- `keyhole-sizer/sizer/platform_budget.py` (add the anchor cell)

---

## Priority 3 — Documentation and framing

### SHARED-P3-001 — Reframe the 90× headline

**Why.** Per question 6 in the Keyhole briefing, the team itself flags that "90× improvement" implies an optimization journey but the actual story is architectural replacement. The first reading risk is that an NXP audience walks away thinking "if I throw harder optimization at SAM 3 I'll also get 30 FPS" — false. The correct read is "SAM 3 cannot fit on edge memory; we identified a 549× DRAM-cheaper architectural alternative meeting the capability bar."

**Do.**

1. In TL;DR (`CLAUDE_REVIEW_BRIEFING.md` and the deck): replace "90× improvement (0.4 → 36 FPS) through 8 sequential bake-offs" with: "SAM 3 (840M params, 119 GB DRAM/forward) is bandwidth-bound on every plausible edge memory subsystem and cannot be saved by quantization. We replaced it with a Hybrid V2 pipeline (YOLO-seg + CLIP @ 1Hz) at 217 MB DRAM/forward — a **549× DRAM reduction via architectural replacement**, achieving 36 FPS on NPU Mid stock LPDDR5X."
2. The 90× number stays as a data point but isn't the headline. The 549× DRAM-reduction is the engineering win and is *measured*, not projected.
3. Re-order the deck if needed so the architectural-replacement story comes before the optimization-journey story.

**Files.**
- `docs/CLAUDE_REVIEW_BRIEFING.md` (TL;DR, § 1, § 5.7)
- `keyhole_results.pptx` (deck — reorder slides)
- `personal-ai-use-cases.pptx` (Skippy deck doesn't reference the 90× directly but should mirror the framing if it does)

---

### KH-P3-001 — OWLv2 framing for the agentic-text-prompt slot

**Why.** Per question 7 in the briefing. OWLv2 (2.82 GB DRAM/forward, 6× faster than SAM 3, retains open-vocab text prompting) is well-positioned as the SAM-3 successor for the agentic role. Hybrid V2 stays as the per-frame default. Both can coexist.

**Do.**

1. Add a deck slide / briefing section: "Agentic role recommendation — OWLv2 for text-prompted segmentation when needed." Frame as "Hybrid V2 handles per-frame open-vocab labeling; OWLv2 slots into the same on-demand 1 Hz duty-cycle slot CLIP currently uses (240 ms × ~1 query/min = 0.4% NPU duty) when text-grounded segmentation is required."
2. Don't displace Hybrid V2 — additive, not substitutive.

**Files.**
- `docs/CLAUDE_REVIEW_BRIEFING.md` § 4 (shipping recommendation)
- Deck slide for ViT alternatives

---

### KH-P3-002 — Latency budget breakdown for shipping pipeline

**Why.** "36 FPS at 720p" implies <28 ms total per frame. The deck reports YOLO ~10 ms + CLIP ~22 ms + per-frame amortization, but doesn't break down ingest, decode, queue, DB write. An NXP customer asking "where's my slack" needs the full breakdown.

**Do.**

1. Profile a real end-to-end frame on the 5090 reference platform: FFmpeg ingest → preproc → YOLO TRT → optional CLIP → SQLite + FTS5 write → response. Record per-stage ms.
2. Project to NPU Mid using the same scaling rules as the existing edge projections (BW-bound stages scale by 16.19; compute-light stages are ~constant).
3. Add a deck slide: "Latency budget per frame at 720p NPU Mid stock LPDDR5X" with stacked-bar visualization showing per-stage ms and total.
4. Identify the slack: at 36 FPS = 27 ms/frame and YOLO+CLIP@1Hz = ~22 ms, there's ~5 ms slack for ingest + DB + overhead. Make this visible.

**Files.**
- New: `scripts/profile_e2e_pipeline.py`
- New deck slide

---

### SK-P3-001 — Document the unbalanced eval set as a known limitation

**Why.** Even after fixing persona and adding variance bounds, the eval is structurally unbalanced (60% rag_datasheet). For NXP-internal review, declare this as a known limitation rather than burying it.

**Do.**

1. Add an "Eval set composition and limitations" section to the white paper and methodology sheet:
   ```
   The v2 eval set contains 132 samples across 10 categories with imbalanced
   sample counts (rag_datasheet=78, multihop+refusal=9 each, most others=6,
   rag_blog+rag_email=3 each). The headline pass rate is dominated by
   rag_datasheet performance (~60% weight). Per-category deltas in small
   categories (3-6 samples) are inherently noisy at sampling-variance scale.
   Customers replicating this recipe should rebuild the eval set against
   their own corpus shape with balanced sample sizes.
   ```
2. Recompute headline pass rates with category-balanced weighting as a *secondary* number alongside the raw count. Don't make it primary; just show both.

**Files.**
- `docs/skippy-white-paper.md`
- `docs/skippy-data-bundle.xlsx` (methodology sheet)

---

## Cross-project coordination

### Two findings live in both repos and must stay in sync.

1. **Mistral v4 finding (gotcha #7 / Keyhole § 5.5).** Per SHARED-P0-001, both must downgrade or pull. After SK-P2-001 + SK-P2-002 land, both must update consistently.
2. **5090 LLM anchors (Qwen 7B 211.7 / Mistral 239.4 / Llama 211.5 tok/s).** These match exactly between Skippy `perf_5090` sheet and Keyhole `llm_anchors_summary`. **Don't drift.** If one project re-measures, the other must update from the same source. Single source of truth: `personal-ai-framework/eval/perf_5090.json` (or wherever the canonical numbers live). Keyhole reads, doesn't fork.

### Boundary that must not be crossed

**Skippy's training-methodology findings (gotcha #7, recipe taxonomy, voice transfer) do not justify Keyhole's silicon-tier decisions.** Keyhole § 5.5 currently cites the recipe-base-coupling finding for context. That's fine. But no Keyhole NPU-tier recommendation should rest on a Skippy training-methodology finding. If a slide currently does this, refactor to remove the dependency.

### Versioning

Adopt a shared `methodology_version` field across both bundles. Bump it whenever:
- BW efficiency value changes
- 5090→edge scale factor changes
- Eval set composition changes (SK-P0-001 will trigger a bump)
- Recipe taxonomy structural changes (SK-P1-001 will trigger a bump)

Both projects' data bundles read the version from a shared source if possible, or document the version they're built against.

---

## Sequencing

Suggested order of execution:

**Week 1 (P0 stop-the-show):**
- SK-P0-001 (eval cleanup) — 1 day
- KH-P0-002 (dtype gating) — 1 day
- KH-P0-003 (recall re-labeling) — 0.5 day
- SHARED-P0-001 (gotcha #7 downgrade) — 0.5 day

**Week 1-2 (parallel):**
- KH-P0-001 (BW reconciliation) — 2-3 days
- SK-P0-002 (variance bounds) — 1 day setup + 5 hr eval wall

**Week 2 (P2 experiments, while P1 is in progress):**
- SK-P2-001 (Mistral full-seq) — 1 day
- SK-P2-002 (Llama v4) — 1 day

**Week 2-3 (P1):**
- KH-P1-001 (BW efficiency derivation) — 0.5 day
- KH-P1-002 (streamlit confidence badges) — 1 day
- KH-P1-003 (compute-ceiling Phase 2) — 1-2 days
- SK-P1-001 (recipe taxonomy re-scope) — 0.5 day
- SK-P1-002 (LLM-judge eval) — 1 day
- SK-P1-003 (probe expansion) — 0.5 day

**Week 3 (P3 framing):**
- SHARED-P3-001 (90× reframe) — 0.5 day
- KH-P3-001, KH-P3-002, SK-P3-001 — 1 day total

**Deferred / hardware-dependent:**
- KH-P2-001 (real edge anchor) — needs silicon access
- SK-P2-003 (MoE r=2 expert LoRA) — needs RunPod time

---

## Definition of done for NXP-internal review

The deck and briefing are ready when:

- [ ] No conflicting numbers exist between data bundle and edge projection JSON for the same workload at the same tier (KH-P0-001).
- [ ] Every projection cell respects the current dtype gating, with historical FP-on-Mid data preserved in JSON but suppressed from default UI rendering (KH-P0-002).
- [ ] No "recall = 1.000" claim appears without "vs FP16 engine" clarification (KH-P0-003).
- [ ] Persona category is fixed, dropped, or quarantined; all headlines re-stated against the cleaned eval (SK-P0-001).
- [ ] Five anchored runs have variance bounds; every delta in the deck is calibrated against ≥2σ (SK-P0-002).
- [ ] Gotcha #7 / Keyhole § 5.5 is downgraded to "preliminary" or pulled, with a falsification or replication run scheduled (SHARED-P0-001).
- [ ] 0.70 BW efficiency derivation document exists and is linked from the methodology section (KH-P1-001).
- [ ] Streamlit sizer surfaces confidence badges (KH-P1-002).
- [ ] Recipe taxonomy is re-scoped to 6 functional dimensions with explicit reproducibility caveats (SK-P1-001).
- [ ] 90× headline reframed to lead with the 549× DRAM reduction via architectural replacement (SHARED-P3-001).

If P2 experiments land before review (Mistral full-seq, Llama v4), update gotcha #7 framing accordingly. If they don't, ship with the downgraded "preliminary" framing.

---

## Out of scope for this remediation

- **Power, thermal, sustained-utilization throttling.** Explicitly out by reviewer note. Don't add.
- **Cross-NPU-vendor benchmarking.** The current methodology is one vendor's silicon class. Expanding to multi-vendor is a larger project.
- **Fine-tuning theory.** The recipe taxonomy is a customer artifact, not a contribution to the LoRA/RLHF/DPO design space.
- **Eval set reconstruction from scratch.** Persona/reasoning/sample-imbalance fixes are tactical. A from-scratch eval set rebuild is a separate project worth at least 2 weeks of work and is not in scope for this review cycle.

---

*Generated 2026-05-08 by an external Claude session reviewing the Keyhole + Skippy bundles. Owned by the Keyhole + Skippy / personal-ai-framework teams jointly. Update task status inline as items complete.*
