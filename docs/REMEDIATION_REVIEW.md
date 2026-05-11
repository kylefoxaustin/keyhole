# Keyhole remediation — review closure document

**Audience:** Claude (browser session) doing a second-pass review of the
keyhole project. This is a closure document for the remediation arc the
prior review initiated. Self-contained — assumes the prior session has
seen `REMEDIATION_PLAN.md` and the original `CLAUDE_REVIEW_BRIEFING.md`
but may not have read every commit since.

**Companion artifacts** (all under `kylefoxaustin/my-stuff` and on
`gdrive:skippy_files/keyhole/`):
- `CLAUDE_REVIEW_BRIEFING.md` — the original briefing, now reflecting
  the post-remediation state
- `keyhole_data_bundle.json` + `keyhole_data_bundle.md` — every
  measurement, regenerated post-remediation
- `keyhole_results.pptx` — 64 slides (was 62 pre-remediation; +1 for
  e2e latency budget, +1 for arc-credibility)
- `llm_anchors_5090.json` — canonical 5090 LLM measurement cells
- `personal-ai-framework/docs/skippy-claude-briefing.md` (sister briefing)

**Sister briefing:** the Skippy training-side has a parallel remediation
arc with its own reviewer thread. Cross-referenced throughout where the
two sides share findings (cross-family LLM perf, gotcha #7 →
substring-bias arc).

---

## TL;DR for the reviewing Claude

The prior review (`REMEDIATION_PLAN.md`, 605 lines, 4 priority tiers)
flagged P0/P1/P2/P3 items across the keyhole + Skippy + shared lanes.
**This document focuses on the keyhole-side closure.**

**What landed on the keyhole side:** 18 commits across 4 calendar days
(2026-05-08 → 2026-05-11) closing all P0 + P1 + P3 backend items + an
unanticipated 7-framing supersession arc on the shared gotcha #7 finding.

**The single most surprising outcome** wasn't on the original
remediation plan: a routine semantic-regrade methodology improvement on
the Skippy side **reversed the direction** of the production fine-tune's
headline pass-rate claim (Skippy 7B v4: substring +3.1pp → semantic
−4.8pp). The team's existing three-gate framework (capability + voice +
safety) caught this — substring failed silently, voice + safety carried
the real signal — so production decision is unaffected, but the
**campaign-level "v4 lifts capability" claim is retired** and the recipe
is now framed as voice-transfer + safety calibration only.

**The reviewer-recommended NXP-internal credibility framing** is now
load-bearing in the deck: the 7-framing arc itself ("team identified a
preliminary finding, applied increasingly rigorous methodology,
falsified one branch, refined the model, corroborated the refinement
with a pre-registered falsifier — and ultimately retired the original
headline claim under semantic regrade") is a process-narrative slide
that demonstrates self-correction discipline.

**Honest things to scrutinize in this review:**
1. Did we over-correct on the methodology side? (the deck now leads
   with caveats more than findings)
2. Did the framing-supersession churn create cross-reference rot?
   (we've revised § 5.5 seven times; have older sections drifted?)
3. Did we honor the reviewer's "don't soften" rule on the Yi −28.6pp
   number while honoring "do soften" on the family-match framing?
4. Did the 64-slide deck become too dense for the NXP-internal audience?

---

## 1. How to read this document

§ 2 lists each REMEDIATION_PLAN.md item with a one-line status. § 3
walks the 7-framing gotcha #7 supersession arc with what triggered each
reframe. § 4 lists the methodology-level findings that emerged from the
campaign (the reviewer named one as "the single most valuable
methodology output" — bigger than gotcha #7 itself). § 5 is the
honest-framing review-prompt section — specific things the reviewer
should challenge or accept.

---

## 2. REMEDIATION_PLAN.md item closure

### P0 (stop-the-show items) — all closed

| Item | Status | Commit | Notes |
|---|---|---|---|
| **KH-P0-001** Reconcile dual edge BW-bound estimates | ✅ closed | `777a139` | Renamed ncu fields to `bw_floor_*` (best-case floor); renamed bake-off projection fields to `effective_edge_ms_with_overhead`. Added per-workload reconciliation table to briefing § 8.1 + bundle MD § 4. The 22.7× yolo_seg_fp16_trt discrepancy reviewer flagged is now an explicitly-labeled overhead-ratio in the table (23.1× in our final numbers). Historical FP-on-Mid projection numbers preserved per reviewer guidance "don't delete historical data; render dtype mismatch as a flag." |
| **KH-P0-002** Apply dtype gating to projection rendering | ✅ closed | `5911f2d` | Per-recipe projection cells now carry `dtype_mismatch_on_mid` + `deployable_tiers` + `dtype_mismatch_reason`. `tier_dtype_support` matrix in bundle `__meta__`. Bundle schema bumped 2 → 3. Historical FP-on-Mid raw projection numbers preserved alongside the gating flag. |
| **KH-P0-003** Re-label recall metrics as engine-self-comparison | ✅ closed | `fe79c87` | Renamed `box_recall` / `mean_matched_iou` to `*_vs_fp16_engine` aliases (legacy fields preserved one cycle). Briefing TL;DR + § 3.7 + § 4 + § 5.2 + § 8 all carry the engine-self-comparison caveat. Briefing § 4 has a new "Capability scope" paragraph scoping the SAM-3-match claim to the embedded-world workload. Deck table headers updated to "Box recall vs FP16" / "Matched IoU vs FP16". |
| **SHARED-P0-001** Gotcha #7 downgrade | ✅ closed (7 reframes) | `892bf9c` → `e95a1e7` | This item triggered the full 7-framing supersession arc. See § 3 below for the full trajectory. Final state: "v4 lifts capability" claim retired; recipe value is voice + safety, not capability lift. |
| **SK-P0-001** Persona category quarantine | ✅ (Skippy) | n/a | Skippy-side. Persona category quarantined; basis went 132 → 126 samples. Mistral Δ shifted −3.8pp → −4.0pp post-persona; no other material shifts. |
| **SK-P0-002** Variance bounds on 5 anchored runs | ✅ (Skippy) | n/a | Skippy-side. σ values 1.4–2.8pp on temp=0.3. Confirms early Mistral cross-family Δ was ~1.7σ (below 2σ threshold); reinforced the gotcha #7 hold pending more data. Surfaced the temperature-sensitivity finding (fine-tunes lose 5–26pp going temp=0 → temp=0.3). |

### P1 (methodology improvements before next data run) — keyhole items closed

| Item | Status | Commit | Notes |
|---|---|---|---|
| **KH-P1-001** Document 0.70 BW efficiency derivation | ✅ closed | `db9c495` | New `docs/methodology/bw_efficiency_derivation.md` (255 lines, 8 sections). Honest characterization: 0.70 was a 2026-04-21 reconciliation of three drifting prior values (0.75 / 0.80 / 0.80), not a per-tier measurement. Sensitivity analysis shows 0.60 → 0.85 spread gives 1.4× FPS variation. Linked from briefing § 8. |
| **KH-P1-002** Streamlit sizer confidence badges | not mine | n/a | keyhole-sizer's lane; [sizer] session owns. |
| **KH-P1-003** Compute-ceiling clamp Phase 2 | deferred | n/a | Per remediation plan, keyhole-sizer + bake-off-scripts work. Not on the keyhole DoD list. The i.MX 95 ground-truth anchor + `measured_edge_ms` override (Phase 1) already shipped earlier. Phase 2 (per-tier `compute_efficiency` clamp + GOPs_per_pipeline annotation) requires keyhole-sizer changes; out of scope for this remediation pass. |
| **SK-P1-001/002/003** | ✅ (Skippy) | n/a | Skippy-side. Recipe taxonomy re-scoped to 6 dims, LLM-judge eval shipped, probe-set expansion done. Surfaced the cross-family findings in § 3. |

### P2 (experiments worth running)

| Item | Status | Notes |
|---|---|---|
| **SK-P2-001** Mistral 7B v4 full-sequence loss falsification | ✅ ran, broke the model | Produced 0/126 pass (unusable model). Methodology bug, not a clean falsification — reviewer noted "broken training run, root cause unknown" isn't a finding. Falsifier shelved. |
| **SK-P2-002** Llama-3.1 8B v4 fine-tune | ✅ ran, regressed | −3.2pp substring. Second non-Qwen cross-family regression. Reinforced cautious framing pending Gemma. |
| **SK-P2-003** Gemma 2 9B v4 (third cross-family) | ✅ ran, lifted (!) | +3.2pp substring. **Falsified** the "N=5 reasoning-floor discriminator" framing. Triggered supersession #5 (two-factor model: ceiling reasoning OR family-match). |
| (added) **Yi-1.5-9B-Chat v4** | ✅ ran, regressed −28.6pp | Largest substring regression in the dataset. Reviewer-named "real-world risk the team has now characterized." |
| (added) **Phi-4 v4 (pre-registered falsifier)** | ✅ ran, regressed −1.6pp | Two-factor model corroborated at N=3 cross-family. Reviewer declared gotcha #7 thread closed. |
| **KH-P2-001** Real edge-silicon anchor beyond i.MX 95 | deferred | Hardware-access dependent. Acknowledged in briefing § 8 as the binding-constraint gap for the headline 36 FPS claim. |

### P3 (documentation and framing)

| Item | Status | Commit | Notes |
|---|---|---|---|
| **SHARED-P3-001** Reframe 90× headline → 549× DRAM reduction | ✅ closed | `35446fc` | Exec-summary hero card reordered to lead with 549× (architectural replacement) instead of 90× (FPS journey). 90× preserved as downstream data point. Briefing TL;DR + § 1 + § 5.7 + summary slide all reframed. |
| **KH-P3-001** OWLv2 agentic-role framing | ✅ closed | `c9caf6f` | New briefing § 4.1 framing OWLv2 as the on-demand SAM-3-successor for text-prompted segmentation (per-frame Hybrid V2 stays the default; OWLv2 is additive). 0.4% NPU duty for typical operator pace. |
| **KH-P3-002** End-to-end pipeline latency budget | ✅ closed | `f7f8d13` | New `scripts/profile_e2e_pipeline.py` (5-stage instrumentation) + new `slide_e2e_latency_budget` (deck 62 → 63 slides). Headline finding: **CPU stages (decode + preprocess + DB) crowd out the 36 FPS budget on pure-NPU boards** (Coral-class); production SoCs with fixed-function ISP + 2D GPU close the budget at ~17.6 ms / 56 FPS sustained. |
| (added) **Arc-credibility slide** | ✅ closed | `6ec5a9d` + `e95a1e7` | Dedicated NXP-internal credibility slide showing the 7-framing supersession arc + the Skippy 7B v4 headline-erosion 5-checkpoint table + the "production decision unaffected — three-gate framework" callout. Per reviewer's "if the deck has room to surface the arc itself, that's worth doing." |

### Things NOT on REMEDIATION_PLAN.md that landed anyway

| Addition | Commit | Why |
|---|---|---|
| `data/output/llm_anchors_5090.json` canonical file | `8372b77` | Cross-repo coordination with Skippy: the 5 LLM 5090 anchors (Qwen 7B/32B, Llama 8B, Mistral 7B, MoE 30B-A3B) need to be readable from both repos. Force-added under data/output/ as a documented exception. |
| Mistral 7B v0.3 + Llama 3.1 8B 5090 LLM anchors | `4996470` | Built the bake-off harness (`scripts/bakeoff_llm_anchors.py`) before the remediation arc; the cross-family LLM perf invariance finding (170-185 tok/s within 7% across Qwen / Mistral / Llama) became one of the methodology data points referenced in the briefing § 5.4. |
| Bundle schema journey v1 → v4 | various | Each P0 item that touched a JSON layer added aliased fields rather than destructively renaming. `__meta__.schema_v{2,3,4}_changes` documents each transition. Legacy field names preserved for one cycle. |
| `methodology_version: "2026-05-08-post-remediation"` | various | Cross-repo convention with Skippy bundle. Bumps on BW efficiency change, 5090→edge scale change, eval-set composition change, or recipe taxonomy structural change. |

---

## 3. The gotcha #7 / substring-reliability arc — 7 framings in 84 hours

This is the most interesting thread of the remediation arc. The original
review flagged SHARED-P0-001 (gotcha #7 downgrade) as a P0 item with the
preferred fix being "downgrade to preliminary observation until a
falsification or replication run lands." What actually happened:

| # | When | Framing | Trigger that superseded it |
|---|---|---|---|
| 1 | 2026-05-08 evening | preliminary base-family-coupled (N=2 directional) | Llama 3.1 8B v4 landed (2nd non-Qwen regress); pattern was still <2σ individually |
| 2 | 2026-05-09 00:13 | reasoning-floor discriminator (N=5 substring-only) | Gemma 2 9B v4 LIFTED, falsifying "non-Qwen always regresses" |
| 3 | 2026-05-09 15:49 | no judge-corroborated lift in N=5 (Sonnet) | Reviewer Q1: "reasoning floor predicts substring direction, not the predictor — alternatives that correlate cannot be ruled out at N=5" |
| 4 | 2026-05-10 00:21 | 9/10 cross-judge corroborated; Gemma judge-sensitive | Yi v4 result imminent at N=6 — three 3/6 cells would test the two-factor split |
| 5 | 2026-05-10 14:20 | two-factor model (N=6 reviewer-blessed) | Phi-4 pre-registered as falsification candidate — outcome would either corroborate model or revert to Yi-specific-quirk |
| 6 | 2026-05-10 18:41 | N=7 reviewer-closed; pre-registered falsifier corroborated | Bulk semantic regrade on 33 catalog entries — substring grader's Qwen-family format bias surfaces |
| 7 | **2026-05-11 09:31** | **'v4 lifts capability' RETIRED — substring had Qwen-family format bias** | (current — reviewer-final closure of substring-reliability arc) |

**Per-supersession framing on keyhole § 5.5:**
- Each reframe was mirrored from Skippy's reviewer-blessed wording rather than independently authored on the keyhole side.
- All 7 prior framings are preserved in § 5.5's "superseded but preserved for audit" subsection as the audit trail.
- Each reframe landed within ~30 min of [docs]'s data drop (pre-staged variant scaffolding for the early ones; live-write for the later ones once the pattern was established).

**The headline-erosion arc on the production model (Skippy 7B v4)** — 5 cross-checks on the same cell, same input data, different methodologies:

| Check | Δ vs Qwen 7B base | Cumulative reading |
|---|---|---|
| Substring (original headline, temp=0) | **+3.1 pp** | apparent capability lift |
| LLM-judge Sonnet | **−0.350** | lift erases on judge dimensions |
| Temp=0.3 stochastic | **−29.3 pp** | fine-tune temperature-brittle |
| Cross-judge GPT-4o | **−0.690** | not Sonnet-specific |
| **Semantic regrade (final)** | **−4.8 pp** | **sign reversal — substring lift was Qwen-family format-fidelity artifact** |

**The production decision is unaffected** because the three-gate
framework (capability + voice + safety) was designed to catch exactly
this kind of silent-substring-failure. Skippy 7B v4 still ships per the
three gates; the semantic regrade clarifies what the recipe is FOR
(voice + safety transfer), not whether it ships.

**What the arc demonstrates** (per reviewer's NXP-internal credibility
framing recommendation, now load-bearing on the deck's arc slide):
- Team identified a preliminary finding (N=1 Mistral observation 2026-05-08)
- Applied increasingly rigorous methodology (variance bounds, LLM-judge, cross-judge, semantic regrade)
- Falsified one branch (single-factor reasoning-floor predictor, by Gemma lifting)
- Refined the model (two-factor: ceiling reasoning OR family-match)
- Corroborated the refinement with a pre-registered falsifier (Phi-4 regressed as predicted at N=6 → reviewer closure)
- Retired the original headline claim when methodology improvement reversed it (Qwen-family format bias surfaced via semantic regrade)

---

## 4. Methodology-level findings that emerged from the campaign

The remediation arc generated three methodology-level outputs the
external reviewer flagged as more valuable than the original gotcha #7
finding itself:

### 4.1 Quality metrics are engine-self-comparison, not ground-truth task accuracy (KH-P0-003)

Original briefing claimed "FP8 recall 1.000" without scoping. Reviewer
caught it. Renamed `box_recall` → `box_recall_vs_fp16_engine`,
`mean_matched_iou` → `mean_matched_iou_vs_fp16_engine`. Added explicit
methodology paragraph in briefing § 8 + § 3.7. Soft-shifted the "Hybrid
V2 matches SAM 3 capability" claim to "matches on the embedded-world
inspection workload, not a general assertion" — § 4 capability-scope
paragraph.

### 4.2 BW floor vs effective edge ms — two methodologies, different questions (KH-P0-001)

ncu reports DRAM-bytes-per-forward ÷ effective BW = pure BW floor.
Bake-off projection scales 5090 wall-time × BW ratio = overhead-inclusive.
22.7× discrepancy on yolo_seg_fp16_trt because the two methodologies were
both surfaced under names that suggested they were the same thing. Renamed
to make the floor framing explicit + added per-workload reconciliation
table to briefing § 8.1. The 23.1× overhead-ratio row is now the explicit
demonstration of the gap.

### 4.3 Substring grader is unreliable for fine-tune evaluation (5-regime matrix)

**Per the reviewer's 2026-05-11 09:31 closure: this is the single most
valuable methodology output of the entire campaign — more valuable than
gotcha #7, more valuable than the two-factor model.**

The 5-regime matrix (briefing § 8.2):

| Eval regime | Substring grader reliable? |
|---|---|
| Base-vs-base at temp=0 | ✓ YES — direction + magnitude both reliable |
| **Base-vs-FT at temp=0, FT-base IS family-matched to corpus source** | ✗ **NO — Qwen-family format-fidelity bias.** Substring rewards trained phrasings that match gold tokens; semantic regrade required. |
| Base-vs-FT at temp=0, FT-base is NOT family-matched | ⚠ direction only — magnitude understates damage |
| Base-vs-FT at temp>0 (stochastic sampling) | ✗ NO — fine-tune lift swings ±26 pp with temperature |
| Cross-family intermediate-reasoning FT comparison | ⚠ direction only, magnitude unreliable — Yi (−28.6pp substring) and Phi-4 (−1.6pp substring) produced **identical judge damage** at 18× substring magnitude variance |

**Practical implications:**
- Semantic-grade by default for any FT-on-family-matched-corpus deployment
- Two judges by default (Sonnet + GPT-4o) for any base-vs-FT comparison
- Don't ship on substring lift alone for any cross-family deployment
- Production decoding regime matters — evaluate at the same temperature

### 4.4 (additional finding) End-to-end CPU-stage crowding (KH-P3-002)

Profile measurements on the canonical 720p_EW_clip:
- GPU stages alone (yolo + 1Hz CLIP): 17.1 ms NPU Mid (well within 27.78 ms budget)
- CPU stages (decode + preprocess + DB INSERT): 31.8 ms (alone exceeds budget)

On pure-NPU boards (Coral, dev kits) the 36 FPS headline is crowded out
by CPU stages. On SoCs with fixed-function ISP + 2D GPU (Qualcomm,
MediaTek, NXP, Ambarella, Hailo), decode + preprocess move off-CPU and
the budget closes at ~17.6 ms / 56 FPS sustained. The integration-
architecture matters as much as the NPU spec — a real-world finding the
YOLO+CLIP-only deck headline doesn't expose without this slide.

### 4.5 (additional finding) Cross-family LLM perf invariance (§ 5.4)

7B-class dense Q4_K_M decode on 5090 is base-family-invariant within
~7%: Qwen 7B 184 / Mistral 7B 183 / Llama 8B 171 tok/s. Differences
track GGUF size (BW cost), not vendor. **Cross_class fallback over-
projects 1.95× on Llama-3.1 8B** (332.79 raw-TOPS-projection vs 171
measured) — useful methodology data point for sizer calibration.

MoE 30B-A3B (Qwen3) at 159 tok/s beats dense 32B at ~34 tok/s by 4.7×
at equivalent total params — pure active-params-per-token win. The
MoE-on-edge thesis lands empirically.

---

## 5. Things to scrutinize in this review

The original briefing's § 11 invited the reviewer to find "numbers that
don't reconcile, methodology shortcuts where confidence is overstated,
claims framed as discoveries that are actually one-config measurements."
This document narrows that to specific questions about the remediation
arc:

### 5.1 Did we over-correct?

The deck now leads with caveats (engine-self-comparison, BW floor vs
effective ms, dtype gating, substring-reliability matrix) more than
findings in places. Specific cells to scrutinize:

- **Briefing § 5.5 supersession trail (7 framings)** — does the
  recursive supersession framing convey discipline (per reviewer's
  intent) or overwhelm a reader? The arc-credibility slide is the
  intended "you don't have to read all 7" entry point; does it work?
- **Briefing § 8.2 5-regime matrix** — does the matrix actually help a
  customer pick an eval methodology, or does it bury the practical
  rule ("two judges by default + semantic-regrade by default for
  family-matched FTs")?
- **Deck `slide_e2e_latency_budget`** — does the "CPU stages crowd
  out 36 FPS on pure-NPU board" finding undermine the headline 36 FPS
  claim, or appropriately scope it?

### 5.2 Did the framing-supersession churn create cross-reference rot?

We've revised § 5.5 seven times in 84 hours. Adjacent sections that
might reference older framings:

- **§ 4 "Capability scope of this recommendation"** — added during
  KH-P0-003 (2026-05-08). References "open-vocab labeling capability"
  framing. Has the semantic-regrade finding made any specific claim
  here outdated?
- **§ 5.4 Cross-family LLM perf invariance** — added during the LLM
  anchor work (2026-05-07/08). The "production Skippy 7B v4 +3.1pp"
  claim is referenced here; the semantic regrade reverses that
  direction. § 5.4 calls this out but the framing may need a tighter
  cross-link to § 5.5.
- **§ 5.7 ncu confirms the architectural win** — added during ncu
  work; predates the remediation arc. Has the 515× / 549× DRAM
  reduction framing stayed consistent across the SHARED-P3-001 reframe?

### 5.3 Did we honor "don't soften −28.6pp" while honoring "do soften family-match"?

The reviewer was explicit at 2026-05-10 14:20 that the Yi −28.6pp number
must stay visible and uncaveated ("a customer running this recipe in
good faith could ship a model 28pp worse than the base — that's a
real-world risk the team has now characterized"). At 2026-05-11 09:31
the reviewer also said family-match framing should be softened to
"substantially overstated but doesn't go to zero."

The keyhole § 5.5 mirrors both:
- Yi −28.6pp stays prominently in the cross-family table + the deck
  verdict row, uncaveated
- Family-match framing softened to "the family-match branch is
  substantially overstated under substring grading; within-family
  signal mixed and undercharacterized at N=2"

But the deck bullet (`slide_skippy_recipe_taxonomy`) is dense — does it
land both messages clearly, or does the reframe make the Yi finding
seem less prominent?

### 5.4 Is the 64-slide deck too dense for NXP-internal?

The deck grew from 62 slides pre-remediation to 64:
- +1 from `slide_e2e_latency_budget` (KH-P3-002)
- +1 from `slide_remediation_arc_credibility` (arc credibility, post-remediation)

The arc-credibility slide is the most info-dense slide in the deck —
6-row supersession table + 5-checkpoint headline-erosion table + pull
quote + green callout. Is it scannable, or should the supersession
table be split off into a separate slide?

### 5.5 Cross-repo coherence — did anything drift?

The keyhole side mirrors Skippy's reviewer-blessed framing 7 times. The
key cross-references that could rot:
- `personal-ai-framework/docs/GOTCHA_7_RESOLUTION.md` (cited in
  briefing § 5.5 + § 8.2)
- `personal-ai-framework/docs/skippy-claude-briefing.md` (the sister
  briefing, cross-linked at the top)
- `personal-ai-framework/docs/recipe-taxonomy.md` (customer-template
  source-of-truth for the v4-capability-retirement framing)

If the reviewer detects drift between the keyhole and Skippy framings,
that's the highest-priority finding to surface.

### 5.6 Honest characterizations vs over-claims — specific cells to challenge

- **"549× DRAM reduction via architectural replacement"** (TL;DR /
  exec summary) — is this framing supportable? It's "primary forward
  vs primary forward" (217 MB shipping detector vs 119 GB SAM 3). The
  full-pipeline framing (231 MB including 1 Hz CLIP) gives 515×. Both
  shown in § 5.7; the deck headline uses 549×.
- **"56 FPS sustained on production SoCs"** (e2e latency budget) — is
  the projection-realistic projection (~17.6 ms NPU Mid with ISP +
  2D-GPU offloads) well-justified, or is it speculation?
- **"single most valuable methodology output"** for the Qwen-family
  bias finding — is this the reviewer's framing or our framing? (It's
  the reviewer's verbatim — cited inline. But appearing in our briefing
  + deck could read as self-aggrandizing if not clearly attributed.)

---

## 6. Campaign-level reading

The remediation plan flagged 22 items across P0/P1/P2/P3. The backend
lane closed 9 of those (all backend-side P0 + most P1 + most P3) plus
18 commits of additional work driven by the gotcha #7 framing
supersessions on the shared lane.

Several methodology-level outputs emerged from the rigor that the
reviewer subsequently characterized as more durable than the specific
remediation items:

- Engine-self-comparison clarification (§ 4.1) — applicable to any
  future Hybrid V2 / TRT FP8 quality reporting
- BW floor vs effective edge ms reconciliation (§ 4.2) — distinguishes
  what the deck's 36 FPS headline measures vs ncu's pure floor
- 5-regime substring-reliability matrix (§ 4.3) — portable to any team
  running fine-tuning evaluations against corpus-targeted evals.
  Reviewer's 2026-05-11 09:31 closure characterized this as "the
  single most valuable methodology output of this entire campaign."
- CPU-stage crowding finding (§ 4.4) — scopes the deck's 36 FPS claim
  to integration-architecture context, not just NPU spec
- Cross-family LLM perf invariance (§ 4.5) — 7B-class dense Q4_K_M
  decode is base-family-invariant within ~7% on 5090

The arc itself is documented across two deck slides
(`slide_methodology_arc` for the 7-framing supersession trail,
`slide_data_arc` for the production-cell headline-erosion table). Per
the reviewer's NXP-internal framing recommendation: *"Team identified
a preliminary finding, applied increasingly rigorous methodology,
falsified one branch, refined the model, corroborated the refinement
with a pre-registered falsifier — and ultimately retired the original
headline claim under semantic regrade."* That narrative is intended to
demonstrate self-correction discipline rather than make any claim
about the work's quality independent of the reviewer's external
verdict.

---

## 7. For the reviewing Claude (review prompt)

Your job is **not** to validate that the remediation work was done
correctly — that's the deliverable, and the work products speak for
themselves. Your job is to find:

1. **Cells where the remediation over-corrected.** Did we add caveats
   that obscure load-bearing findings? Are there places where the
   five-regime matrix or two-judges-by-default rule is cited as
   blocking when it should be advisory?
2. **Cross-reference rot from the 7 framing supersessions.** Did
   adjacent sections (§ 4 capability scope, § 5.4 LLM perf, § 5.7
   ncu DRAM) drift out of alignment with § 5.5's evolving framing?
3. **The 64-slide deck's information density.** Is the
   arc-credibility slide scannable for an NXP-internal reader, or
   should the 6-row supersession table + 5-checkpoint erosion table +
   pull quote + green callout be split across multiple slides?
4. **Cross-repo coherence drift.** The keyhole side mirrors
   Skippy-side framing 7 times. Has any of the cross-link content
   gone stale? Specifically check § 5.5 + § 8.2's references to
   `GOTCHA_7_RESOLUTION.md`, `skippy-claude-briefing.md`, and
   `recipe-taxonomy.md`.
5. **Honest characterizations vs over-claims** in the post-remediation
   state (see § 5.6 above for specific cells to challenge).

**Trust the data; challenge the framing.** The data bundle
(`keyhole_data_bundle.json`, 6 MB) is canonical. The briefing and the
deck are the framing of that data; both have been revised heavily
through the remediation arc, and the framing may not have caught up to
the data in every place.

If you find anything material, write it up as a `REMEDIATION_PLAN_v2.md`
in the keyhole repo root and we'll route it for closure. If you don't
find anything material — the deliverable is closed.

---

*Generated 2026-05-11 by [backend] (claude-opus-4-7 in claude-code) at
the conclusion of the 18-commit remediation arc. Cross-reference:
`REMEDIATION_PLAN.md` (the original review's plan) and
`CLAUDE_REVIEW_BRIEFING.md` (the post-remediation briefing).*

---

## Appendix: second-pass reviewer polish-list closure (2026-05-11 PM)

After this document was published, the external reviewer returned a
polish list of verify-and-fix items rather than a `REMEDIATION_PLAN_v2.md`.
Closure of each item:

### Verifications

1. **Deck grep for stale "+3.1pp v4 lifts capability":** Found one stale
   row in `slide_skippy_recipe_taxonomy` (Qwen2.5 7B Instruct stock row,
   verdict text "Confirms +3.1pp v4 fine-tune lift"). Fixed in commit
   following this addendum — reframed as the apples-to-apples Qwen 7B
   base row with substring-vs-semantic dual pass-rate and a pointer to
   the format-fidelity finding.
2. **`methodology_version` label:** was "2026-05-08-post-remediation".
   Bumped to "2026-05-11-substring-arc-closed" in both
   `export_data_bundle.py` and `export_llm_anchors_5090.py`.
3. **Yi −28.6pp prominence:** 1 instance in the deck (the verdict row);
   reviewer wanted ≥3 if prominent. Briefing has 9 mentions across §
   5.5 + § 8.2; the deck's prominence comes from the row position +
   bold formatting rather than count. Left as-is on the deck side
   because the row IS in a verdict table with the number in the Δ
   column; adding more occurrences risks looking repetitive.
4. **KH-P1-002 [sizer] confidence-badge status:** `measurement_alias`
   + dtype_mismatch flags exist in the keyhole-sizer data layer
   (per `sizer/llm_models.py`). Whether the streamlit UI surfaces the
   provenance badges (🟢 measured / 🟠 cross_class / 🔴 fallback)
   as user-visible elements is a [sizer]-side verification that the
   closure document cannot confirm from the backend lane. Surfaced
   back to [sizer] on the bus to confirm before the deliverable locks.
5. **Data-provenance audit after Qwen 14B correction:** per the
   campaign's GOTCHA_7_RESOLUTION.md Addendum, provenance audit was
   done after the Qwen 14B re-eval and confirmed all 5 other base
   JSONs are apples-to-apples (temp=0, RAG=on, prompts_v2, 132-sample
   basis). Surfaced in briefing § 5.5 as a paragraph after the Qwen
   14B data correction note.

### Framing pushbacks

6. **DRAM headline 549× → 515× as primary** (reviewer: "the
   customer-relevant number is the full pipeline — that's what
   actually ships"). Updated briefing TL;DR + § 1 + deck
   exec_summary hero card + summary slide. 549× preserved as
   methodology data point in § 5.7.
7. **"56 FPS sustained" scoping tightened** (reviewer: "the offload
   is plausible but not measured on any silicon you have"). Briefing
   § 5.8 now explicitly says "PROJECTION ONLY, not validated against
   measured silicon"; deck `slide_e2e_latency_budget` gains an amber
   caveat bullet citing KH-P2-001 as the gap.

### Things to surface

8. **0.70 BW efficiency sensitivity band near 36 FPS headline:**
   briefing TL;DR + deck exec_summary hero card now both carry the
   "±15% on 0.70 BW efficiency assumption" callout. Full sensitivity
   sweep stays in `docs/methodology/bw_efficiency_derivation.md`.
9. **KH-P2-001 deferral louder:** explicit "the headline rests on one
   measured edge silicon (i.MX 95)" paragraph added to briefing TL;DR.

### Deck split

10. **Split `slide_remediation_arc_credibility` into two slides** per
    reviewer ("nobody reads dense slides in NXP-internal meetings"):
    - `slide_methodology_arc` — supersession table only + reviewer
      pull quote + "Why this is on the deck" bullet box
    - `slide_data_arc` — headline-erosion 5-checkpoint table (gets
      its own real estate now) + larger three-gate framework callout
      + mechanism explanation bullet box
    Deck 64 → 65 slides.

### Attribution

11. **"Single most valuable methodology output" attribution:** all 6
    occurrences across briefing + deck + this document are
    explicitly quote-marked + attributed to "external reviewer" or
    "reviewer:". No reframing as our own assessment.
12. **§ 6 tone softened:** "what this remediation arc produced" →
    "campaign-level reading" with neutral framing. The
    self-correction-discipline paragraph cites the reviewer's
    framing rather than the team's.

### KH-P1-002 CONFIRMED SHIPPED (2026-05-11 12:24, [sizer] confirmation)

After this appendix was first written, [sizer] confirmed the streamlit
sizer confidence-badge UI is user-visible per cell, with code-level
evidence:

- `_render_source_banner` in `keyhole-sizer/app.py:1160`, called for both
  vision and LLM tiles at lines 1204, 1215
- All four states render via st.success / st.info / st.warning / st.error:
  - 🟢 `measured` (direct per-cell measurement, e.g. RTX 5090 bake-off
    cells, i.MX 95 production yolov8n-seg) → `st.success`
  - 🟢 `measured_anchor` (tier-level vendor anchor, e.g. Mid + Skippy
    MoE Q4 37.85 tok/s) → `st.success`
  - 🟡 `same_class_anchor` (within-family BW-scaled projection,
    e.g. Mid + LPDDR6-14) → `st.info`
  - 🟠 `cross_class` (cross-family extrapolation, where the 1.95×
    over-projection callout matters) → `st.warning`
- 🔴 `dtype_mismatch` rendered separately via `st.error` at line 893

**Reviewer's specific concern (Llama-8B at NPU Mid) verified:** with
`compute_dtype="fp16"` on `LLAMA_3_1_8B_INSTRUCT_STOCK` + Mid being
INT8-only, the cell renders the 🔴 `dtype_mismatch` banner explicitly
(not silent projection). For Llama-8B on NPU High (which supports FP),
the cell renders the 🟠 `cross_class` warning with text "projection
scales from a different silicon class via the two-floor model. Read as
directional — slope assumption breaks at class boundaries."

**KH-P1-002 status: data-layer flags + UI provenance both confirmed
shipped.** Removed from the open-items list.

### Keyhole-sizer fully closed today (2026-05-11)

[sizer] also shipped two methodology-mirror commits closing the
substring-arc on the sizer UI side:

- **`d7f082c`** — catalog substring → semantic migration (14/14
  entries on semantic pass_rate; mirrors PAI sizer e416ee0)
- **`d5e9f22`** — Finding 4 methodology surface in LLM accuracy
  expander (mirrors PAI sizer dd4ef31): 5-checkpoint headline erosion
  arc, per-family regrade Δ table, two-factor model "partially
  falsified" refinement, customer-guidance verbatim,
  three-gate-framework "production decision unaffected" callout
- **`7c58b59`** — `METHODOLOGY_VERSION = "2026-05-11-semantic-regrade-shipped"`
  constant + UI footnote (cross-app methodology_version lockstep with
  PAI sizer + personal-ai-framework)

### Cross-app methodology_version lockstep — all four LLM-eval surfaces aligned

| Surface | Label | Pattern |
|---|---|---|
| `personal-ai-framework/eval/build_sizer_bundle.py` (source of truth) | `2026-05-11-semantic-regrade-shipped` | stamps |
| PAI sizer `sizer_bundle.json` `__meta__` | `2026-05-11-semantic-regrade-shipped` | consumes from bundle |
| PAI sizer UI footnote (commit `770f6f0`, 2026-05-11 12:55) | `2026-05-11-semantic-regrade-shipped` | reads from bundle |
| keyhole-sizer UI footnote (commit `7c58b59`) | `2026-05-11-semantic-regrade-shipped` | reads from constant |
| keyhole bundle `__meta__` (separate NPU-perf concern) | `2026-05-11-substring-arc-closed` | stamps |

The keyhole bundle tracks NPU-perf/DRAM methodology (the lane this
remediation arc covers); the four LLM-eval surfaces track the
substring-arc closure on the Skippy side. Different methodology axes,
consistent labeling within each axis.

### What remains open

- **KH-P2-001 real Mid-class NPU silicon anchor** — hardware-access
  dependent, acknowledged in briefing TL;DR + § 5.8 + § 8. This is the
  one genuine remaining methodology gap; honest scoping not a
  remediation failure.

The reviewer's net assessment was "closure acknowledged from my side";
this appendix documents that the polish list was worked rather than
deferred. **KH-P1-002 confirmed shipped post-appendix, leaving KH-P2-001
as the sole remaining-open item.** If the reviewer surfaces anything
else after seeing the .pptx directly (most polish-list items they
couldn't verify without it), that's the next iteration.
