# Keyhole plain-deck → conceptual-frame alignment plan

**Status:** deferred. Multi-day cross-session effort. Created 2026-05-17
when Kyle issued the [conceptual frame brief](#references).

This document scopes the work required to align the **plain Keyhole
deck** (`data/output/keyhole_results.pptx`, 65 slides) to the conceptual
frame. The **presenter script** (`docs/PRESENTER_SCRIPT.md`) was aligned
on 2026-05-17 as a precursor — see Closing remarks + Section 8 notes
inside the script for the script-side reference. The deck refactor is
deferred because it cascades through shared data structures, ~10
slides of numerical re-projection, and the keyhole-sizer twin repo.

---

## What the conceptual frame says (summary)

1. **NPU Mid is FP-capable** (200 BF16 / 400 INT8 / 400 FP8 TOPS) at
   128-bit LPDDR5X @ 8.4 GT/s.
2. **NPU High is on a different memory bus** (128-bit LPDDR5X @ 11.2
   GT/s, 179.2 GB/s peak — 1.33× Mid's bandwidth). 275 / 550 / 550
   TOPS. Both Mid and High are FP-capable; tier choice is BW + capacity
   + TDP, not dtype.
3. **Skippy training content does not belong in the Keyhole deck.**
   Recipe taxonomy, methodology arc, headline erosion, sister-model
   confound, cross-family base-selection — all live in the Skippy
   product deck only.
4. **Three operational modes** must be acknowledged: vision-only,
   vision + LLM, LLM-only.
5. **LLM perf numbers must match the Skippy deck exactly.**

## What the plain deck currently shows

The plain deck reflects the **2026-04-29 NPU High redirect** (sizer
commit `239aa7e`, deck source's `MEMORY_BW_GBS` dict): Mid + High
share 128-bit LPDDR5X @ 8.4 GT/s; Mid is INT8-only; High differentiates
on compute + capacity. Most slides post-pivot embed "Mid INT8-only;
FP recipes pin to High" framing throughout.

The **branded Keyhole deck** (`data/output/keyhole_deck_branded.pptx`,
2026-04-24, stale) already shows the **canonical state** per the
conceptual frame (Mid FP-capable, Mid 8.4 / High 11.2 GT/s). The
branded variant therefore predates the redirect; the redirect was
applied to source code + plain deck but never rolled out to the
branded variant. The conceptual frame restores the pre-redirect state
as the target.

The **PAI/Skippy deck** (`personal-ai-use-cases.pptx` v5.9, 28 slides)
shows "Mid INT8-only; Mid and High share 8.4 GT/s bus" framing — same
as plain Keyhole deck, not aligned to the conceptual frame. PAI/Skippy
will need its own alignment effort.

## Scope of the deferred plain-deck refactor

### Data-structure changes (cascading)

| File | Change |
|---|---|
| `scripts/build_deck.py` `MEMORY_BW_GBS` | Add `("NPU High", "LPDDR5X-11.2"): 179.2` as the High stock entry. Decide what to do with the `("NPU High", "LPDDR5X-8.4")` cell (probably remove). |
| `scripts/build_deck.py` `TIER_STOCK_MEM` | Change `"NPU High": "LPDDR5X-8.4"` → `"LPDDR5X-11.2"`. |
| `scripts/build_deck.py` comments (lines 249–253, 271–280) | Rewrite "Mid + High share SAME memory class; High differentiates on COMPUTE only" → "Mid + High on different memory classes; High differentiates on BW + compute + capacity." |
| `~/Documents/GitHub/keyhole-sizer/sizer/npu_model.py` (twin) | Mirror the same data-structure update. Cross-session coord with `[sizer]` required. |
| `~/Documents/GitHub/keyhole-sizer` Streamlit app tier selector | High tier dropdown / Custom-mode defaults should reflect 11.2 GT/s as High stock. |

### Numerical re-projection (~10 slides)

Every slide that uses `bw_ratio_5090_to_npu("NPU High", "LPDDR5X-8.4")`
or that hardcodes 134.4 GB/s for High will shift by **1.33×** when
High's stock BW changes to 179.2. Specifically affected (per grep):

- `slide_trt_clip` (line 2087) — CLIP recipe edge projections
- `slide_efficientsam3p1_textprompt` (line 2997)
- `slide_trt_yoloe26` (line 3091)
- `slide_yoloe26_onemodel` (line 3191)
- `slide_vit_alternatives` (line 3294)
- `slide_efficientsam3_community` (line 3435)
- `slide_trt_takeaways` (line 3523) — comparison table
- `slide_npu_tier_specs` (line 4243) — the tier table itself
- `slide_exec_summary` (line 848) — hero card + NPU tier comparison rows
- `slide_optimization_roadmap` (line 4092) — roadmap projections
- Any per-clip Edge NPU Projection slide that exposes a High column

For each: re-derive the edge ms and FPS using the new BW ratio; verify
the slide reads coherently with the changed numbers.

### Narration changes (~12 slides)

Every slide that contains "Mid INT8-only" / "no FP path" / "FP pins
to High" / "Mid + High share same bus" needs rewording. Per grep:

- `slide_exec_summary` (lines 864, 923, 925, 926, 933, 960–965)
- `slide_trt_yolo` (docstring 1884–1885; bullets 1935–1938)
- `slide_concurrency` (comment 2044–2049)
- `slide_trt_clip` (docstring 2090–2091; bullets 2103, 2144–2150)
- `slide_llm_duty_cycle` (line 2293)
- `slide_efficientsam3p1_textprompt` (3000–3024)
- `slide_trt_yoloe26` (3095–3117)
- `slide_yoloe26_onemodel` (3198–3220)
- `slide_vit_alternatives` (3318–3429)
- `slide_efficientsam3_community` (3438–3459)
- `slide_trt_takeaways` (3556–3584)
- `slide_npu_tier_specs` (lines 250–253 module comment, 4297–4312 bullets)

### Slide removals (Skippy training, ~4 slides)

Remove from `build_dirty()`:
- `slide_skippy_recipe_taxonomy` (current slide 49)
- `slide_methodology_arc` (current slide 50)
- `slide_data_arc` (current slide 51)
- `slide_skippy_sister_confound` (current slide 52)

This drops plain deck from 65 → 61 slides. Downstream slides
renumber. The slide-function code can stay in `build_deck.py` for now
(useful for the Skippy deck's own builder), just stop calling them
from Keyhole's main build path.

### Slide reframings

- `slide_dense_vs_moe_bandwidth` (current slide 53) — strip the
  Mistral / Llama / Yi cross-family rows; keep only Qwen 7B dense vs
  Qwen 2.5 32B dense vs Qwen3-30B-A3B MoE. Reframe narration to focus
  on MoE-on-edge thesis as a vision+LLM coexistence argument.
- `slide_architecture` (current slide 5) — fold in the three-modes
  framing as a left-column callout or right-column legend.

### Slide additions (optional)

- A dedicated **three-modes** slide between slide_architecture and
  slide_sam3_reference. Could replace existing slide 5 sublabel.
- A **LLM identity reference** slide: "Keyhole uses Skippy-product
  artifact unmodified; see Skippy deck for training story." Sits
  between current slide 48 (NPU duty-cycle) and current slide 53
  (Dense vs MoE) — minimal narrative weight, lots of cross-deck
  hygiene value.

## Order of operations

Suggested phase ordering (each phase independently committable):

1. **Phase A — Data structures.** Update `MEMORY_BW_GBS` /
   `TIER_STOCK_MEM` / module comments. Coordinate with `[sizer]` on
   the matching `npu_model.py` change so the twins stay in sync. No
   slide content changes yet.
2. **Phase B — Tier table slide.** Rewrite `slide_npu_tier_specs`
   table + bullets to the canonical state. Numerical changes start
   surfacing here.
3. **Phase C — Numerical re-projection downstream.** For each affected
   slide (the ~10 from the table above), regenerate the projections
   under the new BW ratio. Verify each slide reads coherently.
4. **Phase D — Narration cleanup.** Sweep `build_deck.py` for "Mid
   INT8-only" / "FP pins to High" / etc. and rewrite per the new
   framing. Lots of mechanical text edits.
5. **Phase E — Skippy removal + dense-vs-MoE reframe.** Remove the 4
   Skippy slide calls; reframe slide_dense_vs_moe_bandwidth.
6. **Phase F — Re-script.** Update `PRESENTER_SCRIPT.md` to renumber
   for the new 61-slide structure (drop Section 8's skip-pointer once
   the slides are gone). Re-upload to Drive.
7. **Phase G — Cross-session reconciliation.** PAI/Skippy deck v6.0
   alignment effort. NPU tier framing across PAI deck slide 11 + 13 +
   anywhere else needs the same redirect. Coordinated with `[docs]` /
   `[pai-sizer]`.

## Cross-session coordination needed

- `[sizer]` — `MEMORY_BW_GBS` mirror, tier dropdown defaults, anchor
  catalog if values change with the tier-spec change.
- `[docs]` / `[pai-sizer]` — PAI/Skippy deck alignment. The "must match
  Skippy deck exactly" rule in the conceptual frame can't be satisfied
  until both decks converge. Open question: does PAI session agree
  the new framing (Mid FP-capable + different buses) is target state,
  or do they have a reason the redirect should stay?

## What was *not* done on 2026-05-17

For the record — Kyle picked "script-only today + write plan" on
2026-05-17 when scoping this work. The slide_npu_tier_specs change to
the table itself was reverted at git checkout so the plain deck stays
internally consistent with its accumulated framing. **The plain deck
is unchanged from before the conceptual frame was issued.** Only the
script + this plan document changed.

## References

- `~/.claude/projects/-home-kyle-Documents-GitHub-keyhole/memory/project_conceptual_frame.md`
  — full conceptual frame text from Kyle.
- `data/output/keyhole_deck_branded.pptx` — slide 4 = canonical NPU
  tier specs per the conceptual frame.
- PAI/Skippy deck `personal-ai-use-cases.pptx` v5.9 — slide 11 shows
  the pre-conceptual-frame "Mid + High share 8.4 GT/s, Mid INT8-only"
  framing that the conceptual frame supersedes.
- `docs/PRESENTER_SCRIPT.md` — script-side alignment landed 2026-05-17
  in commit (this PR / next push).
