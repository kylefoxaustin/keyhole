# Keyhole plain-deck → conceptual-frame alignment plan

**Status: ALL PHASES LANDED 2026-05-17.** Plain deck dropped from
65 → 61 slides (Skippy removal), then rose to 63 slides with two
new framing slides (three-modes at slide 6; LLM identity at slide 48).
Branded deck rebuilt from current source via `pptx_template_converter`
(63 slides, NXP corporate template applied). Script renumbered +
uploaded. `CHANGES_2026-05-17.md` is the reviewer-facing summary.
Originally created when the [conceptual frame brief](#references)
landed; revised same-day when the brief's NPU-tier claims were
corrected to PAI golden; revised again when Phase A + B shipped;
revised when Phase C landed; closed end-of-day when Phase E landed.

This document scopes the work required to align the **plain Keyhole
deck** (`data/output/keyhole_results.pptx`, 65 slides) to the conceptual
frame. The **presenter script** (`docs/PRESENTER_SCRIPT.md`) was aligned
on 2026-05-17 (commit `c3df249` + the follow-up that reverted NPU-tier
framing to PAI golden). The remaining deck refactor is **smaller than
initially scoped** because the brief's NPU-tier claims were superseded
— the plain deck already matches PAI golden on tier specs.

---

## What the conceptual frame says (corrected summary)

1. **Keyhole = edge AI video analytics platform.** LLM is an optional
   feature backed by the Skippy product artifact (Qwen3-30B-A3B Q4_K_M),
   not a parallel deliverable.
2. **Skippy training content does not belong in the Keyhole deck.**
   Recipe taxonomy, methodology arc, headline erosion, sister-model
   confound, cross-family base-selection — all live in the Skippy
   product deck only.
3. **Three operational modes** must be acknowledged: vision-only,
   vision + LLM, LLM-only.
4. **LLM perf numbers must match the Skippy deck exactly.**
5. **NPU tier specs — PAI deck slide 11 is golden** (Kyle's same-day
   correction): Mid is INT8-only (200 TOPS, no FP); Mid + High share
   128-bit LPDDR5X @ 8.4 GT/s; High differentiates on compute +
   capacity + TDP, not bandwidth.

## What changed vs initial plan

The initial draft of this plan (also dated 2026-05-17, earlier in the
day) scoped a **major NPU tier refactor**: change MEMORY_BW_GBS so
NPU High stock = LPDDR5X-11.2, re-project ~10 slides numerically,
rewrite ~12 narrations, coordinate with `[sizer]` on the twin data
structure. **All of that is now moot.** PAI golden = current plain
deck state = current `keyhole-sizer` state. No data-structure changes
needed; no numerical re-projection needed; no narration cleanup for
"Mid INT8-only" framing needed (the framing is correct).

The remaining alignment work is just the scope-and-content cleanup
items in the conceptual frame:

## Remaining scope

### Slide removals (Skippy training, ~4 slides) — ✅ DONE 2026-05-17

Removed from `build_dirty()` in `scripts/build_deck.py`:
- ~~`slide_skippy_recipe_taxonomy`~~ (was slide 49)
- ~~`slide_methodology_arc`~~ (was slide 50)
- ~~`slide_data_arc`~~ (was slide 51)
- ~~`slide_skippy_sister_confound`~~ (was slide 52)

Plain deck dropped from 65 → 61 slides. Downstream slides renumbered.
Slide-function code remains in `build_deck.py` for use by the Skippy
deck's own builder.

### Slide reframings — ✅ DONE 2026-05-17

- ~~`slide_dense_vs_moe_bandwidth`~~ — Mistral / Llama / Yi cross-family
  rows removed; kept Qwen 7B dense + Qwen 2.5 32B dense + Qwen3-30B-A3B
  MoE. Title reframed as "MoE-on-edge — why Keyhole's optional LLM
  coexists with vision". Bullets now frame the MoE win as vision+LLM
  coexistence argument rather than cross-family base-selection. Now
  slide 49 (formerly 53).

### Slide additions — ✅ DONE 2026-05-17 (Phase C)

- **`slide_operational_modes`** — dedicated three-modes slide added
  between slide_architecture and slide_sam3_reference. New slide 6
  (3-row table: vision-only / vision+LLM / LLM-only × pipeline state ×
  engineering question, plus follow-on bullets explaining what the
  framing buys and what training-side content is intentionally
  out-of-scope).
- **`slide_llm_identity`** — dedicated LLM identity reference inserted
  before slide_llm_bakeoff. New slide 48 (two-column layout: "what
  Keyhole uses" vs "what's documented elsewhere", followed by
  "why this cross-reference matters" bullets). Plain deck now
  formally cross-references the Skippy product deck rather than
  embedding the cross-reference inline in the bake-off slide
  narration.

### Branded deck divergence (out of scope for plain-deck refactor)

The branded Keyhole deck (`data/output/keyhole_deck_branded.pptx`,
2026-04-24) is now the OFF-canonical variant — it shows the
brief's pre-correction framing (Mid FP-capable, Mid 8.4 / High 11.2
GT/s, different TOPS for High). Its slide 4 table needs revising to
match PAI golden + plain deck. Separate effort; not gated on the
plain-deck cleanup above.

## Order of operations

This is a much smaller refactor than initially scoped:

1. **Phase A — Skippy removal.** Comment out the four
   `slide_skippy_*` / `slide_methodology_arc` / `slide_data_arc`
   calls in `build_dirty()`. Plain deck drops to 61 slides.
2. **Phase B — Dense-vs-MoE reframe.** Strip cross-family rows from
   `slide_dense_vs_moe_bandwidth`; reframe narration.
3. **Phase C — Three-modes + LLM identity** (optional, low-priority).
   Add dedicated slides or fold into existing ones.
4. **Phase D — Re-script.** Update `PRESENTER_SCRIPT.md` to renumber
   for the new 61-slide structure once Phase A lands. Drop the
   Section 8 skip-pointer.
5. **Phase E — Branded deck rebuild** — ✅ DONE 2026-05-17.
   Used the local `~/Documents/GitHub/pptx_template_converter` repo
   (Strategy A theme-swap: graft template's blank-layout onto source
   slides; preserve absolute positioning; remap source RGBs to theme
   slots via `mappings/keyhole_to_corporate.json`). Workflow:
   ```
   # 1. Build merge-ready plain deck (light bg PNGs, no footer/accent stripe)
   KEYHOLE_DECK_MERGE_TARGET=1 python scripts/build_deck.py
   cp data/output/keyhole_results.pptx \
       ~/Documents/GitHub/pptx_template_converter/input/keyhole_merge_ready_63.pptx
   # 2. Convert
   cd ~/Documents/GitHub/pptx_template_converter
   python convert.py \
     --input input/keyhole_merge_ready_63.pptx \
     --template template/corporate_template.pptx \
     --output output/keyhole_deck_branded_63.pptx \
     --color-map mappings/keyhole_to_corporate.json
   # 3. Copy back + restore plain deck
   cp output/keyhole_deck_branded_63.pptx ~/Documents/GitHub/keyhole/data/output/keyhole_deck_branded.pptx
   cd ~/Documents/GitHub/keyhole
   python scripts/build_deck.py    # rebuilds plain (dark) variant
   ```
   `scripts/update_branded_deck.py` is now obsolete — its in-place
   patches (slide 1 date, slide 4 anchor rows, slide 46 TRT CLIP,
   slide 54 TRT takeaways) targeted the static 58-slide branded
   variant whose data was getting stale. After a fresh rebuild from
   current source the data is already current in every slide, and
   the slide indices the patcher hardcodes don't line up with the
   63-slide structure anyway. Keep the script for reference; don't
   run it.

## Cross-session coordination needed

- `[docs]` / `[pai-sizer]` — Skippy deck deployment-related slides
  may want to add cross-references back to Keyhole now that the
  decks formally cite each other (Keyhole says "see Skippy deck for
  training story"; Skippy could say "see Keyhole deck for vision
  pipeline + edge deployment").
- `[sizer]` — no longer needs the MEMORY_BW_GBS change. The original
  alignment-plan implication that `keyhole-sizer/sizer/npu_model.py`
  needed a same-side update is now moot.

## What was *not* done on 2026-05-17

For the record — Kyle picked "script-only today + write plan" on
2026-05-17 when scoping this work. Then he corrected the NPU-tier
claim same day. Net state at end of day:
- Plain deck (`keyhole_results.pptx`) — **unchanged**. Still 65
  slides. NPU tier framing is correct per PAI golden.
- Script — **updated**, aligned to PAI golden NPU framing + brief's
  scope rules (Skippy content skipped, three-modes acknowledged, LLM
  identity referenced).
- `slide_npu_tier_specs` in `build_deck.py` — **unchanged** from
  before today's session (`git checkout` reverted the brief-driven
  draft change).

## References

- `~/.claude/projects/-home-kyle-Documents-GitHub-keyhole/memory/project_conceptual_frame.md`
  — full conceptual frame text + Kyle's 2026-05-17 NPU-tier correction.
- PAI/Skippy deck `personal-ai-use-cases.pptx` v5.9 slide 11 — golden
  source for NPU tier specs across both decks.
- `data/output/keyhole_deck_branded.pptx` slide 4 — now OFF-canonical
  (shows brief's pre-correction framing). Branded variant needs
  rebuild per Phase E above.
- `docs/PRESENTER_SCRIPT.md` — script-side alignment landed 2026-05-17
  in commit `c3df249` + follow-up tier-revert commit.
