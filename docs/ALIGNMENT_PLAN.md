# Keyhole plain-deck → conceptual-frame alignment plan

**Status:** scope reduced 2026-05-17 after Kyle's NPU-tier correction.
Multi-day cross-session effort no longer required. Created when the
[conceptual frame brief](#references) landed; revised same-day when
the brief's NPU-tier claims were corrected to PAI golden.

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

### Slide removals (Skippy training, ~4 slides)

Remove from `build_dirty()` in `scripts/build_deck.py`:
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

### Slide additions (optional)

- A dedicated **three-modes** slide between slide_architecture and
  slide_sam3_reference. Could replace existing slide 5 sublabel.
  Alternative: fold three-modes content into `slide_architecture` as
  a left-column callout. Script already covers this verbally in
  slide 5 narration.
- A **LLM identity reference** slide: "Keyhole uses Skippy-product
  artifact unmodified; see Skippy deck for training story." Sits
  between current slide 48 (NPU duty-cycle) and current slide 53
  (Dense vs MoE). Minimal narrative weight, lots of cross-deck
  hygiene value. Script already covers this verbally in slide 47.

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
5. **Phase E — Branded deck rebuild** (separate effort).
   `scripts/update_branded_deck.py` only patches 4 slides — full
   rebuild needs the corporate-template import operation that
   `update_branded_deck.py` doesn't automate. Manual NXP-template
   work; not blocked by the above.

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
