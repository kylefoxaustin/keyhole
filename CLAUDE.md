# CLAUDE.md — keyhole backend

FastAPI service + ncu profiling + bake-off harness backend behind the Keyhole
silicon-comparison story. This is **not** a sizer dashboard — sizers
(PAI sizer, keyhole-sizer) live in their own repos and consume bake-off output
from here.

## ratchet retrofit (v1.0.1) — why this is a no-op

ratchet is the shared SoC sizing engine being consolidated from the ecosystem
surfaces (phase plan in `personal-ai-framework/docs/`). Phase 4 was scheduled as
the keyhole backend retrofit, expected to mirror the two sizer retrofits
(adopt `Hardware`/`TIERS`/loader/capability from ratchet, keep
surface-specific projection). The empirical recon found that prediction
**doesn't match this repo's structure** — the duplication the design assumed
isn't really here. The full empirical reasoning lives in
[`docs/decisions/phase4-scope-recon.md`](docs/decisions/phase4-scope-recon.md);
this section captures what future contributors need to know.

**The pin is real.** `ratchet>=0.2.4,<0.3.0` is in `requirements.txt`. ratchet
is not imported anywhere in this repo today, but the dependency is wired so
that if/when something here needs canonical types or the shared anchor
schema, the import is available. The `<0.3.0` upper bound is the same
discipline the sizers use — ratchet bumps to v0.3.0 are deliberate
consumer-side opt-ins, not auto-upgrades.

### Why we keep our own `HardwareSpec` (not ratchet's `Hardware`)

[`src/emulate/npu_emulator.py`](src/emulate/npu_emulator.py) defines
`HardwareSpec` — a thinner dataclass with:
- silicon facts (`tops_bf16` / `tops_int8` / **`tops_int4`** / `mem_*`)
- two efficiency factors + TDP
- two derived properties (`effective_tops_bf16`, `effective_bandwidth_gbs`)
- `to_dict()`

…and **two instances total** (`RTX_5090` + `EDGE_MPU_TARGET`). It's not a
tier ladder, has no capability classifier, no calibration provenance, no
measurement-attachment fields, no tier-family taxonomy. The associated method
`project_workload()` projects an *ncu-profiled* workload from the 5090
baseline to an edge MPU — a different abstraction from the sizers'
BW/compute-floor LLM cascade.

Three concrete divergences from ratchet's `Hardware` that matter:

1. **Field-name mismatch**: keyhole backend uses `tops_*`; ratchet uses
   `peak_tops_*`. Adopting ratchet would rename every reference.
2. **INT4 vs FP8 dtype-coverage gap**: keyhole backend tracks `tops_int4`
   (real silicon need for bake-offs); ratchet has `peak_tops_fp8` but no
   INT4 field. Adopting ratchet would either drop INT4 here (loss of
   coverage) or require an engine change to add it — which the rule-of-three
   discipline says wait for: ratchet adds INT4 when ≥2 surfaces need it,
   not for one.
3. **15 unused runtime fields**: ratchet's `Hardware` carries
   `tier_family`, `compute_util_factor`, `llm_prefill_util_factor`,
   `capability_levels`, `calibration_source`, `measured_decode_overrides`,
   `measured_llm`, etc. — none of which `project_workload()` uses. Adopting
   ratchet just to leave 15 fields `None` is bloat, not consolidation.

### Why we keep our own `tomllib` anchor loader (not ratchet's)

[`src/anchors/private_anchors.py`](src/anchors/private_anchors.py) loads
private anchor secrets via `tomllib` directly — used at **build time** by
`scripts/build_deck.py --include-private` and by ncu / bake-off scripts that
run **outside Streamlit**.

ratchet's anchor loader uses `st.secrets` (with a
`try/except ImportError: st = None` guard for headless installability — but
the guard makes every lookup return `None`, useful only at the Streamlit
boundary). keyhole backend genuinely needs values outside Streamlit, so a
returns-None loader doesn't work here.

The two loaders **share the schema** (intentional cross-project compatibility,
documented in `personal-ai-framework/docs/private_anchor_secrets_spec.md`),
but they serve structurally different runtime contexts. This is not a
duplicate to consolidate.

### Future-revisit hook

If a future surface emerges with the same emulator-domain needs (ncu-profiled
workload projection, INT4 dtype, build-time non-Streamlit anchor loading), the
rule-of-three triggers and the right move is to add INT4 to ratchet, add a
tomllib loader variant, and revisit this retrofit. Until then, the divergence
is the right answer — and the pin keeps the option open.

## Cumulative migration shape (after phase 4)

- **Phase 2:** PAI sizer v1.1.0 — Option C, adopt ratchet at the
  Hardware/TIERS/loader/capability layer. Substantial retrofit.
- **Phase 3:** keyhole-sizer v1.1.0 — Option C analog, full Hardware adoption
  + surface-side `capability_level`/`_measured_edge_ms` adapters. Substantial.
- **Phase 4:** keyhole backend v1.0.1 — *this*. Documented divergence, no
  code retrofit. Acknowledgment + pin.
- **Phase 5:** Skippy framework — next.

The pattern: *the engine sharpened during the sizer retrofits, then stabilized*.
Phase 4 didn't sharpen the engine because phase 4 didn't consume it. The
total consolidation is smaller than the design predicted, which is the
correct shape — ratchet's abstraction targets the two surfaces that actually
had the duplication; the other two acknowledge it and move on.

## Running

Standard keyhole backend usage is unchanged by v1.0.1. See `README.md` /
`REPRODUCE.md` for bake-off + ncu profiling instructions.
