# Phase 4 scope review — keyhole backend retrofit

**Date:** 2026-05-23
**From:** ratchet consolidation, phase 4 (keyhole backend) read-and-acknowledge
**For:** design reviewer (read-cold; no repo access required)
**Status:** Holding — no code edits — pending this decision.

---

## 1. Where we are (for a reviewer with no context)

`ratchet` is a pure-Python edge-SoC sizing engine being consolidated from four
production surfaces. The plan runs in sequential phases, each a separate
session ending in a tagged release. Progress so far:

- **Phase 1 — DONE.** ratchet v0.2.0 → v0.2.4 (Amendments 3/4/5/6 landed across
  the retrofits: anchor loader, tier specs, memory-upgrade anchor BW-scaling,
  i.MX 95 TDP correction).
- **Phase 2 — DONE.** PAI sizer v1.1.0 (Option C: adopt ratchet for
  Hardware/TIERS/loader/capability; keep PAI's `project_llm`).
- **Phase 3 — DONE.** keyhole-sizer v1.1.0 (Option C analog, D5(i): full
  Hardware adoption + surface-side `capability_level` / `_measured_edge_ms`
  adapters; per-tier anchors re-attached at import).
- **Phase 4 — THIS.** keyhole backend (the FastAPI/bake-off service repo
  `keyhole`).
- Phases 5–6 — Skippy framework, then a future drone surface.

The design doc (§12) predicted phase 4 would mirror phases 2–3: *"Replace local
Hardware/tier definitions with ratchet imports, local capability classifiers
with ratchet.precision, and ad-hoc projection helpers with ratchet's projection
API. Tag v1.1.0, push."* My recon read found that prediction **doesn't match
keyhole backend's actual structure** — the duplication the design assumed
isn't really there. This doc lays out what I found and asks for a scope
decision.

---

## 2. Repo shape

`keyhole` is the bake-off / profiling / FastAPI service backend (~180k Python
lines, but most is the SAM3 third-party tree). Layout (depth 2):

```
src/
├── api/           FastAPI server (1,081 lines — server.py)
├── emulate/       npu_emulator.py (816 lines — the engine-equivalent)
├── anchors/       private_anchors.py (260 lines — tomllib-based loader)
├── detect/ enrich/ ingest/ profiling/ store/ render/ query/   (keyhole-domain)
└── ...
scripts/           bake-off harnesses (bakeoff_llm.py, bakeoff_fp8.py, ncu, etc.)
third_party/       sam3 multiplex models
```

Searching for ratchet-target symbols (`class Hardware` / `TIERS` /
`measured_llm_q4_decode_tok_s` / `capability_levels` / `npu_anchors` /
`tensor_native` / `sm120`) returned **exactly one file** outside third_party:
`src/emulate/npu_emulator.py`. There's also `src/anchors/private_anchors.py`.
That's the entire retrofit candidate surface.

Cross-repo "keyhole-sizer" references in scripts/ are all **text/comments**
(documentation), **not imports** — no cross-repo coupling.

---

## 3. What the two candidate files actually look like

### 3a. `src/emulate/npu_emulator.py` — `HardwareSpec`

```python
@dataclass
class HardwareSpec:
    """Hardware specification for a compute target."""
    name: str
    tops_bf16: float
    tops_int8: float
    tops_int4: float                # ← keyhole backend tracks INT4 (ratchet doesn't)
    mem_bandwidth_gbs: float
    mem_capacity_gb: float
    mem_bus_width_bits: int
    mem_type: str
    mem_data_rate_gtps: float

    compute_efficiency: float = 0.65
    bandwidth_efficiency: float = 0.80
    tdp_watts: float = 0.0

    @property
    def effective_tops_bf16(self) -> float:
        return self.tops_bf16 * self.compute_efficiency
    @property
    def effective_bandwidth_gbs(self) -> float:
        return self.mem_bandwidth_gbs * self.bandwidth_efficiency
    def to_dict(self) -> dict: ...

# Two instances total — NOT a tier ladder:
RTX_5090 = HardwareSpec(...)
EDGE_MPU_TARGET = HardwareSpec(...)
```

**Method:** `project_workload(workload: WorkloadProfile) -> EmulationResult` —
projects an **ncu-profiled** workload from the 5090 baseline to the edge MPU.

### 3b. For comparison, ratchet's `Hardware`

```python
@dataclass
class Hardware:
    name: str
    peak_tops_bf16: float           # ← different field name
    peak_tops_int8: float
    peak_tops_fp8: float             # ← ratchet has FP8, NOT int4
    mem_bandwidth_gbs: float
    mem_capacity_gb: float
    mem_bus_width_bits: int
    mem_type: str
    mem_data_rate_gtps: float
    # ── plus 15 more runtime fields keyhole backend doesn't use ──
    compute_efficiency: float = 0.65
    bandwidth_efficiency: float = 0.70
    tdp_watts: float = 0.0
    tier_family: Optional[str] = None
    compute_util_factor: float = 0.45
    llm_prefill_util_factor: float = 0.10
    llm_decode_bw_realization: float = 1.0
    compute_overhead_ms: float = 1.0
    npu_share_default: float = 0.75
    capability_levels: Optional[dict[str, CapabilityInfo]] = None
    calibration_source: Optional[CalibrationSource] = None
    bw_projected: bool = False
    stock_mem_bandwidth_gbs: Optional[float] = None
    stock_name: Optional[str] = None
    measured_decode_overrides: Optional[dict[str, float]] = None
    measured_prefill_overrides: Optional[dict[str, float]] = None
    measured_vision_overrides: Optional[dict[str, dict[str, dict[str, float]]]] = None
    measured_llm: Optional[dict[str, dict[str, dict[str, float]]]] = None
```

### 3c. `src/anchors/private_anchors.py` — tomllib loader

Its docstring says: *"Same schema as keyhole-sizer + PAI sizer — cross-project
compatible by design."* It reads `.streamlit/secrets.toml` **via `tomllib`
directly** (not `st.secrets`). It's the build-time loader used by
`scripts/build_deck.py --include-private` and ncu/bake-off scripts that run
**outside Streamlit**.

ratchet's loader uses `st.secrets` (with a `try/except ImportError: st = None`
guard for headless installability — but the guard returns `None` for every
lookup, so it's only useful at the Streamlit boundary). keyhole backend
genuinely *needs* values at build time outside Streamlit, so a returns-None
loader is useless to it.

The two loaders share the *schema* (intentional, cross-project) but serve
different runtime contexts.

---

## 4. The mismatch in plain terms

The design §12 said *"replace local Hardware/tier definitions with ratchet
imports, local capability classifiers with ratchet.precision, ad-hoc projection
helpers with ratchet's projection API."* In keyhole backend:

- There are **no tier definitions** — just two `HardwareSpec` instances (5090,
  edge target). No NPU/i.MX 95/Low-LP5* ladder. Nothing to replace with a tier
  registry.
- There is **no capability classifier** — `HardwareSpec` has no
  `capability_levels` field, no enum, no tables. Nothing to replace with
  `ratchet.precision`.
- There are **no ad-hoc projection helpers in the sizer sense** —
  `project_workload` is an ncu-driven bake-off projection (different from
  ratchet's BW/compute-floor LLM cascade), and it's the keyhole-domain
  abstraction the rest of the file (bake-off harnesses, FastAPI, ncu pipelines)
  is built around.
- The shared anchor loader is **deliberately a different runtime** (tomllib vs
  st.secrets), not a duplicate.
- INT4 vs FP8 is a real dtype-coverage gap: keyhole backend tracks INT4 (which
  ratchet has no field for), ratchet tracks FP8 (which keyhole backend doesn't).

The "duplicated engine code" the design expected to dedup *isn't actually
there*. The two surfaces that did duplicate canonical engine code (PAI sizer +
keyhole-sizer) are done. keyhole backend is a different domain — it sizes
silicon, but as part of an ncu-profiling + bake-off + FastAPI pipeline, not as
a sizer dashboard.

---

## 5. Three scope options

### (α) No-op retrofit — RECOMMEND
Acknowledge the divergence; no code adoption today. Concretely:
- Pin `ratchet>=0.2.4,<0.3.0` in keyhole's `requirements.txt` as an *optional
  future-use* dependency (so the import is available if anyone wants it later).
- Add `CLAUDE.md` documenting **why keyhole backend keeps its own
  `HardwareSpec` and tomllib loader** (different abstraction; different runtime;
  INT4 dtype; ncu-projection focus) — useful for future contributors who'd
  otherwise assume "phase 4 was skipped, do it now."
- Tag a small docs-only **v1.0.1** (or skip the tag if you'd rather not bump).
- Net effort: ~30 minutes. No risk to a production backend.

### (β) Selective adoption
If you see something concrete worth adopting (I can't identify a clear
canonical piece — schema overlap is already documented intentional, loader
runtime is different, no tier ladder exists). Falls back to (α) if nothing
emerges.

### (γ) Force-fit Option C
Map `HardwareSpec ↔ ratchet.Hardware` (rename `tops_*` → `peak_tops_*`, add
INT4 to ratchet *or* drop it from keyhole backend, port `project_workload` to
use ratchet types, possibly add tier metadata that keyhole backend doesn't use).
High effort, low payoff — keyhole backend doesn't use tier ladders / capability
tables / projection cascade / measurement attachment, which is most of
ratchet's value.

---

## 6. What I'm asking
1. **Scope ruling — (α), (β), or (γ)?** I recommend (α).
2. Under (α): do we **tag v1.0.1** (a documentation-only release that pins
   ratchet for future opt-in) or **skip the tag** and move straight to phase 5?
3. Sanity check on the bigger framing: was the design §12's phase-4 prediction
   based on an assumption that has since proven wrong (the duplicated engine
   only existed in the two *sizer* surfaces), or is there something I'm missing
   in keyhole backend that *does* duplicate ratchet?

Once you rule, I'll either execute (α) (~30 min) or move to phase 5.

---

## 7. What I will NOT do without sign-off
- No edits to `src/emulate/npu_emulator.py`, `src/anchors/private_anchors.py`,
  the FastAPI server, the bake-off scripts, or anywhere else in this repo.
- No `requirements.txt` change.
- No new file beyond this review doc.
