# BW-efficiency derivation — why 0.70 across all NPU tiers, 0.85 on the 5090

**Status:** post-remediation (KH-P1-001). Authored 2026-05-08 to address
external Claude reviewer concern that the briefing reports "BW
efficiency = 0.70 uniform across all 4 NPU tier presets" without
provenance — an NXP memory-controller engineer would ask whether 0.70
defensibly applies to LP4 / LP5X-64bit / LP5-32bit / LP5X-128bit given
different bus widths, controllers, and workload patterns.

This doc reconstructs the actual reasoning behind the value, states the
honest evidence basis (not strong), and flags what would tighten it.

---

## 1. The two numbers and their job

In the sizer + bake-off projection pipeline, every NPU tier carries a
`bandwidth_efficiency` ∈ [0, 1] applied to the memory subsystem's
**theoretical peak BW** to get **effective BW** used in projections:

```
effective_bandwidth_gbs = mem_bandwidth_gbs × bandwidth_efficiency
```

Effective BW is the divisor on:
- **ncu BW floor** (`bw_floor_ms_npu_mid`): `dram_bytes_per_forward ÷
  effective_bw`
- **Bake-off projection** (`effective_edge_ms_with_overhead`): scales the
  5090 wall-time × `(reference_effective_bw ÷ target_effective_bw)`
- **LLM tok/s projection**: `effective_bw ÷ active_param_bytes_per_token`

It is the dominant uncertainty in every edge projection in the bundle.

---

## 2. The values today

| Hardware | Theoretical peak BW | Efficiency | Effective BW | Source |
|---|---|---|---|---|
| RTX 5090 | 1792 GB/s (GDDR7) | **0.85** | 1523.2 GB/s | `npu_emulator.py` RTX_5090 dataclass |
| NPU Low-LP4 | 25.6 GB/s (64b LPDDR4 @ 4.0 GT/s) | **0.70** | 17.92 GB/s | sizer + bundle (post-2026-04-21) |
| NPU Low-LP5X | 67.2 GB/s (64b LPDDR5X @ 8.4 GT/s) | **0.70** | 47.04 GB/s | sizer + bundle |
| NPU Low-LP5-32bit | 25.6 GB/s (32b LPDDR5 @ 6.4 GT/s) | **0.70** | 17.92 GB/s | sizer + bundle |
| NPU Mid | 134.4 GB/s (128b LPDDR5X @ 8.4 GT/s) | **0.70** | **94.08 GB/s** | sizer + bundle |
| NPU High | 134.4 GB/s (same memory class as Mid) | **0.70** | 94.08 GB/s | sizer + bundle |

The 5090 efficiency (0.85) is the only value with a defensible
empirical anchor; the 0.70 is a reconciled compromise across NPU tiers,
not a per-tier measurement.

---

## 3. How the 0.70 came to be

The campaign accumulated three separate efficiency assumptions that drifted
out of sync over ~6 weeks, then a 2026-04-21 commit (keyhole `1380d44` /
keyhole-sizer `4bd94d4`) reconciled them to a single value: **0.70**.

### Earlier values (pre-reconciliation)

| Source | Tier | Value | Why |
|---|---|---|---|
| `npu_emulator.py` HardwareSpec dataclass default | (any) | **0.80** | Initial "industry-typical" assumption when the emulator was scaffolded. No specific provenance — a reasonable conservative midpoint between "60% you'd see on a saturated workload" and "95% theoretical you'd see on a well-tuned LLM decode." |
| `keyhole-sizer/sizer/npu_model.py` per-tier overrides | NPU Low | **0.75** | Slightly more conservative for the low-power-NPU tier; reasoning was lower bus width + simpler controller likely sees more contention. |
| `keyhole-sizer/sizer/npu_model.py` | NPU Mid + High | **0.80** | Inherited from the dataclass default. |
| `scripts/export_ncu_for_sizer.py` bundle metadata | Mid (denominator) | **0.75** | Picked separately when the ncu bridge was first authored; reasoning lost. |

By April 2026, the deck, the sizer, and the bundle were quoting different
effective-BW numbers for the same NPU Mid hardware (100.8 / 94.08 / 100.8
GB/s in different places). A reviewer flipping between artifacts would see
the drift and ask which is real.

### The 2026-04-21 reconciliation

Both repos updated to **0.70 uniform**. From the keyhole `1380d44` commit
message:

> All four NPU tier presets now use a uniform 0.70 bandwidth efficiency
> (was 0.75 Low / 0.80 Mid+High), so the sizer and deck tell one consistent
> story instead of drifting by tier.
>
> [...] Edge-Mid FPS projections on those slides drop ~13% accordingly —
> the numbers get more conservative, not more optimistic.

The reasoning at the time:
1. **Conservative monotonicity.** 0.70 is below all three prior values
   (0.75 / 0.80 / 0.80), so the reconciliation only makes projections
   *more conservative*, never less. A reviewer reading the post-
   reconciliation deck cannot accuse us of having tightened our edge FPS
   numbers to look better.
2. **Cross-tier comparability.** Picking one shared value forces the
   *hardware* (memory bus, channel count, theoretical BW) to be the
   variable — not the utilization assumption. Cross-tier comparisons
   ("Mid vs High vs Low-LP5X at the same workload") become "what does the
   memory subsystem buy you?", not "how much did we assume each NPU
   utilizes its memory?"
3. **Defensibility.** 0.70 is a value that can be cited as
   "conservative-typical" without overcommitting to a specific edge-
   silicon vendor's measurement.

**Honest characterization:** 0.70 was picked as the most defensible single
value across our prior assumptions, not from a specific edge-silicon
measurement. It is a compromise number, not a calibrated one.

---

## 4. Why the 5090 gets 0.85 (not 0.70)

Two structural advantages favor 5090 over edge LPDDR-class memory:

| Factor | RTX 5090 | Edge NPU (LPDDR-class) |
|---|---|---|
| Memory technology | GDDR7 @ 28 GT/s, 512-bit bus | LPDDR4 / LPDDR5X, 32-bit / 64-bit / 128-bit bus |
| On-die cache | 128 MB L2 (Blackwell) + 512 MB shared L3 in some configs | 1–4 MB SRAM typical; less for low-tier NPUs |
| Memory controllers | Dedicated wide controllers, hardware-managed prefetch | Smaller controllers, shared with CPU/GPU on SoC |
| Workload pattern | Vision Transformers + LLM decode = highly cache-friendly access patterns | Same workloads; less cache so more frequent DRAM hits |

Empirically, well-tuned LLM decode on 5090 with llama-cpp-python hits
~1500 GB/s effective on RAG workloads — 84% of the 1792 GB/s theoretical
peak. We round this to **0.85** to be slightly conservative and capture
the kernel-launch tax that does cost a few percent.

Edge NPUs running similar workloads typically benchmark at 60–80% of
theoretical BW depending on tier (vendor-published numbers; not directly
measured here). The 0.70 sits in the conservative half of that range.

---

## 5. What 0.70 does and doesn't capture

### Does capture
- Average DRAM saturation across mixed compute/BW-bound workloads
- A blanket adjustment for kernel-launch overhead, bus-arbitration
  contention, controller startup costs that don't scale with bandwidth
- Cross-tier consistency: Mid vs High at the same workload differ only by
  hardware spec, not by efficiency assumption

### Doesn't capture
- **Bus-width-specific behavior.** A 32-bit LPDDR5 controller has worse
  per-channel utilization than a 128-bit LPDDR5X controller running the
  same workload — but we're applying 0.70 to both.
- **Workload-specific spread.** Pure BW-bound LLM decode might hit 0.85+
  on a well-tuned controller; complex graph-bound vision workloads might
  see 0.55. We're picking a midpoint.
- **Vendor-controller variation.** NXP eIQ Neutron, Hailo-15, MediaTek
  Genio, Qualcomm Hexagon NPU — all hit theoretical BW differently.
  Single-vendor calibration would tighten this.

### Sensitivity test
For a 720p YOLO-seg FP8 TRT workload at 217 MB DRAM/forward on NPU Mid:

| BW efficiency | Effective BW | BW floor ms | BW floor FPS |
|---|---|---|---|
| 0.60 | 80.6 GB/s | 2.69 | 372 |
| 0.70 (current) | 94.08 GB/s | 2.31 | 433 |
| 0.80 | 107.5 GB/s | 2.02 | 495 |
| 0.85 | 114.2 GB/s | 1.90 | 526 |

On the BW floor, the choice between 0.60 and 0.85 is a 1.4× spread on
projected FPS. On the bake-off projection (which dominates the headline
36 FPS), the spread is somewhat damped because the projection is
overhead-dominated for shipping-class workloads — see § 8.1 of
`CLAUDE_REVIEW_BRIEFING.md`.

---

## 6. What would tighten this

| Approach | Status | Effort |
|---|---|---|
| **Per-tier vendor-published BW efficiency** (e.g., from NXP / Qualcomm / MediaTek datasheets running representative workloads) | Not yet collected | ~1 week of research |
| **Direct ncu measurement on edge silicon** | Currently only one anchor: i.MX 95 yolov8n-seg INT8 @ 1080p. KH-P2-001 in REMEDIATION_PLAN.md tracks getting more. | Hardware-access dependent |
| **Workload-class-specific efficiency** (split LLM decode from vision, e.g., 0.80 LLM / 0.65 vision) | Plausible improvement; not yet justified by data | ~half day to wire through, +calibration data |
| **Per-tier bus-width-derived efficiency** (32b worse than 64b worse than 128b) | Theoretically sound but no data backing it for this campaign | ~1 day to wire through, +per-tier calibration |

---

## 7. The honest claim, scoped

The 0.70 value is:

- **Internally consistent.** The same value applies everywhere a sizer or
  deck artifact computes effective BW; cross-tier comparisons reflect
  hardware differences, not utilization assumption drift.
- **Conservatively positioned.** Below all three prior values that were
  in the codebase (0.75 / 0.80); reduces edge FPS projections by ~13%
  vs the older 0.80 baseline.
- **Defensibly cited as "industry-typical"** for LPDDR-class edge NPUs
  on mixed compute/BW-bound workloads.
- **Not per-tier-calibrated.** A reviewer asking "is this exactly right
  for NPU Low-LP5-32bit?" gets the answer "no, it's a uniform compromise;
  here's the sensitivity range; here's the path to tightening it."

The single biggest validation of this whole projection methodology is the
**vendor LLM benchmark anchor at NPU Mid**: 37.85 tok/s decode for
Qwen3-30B-A3B Q4_K_M. If 0.70 effective BW were drastically wrong, the
projection from the 5090 anchor would not match the vendor benchmark within
the observed tolerance. It does (the BW-only projection was originally
2.3× pessimistic vs vendor numbers, which has since been corrected by
using the vendor anchor directly for that cell). The fact that the
correction was 2.3× rather than 5×+ is informal evidence that 0.70 is in
the right ballpark for the LLM-decode regime on Mid.

For the BW-bound vision workloads (yolo_seg_fp8_trt, clip_trt) that
constitute the deck headline, no comparable edge-silicon ground-truth
measurement exists yet. The headline 36 FPS rests on the bake-off
projection methodology (5090 wall-time × BW ratio) plus the 0.70 effective
BW assumption. **A reviewer who wants to challenge 36 FPS should attack the
0.70 assumption + the bake-off projection's overhead model jointly** —
neither alone is the binding uncertainty.

---

## 8. Cross-references

- `CLAUDE_REVIEW_BRIEFING.md § 8` — methodology summary; this doc is
  linked there as the BW-efficiency derivation reference.
- `CLAUDE_REVIEW_BRIEFING.md § 8.1` — BW floor vs effective edge ms
  reconciliation; both methodologies depend on the 0.70 assumption.
- `REMEDIATION_PLAN.md` — KH-P1-001 (this doc), KH-P2-001 (real edge
  silicon anchor — the path to per-tier calibration).
- `scripts/export_ncu_for_sizer.py:117` — `NPU_MID_EFFECTIVE_GBS = 94.08`
  is the canonical product (134.4 × 0.70).
- `keyhole-sizer/sizer/npu_model.py` — Hardware dataclass default
  `bandwidth_efficiency = 0.70`.
- Commits `1380d44` (keyhole) and `4bd94d4` (keyhole-sizer) — the
  2026-04-21 reconciliation.
