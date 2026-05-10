# Keyhole — independent review briefing

**Audience:** Claude (browser session) doing an independent review of the
keyhole project's findings. Self-contained — assumes no prior context.

**Sister briefing:** `personal-ai-framework/docs/skippy-claude-briefing.md`
covers the LLM training-side findings (recipe taxonomy, voice/safety/
capability gates, cross-family fine-tunes). The two briefings are
non-overlapping but share one finding (recipe-base-coupling, see § 5.5
here and gotcha #7 there) — a reviewer who reads both has a stronger case
to push than either alone.

**Companion artifacts:**

- `data/output/keyhole_data_bundle.md` — every measurement summary, rendered
  as tables. Read this for raw numbers.
- `data/output/keyhole_data_bundle.json` — same data, machine-parseable. Use
  if you want to cross-correlate fields programmatically.
- `data/output/keyhole_results.pptx` — 62-slide deck (the audience-facing
  artifact this briefing is summarizing).
- `data/output/keyhole_results.xlsx` — 22-sheet companion (one sheet per
  bake-off + index).

---

## TL;DR

**The headline framing:** SAM 3 (Meta's 840M-param open-vocab segmenter)
is bandwidth-bound on every plausible edge memory subsystem and cannot be
saved by quantization alone. We replaced it with a Hybrid V2 pipeline
(YOLO-seg + CLIP @ 1 Hz) at **549× lower DRAM per primary forward** (217
MB shipping detector vs 119,000 MB SAM 3 — measured via Nsight Compute,
not projected). The replacement achieves **36 FPS at 720p on NPU Mid
stock LPDDR5X**, vs SAM 3's 0.4 FPS edge ceiling.

**This is an architectural-replacement story, not an optimization
journey.** No amount of quantization gets SAM 3 from 0.4 FPS to 30 FPS —
a 119 GB DRAM/forward workload at 134 GB/s edge memory has a physical
floor of ~890 ms/frame regardless of bit-width. The engineering win is
recognizing the workload was on the wrong side of the BW physics and
finding a structurally lighter pipeline that meets the open-vocab
capability requirement (scoped to embedded-world inspection — see § 4).

**Headline numbers:** 549× DRAM reduction (per primary forward, ncu-
measured); 515× per full-pipeline frame including 1 Hz CLIP amortization
(231 MB total); 90× edge FPS (0.4 → 36 FPS). The 549× DRAM reduction is
the primary engineering outcome; 90× FPS is downstream.

The shipping pipeline compiles cleanly to TensorRT FP8 with negligible
quantization drift vs the FP16 reference engine
(`box_recall_vs_fp16_engine` 1.000, `mean_matched_iou_vs_fp16_engine`
0.998 for FP8; CLIP top-1 agreement vs BF16 = 0.964). **These are
engine-self-consistency metrics, not absolute task accuracy** — open-
vocab segmentation quality on novel concepts is not characterized in this
study (see § 8 methodology + § 9 question 8).

Two surprising secondary findings:

1. **FP8 and INT8 share the BW-bound edge FPS at 8-bit precision** — the
   silicon dictates which dtype deploys (Mid is INT8-only at 200 TOPS; High
   is FP-capable at 400 FP8/INT8 TOPS, both share LPDDR5X-8.4 stock memory).
   INT8 buys silicon reach but produces detections that match the FP16
   engine on only ~87.5% of boxes at 720p (low-confidence boxes drop); FP8
   matches the FP16 engine essentially perfectly. Both numbers are
   engine-self-consistency, not ground-truth recall.

2. **Recipe transfer is base-family-coupled.** Same fine-tuning recipe +
   same 6,517-example corpus on different LLM bases produces wildly
   different capability outcomes: Qwen 7B +3.1pp, Qwen 14B +5.3pp, Qwen 32B
   −4.6pp, Qwen3-30B-A3B (MoE attn-only) −9.8pp, Mistral 7B v0.3 −3.8pp.
   Voice + safety lifts (refusal, rag_email, numerical_precision) transfer
   cleanly across families; capability cost varies by base.

The ncu (NVIDIA Nsight Compute) measurements close the loop:
- **Shipping primary forward** (yolo_seg_fp8_trt) = 217 MB DRAM
- **Shipping per frame** (with CLIP @ 1 Hz amortized = 14 MB/frame) = 231 MB DRAM
- **SAM 3 baseline forward** = 118,975 MB DRAM

So **549× DRAM reduction per primary forward** (the architectural win) and
**515× per full-pipeline frame** including 1 Hz CLIP. Both numbers measured,
not projected — see § 5.7.

---

## 1. The problem

The application is an open-vocab vision pipeline for an embedded-world
device: capture frames, detect + segment objects, label them with arbitrary
text concepts (e.g. "screwdriver", "cable bundle", "rust spot"), and store
events to a queryable database with optional NLQ (natural language query)
via a local LLM.

The ML capability bar is roughly "what SAM 3 does." The deployment bar is
"runs at real-time (≥30 FPS) on a sub-$50 NPU SoC (Mid-tier silicon: ~2 W
NPU, LPDDR5X memory at 134 GB/s)."

**SAM 3 baseline:**

| Metric | Value | Source |
|---|---|---|
| Total params | 840M (vision: ~840M, text encoder: shared) | Meta release |
| Compute precision | BF16 native (RoPE-locked attention) | model card |
| Edge FPS @ 720p | **0.4 FPS** | bakeoff_sam_variants.py |
| ncu DRAM per forward | **118,975 MB (~119 GB)** | data/output/ncu/sam3_bf16_refs.json |
| 5090 wall-time @ 720p | 95 ms / forward | bakeoff_sam_variants.py |

At 119 GB DRAM per forward and 134 GB/s edge bandwidth, the BW floor alone
is **~890 ms/frame** — physically incapable of real-time at any quantization
level that preserves the BF16 attention path.

**The roadmap question:** can we get to 30 FPS while keeping the
open-vocabulary capability that makes SAM 3 valuable in the first place?

---

## 2. Hardware envelope

We model edge silicon as 5 representative tiers, all real or near-real
silicon classes that ship today:

| Tier | Memory bus | Effective BW | Compute (INT8 / BF16 / FP8) | Notes |
|---|---|---|---|---|
| **NPU Low-LP4** | 64-bit LPDDR4 @ 4.0 GT/s | 17.92 GB/s | ~50 TOPS INT8 | Cheapest credible NPU class |
| **NPU Low-LP5X** | 64-bit LPDDR5X @ 8.4 GT/s | ~37 GB/s | ~100 TOPS INT8 | 2.1× LP4 BW |
| **NPU Low-LP5-32bit** | 32-bit LPDDR5 @ 6.4 GT/s | 17.92 GB/s | 2 TOPS INT8 | NXP i.MX 95 ground truth |
| **NPU Mid** | 128-bit LPDDR5X @ 8.4 GT/s | 94.08 GB/s | **200 TOPS INT8 only** | The deployment target. **No FP path.** |
| **NPU High** | 128-bit LPDDR5X @ 8.4 GT/s | 94.08 GB/s | 200 BF16 / 400 INT8 / 400 FP8 | Same BW as Mid, FP-capable |
| **RTX 5090** (host reference) | GDDR7 @ 1792 GB/s × 0.85 eff | 1523 GB/s | ~210 TFLOPS FP16 | All edge projections start here |

**The Mid–High split matters.** Both tiers ship the same 128-bit LPDDR5X-8.4
memory, so BW-bound workloads project to the same edge FPS on either tier.
The differentiator is *dtype gating*: Mid silicon is INT8-only (200 TOPS, no
FP path); High silicon adds BF16/FP16 + FP8 native at 200/400 TOPS via
Blackwell-class doubling. A pipeline that needs FP8 (e.g. our CLIP recipe)
**dtype_mismatches** on Mid and projects to High — same BW ceiling, but
Mid would need an INT8 port (post-training quantization with activation
calibration) we do not have today.

**Bandwidth efficiency = 0.70** is uniform across all 4 NPU tier presets
(reconciled to a single value 2026-04-21; earlier deck snapshots used
0.75/0.80). 5090 efficiency = 0.85 (Blackwell GDDR7 controller).

**5090→NPU Mid BW ratio = 16.19×** is the canonical scale factor used
across all edge projections. Derivation: `(1792×0.85) / (134.4×0.70) =
1523.2 / 94.08 = 16.19`.

**Compute-ceiling clamp regime (calibrated 2026-04-22).** On weak silicon
(< ~3 TOPS INT8), workloads are compute/overhead-bound, not BW-bound.
i.MX 95 (2 TOPS Neutron NPU, 32-bit LPDDR5) measured yolov8n-seg INT8 at
**32 ms/frame @ 1080p (31 FPS)**, but BW-only sizing projected 18 FPS —
**1.7× optimistic**. Back-solve: compute_efficiency = 0.19 for Neutron-class.
The sizer now exposes a `measured_edge_ms` override (Phase 1, shipped) and
a per-tier `compute_efficiency` clamp (Phase 2, deferred). Mid + High
remain BW-bound at the workloads that matter.

---

## 3. The bake-off journey

8 sequential bake-offs landed across 6 weeks. Each is a self-contained
script under `scripts/bakeoff_*.py` with a JSON summary in
`data/output/bakeoff/<name>_summary.json` (and an
`<name>_edge_projection.json` for those that project to edge silicon).

### 3.1 Mask-model bake-off (`bakeoff_sam_variants.py`)

**Question:** can a smaller open-vocab segmenter replace SAM 3?

**Tested:** SAM 3 BF16 (reference) / EfficientSAM-Small / EfficientSAM-Tiny
/ MobileSAM (TinyViT) / YOLO11s-seg.

**Outcome:** The lightest mask-model alternatives (EfficientSAM-Small,
MobileSAM) still want 8–18 GB DRAM/forward at 720p — better than SAM 3 by
~10× but still **3–5× the BW budget of NPU Mid**. YOLO11s-seg at 620 MB/
forward is the only candidate in BW budget, but it is *not* an open-vocab
segmenter — it segments a fixed COCO vocab. **Conclusion:** mask-model
swaps alone don't close the gap. Need an architectural pivot.

### 3.2 FP8 activation quantization (`bakeoff_fp8.py`)

**Question:** does FP8 activation quantization (torchao, PerTensor) on
EfficientSAM-Small + YOLO-seg unblock edge?

**Outcome:** FP8 lands on 94/95 EfficientSAM-Small Linear layers and ~70%
of YOLO-seg Linears. **Edge gain: ~0%.** The remaining BF16 path
(unquantized Conv layers, attention) dominates BW. **Conclusion:** Linear-
only FP8 is insufficient when Conv weights still travel as FP16/BF16.

### 3.3 SmoothQuant + plain INT8 (`bakeoff_smoothquant.py`)

**Question:** does SmoothQuant fix the activation outlier problem and
unlock INT8 weight + activation quant?

**Outcome:** SmoothQuant CONVERT step blocked by torchao 0.17 incompat
(deferred upstream). Plain INT8 weight quant (without SmoothQuant) matches
FP8 for matmul throughput but doesn't touch activation BW — the same
~0% edge gain. **Conclusion:** weight-only quant doesn't help BW-bound
workloads.

### 3.4 Hybrid V2 architectural pivot (`bakeoff_hybrid_v2.py`)

**Question:** can we replace the monolithic mask-model with two specialized
models — a fast detector-segmenter (YOLO-seg, fixed COCO vocab) + a
zero-shot text-image classifier (OpenCLIP ViT-B/32, open-vocab)?

**Outcome:** YOLO-seg returns crops; CLIP scores each crop against the
user's text concepts. Open-vocab capability preserved structurally; SAM 3
quality matched on the relevant evaluation. **Edge FPS @ 720p: 16 FPS**
(BF16 baseline; YOLO ~10 ms + CLIP ~22 ms per 5 crops).

**Conclusion:** the architectural pivot works. Two specialized smaller
models beat one large open-vocab model on a BW-constrained edge target.

### 3.5 CLIP keyframe debouncing (`bakeoff_keyframe_debounce.py`)

**Question:** can we amortize CLIP cost across multiple frames since most
frames don't need a vocabulary refresh?

**Outcome:** Run CLIP at 1 Hz (every 30th frame at 30 FPS source) and
reuse labels. Quality cost is workload-dependent but small for static
scenes. **Edge FPS @ 720p: 24 FPS** with 1Hz CLIP + BF16 YOLO-seg.

### 3.6 YOLO-seg torchao Conv quant (`bakeoff_yolo_conv_quant.py`)

**Question:** can we extend torchao quant to YOLO-seg's Conv backbone via
the `swap_conv2d_1x1_to_linear` trick?

**Outcome:** 44% of Conv weights quantized; rest blocked by torchao's
1×128 block-size requirement on Conv layers. Edge gain: minor. **Conclusion:**
torchao + Conv is a tool-chain gap, not a fundamental limit.

### 3.7 TensorRT YOLO-seg FP8/INT8 (`bakeoff_trt_yolo.py`)

**Question:** does TensorRT 10.16 on Blackwell (SM 12.0) close the
torchao Conv gap?

**Outcome — the breakthrough:**

| Resolution | Recipe | 5090 ms | Edge ms (NPU Mid) | Edge FPS | Box recall vs FP16 engine | Matched IoU vs FP16 engine |
|---|---|---|---|---|---|---|
| 720p | FP16 | 3.32 | 53.7 | 18.6 | (reference) | (reference) |
| 720p | INT8 | 1.68 | 27.2 | **36.8** | **0.875** | 0.998 |
| 720p | FP8 | 1.68 | 27.2 | **36.8** | **1.000** | 0.998 |

**Two clean conclusions:**

- INT8 and FP8 share the BW-bound edge FPS at 8-bit (both ~27 ms/frame).
  The dtype choice is a *quality-preservation vs silicon-class* trade-off,
  not a speed trade-off.
- **Quality metrics here are engine-self-consistency.** Recall and matched
  IoU compare the quantized engine's output to the FP16 engine's output on
  the same input frames — they measure quantization drift, not ground-truth
  task accuracy. FP8 produces detection sets that match the FP16 engine
  perfectly at IoU 0.998. INT8 matches on 87.5% of boxes; the 12.5% drop is
  low-confidence boxes that fall below score threshold under INT8's tighter
  dynamic range. Open-vocab segmentation quality on novel concepts is not
  characterized in this study (see § 9 question 8).

### 3.8 TensorRT CLIP visual (`bakeoff_trt_clip.py`)

**Question:** can the CLIP visual tower compile cleanly to TRT FP8?

**Outcome:**

| Recipe | 5090 ms | Edge CLIP ms | Top-1 vs BF16 |
|---|---|---|---|
| BF16 (PyTorch) | 2.91 | 47.1 | 1.000 |
| FP16 (TRT) | 1.81 | 29.4 | 0.970 |
| FP8 (TRT) | 1.45 | **15.6** | **0.964** |

**FP8 halves CLIP edge cost** (29.4 → 15.6 ms) at ~0.4 pp top-1 quality
loss — noise-level. **Conclusion:** TRT FP8 closes the recipe story. The
180 MB engine fits NPU High DRAM budget without QDQ-node hand-holding —
TRT auto-selects FP8 layers.

**Caveat:** TRT FP8 is a *FP-only recipe*. NPU Mid (INT8-only) cannot
deploy it without an INT8 port. Path forward = post-training INT8
quantization with activation calibration over a representative
text-image distribution. Not in our toolchain today; would unblock the
full pipeline on Mid.

### 3.9 Multi-stream concurrency (`bakeoff_concurrency.py`)

Measured YOLO-seg FP8 TRT engine at dynamic batch {1,2,4,8,16} on the
5090, projected to edge:

| Batch | Edge ms | Per-stream ms | Streams/sec |
|---|---|---|---|
| 1 | 27.2 | 27.2 | 36.8 |
| 4 | 38.0 | 9.5 | 105 |
| 8 | 65.8 | 8.2 | 122 |

**Conclusion:** at 4 concurrent streams batched at B=4, per-stream FPS is
~26 (not 9 as naive round-robin would suggest). Batching beats serial
scheduling.

### 3.10 LLM bake-off — Qwen3-30B-A3B MoE (`bakeoff_llm.py`)

The application's NLQ subsystem queries event metadata via a local LLM.
Target model: **Qwen3-30B-A3B-Instruct-2507** (30B total / 3B active MoE,
128 experts × 8 routed). Same model the personal-ai-framework / Skippy
session uses for downstream domain fine-tuning.

**5090 measurements:**

| Quant | GGUF | Decode @256 | RAG 8K+2K decode | RAG total |
|---|---|---|---|---|
| Q4_K_M | 18.6 GB | 250 tok/s | **159 tok/s** | 16.5 s |
| Q5_K_M | 21.7 GB | 239 | 163 | 16.7 s |
| Q8_0 | 32.5 GB | 55* | 29* | 90+ s |

*Q8_0 hit partial CPU-offload (32.5 GB GGUF > 32 GB VRAM). Edge projection
reuses Q4 efficiency to avoid CPU-offload penalty that wouldn't apply on
unified-memory edge silicon.

**Vendor-published edge anchors** (NPU vendor benchmarks, authoritative):

| Tier | TTFT 1K prompt | Decode tok/s (Q4) |
|---|---|---|
| NPU Low-LP4 | 1.67 s | 29.3 |
| NPU Mid | 0.351 s | 37.85 |
| NPU High | 0.176 s | 50.5 |

Our 5090 BW-only projection was 2.3× pessimistic vs vendor numbers.
Anchored projections now use the vendor data.

### 3.11 Cross-family LLM 5090 anchors (`bakeoff_llm_anchors.py`)

The latest bake-off (2026-05-07/08, this conversation). Added 5090 anchors
for additional sizer cells: Qwen 2.5 7B/32B dense, Llama-3.1 8B Instruct,
Mistral 7B v0.3 Instruct.

**Result — cross-family RAG decode invariance:**

| Model | GGUF | RAG 8K+2K decode (5090) |
|---|---|---|
| Qwen 2.5 7B Q4_K_M | 4.68 GB | 184 tok/s |
| Mistral 7B v0.3 Q4_K_M | 4.37 GB | 183 tok/s |
| Llama-3.1 8B Q4_K_M | 4.92 GB | **171 tok/s** |
| Qwen 2.5 32B Q4_K_M | 18.5 GB | ~34 tok/s |
| Qwen3-30B-A3B Q4_K_M (MoE) | 18.6 GB | **159 tok/s** |

**Three findings:**

1. **7B-class dense Q4 is base-family-invariant within ~7%** on 5090
   (170–185 tok/s). Differences track GGUF size (BW cost), not vendor.
   Choosing 7B-class base is a *quality* decision, not a perf decision.

2. **MoE 30B-A3B beats dense 32B by 4.7×** on RAG decode despite ~same
   total params and ~same VRAM footprint. Per-token BW pays for active
   params (3B), not total params (30B). VRAM pays for total. This is
   the classic MoE bandwidth thesis, validated empirically.

3. **Cross_class projection over-projects 1.95×.** The sizer's analytical
   fallback (raw TOPS × util_factor) projected Llama-3.1 8B at 333 tok/s
   on 5090; measured was 171 — **1.95× optimistic**. The 🟠 cross_class
   badge correctly flags lower confidence; measured cells (🟢) replace
   them as bake-offs land. Methodology data point: util_factor for
   fp16-dense should drop ~50% on the RTX 5090 anchor for 7B-class
   anchorless cells.

### 3.12 ViT alternatives (`bakeoff_vit_alternatives.py`)

**Question:** can vision transformers (RT-DETR-L, DETR-ResNet50, OWLv2,
Grounding-DINO) replace the YOLO-seg + CLIP two-stage pipeline?

**Outcome — a 1-not-2 decision:**

| Variant | 5090 ms | NPU Mid FPS | DRAM/fwd | Recall vs YOLO11x |
|---|---|---|---|---|
| RT-DETR-L | 14.8 | 4.1 | 2.05 GB | 0.947 |
| DETR ResNet-50 | 10.9 | 5.7 | 2.74 GB | 0.936 |
| **OWLv2-base** | 14.8 | 4.2 | 2.82 GB | 0.926 |
| Grounding DINO Tiny | 69.9 | 0.9 | 38.5 GB | 0.782 |
| YOLO-seg (shipping) | 1.68 | 36.8 | 0.22 GB | reference |
| SAM 3 BF16 | 95 | 0.7 | 119 GB | reference |

**Camera role: don't replace.** ViT detectors are 10–13× heavier per
forward than shipping YOLO-seg. Even with TRT FP8 closing 3×, they bust
real-time at NPU Mid stock memory. *⚡LPDDR6-14 unlocks DETR ResNet-50 to
~28 FPS — within striking distance of 30 FPS.*

**Agentic role: OWLv2 is the SAM 3 successor.** 42× lighter than SAM 3
(2.82 GB vs 119 GB DRAM/forward), 6× faster, retains open-vocab text
prompting natively. Slots into the same on-demand 1 Hz duty-cycle slot
CLIP currently uses (240 ms × ~1 query/min = 0.4% NPU duty). If the
application needs per-frame agentic prompts with text-grounded
segmentation, OWLv2 replaces SAM 3.

### 3.13 EfficientSAM3 / EfficientSAM3.1 (community SAM 3 Lite watch)

Two community releases (April 2026, Apache-2.0): EfficientSAM3 ES-EV-S
(EfficientViT-B0 backbone, 424M total / 26M vision-only) and the
text-prompt-capable EfficientSAM3.1 student (106M params).

**Outcome:** 6.5× faster than SAM 3 (2.6 FPS @ 720p NPU Mid), still 13×
slower than shipping (36 FPS). **Conclusion:** the community Lite
variants are *honest follow-ups* on SAM 3 — but they cannot touch a
purpose-built two-stage pipeline on edge-native kernels.

### 3.14 YOLOE-26 one-model open-vocab (`bakeoff_trt_yoloe26.py`)

Ultralytics YOLOE-26 (Jan 2026) collapses detector + open-vocab labeler
into a single model with built-in 4585-class vocab. Tested both PyTorch
FP16 + TRT FP16 + TRT FP8.

**Outcome — negative result:**

| Recipe | 720p 5090 ms | Edge ms | Edge FPS |
|---|---|---|---|
| PyTorch FP16 | 5.3 | 86 | 11.6 |
| TRT FP16 | 4.5 | 73 | 13.7 |
| **TRT FP8** | 4.5 | 73 | 13.7 |

**FP16 → FP8 gain: ~0%.** YOLOE-26 at 16M params is **kernel-launch-bound**,
not matmul-bound. TRT FP8 helps when matmul is the bottleneck; on a small
open-vocab head with complex graph topology, it doesn't. Orthogonal win:
TRT FP8 still cuts VRAM 73% (360 → 99 MB) — useful for multi-stream.

**Decision:** stay with two-stage YOLO-seg + CLIP. The 3× gap to shipping
is structural (one-model + every-frame open-vocab vs 1 Hz CLIP debounce),
not closeable by FP8 alone.

---

## 4. Shipping recommendation

**Pipeline (NPU High-deployable):**

```
FFmpeg ingest → YOLO-seg FP8 (TRT) → CLIP FP8 (TRT) @ 1 Hz keyframe debounce
              → SQLite + FTS5 → optional Qwen3-30B-A3B MoE NLQ
```

**Pipeline (NPU Mid-deployable, no CLIP path until INT8 port):**

```
FFmpeg ingest → YOLO-seg INT8 (TRT) → SQLite + FTS5 → optional Qwen3-30B-A3B MoE NLQ
```

**Performance @ 720p single-stream NPU Mid stock LPDDR5X:**

- YOLO-seg FP8 TRT alone: **36.8 FPS**
- + CLIP FP8 every frame: **~22 FPS** (CLIP costs 15.6 ms; YOLO+CLIP = 42.8 ms)
- + CLIP @ 1 Hz: **~36 FPS** (CLIP cost amortized to 0.5 ms/frame; YOLO is the ceiling)

**Multi-stream:** 4× 720p concurrent streams batched at B=4 → 26 FPS each.
8 streams batched B=8 → 15 FPS each.

**ncu validation:** shipping per-frame DRAM = **231 MB** (yolo_seg_fp8_trt
217 MB + clip_trt 433 MB ÷ 30 frames at 1 Hz debounce = 14 MB amortized).
SAM 3 reference per forward = **118,975 MB**. **515× DRAM reduction.**

**Capability scope of this recommendation.** Hybrid V2 (YOLO-seg + CLIP)
matches SAM 3's *labeling-on-detected-regions* pattern — the YOLO-seg head
proposes boxes and masks from a fixed COCO-class detector, and CLIP scores
each crop against the user's text concepts. Open-vocab labeling capability
is preserved structurally. **What is *not* characterized in this study:**
SAM 3's open-vocab *segmentation* on novel concepts (unfamiliar object
categories where COCO-class detection misses the proposal entirely). For
applications where text-prompted segmentation of arbitrary concepts is the
binding capability requirement, see § 7 (OWLv2 as the agentic-role
successor) and the limitations note in § 9 question 8. The "Hybrid V2
matches SAM 3 capability" claim is scoped to our embedded-world
inspection workload, not a general assertion.

**LLM co-host:** Qwen3-30B-A3B Q4_K_M at 38 tok/s decode on NPU Mid (vendor
anchor) supports occasional NLQ queries (~5 s for 200 tokens). Per-frame
LLM is not viable on a busy vision NPU; duty-cycle modeling shows 1 Hz
LLM queries cost ~1 FPS on NPU Mid for short answers (200 tok), and full
RAG (8K+2K decode) obliterates vision even on NPU High — reserve RAG for
async or a second-NPU use case.

### 4.1 Agentic role recommendation — OWLv2 when text-prompted segmentation is needed

The shipping pipeline above handles **per-frame open-vocab labeling** at
real-time rates — every frame is analyzed, every detected region is
labeled against the user's vocabulary. This is the right pattern when
the application asks "what's in the scene right now?" continuously.

A different pattern shows up when the application asks "find the
[arbitrary text concept] in this scene" *on-demand* (operator query,
event-triggered analysis, periodic agentic prompt). For that role,
**OWLv2 is the recommended SAM 3 successor** — additive to Hybrid V2,
not a replacement.

**Why OWLv2:**

| Metric | OWLv2-base | SAM 3 BF16 | Shipping (yolo_seg_fp8_trt) |
|---|---|---|---|
| Total params | ~155M | 840M | 10M |
| DRAM/forward (ncu-measured) | 2.82 GB | 119 GB | 217 MB |
| 5090 ms @ 720p | 14.8 | 95 | 0.68 |
| NPU Mid edge ms (BW floor) | 30 (2.82 / 0.094) | 1265 | 2.3 |
| NPU Mid effective edge ms (with overhead) | ~240 | not deployable | ~27 |
| Open-vocab text prompting | ✓ native | ✓ native | only via CLIP labeling |
| License | Apache-2.0 | non-commercial | AGPL-3.0 (Ultralytics) |

OWLv2 is **42× lighter than SAM 3** per forward (2.82 GB vs 119 GB) and
**6× faster** on the 5090 reference. It retains SAM 3's text-prompted
segmentation natively — the agentic capability that Hybrid V2's
detector-then-labeler pattern doesn't cleanly preserve when the prompted
concept doesn't surface from a COCO-class detector first.

**Duty cycle.** At ~240 ms per agentic forward and 1 query/minute typical
operator pace, OWLv2 occupies **0.4% NPU duty** — negligible impact on
the per-frame vision pipeline. Slots into the same on-demand budget CLIP
currently uses for the 1 Hz keyframe debounce.

**Recommendation framing:** ship Hybrid V2 as the per-frame default
(real-time, BW-budget-friendly). Add OWLv2 as the agentic-role on-demand
secondary path **when** the application requires arbitrary text-prompted
segmentation outside the Hybrid V2 detector's COCO vocabulary. Both can
coexist on a single NPU — additive, not substitutive.

**Open work for the OWLv2 path.** Currently characterized in PyTorch
FP16 only. A TRT-FP8 OWLv2 port (analogous to our YOLO-seg + CLIP
recipes) could push it to ~80 ms NPU High edge — making per-frame
agentic queries viable rather than 1 Hz on-demand. KH-P2-001 in
REMEDIATION_PLAN.md tracks this if/when an application case justifies it.

---

## 5. Cross-cutting findings

### 5.1 The FP8 unblock came from TensorRT, not torchao

Three months of torchao work hit a tool-chain wall on Conv-only models
(`swap_conv2d_1x1_to_linear` covers 44%, the rest blocked by 1×128 block
constraints). TensorRT 10.16 on SM 12.0 (Blackwell) compiles full-model
FP8 with *zero QDQ-node hand-holding* — the runtime auto-selects FP8
layers. **Lesson:** FP8 on dense Conv backbones is not a fundamental
PTQ problem, it's a tool-chain maturity problem. TRT solved it.

### 5.2 INT8 + FP8 share BW-bound edge FPS — silicon dictates dtype

At 8-bit precision the matmul throughput is BW-bound on edge LPDDR5X-class
memory; quantized weights + activations at 8-bit set the ceiling
regardless of dtype. **The choice between INT8 and FP8 is a
quantization-drift-vs-silicon-class trade-off:**

- **INT8** deploys on Mid (200 TOPS INT8-only). Box-recall vs FP16 engine =
  0.875 at 720p — 12.5% of low-confidence boxes drop below score threshold
  under INT8's tighter dynamic range. Fine for high-volume coarse detection
  where the dropped boxes are noise.
- **FP8** deploys only on High (FP-capable). Box-recall vs FP16 engine =
  1.000 at 720p — quantization drift is essentially zero. The "right"
  recipe when silicon supports it.

**Caveat (engine-self-comparison).** "Recall 0.875" and "recall 1.000" both
mean *vs the FP16 engine output*, not *vs ground-truth labels*. They
quantify quantization drift, not absolute task accuracy.

This was *not* obvious going in — we expected FP8 to be visibly faster
than INT8 on Blackwell. It isn't. Both are BW-bound at 8-bit weights.

### 5.3 Dense vs MoE LLM on bandwidth-constrained silicon

VRAM scales with *total* params; per-token BW scales with *active* params.
MoE with sparse activation (Qwen3-30B-A3B = 30B total / 3B active) gives
the same model capacity headroom as a dense 32B but pays ~10× less BW per
token. Empirical:

| Model | Total params | Active per fwd | RAG 8K+2K decode (5090) |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M (MoE) | 30B | 3B | 159 tok/s |
| Qwen 2.5 32B Q4_K_M (dense) | 32.5B | 32.5B | ~34 tok/s |

**4.7× speedup** at ~equivalent VRAM cost. The MoE bandwidth thesis lands.

### 5.4 Cross-family LLM perf invariance + cross-family quality NON-invariance

7B-class dense Q4_K_M decode on 5090 is base-family-invariant within ~7%
(Qwen 184 / Mistral 183 / Llama 171 tok/s). Differences track GGUF size,
not vendor.

Same models on the same 132-sample Skippy v2-RAG eval: **Qwen2.5 7B 67.4%
/ Mistral 7B v0.3 60.6% / Llama-3.1 8B 56.8%**. Reasoning category is
the biggest delta (Qwen 6/6 vs Mistral 0/6 vs Llama 1/6) — chain-of-
thought training in Qwen ships visibly in pass rate.

**Practical rule:** at 7B class the hardware budget is family-invariant;
the quality outcome depends on corpus alignment with base capabilities.

### 5.5 Recipe transfer: two-factor model — lift requires ceiling reasoning OR family-match

**Status:** load-bearing finding as of 2026-05-10 N=6 publication.
Sanctioned framing per `personal-ai-framework/docs/GOTCHA_7_RESOLUTION.md`
(Reviewer-blessed two-factor refinement subsection, commit `6398944`).
Customer-template publication shipped reviewer-final.

**Headline (N=6 cross-judge, 2026-05-10):** *Across 12 judge passes
(6 cells × 2 judges, Claude Sonnet + GPT-4o), 11 of 12 confirm v4 ≤
base. The N=6 data is consistent with a parsimonious **two-factor
model**: substring lift requires either **ceiling stock reasoning
(6/6)** OR **family-match to the corpus source distribution
(Qwen-family in our case)**. Cross-family bases without ceiling
reasoning regress, regardless of intermediate reasoning headroom.*

**Falsifiable prediction:** a third cross-family 3/6 base should
regress. Phi-4 (Microsoft, distinct family, 128K context) is running
now on the [docs] side as the falsification candidate. Falsification
outcome: lift on Phi-4 → two-factor model breaks → 'Yi-specific
quirk' framing returns. Reviewer-named expected outcome: regression
(corroborates two-factor model).

**Same 6,517-example Skippy fine-tuning corpus, same recipe, same
hyperparams, only the base model changed:**

| Base | Stock reasoning | Family-match | Substring Δ | Sonnet Δ | GPT-4o Δ | Reading |
|---|---|---|---|---|---|---|
| Qwen 2.5 7B | **6/6** | Qwen | **+3.1 pp** | −0.350 | −0.690 | ceiling — lift erases on both judges |
| Qwen 2.5 14B | 3/6 | **Qwen** | **+8.7 pp** | ±0.000 | −0.214 | family-match — lift erases (cleanest demo, both robust) |
| Gemma 2 9B | **6/6** | cross | **+3.2 pp** | −0.620 | +0.119 | ceiling — judge-sensitive on faithfulness only |
| **Yi-1.5-9B-Chat** | **3/6** | **cross** | **−28.6 pp** | **−0.848** | **−0.714** | **two-factor predicts regression — confirmed (catastrophic)** |
| Mistral 7B v0.3 | 0/6 | cross | **−3.8 pp** | −0.218 | −0.048 | floor — regress confirmed both judges |
| Llama 3.1 8B | 1/6 | cross | **−3.2 pp** | −1.165 | −1.524 | floor — regress strongest both judges |

Judges: Claude Sonnet 4.6/4.7 via Anthropic API + GPT-4o via OpenAI API.
Same semantic rubric (faithfulness to RAG context + instruction-following
+ correctness), same 50-sample held-out subset of `prompts_v2.json`. Full
per-cell breakdown in
`personal-ai-framework/eval/results/yi_n6_falsifies_substring_predictor.md`
+ earlier N=5 analysis files.

**The Yi result is load-bearing.** Largest substring regression in the
dataset (−28.6pp). Both cross-judges corroborate at −0.7 to −0.9
magnitude. Per-category damage: rag_datasheet 55→29/78 (−26), multihop
6→0/9, coding 4→0/6, numerical_precision 6→3/6. Only gain: refusal
6→9/9 (+3). **A customer running this recipe in good faith on an
intermediate cross-family base could ship a model 28pp worse than the
base** — that's not marginal or preliminary, it's a real-world risk
the campaign has now characterized.

**Two damage profiles — same recipe, different mechanisms by direction:**

- **Lift cells (Qwen 7B / Qwen 14B / Gemma 9B):** lose RAG-citation
  discipline. Faithfulness to RAG context drops on v4 (Qwen 7B −0.43,
  Qwen 14B −0.26, Gemma −0.20 on the 0–2 dimension), while conciseness
  and instruction-following hold. Substring grader doesn't penalise
  because trained phrasings still match gold tokens; judges do.
- **Regression cells, especially Yi:** lose capability on the question
  itself. Yi v4: correctness Sonnet −0.470 / GPT-4o −0.214, instruction-
  following Sonnet −0.502 / GPT-4o −0.476. Faithfulness *holds* on Yi
  (+0.091 / +0.071). The recipe damages different things on different
  bases — both directions show real damage, just on different axes.

Customer implication: a judge weighted toward faithfulness catches
lift-cell damage; a judge weighted toward correctness catches regression-
cell damage. Single-judge runs miss whichever axis that judge
underweights — another reason for two judges by default.

**Predictor framing (sanctioned wording per reviewer-blessed two-factor
model):**

> Bases at floor stock reasoning (0–1/6, N=2: Mistral 7B, Llama 8B)
> regressed on substring AND on judge. Bases at ceiling stock reasoning
> (6/6, N=2: Qwen 7B, Gemma 9B) lifted on substring but the lift erased
> on judge. Bases at intermediate stock reasoning (3/6, N=2) split by
> family-match: Qwen 14B (Qwen-family, same as corpus source) lifted
> +8.7pp on substring; Yi-1.5-9B-Chat (cross-family) regressed −28.6pp
> with both judges corroborating. The N=6 data is consistent with a
> two-factor model: lift requires either ceiling reasoning OR
> family-match to the corpus source distribution. Customers fine-tuning
> cross-family bases without ceiling stock reasoning should expect
> regression on this recipe.

**Predictor-vs-proxy caveat (Q1, partially superseded by N=6):** at N=5,
"reasoning floor predicts substring direction" was the cleanest single-
factor predictor identified, but Yi falsified it cleanly — same 3/6
band as Qwen 14B, opposite substring direction. The two-factor model
restores parsimony AND adds falsifiability (Phi-4 third 3/6 cross-family
prediction). At N=6, the two-factor model is a strong directional
predictor; alternatives that happen to correlate (e.g., specific
training-data overlap with eval corpus) still cannot be ruled out
without further cells.

**Qwen 14B data correction note:** the original N=5 table cited Qwen 14B
at 6/6 reasoning / 6/9 refusal / +5.3pp Δ headline (interpolated values).
Running the asymmetry test required a fresh apples-to-apples 14B base
eval. Corrected values: **3/6 reasoning, 9/9 refusal, +8.7pp Δ**.
Provenance audit confirmed all other 5 base JSONs are apples-to-apples
(temp=0, RAG=on, prompts_v2, 132-sample basis); 14B was the sole cell
with interpolated values. The correction was load-bearing for the
two-factor model — repositioning 14B from "ceiling-reasoning lift" to
"intermediate-reasoning Qwen-family lift" is what made the family-match
factor visible when Yi at the same 3/6 band regressed catastrophically.

**Earlier framings superseded but preserved for audit:**
1. 2026-05-08 evening: "preliminary base-family-coupled" (N=2 directional)
2. 2026-05-09 00:13: "reasoning-floor discriminator" (N=5 substring-only)
3. 2026-05-09 15:49: "no judge-corroborated lift in N=5 cells" (Sonnet)
4. 2026-05-10 00:21: "9/10 cross-judge corroborated; Gemma judge-sensitive"
5. **2026-05-10 14:20: "two-factor model — ceiling reasoning OR family-match"** (current)

Each was correct-at-the-time and superseded as new data fired. Full
supersession trail in the GOTCHA_7_RESOLUTION.md Addendum + Reviewer
follow-up sections.

**Methodology hardening — cross-judge ran 2026-05-10:**

- **Cross-judge corroborates regression as real capability damage on
  Mistral and Llama** — both judges ≤ 0 on both regression cells; Llama
  negative *more strongly* under GPT-4o than Sonnet (Sonnet −1.165,
  GPT-4o −1.524). **If the regressions had been judge-bias artifacts,
  cross-judge would have surfaced disagreement; instead it doubled
  down. The regression-is-real reading is the most strengthened claim
  under cross-judge.**
- **Cross-judge corroboration with GPT-4o + Sonnet EXECUTED** across N=6.
  **11 of 12 judge passes confirm v4 ≤ base.** Direction agrees on 5 of 6
  cells; Gemma 9B remains the single judge-sensitive cell. The Qwen 14B
  "biggest substring lift, most evaporative judge result" demo is
  **robust under both judges**. The Yi −28.6pp regression is corroborated
  at −0.7 to −0.9 magnitude under both judges (no judge-sensitivity).
- **Standing methodology going forward: two judges by default.** No
  marginal-Δ qualifier. Reviewer-corrected 2026-05-10 from an earlier
  draft that triggered cross-judge only on marginal Δ. Justification:
  the cell that disagreed (Gemma) had Sonnet at −0.620 —
  meaningfully-negative-not-marginal-looking; the cell that looked
  most marginal on Sonnet alone (Mistral, −0.218) was robust across
  judges. The marginal-Δ filter would have missed Gemma instability.
  Cross-judge cost (~$5 per N=5 pass via OpenAI API) is in the noise
  compared to fine-tune compute; default-on is the right policy.
- **Judge-at-temp=0.3 explicitly NOT pursued.** Reviewer's reasoning:
  temp=0.3 already shows fine-tune fragility (per § 5.9
  temperature-sensitivity); rerunning judge there conflates confounds
  rather than separating them. Production decoding regime (temp=0) is
  where judge stays orthogonal.

**Gemma judge-divergence — locus is RAG-faithfulness, not overall
quality:**

| Dimension (0–2) | Gemma base — Sonnet | Gemma v4 — Sonnet | Gemma base — GPT-4o | Gemma v4 — GPT-4o |
|---|---:|---:|---:|---:|
| Correctness | 1.487 | 1.366 | 1.476 | 1.452 |
| Instruction-following | 1.667 | 1.634 | 1.857 | 1.738 |
| **Faithfulness to RAG context** | **1.564** | **1.366 (−0.198)** | **1.024** | **1.262 (+0.238)** |
| Conciseness | 2.000 | 1.732 | 1.500 | 1.524 |

Both judges agree on correctness, instruction-following, and
conciseness — Gemma v4 is flat-or-slightly-down on all three under
both judges. The disagreement is *only* on RAG-faithfulness scoring:
Sonnet penalises Gemma v4 for citation-faithfulness loss; GPT-4o
rewards it (likely a difference in how each judge interprets
"faithful citation" for Gemma's response style). **Customer
implication:** if your eval weights RAG-faithfulness as a load-bearing
dimension, characterize it under multiple judges before deploying.
Reviewer endorsed surfacing the disagreement explicitly rather than
breaking the tie with a third judge — "a team that surfaces 'here's
the cell where our two judges disagreed and here's why' looks like
it's optimizing for honest characterization, not clean conclusions."

**Stock-baseline measurement on N=6 candidates (2026-05-09):** Phi-3-
mini-4k-instruct, Yi-1.5-9B-Chat, and Gemma 2 2B-it measured for stock
baseline. All three landed at 3/6 reasoning. **Yi-1.5-9B-Chat became
the load-bearing N=6 cell** — fine-tuned 2026-05-10, regressed −28.6pp
substring (largest in dataset), corroborated as real capability damage
by both judges (Sonnet −0.848 / GPT-4o −0.714). The Yi result falsified
the N=5 single-factor "reasoning floor" predictor and surfaced the
two-factor model as the parsimonious replacement. **No 4/6 or 5/6
candidate emerged** from the stock-baseline trio.

**Queued falsification: Phi-4 (Microsoft) as third 3/6 cross-family
base.** Reviewer-named for falsification because (a) distinct family
(Microsoft/Phi vs Qwen/Mistral/Yi/Gemma/Llama already in the dataset),
(b) modern small model (NXP-relevance for "what about Phi"),
(c) 128K context (controls for the Phi-3-mini-4k context-saturation
confound). Two-factor model predicts regression on Phi-4. Falsification
outcome: lift on Phi-4 → two-factor model breaks → "Yi-specific quirk"
framing returns. Same Yi pipeline (stock baseline → fine-tune → both
judges → analysis). Running on [docs] side; not blocking customer-
template publication per reviewer ruling.

**Customer recommendation (reviewer-final, two-factor model, N=6):**
Across 12 judge passes (6 cells × 2 judges, Sonnet + GPT-4o), 11 of 12
confirm v4 ≤ base. Customers fine-tuning **cross-family bases without
ceiling stock reasoning should expect regression on this recipe** —
Yi-1.5-9B-Chat ran with this recipe in good faith and produced a model
**28pp worse than its base** on the Skippy substring eval; both
cross-judges corroborated as real capability damage. That is not a
marginal or preliminary risk — it is the recipe's behavior on
intermediate-reasoning cross-family bases as currently characterized.

The two-factor model gives customers a clean decision rule:
- **Lift expected if:** stock reasoning is at ceiling (6/6) on your
  eval — OR your base is family-matched to the corpus source
  distribution.
- **Regression expected if:** stock reasoning is below ceiling AND your
  base is cross-family to the corpus source.
- **Always run cross-judge corroboration** (two judges by default) on
  any cell whose deployment decision turns on the v4 recipe outcome —
  the Gemma cell shows that single-judge findings can carry interpretation
  risk even at meaningfully-negative Δ.

Run a stock baseline on your eval before transferring this recipe to a
new base. Bases at 2/6, 4/6, or 5/6 stock reasoning are uncharacterized.

**For the full evidence package** (judge per-cell breakdown, mechanism
analysis, reviewer Q&A, customer-template wording, methodology
hardening notes), see
`personal-ai-framework/docs/GOTCHA_7_RESOLUTION.md`.

### 5.6 Sister-model baseline confound

Original framing of the Skippy MoE FT v1 result was "+5.3 pp vs base."
The base used was Qwen3-30B-A3B-Thinking-2507 (0.636 stock pass rate).
Apples-to-apples comparison vs Qwen3-30B-A3B-Instruct-2507 (the correct
sister) is **−2.3 pp**. The 7.6 pp gap is a *base-model property*, not a
fine-tuning win.

Where the gap lives: rag_email 0/3 (Thinking) vs 3/3 (Instruct) — Thinking
is broken at base on rag_email. coding 5/6 vs 6/6. refusal 9/9 vs 6/9
(Thinking SOTA refusal; FT regressed). **Methodology lesson:** always
validate FT recipes against BOTH sister models when the base family ships
Instruct + Thinking variants. Thinking-2507 had a hidden category-level
broken-at-base behavior that confounded the FT lift framing.

### 5.7 ncu confirms the architectural win quantitatively

We measured every shipping + reference workload's per-forward DRAM via
NVIDIA Nsight Compute. The bundle (`data/output/ncu/sizer_bundle.json`)
lists all 23 workloads. Highlights, sorted by DRAM/forward (light → heavy):

| Workload | MB/fwd | NPU Mid BW-bound FPS ceiling |
|---|---|---|
| ResNet-50 INT8 TRT (5090 anchor) | 94 | 999 |
| **yolov8n-seg FP8 TRT** | 106 | **890** |
| **yolo_seg_fp8_trt (shipping)** | 217 | **434** |
| **clip_trt (shipping)** | 433 | 217 |
| yoloe26_trt_fp8 | 498 | 189 |
| EfficientSAM3 ES-EV-S | 8,934 | 10.5 |
| EfficientSAM-Small | 34,359 | 2.7 |
| **Grounding DINO Tiny** | 38,508 | **2.4** |
| **SAM 3 BF16** | **118,975** | **0.8** |

The DRAM reduction from SAM 3 → shipping is the *measured* engineering
win, not a sized projection. Two framings:

- **549× per primary forward** (217 MB yolo_seg_fp8_trt vs 118,975 MB
  SAM 3) — apples-to-apples per primary detector forward.
- **515× per full pipeline frame** (231 MB total: 217 MB YOLO + 14 MB
  amortized CLIP at 1 Hz, vs 118,975 MB SAM 3) — accounts for the CLIP
  open-vocab tower in the shipping pipeline.

The 549× framing is what makes the architectural-replacement story land:
the shipping detector forward alone is 549× lighter than the SAM 3
forward it replaces, and the CLIP tower amortizes to a small marginal
addition.

ncu replay-mode caveat: TRT engines + dynamic NMS use kernel-replay (slow
but robust); PyTorch targets use app-replay.

### 5.8 End-to-end pipeline latency budget (KH-P3-002)

The deck headline "36 FPS at 720p NPU Mid" implies <28 ms total per frame.
Bake-off projections cover the YOLO + CLIP inference stages but don't
break out the full pipeline (FFmpeg ingest → preprocess → YOLO TRT →
CLIP TRT @ 1 Hz → SQLite event-log INSERT). KH-P3-002 fills the gap.

**Profile harness:** `scripts/profile_e2e_pipeline.py` runs the canonical
720p_EW_clip on the 5090 reference platform with `time.perf_counter()`
+ `torch.cuda.synchronize()` instrumentation around each stage. GPU
stages project to NPU Mid via the standard 16.19× BW ratio; CPU stages
(decode, preprocess, DB INSERT) project via 10× ARM Cortex-A55 single-
thread slowdown documented in slide_trt_yolo's preprocessing footnote.
DB commits batched at 1 Hz (every 30 frames) per production pattern
— per-frame `commit()` would fsync per frame and crowd out the budget
catastrophically.

**Measured (200 frames, 720p_EW_clip, FP8 TRT engines):**

| Stage | 5090 p50 ms | NPU Mid p50 ms | Class | Notes |
|---|---|---|---|---|
| ingest_decode | 1.96 | **19.57** | CPU | cv2 H.264 decode, single-thread |
| preprocess | 1.20 | **11.99** | CPU | letterbox 720p → 640×640, normalize |
| yolo_trt_infer | 1.03 | 16.69 | GPU | dynbatch=1 engine, includes input copy |
| clip_trt_infer (× 1/30) | 0.79 | 0.43 | GPU | 1 Hz amortization (12.82 ms full × 1/30) |
| db_insert (~3 dets/frame) | 0.02 | 0.20 | CPU | SQLite INSERT, batched commits |
| **per-frame total** | **4.23** | **48.88** | | |
| **5090 sustained** | **236 FPS** | | | |
| **NPU Mid 36 FPS budget** | — | **27.78 ms** | | |
| **NPU Mid slack** | — | **−21.10 ms** | | over budget |

**Headline finding: CPU stages crowd out the 36 FPS NPU Mid budget on a
pure-NPU board.** The GPU stages (yolo + amortized clip) total ~17 ms,
well within the 27.78 ms budget. The CPU stages (decode + preprocess +
DB) total ~32 ms, which alone exceeds the budget. This is a real
integration-architecture finding the YOLO+CLIP-only headline doesn't
expose.

**Production-realistic projection.** SoCs with fixed-function ISP and
2D GPU (Qualcomm Hexagon, MediaTek Genio, NXP i.MX, Ambarella, Hailo)
move decode + preprocess off-CPU entirely:

- ingest_decode → ~0.3 ms via NVDEC / hardware video decoder block
- preprocess → ~0.5 ms via 2D-GPU letterbox + ISP normalize

With those offloads and batched commits, NPU Mid p50 total ≈ **17.6 ms
= 56 FPS sustained**, well under the 36 FPS budget with ~10 ms
headroom for multi-stream batching or higher source resolution.

**Pure-NPU boards** (Coral, some development kits) without fixed-function
ISP / 2D GPU pay the full CPU cost. For those targets, the practical
deployment recipe is to either (a) lower source resolution to 480p (cuts
decode + preprocess in half), or (b) drop to a smaller detector
(yolov8n-seg, ncu floor 106 MB/forward — half of yolo11s-seg).

**Reviewer takeaways.** The 36 FPS headline is achievable end-to-end
on production SoCs that ship hardware decode + 2D GPU, with comfortable
headroom. It is *not* achievable on a pure-NPU board running CPU
software-decode — the integration-architecture matters as much as the
NPU spec. The deck's prior framing didn't quantify this; § 5.8 + the
new `slide_e2e_latency_budget` deck slide do.

### 5.9 Temperature sensitivity in LLM accuracy citations (caveat)

LLM accuracy citations in this briefing (§ 3.10, § 3.11, § 5.4, § 5.5)
are temp=0 production-grading numbers (greedy decoding + substring
grader). Skippy-side variance-bounds work (see
`personal-ai-framework/docs/skippy-claude-briefing.md` § temperature-
sensitivity) measured the same models at temp=0.3 with stochastic
sampling and found:

- **Base models are temperature-flat** (qwen-7b-base ±1.7 pp; mistral-7b-
  base ±2.9 pp).
- **Fine-tuned models are temperature-brittle.** Skippy 7B v4 dropped
  from 70.5% (temp=0) to 44.5% (temp=0.3), a **−26 pp swing**. Skippy
  Mistral v4 dropped −5.5 pp.

**Interpretation (per [docs]):** the fine-tunes learned high-fidelity
output patterns the substring grader rewards at temp=0; stochastic
sampling at temp=0.3 breaks those patterns even when the answer is
semantically correct. The temp=0 production headline numbers may
therefore reflect format fidelity rather than absolute task accuracy.

**Implication for this briefing:** the LLM-accuracy citations are
mechanical pass-rate measurements as graded, and the percentages stand
as reported. But "fine-tune adds X pp lift" claims should be read as
"fine-tune adds X pp lift *under temp=0 substring grading*" — a
narrower scope than absolute capability. The cross-family regression
finding (§ 5.5) is partially insulated from this caveat because both
sides of the cross-family comparison are graded the same way; the
relative direction holds even if absolute magnitudes are
substring-grader-coupled.

For the full grader-methodology deep dive, see the Skippy briefing
linked above.

---

## 6. What we ruled out

Each of these was tested empirically before being shelved:

- **SAM 3 BF16 at any plausible edge BW.** 119 GB/forward × 30 FPS =
  3.6 TB/s — exceeds even GDDR7 high-end host silicon, and edge silicon
  is 1/10× to 1/40× of that. Not feasible without model replacement.
- **INT8 weight-only quantization.** Doesn't touch activation traffic;
  zero edge gain on BW-bound workloads.
- **SAM 3 resolution / prompt cuts.** RoPE-locked attention won't let us
  shrink the input while keeping the architecture intact.
- **torchao FP8 on Conv-only models.** Tool-chain gap, not a fundamental
  limit. Use TensorRT instead.
- **Generative LLM on a busy vision NPU at non-trivial query rates.**
  RAG (8K+2K decode) obliterates vision FPS even on NPU High. Reserve LLM
  for async or a second-NPU use case.
- **Camera-side ViT replacement of YOLO-seg + CLIP.** RT-DETR / DETR /
  Grounding-DINO are 10–13× heavier per forward; bust real-time on
  stock LPDDR5X memory.
- **YOLOE-26 one-model pivot.** TRT FP8 gives ~17% speedup, not 3×.
  Kernel-launch-bound at 16M params; the structural gap to two-stage +
  1Hz CLIP isn't closeable by FP8.

---

## 7. Open questions / limitations

### 7.1 INT8 CLIP port

The full open-vocab pipeline currently pins to NPU High. An INT8 CLIP
port (post-training quantization with activation calibration over a
representative text-image distribution) would unlock the full pipeline
on NPU Mid silicon. Not in our toolchain today; budget would be ~1–2
weeks of careful PTQ work.

### 7.2 INT4 / FP4 detection-head

If an NPU exposes INT4 / FP4 native compute, detection-head quantization
might claw back another ~1.5× edge FPS. Accuracy risk on detection-head
logits (where FP8 already wins over INT8); warrants a targeted study
when a candidate NPU ships.

### 7.3 LPDDR6 memory upgrade unlocks a class transition

LPDDR6-14 (14 GT/s) at 128-bit raises NPU Mid BW from 94 → ~165 GB/s.
Sized projections show this lifts:

- shipping pipeline @ 720p single-stream: 36 → ~50 FPS
- camera-ViT alternatives (DETR ResNet-50): 5.7 → ~28 FPS — close to
  real-time, would re-open the 1-not-2 ViT question

This is a *future-silicon* question, not a today-deployment one.

### 7.4 v4-on-Llama-8B fine-tune (predicted)

White paper pre-registered a prediction: **Llama-3.1 8B + v4 recipe will
land ~60–63% headline pass rate** (60.6% Mistral v4 falsified the
prediction by regressing below stock; same falsification risk on Llama).
Per the recipe-base-coupling finding, expect Llama v4 FT to regress
further than Mistral did, given Llama stock starts at 56.8%. Bake-off
infrastructure ready when GGUF lands.

### 7.5 Cross_class util_factor recalibration

The 1.95× over-projection on Llama-3.1 8B suggests the sizer's
fp16-dense util_factor on RTX 5090 should drop ~50%. Trade-off: a
global recalibration risks over-correcting cells where the over-
projection happens to be smaller (Qwen, MoE). Per-cell measurement_alias
is the cleaner path; util_factor recalibration is the bigger-hammer
alternative.

### 7.6 Compute-ceiling clamp Phase 2

i.MX 95 ground truth (32 ms @ 1080p) showed BW-only sizing under-
projects on weak silicon. Phase 1 (`measured_edge_ms` override field on
Hardware) shipped. Phase 2 (per-tier `compute_efficiency` clamp +
GOPs_per_pipeline annotation) deferred. Mid + High remain BW-bound at
the workloads that matter; the clamp matters for sub-5-TOPS silicon
where per-kernel overhead dominates.

---

## 8. Methodology notes (worth scrutiny)

- **Quality metrics are engine-self-comparison, not ground-truth task
  accuracy.** When this briefing reports "FP8 box recall 1.000" or
  "matched IoU 0.998," the reference is the FP16 TRT engine output on the
  same input frames — *not* hand-labeled ground truth. These metrics
  measure quantization drift (how faithfully the quantized engine
  reproduces the FP16 engine's outputs), not absolute task accuracy.
  Open-vocab segmentation quality on novel concepts is uncharacterized in
  this study; see § 9 question 8 for what would be needed to measure it.
  JSON field naming bumped to `box_recall_vs_fp16_engine` /
  `mean_matched_iou_vs_fp16_engine` to make this explicit (schema
  version 2). Legacy fields preserved as aliases for one cycle.
- **Dtype gating applied to projection JSONs** (schema v3, KH-P0-002 in
  REMEDIATION_PLAN.md). Per-recipe projection cells now carry
  `dtype_mismatch_on_mid` (boolean) + `deployable_tiers` (list) +
  `dtype_mismatch_reason` (string) fields. NPU Mid is INT8-only at
  200 TOPS; FP-class recipes (`fp8`, `fp16`, `bf16`, `fp32`) are flagged
  `dtype_mismatch_on_mid=True` and project to NPU High (BW-equal at stock
  LPDDR5X-8.4 memory class). **Historical FP-on-Mid raw projection
  numbers are preserved** alongside the gating flag — the reviewer's
  guidance was render dtype mismatch as a flag, not delete data. The
  matrix is in the bundle's `__meta__.tier_dtype_support` field and
  rendered as a markdown table in the bundle MD § 5.

### 8.1 Two BW estimates — when to use which (KH-P0-001 reconciliation)

Same workload, same target tier, two different numbers — this came up in
the external review. They answer different questions:

- **`bw_floor_ms_npu_mid`** (ncu side): pure DRAM-bytes/forward ÷ NPU
  effective BW. Best-case minimum; **cannot be achieved in practice**
  because real silicon pays kernel-launch overhead, NMS dispatch, memory
  hierarchy stalls, sync overhead.
- **`effective_edge_ms_with_overhead`** (bake-off side): 5090 GPU-kernel
  wall-time × BW ratio (16.19×) + 5090-derived CPU overhead. Captures all
  the overhead the 5090 actually paid; **assumes that overhead profile
  transfers to edge silicon** (probably pessimistic since edge ARM +
  tightly-integrated NPU may have lighter dispatch tax than 5090 + x86
  CPU).

**Real edge latency sits BETWEEN the two.** The bake-off projection is
the more conservative (slower) estimate and is what the deck + sizer use
as the headline FPS. The BW floor is the engineering lower bound — useful
for "is this workload BW-bound or compute-bound?" questions.

| Workload (720p) | DRAM MB/fwd | BW floor ms (Mid) | Effective edge ms (Mid, w/ overhead) | Overhead ratio |
|---|---|---|---|---|
| yolo_seg_fp8_trt (shipping) | 217 | 2.30 | 27.19 | 11.8× |
| yolo_seg_fp16_trt (was the 22.7× discrepancy reviewer caught) | 219 | 2.32 | 53.63 | **23.1×** |
| clip_trt (shipping) | 433 | 4.61 | 15.57 | 3.4× |
| sam3_bf16_reference | 119,000 | 1265 | (not deployable) | n/a |

**Reading this:** `yolo_seg_fp16_trt` was the headline discrepancy — 23.1×
between the two methodologies because at 219 MB DRAM/forward the
workload is overhead-dominated, not BW-bound. The shipping `_fp8_trt`
variant lands at 11.8× because halving activation bytes drops the
overhead-fraction modestly but doesn't change the absolute overhead pool.
`clip_trt` at 3.4× is the cleanest BW-bound case in the table (433 MB/
forward = enough DRAM traffic that overhead amortizes well).

`sam3_bf16_reference` has no bake-off projection (we don't deploy it) —
the 1265 ms BW floor alone tells the story: at 119 GB DRAM/forward, real-
time is physically impossible at any plausible edge BW.

**For reviewers:** if you want to challenge the headline 36 FPS shipping
number, the right attack is "is the bake-off projection methodology's
overhead model right for edge silicon?" — not "your numbers don't
match." The numbers don't match by design; they answer different
questions.
- **5090 → NPU Mid scale = 16.19×.** Effective: (1792 × 0.85) / (134.4 ×
  0.70) = 1523.2 / 94.08 = 16.19. Used as the canonical scale factor
  across every edge projection. Sensitivity: ±10% on either efficiency
  factor changes edge FPS by ±15%.
- **0.70 BW efficiency** uniform across all 4 NPU tiers. Reconciled to
  this value 2026-04-21; earlier deck snapshots used 0.75/0.80. Full
  derivation + sensitivity analysis + caveats in
  `docs/methodology/bw_efficiency_derivation.md` (KH-P1-001). Headline:
  0.70 was picked as the most defensibly-conservative single value across
  three drifting prior assumptions (0.75 / 0.80 / 0.80), not from a
  per-tier calibration measurement. The single biggest informal
  validation is the NPU Mid vendor LLM anchor (37.85 tok/s) which
  projects within ballpark from the 5090 anchor under the 0.70 assumption.
  For BW-bound vision workloads (yolo_seg_fp8_trt, clip_trt) no edge
  ground-truth exists yet; the headline 36 FPS rests on this assumption
  plus the bake-off projection methodology.
- **Vendor anchors override BW-only projections.** 5090 → NPU Mid was
  2.3× pessimistic on LLM decode vs vendor numbers; sizer uses vendor
  anchors when they exist.
- **ncu replay modes.** App-replay (~1 hr per PyTorch target, fast) for
  ML targets without dynamic kernels; kernel-replay (~80 min for
  trt_yolo, ~3 hrs for trt_yoloe26) for TRT + NMS targets where
  per-pass kernel sets vary.
- **NVTX label parsing for ncu 2026+** requires extracting from
  `thread Domain:Push/Pop_Range:...` column (legacy NVTX column was
  dropped). Fixed in `scripts/profile_ncu.py`.
- **5090 anchor uses GDDR7 effective 0.85.** Memory subsystem is the
  binding factor; compute is rarely the ceiling at 8-bit on Blackwell
  for the workloads we measure.

---

## 9. Questions for the reviewing Claude

We're asking an external session to evaluate. Specific things to scrutinize:

### Soundness questions

1. **Is the BW-bound argument quantitatively correct?** Sizer pipeline:
   per-forward DRAM ÷ effective BW = ms/frame. Cross-check with the ncu
   bundle's `bw_bound_ms_min` field. Does the math hold?

2. **Is the 5090→NPU Mid 16.19× scale factor sensible?** It assumes
   linear BW scaling with effective bandwidth. ncu measurements show
   workloads sit at 70–95% of theoretical BW on 5090; do they on edge?
   We don't have direct ncu measurements on edge silicon — only vendor
   benchmarks.

3. **Cross_class projection 1.95× over-projection on Llama-3.1 8B.** We
   call this a calibration data point. Is it methodologically a problem
   for any other cells in the sizer that lack anchors? We have anchors
   for Qwen 2.5 (7B + 32B), Llama-3.1 8B, Mistral 7B v0.3, MoE 30B-A3B.
   What's missing?

4. **INT8 + FP8 sharing edge FPS at 8-bit.** The deck and briefing
   present this as a clean story. Is it? Could there be a kernel-launch
   amortization regime where FP8 wins on small models we haven't tested?

5. **Recipe-base-coupling finding.** The "voice transfers, capability
   doesn't" pattern is novel — is it methodologically sound? Per-category
   data for Mistral v4: refusal/rag_email/numerical_precision +3,
   coding/rag_blog/rag_datasheet −3/−3/−8. Could this be eval-set
   bias rather than a recipe-base interaction?

### Story questions

6. **The 90× headline (0.4 → 36 FPS).** Is this the right framing? It
   implies "optimization journey" but the actual story is "architectural
   replacement (SAM 3 → Hybrid V2) + tool-chain unlock (TRT FP8) +
   per-frame amortization (1 Hz CLIP)." Should the deck reframe?

7. **The 1-not-2 ViT decision.** We tested 4 ViT alternatives. Did we
   miss any? OWLv2 won the agentic role; should the deck recommend it
   for replacing SAM 3's text-prompted segmentation, or is Hybrid V2 +
   on-demand SAM 3 still the better integration story?

8. **Open-vocab capability claim.** We assert Hybrid V2 (YOLO-seg + CLIP)
   matches SAM 3's open-vocab capability. The actual evaluation is
   weaker than this — we matched detection + labeling on a curated
   embedded-world clip. SAM 3's segmentation quality on novel concepts
   is harder to evaluate. Should the deck weaken the claim?

### Honesty questions

9. **"Recommended stack" framing.** We label Hybrid V2 + TRT FP8 + 1Hz
   CLIP "recommended." It's recommended *for our specific application*
   (embedded-world inspection). For higher-fidelity SAM 3-like
   capabilities (medical imaging, agricultural inspection where the
   "open-vocab" requirement is harder), this stack may be insufficient.
   The deck doesn't always make this scope explicit.

10. **Sister-model confound.** The original "+5.3 pp domain FT win"
    framing was wrong. We caught it. What other framings might still
    have hidden base-model dependencies? Our FT cross-family work surfaced
    one; are there others?

---

## 10. Sister artifacts + where to find data

This repo (`keyhole`) holds the bake-off harness + deck builder + API
server. Sister artifacts:

- **keyhole-sizer** (`github.com/kylefoxaustin/keyhole-sizer`, hosted at
  `https://keyhole-sizer.streamlit.app`) — interactive NPU-tier sizing
  tool. Pick a pipeline + tier + concurrency, get edge FPS + DRAM
  bandwidth + duty-cycle math. 20 pipelines, 5 tiers, includes RTX 5090
  reference cell + i.MX 95 ground-truth cell.

- **personal-ai-framework / Skippy** — LLM training campaign for the
  domain-specialized voice + capability fine-tunes. White paper at
  `docs/skippy-white-paper.md` + recipe taxonomy at
  `docs/recipe-taxonomy.md`.

- **my-stuff** (`github.com/kylefoxaustin/my-stuff`) — deck + xlsx
  artifacts, refreshed each commit. The 62-slide
  `keyhole_results.pptx` and 22-sheet `keyhole_results.xlsx` live there.

### Files in this repo worth reading

- `docs/CLAUDE_REVIEW_BRIEFING.md` — this file.
- `data/output/keyhole_data_bundle.md` — every measurement, rendered.
- `data/output/keyhole_data_bundle.json` — same, machine-parseable.
- `REPRODUCE.md` — operator guide for re-running every bake-off.
- `API.md` — backend HTTP API (the embedded-world demo runs against this).
- `scripts/build_deck.py` — the deck builder; each `slide_*` function
  reads a specific bake-off JSON and renders a slide.
- `scripts/bakeoff_*.py` — one per stage of the journey.
- `scripts/profile_ncu.py` + `scripts/profile_all_ncu.sh` — Nsight
  Compute sweep harness.
- `scripts/export_ncu_for_sizer.py` — bridge that turns raw ncu CSVs
  into the sizer's per-workload DRAM bundle.

---

## 11. Honest framing for the review

This briefing is *our framing of our work*, written by the team that did
the work. The reviewing Claude's job is **not** to validate the
conclusions — it's to find:

- Numbers that don't reconcile across sources.
- Methodology shortcuts where confidence is overstated.
- Architectural choices that make sense in our application but wouldn't
  generalize (and that we're presenting as if they do).
- Claims framed as "we discovered X" that are actually "we measured X in
  one specific configuration."
- Missing work that a senior engineer would expect to see (e.g., do we
  have variance / std-dev across runs? do we have eval coverage on
  out-of-distribution inputs?).
- The shape of arguments we should but don't make (e.g., what are the
  *right* questions for selecting between Mid and High silicon for a
  specific application? we hint but don't quite answer).

The data bundle (`keyhole_data_bundle.json`) is the ground truth. The deck
is the polished narrative. This briefing is the bridge. Trust the data;
challenge the framing.
