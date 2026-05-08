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

We took an open-vocab vision pipeline targeted at edge-class NPU silicon
(LPDDR5X-class memory, ~134 GB/s) from **0.4 FPS unusable** to **36 FPS
real-time** at 720p — a **90× improvement** — through a sequence of 8
sequential bake-offs and 3 follow-on investigations.

The headline architectural call: **SAM 3 (Meta's 840M-param open-vocab
segmenter) is bandwidth-bound on every plausible edge memory subsystem and
cannot be saved by quantization alone**. The shipping pipeline replaces it
with a two-stage Hybrid V2: a lightweight detector-segmenter (YOLO-seg, 10M
params) + an open-vocab labeler (OpenCLIP ViT-B/32) running at 1 Hz keyframe
debounce. Both halves compile cleanly to TensorRT FP8 with negligible
quantization drift vs the FP16 reference engine (`box_recall_vs_fp16_engine`
1.000, `mean_matched_iou_vs_fp16_engine` 0.998 for FP8; CLIP top-1 agreement
vs BF16 = 0.964). **These are engine-self-consistency metrics, not absolute
task accuracy** — open-vocab segmentation quality on novel concepts is not
characterized in this study (see § 8 methodology + § 9 question 8).

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

The ncu (NVIDIA Nsight Compute) measurements close the loop: shipping
pipeline = **231 MB DRAM per frame**, vs SAM 3 = **119,000 MB per frame**.
A **515× DRAM reduction** is the engineering win — not optimization, but
architectural replacement.

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

### 5.5 Recipe transfer is base-family-coupled (new, 2026-05-08)

Same 6,517-example Skippy fine-tuning corpus, same recipe, same hyperparams,
only the base model changed:

| Base | Δ vs stock baseline |
|---|---|
| Qwen 2.5 7B → v4 | **+3.1 pp** (production) |
| Qwen 2.5 14B → v4 | +5.3 pp (fabricates peripherals — not shipped) |
| Qwen 2.5 32B → v4 | −4.6 pp (corpus too small for 32B) |
| Qwen3-30B-A3B (MoE attn-only) → FT v1 | −9.8 pp (recipe MoE-incompatible without router) |
| **Mistral 7B v0.3 → v4** | **−3.8 pp** (recipe damages retrieval on non-Qwen dense) |

Per-category split for Mistral v4: **refusal +3 / rag_email +3 /
numerical_precision +3** transfer cleanly; **coding −3 / rag_blog −3 /
rag_datasheet −8** regress hard. Voice + safety lifts are
family-invariant; capability cost varies wildly by base.

**Hypothesis (untested):** Mistral's chat template required
`{% generation %}` patching before training (similar to Qwen3-MoE). The
patched template + assistant_only_loss combination may reweight away from
RAG-following more than it did on Qwen 2.5 dense.

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

The 515× DRAM reduction from SAM 3 → shipping is the *measured* engineering
win, not a sized projection. ncu replay-mode caveat: TRT engines + dynamic
NMS use kernel-replay (slow but robust); PyTorch targets use app-replay.

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
- **5090 → NPU Mid scale = 16.19×.** Effective: (1792 × 0.85) / (134.4 ×
  0.70) = 1523.2 / 94.08 = 16.19. Used as the canonical scale factor
  across every edge projection. Sensitivity: ±10% on either efficiency
  factor changes edge FPS by ±15%.
- **0.70 BW efficiency** uniform across all 4 NPU tiers. Reconciled to
  this value 2026-04-21; earlier deck snapshots used 0.75/0.80. Derivation
  doc pending (KH-P1-001 in REMEDIATION_PLAN.md).
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
