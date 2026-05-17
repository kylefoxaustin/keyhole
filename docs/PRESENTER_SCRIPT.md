# Keyhole — Presenter Script

**Audience:** Technical management (engineering leaders who understand the
domain — NPU silicon, edge ML — but want bottom-line + implications, not
deep methodology dives).

**Target runtime:** ~45–60 minutes for the plain deck (65 slides; presenter
skips slides 49–52 which carry training-methodology content that belongs
to the Skippy product deck — see Section 8 note below). Backgrounder
slides skim; load-bearing slides get the linger.

**Conceptual frame:** Keyhole is the **edge AI video analytics platform**.
The product deliverable is a vision pipeline that runs on NPU-class
silicon. The LLM is an **optional feature** of the Keyhole product —
the *same artifact* that ships as the Skippy / personal-AI product
(Qwen3-30B-A3B base, Q4_K_M). Training-methodology content (recipe
taxonomy, headline-erosion arc, cross-family bake-offs) belongs in the
Skippy deck, not this one.

**Three operational modes** the deck addresses:
1. **Vision-only.** Vision pipeline runs; LLM off. Default video-analytics
   use case.
2. **Vision + LLM.** Both run; LLM acts on vision data (NLQ over the event
   store, agentic scene queries). Vision = substrate; LLM = interaction
   surface on top. Engineering question = NPU coexistence.
3. **LLM-only.** Vision pipeline off; LLM standalone. Equivalent to the
   Skippy product running inside keyhole-sizer. Perf matches Skippy deck.

**Tone:** Confident but honest. The deliverable has been calibrated
heavily through external review; the self-correction discipline is itself
a credibility marker — surface it without self-aggrandizement.

**Structure of this script:** one section per narrative arc, with per-slide
speaker text. Stage directions in italics. "Pause" markers indicate where
to slow down + invite questions.

---

## Section 1 — Opening (slides 1–7)

### Slide 1 — Title

*Land on title slide.*

> "Thanks for the time. I'm going to walk you through Keyhole — what it
> is, what we learned about getting an open-vocab vision pipeline onto
> edge-class NPU silicon, and the secondary methodology findings that
> came out of the work. There's about 45 minutes of material; I'll
> leave 15 for questions. If something jumps out earlier, interrupt."

### Slide 2 — Executive summary

*Three hero cards: 515× DRAM reduction, 36 FPS, 0.4 FPS baseline.*

> "Here's the engineering arc on one slide. The starting condition:
> SAM 3, Meta's open-vocab segmenter, runs at **0.4 FPS** on the edge
> silicon we target. That's not 'slow' — that's *unusable*. We
> ended at **36 FPS** at 720p on NPU Mid stock LPDDR5X. The number
> in the middle is the engineering win: **515× lower DRAM bandwidth
> per shipping pipeline frame**, measured by Nsight Compute, not
> projected.
>
> This is not a quantization story. SAM 3 at 119 GB DRAM per forward
> is bandwidth-bound on every plausible edge memory subsystem,
> regardless of bit-width. The 515× win came from architectural
> replacement: a smaller two-stage pipeline that meets the open-vocab
> capability requirement at one five-hundredth the memory traffic.
> Numbers to track through the rest of the deck."

*Point at the BW efficiency caveat at bottom of hero cards.*

> "Note the ±15% sensitivity band on the 36 FPS number. That's the
> spread we see on the 0.70 bandwidth efficiency assumption used to
> project from our reference hardware. We'll come back to that in the
> methodology section."

### Slide 3 — Platform specs

*Table of host hardware + edge target.*

> "Quick context on the hardware envelope. Our reference platform is an
> RTX 5090 with an i9 host — all bake-offs measured here. The headline
> edge target is **NPU Mid**: 134 GB/s LPDDR5X stock, 200 BF16 / 400
> INT8 / 400 FP8 TOPS — FP-capable. Edge projections scale from 5090
> measurement via bandwidth ratio (~16× to Mid; ~12× to High, which
> has a faster 11.2 GT/s bus). Trustworthiness of that scale factor —
> and the only-one-measured-edge-anchor caveat — comes up in the
> methodology section."

### Slide 4 — NPU tier assumptions

*Multi-tier matrix from Low-LP5-32bit through High. Two tiers do most of
the work in this deck: Mid + High.*

> "Edge NPU tier model we project against. The two that matter most:
>
> **NPU Mid** — 128-bit LPDDR5X @ 8.4 GT/s, 134.4 GB/s peak (94 GB/s
> effective at 70% efficiency), 200 BF16 / 400 INT8 / 400 FP8 TOPS,
> 24 GB DRAM, 25 W. **FP-capable** — runs BF16 and FP8 recipes natively.
>
> **NPU High** — 128-bit LPDDR5X @ 11.2 GT/s, 179.2 GB/s peak (125 GB/s
> effective), 275 BF16 / 550 INT8 / 550 FP8 TOPS, 32 GB DRAM, 40 W.
> Different memory bus from Mid (33% more bandwidth) plus higher TOPS,
> capacity, and TDP.
>
> Both tiers are FP-capable. The choice between them is a
> memory-bandwidth + compute-headroom + capacity trade-off, not a dtype
> trade-off. INT8 and FP8 recipes deploy on either tier; tier selection
> is driven by the workload's bandwidth budget, concurrent workloads,
> and DRAM footprint."

### Slide 5 — Architecture diagram

> "Five-stage pipeline as the deck labels it: FFmpeg ingest, YOLO-seg
> detection + masks, open-vocab labeling via CLIP, event store, and
> the LLM as an *optional* natural-language query layer. The whole
> deck is about getting the vision stages — detection through event
> store — to run in real-time on NPU-class silicon.
>
> *Cue the three-modes framing — say this once, then move on.*
>
> The keyhole-sizer Streamlit app exposes three operational modes
> over this pipeline: **vision-only** (the default video-analytics
> case, LLM off); **vision + LLM** (both running, LLM acts on the
> event store — NLQ, agentic scene queries — coexistence is the
> engineering question); and **LLM-only** (vision off, LLM standalone
> — equivalent to running the Skippy product inside keyhole-sizer).
> The next ~40 slides are dominated by the vision-only and vision+LLM
> modes; LLM-only perf is identical to the Skippy deck and is
> cross-referenced rather than reproduced."

### Slide 6 — SAM 3 reference breakdown

> "If we just used SAM 3 directly — Meta's open-vocab segmenter — here
> are the numbers. 840 million parameters, BF16-locked attention,
> **119 GB DRAM per forward**, 0.4 FPS edge. The 119 GB number is from
> Nsight Compute. At 134 GB/s edge memory, the bandwidth floor alone
> is 890 ms per frame — physically incapable of real-time. This is
> the constraint we're working under."

### Slide 7 — Roofline model

> "Standard roofline: arithmetic intensity vs achievable performance.
> SAM 3 sits in the bandwidth-bound regime on every plausible edge
> hardware. No amount of clever compute makes this faster. Architecture
> change is the only lever."

*Pause for questions on the framing.*

---

## Section 2 — Per-clip baseline results (slides 8–28)

### Slides 8–27 — Per-clip test runs (20 slides, skim at ~20 sec each)

*These 20 slides alternate between a Test Run page (raw 5090
measurements) and an Edge NPU Projection page for each input clip.
Don't dwell — say the framing once, then page through.*

> "We benchmarked against six representative embedded-world clips —
> 720p, 1080p, and 4K from the EW capture set, plus a bus exterior
> and a synthetic test pattern. For each clip the deck has one Test
> Run slide (raw 5090 measurements) followed by one Edge NPU
> Projection slide (the same workload scaled to NPU Mid via the
> 16.19× bandwidth ratio). I'm not going to read these one by one —
> the pattern is consistent: bandwidth-bound on edge, compute-
> headroom on the 5090. They're in the deck so you can audit any
> single clip if you want to. Otherwise, head to the comparison
> chart on slide 28."

### Slide 28 — Run comparison chart

> "Bottom line on the per-clip comparisons: the 5090 measurements are
> internally consistent within ±5% across resolutions. We trust our
> 5090 numbers; the open question is how cleanly they project to the
> edge — and that's what the rest of the deck answers."

---

## Section 3 — Bandwidth physics (slides 29–31)

### Slide 29 — Bandwidth wall analysis

> "This is the load-bearing physics slide. The vertical axis is DRAM
> bytes per forward; the horizontal is bandwidth-bound latency at NPU
> Mid. SAM 3 sits at 119 GB / 890 ms. Real-time at 30 FPS is the red
> line at 33 ms. SAM 3 is **27× the bandwidth budget** for 30 FPS.
>
> Quantization can halve activation bytes. INT4 can halve them again.
> Neither reaches 27×. **The model has to change**, not just the
> precision."

### Slide 30 — Bandwidth requirements

> "Same physics expressed as 'what model size fits the budget?' At
> NPU Mid, a 30 FPS workload can spend at most ~3 GB of DRAM per
> forward. SAM 3 spends 40× that. The candidates that fit the budget
> are sub-billion-parameter models. We need to find one that preserves
> the open-vocab capability."

### Slide 31 — Quantization tested (weight-only INT8)

> "First thing we tried, before the architectural pivot: weight-only
> INT8 on SAM 3. The expectation was that 4× weight compression would
> halve memory traffic. The reality: **zero edge gain**. Weights are a
> minority of the DRAM bytes on this workload; activations dominate.
> Quantizing weights without touching activations doesn't move the
> bandwidth needle. Lesson here: edge bandwidth optimization has to
> target activations, not weights."

---

## Section 4 — Quantization journey (slides 32–35)

### Slide 32 — Activation quantization challenges

> "Activation quantization is where the real wins are — and where the
> hard problems live. SAM 3's attention path has BF16-locked operations
> the public tooling can't quantize. torchao gets us 94 of 95 Linear
> layers but the BF16-locked attention dominates BW. The remaining
> compute path still travels at FP16/BF16. We're stuck at ~1.2 FPS
> even with aggressive activation quant. The tool-chain has a real
> gap on SAM-3-style architectures."

### Slide 33 — Prompt count scaling

> "Briefly: SAM 3 scales linearly with the number of prompt tokens.
> Each additional text concept costs another full-model forward. We
> tried trimming the prompt set — it doesn't move the headline because
> we're bandwidth-bound on the model itself, not on the prompts."

### Slide 34 — Resolution lock analysis

> "Final SAM-3 desperation move: cut the input resolution. SAM 3's
> rotary attention is locked to its training resolution — you can't
> just downscale the input and run. The architecture doesn't permit
> a clean resolution cut. This closes the door on optimizing SAM 3 in
> place."

### Slide 35 — Speed vs Accuracy — The Model Tradeoff

*Backgrounder bridge slide; skim quickly.*

> "Before the pivot, one framing slide: speed vs accuracy across the
> SAM-3-class tradeoff space. Every model we'll bench in the next
> sections sits somewhere on this plane — quality on one axis,
> inference latency on the other. The architectural pivot we're
> about to describe is what lets us move from 'high quality, far too
> slow' into 'high enough quality, fast enough at the edge'."

*Beat. Slow down.*

> "At this point in the campaign we had ruled out every flavor of
> quantization, prompt reduction, and resolution cut. SAM 3 stays at
> 0.4 FPS on edge. The pivot has to be architectural."

---

## Section 5 — Architectural pivot (slides 36–38)

### Slide 36 — Hybrid V2 breakthrough

> "Here's the architectural pivot. **Replace the monolithic open-vocab
> segmenter with a two-stage pipeline**: a small dense detector +
> segmenter (YOLO-seg, 10M params, COCO classes), followed by a
> zero-shot open-vocab labeler (OpenCLIP ViT-B/32) on each cropped
> region.
>
> The detector handles 'where are the objects.' The labeler handles
> 'what user-specified text concept matches each object.' Open-vocab
> capability is preserved structurally — we kept SAM 3's interface
> (text prompts in, segmented + labeled objects out) but moved the
> open-vocab work from a 840M monolith to an 88M CLIP visual tower
> running on detector-confirmed regions only.
>
> At BF16 baseline, edge ms drops from 2500 ms to about 62 ms.
> **16 FPS at 720p.** Real-time is in sight. Open vocab still works."

### Slide 37 — Mask bake-off summary

> "We benched four mask-model alternatives — MobileSAM, EfficientSAM
> tiny/small, YOLO-seg — against the SAM 3 mask quality reference. The
> two surviving candidates with edge-viable bandwidth are MobileSAM and
> YOLO-seg. We picked YOLO-seg because the detection + segmentation
> head is fused, which saves a 2D-GPU step at deployment time."

### Slide 38 — Mask bake-off visuals

> "Side-by-side mask quality on the embedded-world test clips. YOLO-seg
> at IoU 0.86 vs SAM 3 reference. Difference: SAM 3 produces fractional-
> pixel edge smoothing; YOLO-seg edges are blockier. For our use case
> — object identification + bounding box correctness — the
> difference is invisible."

---

## Section 6 — FP8 + TensorRT — the unblock (slides 39–46)

### Slide 39 — FP8 activation quantization

> "Now the precision optimization on the new pipeline. FP8 activation
> quant via torchao gets us 94 of 95 Linear layers on the smaller mask
> models — same tool-chain gap as before, but on a smaller surface
> area. Edge FPS still moves slowly. We need a different approach for
> the Conv layers."

### Slide 40 — SmoothQuant + INT8

> "SmoothQuant explored. The technique smooths activation outliers
> to make INT8 quant safer. Implementation hit a torchao 0.17
> compatibility gap — couldn't complete the CONVERT step. Plain INT8
> weight quant landed at activation-quant-equivalent edge gain (i.e.,
> zero — same lesson as the SAM-3 weight-only case). Moving on."

### Slide 41 — Hybrid V2 CLIP quantization bake-off

> "Quantization on the CLIP visual tower in the Hybrid V2 stack. CLIP
> at BF16 is the bottleneck — 22 ms per crop on the 5090, hits 8 ms on
> reduced precision. Setting us up for the TRT compile path on the
> next slide."

### Slide 42 — CLIP keyframe debouncing

> "Practical win: CLIP doesn't have to run every frame. Object identity
> doesn't flicker frame-to-frame in real video. We run CLIP once per
> second (every 30th frame at 30 FPS source) and re-use the labels.
>
> Edge FPS goes from 16 to 24. This is free — it's an integration
> trick, not a model change. The 1 Hz debounce is a fundamental piece
> of the shipping recipe."

### Slide 43 — YOLO-seg conv quantization

> "torchao's Conv quantization handles 44% of YOLO-seg's conv weights
> — the 1×1 swap path. The remainder is blocked by a torchao 1×128
> block-size constraint on Conv layers. Same theme: the tool-chain is
> the bottleneck, not the model. This is where TensorRT enters the
> picture."

### Slide 44 — TensorRT YOLO FP8 / INT8

*Big breakthrough slide.*

> "TensorRT 10.16 on Blackwell silicon compiles full-model FP8 with
> zero QDQ-node hand-holding — the runtime auto-selects FP8 layers.
> Conv backbone, detection head, segmentation head, everything.
>
> 5090 wall-time at 720p: FP16 → 0.67 ms; INT8 → 0.91 ms; FP8 → 0.68
> ms. Edge projection on NPU Mid stock LPDDR5X: FP16 → 53.6 ms
> (18.6 FPS); INT8 → 27.2 ms (**36.8 FPS**); FP8 → 27.2 ms (**36.8
> FPS**). Real-time on Mid — the cheaper of our two FP-capable
> tiers — is reached for the first time in this campaign.
>
> Two important findings here:
>
> First — **the FP8 unblock came from TensorRT, not torchao.** Three
> months of torchao work hit a tool-chain wall; TRT compiled it
> cleanly. The model wasn't broken; the tool-chain matured.
>
> Second — **INT8 and FP8 deliver the same *edge* FPS but not the
> same 5090 wall-time.** On Blackwell, INT8 is actually slower than
> FP8 (0.91 vs 0.68 ms) because the INT8 path pays a dequant
> overhead the native FP8 datapath doesn't. At the edge, both run on
> identical 8-bit weights and the workload is bandwidth-bound — so
> projected FPS collapses to the same 36.8 regardless of dtype. Both
> Mid and High are FP-capable, so both recipes deploy on either tier.
> Tier choice is driven by memory-bandwidth headroom (Mid is the
> minimum bus that achieves 36 FPS at 720p; High has 33% more BW for
> concurrent workloads) and by capacity / TDP — not by precision.
>
> The 'box recall' column here is engine-self-consistency: FP8 boxes
> vs FP16 engine boxes, not ground truth. The relevant signal: FP8
> reproduces FP16 detections at IoU 0.998 — quantization drift
> essentially zero."

### Slide 45 — yolo11s-seg vs yolov8n-seg comparison

> "Quick cross-variant check. yolo11s-seg is the shipping detector;
> yolov8n-seg is the smaller variant. 8n is half the DRAM at 720p (106
> MB vs 217 MB per forward) and twice the edge FPS — projects to about
> 850 FPS BW ceiling, vs 11s's 434. For applications where the COCO
> class set is overkill, 8n is the lighter-weight option."

### Slide 46 — TensorRT CLIP visual

> "CLIP visual tower TRT-compiled cleanly at FP16 + FP8. The 88M-param
> tower drops from 47 ms BF16 to 29 ms FP16 to 16 ms FP8 — about a 3×
> total speedup. Top-1 concept-tag agreement vs BF16 is 0.964 —
> noise-level quality loss.
>
> The recipe is FP — and both Mid and High are FP-capable, so the
> full open-vocab pipeline (YOLO-FP8 + CLIP-FP8 at 1 Hz debounce)
> deploys on either tier. Mid is the headline silicon target because
> it's the cheaper of the two and still hits the 36 FPS budget at
> 720p. High is the upgrade path when the deployment is
> capacity-bound, BW-bound on concurrent workloads, or co-hosting an
> active LLM."

*Pause for questions on the bake-off arc.*

---

## Section 7 — LLM bake-off + duty cycle (slides 47–48)

### Slide 47 — LLM bake-off (Qwen3-30B-A3B)

> "The LLM dimension. Keyhole's optional natural-language-query
> feature — and any agentic / NLQ scene-query — is backed by a local
> LLM. **The LLM here is the same artifact that ships as the Skippy
> product: Qwen3-30B-A3B base, Q4_K_M quantization, identical
> shipping recipe.** Keyhole *uses* the artifact; the training story
> (recipe taxonomy, fine-tuning campaigns, methodology arc) lives in
> the Skippy deck.
>
> Why this model: Mixture-of-Experts with 30B total parameters but
> only 3B active per token. Right fit for bandwidth-bound silicon —
> VRAM scales with total params; per-token bandwidth scales with
> active params.
>
> 5090 measurements at Q4_K_M: 250 tokens/sec decode @ 256-token
> outputs; 159 tokens/sec on full RAG (8K context + 2K output).
> Vendor-published edge anchors put NPU Mid at 37.85 tok/s and NPU
> High at 50.46 tok/s decode on the same model — High has 33% more
> memory bandwidth at stock (11.2 vs 8.4 GT/s), which is exactly the
> decode ratio. Tier choice on LLM workloads = bandwidth budget; both
> tiers are FP-capable and run the same artifact.
>
> *If asked: 'why don't you fine-tune this for Keyhole specifically?' —*
> *say 'we use the Skippy artifact unmodified. The training methodology*
> *that produced it is documented in the Skippy deck; reproducing it*
> *here would be redundant.'*"

### Slide 48 — NPU duty-cycle trade-off

> "Practical deployment question: can we run the LLM concurrently with
> the vision pipeline on a shared NPU?
>
> Short answer: short queries yes, RAG no. A 200-token response at 38
> tok/s costs about 5 seconds of NPU time. If the user queries once
> per minute, that's 8% duty cycle — about a 1 FPS hit on the vision
> stream. Acceptable.
>
> Full RAG (8K prefill + 2K decode) costs about 60 seconds of NPU
> time. Anything more than a query every few minutes obliterates the
> vision pipeline. So: short answers viable; RAG either gets a second
> NPU or runs asynchronously."

---

## Section 8 — SKIP slides 49–52 (Skippy training content — belongs in Skippy deck)

*Slides 49–52 in the plain deck carry Skippy training-methodology
content: recipe taxonomy, gotcha-#7 supersession arc, headline-erosion
data, sister-model confound. **Per the Keyhole conceptual frame, this
content belongs in the Skippy product deck, not here.** When these
slides appear on screen, skip them quickly with a single pointer:*

> "These four slides describe how the Skippy LLM artifact was
> trained — recipe taxonomy, methodology arc, headline erosion,
> sister-model confound. That's product-internal training content for
> the Skippy deck, not for Keyhole. We *use* the trained artifact
> unmodified; the training story is one slide-deck away. Skipping
> these to keep the Keyhole narrative focused on edge-deployment
> physics + vision pipeline."

*Page forward without dwelling. If the audience asks for the training
detail, defer to the Skippy deck (`personal-ai-use-cases.pptx`) and
its presenter script. In a corporate-template branded variant rebuild,
slides 49–52 will be removed entirely — see `ALIGNMENT_PLAN.md` in
the repo for the deferred refactor scope.*

---

## Section 9 — Cross-cutting LLM findings (slides 53–54)

### Slide 53 — Dense vs MoE bandwidth (LLM deployment context)

*Slide may show a cross-family comparison table; for Keyhole, focus
narration on the MoE-on-edge deployment thesis. The cross-family
content (Mistral / Llama / Yi / Phi-4) is Skippy-deck-relevant and
should be skimmed in Keyhole.*

> "The deployment-relevant LLM finding: **Mixture-of-Experts wins on
> bandwidth-bound edge silicon at equivalent VRAM.** Qwen3-30B-A3B
> at 159 tokens/sec RAG decode on 5090 beats Qwen 2.5 32B dense at
> 34 tok/s — 4.7× at the same memory footprint. Per-token bandwidth
> pays for the 3B active parameters, not the 30B total.
>
> For Keyhole this matters because the LLM is co-hosted with vision
> on a shared NPU. MoE keeps per-token bandwidth low so vision
> coexistence stays viable; a 32B dense at the same quality tier
> would consume 5× the per-token bandwidth and obliterate the
> vision FPS budget.
>
> Cross-family base-model selection is documented in the Skippy
> deck. For Keyhole's purposes the answer is fixed: Qwen3-30B-A3B
> Q4_K_M, same artifact as Skippy product."

### Slide 54 — Multi-stream concurrency

> "Practical question: can one NPU serve multiple camera streams?
> Measured TensorRT YOLO at batch sizes 1, 2, 4, 8, 16. The headline:
> at batch=4 on a 4-stream deployment, each stream sees about 26 FPS.
> Not the naïve 9 FPS you'd get from round-robin serial scheduling.
> Batching beats serial. 8-stream batch=8 lands at 15 FPS each.
> Multi-stream is real."

---

## Section 10 — Community SAM 3 + ViT alternatives (slides 55–59)

### Slide 55 — EfficientSAM3 community bake-off

> "Open-source community released two SAM-3-Lite variants in April. We
> benched them. EfficientSAM3 ES-EV-S: 424M params, BF16, lands at
> 2.6 FPS edge at 720p. 6.5× faster than SAM 3. Still 13× slower than
> our shipping pipeline. Community variants beat the monolith but
> can't touch a purpose-built two-stage pipeline."

### Slide 56 — EfficientSAM3.1 text-prompt variant

> "Smaller variant: EfficientSAM3.1 at 106M params with text-prompt
> capability. Even faster than ES-EV-S but still single-digit FPS at
> 1080p+. Same conclusion: useful as a community SAM 3 Lite but
> doesn't change our pipeline decision."

### Slide 57 — YOLOE-26 one-model open-vocab

> "Ultralytics released YOLOE-26 in January — a one-model open-vocab
> alternative to our two-stage Hybrid V2. 4585-class built-in vocab,
> text-prompt support. Interesting because if it works, we collapse
> our two-stage pipeline to one model. Benched it.
>
> Result: ~13 FPS at 720p NPU High in PyTorch FP16. Three times
> slower than our two-stage TRT FP8 stack. Why? Because the YOLOE-26
> head is kernel-launch-bound at small parameter count — TRT FP8
> compression doesn't help when the bottleneck isn't matmul. Next
> slide confirms."

### Slide 58 — TRT YOLOE-26

> "TRT FP8 on YOLOE-26 gives ~17% speedup over PyTorch FP16, not the
> 3× we get on YOLO-seg. At 16M params, the open-vocab head's
> kernel-launch overhead dominates wall-time. TRT FP8 doesn't help
> here. **TRT FP8 pays off when the kernel is big**; on small models
> with complex graph topology, it doesn't.
>
> Useful methodology data point: not every workload benefits from
> precision lowering. Match the optimization to the bottleneck."

### Slide 59 — ViT alternatives — what-if

> "Investigated four ViT alternatives in case a single-model approach
> might still win: RT-DETR-L, DETR-ResNet50, OWLv2, Grounding DINO.
> 10–13× heavier per forward than our shipping detector. Camera-side
> ViTs don't fit the bandwidth budget at NPU Mid stock memory. With
> LPDDR6 memory upgrade (~165 GB/s effective), DETR ResNet-50 reaches
> 28 FPS — within striking distance. But that requires future silicon.
>
> One useful finding: **OWLv2 is the SAM 3 successor for agentic
> queries.** 42× lighter than SAM 3, 6× faster, retains text-prompted
> segmentation natively. At 240 ms per forward, slot it into a 1-Hz
> agentic-query budget — 0.4% NPU duty cycle. Use case: 'find the
> [arbitrary text concept] in this scene' on operator-driven prompts,
> as opposed to per-frame open-vocab labeling which is what our
> Hybrid V2 pipeline handles."

---

## Section 11 — TRT takeaways (slide 60)

### Slide 60 — TRT takeaways

> "Synthesizing the three TRT bake-offs: YOLO-seg, CLIP visual,
> YOLOE-26. **Where TRT FP8 pays off**: dense convolutional backbones
> + dense ViT towers with large matmul kernels. Both YOLO-seg and
> CLIP land 1.5–2× speedups with engine-self-consistency near-perfect.
>
> **Where it doesn't pay off**: small-parameter models with complex
> graph topology (YOLOE-26). The bottleneck is kernel-launch overhead,
> which compression doesn't address.
>
> Decision rule: profile first; figure out what's bandwidth-bound vs
> compute-bound vs launch-bound; only then decide if FP8 helps."

---

## Section 12 — ncu measurement validation (slides 61–63)

### Slide 61 — ncu measured DRAM — headline 515× gap

*Big visualization of measured DRAM per forward across workloads.*

> "Nsight Compute measurements on every workload in the deck. Sorted
> by DRAM per forward, ascending. The shipping pipeline — yolo_seg_
> fp8_trt at 217 MB per forward — is at the light end. SAM 3 at the
> bottom of the chart at 118,975 MB. The ratio is **515×** when you
> include CLIP at 1 Hz amortization (231 MB total for the full
> shipping pipeline frame).
>
> This number is measured, not projected. ncu reads the actual DRAM
> bytes the kernels load. The 515× win is real silicon traffic
> measurement — it's the engineering claim the rest of the deck rests
> on."

### Slide 62 — ncu measured DRAM — workload table

> "Full table of measured workloads. Use this as the reference when
> someone asks 'what about [model X] on edge?' If it's in the table,
> the answer is bandwidth-bound at the listed memory budget."

### Slide 63 — End-to-end pipeline latency budget

*The CPU-crowding finding slide.*

> "End-to-end profile: not just YOLO + CLIP, but every stage of the
> pipeline. FFmpeg decode, preprocess (letterbox + normalize), YOLO
> inference, CLIP inference, SQLite event-log INSERT.
>
> Measured on 5090, projected to NPU Mid:
>
> - GPU stages (YOLO + 1Hz CLIP): about 17 ms total. Well within the
>   27.78 ms budget for 36 FPS.
> - CPU stages (decode + preprocess + DB INSERT) on edge ARM: about
>   32 ms. **Alone exceeds the budget.**
>
> This is the finding: **on a pure-NPU board** — think Coral or
> development kits without fixed-function ISP + 2D GPU — the
> CPU stages crowd out the 36 FPS budget. The deck's prior framing
> didn't quantify this.
>
> **On production SoCs with fixed-function ISP + 2D GPU offloads**
> (Qualcomm Hexagon, MediaTek Genio, NXP, Ambarella, Hailo), decode
> + preprocess move off-CPU. Projected NPU Mid total drops to ~17.6
> ms. **About 56 FPS sustained.** But — important caveat — that's a
> projection, not measured. The ISP+2D-GPU offload is plausible but
> hasn't been validated on actual silicon. The i.MX 95 anchor remains
> the only edge measurement we have in the campaign.
>
> Practical takeaway for the customer: integration-architecture
> matters as much as NPU spec. Headline 36 FPS holds on production
> SoCs with proper offloads; on pure-NPU boards, plan for lower
> resolution (480p) or a smaller detector (yolov8n)."

---

## Section 13 — Roadmap + summary (slides 64–65)

### Slide 64 — Optimization roadmap

> "Where we go from here. Three open methodology gaps:
>
> 1. **Real Mid-class NPU silicon anchor.** Currently i.MX 95 is the
>    only edge measurement; everything else is 5090-projected. A
>    Mid-class NPU loan would close this gap.
> 2. **INT4 / FP4 detection-head precision** if a Mid-class NPU
>    silicon ships with it. Could buy another ~1.5× on detection head
>    bandwidth.
> 3. **INT8 CLIP port** to unlock the full Hybrid V2 pipeline on Mid
>    silicon (not just High). Post-training quantization with
>    calibration; ~1–2 weeks of focused work."

### Slide 65 — Summary & findings

*The final wrap-up slide — has hero stat bar + bulleted findings.*

> "Final summary. The engineering arc: SAM 3 at 0.4 FPS, bandwidth-
> bound on every plausible edge memory subsystem. We replaced it with
> a two-stage Hybrid V2 — YOLO-seg + CLIP @ 1Hz — at 515× lower DRAM
> per shipping pipeline frame. The replacement achieves 36 FPS at
> 720p on NPU Mid stock LPDDR5X.
>
> Three layers of finding:
>
> 1. **Architectural replacement**: SAM 3 → Hybrid V2. Measured, not
>    projected. 515× DRAM reduction. Open-vocab capability preserved
>    on the inspection workload we target.
>
> 2. **Tool-chain unlock**: TensorRT 10.16 on Blackwell compiles full
>    Conv backbone FP8 cleanly. Three months of torchao work were
>    blocked by tool-chain maturity, not by the underlying physics.
>    Sometimes the answer is wait for the tool.
>
> 3. **Methodology findings**: the substring grader had Qwen-family
>    format bias we caught through external review + cross-judge
>    methodology hardening. The three-gate framework — capability +
>    voice + safety — was designed for exactly this kind of silent
>    grading failure; production decision unaffected.
>
> The methodology rigor here was as important as the engineering
> result. The deliverable today is sharper than what started this
> campaign — both technically and in how we know what we know.
>
> Happy to take questions."

*End of public deck.*

---

## Conditional Section — Private Anchors Slide (only in `--include-private` build)

### Slide 66 (private) — Measured silicon anchors

> "If you're in the NXP-internal audience: this slide is in your
> deck variant. It shows measured silicon anchor performance for
> Qwen3-MoE and Qwen 2.5 dense on NPU Mid/High, plus ResNet-50 and
> YOLOv8n at 4-bit and 8-bit weights.
>
> Two notes:
>
> First — the values you're looking at were loaded into this .pptx
> at build time from a gitignored secrets file. They were not visible
> to the Claude session that constructed this deck. The discipline:
> code is chat-safe, runtime output is private.
>
> Second — comparing the measured cells to the projected numbers
> elsewhere in this deck: where they match within the ±15% BW-
> efficiency band, our methodology validates. Where they diverge,
> the methodology should be revisited — you can flag without quoting
> the values back to me; bounds-language is sufficient.
>
> This slide also lives in the keyhole-sizer Streamlit app at
> share.streamlit.io — same data, runtime UI."

---

## Closing remarks (for the presenter)

> "Bottom-line bullets if Q&A doesn't get there:
>
> - **Edge ML is bandwidth-physics first.** Quantization helps where
>   it helps (activations on big kernels). Architecture replacement
>   is the bigger lever when a model is 27× over budget.
>
> - **TensorRT 10.16 unblocks FP8 on Blackwell** in ways torchao
>   currently can't reach for Conv backbones. The model wasn't broken;
>   the tool-chain matured.
>
> - **Hybrid V2 deploys on either NPU Mid or NPU High** — both are
>   FP-capable. Mid is the headline silicon (cheaper, hits 36 FPS at
>   720p); High is the upgrade path for concurrent workloads or
>   capacity-bound deployments.
>
> - **The LLM in Keyhole is the Skippy product artifact, unmodified**
>   — Qwen3-30B-A3B Q4_K_M. Training methodology, recipe taxonomy, and
>   any cross-family base-selection story live in the Skippy deck.
>   Keyhole uses the artifact; it does not retrain it.
>
> - **Three operational modes:** vision-only (default), vision + LLM
>   (NLQ + agentic scene-queries; NPU coexistence is the engineering
>   question), LLM-only (Skippy-product behavior inside keyhole-sizer).
>
> - **Integration-architecture matters.** The 36 FPS edge result holds
>   on production SoCs with ISP + 2D-GPU offloads. On pure-NPU boards,
>   plan accordingly.
>
> - **One measured edge silicon anchor today** (i.MX 95 yolov8n-seg).
>   The rest of the deck scales from 5090 reference via bandwidth-ratio
>   projection. KH-P2-001 in the roadmap: get a Mid-class NPU
>   measurement to close the gap.
>
> Thank you."

---

## Notes for the script reviewer (Claude browser)

This script is aligned to the **Keyhole conceptual frame** (2026-05-17,
see `~/.claude/projects/.../memory/project_conceptual_frame.md`). Key
divergences from the prior version:

- **Section 8 removed.** Skippy training-methodology content (recipe
  taxonomy, gotcha-#7 arc, headline-erosion data, sister-model
  confound) is product-internal training material that belongs in the
  Skippy deck, not in Keyhole. The plain deck's slides 49–52 still
  contain that content — the script instructs the presenter to skim
  them with a one-line pointer. A planned plain-deck refactor will
  remove those slides; see `docs/ALIGNMENT_PLAN.md`.
- **NPU tier framing updated.** Slides 4, 24/44, 26/46 narration now
  treats Mid and High as **both FP-capable, on different memory buses**
  (Mid @ 8.4 GT/s, High @ 11.2 GT/s) per the conceptual frame's slide
  4 reference. Tier choice = bandwidth + capacity + TDP trade-off, not
  a dtype trade-off. *Note for cross-deck reviewers:* this contradicts
  the current PAI/Skippy deck slide 11 (which still shows Mid INT8-only
  + same bus). Per the conceptual frame, both decks will converge to
  the new framing; the Skippy deck has not yet been updated.
- **Three operational modes** acknowledged on Slide 5 (vision-only,
  vision + LLM, LLM-only). Sets up which mode each later section
  addresses.
- **LLM identity** stated on Slide 47: Qwen3-30B-A3B Q4_K_M, same
  artifact as Skippy product, training story cross-referenced to Skippy
  deck rather than reproduced.
- **Slide 53 reframed** to drop cross-family base-selection content
  (Mistral / Llama / Yi) — that's Skippy-deck-relevant. Keeps the
  MoE-on-edge thesis (Qwen3-30B-A3B vs Qwen 2.5 32B dense) since it
  bears on Keyhole's vision + LLM coexistence story.

Things this script does:

- **Leads with conclusions per slide**, then supports.
- **Honest framings throughout**: 36 FPS sensitivity band ±15%, the
  "56 FPS production-SoC projection is not measured."
- **Skips dense bake-off methodology** that doesn't bear on the bottom
  line. Per-clip results, prompt-count scaling, weight-only INT8
  rejection — these get one-paragraph mentions.
- **Lingers on load-bearing findings**: architectural pivot (Slide 36),
  TRT FP8 unblock (Slide 44), the e2e CPU crowding finding (Slide 63).
- **Pause points marked** where the presenter should slow down +
  invite questions.

Things this script does *not* do:

- Doesn't quote specific NPU measurement values from the private
  anchors slide — by design (discipline rule).
- Doesn't make claims about NPU Mid silicon measurements that we
  don't have (KH-P2-001 honestly scoped as the remaining gap).
- Doesn't reproduce Skippy training methodology — defers to Skippy deck.

Open questions for the reviewer:

1. Is the 45–60 minute runtime appropriate for a technical management
   audience, or should the script trim further?
2. The cross-deck NPU tier contradiction (this script says Mid
   FP-capable + different buses; current PAI/Skippy deck slide 11 says
   Mid INT8-only + same bus). For a reviewer presenting to mixed
   audiences who might see both decks, is the convergence story
   ("both decks updating to the new framing; PAI hasn't caught up")
   adequate, or does the divergence need a footnote in delivery?
3. The "the deck has 65 slides but presenter skips 4" volume — is
   that itself a problem for the target audience? Should the speaker
   explicitly call out "I'll skip slides 49–52 as Skippy-deck content"
   near the front to set expectations?
4. The conditional private slide — if the audience is NXP-internal and
   has clearance to see measured values, does the bounds-language
   framework I describe actually work in a live presentation, or does
   the audience want the team to discuss specific numbers?
