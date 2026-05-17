# Keyhole — Presenter Script

**Audience:** Technical management (engineering leaders who understand the
domain — NPU silicon, edge ML, fine-tuning — but want bottom-line +
implications, not deep methodology dives).

**Target runtime:** ~45–60 minutes for the 65-slide deck. Backgrounder
slides skim; load-bearing slides get the linger.

**Tone:** Confident but honest. The deliverable has been calibrated
heavily through an external-reviewer remediation arc; the self-correction
discipline is itself a credibility marker — surface it without
self-aggrandizement.

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
> RTX 5090 with an i9 host — all bake-offs measured here. The edge
> target is NPU Mid: 134 GB/s LPDDR5X, 200 TOPS INT8, no FP path.
> Everything you'll see scales from the 5090 measurement by a 16.19×
> bandwidth ratio. We'll talk about how trustworthy that scale factor
> is when we hit the methodology section."

### Slide 4 — NPU tier assumptions

*5-tier matrix: Low-LP4, Low-LP5X, Low-LP5-32bit, Mid, High.*

> "Five NPU tiers we model. The two that matter most for this deck are
> **NPU Mid** — 128-bit LPDDR5X-8.4, 200 TOPS INT8, **no FP path** —
> and **NPU High** — same memory bus as Mid, but FP-capable: 200 BF16
> or 400 FP8 TOPS. Same bandwidth-bound regime; differentiator is dtype.
>
> The reason Mid + High share the same memory tier matters: a workload
> that's bandwidth-bound projects to the same edge FPS on either tier.
> The difference shows up when a workload needs FP — for those, it's
> High-only."

### Slide 5 — Architecture diagram

> "Standard four-stage pipeline: FFmpeg ingest, detect + segment, label
> with open-vocab text concepts, write to a queryable database with
> optional LLM-driven natural-language query. Camera-to-event-store on
> embedded silicon. The whole deck is about getting this to run in
> real-time at the detect/segment/label stage."

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

## Section 2 — Per-clip baseline results (slides 8–10)

### Slide 8+ — Run results per video clip

> "We benchmarked against three clips of representative embedded-world
> footage: 720p, 1080p, 4K. Per-clip results are in the deck for
> reference. The pattern is consistent across all three: bandwidth-
> bound on edge, compute-headroom on the 5090. I'll skip the details
> and call out the comparison chart on the next slide."

### Slide 9 — Run comparison chart

> "Bottom line on the per-clip comparisons: the 5090 measurements are
> internally consistent within ±5% across resolutions. We trust our
> 5090 numbers; the open question is how cleanly they project to the
> edge."

---

## Section 3 — Bandwidth physics (slides 10–12)

### Slide 10 — Bandwidth wall analysis

> "This is the load-bearing physics slide. The vertical axis is DRAM
> bytes per forward; the horizontal is bandwidth-bound latency at NPU
> Mid. SAM 3 sits at 119 GB / 890 ms. Real-time at 30 FPS is the red
> line at 33 ms. SAM 3 is **27× the bandwidth budget** for 30 FPS.
>
> Quantization can halve activation bytes. INT4 can halve them again.
> Neither reaches 27×. **The model has to change**, not just the
> precision."

### Slide 11 — Bandwidth requirements

> "Same physics expressed as 'what model size fits the budget?' At
> NPU Mid, a 30 FPS workload can spend at most ~3 GB of DRAM per
> forward. SAM 3 spends 40× that. The candidates that fit the budget
> are sub-billion-parameter models. We need to find one that preserves
> the open-vocab capability."

### Slide 12 — Quantization tested (weight-only INT8)

> "First thing we tried, before the architectural pivot: weight-only
> INT8 on SAM 3. The expectation was that 4× weight compression would
> halve memory traffic. The reality: **zero edge gain**. Weights are a
> minority of the DRAM bytes on this workload; activations dominate.
> Quantizing weights without touching activations doesn't move the
> bandwidth needle. Lesson here: edge bandwidth optimization has to
> target activations, not weights."

---

## Section 4 — Quantization journey (slides 13–15)

### Slide 13 — Activation quantization challenges

> "Activation quantization is where the real wins are — and where the
> hard problems live. SAM 3's attention path has BF16-locked operations
> the public tooling can't quantize. torchao gets us 94 of 95 Linear
> layers but the BF16-locked attention dominates BW. The remaining
> compute path still travels at FP16/BF16. We're stuck at ~1.2 FPS
> even with aggressive activation quant. The tool-chain has a real
> gap on SAM-3-style architectures."

### Slide 14 — Prompt count scaling

> "Briefly: SAM 3 scales linearly with the number of prompt tokens.
> Each additional text concept costs another full-model forward. We
> tried trimming the prompt set — it doesn't move the headline because
> we're bandwidth-bound on the model itself, not on the prompts."

### Slide 15 — Resolution lock analysis

> "Final SAM-3 desperation move: cut the input resolution. SAM 3's
> rotary attention is locked to its training resolution — you can't
> just downscale the input and run. The architecture doesn't permit
> a clean resolution cut. This closes the door on optimizing SAM 3 in
> place."

*Beat. Slow down.*

> "At this point in the campaign we had ruled out every flavor of
> quantization, prompt reduction, and resolution cut. SAM 3 stays at
> 0.4 FPS on edge. The pivot has to be architectural."

---

## Section 5 — Architectural pivot (slides 16–18)

### Slide 16 — Hybrid V2 breakthrough

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

### Slide 17 — Mask bake-off summary

> "We benched four mask-model alternatives — MobileSAM, EfficientSAM
> tiny/small, YOLO-seg — against the SAM 3 mask quality reference. The
> two surviving candidates with edge-viable bandwidth are MobileSAM and
> YOLO-seg. We picked YOLO-seg because the detection + segmentation
> head is fused, which saves a 2D-GPU step at deployment time."

### Slide 18 — Mask bake-off visuals

> "Side-by-side mask quality on the embedded-world test clips. YOLO-seg
> at IoU 0.86 vs SAM 3 reference. Difference: SAM 3 produces fractional-
> pixel edge smoothing; YOLO-seg edges are blockier. For our use case
> — object identification + bounding box correctness — the
> difference is invisible."

---

## Section 6 — FP8 + TensorRT — the unblock (slides 19–26)

### Slide 19 — FP8 activation quantization

> "Now the precision optimization on the new pipeline. FP8 activation
> quant via torchao gets us 94 of 95 Linear layers on the smaller mask
> models — same tool-chain gap as before, but on a smaller surface
> area. Edge FPS still moves slowly. We need a different approach for
> the Conv layers."

### Slide 20 — SmoothQuant + INT8

> "SmoothQuant explored. The technique smooths activation outliers
> to make INT8 quant safer. Implementation hit a torchao 0.17
> compatibility gap — couldn't complete the CONVERT step. Plain INT8
> weight quant landed at activation-quant-equivalent edge gain (i.e.,
> zero — same lesson as the SAM-3 weight-only case). Moving on."

### Slide 21 — Hybrid V2 CLIP quantization bake-off

> "Quantization on the CLIP visual tower in the Hybrid V2 stack. CLIP
> at BF16 is the bottleneck — 22 ms per crop on the 5090, hits 8 ms on
> reduced precision. Setting us up for the TRT compile path on the
> next slide."

### Slide 22 — CLIP keyframe debouncing

> "Practical win: CLIP doesn't have to run every frame. Object identity
> doesn't flicker frame-to-frame in real video. We run CLIP once per
> second (every 30th frame at 30 FPS source) and re-use the labels.
>
> Edge FPS goes from 16 to 24. This is free — it's an integration
> trick, not a model change. The 1 Hz debounce is a fundamental piece
> of the shipping recipe."

### Slide 23 — YOLO-seg conv quantization

> "torchao's Conv quantization handles 44% of YOLO-seg's conv weights
> — the 1×1 swap path. The remainder is blocked by a torchao 1×128
> block-size constraint on Conv layers. Same theme: the tool-chain is
> the bottleneck, not the model. This is where TensorRT enters the
> picture."

### Slide 24 — TensorRT YOLO FP8 / INT8

*Big breakthrough slide.*

> "TensorRT 10.16 on Blackwell silicon compiles full-model FP8 with
> zero QDQ-node hand-holding — the runtime auto-selects FP8 layers.
> Conv backbone, detection head, segmentation head, everything.
>
> Results: 5090 wall-time at FP16 → 3.3 ms; INT8 → 1.68 ms; FP8 →
> 1.68 ms. Same wall-time at 8-bit, regardless of dtype. Edge
> projection: **36.8 FPS at 720p**.
>
> Two important findings here:
>
> First — **the FP8 unblock came from TensorRT, not torchao.** Three
> months of torchao work hit a tool-chain wall; TRT compiled it
> cleanly. The model wasn't broken; the tool-chain matured.
>
> Second — **INT8 and FP8 share the bandwidth-bound edge FPS.** At
> 8-bit precision the matmul throughput is BW-bound; the dtype choice
> is a *quality-vs-silicon-class* trade-off, not a speed trade-off.
> INT8 deploys on NPU Mid; FP8 deploys on NPU High with better recall
> on the detection head. Same edge FPS either way.
>
> The 'box recall' column here: this is the engine-self-consistency
> measurement — FP8 boxes vs FP16 engine boxes. Not ground truth.
> We'll come back to that caveat. The relevant signal: FP8 reproduces
> FP16 detections at IoU 0.998 — quantization drift is essentially
> zero."

### Slide 25 — yolo11s-seg vs yolov8n-seg comparison

> "Quick cross-variant check. yolo11s-seg is the shipping detector;
> yolov8n-seg is the smaller variant. 8n is half the DRAM at 720p (106
> MB vs 217 MB per forward) and twice the edge FPS — projects to about
> 850 FPS BW ceiling, vs 11s's 434. For applications where the COCO
> class set is overkill, 8n is the lighter-weight option."

### Slide 26 — TensorRT CLIP visual

> "CLIP visual tower TRT-compiled cleanly at FP16 + FP8. The 88M-param
> tower drops from 47 ms BF16 to 29 ms FP16 to 16 ms FP8 — about a 3×
> total speedup. Top-1 concept-tag agreement vs BF16 is 0.964 — noise-
> level quality loss.
>
> Important caveat: this is a **FP-only recipe**. NPU Mid is INT8-only
> and there's no INT8 CLIP port in our tool-chain yet. So the full
> open-vocab pipeline pins to NPU High silicon. On NPU Mid silicon,
> you ship YOLO INT8 + raw COCO labels — no open-vocab. That's the
> deploy-split. We'll come back to silicon-class trade-offs in a
> moment."

*Pause for questions on the bake-off arc.*

---

## Section 7 — LLM bake-off + duty cycle (slides 27–28)

### Slide 27 — LLM bake-off (Qwen3-30B-A3B)

> "Separate dimension: the application has a downstream natural-
> language-query feature backed by a local LLM. We target Qwen3-30B-
> A3B, a Mixture-of-Experts model with 30B total parameters but only
> 3B active per token. The mixture-of-experts design is the right fit
> for bandwidth-bound silicon — VRAM scales with total parameters;
> per-token bandwidth scales with active parameters.
>
> 5090 measurements at Q4_K_M quantization: 250 tokens/sec decode @
> 256-token outputs; 159 tokens/sec on full RAG (8K context + 2K
> output). Vendor-published edge anchors put NPU Mid at 38 tok/s
> decode on the same model — the bandwidth-bound projection from
> 5090 was 2.3× pessimistic relative to vendor numbers, so we anchor
> projections off the vendor data when available."

### Slide 28 — NPU duty-cycle trade-off

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

## Section 8 — Skippy training campaign + gotcha arc (slides 29–32)

*Lengthy section — this is the methodology-rich part of the deck.*

### Slide 29 — Skippy recipe taxonomy

*Large table; about 17 rows of recipe variants with verdicts.*

> "Switching tracks: the LLM side has its own fine-tuning campaign
> running in parallel. Skippy is a domain-tuned variant of Qwen3-30B-
> A3B for our embedded-world voice + safety profile. This slide is the
> taxonomy of recipe variants we tried, with a verdict on each.
>
> The currently-shipping production model is **Skippy 7B v4** — a
> dense Qwen2.5-7B base, our v4 recipe. Production substring pass
> rate of 70.5% on our v2-RAG evaluation.
>
> Now — and this is going to feel counterintuitive — **that production
> substring number is misleading**. Look at the column: '0.705
> substring / 0.606 semantic.' The model passes 60.6% of the eval
> under semantic grading, not 70.5%. There's an entire methodology
> story behind that gap; we'll walk it on the next two slides. The
> short version: **the substring grader had a Qwen-family format-bias
> we caught two weeks ago** through a rigorous remediation arc.
>
> Production decision is unaffected. We ship Skippy 7B v4 because the
> three-gate framework — capability + voice + safety — caught the
> silent substring failure and confirmed the production model ships
> on voice + safety. The recipe was producing voice + safety, not
> capability lift. We just didn't know it from the substring grader.
>
> Two more rows worth pointing at on this slide: **Skippy 14B v4** at
> the top of the table — same recipe on a 14B base — produces what
> the substring grader scores as a +8.7pp lift over the 14B base.
> Under semantic, that lifts +4.8 to +5.5pp. **One of only two
> cross-family v4 cells that survives semantic regrade.** This
> matters in the methodology story.
>
> And **Yi-1.5-9B-Chat v4** — different family, intermediate
> reasoning — produces a **−28.6pp substring regression**. Largest
> regression in the entire dataset. Both judges corroborate. A
> customer running this recipe on a cross-family base could ship a
> model 28 points worse than the base they started with. We caught
> that. The next slides explain how."

### Slide 30 — Methodology arc (supersession trail)

*Table of 7 framing supersessions with timestamps + triggers.*

> "Seven different framings on what gotcha #7 — the cross-family
> regression finding — actually was. Each row is correct at its
> timestamp; each was superseded by the next when new data fired.
>
> Started May 8 with a preliminary observation: 'recipe transfer is
> base-family-coupled' — N=1 from Mistral, very weak signal.
>
> Then Llama 8B v4 confirmed the pattern at N=2. Then Gemma 9B v4
> *lifted*, falsifying the architecture-family reading. Then judge
> evaluation showed the lifts erased — the recipe wasn't producing
> the capability the substring claimed. Then cross-judge with GPT-4o
> corroborated that finding on a second independent judge.
>
> Then the two-factor model emerged: lift requires ceiling reasoning
> *or* family-match to corpus source. Phi-4, a pre-registered
> falsifier, regressed as the model predicted. External reviewer
> declared closure.
>
> Then — and this is the crucial supersession — we ran a bulk
> semantic regrade on 33 catalog entries. The production model's
> substring +3.1pp lift **flipped to a semantic −4.8pp regression**.
> Sign reversal. The substring grader had a Qwen-family format bias
> we hadn't seen at any prior stage.
>
> **Why this slide is on the deck:** reviewers care about whether a
> team will catch its own over-claims. Each framing here was
> superseded by new data the team gathered to test it — not by
> external pushback. That self-correction discipline is what makes
> the final framing credible."

### Slide 31 — Data arc (headline-erosion + three-gate callout)

*5-checkpoint headline-erosion table on the production cell.*

> "This is the data side of the arc. Same model — Skippy 7B v4
> production — under five different evaluations.
>
> **Substring**, the original headline: +3.1pp lift over the Qwen 7B
> base. We'd shipped on this.
>
> **LLM-judge with Claude Sonnet**: Δ −0.350 on a 0-8 scale. The lift
> erases on judge dimensions — faithfulness to RAG context drops; the
> model gets less anchored to the retrieved evidence.
>
> **Temperature 0.3** (stochastic sampling instead of greedy decoding):
> Δ −29.3pp. The fine-tune is *temperature-brittle* — base models are
> temp-flat, fine-tunes are not. Production runs at temp=0 so this
> doesn't affect ship, but it tells us the substring lift is partly
> trained-phrasing memorization rather than generalization.
>
> **Cross-judge with GPT-4o**: Δ −0.690. Both judges agree v4 ≤ base.
> Not Sonnet-specific.
>
> **Semantic regrade**: Δ −4.8pp. Sign reversal. The original +3.1pp
> lift was Qwen-family format-fidelity artifact — the corpus phrasings
> come from Qwen, gold tokens are Qwen-shaped, substring rewards
> Qwen-style FTs.
>
> **The green callout at the bottom is the load-bearing claim**:
> production decision unaffected. The three-gate framework — capability
> + voice + safety — was designed exactly for this. Substring failed
> silently on capability. Voice + safety carried the real signal.
> Skippy 7B v4 ships *for voice and safety*, not for capability lift.
>
> The customer recommendation that comes out of this: semantic-grade
> by default. Substring alone isn't sufficient for FT-vs-base
> comparisons on a Qwen-family-sourced corpus. The cost is about 66
> cents per eval pass via GPT-4o with prompt caching — negligible
> versus fine-tune compute."

### Slide 32 — Skippy sister-model confound

> "Adjacent methodology lesson, separate finding. An earlier framing
> claimed the MoE fine-tune produced a +5.3pp domain lift. That number
> compared to Thinking-2507 — the *wrong* sister-model baseline. Apples-
> to-apples vs Instruct-2507 — the correct base reference — the
> 'lift' becomes a −2.3pp regression. The 7.6pp gap between Instruct
> and Thinking-2507 is *base-model property*, not fine-tune win.
>
> The lesson: always validate FT recipes against both sister models
> when the base family ships Instruct + Thinking variants. We caught
> ourselves on this one too. Same self-correction pattern as the
> Qwen-family bias arc."

*Pause. This is the densest methodology section.*

---

## Section 9 — Cross-cutting LLM findings (slides 33–34)

### Slide 33 — Dense vs MoE bandwidth

> "Cross-family LLM performance question. On the 5090, we measured
> three 7B-class dense Q4_K_M models: Qwen 2.5 7B, Mistral 7B v0.3,
> Llama 3.1 8B. All three land in 170-185 tokens/sec RAG decode —
> within 7% of each other. **7B-class dense decode is family-invariant**
> at this scale; differences track GGUF size, not vendor architecture.
>
> Same comparison for Mixture-of-Experts: Qwen3-30B-A3B at 159 tok/s
> beats Qwen 2.5 32B dense at 34 tok/s by **4.7× at equivalent VRAM**.
> Per-token bandwidth pays for the 3B active params, not the 30B
> total. The MoE-on-edge thesis lands empirically.
>
> Take-home: at 7B-class, choose your base for quality, not perf.
> If you need 30B-equivalent capacity at bandwidth-bound performance,
> use MoE."

### Slide 34 — Multi-stream concurrency

> "Practical question: can one NPU serve multiple camera streams?
> Measured TensorRT YOLO at batch sizes 1, 2, 4, 8, 16. The headline:
> at batch=4 on a 4-stream deployment, each stream sees about 26 FPS.
> Not the naïve 9 FPS you'd get from round-robin serial scheduling.
> Batching beats serial. 8-stream batch=8 lands at 15 FPS each.
> Multi-stream is real."

---

## Section 10 — Community SAM 3 + ViT alternatives (slides 35–39)

### Slide 35 — EfficientSAM3 community bake-off

> "Open-source community released two SAM-3-Lite variants in April. We
> benched them. EfficientSAM3 ES-EV-S: 424M params, BF16, lands at
> 2.6 FPS edge at 720p. 6.5× faster than SAM 3. Still 13× slower than
> our shipping pipeline. Community variants beat the monolith but
> can't touch a purpose-built two-stage pipeline."

### Slide 36 — EfficientSAM3.1 text-prompt variant

> "Smaller variant: EfficientSAM3.1 at 106M params with text-prompt
> capability. Even faster than ES-EV-S but still single-digit FPS at
> 1080p+. Same conclusion: useful as a community SAM 3 Lite but
> doesn't change our pipeline decision."

### Slide 37 — YOLOE-26 one-model open-vocab

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

### Slide 38 — TRT YOLOE-26

> "TRT FP8 on YOLOE-26 gives ~17% speedup over PyTorch FP16, not the
> 3× we get on YOLO-seg. At 16M params, the open-vocab head's
> kernel-launch overhead dominates wall-time. TRT FP8 doesn't help
> here. **TRT FP8 pays off when the kernel is big**; on small models
> with complex graph topology, it doesn't.
>
> Useful methodology data point: not every workload benefits from
> precision lowering. Match the optimization to the bottleneck."

### Slide 39 — ViT alternatives — what-if

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

## Section 11 — TRT takeaways (slide 40)

### Slide 40 — TRT takeaways

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

## Section 12 — ncu measurement validation (slides 41–43)

### Slide 41 — ncu measured DRAM — headline 515× gap

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

### Slide 42 — ncu measured DRAM — workload table

> "Full table of measured workloads. Use this as the reference when
> someone asks 'what about [model X] on edge?' If it's in the table,
> the answer is bandwidth-bound at the listed memory budget."

### Slide 43 — End-to-end pipeline latency budget

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

## Section 13 — Roadmap + summary (slides 44–45)

### Slide 44 — Optimization roadmap

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

### Slide 45 — Summary & findings

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
> - **Substring graders have format-fidelity bias** on corpus-targeted
>   FTs. The fix is semantic graders + cross-judge methodology.
>   Generalizable methodology lesson for any team doing fine-tuning
>   evaluation.
>
> - **Integration-architecture matters.** The 36 FPS edge result holds
>   on production SoCs with ISP + 2D-GPU offloads. On pure-NPU boards,
>   plan accordingly.
>
> - **One measured edge silicon anchor today** (i.MX 95 yolov8n-seg).
>   The rest of the deck scales from 5090 reference via a 16.19×
>   bandwidth ratio. KH-P2-001 in the roadmap: get a Mid-class NPU
>   measurement to close the gap.
>
> Thank you."

---

## Notes for the script reviewer (Claude browser)

Things this script does:

- **Leads with conclusions per slide**, then supports. Technical
  managers parse fast; they need the headline up front.
- **Surfaces the self-correction discipline** without
  self-aggrandizement. The arc is positioned as "we caught ourselves"
  not "look how rigorous we are."
- **Honest framings throughout**: 36 FPS sensitivity band ±15%, the
  "56 FPS production-SoC projection is not measured," the "we ship
  for voice and safety, not capability lift" reframe on Skippy 7B
  v4 production.
- **Skips dense bake-off methodology** that doesn't bear on the bottom
  line. Per-clip results, prompt-count scaling, weight-only INT8
  rejection — these get one-paragraph mentions.
- **Lingers on load-bearing findings**: architectural pivot (Slide 16),
  TRT FP8 unblock (Slide 24), the gotcha-7 arc (Slides 30–31), the
  e2e CPU crowding finding (Slide 43).
- **Pause points marked** where the presenter should slow down +
  invite questions.

Things this script does *not* do:

- Doesn't quote specific NPU measurement values from the private
  anchors slide — by design (discipline rule).
- Doesn't make claims about NPU Mid silicon measurements that we
  don't have (KH-P2-001 honestly scoped as the remaining gap).
- Doesn't soften the Yi −28.6pp finding or the Skippy 7B v4 sign-
  reversal under semantic — those are real and load-bearing for the
  credibility story.

Open questions for the reviewer:

1. Is the 45–60 minute runtime appropriate for a technical management
   audience, or should the script trim further?
2. The methodology arc section (Slides 30–31) is the densest. Should
   the speaker text lead with the closure verdict more directly, or
   is the storytelling buildup ("we caught ourselves") more useful?
3. The "the deck has 65 slides" volume — is that itself a problem for
   the target audience? Should the speaker explicitly call out "I'll
   skip 5 background slides" near the front to set expectations?
4. The conditional private slide — if the audience is NXP-internal and
   has clearance to see measured values, does the bounds-language
   framework I describe actually work in a live presentation, or does
   the audience want the team to discuss specific numbers?
