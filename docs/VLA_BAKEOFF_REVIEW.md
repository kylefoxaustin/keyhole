# VLA Bake-off — reviewer briefing

**Audience:** an independent reviewer (Claude browser session or a human) doing a
critical review of the Keyhole VLA latency bake-off. Self-contained — assumes no
prior context. **Please push back** — the methodology has real judgment calls and
caveats flagged in § 6; the goal of this doc is to make them easy to attack.

**Scope:** this covers the *VLA inference-latency* bake-off only (the
vision-language-action robot models). It is separate from the core Keyhole vision
story (SAM 3 → Hybrid V2), which has its own briefing in
`docs/CLAUDE_REVIEW_BRIEFING.md`.

**Companion artifacts:**
- `scripts/bakeoff_vla.py` — the measurement harness (all five models).
- `data/inputs/vla_model_data.csv` — the catalog (one row per model; canonical for HF paths).
- `data/output/bakeoff/vla_summary_<key>.json` — per-model measurement output
  (gitignored; regenerate by running the harness). The schema is described in § 4.
- Consumer: `keyhole-sizer` (separate repo) projects these 5090 anchors to edge NPUs.

---

## 1. TL;DR

We measured **five published VLA models on an RTX 5090**, spanning **three
architecturally distinct action-generation topologies**. The headline is that the
topology — not parameter count — determines edge viability:

| Topology | Models | Per-action mechanism | Edge bottleneck |
|---|---|---|---|
| **Single-loop AR** | NORA-3B, OpenVLA-7B | autoregressive token decode through the full LLM | hard **bandwidth-wall** (every token streams all weights) |
| **Dual-loop flow-matching** | NORA-1.5, π0.5 | VLM once → small action expert runs N denoise steps → H-action chunk | VLM **amortized over the chunk**; fast loop has optimization headroom |
| **OFT parallel-chunk** | BitVLA | ONE VLM forward → H-action chunk via a parallel regression head | **prefill-shaped** (compute-bound); no AR-decode wall |

**The deck beat:** single-loop is bandwidth-walled on edge with no escape;
dual-loop turns *chunk size* into an amortization knob (π0.5's 50-action chunk →
367 Hz vs NORA-1.5's 5 → 27 Hz); OFT sidesteps the AR wall entirely by predicting
the chunk in one parallel pass.

**Everything here is an un-optimized stock-framework floor** — no CUDA graphs,
static KV cache, torch.compile, or specialized (e.g. ternary) kernels. These are
*lower bounds on achievable speed*, deliberately, so the edge projections are
conservative. See § 6 for why that's a feature and where it bites.

---

## 2. The five measurements (RTX 5090, bf16, n=20, p50)

| Model | Key | Topology | Action latency | Rate | Peak VRAM | Notes |
|---|---|---|---|---|---|---|
| NORA 3B | `nora_3b` | single-loop AR | e2e 79 ms (VLM 30.6 + FAST+ decode) | 12.6 Hz | 7.1 GB | Qwen2.5-VL + FAST+ |
| OpenVLA 7B | `openvla_7b_single` | single-loop AR | e2e 126.5 ms (VLM 46.6 + 7 tok) | 7.9 Hz | 14.4 GB | Llama-2 + discrete 256-bin |
| NORA-1.5 | `nora_1p5` | dual-loop | chunk 182.8 ms / 5 actions | **27 Hz** (fast-loop 32) | 7.6 GB | Qwen2.5-VL + ~228M expert |
| π0.5 | `pi_0p5` | dual-loop | chunk 136 ms / 50 actions | **367 Hz** (fast-loop 682) | 20.9 GB¹ | PaliGemma + ~430M Gemma expert |
| BitVLA | `bitvla` | OFT parallel-chunk | forward 123 ms / 8 actions | **65 Hz** | 6.07 GB | ternary SigLIP+LLM, OFT |

¹ π0.5 VRAM is high because it runs **bf16-AMP with float32 master weights** (see § 6.2).

**Per-component splits (vision encoder / LLM, p50 ms):**
- NORA-3B VLM 30.6 → split not isolated in the first run; later runs ~46/54%.
- OpenVLA-7B VLM 46.6 = vision 6.2 / LLM-prefill 40.4.
- NORA-1.5 VLM 26.5 = vision 13.3 / prefill 13.2; denoise step 15.8 ms ×10.
- π0.5 VLM 52.8 = vision 14.6 (×3 cameras) / prefill 38.2 (over 915 tokens);
  denoise step 7.2 ms ×10.
- BitVLA forward 123 = vision 22.6 (×2 cameras) / ternary-LLM 100.5.

---

## 3. The three topologies, precisely

### 3.1 Single-loop autoregressive (NORA-3B, OpenVLA-7B)
One VLM prefill produces the perception context, then the action is decoded
**token-by-token through the full LLM** (NORA emits variable-length FAST+ tokens to
EOS; OpenVLA emits 7 discrete 256-bin tokens). Each decode step re-reads the entire
weight set from memory → **bandwidth-bound**. Measured 5090 decode achieved-util is
**0.3% (NORA) / 0.5% (OpenVLA)** of peak compute — i.e. the GPU is ~99.7% idle on
compute during decode, streaming weights. This is the quantified bandwidth-wall and
it does not improve with more edge compute (only more edge bandwidth).

### 3.2 Dual-loop flow-matching (NORA-1.5, π0.5)
The VLM backbone runs **once per action chunk** → a frozen KV cache. A **separate,
much smaller action expert** then runs **N fixed flow-matching denoise steps**
(default 10) against that cache and emits a **whole chunk of H actions at once**.
- NORA-1.5: H=5 actions, expert ~228M (Qwen2.5-VL MoT cross-attention to the VLM KV).
- π0.5: H=50 actions, expert ~430M (Gemma-class, PaliGemma VLM).

Because the expensive VLM is amortized over H actions, **chunk size is the
amortization knob**. The fast loop's edge behavior depends on how bandwidth-bound
each denoise step is (§ 5.3).

### 3.3 OFT parallel-chunk (BitVLA)
OpenVLA-OFT does **ONE VLM forward** over `[image tokens ×cams + prompt +
ACTION_DIM·H action-placeholder tokens + proprio]`, and an L1-regression action head
reads the action-position hidden states **in parallel** → H actions (H=8) from a
single forward. **No AR loop, no denoise loop, no decode-per-token term.** The
forward is *prefill-shaped* (compute-bound, util ~9–14%), so it avoids the AR-decode
bandwidth-wall entirely — that's why BitVLA hits 65 Hz vs OpenVLA's 7.9 Hz despite a
similar VLM scale.

---

## 4. Measurement methodology

All in `scripts/bakeoff_vla.py`, dispatched by `resolve_family(vla_key)`:
- **Timing:** CUDA events around the relevant GPU call (generate / VLM forward /
  denoise loop / predict_action), p50/p95/p99 over n=20 trials after 3 warmups.
  Per-component splits use forward hooks (or, where the framework calls `.forward()`
  directly, a wrapped bound method) with paired CUDA events around the vision tower
  and the LLM/expert.
- **Validation:** every model runs an action-path validation (real 7-DOF vector or
  correct-shape finite action chunk) so we prove we're timing the real inference
  path, not a degenerate one. (This caught OpenVLA running text-only under
  transformers ≥4.57 — see § 6.4.)
- **Physical FLOP (hardware-independent):** `2·P·T` per module with **real parameter
  counts read off the loaded model** (attention-quadratic term omitted, <~2% at these
  seqlens). Stored at `result.flops`.
- **Achieved util cross-check:** physical FLOP ÷ measured p50 ÷ 209 TF bf16 peak
  (and, for the dual-loop fast loop, an effective-bandwidth estimate vs ~1.79 TB/s).
  This is *measured*, not assumed — it replaced an earlier effective-FLOP back-solve
  that baked in a wrong 0.85 vision-util assumption.
- **Schema** (`result.*`): `vlm_forward` (+ `components`), `action_forward` /
  `dual_loop`, `derived` (amortized ms/action + Hz), `dram`, `flops`
  (+ `achieved_util_5090`), `calibration` (narrative), `nvtx_labels`.

---

## 5. Cross-cutting findings (the ones worth scrutinizing)

### 5.1 Vision encoders are NOT compute-saturated at batch-1
Measured vision-encoder achieved util is **11% (NORA) / 29% (OpenVLA)**, not the
0.85 a prior model assumed. The 0.85 assumption had inflated NORA's vision FLOP ~7×
(2458 → 342 GF). Edge projections that assume saturated ViTs are wrong.

### 5.2 The AR-decode bandwidth-wall is real and quantified
0.3–0.5% decode util ⇒ single-loop VLAs are pure weight-streaming on decode. On a
94 GB/s-effective edge NPU, NORA-3B projects 12.6 → ~1.8 Hz, and a higher-compute
tier buys ~nothing (still BW-bound). This is the strongest "needs architectural
change, not more TOPS" result.

### 5.3 The dual-loop denoise bottleneck is per-model, not a single label
We initially labeled NORA-1.5's denoise "launch-bound" and were tempted to reuse it.
**Don't.** The honest decomposition (which the sizer adopted) splits each denoise
step into a silicon-independent launch constant + a genuinely bandwidth-bound
fraction, driven by the **measured per-model effective bandwidth**:
- NORA-1.5: denoise step ~1.6% of peak BW (29 GB/s eff) → **launch-dominated**,
  big edge headroom (~1.4× degradation to a High NPU).
- π0.5: denoise step ~13.4% of peak BW (240 GB/s eff) → **partial-BW**, degrades
  ~4.2× on the same edge tier. (Its 430M fp32 expert over 50 tokens streams real bytes.)

### 5.4 Chunk size is the amortization knob
π0.5 (50-action chunk) and NORA-1.5 (5-action chunk) have similar *chunk latency*
(~136 vs 183 ms) but π0.5 gets 10× the actions per VLM forward → 367 Hz vs 27 Hz.
This is the cleanest "dual-loop wins by amortization" demonstration.

### 5.5 OFT parallel-chunk dodges the AR wall
BitVLA's single parallel forward is prefill-shaped → 65 Hz, ~8× OpenVLA's AR rate at
comparable VLM scale, with no per-token weight re-streaming.

---

## 6. Caveats & judgment calls — **attack these**

### 6.1 Stock-framework, un-optimized floor (all models)
No CUDA graphs / static cache / torch.compile / fused kernels. Real deployments
would be faster (the cited papers use optimized stacks: e.g. NORA's 33 ms anchor,
π0.5's 50 Hz, BitVLA's 4.4× claim). **We measure the floor on purpose** so edge
projections are conservative, but it means our absolute Hz are *lower bounds*, and
the gap-to-paper is "optimization headroom," not error. Push back if you think a
floor is the wrong baseline for an exec story.

### 6.2 π0.5 runs bf16-AMP (float32 master weights), not true bf16 weights
pi05's flow-matching expert hardcodes float32 internals (sinusoidal time embeddings,
the openpi convention), so casting weights to bf16 breaks it. We run lerobot's
mixed-precision path (autocast bf16 matmuls over float32 master weights). Consequence:
the **20.9 GB VRAM and the denoise effective-BW are float32-weighted upper bounds**;
true bf16-weight deployment would roughly halve both (→ more launch-leaning denoise).
The *compute-bound stages' latency* is representative of bf16; the *bandwidth* terms
are not. This is the biggest apples-to-oranges in the table.

### 6.3 BitVLA's ternary is NOT realized as a speed/bandwidth win here
The HF "bf16" checkpoint runs ternary BitLinear as **dense bf16 matmuls** (weights
stored bf16, 5.47 GB — not packed ~1.58-bit/0.2-byte). So:
- The **6 GB VRAM** is the only realized ternary win, and it's bf16-stored (not the
  paper's ~1.4 GB packed figure).
- The measured **latency is OpenVLA-OFT-class bf16** — ternary gives no compute/BW/
  latency benefit in this path.
- The paper's 4.4× speedup / 11× memory and the "0.2 byte/param decode-BW" story
  **require bitblas/LUT ternary kernels we did not run.** Treat any ternary BW/speed
  number as a *separate, optimistic, kernel-dependent projection*, explicitly flagged
  — not as measured. This is the single most important caveat for anyone citing
  BitVLA as an "INT/ternary edge floor."

### 6.4 Environment pins are load-bearing and fragile (per-model venvs)
Each model needed a different, sometimes painful, environment. Wrong env → silently
wrong results (OpenVLA ran **text-only** under transformers ≥4.57, dropping
pixel_values — caught only by the acceptance gate). Recipes in § 7. A reviewer should
sanity-check that each measurement used the right env (the JSON records torch/
transformers/lerobot versions).

### 6.5 Content-invariant inputs / synthetic observations
VLA latency at fixed input size is content-invariant (token counts + graph dominate),
so we use one representative frame, zero proprio state, and (BitVLA) center_crop off.
This is fine for latency; it means the *action values* aren't task-meaningful (the
shape/finiteness validation is what we rely on). Push back if you think content
matters for any timed path (we argue it doesn't).

### 6.6 BitVLA console logs are suppressed; CSV/param corrections
prismatic's `overwatch` raises the root log level, so BitVLA's console result block
is swallowed — **the JSON artifact is canonical**, not stdout. Several CSV figures
were corrected from measurement (NORA-1.5 expert 800M→228M; π0.5 expert 300M→430M,
hf_repo `lerobot/pi0_5`→`lerobot/pi05_base`; BitVLA single_loop→oft_parallel_chunk,
hf_repo filled). The CSV is now reconciled to measured values.

### 6.7 FLOP convention
`2·P·T` matmul-only, attention-quadratic omitted. For π0.5's 915-token prefix and
BitVLA's long action-placeholder sequence this is a (small) underestimate. Decode/
denoise FLOP runs through the LLM/expert body + head with no standalone action-head
term (no double-count).

---

## 7. Reproduction — environments (the load-bearing part)

| Model | venv | Stack | Gotcha |
|---|---|---|---|
| NORA-3B | `~/.virtualenvs/keyhole` | transformers 5.x, torch 2.11 cu128 | vanilla Qwen2.5-VL; action logic mirrored from declare-lab/nora |
| OpenVLA-7B | `~/.virtualenvs/openvla` | **transformers 4.40.1** | runs **text-only** on ≥4.57 — pinned venv + acceptance gate mandatory |
| NORA-1.5 | `~/.virtualenvs/nora15` | **transformers 4.54.1** | vendored `scripts/vendor/nora15_modelling_expert.py` (TF-stripped); MoT attn reads legacy tuple KV cache |
| π0.5 | `~/.virtualenvs/pi05` | **Python 3.12** + lerobot **0.5.2 from git** (PyPI 0.4.4 gates on an unshipped patched-transformers); built with `uv` | bf16-AMP only (§ 6.2); pass `noise=` of matching dtype |
| BitVLA | `~/.virtualenvs/bitvla` | Python 3.10, torch cu128, BitVLA transformers fork v4.51 + openvla-oft `prismatic` editable from a clone (`$KEYHOLE_BITVLA_OFT`) | prismatic eagerly imports RLDS/dlimp/TF training pipeline — `run_bitvla` injects permissive stub modules; `set_constant()` required after load |

Run: `python scripts/bakeoff_vla.py --model-key <key> --n-trials 20 --warmup 3`
(in the matching venv). Output → `data/output/bakeoff/vla_summary_<key>.json`.

---

## 8. Questions for the reviewer

1. Is the **un-optimized floor** (§ 6.1) the right baseline for the edge-viability
   story, or should we report an optimized number alongside?
2. π0.5's **bf16-AMP vs true-bf16** (§ 6.2) — is the float32-weighted VRAM/BW an
   acceptable caveat, or does it need a true-bf16 re-measurement to be comparable?
3. BitVLA's **ternary-not-realized** (§ 6.3) — is flagging it as a separate optimistic
   projection sufficient, or does an honest "BitVLA edge floor" require running the
   bitblas kernels?
4. Is the **launch + BW decomposition** of the dual-loop denoise step (§ 5.3) sound,
   or is there a better way to project a launch-bound op to a different silicon?
5. Anything in the **per-component split methodology** (§ 4) that could mis-attribute
   latency between vision and LLM (e.g. hook placement, embed-merge accounting)?
