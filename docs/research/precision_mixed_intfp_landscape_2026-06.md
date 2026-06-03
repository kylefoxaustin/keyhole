# Mixed INT/FP precision landscape — survey + bake-off candidates (2026-06)

**Purpose.** Forward-looking (to ~2030) survey of emerging models and quantization
formats that exemplify **mixed INT/FP precision**, scoped to what is *available now in
early form* and *runnable on a single RTX 5090 (sm_120 Blackwell, ~32 GB)*. Feeds the
precision/FP4 deck — specifically the slide-7 **INT4-vs-FP4 asymmetry** argument.

**The load-bearing claim this bolsters.**
INT4 (GPTQ/AWQ) is a *memory* format: 4-bit weights cut bandwidth/footprint and win
**decode** (bandwidth-bound), but the GEMM dequantizes weights to bf16, so **prefill
stays on the bf16 compute floor**. FP4 (NVFP4/MXFP4) wins **both** memory *and* compute,
because Blackwell sm_120 has a **native FP4 tensor-core datapath**. FP8 sits in the
middle. Mixed schemes like **W4A8** (4-bit weights + 8-bit compute path) are the
textbook proof that splitting the weight-memory format from the compute format is what
escapes the prefill floor.

Method: 4 parallel web-research agents (one per angle below), claims cited inline.
HF repo existence + weight sizes verified against the HF API on 2026-06-03.

---

## 0. Existing measured anchor (this repo, vLLM 0.22, RTX 5090, single-stream)

Qwen3-8B, identical methodology (prefill pp128–4096, prefill-subtracted decode tg256):

| precision | decode tg256 (tok/s) | prefill peak (tok/s) | vs BF16 decode |
|---|---|---|---|
| BF16  | 96.7  | ~13.4k | 1.00× |
| FP8   | 150.0 | ~23.0k | 1.55× |
| NVFP4 | 217.0 | ~48.4k | **2.24×** |
| AWQ-INT4 | *(pending bake-off)* | *(predict ~bf16 floor on prefill)* | — |

NVFP4 already wins **both** axes (decode 2.24×, prefill 3.6× BF16). The pending AWQ-INT4
run on the *same base* completes the four-way and tests the asymmetry prediction
directly: INT4 decode should be high (BW-bound win) while INT4 prefill should sit near
the ~13k BF16 floor (dequant-to-bf16 compute path).

---

## 1. Within-model mixed dtype (W4A8, per-layer, mixed KV)

The on-theme point: **W4A8 keeps 4-bit weight memory savings AND runs the matmul on
INT8/FP8 tensor cores**, partially escaping the W4A16 prefill floor.

- **QServe / QoQ** (MIT HAN Lab, MLSys'25) — the canonical W4A8KV4. Authors' own line:
  *"all GEMM layers operate on W4A8 inputs, perform computation on INT8 tensor cores,
  and generate FP16 outputs"* — the citable hook for the whole asymmetry argument.
  ([arxiv 2405.04532](https://arxiv.org/abs/2405.04532),
  [hanlab project](https://hanlab.mit.edu/projects/qserve))
  Checkpoints exist: `mit-han-lab/Llama-3-8B-QServe` (+ Instruct, Qwen/Mistral/VILA).
  **5090 caveat:** OmniServe ships custom CUDA kernels with **no documented Blackwell
  build** → source-compile only, not turnkey.
- **Atom** (MLSys'24) — W4A4KV4 with INT8 outlier channels (mixed inside one matmul).
  Research kernels, pre-Blackwell. ([github efeslab/Atom](https://github.com/efeslab/Atom))
- **TensorRT-LLM W4A8 family** — explicit modes `W4A8_AWQ` (INT4 weights + **FP8**
  activations), and the Blackwell FP4-mixed modes `W4A8_NVFP4_FP8`, `W4A8_MXFP4_FP8`,
  `W4A8_MXFP4_MXFP8`. Self-quantize via TensorRT Model Optimizer; runs on Blackwell
  ≥ v0.17. ([TRT-LLM modes](https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/quantization/mode.html))
- **MoE mixed precision** (research): MxMoE (per-expert mixed-precision GroupGEMM, up to
  3.4×), EAQuant, DyMoE, MoQAE (mixed-precision KV). The *shipped* analog is
  MXFP4-experts + higher-precision-attention (gpt-oss pattern).
- **Mixed KV today:** mainline **vLLM KV cache is FP8-only** (`--kv-cache-dtype fp8`,
  needs cc > 8.9 → 5090 qualifies); INT8/INT4 KV not supported. Weights and KV are
  independent knobs, so INT4/NVFP4 weights + FP8 KV is a valid mixed config now.
  ([vllm docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/))

### ⚑ MEASURED on this box (2026-06-03): W4A8 does NOT run on sm_120 — the mixed-INT/FP "escape hatch" is Hopper-only

W4A8 is the textbook proof that splitting weight-memory (4-bit) from the compute path
(8-bit tensor cores) escapes the bf16 prefill floor. We tried to put a real W4A8 point on
the 5090 to sit *between* the INT4 floor and the FP4 high. **It cannot be done on consumer
Blackwell + current vLLM.** Probe: `scripts/probe_w4a8_5090.sh` attempted to load 4
downloadable W4A8 checkpoints (1B–8B) on vLLM 0.22.0 (torch 2.11+cu130, cap 12.0).
**Result: 0/4 loaded**, two distinct failure modes:

1. **W4A8-int** (int4 weight + int8 activation, e.g. `zera09/Llama-3.2-1B-Instruct-W4A8-GPTQ`):
   the compressed-tensors parser *accepts* the scheme, but **every one of vLLM's
   mixed-precision kernels rejects sm_120** (verbatim from the engine-init log):
   - `CutlassW4A8LinearKernel … CUTLASS W4A8 requires compute capability of 90 (Hopper)`
   - `MacheteLinearKernel … Machete requires compute capability of 90 (Hopper)`
   - `AllSparkLinearKernel … AllSpark currently does not support device_capability = 120`
   - `MarlinLinearKernel … Quant type (int4) not supported by Marlin, supported types are: [uint4b8, uint8b128, float8_e4m3fn, float4_e2m1f]`
   - `ConchLinearKernel … Weight type (int4) not supported`
   - `ExllamaLinearKernel … Exllama only supports float16 activations`
   → `ValueError` in `compressed_tensors_w4a8_int.py` → engine init fails. **No int4-weight +
   int8-activation kernel exists for cap 120.**
2. **W4A8-fp8 / W4AFP8** (int4 weight + fp8 activation, e.g.
   `czhu-cohere/Meta-Llama-3-8B-Instruct-W4A8-compressed-tensors-test`): vLLM 0.22.0
   doesn't even *register* the scheme — `NotImplementedError: No compressed-tensors
   compatible scheme was found` (the fp8-activation W4A8 path is unimplemented in this
   build; the CUTLASS kernel it would need is SM90-only anyway).

**The one-line takeaway (Marlin's own supported-types list):** Blackwell's sub-8-bit
compute path is **`float4_e2m1f` (NVFP4), not int4×int8**. The mixed-INT/FP escape hatch
that works on Hopper (QServe/QoQ, CUTLASS-W4A8, Machete) has **no sm_120 kernel** — so on
consumer Blackwell the *only* 4-bit-weight format that clears the bf16 prefill floor is
one with a native low-precision **compute** kernel: FP4/MXFP4. This is the same family of
sm_120-coverage gap as the broken FP4-MoE path (vllm#31085) and strengthens the slide-7
thesis: INT4 is memory-only here **and so is W4A8** — there's no INT-mixed way out, only FP4.
**Which GPUs DO run W4A8 (verified 2026-06-03) — the counterintuitive part:**

| GPU | cap | vLLM W4A8 | note |
|---|---|---|---|
| H100 / H200 (Hopper) | sm_90 | ✅ yes | CUTLASS-W4A8 + Machete; the supported tier |
| B200 / GB200 (DC Blackwell) | sm_100 | ❌ no | issue #35439 open/unfixed Jun 2026 |
| RTX 5090 (consumer Blackwell) | sm_120 | ❌ no | measured above |
| A100 / L40S / 4090 (Ampere/Ada) | sm_80/89 | ❌ no | no working real-W4A8 path |

**vLLM W4A8 is Hopper-only.** A $30–40k B200 is in the *same boat* as the 5090 — the
Machete/CUTLASS kernels use Hopper `wgmma` PTX never ported to either Blackwell variant.
More-expensive ≠ supported; it's the *previous* generation that has it. Two caveats:
(1) even on Hopper only the **fp8-activation scheme (W4AFP8)** genuinely works — the
int8-activation path is broken on *all* GPUs (vLLM #38064: silently runs as W4A16).
(2) **TensorRT-LLM** (different runtime) *does* run `W4A8_AWQ` on B200/sm_100 — but its
matrix is `Y` for sm_100/103 and `.` for **sm_120**, so the 5090 is excluded in *both*
vLLM and TRT-LLM. Net: W4A8 is a Hopper-generation artifact; the whole Blackwell line
(consumer + data-center) deprioritized INT4×INT8 for native FP4. To run W4A8 at all:
H100/H200 + vLLM (W4A8-FP8), or B200 + TensorRT-LLM (W4A8_AWQ).
([vLLM #35439](https://github.com/vllm-project/vllm/issues/35439),
[#38064](https://github.com/vllm-project/vllm/issues/38064),
[TRT-LLM quant matrix](https://nvidia.github.io/TensorRT-LLM/features/quantization.html))

## 2. Microscaling formats (MXFP4 / MXFP6 / MXFP8 / NVFP4)

The technical spine of slide 7. **Same E2M1 element; the entire difference is the scale:**

| | MXFP4 (OCP) | NVFP4 (NVIDIA) |
|---|---|---|
| element | E2M1 (4-bit) | E2M1 (4-bit) |
| block size | **32** | **16** (finer) |
| per-block scale | **E8M0** (8-bit, power-of-2 only) | **E4M3 FP8** (fractional) |
| 2nd-level scale | none | per-tensor FP32 |
| eff. bits/value | 4.25 | 4.5 |

NVIDIA's worked example: **E4M3 scale 0.08 MSE vs E8M0 0.72 MSE** on the same block —
that's the whole accuracy gap.
([OCP MX v1.0 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
[NVFP4 blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/))

**Native-format shipping models (downloadable now):**
- **gpt-oss-20b / -120b** — trained natively in MXFP4 (block-32/E8M0, 4.25 bpp, MoE
  experts). `openai/gpt-oss-20b` runs ~16 GB on a 5090 (256 tok/s).
  ([HF welcome-gpt-oss](https://huggingface.co/blog/welcome-openai-gpt-oss))
- **NVIDIA Model-Optimizer NVFP4 collection** (`nvidia/*-NVFP4`) — verified live
  2026-06-03: `nvidia/Qwen3-8B-NVFP4`, `nvidia/Llama-3.1-8B-Instruct-NVFP4`,
  `nvidia/Phi-4-reasoning-plus-NVFP4`, `nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4`,
  `nvidia/Qwen3-30B-A3B-NVFP4`, and newer mid-2026 releases
  `nvidia/Qwen3.6-35B-A3B-NVFP4` (23.4 GB, MoE), `nvidia/GLM-5-NVFP4` (480 GB,
  data-center). Most `nvidia/*` cards target TensorRT-LLM; for **vLLM** prefer the
  **`RedHatAI/*-NVFP4`** compressed-tensors mirrors.

**Hardware:** Blackwell sm_120 (RTX 5090, B200) = first native FP4 tensor cores. Roadmap:
**Rubin (H2 2026)** — 35 PFLOPS NVFP4 *training* / 50 PFLOPS inference, 3rd-gen
Transformer Engine with two-level NVFP4 micro-block scaling; **Rubin Ultra (H2 2027)**
100 PFLOPS FP4.
([ServeTheHome CES 2026](https://www.servethehome.com/nvidia-launches-next-generation-rubin-ai-compute-platform-at-ces-2026/))

## 3. Low-precision native *training* — the "2030 in early form today" angle

Distinction: native training = forward-pass GEMMs in FP8/FP4 *during* pretraining (vs the
PTQ-for-inference NVFP4 checkpoints above).

- **FP8 training is now routine.** DeepSeek-V3 (671B MoE, 14.8T tokens, loss err
  <0.25% vs BF16) was the anchor; **NVIDIA Nemotron-H-56B** pushed it to **20T tokens
  in FP8** (within 0.1% of BF16); Kimi-K2 (1T MoE) ships block-FP8. Open recipe:
  NeMo + Transformer Engine + Megatron.
  ([DeepSeek-V3 2412.19437](https://arxiv.org/abs/2412.19437),
  [Nemotron-H 2504.03624](https://arxiv.org/abs/2504.03624))
- **FP4 training is the active frontier — and it crossed the credibility line.**
  **NVIDIA "Pretraining LLMs with NVFP4"** trained a **12B model on 10 TRILLION tokens
  in NVFP4** and matched the FP8 baseline (MMLU-pro 62.58 vs 62.62).
  ([2509.25149](https://arxiv.org/abs/2509.25149))
  Independent confirmations converging on NVFP4: **Quartet** (IST Austria, NeurIPS'25;
  all-MXFP4 matmuls, ~2× FP8 on **RTX 5090**, [github IST-DASLab/Quartet](https://github.com/IST-DASLab/Quartet)),
  **FP4 All the Way** (7B/200B on Gaudi2, found NVFP4 optimal, [2505.19115](https://arxiv.org/abs/2505.19115)),
  **Metis**, **Microsoft FP4** (13B/100B, [2501.17116](https://arxiv.org/abs/2501.17116)).
- **Runnable hook:** no FP4-*trained* checkpoints are downloadable yet, but **Quartet's
  kernels are tuned for the RTX 5090** — the one local-reproducible FP4-native-training
  artifact.

### ⚑ MEASURED on this box (2026-06-03): FP4 training GEMMs run ~5.5× BF16 / ~2.9× FP8 on the 5090

We built the Quartet FP4-training kernels ([IST-DASLab/qutlass](https://github.com/IST-DASLab/qutlass)
v0.2.0, sm_120a) on the 5090 and measured the **forward-pass GEMM** speedup — the training-side
twin of the inference asymmetry. Same MXFP4/NVFP4 kernels Quartet trains in; weight pre-quantized,
activation quantized on-the-fly (cost included), output bf16. Real transformer layer shapes,
high-arithmetic-intensity token counts (M=4k–16k). Harness: `scripts/bench_fp4_training_gemm_5090.py`
→ `data/output/fp4_training_gemm_5090.json`.

| dtype | TFLOP/s (5090) | × BF16 | × FP8 |
|---|---|---|---|
| BF16 | ~230 | 1.0× | — |
| FP8 (torch `_scaled_mm` e4m3) | ~440 | ~1.9× | 1.0× |
| **MXFP4** (qutlass, block-32/E8M0) | **~1300** | **~5.5×** | **~2.9×** |
| **NVFP4** (qutlass, block-16/E4M3) | **~1270** | **~5.4×** | **~2.85×** |

Findings:
- **FP4 wins the *training* compute, not just inference** — ~5.5× BF16 / ~2.9× FP8 on the
  forward GEMM. This *exceeds* Quartet's headline (~4× BF16 / ~2.4× FP8) because our shapes are
  high-AI (the speedup grows with M/N/K, exactly as the paper predicts). Physically sane: tracks
  the 5090's tensor-core ratios (FP8 ~2× BF16 peak; FP4 ~1300 of ~1676 dense-FP4 peak).
- **MXFP4 ≈ NVFP4 in speed** → NVFP4's better numerics (the slide-7 spine) are **~free on
  throughput**. Clean "use NVFP4" conclusion for training too.
- Honest unit: this is the *forward-GEMM* number (cleanest kernel-level win, analogous to our
  inference prefill-GEMM result). A full training step (fwd + 2× bwd + non-GEMM) lands lower
  (~1.8× FP8 end-to-end per Quartet).
- 🚨 **qutlass MXFP8 is sm_100-only** (raises "Unsupported CUDA arch" on sm_120) — another
  consumer-Blackwell gap; we used torch-native FP8 as the baseline instead.
- 🚨 Toolchain: torch **cu128** (NOT cu130) so its major matches the CUDA 12.9 toolkit the
  `sm_120a` build needs; uv-managed Python 3.12 (system python3.12 ensurepip is broken);
  clone qutlass `--recursive` (CUTLASS submodule); `CUDA_HOME=/home/kyle/cuda-12.9`. Env
  `~/.virtualenvs/quartet_fp4`.

### ⚑ MEASURED (2026-06-03): FP4 training CONVERGES to BF16 quality (the accuracy half)

End-to-end run in the Quartet harness (`scripts/plot_fp4_convergence.py` parses the logs →
`data/output/fp4_training_convergence_5090.json`, slide 9). Trained a 30M-param Llama on
WikiText-103 twice, identical config + init, single 5090: BF16 baseline vs FP4 — the **full
Quartet recipe**: QuestMXFP4 4-bit weights+activations **AND** AlbertTseng 4-bit gradients
(stochastic) with the `Q(E)Q(Wt)t_Q(Et)Q(Xt)t` backward scheme. The FP4 val-loss curve
**tracks BF16 within ~0.07 nats the whole way**; final **BF16 4.914 (pp 136.2) vs FP4 4.984
(pp 146.0) — ~1.4% loss / ~7.2% perplexity gap.** So fully-4-bit training (incl. gradients)
reaches ~BF16 quality — the Quartet thesis ("native FP4 training can be optimal") confirmed
locally. Together with the GEMM-speed result above: **FP4 training is FAST *and* ACCURATE.**
- Honest framing: pseudo-quant (emulated FP4 numerics for an accuracy/convergence measurement;
  the *speed* is the kernel result, not this run).
- 🔧 **We fixed Quartet's triton-3.6 incompat ourselves (1 line):** the custom MXFP4 kernel
  (`mxfp4_triton.py`) passed `seed=None` to a non-constexpr `seed: int` kernel arg when
  `stochastic_round=False` → triton ≥3.x rejects it (`'NoneType' cannot be interpreted as
  integer`). `seed` is only read under the `stochastic_round=True` branch (dead-code-eliminated),
  so setting `seed=0` is safe — full recipe then runs at ~10 it/s (faster than the pure-pytorch
  HadamardFP4Clip fallback, which gave +0.044/1.04× forward-path-only as an interim result).
- 🚨 Quartet-harness gotchas (patched locally in `~/Documents/GitHub/Quartet`): missing
  `schedulefree`; `src/data/c4.py` does a module-level load of the **gated** `meta-llama/Llama-2-7b-hf`
  tokenizer (fires on any data import → 403) — guard it; the WikiText-103 S3 URL is dead (301) →
  Smerity mirror + a browser User-Agent (urllib default is 403'd); bits are NOT separate args
  (`--w-bits` etc. are unrecognized — the quantizer name encodes FP4); BF16 baseline = the
  defaults (`NoQuantizer` + `EW_EtX`).

## 4. INT4-vs-FP4 head-to-head (same base, both formats) — bake-off candidates

Same base in both INT4 and NVFP4, **dense** (so sm_120 FP4 tensor cores actually engage),
all HF paths verified live 2026-06-03 with real weight sizes:

| base | INT4 path | size | NVFP4 path | size | fit on 5090 |
|---|---|---|---|---|---|
| **Qwen3-8B** ⭐ | `Qwen/Qwen3-8B-AWQ` | 6.1 GB | `RedHatAI/Qwen3-8B-NVFP4` | 6.4 GB | both, twice over |
| Llama-3.1-8B | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` | ~5 GB | `nvidia/Llama-3.1-8B-Instruct-NVFP4` | ~5 GB | both |
| Qwen3-14B | `Qwen/Qwen3-14B-AWQ` | 10.0 GB | `RedHatAI/Qwen3-14B-NVFP4` | 10.5 GB | both |
| Qwen3-32B | `Qwen/Qwen3-32B-AWQ` | ~18 GB | `RedHatAI/Qwen3-32B-NVFP4` | ~19 GB | one format at a time |

Independent external validation of the asymmetry: a published guide measured
`Qwen3-8B-NVFP4` at **411 TPS vs 314 TPS for W4A16-AWQ on an RTX 5090** (vLLM 0.12,
~31% faster). ([2601.09527](https://arxiv.org/html/2601.09527v1))

### Gotchas (match this repo's prior findings)
- 🚨 **Avoid MoE NVFP4 for the compute-win claim** — native CUTLASS FP4 MoE kernels are
  broken/not selected on sm_120; vLLM falls back to Marlin with *"GPU does not have
  native support for FP4."* ([vllm#31085](https://github.com/vllm-project/vllm/issues/31085))
  Use **dense** models (8B/14B/32B).
- 🚨 **llama.cpp can only do same-base INT4-vs-FP4 for gpt-oss** (Q4_K vs native MXFP4);
  no NVFP4 in llama.cpp, no general MXFP4 requant. Keep the asymmetry on **vLLM**.
- 🚨 NVFP4/MXFP4 JIT on sm_120 needs `CUDA_HOME=/home/kyle/cuda-12.9` (12.6 fails);
  env `~/.virtualenvs/vllm_fp4`.

---

## Verification note (resolves an inter-agent conflict)

One research agent flagged `Qwen3.6-*`, `DeepSeek-V4-Pro`, `Gemma-4-31B`, `GLM-5` NVFP4
repos as likely hallucinated future names. **HF API check on 2026-06-03 returned 200 with
real weights for all of them** — they are genuine mid-2026 releases (e.g.
`nvidia/Qwen3.6-35B-A3B-NVFP4`, modified 2026-05-29, 313k downloads). They were simply
released after the agents' effective search horizon. Lesson: confirm HF repo + weight
files before trusting *or* dismissing a model name.

## Recommended next bake-off

Lead with **Qwen3-8B**: add the missing `Qwen/Qwen3-8B-AWQ` (INT4) run to the existing
NVFP4/FP8/BF16 trio → clean four-way on identical weights. Prediction to confirm: AWQ-INT4
decode high (BW-bound), prefill near the ~13k BF16 floor. Then Qwen3-14B as the scale-up.
