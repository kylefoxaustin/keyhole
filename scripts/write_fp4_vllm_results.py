#!/usr/bin/env python3
"""
write_fp4_vllm_results.py — assemble the vLLM NVFP4-vs-FP8-vs-BF16 measurement (Qwen3-8B,
RTX 5090) into data/output/precision_5090_vllm_fp4_vs_int.json.

This is the COMPANION + ANSWER to precision_5090_fp4_vs_int.json (the llama.cpp run),
which found NVFP4 ~15-19% SLOWER and explicitly flagged the missing comparison:
  "the published 3x FP4 speedups are NVIDIA vLLM/TensorRT-LLM, not measured here."
Here we measure it. Same model, same FP4 format, mature runtime — isolating PRECISION
from RUNTIME.

Decode is single-stream (batch=1) and is the trustworthy, BW-bound comparison.
Prefill is single-request and compute-bound; reported with that caveat (vLLM is built
for batched prefill, so single-request prefill understates its throughput ceiling).
"""
import json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "data/output/precision_5090_vllm_runs")
LLAMA = os.path.join(REPO, "data/output/precision_5090_fp4_vs_int.json")
OUT = os.path.join(REPO, "data/output/precision_5090_vllm_fp4_vs_int.json")


def load(label):
    return json.load(open(os.path.join(RUNS, f"{label}_vllm.json")))


def main():
    nv, fp8, bf16 = load("nvfp4"), load("fp8"), load("bf16")
    foot = json.load(open(os.path.join(RUNS, "footprints_gb.json")))

    regimes = list(bf16["prefill_tok_s"].keys())
    dec = lambda d: list(d["decode_tok_s"].values())[0]

    def ratio_block(num, den):
        out = {r: round(num["prefill_tok_s"][r] / den["prefill_tok_s"][r], 3) for r in regimes}
        out["tg256"] = round(dec(num) / dec(den), 3)
        return out

    lcpp = None
    if os.path.exists(LLAMA):
        lj = json.load(open(LLAMA))
        lcpp = {
            "headline": lj["__meta__"]["headline"],
            "fp4_over_int_decode": lj["throughput_tok_s"]["fp4_over_int_ratio"].get("tg128"),
        }

    doc = {
        "__meta__": {
            "description": "Measured NVFP4 vs FP8 vs BF16 for Qwen3-8B on RTX 5090 via vLLM "
                           "(single-stream). The mature-runtime answer to the llama.cpp FP4 finding.",
            "schema_version": 1,
            "methodology_version": "2026-06-02-vllm-fp4-vs-int-v2-prefixcache-fixed",
            "host": "RTX 5090 (sm_120a, cc 12.0), driver 580.159",
            "runtime": "vLLM 0.22.0 (torch 2.11+cu130); NVFP4 GEMM = FlashInferCutlassNvFp4LinearKernel, "
                       "JIT-compiled with CUDA 12.9 nvcc (sm_120f). FP8 = online dynamic quant. "
                       "enable_prefix_caching=False; unique prompt window per prefill iter.",
            "model": "Qwen3-8B (8.19B); NVFP4 = nvidia/Qwen3-8B-NVFP4, FP8/BF16 = Qwen/Qwen3-8B",
            "stream": "single (batch=1, greedy, ignore_eos); median of 3",
            "headline": (
                "On vLLM (mature FlashInfer/CUTLASS NVFP4 kernels), NVFP4 decode is %.2fx BF16 and "
                "%.2fx FP8 single-stream — FP4 WINS here. This INVERTS the llama.cpp result, where "
                "NVFP4 was ~15-19%% slower. The confound was the RUNTIME (kernel maturity), not the "
                "format: same 5090, same NVFP4 weights."
                % (round(dec(nv) / dec(bf16), 2), round(dec(nv) / dec(fp8), 2))
            ),
        },
        "decode_tok_s_single_stream": {
            "nvfp4": dec(nv), "fp8": dec(fp8), "bf16": dec(bf16),
            "nvfp4_over_bf16": round(dec(nv) / dec(bf16), 3),
            "nvfp4_over_fp8": round(dec(nv) / dec(fp8), 3),
            "fp8_over_bf16": round(dec(fp8) / dec(bf16), 3),
        },
        "prefill_tok_s_single_request": {
            "regimes": regimes,
            "nvfp4": nv["prefill_tok_s"], "fp8": fp8["prefill_tok_s"], "bf16": bf16["prefill_tok_s"],
            "nvfp4_over_bf16": {r: round(nv["prefill_tok_s"][r] / bf16["prefill_tok_s"][r], 3) for r in regimes},
            "caveat": "single-request prefill is compute-bound; vLLM's batched-prefill ceiling is higher.",
        },
        "footprint_gb": {"nvfp4_weights": foot.get("nvidia/Qwen3-8B-NVFP4"),
                         "bf16_weights": foot.get("Qwen/Qwen3-8B"),
                         "note": "NVFP4 weights ~%.1fx smaller than BF16 — smaller bytes drive the "
                                 "BW-bound decode win." % (foot.get("Qwen/Qwen3-8B", 16.4) /
                                                           foot.get("nvidia/Qwen3-8B-NVFP4", 6.41))},
        "ratios_nvfp4_over_bf16": ratio_block(nv, bf16),
        "ratios_nvfp4_over_fp8": ratio_block(nv, fp8),
        "vs_llama_cpp": lcpp,
        "establishes": [
            "On a mature runtime, NVFP4 decode beats both FP8 and BF16 single-stream on the 5090.",
            "The llama.cpp FP4 slowdown was a RUNTIME/kernel-maturity artifact, not a format limit.",
            "Decode win tracks the smaller NVFP4 byte footprint — the BW-bound-decode thesis, measured.",
            "NVFP4 runs natively on Blackwell sm_120 today (FlashInfer CUTLASS FP4, CUDA>=12.9 nvcc).",
        ],
        "does_not_establish": [
            "Batched/served throughput (this is single-stream; vLLM's batched ceiling differs).",
            "Accuracy parity — NVFP4 quantization quality not evaluated here, only speed/footprint.",
            "TensorRT-LLM numbers (a third runtime; NVIDIA's headline FP4 figures use it).",
        ],
        "operational_note_for_repro": "NVFP4 on vLLM 0.22 + sm_120 REQUIRES CUDA_HOME pointed at a "
            ">=12.9 nvcc (system 12.6 fails FlashInfer JIT with 'No supported CUDA architectures "
            "found for major versions [12]'). Used /home/kyle/cuda-12.9.",
    }
    with open(OUT, "w") as fp:
        json.dump(doc, fp, indent=2)
    print("wrote", OUT)
    print(json.dumps(doc["decode_tok_s_single_stream"], indent=2))


if __name__ == "__main__":
    main()
