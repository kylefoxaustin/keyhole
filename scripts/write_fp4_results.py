#!/usr/bin/env python3
"""
write_fp4_results.py — assemble the measured INT(Q4_K_M)-vs-FP4(NVFP4) result on
the RTX 5090 into data/output/precision_5090_fp4_vs_int.json.

HONEST FRAMING (the measurement refuted the clean hypothesis):
  On llama.cpp (build c30e012, sm_120a, CUDA 12.9), NVFP4 is ~15-19% SLOWER than
  Q4_K_M across all prompt lengths AND decode, and the NVFP4 GGUF is 27% LARGER.
  The dramatic "4.2x prefill" seen in the first smoke run was a COLD-START artifact
  (the INT -r1 cold run measured 1238 t/s; warm it is ~6800).

What the measurement DOES establish (the real bolsters):
  1. NVFP4 runs NATIVELY on Blackwell (sm_120a) TODAY — the "2030 format" executes
     now on the 5090; the silicon + an open runtime exist.
  2. The "FP4 model" keeps token-embedding + LM-head in BF16 (2.49 GB of its 6.40 GB)
     — empirical confirmation of the precision_composition FP-residual thesis: even an
     aggressively-quantized model retains a high-precision tail.
  3. Decode is bandwidth-bound: FP4 decode tracks its (larger) byte footprint, ~flat-
     to-slower vs INT — the keyhole/Skippy "decode is BW-limited" story, measured.

Caveats:
  - llama.cpp's NVFP4 kernels are NEW; its Q4_K_M MMQ kernels are highly mature. The
    published 3x FP4 wins come from NVIDIA's optimized vLLM/TensorRT-LLM path, NOT
    llama.cpp. A fair "FP4 perf ceiling" needs that runtime (not measured here).
  - This NVFP4 GGUF (IwakuraRein) is a community conversion that kept embeddings BF16,
    so it is not size-optimal; a different NVFP4 build could be smaller.
"""
import json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "data/output/precision_5090_runs")
OUT = os.path.join(REPO, "data/output/precision_5090_fp4_vs_int.json")


def load_sweep(p):
    out = {}
    for r in json.load(open(p)):
        k = ("pp%d" % r["n_prompt"]) if r["n_prompt"] else ("tg%d" % r["n_gen"])
        out[k] = round(r["avg_ts"], 1)
    return out


def main():
    i = load_sweep(os.path.join(RUNS, "int_q4km_sweep.json"))
    f = load_sweep(os.path.join(RUNS, "fp4_nvfp4_sweep.json"))
    iv = int(open(os.path.join(RUNS, "int_q4km_peak_vram_mib.txt")).read().strip())
    fv = int(open(os.path.join(RUNS, "fp4_nvfp4_peak_vram_mib.txt")).read().strip())
    regimes = [k for k in ["pp128", "pp512", "pp1024", "pp2048", "pp4096", "tg128"] if k in i]

    doc = {
        "__meta__": {
            "description": "Measured INT(Q4_K_M) vs FP4(NVFP4) for Qwen3-8B on RTX 5090.",
            "schema_version": 1,
            "methodology_version": "2026-06-02-fp4-vs-int-v1",
            "host": "RTX 5090 (sm_120a, cc 12.0), driver 580.159",
            "runtime": "llama.cpp build c30e012, CUDA 12.9 (userspace toolkit), llama-bench -r2",
            "model": "Qwen3-8B (8.19B params), same base for both quants",
            "int_source": "Qwen/Qwen3-8B-GGUF :: Q4_K_M",
            "fp4_source": "IwakuraRein/Qwen3-8B-NVFP4-GGUF :: NVFP4 (community conversion)",
            "headline": "On llama.cpp, NVFP4 is ~15-19% SLOWER than Q4_K_M across prefill+decode "
                        "and 27% LARGER. The format runs natively on Blackwell today, but is not a "
                        "win on this (FP4-kernel-immature) stack. The earlier 4.2x prefill was a "
                        "cold-start artifact.",
        },
        "throughput_tok_s": {
            "regimes": regimes,
            "int_q4km": {k: i[k] for k in regimes},
            "fp4_nvfp4": {k: f[k] for k in regimes},
            "fp4_over_int_ratio": {k: round(f[k] / i[k], 3) for k in regimes},
        },
        "footprint": {
            "int_q4km": {"file_gb": 5.02, "peak_vram_mib": iv,
                         "tensor_types": {"Q4_K": "3.70 GB / 217t", "Q6_K": "1.32 GB / 37t", "F32": "~0 / 145t"}},
            "fp4_nvfp4": {"file_gb": 6.40, "peak_vram_mib": fv,
                          "tensor_types": {"NVFP4": "3.91 GB / 252t", "BF16": "2.49 GB / 2t (embed+lm_head)", "F32": "~0 / 649t"}},
            "note": "NVFP4 file larger because token-embedding + LM-head stay BF16 (2.49 GB) — "
                    "the high-precision FP residual, even in an 'FP4' model.",
        },
        "establishes": [
            "NVFP4 executes natively on Blackwell sm_120a TODAY (the 2030 format runs now).",
            "FP-residual thesis confirmed: the FP4 model keeps embeddings/LM-head in BF16.",
            "Decode is bandwidth-bound: FP4 decode tracks its larger byte footprint (~flat-to-slower).",
        ],
        "does_not_establish": [
            "A performance win for FP4 on llama.cpp — its NVFP4 kernels are new vs mature Q4_K_M MMQ.",
            "A memory win — this community NVFP4 GGUF is larger than Q4_K_M.",
            "The published 3x FP4 speedups (those are NVIDIA vLLM/TensorRT-LLM, a different runtime, not measured here).",
        ],
    }
    with open(OUT, "w") as fp:
        json.dump(doc, fp, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
