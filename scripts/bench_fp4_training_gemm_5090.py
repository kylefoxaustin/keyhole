#!/usr/bin/env python3
"""
bench_fp4_training_gemm_5090.py — FP4 TRAINING-kernel speedup on the RTX 5090 (sm_120).

The inference bake-off showed FP4 (NVFP4) wins decode AND prefill at inference. This is the
TRAINING-side complement: do the forward-pass GEMMs of low-precision *training* actually run
faster in FP4 than BF16/FP8 on consumer Blackwell? Reproduces the Quartet result
("Native FP4 Training Can Be Optimal for LLMs", arXiv 2505.14669, Fig.3 — up to ~4x BF16 /
~2.4x FP8 forward) using IST-DASLab/qutlass kernels, the same MXFP4/NVFP4 GEMMs Quartet trains in.

Providers (all output bf16):
  - torch-bf16   : F.linear (the baseline a low-precision run must beat)
  - mxfp4        : qutlass MXFP4 (block-32 / E8M0) with on-the-fly activation quant (realistic)
  - nvfp4        : qutlass NVFP4 (block-16 / E4M3) with on-the-fly activation quant (Quartet-II fmt)
  - fp8          : torch._scaled_mm e4m3 (qutlass MXFP8 is sm_100-only, so use torch-native FP8)

Weight is pre-quantized once (as in training: quantize-per-step is amortized by the matmul);
activation is quantized on the fly each call (the honest per-step cost). Speedup grows with
arithmetic intensity (M, N, K), so we sweep large token counts on real Llama/Qwen layer shapes.

Run:  CUDA_HOME=/home/kyle/cuda-12.9 ~/.virtualenvs/quartet_fp4/bin/python \
        scripts/bench_fp4_training_gemm_5090.py
Writes data/output/fp4_training_gemm_5090.json
"""
import json, os
import torch
import triton

import qutlass
from qutlass import (matmul_mxf4_bf16_tn, fusedQuantizeMx,
                     matmul_nvf4_bf16_tn, fusedQuantizeNv)
from qutlass.utils import to_blocked
from scipy.linalg import hadamard

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTJSON = os.path.join(REPO, "data/output/fp4_training_gemm_5090.json")
DEV = "cuda"
HAD = 128  # hadamard rotation block (divides every K below)

# (label, K, N) — real transformer linear-layer shapes (weight is N x K).
SHAPES = [
    ("Qwen3-8B qkv",  4096, 12288),
    ("Qwen3-8B FFN",  4096, 24576),
    ("Llama-70B FFN", 8192, 57344),
]
TOKENS = [4096, 16384]   # M = tokens (batch*seq); compute-bound regime where FP4 pays off


def hadamard_matrix(n, dtype, device):
    return torch.tensor(hadamard(n) * n ** -0.5, dtype=dtype, device=device)


def bf16_runner(a, b):
    return lambda: torch.nn.functional.linear(a, b)


def mxfp4_runner(a, b, H):
    w_e2m1, w_e8m0 = fusedQuantizeMx(b, H)
    w_scale = to_blocked(w_e8m0)
    alpha = torch.tensor([1.0], device=DEV)

    def run():  # on-the-fly activation quant (realistic per-step cost)
        a_e2m1, a_e8m0 = fusedQuantizeMx(a, H)
        return matmul_mxf4_bf16_tn(a_e2m1, w_e2m1, to_blocked(a_e8m0), w_scale, alpha)
    return run


def nvfp4_runner(a, b, H):
    alpha = torch.tensor([1.0], device=DEV)
    gscale = torch.tensor([1.0], device=DEV)
    w_e2m1, w_e8m0 = fusedQuantizeNv(b, H, gscale)
    w_scale = to_blocked(w_e8m0)

    def run():
        a_e2m1, a_e8m0 = fusedQuantizeNv(a, H, gscale)
        return matmul_nvf4_bf16_tn(a_e2m1, w_e2m1, to_blocked(a_e8m0), w_scale, alpha)
    return run


def fp8_runner(a, b):
    # torch-native scaled FP8 (e4m3). b is N x K -> b.t() gives K x N (column-major) for _scaled_mm.
    b8 = b.to(torch.float8_e4m3fn)
    bt = b8.t()
    sa = torch.tensor(1.0, device=DEV)
    sb = torch.tensor(1.0, device=DEV)

    def run():
        a8 = a.to(torch.float8_e4m3fn)
        return torch._scaled_mm(a8, bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    return run


def tflops(ms, M, N, K):
    return (2 * M * N * K) * 1e-12 / (ms * 1e-3)


def bench(fn):
    ms = triton.testing.do_bench(fn, warmup=10, rep=50, quantiles=[0.5])
    return float(ms if not isinstance(ms, (list, tuple)) else ms[0])


def main():
    os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
    print(f"device {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)} "
          f"torch {torch.__version__} qutlass {getattr(qutlass,'__version__','?')}")
    rows = []
    for sl, K, N in SHAPES:
        H = hadamard_matrix(HAD, torch.bfloat16, DEV)
        for M in TOKENS:
            a = torch.randn((M, K), device=DEV, dtype=torch.bfloat16)
            b = torch.randn((N, K), device=DEV, dtype=torch.bfloat16)
            res = {"shape": sl, "M": M, "N": N, "K": K, "tflops": {}, "speedup_vs_bf16": {}}
            runners = {"bf16": bf16_runner(a, b),
                       "mxfp4": mxfp4_runner(a, b, H),
                       "nvfp4": nvfp4_runner(a, b, H)}
            try:
                runners["fp8"] = fp8_runner(a, b)
            except Exception as e:
                print(f"  [fp8 skipped: {e}]")
            for name, fn in runners.items():
                try:
                    fn()  # correctness/warm touch
                    ms = bench(fn)
                    tf = tflops(ms, M, N, K)
                    res["tflops"][name] = round(tf, 1)
                except Exception as e:
                    print(f"  [{name} failed @ {sl} M={M}: {e}]")
            bf = res["tflops"].get("bf16")
            for name, tf in res["tflops"].items():
                if bf:
                    res["speedup_vs_bf16"][name] = round(tf / bf, 2)
            # vs-fp8 too (the harder baseline)
            f8 = res["tflops"].get("fp8")
            if f8:
                for name in ("mxfp4", "nvfp4"):
                    if name in res["tflops"]:
                        res.setdefault("speedup_vs_fp8", {})[name] = round(res["tflops"][name] / f8, 2)
            rows.append(res)
            su = res["speedup_vs_bf16"]
            print(f"{sl:14s} M={M:6d} N={N:6d} K={K:6d} | "
                  f"bf16 {res['tflops'].get('bf16','-')}  fp8 {res['tflops'].get('fp8','-')}  "
                  f"mxfp4 {res['tflops'].get('mxfp4','-')}  nvfp4 {res['tflops'].get('nvfp4','-')} TFLOP/s "
                  f"|| xBF16: mxfp4 {su.get('mxfp4','-')} nvfp4 {su.get('nvfp4','-')} fp8 {su.get('fp8','-')}")

    doc = {
        "__meta__": {
            "description": "FP4 training-GEMM speedup on RTX 5090 (sm_120) via qutlass MXFP4/NVFP4 "
                           "kernels (the Quartet FP4-native-training kernels). Forward-pass GEMM, "
                           "weight pre-quantized + on-the-fly activation quant. Output bf16.",
            "reproduces": "Quartet (arXiv 2505.14669) Fig.3 forward-pass FP4 speedup, measured on 5090",
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "fp8_baseline": "torch._scaled_mm e4m3 (qutlass MXFP8 is sm_100-only, unsupported on sm_120)",
            "hadamard_block": HAD,
        },
        "rows": rows,
    }
    json.dump(doc, open(OUTJSON, "w"), indent=2)
    print("wrote", OUTJSON)


if __name__ == "__main__":
    main()
