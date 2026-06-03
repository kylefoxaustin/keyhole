#!/usr/bin/env python3
"""
plot_fp4_lifecycle.py — slide 8: FP4 wins across the whole lifecycle, and the win GROWS
with compute-intensity. Ties the inference asymmetry (slides 5-7) to the new training-GEMM
result into one message.

Three workloads, ordered by compute-intensity (memory-bound -> compute-bound):
  Decode (inference, BW-bound)  ->  Prefill (inference, compute-bound)  ->  Training fwd-GEMM
Each shows FP8 and FP4(NVFP4) speedup over BF16, all on the same RTX 5090. FP4 wins every
stage, and its margin widens as the work gets more compute-bound — because FP4 is a COMPUTE
format (native sm_120 tensor cores), so it pays off most where compute dominates.

Inference numbers: data/output/precision_5090_breadth.json (same-base Qwen3-8B quad).
Training numbers:  data/output/fp4_training_gemm_5090.json (qutlass MXFP4/NVFP4 fwd GEMM).
Writes data/output/precision_5090_fp4_lifecycle.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data/output/precision_5090_fp4_lifecycle.png")
BREADTH = os.path.join(REPO, "data/output/precision_5090_breadth.json")
TRAIN = os.path.join(REPO, "data/output/fp4_training_gemm_5090.json")

FP8_C = "#2b6cb0"
FP4_C = "#2f855a"


def mean(xs):
    return sum(xs) / len(xs)


def main():
    bd = json.load(open(BREADTH))["same_base_quad_qwen3_8b"]
    bf16_dec = bd["BF16"]["decode_tok_s"]; bf16_pre = bd["BF16"]["prefill_peak_tok_s"]
    decode = {"fp8": bd["FP8"]["decode_tok_s"] / bf16_dec, "fp4": bd["NVFP4"]["decode_tok_s"] / bf16_dec}
    prefill = {"fp8": bd["FP8"]["prefill_peak_tok_s"] / bf16_pre, "fp4": bd["NVFP4"]["prefill_peak_tok_s"] / bf16_pre}

    tr = json.load(open(TRAIN))["rows"]
    train = {"fp8": mean([r["speedup_vs_bf16"]["fp8"] for r in tr if "fp8" in r["speedup_vs_bf16"]]),
             "fp4": mean([r["speedup_vs_bf16"]["nvfp4"] for r in tr if "nvfp4" in r["speedup_vs_bf16"]])}

    groups = [("Decode\n(inference · BW-bound)", decode),
              ("Prefill\n(inference · compute-bound)", prefill),
              ("Training fwd-GEMM\n(most compute-bound)", train)]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = range(len(groups)); w = 0.36
    for i, (lab, d) in enumerate(groups):
        b8 = ax.bar(i - w/2, d["fp8"], w, color=FP8_C, edgecolor="white",
                    label="FP8" if i == 0 else None, zorder=3)
        b4 = ax.bar(i + w/2, d["fp4"], w, color=FP4_C, edgecolor="white",
                    label="FP4 (NVFP4)" if i == 0 else None, zorder=3)
        ax.annotate(f"{d['fp8']:.1f}×", (i - w/2, d["fp8"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color=FP8_C, fontweight="bold")
        ax.annotate(f"{d['fp4']:.1f}×", (i + w/2, d["fp4"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10, color=FP4_C, fontweight="bold")

    ax.axhline(1.0, ls=":", color="#a0aec0", lw=1.2, zorder=1)
    ax.annotate("BF16 baseline (1×)", (2.42, 1.0), textcoords="offset points", xytext=(0, 4),
                fontsize=8, color="#718096", ha="right")
    # the growth arrow under the x-axis story
    ax.annotate("", xy=(2.25, 5.9), xytext=(-0.25, 5.9),
                arrowprops=dict(arrowstyle="->", color="#9b2c2c", lw=1.6), zorder=2)
    ax.annotate("FP4's margin GROWS as work gets more compute-bound", (1.0, 6.0),
                ha="center", fontsize=9, color="#9b2c2c", fontweight="bold")

    ax.set_xticks(list(x)); ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
    ax.set_ylabel("speedup over BF16  (same RTX 5090)")
    ax.set_ylim(0, 6.6)
    ax.set_title("FP4 wins the whole lifecycle — inference AND training\n"
                 "RTX 5090 · Qwen3-8B inference (vLLM) + FP4-native training GEMMs (qutlass/Quartet)",
                 fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.012, 0.74,
            "FP4 is a COMPUTE format (native sm_120 tensor cores),\n"
            "so it pays off most where compute dominates.\n"
            "Training fwd-GEMM: MXFP4 ≈ NVFP4 in speed → NVFP4's\n"
            "better numerics are ~free.",
            transform=ax.transAxes, va="top", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7fafc", ec="#cbd5e0", lw=0.8))
    fig.tight_layout(); fig.savefig(OUT, dpi=130)
    print("wrote", OUT)
    print(f"decode  fp8 {decode['fp8']:.2f} fp4 {decode['fp4']:.2f}")
    print(f"prefill fp8 {prefill['fp8']:.2f} fp4 {prefill['fp4']:.2f}")
    print(f"train   fp8 {train['fp8']:.2f} fp4 {train['fp4']:.2f}")


if __name__ == "__main__":
    main()
