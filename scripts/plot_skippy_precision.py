#!/usr/bin/env python3
"""plot_skippy_precision.py — measured BF16/FP8/NVFP4 reader at the knee (vLLM, 5090)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(REPO, "data", "output", "skippy_precision_measured.json")))
order = ["bf16", "fp8", "nvfp4"]
lab = {"bf16": "BF16", "fp8": "FP8", "nvfp4": "NVFP4"}
C = {"bf16": "#9aa3b0", "fp8": "#4f9fd6", "nvfp4": "#10325f"}

ttft = [d[k]["ttft_ms"] for k in order]
dec = [d[k]["decode_ms_per_tok"] for k in order]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.6))
fig.suptitle("Visual-RAG reader at the 768px knee: precision measured directly on vLLM (RTX 5090)",
             fontsize=13.5, fontweight="bold", y=0.98)

for ax, vals, title, unit in [(a1, ttft, "Prefill / TTFT", "ms"),
                              (a2, dec, "Decode", "ms / token")]:
    bars = ax.bar([lab[k] for k in order], vals, color=[C[k] for k in order], edgecolor="white")
    for b, k in zip(bars, order):
        x = d[k].get("prefill_x" if unit == "ms" else "decode_x")
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.02,
                f"{b.get_height():.1f}{'ms' if unit=='ms' else ''}" + (f"\n{x}×" if x else ""),
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_ylabel(unit); ax.set_ylim(0, max(vals) * 1.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

a1.text(0.5, -0.22, "NVFP4 prefill only 1.44× (not the deck's pure-LLM 3.59×):\nthe vision encoder runs UN-quantized BF16 and dilutes it",
        transform=a1.transAxes, ha="center", fontsize=8.5, color="#c1121f", style="italic")
a2.text(0.5, -0.22, f"answer agreement vs BF16:  FP8 {d['fp8']['answer_agreement_vs_bf16']:.0%}  ·  "
        f"NVFP4 {d['nvfp4']['answer_agreement_vs_bf16']:.0%}\n(NVFP4 PTQ trades more reader consistency on this 4B VLM)",
        transform=a2.transAxes, ha="center", fontsize=8.5, color="#c1121f", style="italic")
fig.text(0.5, 0.02, "Qwen3-VL-4B (BF16 / Qwen-FP8 / nm-testing-NVFP4) · single-stream · "
         "Skippy gold pages @768px · NVFP4 = FlashInferCutlass kernel · aggregate only.",
         ha="center", fontsize=7.8, color="#888", style="italic")
fig.tight_layout(rect=[0, 0.10, 1, 0.94])
out = os.path.join(REPO, "data", "output", "skippy_precision_arm")
fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".svg")
print("wrote", out + ".png")
