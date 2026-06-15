#!/usr/bin/env python3
"""plot_vision_quant_combined.py — FP8 vs FP4: where prefill savings come from.
Reads vision_quant_uncap.json (FP8) + vision_quant_uncap_fp4.json (FP4). Synthetic image, no personal data."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f8 = json.load(open(f"{REPO}/data/output/vision_quant_uncap.json"))["arms"]
f4 = json.load(open(f"{REPO}/data/output/vision_quant_uncap_fp4.json"))["arms"]
sets = [("FP8 (LM)", f8["A"]["ttft_ms_mean"], f8["B"]["ttft_ms_mean"], f8["C"]["ttft_ms_mean"]),
        ("FP4 / NVFP4 (LM)", f4["A"]["ttft_ms_mean"], f4["B"]["ttft_ms_mean"], f4["C"]["ttft_ms_mean"])]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0), sharey=True)
fig.suptitle("Mixed precision: the language model is the lever; the vision tower is a sliver",
             fontsize=14, fontweight="bold", y=0.98)
labels = ["BF16", "LM low-prec\n(vision BF16)", "LM + vision\nlow-prec"]
cols = ["#9aa3b0", "#4f9fd6", "#10325f"]
ymax = max(s[1] for s in sets) * 1.2
for ax, (name, A, B, C) in zip(axes, sets):
    gap = A - C
    bars = ax.bar(labels, [A, B, C], color=cols, edgecolor="white", width=0.62)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.annotate("", xy=(0.5, B+0.5), xytext=(0.5, A+0.5), arrowprops=dict(arrowstyle="<->", color="#c1121f", lw=1.3))
    ax.text(0.5, (A+B)/2+1.2, f"LM −{A-B:.1f} ms\n({(A-B)/gap*100:.0f}% of gap)", ha="center", fontsize=8.5, color="#c1121f", fontweight="bold")
    ax.annotate("", xy=(1.5, C+0.5), xytext=(1.5, B+0.5), arrowprops=dict(arrowstyle="<->", color="#1f6f3f", lw=1.3))
    ax.text(1.55, (B+C)/2, f"vision −{B-C:.1f} ms\n({(B-C)/gap*100:.0f}%)", ha="left", va="center", fontsize=8.5, color="#1f6f3f", fontweight="bold")
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylim(0, ymax)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("single-stream prefill / TTFT (ms) — lower = faster", fontsize=10)
fig.text(0.5, 0.015,
         "Qwen3-VL-4B · 768px · vLLM · RTX 5090 · synthetic image. FP4 on the LM nearly DOUBLES the win vs FP8 (LM bar 24.8→13.0 ms). "
         "Quantizing the vision tower is a minor lever in both — and SMALLER at FP4 (1% vs 8%). Vision can stay INT8; its share grows with image resolution. "
         "(FP4 arm C = FP4 LM + FP8 vision — pure-FP4 vision needs a custom checkpoint; the vision delta is a conservative lower bound.)",
         ha="center", fontsize=7.4, color="#888", style="italic", wrap=True)
fig.tight_layout(rect=[0,0.06,1,0.95])
out = f"{REPO}/data/output/vision_quant_combined"
fig.savefig(out+".png", dpi=150); fig.savefig(out+".svg")
print("wrote", out+".png")
