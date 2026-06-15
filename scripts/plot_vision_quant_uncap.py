#!/usr/bin/env python3
"""plot_vision_quant_uncap.py — where prefill savings come from under mixed precision.
Reads data/output/vision_quant_uncap.json (synthetic-image experiment, no personal data)."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(REPO, "data", "output", "vision_quant_uncap.json")))
A = d["arms"]["A"]["ttft_ms_mean"]; B = d["arms"]["B"]["ttft_ms_mean"]; C = d["arms"]["C"]["ttft_ms_mean"]
gap = A - C

fig, ax = plt.subplots(figsize=(9.6, 5.8))
labels = ["BF16\n(nothing quantized)", "FP8 LM only\n(vision stays BF16)", "FP8 LM + vision\n(both quantized)"]
vals = [A, B, C]; cols = ["#9aa3b0", "#4f9fd6", "#10325f"]
bars = ax.bar(labels, vals, color=cols, edgecolor="white", width=0.6)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{b.get_height():.1f} ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
# delta brackets
ax.annotate("", xy=(0.5, B+0.6), xytext=(0.5, A+0.6), arrowprops=dict(arrowstyle="<->", color="#c1121f", lw=1.4))
ax.text(0.5, (A+B)/2+1.4, f"quantize the LM\n−{A-B:.1f} ms  ({(A-B)/gap*100:.0f}% of the gap)", ha="center", fontsize=9, color="#c1121f", fontweight="bold")
ax.annotate("", xy=(1.5, C+0.6), xytext=(1.5, B+0.6), arrowprops=dict(arrowstyle="<->", color="#2a9d8f", lw=1.4))
ax.text(1.55, (B+C)/2, f"+ quantize the\nvision tower\n−{B-C:.1f} ms  ({(B-C)/gap*100:.0f}%)", ha="left", va="center", fontsize=9, color="#1f6f3f", fontweight="bold")
ax.set_ylabel("single-stream prefill / TTFT (ms)  — lower = faster", fontsize=10)
ax.set_ylim(0, A*1.18)
ax.set_title("Mixed precision: where the prefill savings come from\nQwen3-VL-4B · 768px · vLLM FP8 · RTX 5090 (synthetic image)", fontsize=12.5, fontweight="bold")
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.text(0.5, 0.01, "The language model is ~2/3 of the quantizable prefill; the vision tower is only ~1/3 (~8%, ~2 ms) at this resolution — "
         "its share grows with image resolution. Vision can stay INT8 (bounded, near-lossless) for little latency cost.",
         ha="center", fontsize=7.8, color="#888", style="italic", wrap=True)
fig.tight_layout(rect=[0,0.04,1,1])
out = os.path.join(REPO, "data", "output", "vision_quant_uncap")
fig.savefig(out+".png", dpi=150); fig.savefig(out+".svg")
print("wrote", out+".png")
