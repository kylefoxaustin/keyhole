#!/usr/bin/env python3
"""plot_accuracy_study.py — answer accuracy (DocVQA ANLS): precision x pipeline. Public data."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(f"{REPO}/data/output/accuracy_study.json"))
order = ["bf16", "fp8", "nvfp4"]; disp = {"bf16": "BF16", "fp8": "FP8", "nvfp4": "NVFP4"}
img = {c["label"]: c["anls"] for c in d["configs"] if c["input"] == "image"}
txt = {c["label"]: c["anls"] for c in d["configs"] if c["input"] == "text"}

fig, ax = plt.subplots(figsize=(10.2, 6.0))
x = range(len(order)); bw = 0.38
b1 = ax.bar([i - bw/2 for i in x], [img[k] for k in order], bw, label="IMAGE  (visual-RAG / PixelRAG)", color="#10325f", edgecolor="white")
b2 = ax.bar([i + bw/2 for i in x], [txt[k] for k in order], bw, label="TEXT  (OCR → text-RAG)", color="#9aa3b0", edgecolor="white")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels([disp[k] for k in order], fontsize=12, fontweight="bold")
ax.set_ylabel("answer accuracy — ANLS (higher = better)", fontsize=10.5)
ax.set_ylim(0, 1.05); ax.set_xlabel("reader numeric precision", fontsize=10.5)
ax.set_title("Answer accuracy on DocVQA (ANLS, n=200): visual-RAG beats text-RAG, and FP4 barely costs anything",
             fontsize=12, fontweight="bold")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=10, frameon=False, loc="lower center")
gap = img["bf16"] - txt["bf16"]
ax.annotate(f"+{gap*100:.0f} ANLS pts\n(image vs text)", xy=(0, (img["bf16"]+txt["bf16"])/2), xytext=(0.0, 0.50),
            ha="center", fontsize=9, color="#1f6f3f", fontweight="bold",
            arrowprops=dict(arrowstyle="-[, widthB=2.2", color="#1f6f3f", lw=1.3))
ax.text(2, img["nvfp4"]+0.06, f"FP4 costs only\n−{(img['bf16']-img['nvfp4'])*100:.1f} pts vs BF16", ha="center", fontsize=8.5, color="#c1121f", fontweight="bold")
fig.text(0.5, 0.015, "lmms-lab/DocVQA validation · Qwen3-VL-4B reader · gold page given (retrieval held correct) · text path = rapidocr OCR · "
         "image input sidesteps OCR errors; quantization (FP8/FP4) is near-free for answer correctness.",
         ha="center", fontsize=7.6, color="#888", style="italic", wrap=True)
fig.tight_layout(rect=[0, 0.05, 1, 1])
out = f"{REPO}/data/output/accuracy_study"
fig.savefig(out+".png", dpi=150); fig.savefig(out+".svg")
print("wrote", out+".png")
