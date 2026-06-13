#!/usr/bin/env python3
"""
plot_skippy_knee.py — the resolution-knee chart for the Skippy PixelRAG bake-off.

One dual-axis figure: as tile resolution rises, pixel retrieval quality climbs to a
knee (~768px) then plateaus, while reader image-tokens keep climbing. The text-RAG
baseline is a flat reference on both axes. The story: at the knee, visual-RAG beats
native-text RAG on retrieval AND costs fewer reader tokens — a double win you only get
by tuning resolution (at full res, pixels are a token loser).

Reads data/output/skippy_pixelrag_sweep.json (aggregate metrics only — no personal
content). Output: data/output/skippy_pixelrag_knee.png (+ svg).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(REPO, "data", "output", "skippy_pixelrag_sweep.json")))

PIX = "#10325f"; PIX_L = "#4f9fd6"; TXT = "#9aa3b0"; TOK = "#c1666b"; GREEN = "#2a9d8f"

res = [int(r) for r in D["pixel_by_resolution"]]
ndcg = [D["pixel_by_resolution"][str(r)]["all"]["ndcg@5"] for r in res]
r5 = [D["pixel_by_resolution"][str(r)]["all"]["recall@5"] for r in res]
ptok = [D["reader"][str(r)]["input_tokens_mean"] for r in res]
t_ndcg = D["text"]["all"]["ndcg@5"]; t_r5 = D["text"]["all"]["recall@5"]
t_tok = D["reader"]["text"]["input_tokens_mean"]

fig, axL = plt.subplots(figsize=(11.5, 6.6))
axR = axL.twinx()

# retrieval quality (left axis)
axL.plot(res, r5, "-o", color=PIX, lw=2.4, ms=8, label="pixel  recall@5", zorder=5)
axL.plot(res, ndcg, "--s", color=PIX_L, lw=2.0, ms=7, label="pixel  nDCG@5", zorder=5)
axL.axhline(t_r5, color=TXT, lw=2.0, ls="-", label=f"text  recall@5 ({t_r5:.2f})")
axL.axhline(t_ndcg, color=TXT, lw=1.4, ls="--", alpha=0.8, label=f"text  nDCG@5 ({t_ndcg:.2f})")
axL.set_ylabel("retrieval quality  (higher = better)", fontsize=11, color=PIX)
axL.set_ylim(0.3, 0.78)

# reader tokens (right axis)
axR.plot(res, ptok, "-^", color=TOK, lw=2.4, ms=8, label="pixel  reader tokens", zorder=4)
axR.axhline(t_tok, color=TOK, lw=1.6, ls=":", alpha=0.85, label=f"text  reader tokens ({t_tok:.0f})")
axR.set_ylabel("reader input tokens / page  (lower = cheaper)", fontsize=11, color=TOK)
axR.set_ylim(0, 1800)

# knee marker + double-win shading (where pixel acc>text AND pixel tokens<text)
knee = 768
axL.axvline(knee, color=GREEN, lw=1.6, ls=(0, (4, 3)), zorder=2)
axL.axvspan(640, 900, color=GREEN, alpha=0.07, zorder=0)
axL.text(knee, 0.755, "knee ≈ 768px", color=GREEN, fontweight="bold", ha="center", fontsize=10.5)
axL.annotate("double win:\npixel beats text on retrieval\nAND costs fewer reader tokens",
             xy=(knee, 0.704), xytext=(880, 0.40), fontsize=9.2, color=GREEN, fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4))

axL.set_xlabel("page-image resolution (long side, px)", fontsize=11)
axL.set_xticks(res); axL.set_xticklabels([str(r) for r in res])
axL.set_title("Tune the resolution: on Skippy's own corpus, visual-RAG beats native-text RAG on retrieval —\n"
              "and at the knee it ALSO costs fewer reader tokens than text (measured, RTX 5090)",
              fontsize=12.5, fontweight="bold")
l1, lab1 = axL.get_legend_handles_labels()
l2, lab2 = axR.get_legend_handles_labels()
axL.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8.8, frameon=False, ncol=1)
for s in ("top",):
    axL.spines[s].set_visible(False); axR.spines[s].set_visible(False)
fig.text(0.5, 0.012,
         f"Skippy corpus: {D['__meta__']['corpus_pages']} pages (datasheets/documents/writing), "
         f"{D['__meta__']['n_queries']} local-generated queries · Qwen3-VL-Embedding-2B + Qwen3-VL-4B · "
         "aggregate metrics only (no personal content).",
         ha="center", fontsize=8, color="#888", style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 1])
out = os.path.join(REPO, "data", "output", "skippy_pixelrag_knee")
fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".svg")
print("wrote", out + ".png")
