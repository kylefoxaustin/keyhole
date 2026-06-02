#!/usr/bin/env python3
"""
plot_precision_composition.py — faceted INT-vs-FP composition chart.

Reads data/output/precision_composition.json and renders three side-by-side
100%-stacked horizontal-bar panels (by parameters / by FLOPs / by stored bytes).
Models share a y-axis ordering (vision, then LLM, then VLA) so you can read a
single model straight across the three facets.

  teal  = INT-capable (quantizable: matmul/conv/embedding weights, GEMM FLOPs,
          low-precision weight body)
  coral = FP-required / FP-tail / high-precision residual
  grey  = not measured for that axis (honest coverage gap)

Output: data/output/precision_composition.png + .svg
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "output", "precision_composition.json")
OUT = os.path.join(REPO, "data", "output", "precision_composition")

INT_C = "#2a9d8f"
FP_C = "#e76f51"
NA_C = "#dcdcdc"

# Display order (top -> bottom on the chart we reverse), grouped by family.
ORDER = [
    ("vision", "sam3_full", "SAM 3 (full)"),
    ("vision", "yolov8n_seg_clip_hybrid", "YOLOv8n-seg + CLIP"),
    ("llm", "llama3.1-8b-dense", "Llama-3.1-8B"),
    ("llm", "mistral-7b-v0.3-dense", "Mistral-7B-v0.3"),
    ("llm", "qwen2.5-7b-dense", "Qwen2.5-7B"),
    ("llm", "qwen2.5-32b-dense", "Qwen2.5-32B"),
    ("vla", "nora_3b", "NORA-3B (single-loop)"),
    ("vla", "openvla_7b_single", "OpenVLA-7B (single-loop)"),
    ("vla", "bitvla", "BitVLA (ternary)"),
    ("vla", "pi_0p5", "π0.5 (dual-loop)"),
    ("vla", "nora_1p5", "NORA-1.5 (dual-loop)"),
]


def params_split(m):
    bp = m.get("by_params", {})
    if "int_capable_pct" in bp:
        return bp["int_capable_pct"], bp["fp_required_pct"]
    if "linear_layers_quantized_pct" in bp:           # YOLO+CLIP recipe
        return bp["linear_layers_quantized_pct"], bp["linear_layers_fp_kept_pct"]
    return None


def flops_split(m):
    bf = m.get("by_flops", {})
    if bf.get("provenance") == "not_computed" or "int_capable_pct" not in bf:
        return None
    return bf["int_capable_pct"], bf["fp_tail_pct"]


def bytes_split(m):
    """INT/FP byte split. Only LLM GGUF quants decompose into low-precision body
    vs high-precision residual; pick the lowest-bit quant available."""
    bb = m.get("by_bytes", {})
    quants = bb.get("quants")
    if not quants:
        return None
    for q in ("Q4_K_M", "Q5_K_M", "Q8_0"):
        if q in quants and quants[q].get("high_precision_residual_pct") is not None:
            resid = quants[q]["high_precision_residual_pct"]
            return (100 - resid, resid, q)
    return None


def main():
    models = json.load(open(DATA))["models"]
    rows = list(reversed(ORDER))  # so SAM3 ends up at the top
    labels = [r[2] for r in rows]
    y = range(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), sharey=True)
    fig.suptitle("TODAY (2026): measured INT-vs-FP composition — the roadmap's anchor, not its conclusion",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.945,
             "INT-capable @ 8-bit ≠ INT-sufficient below 8-bit.  As models descend to FP8 (~2028) / FP4 (~2030), "
             "this green migrates to FP — see precision_migration.",
             ha="center", fontsize=9.5, color="#c1121f")

    panels = [
        ("By parameters\n(quantizable weights vs FP-only)", params_split, "params"),
        ("By FLOPs\n(GEMM compute vs FP-residual tail)", flops_split, "flops"),
        ("By stored bytes\n(low-precision body vs hi-precision residual)", bytes_split, "bytes"),
    ]

    for ax, (title, fn, kind) in zip(axes, panels):
        for i, (_, key, _) in enumerate(rows):
            res = fn(models[key])
            if res is None:
                ax.barh(i, 100, color=NA_C, hatch="///", edgecolor="white")
                ax.text(50, i, "not measured", va="center", ha="center",
                        fontsize=7.5, color="#888", style="italic")
                continue
            intp, fpp = res[0], res[1]
            ax.barh(i, intp, color=INT_C, edgecolor="white")
            ax.barh(i, fpp, left=intp, color=FP_C, edgecolor="white")
            # INT label at left
            ax.text(2, i, f"{intp:.0f}", va="center", ha="left",
                    fontsize=8, color="white", fontweight="bold")
            # FP label only when the segment is visible
            if fpp >= 1.5:
                ax.text(min(intp + fpp - 1, 99), i, f"{fpp:.0f}", va="center",
                        ha="right", fontsize=8, color="white", fontweight="bold")
            if kind == "bytes":  # annotate which quant
                ax.text(101, i, res[2], va="center", ha="left", fontsize=7, color="#555")

        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("% of model")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # faint family separators (rows are reversed: vla block at bottom)
        ax.axhline(2.5, color="#eee", lw=1)  # vla | llm
        ax.axhline(6.5, color="#eee", lw=1)  # llm | vision

    legend = [
        Patch(facecolor=INT_C, label="INT8-sufficient TODAY (at 8-bit)"),
        Patch(facecolor=FP_C, label="FP-required / FP tail / hi-precision residual"),
        Patch(facecolor=NA_C, hatch="///", label="not measured for this axis"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.05,
             "Green = INT8 works for this model TODAY — it does NOT mean FP is unneeded. The FP tail is <0.1% by FLOPs "
             "yet dominates latency (memory-/launch-bound);",
             ha="center", fontsize=8, color="#666", style="italic")
    fig.text(0.5, 0.028,
             "flow-matching heads (π0.5, NORA-1.5) already require FP; and below 8-bit (FP8 '28 / FP4 '30) the green itself migrates to FP.",
             ha="center", fontsize=8, color="#666", style="italic")

    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT + ".svg", bbox_inches="tight")
    print("wrote", OUT + ".png", "and", OUT + ".svg")


if __name__ == "__main__":
    main()
