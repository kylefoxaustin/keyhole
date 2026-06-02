#!/usr/bin/env python3
"""
plot_precision_migration.py — the precision-migration timeline (the thesis).

Reads data/output/precision_migration.json and renders a workload × horizon grid
where each cell is a format chip colored on a bit-width/format SPECTRUM. Layered on:
  - real measured/roadmap models anchored in each cell,
  - a "FP silicon already shipping (ahead of need)" strip on the time axis,
  - Keyhole NPU-tier asks tagged on the 2028 / 2030 columns,
  - a bit-width descent strip showing the 8-bit INT/FP threshold.

The eye should see vision stay integer (cool) while LLM descends INT8→FP8→FP4
(warm), with FP silicon already sitting left of the 2028 "mandatory" line.

Output: data/output/precision_migration.png + .svg
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, FancyArrowPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "output", "precision_migration.json")
OUT = os.path.join(REPO, "data", "output", "precision_migration")

CLASS_COLOR = {
    "int8": "#2a9d8f", "int4": "#1d6f8b", "mixed": "#8aab6b",
    "mixed_fp8": "#e9c46a", "fp8": "#f4a261", "fp4": "#e76f51",
    "layer_adaptive": "#c1666b",
}
TEXT_DARK = {"int8", "int4", "mixed", "mixed_fp8"}
RED = "#c1121f"

# Map a calendar year to grid-x. Columns: H1 center 0.5, H2 1.5, H3 2.5.
# Treat x=0.5 ≈ 2027, and 1 grid unit ≈ 2.33 yr (the horizon width).
YR_PER_X = (2033 - 2026) / 3.0
def year_to_x(yr):
    return (yr - 2026) / YR_PER_X


def main():
    doc = json.load(open(DATA))
    horizons = doc["__meta__"]["horizons"]
    workloads = doc["workloads"]
    tier_actions = doc["npu_tier_actions"]
    bwd = doc["bit_width_descent"]
    silicon = doc["competitive_silicon"]
    wkeys = ["vision", "llm", "vla"]

    fig = plt.figure(figsize=(15.5, 8.6))
    ax = fig.add_axes([0.15, 0.13, 0.82, 0.85])  # leave bottom 13% for legend/source
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(-1.6, 5.35)
    ax.axis("off")

    fig.suptitle("7-year precision roadmap — INT8 holds the floor, FP stops being optional",
                 fontsize=17, fontweight="bold", y=0.975)
    ax.text(1.5, 4.92,
            "Edge-AI deploy precision by workload, 2026 → 2033.  "
            "INT-capable @ 8-bit ≠ INT-sufficient below 8-bit — descend sub-8-bit and FP becomes mandatory.",
            ha="center", va="center", fontsize=10.5, color="#444")

    # ---- Silicon-shipping strip (tweak 2): FP silicon arriving AHEAD of need ----
    # Most parts shipped 2024–2026 (left of the axis start), so place them spread
    # across the left half — all sit LEFT of the 2028 mandate line (x=1.0).
    strip_y = 4.50
    strip_x = [0.12, 0.40, 0.66, 0.90]
    ax.text(-0.08, strip_y, "FP silicon\nalready shipping →", ha="right", va="center",
            fontsize=8.3, color="#555", fontweight="bold")
    ax.annotate("", xy=(1.0, strip_y), xytext=(0.0, strip_y),
                arrowprops=dict(arrowstyle="-", color="#ccc", lw=1))
    for s, x in zip(silicon, strip_x):
        ax.plot(x, strip_y, "o", ms=7, color="#5a5a5a", zorder=5)
        name = s["part"].replace("Qualcomm ", "").replace("NVIDIA ", "")
        ax.text(x, strip_y + 0.15, name, ha="center", va="bottom", fontsize=7.0, color="#333")
        ax.text(x, strip_y - 0.15, s["shipping"], ha="center", va="top", fontsize=6.4, color="#999")
    ax.text(1.9, strip_y, "← all shipping FP before the 2028 mandate (hardware precedes the workload need)",
            ha="left", va="center", fontsize=7.6, color="#999", style="italic")

    # ---- FP-mandatory marker (top of the crossover line) ----
    ax.text(1.0, 4.06, "FP becomes mandatory  →", ha="center", va="center",
            fontsize=10, color=RED, fontweight="bold")

    # ---- Horizon headers + NPU tier tags (tweak 3) ----
    for j, h in enumerate(horizons):
        ax.text(j + 0.5, 3.70, h["label"], ha="center", va="bottom",
                fontsize=12.5, fontweight="bold")
        ax.text(j + 0.5, 3.57, h["subtitle"], ha="center", va="bottom",
                fontsize=8.5, color="#777", style="italic")
        if h["key"] in tier_actions:
            ax.text(j + 0.5, 3.33, tier_actions[h["key"]], ha="center", va="center",
                    fontsize=8.6, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=RED, edgecolor="none"))

    # ---- Grid rows (vision top -> vla bottom) ----
    for i, wk in enumerate(wkeys):
        row_y = 2 - i
        w = workloads[wk]
        ax.text(-0.08, row_y + 0.62, w["display_name"], ha="right", va="center",
                fontsize=12, fontweight="bold")
        ax.text(-0.08, row_y + 0.30, w["verdict"], ha="right", va="center",
                fontsize=7.6, color="#888", style="italic")

        for j, h in enumerate(horizons):
            cell = w["cells"][h["key"]]
            color = CLASS_COLOR[cell["precision_class"]]
            ax.add_patch(FancyBboxPatch((j + 0.06, row_y + 0.06), 0.88, 0.88,
                         boxstyle="round,pad=0.01,rounding_size=0.04",
                         linewidth=0, facecolor=color))
            tc = "#1a1a1a" if cell["precision_class"] in TEXT_DARK else "white"
            ax.text(j + 0.5, row_y + 0.70, cell["format"], ha="center", va="center",
                    fontsize=12.5, fontweight="bold", color=tc)
            ax.text(j + 0.5, row_y + 0.46, cell["rationale"], ha="center", va="center",
                    fontsize=7.2, color=tc)
            # model anchors (tweak 1)
            tag = ("● " if cell.get("measured") else "") + cell["anchor_models"]
            ax.text(j + 0.5, row_y + 0.20, tag, ha="center", va="center",
                    fontsize=6.9, color=tc, fontweight="bold" if cell.get("measured") else "normal")

    # measured-today anchor under column 1
    ax.text(0.5, -0.18, "● = measured today (precision_composition)", ha="center", va="top",
            fontsize=7.6, color="#2a9d8f", fontweight="bold")

    # Measured-validation badge on the 2030 LLM cell (the FP4 datapoint).
    mv = doc.get("measured_validation_2026")
    if mv:
        ax.text(2.5, 1.5, "✓ measured on RTX 5090\n(Jun 2026)", ha="center", va="center",
                fontsize=7.3, color="white", fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1a1a1a", alpha=0.55, edgecolor="none"))

    # ---- FP-mandatory line at the 2028 crossover (x=1.0) ----
    ax.plot([1.0, 1.0], [0.06, 4.06], color=RED, lw=2.4, ls=(0, (5, 3)), zorder=6)
    ax.text(1.04, 0.10, "INT8-only past here\n= feature gap, not a cost win", ha="left",
            va="bottom", fontsize=8, color=RED, style="italic")

    # ---- Bit-width descent strip (tweak 4) ----
    by = -0.85
    ax.add_patch(FancyArrowPatch((0.0, by), (3.0, by), arrowstyle="-|>",
                 mutation_scale=18, lw=2.2, color="#666", zorder=4))
    ax.text(-0.08, by, "deploy\nbit-width ↓", ha="right", va="center",
            fontsize=8.2, color="#555", fontweight="bold")
    for j, h in enumerate(horizons):
        bits = bwd["per_horizon_bits"][h["key"]]
        below = j >= 1  # H2/H3 are sub-8-bit -> FP territory
        ax.text(j + 0.5, by + 0.20, bits, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=(RED if below else "#2a9d8f"))
        ax.text(j + 0.5, by - 0.20, "FP territory" if below else "INT holds",
                ha="center", va="top", fontsize=7.6,
                color=(RED if below else "#2a9d8f"), style="italic")
    ax.text(1.5, by - 0.62, bwd["note"], ha="center", va="top", fontsize=7.6, color="#777")

    # ---- Legend (format spectrum) ----
    order = [("int8", "INT8 (integer floor)"), ("int4", "INT8+INT4"),
             ("mixed", "mixed INT"), ("mixed_fp8", "mixed INT/FP8"),
             ("fp8", "FP8 (E4M3)"), ("fp4", "FP4 / NVFP4 (E2M1)"),
             ("layer_adaptive", "layer-adaptive")]
    handles = [Patch(facecolor=CLASS_COLOR[k], label=lbl) for k, lbl in order]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               ncol=7, fontsize=8.4, frameon=False, title="deploy precision format",
               title_fontsize=9.2)

    fig.text(0.5, 0.038,
             "Measured (RTX 5090, Jun 2026): NVFP4 runs natively on Blackwell TODAY; the 'FP4' model keeps embed+LM-head in BF16 (FP-residual, confirmed); "
             "decode is BW-bound. Perf win is runtime-dependent — not yet on llama.cpp; NVIDIA vLLM/TensorRT-LLM report ~3×.",
             ha="center", fontsize=7.1, color="#555", style="italic")
    fig.text(0.5, 0.014, "Source: precision-roadmap-combined.pptx (Executive Brief, May 2026) · "
             "measured datapoint in precision_5090_fp4_vs_int.json · citations in precision_migration.json",
             ha="center", fontsize=7, color="#999")

    fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT + ".svg", bbox_inches="tight")
    print("wrote", OUT + ".png", "and", OUT + ".svg")


if __name__ == "__main__":
    main()
