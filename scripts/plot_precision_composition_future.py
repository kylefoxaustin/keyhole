#!/usr/bin/env python3
"""
plot_precision_composition_future.py — the 2030 & 2033 companions to slide 4.

Same 3-axis (params / FLOPs / bytes) per-model layout as plot_precision_composition,
but for the NEXT GENERATION of models at each horizon's deploy format. This is the
per-model breakout that the over-time summary (precision_blue_overtime) summarizes.

Honesty: today's BF16-trained models do NOT "become" FP4 — so these rows are
next-gen *model classes* (forecast), NOT today's models relabelled. The transformer
STRUCTURE (matmul engine vs FP-mandatory tail) is architecture-stable, so each
class borrows the measured structural fractions of an analogous 2026 model
(`proxy`); only the DEPLOY-FORMAT colouring and the model generation change.

  grey       = INT (deployed integer — vision throughout; edge LLMs until they move)
  light blue = FP-optional (FP4-native weights: could be INT, but you lose the
               compute-format speed-up + accuracy)
  dark blue  = FP-mandatory (softmax/norm/outlier tail; flow-matching action heads;
               and — at sub-4-bit, 2033 — the outlier handling that INT can't do)

Consistent with precision_blue_overtime: LLM dark-blue (mandatory) grows 2030→2033
as deploy bit-width descends. Output: precision_composition_2030/2033.png + .svg
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = os.path.join(REPO, "data", "output", "precision_composition.json")

INT_C = "#9aa3b0"      # grey  — INT
OPT_C = "#4f9fd6"      # light blue — FP-optional (FP4-native)
MAND_C = "#10325f"     # dark blue — FP-mandatory

# Next-gen classes. proxy = 2026 model whose structural fractions we borrow.
# fmt[h] = "INT" (engine stays integer) or "FP4" (engine is FP4-native).
CATALOG = [
    ("vision", "Vision ViT / CNN", "sam3_full",
     {"2030": "INT", "2033": "INT"}),
    ("llm", "Edge LLM  ~1–3B", "qwen2.5-7b-dense",
     {"2030": "INT", "2033": "FP4"}),          # edge holds INT longer, moves by '33
    ("llm", "Mid LLM  ~8B", "llama3.1-8b-dense",
     {"2030": "FP4", "2033": "FP4"}),
    ("llm", "Flagship dense  ~32B", "qwen2.5-32b-dense",
     {"2030": "FP4", "2033": "FP4"}),
    ("llm", "Flagship MoE", "qwen2.5-32b-dense",
     {"2030": "FP4", "2033": "FP4"}),
    ("vla", "Single-loop VLA", "nora_3b",
     {"2030": "FP4", "2033": "FP4"}),
    ("vla", "Dual-loop VLA (flow-matching)", "pi_0p5",
     {"2030": "FP4", "2033": "FP4"}),
]

# Sub-4-bit outlier handling turns part of the FP4 engine FP-MANDATORY by 2033
# (mirrors precision_blue_overtime: LLM mandatory 15→22%). Points of the bar moved
# from optional→mandatory for FP4 classes in 2033.
EXTRA_MAND_2033 = {"llm": 10.0, "vla": 6.0, "vision": 0.0}


def axis_split(m, kind):
    """(int_capable_pct, fp_pct) for the given axis of a 2026 proxy model."""
    if kind == "params":
        bp = m["by_params"]
        if "int_capable_pct" in bp:
            return bp["int_capable_pct"], bp["fp_required_pct"]
        return bp["linear_layers_quantized_pct"], bp["linear_layers_fp_kept_pct"]
    if kind == "flops":
        bf = m["by_flops"]
        return bf["int_capable_pct"], bf.get("fp_tail_pct", 100 - bf["int_capable_pct"])
    bb = m["by_bytes"]                       # bytes
    quants = bb.get("quants")
    if quants:
        for q in ("Q4_K_M", "Q5_K_M", "Q8_0"):
            if q in quants and quants[q].get("high_precision_residual_pct") is not None:
                r = quants[q]["high_precision_residual_pct"]
                return 100 - r, r
    return bb["int_capable_pct"], bb.get("fp_required_pct", 100 - bb["int_capable_pct"])


def three_way(ic, fp, fmt, family, horizon):
    """Map (int_capable, fp_mandatory) → (int_grey, fp_optional, fp_mandatory)."""
    mand = fp
    if fmt == "INT":
        return ic, 0.0, mand
    # FP4 engine: int_capable becomes FP-optional, minus an outlier-mandatory slice in 2033
    opt = ic
    if horizon == "2033":
        extra = min(EXTRA_MAND_2033.get(family, 0.0), opt)
        opt -= extra
        mand += extra
    return 0.0, opt, mand


def plot_horizon(models, horizon, out):
    rows = list(reversed(CATALOG))           # first entry ends up on top
    labels = [r[1] for r in rows]
    y = range(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0), sharey=True)
    fig.suptitle(f"{horizon} — projected per-model INT vs FP composition (next-gen, FP4-native)",
                 fontsize=13.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.945,
             "Same breakout as the 2026 slide, for the next model generation. "
             "Grey = INT · light blue = FP4 weights (FP-optional) · dark blue = FP-mandatory. "
             "Forecast — structure borrowed from measured analogues.",
             ha="center", fontsize=9.2, color="#10325f")

    panels = [
        ("By parameters\n(weights)", "params"),
        ("By FLOPs\n(GEMM vs FP tail)", "flops"),
        ("By stored bytes\n(weight footprint)", "bytes"),
    ]
    for ax, (title, kind) in zip(axes, panels):
        for i, (family, _, proxy, fmt_map) in enumerate(rows):
            ic, fp = axis_split(models[proxy], kind)
            g, opt, mand = three_way(ic, fp, fmt_map[horizon], family, horizon)
            left = 0
            for val, color in ((g, INT_C), (opt, OPT_C), (mand, MAND_C)):
                if val <= 0:
                    continue
                ax.barh(i, val, left=left, color=color, edgecolor="white")
                if val >= 7:
                    tc = "#0d2235" if color == OPT_C else "white"
                    ax.text(left + val / 2, i, f"{val:.0f}", va="center", ha="center",
                            fontsize=7.5, fontweight="bold", color=tc)
                left += val
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("% of model")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axhline(1.5, color="#eee", lw=1)   # vla | llm
        ax.axhline(5.5, color="#eee", lw=1)   # llm | vision

    legend = [
        Patch(facecolor=INT_C, label="INT — deployed integer"),
        Patch(facecolor=OPT_C, label="FP-optional — FP4-native weights (could be INT, lose compute/accuracy)"),
        Patch(facecolor=MAND_C, label="FP-mandatory — cannot be INT (FP tail / flow-matching head / sub-4-bit outliers)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=1, fontsize=8.8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, 0.04,
             "Vision stays all-INT. Edge LLMs hold INT longest; flagships lead to FP4. "
             "By 2033 sub-4-bit outlier handling turns part of even the FP4 engine FP-mandatory (dark grows).",
             ha="center", fontsize=8, color="#666", style="italic")
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print("wrote", out + ".png")


def main():
    models = json.load(open(COMP))["models"]
    for h in ("2030", "2033"):
        plot_horizon(models, h, os.path.join(REPO, "data", "output", f"precision_composition_{h}"))


if __name__ == "__main__":
    main()
