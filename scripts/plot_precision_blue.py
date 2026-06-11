#!/usr/bin/env python3
"""
plot_precision_blue.py — the two management visuals from precision_blue.json.

  precision_blue_overtime.png
      Per workload (Vision / LLM / VLA), a 100%-stacked horizontal bar PER horizon
      (2026→2033), split INT (grey) | optional-FP (light blue) | mandatory-FP (dark
      blue). The eye sees the BLUE half grow over time — dramatically for LLM, not
      at all for vision. This is the "see these blue boxes" slide.

  precision_blue_weights.png
      The weights-only question. Left: weight-format trajectory INT4 → FP8 → FP4.
      Right: the decisive MEASURED comparison — INT4 vs FP4 speed-up vs BF16 across
      decode / prefill / training. They TIE on decode (both 4-bit memory) and diverge
      hard on compute (INT4 stuck on the BF16 floor; FP4 climbs) — which is exactly
      why the weight format is moving to FP4.

Run:  python3 scripts/plot_precision_blue.py
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "output", "precision_blue.json")

INT_C = "#9aa3b0"
OPT_C = "#4f9fd6"
MAND_C = "#10325f"
RED = "#c1121f"
GREEN = "#2a9d8f"


def text_color(seg, frac):
    """White on the dark-blue / int segments, dark on the light-blue."""
    if seg == "fp_optional":
        return "#0d2235"
    return "white"


def plot_overtime(doc, out):
    ot = doc["over_time"]
    horizons = doc["__meta__"]["horizons"]
    wkeys = ["vision", "llm", "vla"]

    fig, axes = plt.subplots(3, 1, figsize=(12.6, 8.6), sharex=True)
    fig.suptitle("How each workload splits INT vs FP — and the FP (blue) half grows over time",
                 fontsize=16.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.937,
             "Deploy-precision mix, 2026 → 2033.  Grey = stays INT.  "
             "Light blue = runs FP but COULD be INT (you'd lose compute speed / accuracy).  "
             "Dark blue = CANNOT be INT.",
             ha="center", fontsize=10.3, color="#444")

    for ax, wk in zip(axes, wkeys):
        w = ot[wk]
        rows = list(reversed(horizons))     # 2026 at top
        y = range(len(rows))
        for i, h in enumerate(rows):
            intp, optp, mandp = w["cells"][h]
            segs = [("int", intp, INT_C), ("fp_optional", optp, OPT_C),
                    ("fp_mandatory", mandp, MAND_C)]
            left = 0
            for seg, val, color in segs:
                if val <= 0:
                    continue
                ax.barh(i, val, left=left, color=color, edgecolor="white", linewidth=1.2)
                if val >= 6:
                    ax.text(left + val / 2, i, f"{val:.0f}", va="center", ha="center",
                            fontsize=9, fontweight="bold", color=text_color(seg, val))
                left += val
            ax.text(-1.5, i, h, va="center", ha="right", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        # workload label + verdict, left of the bars
        ax.set_title(f"{w['display_name']}   —   {w['verdict']}",
                     fontsize=12, fontweight="bold", loc="left", color="#222", pad=6)
        # blue-growth callout on the LLM panel (the headline): point at the 2030
        # crossover (INT shrinks to 18% → FP is the majority), in a white pill so
        # the red text reads on any segment behind it.
        if wk == "llm":
            ax.annotate("FP overtakes INT by 2030", xy=(18, 1), xytext=(30, 1.6),
                        ha="left", va="center", fontsize=9, color=RED, fontweight="bold",
                        zorder=10,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))

    axes[-1].set_xlabel("share of deployed model (%)", fontsize=10)

    legend = [
        Patch(facecolor=INT_C, label="INT — quantized, stays integer"),
        Patch(facecolor=OPT_C, label="FP, optional — could be INT, but you lose compute speed / accuracy"),
        Patch(facecolor=MAND_C, label="FP, mandatory — cannot be INT (outliers, softmax/norm, action heads)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=1, fontsize=9.6, frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.10,
             "2026 anchored to measured INT/FP composition + 5090 bake-offs; 2028–2033 are a directional forecast "
             "(adoption precedent + shipping FP silicon).",
             ha="center", fontsize=8, color="#888", style="italic")
    fig.tight_layout(rect=[0.02, 0.15, 1, 0.925])
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print("wrote", out + ".png")


def plot_weights(doc, out):
    w = doc["weights"]
    mm = w["measured_multipliers_vs_bf16"]

    fig = plt.figure(figsize=(13.4, 8.2))
    fig.suptitle("The weights question:  do they stay INT4/INT8, or does FP4 take over?",
                 fontsize=16.5, fontweight="bold", y=0.972)
    fig.text(0.5, 0.928, w["headline"], ha="center", fontsize=11, color=RED, fontweight="bold")

    # ---- LEFT: weight-format trajectory over time -------------------------
    axL = fig.add_axes([0.045, 0.40, 0.40, 0.47])
    axL.axis("off")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    axL.text(0.5, 1.02, "LLM / VLA weight format over time", ha="center",
             fontsize=12, fontweight="bold")
    traj = w["trajectory"]
    chip_colors = [INT_C, "#7a8aa0", OPT_C, MAND_C]   # grey → blue as it migrates
    n = len(traj)
    for i, (t, c) in enumerate(zip(traj, chip_colors)):
        y = 0.86 - i * 0.225
        axL.add_patch(plt.Rectangle((0.08, y - 0.075), 0.84, 0.15, facecolor=c,
                                    edgecolor="none", transform=axL.transData))
        tc = "white" if c != OPT_C else "#0d2235"
        axL.text(0.13, y + 0.018, f"{t['horizon']}   {t['dominant']}", va="center",
                 ha="left", fontsize=11, fontweight="bold", color=tc)
        axL.text(0.13, y - 0.040, t["detail"], va="center", ha="left",
                 fontsize=8.2, color=tc)
        if i < n - 1:
            axL.annotate("", xy=(0.5, y - 0.088), xytext=(0.5, y - 0.137),
                         arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.6))
    # vision caveat sits in the gap below the trajectory panel (fig coords)
    fig.text(0.245, 0.355, w["vision_caveat"], ha="center", va="top",
             fontsize=8.2, color=GREEN, style="italic", wrap=True)

    # ---- RIGHT: the measured WHY — INT4 vs FP4 across the three regimes ----
    axR = fig.add_axes([0.55, 0.46, 0.42, 0.40])
    axes_lbl = mm["axes"]
    int4 = mm["int4"]; fp8 = mm["fp8"]; fp4 = mm["fp4"]
    x = range(len(axes_lbl))
    bw = 0.26
    FP8_C = "#3d7cc0"
    b1 = axR.bar([i - bw for i in x], int4, bw, label="INT4 (memory-only)",
                 color=INT_C, edgecolor="white")
    b3 = axR.bar([i for i in x], fp8, bw, label="FP8 / E4M3 (mid anchor)",
                 color=FP8_C, edgecolor="white")
    b2 = axR.bar([i + bw for i in x], fp4, bw, label="FP4 / NVFP4 (memory + compute)",
                 color=MAND_C, edgecolor="white")
    axR.axhline(1.0, color="#aaa", lw=1, ls="--")
    axR.text(len(axes_lbl) - 0.5, 1.05, "BF16 = 1×", fontsize=8, color="#888", ha="right")
    for bars in (b1, b3, b2):
        for b in bars:
            h = b.get_height()
            axR.text(b.get_x() + b.get_width() / 2, h + 0.12, f"{h:.2f}×",
                     ha="center", va="bottom", fontsize=8, fontweight="bold",
                     color="#222")
    axR.set_xticks(list(x))
    axR.set_xticklabels(axes_lbl, fontsize=9.5)
    axR.set_ylabel("speed-up vs BF16  (RTX 5090)", fontsize=9.5)
    axR.set_ylim(0, 6.4)
    axR.set_title("Why FP4 wins the weight format — measured, same weights",
                  fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    axR.legend(fontsize=8.6, frameon=False, loc="upper left")
    # the crux annotation: tie on decode, split on compute
    axR.annotate("TIE\n(both 4-bit memory)", xy=(0, 2.3), xytext=(0, 3.5),
                 ha="center", fontsize=8, color="#555",
                 arrowprops=dict(arrowstyle="-", color="#bbb", lw=1))
    axR.annotate("INT4 stuck on\nthe BF16 floor", xy=(1 - bw, 1.04), xytext=(0.55, 4.6),
                 ha="center", fontsize=8, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))

    # ---- bottom: the "why" — concise full-width one-liners (own clean band) ----
    why = [
        "Decode ties because INT4 and FP4 stream the same 4 bits (bandwidth-bound); prefill & training "
        "split because only FP4 has a 4-bit compute path — INT4 dequantizes to BF16 and inherits its floor.",
        "Below 8 bits, FP's exponent (E4M3 / E2M1) preserves the heavy-tail activation outliers that INT "
        "clipping loses — DeepSeek-R1: <1% loss FP8→FP4 PTQ.",
        "Silicon already bet on FP4: Blackwell (FP4/FP6/FP8), Qualcomm Hexagon (FP8), TI ADAS — the roadmap "
        "is FP, not INT-mixed.",
    ]
    fig.text(0.06, 0.235, "Why FP4 takes the weights:", fontsize=10, fontweight="bold", color="#222")
    by = 0.195
    for b in why:
        fig.text(0.06, by, "•  " + b, fontsize=8.4, color="#444", ha="left", va="top")
        by -= 0.052
    fig.text(0.5, 0.018, "Source: " + mm["source"], ha="center", fontsize=7.2,
             color="#999", style="italic")

    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print("wrote", out + ".png")


def main():
    doc = json.load(open(DATA))
    plot_overtime(doc, os.path.join(REPO, "data", "output", "precision_blue_overtime"))
    plot_weights(doc, os.path.join(REPO, "data", "output", "precision_blue_weights"))


if __name__ == "__main__":
    main()
