#!/usr/bin/env python3
"""
plot_fp4_convergence.py — slide 9: FP4 training CONVERGES to BF16 quality.

The GEMM bench (slide 8) showed FP4 training is FAST. This is the other half of
"native FP4 training can be optimal" (Quartet, arXiv 2505.14669): is it ACCURATE?
We trained a 30M-param Llama on WikiText-103 twice in the Quartet harness on the RTX 5090,
identical config + init, single GPU:
  - BF16 baseline (NoQuantizer)
  - FP4: the FULL Quartet recipe — 4-bit (E2M1) weights + activations (QuestMXFP4) AND 4-bit
    gradients (AlbertTseng, stochastic) with the Q(E)Q(Wt)t_Q(Et)Q(Xt)t backward scheme
and compare validation-loss convergence. (Pseudo-quant = emulated FP4 numerics for an
accuracy/convergence measurement; the SPEED is the separate kernel result on slide 8.
Quartet's custom MXFP4 triton kernel needed a 1-line fix for triton 3.6 — a None seed passed
to a non-constexpr `int` arg; set to 0 since it's only read under stochastic_round=True.)

Parses ~/quartet_runs/{bf16.log,mxfp4_full.log} for the '>Eval ... val_loss=' curve,
writes data/output/fp4_training_convergence_5090.json, renders
data/output/precision_5090_fp4_convergence.png.
"""
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTJSON = os.path.join(REPO, "data/output/fp4_training_convergence_5090.json")
OUTPNG = os.path.join(REPO, "data/output/precision_5090_fp4_convergence.png")
RUNS = os.path.expanduser("~/quartet_runs")

EVAL_RE = re.compile(r">Eval: Iter=(\d+).*val_loss=([\d.]+).*val_pp=([\d.]+)")


def parse(logname):
    iters, loss, pp = [], [], []
    path = os.path.join(RUNS, logname)
    with open(path) as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                iters.append(int(m.group(1))); loss.append(float(m.group(2))); pp.append(float(m.group(3)))
    return {"iters": iters, "val_loss": loss, "val_pp": pp}


def main():
    bf16 = parse("bf16.log")
    fp4 = parse("mxfp4_full.log")  # full Quartet recipe: 4-bit W+A+gradients
    final_gap = round(fp4["val_loss"][-1] - bf16["val_loss"][-1], 3)
    pp_ratio = round(fp4["val_pp"][-1] / bf16["val_pp"][-1], 3)
    doc = {
        "__meta__": {
            "description": "FP4 vs BF16 training convergence on RTX 5090 (Quartet harness). 30M Llama, "
                           "WikiText-103, identical config/init, single GPU. FP4 = full Quartet recipe: "
                           "QuestMXFP4 W+A (E2M1) + AlbertTseng 4-bit gradients + QEQWtt backward (pseudo-quant).",
            "claim": "native FP4 training converges to ~BF16 quality (Quartet arXiv 2505.14669)",
            "final_val_loss_bf16": bf16["val_loss"][-1],
            "final_val_loss_fp4": fp4["val_loss"][-1],
            "final_loss_gap": final_gap,
            "final_pp_bf16": bf16["val_pp"][-1],
            "final_pp_fp4": fp4["val_pp"][-1],
            "fp4_pp_vs_bf16": pp_ratio,
        },
        "bf16": bf16, "fp4": fp4,
    }
    os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
    json.dump(doc, open(OUTJSON, "w"), indent=2)
    print("wrote", OUTJSON)

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.plot(bf16["iters"], bf16["val_loss"], "-o", color="#718096", lw=2.2, ms=7,
            label="BF16 baseline", zorder=3)
    ax.plot(fp4["iters"], fp4["val_loss"], "-s", color="#2f855a", lw=2.2, ms=7,
            label="FP4 (4-bit W+A+gradients)", zorder=3)
    # annotate the final gap
    xf = bf16["iters"][-1]
    ax.annotate(f"final: BF16 {bf16['val_loss'][-1]:.3f}  ·  FP4 {fp4['val_loss'][-1]:.3f}\n"
                f"gap {final_gap:+.3f} nats  ·  FP4 perplexity {pp_ratio:.2f}× BF16",
                (xf, fp4["val_loss"][-1]), textcoords="offset points", xytext=(-12, 28),
                ha="right", fontsize=9, color="#2f855a", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f0fff4", ec="#2f855a", lw=0.8))
    ax.set_xlabel("training iteration (WikiText-103, 30M Llama, batch 32 × seq 512)")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("FP4 training converges to BF16 quality\n"
                 "RTX 5090 · Quartet harness · the accuracy half of 'native FP4 training can be optimal'",
                 fontsize=11.5)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.text(0.97, 0.55,
            "Full Quartet recipe: 4-bit weights + activations + gradients.\n"
            "Pairs with the speed result (FP4 ~5.5× BF16 GEMM):\n"
            "FP4 training is FAST and ACCURATE → optimal.\n"
            "(pseudo-quant convergence run; speed from kernels.)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7fafc", ec="#cbd5e0", lw=0.8))
    fig.tight_layout(); fig.savefig(OUTPNG, dpi=130)
    print("wrote", OUTPNG)
    print(f"final BF16 {bf16['val_loss'][-1]} / FP4 {fp4['val_loss'][-1]} gap {final_gap} pp_ratio {pp_ratio}")


if __name__ == "__main__":
    main()
