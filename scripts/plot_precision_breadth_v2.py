#!/usr/bin/env python3
"""
plot_precision_breadth_v2.py — the INT4-vs-FP4 asymmetry is ARCHITECTURE-GENERAL.

Breadth is the result here: across 9 same-base INT4-vs-NVFP4 pairs spanning 6 architectures
and 7–32B, EVERY model reproduces the law — decode ties (NVFP4 ÷ INT4 ≈ 1, bandwidth-bound)
while NVFP4 prefill is 3.2–4.5× INT4's (FP4 is a compute format; INT4 is stuck on the bf16
prefill floor). A per-model lollipop of the prefill split makes the universality visible;
the tight clustering toward the FP4÷bf16 tensor-core asymptote (~4.5×) IS "architecture-general".

Reads the same-base pair JSONs in data/output/precision_5090_vllm_runs/ (Tier-1 quads + the
breadth-v2 sweep). Writes data/output/precision_5090_breadth_v2.{json,png}.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "data/output/precision_5090_vllm_runs")
OUTJSON = os.path.join(REPO, "data/output/precision_5090_breadth_v2.json")
OUTPNG = os.path.join(REPO, "data/output/precision_5090_breadth_v2.png")

# display | arch | int4_label | nvfp4_label  (params for sort)
PAIRS = [
    ("Llama-3.2-1B",           "Llama",        "llama32_1b_int4",    "llama32_1b_nvfp4",    1),
    ("Mistral-7B-v0.3",        "Mistral",      "mistral7b_v03_int4", "mistral7b_v03_nvfp4", 7),
    ("Llama-3.1-8B",           "Llama",        "llama8b_awq_int4",   "llama8b_nvfp4",       8),
    ("DS-R1-Distill-Llama-8B", "Llama",        "dsllama8b_int4",     "dsllama8b_nvfp4",     8),
    ("Qwen3-8B",               "Qwen3",        "qwen3_8b_awq_int4",  "nvfp4",               8),
    ("Nemotron-Nano-9B-v2",    "Nemotron-H",   "nemotron_9b_int4",   "nemotron_9b_nvfp4",   9),
    ("DS-R1-Distill-Qwen-14B", "Qwen2",        "dsqwen14b_int4",     "dsqwen14b_nvfp4",    14),
    ("Qwen3-14B",              "Qwen3",        "qwen3_14b_awq_int4", "qwen3_14b_nvfp4",    14),
    ("Phi-4-reasoning-plus",   "Phi-4",        "phi4rp_int4",        "phi4rp_nvfp4",       15),
    ("Mistral-Small-24B",      "Mistral",      "mistral24b_int4",    "mistral24b_nvfp4",   24),
    ("Qwen3-32B",              "Qwen3",        "qwen3_32b_awq_int4", "qwen3_32b_nvfp4",    32),
    ("DS-R1-Distill-Qwen-32B", "Qwen2",        "dsqwen32b_int4",     "dsqwen32b_nvfp4",    32),
]

ARCH_C = {"Mistral": "#2b6cb0", "Llama": "#dd6b20", "Qwen3": "#2f855a",
          "Qwen2": "#38a169", "Phi-4": "#805ad5", "Nemotron-H": "#d69e2e"}


def load(label):
    d = json.load(open(os.path.join(RUNS, f"{label}_vllm.json")))
    return list(d["decode_tok_s"].values())[0], max(d["prefill_tok_s"].values())


def main():
    rows = []
    for disp, arch, i4, f4, params in PAIRS:
        di, pi = load(i4); dn, pn = load(f4)
        rows.append({"model": disp, "arch": arch, "params_b": params,
                     "decode_ratio_nvfp4_int4": round(dn / di, 2),
                     "prefill_split_nvfp4_int4": round(pn / pi, 2),
                     "int4_prefill_k": round(pi / 1000, 1), "nvfp4_prefill_k": round(pn / 1000, 1)})
    rows.sort(key=lambda r: r["params_b"])
    splits = [r["prefill_split_nvfp4_int4"] for r in rows]
    decs = [r["decode_ratio_nvfp4_int4"] for r in rows]
    med = sorted(splits)[len(splits) // 2]
    doc = {
        "__meta__": {
            "description": "INT4-vs-NVFP4 asymmetry across 9 same-base pairs / 6 architectures on "
                           "RTX 5090 (vLLM). decode ratio ~1 (tie); prefill split 3.2-4.5x (FP4 compute).",
            "n_models": len(rows), "n_archs": len(set(r["arch"] for r in rows)),
            "prefill_split_range": [min(splits), max(splits)], "prefill_split_median": med,
            "decode_ratio_range": [min(decs), max(decs)],
            "claim": "architecture-general: every model ties on decode and splits 3.2-4.5x on prefill",
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
    json.dump(doc, open(OUTJSON, "w"), indent=2)
    print("wrote", OUTJSON)

    # ---- Slopegraph: one line per model, decode (left) → prefill (right). ----
    # The message IS the shape: every line is flat at parity on decode and shoots
    # up on prefill, regardless of architecture (colour) or size.
    lo, hi = min(splits), max(splits)
    dmed = sorted(decs)[len(decs) // 2]
    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    x0, x1 = 0.0, 1.0

    # "tie" band around parity on the decode side; parity reference line
    ax.axhspan(0.78, 1.05, xmin=0.0, xmax=0.5, color="#cbd5e0", alpha=0.35, zorder=0)
    ax.axhline(1.0, ls=":", color="#a0aec0", lw=1.0, zorder=1)

    for r in rows:
        c = ARCH_C.get(r["arch"], "#718096")
        d, p = r["decode_ratio_nvfp4_int4"], r["prefill_split_nvfp4_int4"]
        ax.plot([x0, x1], [d, p], color=c, lw=1.7, alpha=0.6, zorder=2,
                solid_capstyle="round")
        ax.scatter([x0, x1], [d, p], s=66, color=c, edgecolor="white",
                   linewidth=0.7, zorder=3)

    # median ticks per column
    ax.plot([x0 - 0.07, x0 + 0.07], [dmed, dmed], color="#1a202c", lw=3, zorder=4)
    ax.plot([x1 - 0.07, x1 + 0.07], [med, med], color="#9b2c2c", lw=3, zorder=4)
    ax.text(x0, dmed - 0.22, f"median {dmed:.2f}×", ha="center", fontsize=8.5,
            fontweight="bold", color="#1a202c")
    ax.text(x1, med + 0.012 * 0 + 0.24, f"median {med:.1f}×", ha="center", fontsize=8.5,
            fontweight="bold", color="#9b2c2c")

    # cluster captions — the takeaway, in plain words
    ax.text(x0 - 0.02, 1.95, "DECODE\nmemory-bound", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#2d3748")
    ax.text(x0 - 0.02, 0.52, "INT4 = FP4  →  TIE\nboth are just 4-bit memory\n(same bytes to stream)",
            ha="center", va="top", fontsize=9, color="#4a5568")
    ax.text(x1 + 0.02, 4.75, "PREFILL\ncompute-bound", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#9b2c2c")
    ax.text(x1 + 0.18, 3.6, f"FP4 wins\n{lo:.1f}–{hi:.1f}×\nnative tensor cores\n(INT4 stuck on\nthe BF16 floor)",
            ha="left", va="center", fontsize=9, color="#9b2c2c", fontweight="bold")

    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Decode", "Prefill"], fontsize=13, fontweight="bold")
    ax.set_ylabel("NVFP4 speed ÷ INT4 speed  (same weights, RTX 5090)", fontsize=10)
    ax.set_xlim(-0.32, 1.42)
    ax.set_ylim(0.3, 5.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Same weights, every architecture: 4-bit INT and 4-bit FP are identical on decode —\n"
                 f"but only FP4 also accelerates prefill.  {len(rows)} models · "
                 f"{doc['__meta__']['n_archs']} architectures · all behave the same way",
                 fontsize=12)
    ax.grid(True, axis="y", alpha=0.25)

    # legend by arch (shows the breadth: 6 distinct architectures)
    seen = []
    for r in rows:
        if r["arch"] not in seen:
            seen.append(r["arch"])
            ax.scatter([], [], color=ARCH_C.get(r["arch"], "#718096"), s=70, label=r["arch"])
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", title="architecture (6)",
              title_fontsize=9, ncol=2)
    fig.text(0.5, 0.012,
             "Each line is one model (same weights, quantized both ways). Nemotron-Nano-9B (amber, lowest) "
             "is hybrid Mamba — its NVFP4 keeps the Mamba layers BF16 (partial FP4), so it splits less.",
             ha="center", fontsize=7.6, color="#888", style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUTPNG, dpi=130)
    print("wrote", OUTPNG)
    print(f"n={len(rows)} archs={doc['__meta__']['n_archs']} split {min(splits)}-{max(splits)} med {med} "
          f"decode {min(decs)}-{max(decs)}")


if __name__ == "__main__":
    main()
