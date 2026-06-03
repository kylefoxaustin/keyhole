#!/usr/bin/env python3
"""
plot_precision_breadth.py — the INT4-vs-FP4 asymmetry chart from the vLLM breadth sweep.

Plots each (format, model) run on a decode-vs-prefill plane, where the two axes are the
two resources a quantization format can help with:
  x = decode throughput (single-stream tg256)  -> BANDWIDTH-bound  (memory-format win)
  y = prefill plateau    (peak tok/s)           -> COMPUTE-bound    (tensor-core win)

The headline the chart makes visible:
  - 4-bit INT (AWQ/GPTQ) sits FAR-RIGHT but LOW — it wins decode (small bytes) yet its
    prefill stays at bf16 level, because INT4 dequantizes to compute (no low-precision
    tensor-core path). INT4 = a MEMORY format.
  - FP4 (NVFP4 / MXFP4) sits far-right AND high — it wins decode AND prefill, because
    Blackwell has native FP4 tensor cores. FP4 = a MEMORY + COMPUTE format.
  - FP8 is the mid anchor (native fp8 tensor cores): ~1.6x decode / ~1.7x prefill over
    bf16, reproduced across 3 architectures (Qwen3-8B, Llama-8B, Mistral-7B).

Size note: the decode axis is partly size-confounded across models (gpt-oss-20b is 20B,
DeepSeek-V2-Lite is a 16B MoE). The size-CONTROLLED comparison is the ~7-8B class
(circles); the two larger breadth models are drawn as diamonds so the size break is
visible. The prefill axis cleanly separates INT4 (~bf16) from FP4/FP8 regardless of size,
which is the actual claim.

Reads data/output/precision_5090_vllm_runs/<label>_vllm.json (the Tier-1 + breadth runs).
Writes data/output/precision_5090_breadth.{json,png}.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "data/output/precision_5090_vllm_runs")
OUTJSON = os.path.join(REPO, "data/output/precision_5090_breadth.json")
OUTPNG = os.path.join(REPO, "data/output/precision_5090_breadth.png")

# Format family -> colour. The four regimes the chart is about.
# Short legend labels — the title + floor annotation carry the "memory vs compute" teaching,
# so keeping these compact avoids the legend creeping into the FP8 ratio box.
FAM = {
    "bf16": ("#718096", "BF16"),
    "fp8":  ("#2b6cb0", "FP8"),
    "int4": ("#dd6b20", "INT4 (memory-only)"),
    "fp4":  ("#2f855a", "FP4 (memory + compute)"),
}

# label | display | format-family | size-class (circle = size-controlled ~7-8B, diamond = larger)
RUNS_SPEC = [
    # Tier-1 size-controlled anchor: Qwen3-8B at FOUR precisions on IDENTICAL weights —
    # the clean same-base quad (INT4 and NVFP4 below differ only in format, not model).
    ("bf16",               "Qwen3-8B BF16",        "bf16", "ctrl"),
    ("fp8",                "Qwen3-8B FP8",         "fp8",  "ctrl"),
    ("qwen3_8b_awq_int4",  "Qwen3-8B INT4 (AWQ)",  "int4", "ctrl"),
    ("nvfp4",              "Qwen3-8B NVFP4",       "fp4",  "ctrl"),
    # Breadth FP8/BF16 reproducibility (other ~7-8B archs).
    ("llama8b_bf16",       "Llama-8B BF16",        "bf16", "ctrl"),
    ("llama8b_fp8",        "Llama-8B FP8",         "fp8",  "ctrl"),
    # Cross-arch same-base quad (Llama-3.1-8B): asymmetry is architecture-general (3.2x split).
    ("llama8b_awq_int4",   "Llama-8B INT4 (AWQ)",  "int4", "ctrl"),
    ("llama8b_nvfp4",      "Llama-8B NVFP4",       "fp4",  "ctrl"),
    ("mistral7b_bf16",     "Mistral-7B BF16",      "bf16", "ctrl"),
    ("mistral7b_fp8",      "Mistral-7B FP8",       "fp8",  "ctrl"),
    # The INT4 stars (Qwen2.5-7B, ~size-matched to the Qwen3-8B anchor).
    ("qwen7b_awq_int4",    "Qwen2.5-7B INT4 (AWQ)",  "int4", "ctrl"),
    ("qwen7b_gptq_int4",   "Qwen2.5-7B INT4 (GPTQ)", "int4", "ctrl"),
    # Same-base scale family: split widens then plateaus (3.5x @8B -> 4.4x @14B -> 4.5x @32B).
    ("qwen3_14b_awq_int4", "Qwen3-14B INT4 (AWQ)",  "int4", "q14"),
    ("qwen3_14b_nvfp4",    "Qwen3-14B NVFP4",       "fp4",  "q14"),
    ("qwen3_32b_awq_int4", "Qwen3-32B INT4 (AWQ)",  "int4", "q32"),
    ("qwen3_32b_nvfp4",    "Qwen3-32B NVFP4",       "fp4",  "q32"),
    # Larger breadth models — size break flagged as diamonds.
    ("gptoss20b_mxfp4",    "gpt-oss-20B MXFP4",    "fp4",  "big"),
    ("deepseekv2lite_fp8", "DeepSeek-V2-Lite FP8 (MoE)", "fp8", "big"),
]

# Inline labels (rest stay clean). AWQ/GPTQ are near-coincident -> one combined label.
CUSTOM_LABEL = {
    "fp8":                 "Qwen3-8B FP8",
    "nvfp4":               "Qwen3-8B NVFP4",
    "qwen3_8b_awq_int4":   "Qwen3-8B INT4 (AWQ)",
    "qwen7b_awq_int4":     "Qwen2.5-7B INT4 (AWQ, GPTQ)",
    "qwen3_14b_awq_int4":  "Qwen3-14B INT4 (AWQ)",
    "qwen3_14b_nvfp4":     "Qwen3-14B NVFP4",
    "qwen3_32b_awq_int4":  "Qwen3-32B INT4 (AWQ)",
    "qwen3_32b_nvfp4":     "Qwen3-32B NVFP4",
    "llama8b_nvfp4":       "Llama-8B NVFP4",
    "gptoss20b_mxfp4":     "gpt-oss-20B MXFP4",
    "deepseekv2lite_fp8":  "DeepSeek-V2-Lite FP8 (MoE)",
}
# Per-label annotation offsets (pts). INT4 labels drop below the floor to avoid the marker.
LABEL_DXY = {
    "qwen3_8b_awq_int4":  (7, -11),
    "qwen7b_awq_int4":    (7, -27),   # extra drop so the two INT4 labels sit on distinct lines
    "qwen3_14b_awq_int4": (-150, -2), # 14B INT4 sits low-left; label to its left
    "qwen3_14b_nvfp4":    (-118, 4),  # 14B NVFP4 label to its left to clear the 8B FP8 cluster
    "qwen3_32b_awq_int4": (8, -4),    # 32B INT4 far-left low; label to its right
    "qwen3_32b_nvfp4":    (8, 2),     # 32B NVFP4 far-left; label to its right
    "llama8b_nvfp4":      (7, 4),     # Llama-8B NVFP4 near Qwen3-8B NVFP4; label up-right
}


def load(label):
    d = json.load(open(os.path.join(RUNS, f"{label}_vllm.json")))
    decode = list(d["decode_tok_s"].values())[0]
    prefill_peak = max(d["prefill_tok_s"].values())
    return decode, prefill_peak


def main():
    rows = []
    for label, disp, fam, size in RUNS_SPEC:
        decode, prefill = load(label)
        rows.append({"label": label, "display": disp, "family": fam, "size_class": size,
                     "decode_tok_s": round(decode, 1), "prefill_peak_tok_s": round(prefill, 1)})

    # Controlled same-model FP8/BF16 ratios (the defensible number, size-confound-free).
    by = {r["label"]: r for r in rows}
    ratios = {}
    for fp8l, bf16l, name in [("llama8b_fp8", "llama8b_bf16", "Llama-8B"),
                              ("mistral7b_fp8", "mistral7b_bf16", "Mistral-7B"),
                              ("fp8", "bf16", "Qwen3-8B")]:
        ratios[name] = {
            "decode_x": round(by[fp8l]["decode_tok_s"] / by[bf16l]["decode_tok_s"], 2),
            "prefill_x": round(by[fp8l]["prefill_peak_tok_s"] / by[bf16l]["prefill_peak_tok_s"], 2),
        }

    # Same-base quad: Qwen3-8B at BF16/FP8/INT4/NVFP4 on IDENTICAL weights — the
    # size-confound-free core of the claim (no cross-model comparison needed).
    quad = {}
    for lab, nm in [("bf16", "BF16"), ("fp8", "FP8"),
                    ("qwen3_8b_awq_int4", "INT4-AWQ"), ("nvfp4", "NVFP4")]:
        quad[nm] = {"decode_tok_s": by[lab]["decode_tok_s"],
                    "prefill_peak_tok_s": by[lab]["prefill_peak_tok_s"],
                    "decode_x_bf16": round(by[lab]["decode_tok_s"] / by["bf16"]["decode_tok_s"], 2),
                    "prefill_x_bf16": round(by[lab]["prefill_peak_tok_s"] / by["bf16"]["prefill_peak_tok_s"], 2)}
    quad["int4_vs_nvfp4_prefill_split"] = round(
        by["nvfp4"]["prefill_peak_tok_s"] / by["qwen3_8b_awq_int4"]["prefill_peak_tok_s"], 2)

    # Helper: same-base INT4 vs NVFP4 pair summary.
    def pair_summary(int4_lab, fp4_lab):
        return {"INT4-AWQ": {"decode_tok_s": by[int4_lab]["decode_tok_s"],
                             "prefill_peak_tok_s": by[int4_lab]["prefill_peak_tok_s"]},
                "NVFP4": {"decode_tok_s": by[fp4_lab]["decode_tok_s"],
                          "prefill_peak_tok_s": by[fp4_lab]["prefill_peak_tok_s"]},
                "int4_vs_nvfp4_prefill_split": round(
                    by[fp4_lab]["prefill_peak_tok_s"] / by[int4_lab]["prefill_peak_tok_s"], 2),
                "decode_ratio_nvfp4_over_int4": round(
                    by[fp4_lab]["decode_tok_s"] / by[int4_lab]["decode_tok_s"], 2)}

    quad14 = pair_summary("qwen3_14b_awq_int4", "qwen3_14b_nvfp4")
    quad32 = pair_summary("qwen3_32b_awq_int4", "qwen3_32b_nvfp4")
    llama_quad = pair_summary("llama8b_awq_int4", "llama8b_nvfp4")
    # Scale trend (same-base Qwen3): widens then plateaus as prefill saturates compute-bound.
    prefill_split_by_scale = {"qwen3_8b": quad["int4_vs_nvfp4_prefill_split"],
                              "qwen3_14b": quad14["int4_vs_nvfp4_prefill_split"],
                              "qwen3_32b": quad32["int4_vs_nvfp4_prefill_split"]}

    doc = {
        "__meta__": {
            "description": "vLLM breadth sweep on RTX 5090: decode (BW) vs prefill (compute) by "
                           "quantization format. INT4 = memory-only; FP4 = memory + compute.",
            "schema_version": 2,
            "methodology_version": "2026-06-03-precision-breadth-v2-samebase-quad",
            "runtime": "vLLM 0.22.0, single-stream (batch=1, greedy, ignore_eos), tg256 / prefill plateau",
            "headline": "Same-base Qwen3-8B quad (IDENTICAL weights): INT4-AWQ and NVFP4 TIE on decode "
                        f"({quad['INT4-AWQ']['decode_tok_s']} vs {quad['NVFP4']['decode_tok_s']} tok/s, both "
                        f"~{quad['NVFP4']['decode_x_bf16']}x bf16, bandwidth-bound) — but INT4 prefill is pinned "
                        f"to the bf16 floor ({quad['INT4-AWQ']['prefill_x_bf16']}x) while NVFP4 prefill is "
                        f"{quad['NVFP4']['prefill_x_bf16']}x bf16 — a {quad['int4_vs_nvfp4_prefill_split']}x prefill split "
                        "on the same weights. INT4 dequantizes to compute (memory-only); FP4 has native sm_120 "
                        "tensor cores (memory + compute). FP8 is the mid anchor (~1.6x/1.7x), reproduced across "
                        "Qwen3-8B, Llama-8B, Mistral-7B.",
        },
        "rows": rows,
        "same_base_quad_qwen3_8b": quad,
        "same_base_pair_qwen3_14b": quad14,
        "same_base_pair_qwen3_32b": quad32,
        "same_base_pair_llama_8b": llama_quad,
        "prefill_split_by_scale": prefill_split_by_scale,
        "fp8_over_bf16_controlled": ratios,
        "establishes": [
            "Same-base Qwen3-8B: INT4 and NVFP4 tie on decode but NVFP4 prefill is "
            f"{quad['int4_vs_nvfp4_prefill_split']}x INT4's — controlled proof, no size confound.",
            "The prefill split widens then PLATEAUS with model size: "
            f"{prefill_split_by_scale['qwen3_8b']}x (8B) -> {prefill_split_by_scale['qwen3_14b']}x (14B) -> "
            f"{prefill_split_by_scale['qwen3_32b']}x (32B) — it asymptotes to the FP4-vs-bf16 tensor-core "
            "throughput ratio as prefill becomes fully compute-bound.",
            "Architecture-general, not a Qwen artifact: Llama-3.1-8B same-base split is "
            f"{llama_quad['int4_vs_nvfp4_prefill_split']}x (vs Qwen3-8B {quad['int4_vs_nvfp4_prefill_split']}x).",
            "INT4 is a memory format: decode win (BW) but prefill ~= bf16 (no tensor-core compute path).",
            "FP4 (NVFP4/MXFP4) is a memory + compute format: wins decode AND prefill on native FP4 cores.",
            "FP8/BF16 advantage is architecture-general (~1.6x decode, ~1.7x prefill across 3 models).",
        ],
    }
    json.dump(doc, open(OUTJSON, "w"), indent=2)
    print("wrote", OUTJSON)

    # ================= two panels: (L) regime scatter, (R) scale curve =================
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6.0),
                                   gridspec_kw={"width_ratios": [1.75, 1]})
    MARKER = {"big": "D", "q14": "s", "q32": "^"}  # diamond=breadth; square=14B; triangle=32B (scale family)
    # Only the hero pair + the breadth diamonds get inline labels in the scatter; the scale
    # family (14B/32B) and Llama are named in the right panel, so they stay as clean dots here.
    AXL_LABELS = {"nvfp4", "qwen3_8b_awq_int4", "fp8", "qwen7b_awq_int4",
                  "gptoss20b_mxfp4", "deepseekv2lite_fp8"}

    # ---- (L) decode (x, bandwidth) vs prefill plateau (y, compute) ----
    seen_fam = set()
    for r in rows:
        color, flabel = FAM[r["family"]]
        marker = MARKER.get(r["size_class"], "o")
        size = 130 if r["size_class"] == "big" else 95
        lbl = flabel if r["family"] not in seen_fam else None
        seen_fam.add(r["family"])
        axL.scatter(r["decode_tok_s"], r["prefill_peak_tok_s"] / 1000.0,
                    s=size, color=color, marker=marker, zorder=3,
                    edgecolor="white", linewidth=0.6, label=lbl)
        if r["label"] in CUSTOM_LABEL and r["label"] in AXL_LABELS:
            dxy = LABEL_DXY.get(r["label"], (7, 5))
            axL.annotate(CUSTOM_LABEL[r["label"]],
                         (r["decode_tok_s"], r["prefill_peak_tok_s"] / 1000.0),
                         textcoords="offset points", xytext=dxy, fontsize=7.5)

    # Guide: the bf16 prefill floor — INT4 sits ON it despite winning decode.
    bf16_prefill_k = by["bf16"]["prefill_peak_tok_s"] / 1000.0
    axL.axhline(bf16_prefill_k, ls=":", color="#a0aec0", lw=1.1, zorder=1)
    axL.annotate("← bf16 prefill floor: every INT4 point sits on it\n   (no FP tensor-core path — it dequantizes to compute)",
                 (150, 16.4), fontsize=8, color="#dd6b20")

    # Hero money shot: Qwen3-8B INT4 vs NVFP4 on IDENTICAL weights — near-equal decode but
    # NVFP4 prefill is 3.5x INT4's. (The 14B/32B same-base pairs are the right panel.)
    i4, f4 = by["qwen3_8b_awq_int4"], by["nvfp4"]
    axL.annotate("", xy=(f4["decode_tok_s"], f4["prefill_peak_tok_s"] / 1000.0),
                 xytext=(i4["decode_tok_s"], i4["prefill_peak_tok_s"] / 1000.0),
                 arrowprops=dict(arrowstyle="<->", color="#9b2c2c", lw=1.4), zorder=4)
    axL.annotate(f"same weights · same decode\n{quad['int4_vs_nvfp4_prefill_split']:.1f}× prefill split (Qwen3-8B)",
                 ((i4["decode_tok_s"] + f4["decode_tok_s"]) / 2.0,
                  (i4["prefill_peak_tok_s"] + f4["prefill_peak_tok_s"]) / 2000.0),
                 textcoords="offset points", xytext=(12, 0), fontsize=8, color="#9b2c2c", va="center")

    axL.set_xlabel("decode throughput (tok/s, single-stream tg256)  →  bandwidth-bound")
    axL.set_ylabel("prefill plateau (k tok/s)  →  compute-bound")
    axL.set_title("INT4 wins only decode; FP4 wins both axes", fontsize=10.5)
    axL.grid(True, alpha=0.3)
    axL.legend(frameon=False, fontsize=8, loc="upper left")
    q = ratios["Qwen3-8B"]; l = ratios["Llama-8B"]; m = ratios["Mistral-7B"]
    axL.text(0.40, 0.985,
             "FP8 ÷ BF16, same model (size-controlled)\n"
             f"   Qwen3-8B   {q['decode_x']}× / {q['prefill_x']}×\n"
             f"   Llama-8B    {l['decode_x']}× / {l['prefill_x']}×\n"
             f"   Mistral-7B  {m['decode_x']}× / {m['prefill_x']}×\n"
             "   (decode / prefill)",
             transform=axL.transAxes, ha="left", va="top", fontsize=7.5, color="#2b6cb0",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ebf4ff", ec="#2b6cb0", lw=0.8))

    # ---- (R) scale curve: prefill split (NVFP4 ÷ INT4) vs model size, same weights ----
    sizes = [8, 14, 32]
    splits = [prefill_split_by_scale["qwen3_8b"], prefill_split_by_scale["qwen3_14b"],
              prefill_split_by_scale["qwen3_32b"]]
    axR.plot(sizes, splits, "-o", color="#9b2c2c", lw=2.0, ms=8, zorder=3, label="Qwen3 (same weights)")
    for x, y in zip(sizes, splits):
        axR.annotate(f"{y:.1f}×", (x, y), textcoords="offset points", xytext=(6, 7),
                     fontsize=9, color="#9b2c2c", fontweight="bold")
    # asymptote: the split saturates toward the FP4-vs-bf16 tensor-core throughput ratio.
    axR.axhline(splits[-1], ls="--", color="#a0aec0", lw=1.1, zorder=1)
    axR.annotate("compute-bound asymptote\n(FP4 ÷ bf16 tensor-core ratio)", (15.5, splits[-1]),
                 textcoords="offset points", xytext=(0, 6), fontsize=7.5, color="#718096")
    axR.axhline(1.0, ls=":", color="#cbd5e0", lw=1.0, zorder=1)
    axR.annotate("1× = no split (INT4 = FP4)", (8, 1.0), textcoords="offset points",
                 xytext=(0, 6), fontsize=7.5, color="#a0aec0")
    # cross-arch point: Llama-8B at 8B sits right on the Qwen curve -> not a Qwen artifact.
    sL = llama_quad["int4_vs_nvfp4_prefill_split"]
    axR.scatter([8], [sL], s=110, marker="*", color="#2f855a", zorder=4,
                edgecolor="white", linewidth=0.6, label="Llama-3.1-8B (cross-arch)")
    axR.annotate(f"Llama-8B {sL:.1f}×", (8, sL), textcoords="offset points", xytext=(8, -14),
                 fontsize=8, color="#2f855a")
    axR.set_xlabel("model size (B params)")
    axR.set_ylabel("prefill split  =  NVFP4 ÷ INT4  (same weights)")
    axR.set_title("Split widens with size, then plateaus", fontsize=10.5)
    axR.set_xticks(sizes); axR.set_xticklabels(["8B", "14B", "32B"])
    axR.set_ylim(0.5, 5.2)
    axR.grid(True, alpha=0.3)
    axR.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("INT4 is a memory format; FP4 is a memory + compute format  ·  "
                 "RTX 5090 · vLLM single-stream", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUTPNG, dpi=130)
    print("wrote", OUTPNG)


if __name__ == "__main__":
    main()
