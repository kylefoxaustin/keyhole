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
FAM = {
    "bf16": ("#718096", "BF16 (baseline)"),
    "fp8":  ("#2b6cb0", "FP8 (native fp8 tensor cores)"),
    "int4": ("#dd6b20", "INT4 AWQ/GPTQ (memory format)"),
    "fp4":  ("#2f855a", "FP4 NVFP4/MXFP4 (memory + compute)"),
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
    ("mistral7b_bf16",     "Mistral-7B BF16",      "bf16", "ctrl"),
    ("mistral7b_fp8",      "Mistral-7B FP8",       "fp8",  "ctrl"),
    # The INT4 stars (Qwen2.5-7B, ~size-matched to the Qwen3-8B anchor).
    ("qwen7b_awq_int4",    "Qwen2.5-7B INT4 (AWQ)",  "int4", "ctrl"),
    ("qwen7b_gptq_int4",   "Qwen2.5-7B INT4 (GPTQ)", "int4", "ctrl"),
    # 2nd same-base quad at 14B (squares): the prefill split WIDENS with scale (3.5x -> 4.4x).
    ("qwen3_14b_awq_int4", "Qwen3-14B INT4 (AWQ)",  "int4", "q14"),
    ("qwen3_14b_nvfp4",    "Qwen3-14B NVFP4",       "fp4",  "q14"),
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
    "gptoss20b_mxfp4":     "gpt-oss-20B MXFP4",
    "deepseekv2lite_fp8":  "DeepSeek-V2-Lite FP8 (MoE)",
}
# Per-label annotation offsets (pts). INT4 labels drop below the floor to avoid the marker.
LABEL_DXY = {
    "qwen3_8b_awq_int4":  (7, -11),
    "qwen7b_awq_int4":    (7, -27),   # extra drop so the two INT4 labels sit on distinct lines
    "qwen3_14b_awq_int4": (-150, -2), # 14B INT4 sits low-left; label to its left
    "qwen3_14b_nvfp4":    (-118, 4),  # 14B NVFP4 label to its left to clear the 8B FP8 cluster
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

    # 2nd same-base quad at 14B + the scale trend (the split widens with model size).
    quad14 = {}
    for lab, nm in [("qwen3_14b_awq_int4", "INT4-AWQ"), ("qwen3_14b_nvfp4", "NVFP4")]:
        quad14[nm] = {"decode_tok_s": by[lab]["decode_tok_s"],
                      "prefill_peak_tok_s": by[lab]["prefill_peak_tok_s"]}
    quad14["int4_vs_nvfp4_prefill_split"] = round(
        by["qwen3_14b_nvfp4"]["prefill_peak_tok_s"] / by["qwen3_14b_awq_int4"]["prefill_peak_tok_s"], 2)
    quad14["decode_ratio_nvfp4_over_int4"] = round(
        by["qwen3_14b_nvfp4"]["decode_tok_s"] / by["qwen3_14b_awq_int4"]["decode_tok_s"], 2)
    prefill_split_by_scale = {"qwen3_8b": quad["int4_vs_nvfp4_prefill_split"],
                              "qwen3_14b": quad14["int4_vs_nvfp4_prefill_split"]}

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
        "prefill_split_by_scale": prefill_split_by_scale,
        "fp8_over_bf16_controlled": ratios,
        "establishes": [
            "Same-base Qwen3-8B: INT4 and NVFP4 tie on decode but NVFP4 prefill is "
            f"{quad['int4_vs_nvfp4_prefill_split']}x INT4's — controlled proof, no size confound.",
            "The prefill split WIDENS with model size: "
            f"{prefill_split_by_scale['qwen3_8b']}x at 8B -> {prefill_split_by_scale['qwen3_14b']}x at 14B "
            "(larger GEMMs are more compute-bound, so the native-FP4 tensor-core win grows).",
            "INT4 is a memory format: decode win (BW) but prefill ~= bf16 (no tensor-core compute path).",
            "FP4 (NVFP4/MXFP4) is a memory + compute format: wins decode AND prefill on native FP4 cores.",
            "FP8/BF16 advantage is architecture-general (~1.6x decode, ~1.7x prefill across 3 models).",
        ],
    }
    json.dump(doc, open(OUTJSON, "w"), indent=2)
    print("wrote", OUTJSON)

    # ---- scatter: decode (x, bandwidth) vs prefill plateau (y, compute) ----
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    seen_fam = set()
    MARKER = {"big": "D", "q14": "s"}   # diamond = larger breadth model; square = 14B same-base quad
    for r in rows:
        color, flabel = FAM[r["family"]]
        marker = MARKER.get(r["size_class"], "o")
        size = 130 if r["size_class"] == "big" else 95
        lbl = flabel if r["family"] not in seen_fam else None
        seen_fam.add(r["family"])
        ax.scatter(r["decode_tok_s"], r["prefill_peak_tok_s"] / 1000.0,
                   s=size, color=color, marker=marker, zorder=3,
                   edgecolor="white", linewidth=0.6, label=lbl)
        if r["label"] in CUSTOM_LABEL:
            dxy = LABEL_DXY.get(r["label"], (7, 5))
            ax.annotate(CUSTOM_LABEL[r["label"]],
                        (r["decode_tok_s"], r["prefill_peak_tok_s"] / 1000.0),
                        textcoords="offset points", xytext=dxy, fontsize=7.5)

    # Guide: the bf16 prefill floor — INT4 sits ON it despite winning decode.
    bf16_prefill_k = by["bf16"]["prefill_peak_tok_s"] / 1000.0
    ax.axhline(bf16_prefill_k, ls=":", color="#a0aec0", lw=1.1, zorder=1)
    ax.annotate("← bf16 prefill floor: INT4 wins decode but never clears it\n   (no FP tensor-core path — it dequantizes to compute)",
                (150, 16.6), fontsize=8, color="#dd6b20")

    # Same-base money shots: INT4 vs NVFP4 on IDENTICAL weights — near-equal decode, but
    # NVFP4 prefill is several-x INT4's (pinned to its bf16 floor). The split WIDENS with
    # scale: 3.5x at 8B, 4.4x at 14B. Connect each same-base pair with a labelled arrow.
    def money_shot(int4_lab, fp4_lab, name, label_dx, full_text=True):
        i4, f4 = by[int4_lab], by[fp4_lab]
        split = f4["prefill_peak_tok_s"] / i4["prefill_peak_tok_s"]
        ax.annotate("", xy=(f4["decode_tok_s"], f4["prefill_peak_tok_s"] / 1000.0),
                    xytext=(i4["decode_tok_s"], i4["prefill_peak_tok_s"] / 1000.0),
                    arrowprops=dict(arrowstyle="<->", color="#9b2c2c", lw=1.4), zorder=4)
        txt = (f"same weights · same decode\n{split:.1f}× prefill split ({name})"
               if full_text else f"{split:.1f}× split ({name})")
        ha = "left" if label_dx >= 0 else "right"
        ax.annotate(txt,
                    ((i4["decode_tok_s"] + f4["decode_tok_s"]) / 2.0,
                     (i4["prefill_peak_tok_s"] + f4["prefill_peak_tok_s"]) / 2000.0),
                    textcoords="offset points", xytext=(label_dx, 0),
                    fontsize=8, color="#9b2c2c", va="center", ha=ha)

    money_shot("qwen3_8b_awq_int4", "nvfp4", "Qwen3-8B", 12, full_text=True)
    money_shot("qwen3_14b_awq_int4", "qwen3_14b_nvfp4", "Qwen3-14B", -10, full_text=False)

    ax.set_xlabel("decode throughput (tok/s, single-stream tg256)  →  bandwidth-bound")
    ax.set_ylabel("prefill plateau (k tok/s)  →  compute-bound")
    ax.set_title("INT4 is a memory format; FP4 is a memory + compute format\n"
                 "RTX 5090 · vLLM single-stream · 4-bit INT wins only decode, FP4 wins both",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    # Controlled-ratio callout (size-confound-free claim) — boxed, in the empty centre.
    q = ratios["Qwen3-8B"]; l = ratios["Llama-8B"]; m = ratios["Mistral-7B"]
    ax.text(0.30, 0.78,
            "FP8 ÷ BF16, same model (size-controlled)\n"
            f"   Qwen3-8B   {q['decode_x']}× / {q['prefill_x']}×\n"
            f"   Llama-8B    {l['decode_x']}× / {l['prefill_x']}×\n"
            f"   Mistral-7B  {m['decode_x']}× / {m['prefill_x']}×\n"
            "   (decode / prefill)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#2b6cb0",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ebf4ff", ec="#2b6cb0", lw=0.8))
    fig.tight_layout(); fig.savefig(OUTPNG, dpi=130)
    print("wrote", OUTPNG)


if __name__ == "__main__":
    main()
