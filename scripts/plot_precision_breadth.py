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
    # Tier-1 size-controlled anchor: Qwen3-8B at three precisions.
    ("bf16",               "Qwen3-8B BF16",        "bf16", "ctrl"),
    ("fp8",                "Qwen3-8B FP8",         "fp8",  "ctrl"),
    ("nvfp4",              "Qwen3-8B NVFP4",       "fp4",  "ctrl"),
    # Breadth FP8/BF16 reproducibility (other ~7-8B archs).
    ("llama8b_bf16",       "Llama-8B BF16",        "bf16", "ctrl"),
    ("llama8b_fp8",        "Llama-8B FP8",         "fp8",  "ctrl"),
    ("mistral7b_bf16",     "Mistral-7B BF16",      "bf16", "ctrl"),
    ("mistral7b_fp8",      "Mistral-7B FP8",       "fp8",  "ctrl"),
    # The INT4 stars (Qwen2.5-7B, ~size-matched to the Qwen3-8B anchor).
    ("qwen7b_awq_int4",    "Qwen2.5-7B INT4 (AWQ)",  "int4", "ctrl"),
    ("qwen7b_gptq_int4",   "Qwen2.5-7B INT4 (GPTQ)", "int4", "ctrl"),
    # Larger breadth models — size break flagged as diamonds.
    ("gptoss20b_mxfp4",    "gpt-oss-20B MXFP4",    "fp4",  "big"),
    ("deepseekv2lite_fp8", "DeepSeek-V2-Lite FP8 (MoE)", "fp8", "big"),
]

# Inline labels (rest stay clean). AWQ/GPTQ are near-coincident -> one combined label.
CUSTOM_LABEL = {
    "fp8":                 "Qwen3-8B FP8",
    "nvfp4":               "Qwen3-8B NVFP4",
    "qwen7b_awq_int4":     "Qwen2.5-7B INT4 (AWQ, GPTQ)",
    "gptoss20b_mxfp4":     "gpt-oss-20B MXFP4",
    "deepseekv2lite_fp8":  "DeepSeek-V2-Lite FP8 (MoE)",
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

    doc = {
        "__meta__": {
            "description": "vLLM breadth sweep on RTX 5090: decode (BW) vs prefill (compute) by "
                           "quantization format. INT4 = memory-only; FP4 = memory + compute.",
            "schema_version": 1,
            "methodology_version": "2026-06-02-precision-breadth-v1",
            "runtime": "vLLM 0.22.0, single-stream (batch=1, greedy, ignore_eos), tg256 / prefill plateau",
            "headline": "4-bit INT (AWQ/GPTQ) wins decode (~245-250 tok/s, bandwidth) but its prefill "
                        "stays at bf16 level (~14.9k tok/s) — no low-precision tensor-core path; it "
                        "dequantizes to compute. FP4 (NVFP4/MXFP4) wins BOTH decode and prefill on "
                        "Blackwell's native FP4 tensor cores. FP8 is the mid anchor, ~1.6x decode / "
                        "~1.7x prefill over bf16, reproduced across Qwen3-8B, Llama-8B, Mistral-7B.",
        },
        "rows": rows,
        "fp8_over_bf16_controlled": ratios,
        "establishes": [
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
    for r in rows:
        color, flabel = FAM[r["family"]]
        marker = "D" if r["size_class"] == "big" else "o"
        size = 130 if r["size_class"] == "big" else 95
        lbl = flabel if r["family"] not in seen_fam else None
        seen_fam.add(r["family"])
        ax.scatter(r["decode_tok_s"], r["prefill_peak_tok_s"] / 1000.0,
                   s=size, color=color, marker=marker, zorder=3,
                   edgecolor="white", linewidth=0.6, label=lbl)
        if r["label"] in CUSTOM_LABEL:
            dy = -12 if r["label"] == "qwen7b_awq_int4" else 5   # INT4 label drops below floor
            ax.annotate(CUSTOM_LABEL[r["label"]],
                        (r["decode_tok_s"], r["prefill_peak_tok_s"] / 1000.0),
                        textcoords="offset points", xytext=(7, dy), fontsize=7.5)

    # Guide: the bf16 prefill floor — INT4 sits ON it despite winning decode.
    bf16_prefill_k = by["bf16"]["prefill_peak_tok_s"] / 1000.0
    ax.axhline(bf16_prefill_k, ls=":", color="#a0aec0", lw=1.1, zorder=1)
    ax.annotate("← bf16 prefill floor: INT4 wins decode but never clears it\n   (no FP tensor-core path — it dequantizes to compute)",
                (150, 16.6), fontsize=8, color="#dd6b20")

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
