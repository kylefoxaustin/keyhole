#!/usr/bin/env python3
"""
precision_ladder.py — assemble the Tier-2 size/quant ladder (llama.cpp, RTX 5090) into
data/output/precision_5090_ladder.json + a decode-vs-footprint scaling plot.

Turns the single-point "decode is bandwidth-bound" claim into a measured curve:
  - QUANT axis (Qwen2.5-7B, fixed params): decode tok/s * weight_GB ~ const => speed ∝ 1/bytes.
  - SIZE  axis (Q4_K_M, 7B vs 32B dense): decode scales ~inverse with weight bytes.
  - MoE decoupling: Qwen3-30B-A3B occupies 32B-class VRAM but decodes faster than 7B dense,
    because decode bandwidth tracks ~3B ACTIVE params, not the 30B total footprint.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "data/output/precision_5090_runs")
OUTJSON = os.path.join(REPO, "data/output/precision_5090_ladder.json")
OUTPNG = os.path.join(REPO, "data/output/precision_5090_ladder.png")

# label -> (display, weight_GB on disk, params, active_params, family/axis tag)
MODELS = [
    ("qwen7b_q4",    "Qwen2.5-7B Q4_K_M",  4.36, "7.6B", "7.6B", "dense"),
    ("qwen7b_q5",    "Qwen2.5-7B Q5_K_M",  5.07, "7.6B", "7.6B", "dense"),
    ("qwen7b_q8",    "Qwen2.5-7B Q8_0",    7.54, "7.6B", "7.6B", "dense"),
    ("qwen32b_q4",   "Qwen2.5-32B Q4_K_M", 18.49, "32.8B", "32.8B", "dense"),
    ("qwen30bA3b_q4","Qwen3-30B-A3B Q4_K_M (MoE)", 17.28, "30.5B", "~3.3B", "moe"),
]


def load(label):
    rows = json.load(open(os.path.join(RUNS, f"{label}_bench.json")))
    pp = next((r["avg_ts"] for r in rows if r.get("n_prompt")), None)
    tg = next((r["avg_ts"] for r in rows if r.get("n_gen")), None)
    try:
        vram = int(open(os.path.join(RUNS, f"{label}_peak_vram_mib.txt")).read().strip())
    except OSError:
        vram = None
    return pp, tg, vram


def main():
    rows = []
    for label, disp, gb, params, active, kind in MODELS:
        pp, tg, vram = load(label)
        rows.append({"label": label, "display": disp, "weight_gb": gb, "params": params,
                     "active_params": active, "kind": kind,
                     "prefill_tok_s_pp2048": round(pp, 1), "decode_tok_s_tg256": round(tg, 1),
                     "peak_vram_mib": vram, "decode_x_gb": round(tg * gb, 0)})

    doc = {
        "__meta__": {
            "description": "Tier-2 size/quant ladder: decode/prefill vs weight footprint, Qwen on RTX 5090.",
            "schema_version": 1,
            "methodology_version": "2026-06-02-precision-ladder-v1",
            "runtime": "llama.cpp build c30e012 (sm_120a), llama-bench -p 2048 -n 256 -r 3, all layers on GPU",
            "headline": "Decode is bandwidth-bound: decode tok/s * weight_GB is ~constant on the 7B quant "
                        "ladder (250*4.4=1101, 224*5.1=1144, 171*7.5=1303). The MoE decouples footprint "
                        "from decode: Qwen3-30B-A3B sits in 32B-class VRAM (19.4 GB ~ the 32B dense's 20.7) "
                        "yet decodes 286.9 tok/s vs the 32B dense's 67.2 (4.3x) — decode BW tracks ~3.3B "
                        "ACTIVE params, not the 30B total.",
        },
        "rows": rows,
        "quant_axis_bw_invariant": {  # fixed params, decode*bytes should be ~flat if BW-bound
            r["label"]: r["decode_x_gb"] for r in rows if r["label"].startswith("qwen7b")},
        "moe_decoupling": {
            "dense_32b": {"vram_mib": rows[3]["peak_vram_mib"], "decode_tok_s": rows[3]["decode_tok_s_tg256"]},
            "moe_30b_a3b": {"vram_mib": rows[4]["peak_vram_mib"], "decode_tok_s": rows[4]["decode_tok_s_tg256"]},
            "moe_decode_speedup_at_same_footprint": round(rows[4]["decode_tok_s_tg256"] /
                                                          rows[3]["decode_tok_s_tg256"], 2),
        },
        "establishes": [
            "Decode is bandwidth-bound: tok/s scales ~1/weight_bytes (quant axis, fixed params).",
            "Decode scales ~inverse with weight bytes across model size (7B vs 32B dense, Q4).",
            "MoE decouples footprint from decode: 32B-class memory, decode tracks ~3.3B active params.",
        ],
    }
    json.dump(doc, open(OUTJSON, "w"), indent=2)
    print("wrote", OUTJSON)

    # ---- plot: decode tok/s vs weight footprint ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    dense = [r for r in rows if r["kind"] == "dense"]
    moe = [r for r in rows if r["kind"] == "moe"]
    ax.scatter([r["weight_gb"] for r in dense], [r["decode_tok_s_tg256"] for r in dense],
               s=90, color="#2b6cb0", zorder=3, label="dense")
    ax.scatter([r["weight_gb"] for r in moe], [r["decode_tok_s_tg256"] for r in moe],
               s=140, color="#d69e2e", marker="D", zorder=4, label="MoE (active≈3.3B)")
    for r in rows:
        ax.annotate(r["display"].replace(" Q", "\nQ"), (r["weight_gb"], r["decode_tok_s_tg256"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)
    # 1/x bandwidth-bound reference through the 7B-Q4 point
    import numpy as np
    k = dense[0]["decode_tok_s_tg256"] * dense[0]["weight_gb"]
    xs = np.linspace(3.5, 20, 100)
    ax.plot(xs, k / xs, "--", color="#a0aec0", lw=1.2, zorder=1,
            label=f"BW-bound 1/bytes ref (k={k:.0f})")
    ax.set_xlabel("weight footprint on disk (GB)")
    ax.set_ylabel("decode throughput (tok/s, single-stream tg256)")
    ax.set_title("RTX 5090 decode is bandwidth-bound — except MoE decouples footprint from speed")
    ax.grid(True, alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUTPNG, dpi=130)
    print("wrote", OUTPNG)


if __name__ == "__main__":
    main()
