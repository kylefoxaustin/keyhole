#!/usr/bin/env python3
"""
skippy_edge_projection.py — project the measured PixelRAG reader (at the 768px knee)
from the RTX 5090 onto the keyhole EDGE_MPU_TARGET (NXP-class NPU), using keyhole's own
HardwareSpec. Prefill is compute-bound → scales by the effective-TOPS ratio; decode is
bandwidth-bound → scales by the effective-bandwidth ratio. The point: edge has ~same
compute but ~15x less bandwidth, so decode explodes and FP4 becomes the enabler (both a
decode speedup AND the only way the reader weights fit in 8GB).

Reads data/output/skippy_precision_measured.json (5090 reader, aggregate). Writes
data/output/skippy_edge_projection.json + a chart. Aggregate only; no personal content.
"""
import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from emulate.npu_emulator import RTX_5090, EDGE_MPU_TARGET

m = json.load(open(REPO / "data" / "output" / "skippy_precision_measured.json"))

def eff_tops(hw, dt):
    return getattr(hw, f"tops_{dt}") * hw.compute_efficiency
def eff_bw(hw):
    return hw.mem_bandwidth_gbs * hw.bandwidth_efficiency

# compute ratio ~ same across dtypes here (peak TOPS ~ matched); decode = BW ratio
COMPUTE_R = eff_tops(RTX_5090, "int8") / eff_tops(EDGE_MPU_TARGET, "int8")   # prefill
BW_R = eff_bw(RTX_5090) / eff_bw(EDGE_MPU_TARGET)                            # decode

# reader weight footprint (4B): bf16 ~8GB, nvfp4 ~2GB (4-bit + fp residual)
FOOT = {"bf16": 8.0, "nvfp4": 2.2}
EDGE_CAP = EDGE_MPU_TARGET.mem_capacity_gb

rows = {}
for dt in ("bf16", "nvfp4"):
    p5, d5 = m[dt]["ttft_ms"], m[dt]["decode_ms_per_tok"]
    rows[dt] = {
        "5090_prefill_ms": p5, "5090_decode_ms_per_tok": d5,
        "edge_prefill_ms": round(p5 * COMPUTE_R, 1),
        "edge_decode_ms_per_tok": round(d5 * BW_R, 1),
        "weights_gb": FOOT[dt], "fits_8gb_with_kv": FOOT[dt] < EDGE_CAP - 1.0,
    }

out = {
    "__meta__": {
        "reader": "Qwen3-VL-4B @768px knee", "source_5090": "skippy_precision_measured.json",
        "edge": EDGE_MPU_TARGET.name,
        "compute_ratio_5090_over_edge": round(COMPUTE_R, 2),
        "bandwidth_ratio_5090_over_edge": round(BW_R, 2),
        "5090_eff_int8_tops": round(eff_tops(RTX_5090, "int8")), "edge_eff_int8_tops": round(eff_tops(EDGE_MPU_TARGET, "int8")),
        "5090_eff_bw_gbs": round(eff_bw(RTX_5090)), "edge_eff_bw_gbs": round(eff_bw(EDGE_MPU_TARGET)),
        "thesis": "edge ≈ same compute (prefill scales ~%.1fx) but ~%.0fx less bandwidth "
                  "(decode scales ~%.0fx) → decode dominates on edge; FP4 is the enabler "
                  "(decode speedup + the only fit in 8GB)." % (COMPUTE_R, BW_R, BW_R),
    },
    "projection": rows,
}
dst = REPO / "data" / "output" / "skippy_edge_projection.json"
json.dump(out, open(dst, "w"), indent=2)
print(json.dumps(out, indent=2)); print("wrote", dst)

# ---- chart: 5090 vs edge decode/prefill, bf16 vs nvfp4 ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.4))
fig.suptitle("PixelRAG reader projected to the edge NPU: bandwidth-starved → FP4 is the enabler",
             fontsize=13, fontweight="bold", y=0.98)
labels = ["5090\nBF16", "5090\nNVFP4", "edge\nBF16", "edge\nNVFP4"]
pre = [rows["bf16"]["5090_prefill_ms"], rows["nvfp4"]["5090_prefill_ms"],
       rows["bf16"]["edge_prefill_ms"], rows["nvfp4"]["edge_prefill_ms"]]
dec = [rows["bf16"]["5090_decode_ms_per_tok"], rows["nvfp4"]["5090_decode_ms_per_tok"],
       rows["bf16"]["edge_decode_ms_per_tok"], rows["nvfp4"]["edge_decode_ms_per_tok"]]
cols = ["#9aa3b0", "#10325f", "#c9a14f", "#1f6f3f"]
for ax, vals, t in [(a1, pre, "Prefill / TTFT (ms) — compute-bound, ~%.1fx" % COMPUTE_R),
                    (a2, dec, "Decode (ms/tok) — bandwidth-bound, ~%.0fx" % BW_R)]:
    b = ax.bar(labels, vals, color=cols, edgecolor="white")
    for bb in b: ax.text(bb.get_x()+bb.get_width()/2, bb.get_height(), f"{bb.get_height():.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(t, fontsize=10.5); ax.set_ylim(0, max(vals)*1.2)
    for s in ("top","right"): ax.spines[s].set_visible(False)
a1.text(0.5,-0.18,"edge BF16 (8GB weights) does NOT fit 8GB DRAM with KV — NVFP4 (~2GB) does",
        transform=a1.transAxes, ha="center", fontsize=8.3, color="#c1121f", style="italic")
fig.text(0.5,0.015,"Projected from measured 5090 via keyhole HardwareSpec (EDGE_MPU_TARGET: ~400 INT8 TOPS, 134 GB/s LPDDR5X, 8GB). Aggregate only.",
         ha="center", fontsize=7.6, color="#888", style="italic")
fig.tight_layout(rect=[0,0.07,1,0.94])
png = REPO / "data" / "output" / "skippy_edge_projection.png"
fig.savefig(png, dpi=150); print("wrote", png)
