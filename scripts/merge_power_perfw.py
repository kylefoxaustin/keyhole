#!/usr/bin/env python3
"""
merge_power_perfw.py — fold measured Orin power into the three-platform matrix and
answer the perf/W question the corpus has been carrying as INCONCLUSIVE.

WHY THIS IS NOT A ONE-LINER
---------------------------
The two platforms do not expose the same KIND of watt, and pretending they do is
exactly the denominator error the fleet has hit 13 times out of 13 — always in the
direction that flatters the accelerator.

  RTX 5090   nvidia-smi power.draw = WHOLE BOARD: GPU die + GDDR7 + VRM losses.
             Idle 64.97 W. Models 176-554 W. (Nameplate is 575 W — measuring it is
             what killed the nameplate perf/W claim in the first place.)
  Jetson Orin tegrastats rails = SoC DIE ONLY:
             VDD_GPU_SOC (GPU+SoC) + VDD_CPU_CV (CPU+CV) = `soc_w`.
             This EXCLUDES the LPDDR5 DRAM and the carrier board.
             VIN_SYS_5V0 is the 5V system/carrier input, measured separately.

So `soc_w` vs board-W is NOT apples-to-apples: it omits, on the Orin side, the very
component (DRAM) that the 5090's number includes. Left unstated, that asymmetry is a
free gift to the edge part.

THE FIX: don't pick a convention and hope. Compute perf/W under THREE defensible ones
and report whether the WINNER IS INVARIANT across them.

  A  as-measured      Orin soc_w                 vs  5090 board W
                      (favours Orin: excludes its DRAM; includes the 5090's)
  B  incl. carrier    Orin soc_w + VIN_SYS_5V0   vs  5090 board W
                      (favours 5090: adds Orin peripherals the 5090 number has no analog for)
  C  delta over idle  Orin soc_w - idle          vs  5090 board W - 65 W idle
                      ("marginal watts to do the work" — the fairest single number,
                       and the one least sensitive to what else sits on the rail)

A and B BRACKET the truth: the real Orin device power sits between them. If the same
platform wins under A, B and C, the conclusion is robust to the ambiguity and we can
finally retire "perf/W INCONCLUSIVE". If the winner flips, perf/W STAYS inconclusive —
but quantified, with the flip point named, instead of hand-waved.

perf/W here = inferences per second per watt = (1000 / compute_p50_ms) / W.
Both sides use compute_p50_ms (pure GPU kernel time), the same cross-platform axis the
FP16 matrix already uses.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"

# maps our canonical model name -> the key used in rtx5090_power_by_model.json
POWER_KEY = {
    "resnet50": "resnet50v1",
    "clip_vit_b32": "clip_vit_b32_visual",
    "yolov8n_seg": "yolov8n-seg",
    "yolo11s_seg": "yolo11s-seg",
    "yoloe_26s_seg": "yoloe-26s-seg-pf",
    "efficientsam_encoder": "efficient_sam_vitt_encoder",
    "efficientsam_decoder": "efficient_sam_vitt_decoder",
}

RTX_IDLE_W = 21.73   # clean card, measured 2026-07-12. The old 64.97 W was a DIRTY-CARD idle.


def ips(ms):
    return 1000.0 / ms if ms else None


# matrix is keyed by ONNX basename; the power pass uses canonical short names.
MATRIX_KEY = {
    "resnet50": "resnet50v1",
    "clip_vit_b32": "clip_vit_b32_visual",
    "yolov8n_seg": "yolov8n-seg",
    "yolo11s_seg": "yolo11s-seg",
    "yoloe_26s_seg": "yoloe-26s-seg-pf",
    "efficientsam_encoder": "efficient_sam_vitt_encoder",
}


def main():
    mpath = OUT / "vision_corpus_three_platform.json"
    matrix = json.loads(mpath.read_text())
    orin_pwr = json.loads((OUT / "orin_power.json").read_text())
    rtx_pwr = json.loads((OUT / "rtx5090_power_by_model_v2.json").read_text())

    orin_idle = orin_pwr["idle"]["soc_mean_w"]
    orin_by_model = {r["model"]: r for r in orin_pwr["results"]}

    rows, invalid = [], []
    for short, o in orin_by_model.items():
        if not o.get("valid"):
            invalid.append({"model": short, "reason": o.get("INVALID_REASON")})
            continue
        key = MATRIX_KEY[short]
        cell = matrix["matrix"][key]["fp16"]
        rtx, orin = cell["rtx-5090"], cell["orin-agx-64gb"]

        rtx_ms, orin_ms = rtx["compute_p50_ms"], orin["compute_p50_ms"]
        rtx_w = rtx_pwr["models"][key]["power_w_median"]
        rtx["power_w_measured"] = rtx_w
        rtx["power_w_measured_runs"] = rtx_pwr["models"][key]["runs_w"]
        rtx["power_w_spread_pct"] = rtx_pwr["models"][key]["spread_pct"]
        rtx["power_w_PREVIOUS_dirty_card"] = rtx_pwr["models"][key]["old_dirty_w"]

        soc = o["power"]["soc_mean_w"]
        vin = o["power"]["vin_sys_5v0_mean_mw"] / 1000.0
        delta = o["soc_delta_over_idle_w"]

        # --- write the MEASURED power into the Orin cell, replacing the nameplate ---
        orin.pop("power_w_nameplate_ceiling", None)
        orin.pop("inferences_per_joule_lower_bound", None)
        orin["power_w_measured_soc"] = round(soc, 2)
        orin["power_w_measured_soc_delta_over_idle"] = delta
        orin["power_w_vin_sys_5v0"] = round(vin, 2)
        orin["power_w_nameplate_for_reference_only"] = 60.0
        orin["inferences_per_joule"] = round(ips(orin_ms) / soc, 2)
        orin["perf_per_watt_basis"] = (
            "MEASURED tegrastats under sustained load, MAXN. soc = VDD_GPU_SOC + VDD_CPU_CV "
            "(die rails; EXCLUDES LPDDR5 + carrier). VIN_SYS_5V0 recorded separately."
        )
        orin["compute_p50_ms_repro_this_pass"] = o.get("compute_p50_ms_trtexec")

        o_ips, r_ips = ips(orin_ms), ips(rtx_ms)
        conv = {}
        for tag, ow, rw in (
            ("A_as_measured", soc, rtx_w),
            ("B_orin_incl_carrier", soc + vin, rtx_w),
            ("C_delta_over_idle", delta, rtx_w - RTX_IDLE_W),
        ):
            op, rp = o_ips / ow, r_ips / rw
            conv[tag] = {
                "orin_w": round(ow, 2), "rtx5090_w": round(rw, 2),
                "orin_inf_per_joule": round(op, 2),
                "rtx5090_inf_per_joule": round(rp, 2),
                "orin_advantage_x": round(op / rp, 2),
                "winner": "orin-agx" if op > rp else "rtx-5090",
            }
        winners = {c["winner"] for c in conv.values()}
        rows.append({
            "model": key,
            "orin_compute_p50_ms": orin_ms, "rtx5090_compute_p50_ms": rtx_ms,
            "orin_soc_w": round(soc, 2), "rtx5090_board_w": rtx_w,
            "conventions": conv,
            "winner_invariant": len(winners) == 1,
            "winner": list(winners)[0] if len(winners) == 1 else "FLIPS_BY_CONVENTION",
        })

    all_inv = bool(rows) and all(r["winner_invariant"] for r in rows)
    overall = {r["winner"] for r in rows}

    matrix["perf_per_watt"] = {
        "status": ("RESOLVED — winner is invariant across all three power conventions"
                   if all_inv and len(overall) == 1 else
                   "SEE PER-MODEL — winner is not uniform across models and/or conventions"),
        "orin_idle_soc_w": orin_idle,
        "rtx5090_idle_board_w": RTX_IDLE_W,
        "orin_power_mode": orin_pwr["__meta__"]["power_mode"],
        "rail_convention": orin_pwr["__meta__"]["rail_convention"],
        "why_three_conventions": (
            "The 5090's nvidia-smi figure is WHOLE-BOARD (die + GDDR7 + VRM). The Orin's "
            "tegrastats SoC rails are DIE-ONLY: they exclude the LPDDR5 and the carrier. "
            "That asymmetry silently FLATTERS THE EDGE PART, which is the fleet's 16-for-16 "
            "failure direction. So we bracket rather than pick: A (as-measured) favours the "
            "Orin, B (incl. carrier) favours the 5090, C (marginal watts over idle) is the "
            "fairest single number. A conclusion is reported only if it survives all three."
        ),
        "supersedes": (
            "RETIRES the nameplate claim '5090 575 W vs edge ~15-60 W -> perf/W is the edge "
            "story'. It was nameplate on BOTH sides. Measured: the 5090 draws 176-554 W (not "
            "575) and the Orin draws 22.4-40.6 W (not 60)."
        ),
        "per_model": rows,
    }
    if invalid:
        matrix["perf_per_watt"]["invalid_rows_not_quoted"] = invalid

    mpath.write_text(json.dumps(matrix, indent=2))

    print(f"Orin idle {orin_idle} W (SoC rails)   5090 idle {RTX_IDLE_W} W (board)\n")
    h = f"{'model':24s} {'orin ms':>8s} {'5090 ms':>8s} {'orinW':>6s} {'5090W':>6s} | " \
        f"{'A':>14s} {'B':>14s} {'C':>14s} | invariant"
    print(h); print("-" * len(h))
    for r in rows:
        c = r["conventions"]
        f = lambda t: f"{c[t]['winner'][:4]} {c[t]['orin_advantage_x']:>5.2f}x"
        print(f"{r['model']:24s} {r['orin_compute_p50_ms']:>8.3f} {r['rtx5090_compute_p50_ms']:>8.3f} "
              f"{r['orin_soc_w']:>6.1f} {r['rtx5090_board_w']:>6.1f} | "
              f"{f('A_as_measured'):>14s} {f('B_orin_incl_carrier'):>14s} "
              f"{f('C_delta_over_idle'):>14s} | {'YES' if r['winner_invariant'] else 'FLIPS'}")
    print("\nx = Orin inferences/joule ÷ 5090 inferences/joule  (>1 = Orin more efficient)")
    print(f"\nSTATUS: {matrix['perf_per_watt']['status']}")


if __name__ == "__main__":
    main()
