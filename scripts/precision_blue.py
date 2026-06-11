#!/usr/bin/env python3
"""
precision_blue.py — the "blue boxes over time" data model for management.

Two structured outputs, both written to data/output/precision_blue.json:

  over_time  — per workload (Vision / LLM / VLA) × horizon (2026→2033), the deploy
               mix split into THREE precision fates:
                 int           — quantized to integer, stays INT (commodity, grey)
                 fp_optional    — runs FP but COULD be INT; you keep FP because INT
                                  costs you a benefit (compute speed / accuracy)   [light blue]
                 fp_mandatory   — CANNOT be INT: heavy-tail-outlier paths, softmax/
                                  exp/normalisation at low bit, embeddings, and
                                  flow-matching / diffusion action heads           [dark blue]
               The story: the blue (FP) share GROWS over time, and the *dark* blue
               (mandatory) grows too as deploy bit-width descends below 8 bits.

  weights    — the weights-only question: do LLM/VLA weights stay INT4/INT8 forever,
               or does FP4/FP8 take over, and WHY. Carries the MEASURED RTX-5090
               INT4-vs-FP4 multipliers (decode / prefill / training) that decide it.

PROVENANCE DISCIPLINE (same as precision_migration.py):
  - the 2026 column is ANCHORED to measured composition (precision_composition.json)
    and the measured 5090 bake-offs;
  - 2028 / 2030 / 2033 columns are a DIRECTIONAL FORECAST built on the adoption
    precedent + competitive-silicon timeline in precision_migration.py. They are
    labelled as such on the chart. They are NOT measured.
  - the weights multipliers ARE measured (precision_5090_breadth / vllm_fp4_vs_int).

Run:  python3 scripts/precision_blue.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "output", "precision_blue.json")

HORIZONS = ["2026", "2028", "2030", "2033"]

# Three precision fates. Order = stacking order left→right on the bar.
# Each cell is (int_pct, fp_optional_pct, fp_mandatory_pct), summing to 100.
# 2026 = measured-anchored; 2028+ = directional forecast.
OVER_TIME = {
    "vision": {
        "display_name": "Vision  (CNN / SAM / YOLO)",
        "verdict": "all-integer engine — bounded activations, no outlier problem",
        "cells": {
            "2026": (100, 0, 0),
            "2028": (100, 0, 0),
            "2030": (100, 0, 0),
            "2033": (100, 0, 0),
        },
        "note": "The matmul/conv engine is INT throughout (INT8 → INT4 weights). "
                "Vision needs no FP path on the accelerator — it is the all-integer "
                "baseline the other workloads migrate away from.",
    },
    "llm": {
        "display_name": "LLM  (chat / reasoning)",
        "verdict": "the strong migration: INT8 → FP8 → FP4",
        "cells": {
            "2026": (90, 3, 7),
            "2028": (55, 35, 10),
            "2030": (18, 67, 15),
            "2033": (8, 70, 22),
        },
        "note": "Weights move INT4/INT8 → FP8 → FP4 as deploy bit-width descends "
                "sub-8-bit. The migrated weights are 'optional FP' (you COULD run "
                "INT4, but lose the compute-format speed-up + accuracy). The dark-blue "
                "(mandatory) share grows too: below 8 bits, INT clipping loses the "
                "heavy-tail activation outliers and FP's exponent becomes required.",
    },
    "vla": {
        "display_name": "VLA  (robotics / ADAS control)",
        "verdict": "already mixed; becomes more FP / layer-adaptive",
        "cells": {
            "2026": (65, 5, 30),
            "2028": (45, 25, 30),
            "2030": (25, 40, 35),
            "2033": (15, 47, 38),
        },
        "note": "INT VLM backbone + an FP-MANDATORY action head (flow-matching / "
                "diffusion experts break under INT, per QuantVLA). As the LLM path "
                "follows the FP8/FP4 curve, VLA becomes layer-adaptive — a different "
                "dtype per layer.",
    },
}

# ---------------------------------------------------------------------------
# The weights question — measured INT4 vs FP4 on RTX 5090 (the WHY).
# Multipliers are speed-up vs BF16, same-base Qwen3-8B (vLLM 0.22, single-stream).
#   decode  : INT4 2.35x / NVFP4 2.24x  -> TIE (both 4-bit memory, bandwidth-bound)
#   prefill : INT4 1.04x (pinned to bf16 floor) / NVFP4 3.59x  -> the split
#   train   : INT4 = not a training format / MXFP4 GEMM ~5.5x bf16
# Source: precision_5090_breadth.json, precision_5090_vllm_fp4_vs_int.json,
#         precision_5090_fp4_lifecycle (qutlass/Quartet training GEMM).
# ---------------------------------------------------------------------------
WEIGHTS = {
    "headline": "LLM/VLA weights are NOT staying INT. They are moving to FP4 — "
                "because FP4 is the only 4-bit format that also accelerates COMPUTE.",
    "trajectory": [
        {"horizon": "2026", "dominant": "INT4 / INT8 weights",
         "detail": "GGUF k-quants (Q4_K_M), AWQ INT4 — mature, ubiquitous"},
        {"horizon": "2028", "dominant": "INT4 ↔ FP8 weights",
         "detail": "FP8 (E4M3) on the prefill/compute path; DeepSeek-class native FP8"},
        {"horizon": "2030", "dominant": "FP4 / NVFP4 weights",
         "detail": "flagship LLMs ship 4-bit FLOAT; INT4 relegated to memory-only edge"},
        {"horizon": "2033", "dominant": "FP4 default",
         "detail": "INT8-only silicon is locked out of the frontier weight format"},
    ],
    # The decisive measured comparison: INT4 and FP4 are IDENTICAL on decode but
    # diverge hard everywhere compute matters. That divergence is the whole reason
    # the weight format is moving to FP4.
    "measured_multipliers_vs_bf16": {
        "axes": ["decode\n(memory-bound)", "prefill\n(compute-bound)", "training GEMM\n(compute-bound)"],
        "int4": [2.35, 1.04, 1.00],
        "fp8":  [1.55, 1.71, 1.90],
        "fp4":  [2.24, 3.59, 5.50],
        "int4_note": "INT4 dequantises to BF16 to compute → a MEMORY-only format. "
                     "Wins decode (fewer bytes to stream) but pinned to the BF16 floor "
                     "for prefill, and not a training format at all.",
        "fp8_note": "FP8 (E4M3) is the mid anchor — a real compute format, but 8-bit: "
                    "~1.6×/1.7× inference, ~1.9× training. The stepping-stone, not the "
                    "destination.",
        "fp4_note": "FP4 (NVFP4/MXFP4) runs on native Blackwell tensor cores → a "
                    "MEMORY + COMPUTE format. Same decode win as INT4, plus 3.6× prefill "
                    "and 5.5× training-GEMM. The win grows with compute intensity.",
        "source": "same-base Qwen3-8B, RTX 5090 sm_120a, vLLM 0.22 single-stream; "
                   "training GEMM from qutlass/Quartet. prefill split is architecture-"
                   "general (median 3.86× across 12 models / 6 archs).",
    },
    "why_fp_wins_below_8bit": [
        "Decode is bandwidth-bound: INT4 and FP4 stream the same 4 bits → they TIE. "
        "Bit-count is what matters there, not INT-vs-FP.",
        "Prefill & training are compute-bound: INT4 has no 4-bit integer matmul path "
        "on the GPU — it dequantises to BF16, so it inherits the BF16 compute wall. "
        "FP4 has native 4-bit tensor cores → it breaks through that wall.",
        "Accuracy below 8 bits: LLM activations have heavy-tail outliers. INT clipping "
        "loses them; FP's exponent (E4M3 / E2M1) preserves dynamic range. DeepSeek-R1: "
        "<1% loss FP8→FP4 PTQ.",
        "Silicon already bet on it: the whole Blackwell line (FP4/FP6/FP8), Qualcomm "
        "Hexagon NPU6 (FP8), TI ADAS NPUs — FP, not INT-mixed, is the roadmap.",
    ],
    "vision_caveat": "Vision weights stay INT: bounded activations, no outlier tail, "
                     "INT8 PTQ is lossless. The FP-weight migration is an LLM/VLA story.",
}

COLORS = {
    "int": "#9aa3b0",          # grey  — INT, commodity / quantized
    "fp_optional": "#4f9fd6",  # light blue — FP but could be INT (lose a benefit)
    "fp_mandatory": "#10325f", # dark blue  — FP mandatory, cannot be INT
    "bf16_ref": "#c9ccd3",
}


def main():
    doc = {
        "__meta__": {
            "description": "The 'blue boxes over time' data: how each workload's deploy "
                           "mix splits into INT / optional-FP / mandatory-FP across "
                           "2026→2033, plus the weights-format question with measured "
                           "RTX-5090 INT4-vs-FP4 multipliers.",
            "schema_version": 1,
            "methodology_version": "2026-06-10-precision-blue-v1",
            "horizons": HORIZONS,
            "fate_legend": {
                "int": "quantized to integer; stays INT (grey)",
                "fp_optional": "runs FP but COULD be INT — you keep FP because INT "
                               "costs a benefit (compute speed / accuracy) [light blue]",
                "fp_mandatory": "CANNOT be INT: outlier paths, softmax/norm at low bit, "
                                "embeddings, flow-matching action heads [dark blue]",
            },
            "provenance": "2026 column anchored to measured composition + 5090 bake-offs; "
                          "2028/2030/2033 are a directional forecast (adoption precedent + "
                          "competitive-silicon timeline in precision_migration.json). "
                          "Weight multipliers are MEASURED.",
            "colors": COLORS,
        },
        "over_time": OVER_TIME,
        "weights": WEIGHTS,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
