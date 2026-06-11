#!/usr/bin/env python3
"""
precision_migration.py — encode the 7-year precision roadmap as structured data.

Writes data/output/precision_migration.json: per workload class (Vision / LLM /
VLA), the deploy precision FORMAT at each of three horizons, the rationale, and
citations. This is the "tomorrow" companion to precision_composition.json (the
"today" measured anchor).

Central thesis (do NOT let the today-snapshot undercut it):
  INT8 holds as the deployment FLOOR through 2033, but FP support stops being
  optional. INT-capable @ 8-bit != INT-sufficient below 8-bit: as memory/BW
  pressure pushes models sub-8-bit, INT clipping loses heavy-tail activation
  outliers and FP's exponent (FP8 E4M3, NVFP4 E2M1) becomes mandatory.
    - FP8 needed by ~2028 (DeepSeek-class), FP4/NVFP4 by ~2030 (flagship).
    - Shipping INT8-only past 2028 is a competitive FEATURE GAP, not a cost win.

Source: precision-roadmap-combined.pptx (Executive Brief, May 2026). Citations
below are the slide-3 reference list.

Run:  python3 scripts/precision_migration.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "output", "precision_migration.json")

HORIZONS = [
    {"key": "2026_2028", "label": "2026–2028", "subtitle": "today → next SKU"},
    {"key": "2028_2030", "label": "2028–2030", "subtitle": "mid horizon"},
    {"key": "2030_2033", "label": "2030–2033", "subtitle": "long horizon"},
]

# precision_class drives chart color; it is the *dominant* format the workload
# deploys at in that horizon. "mixed"/"layer_adaptive" = genuinely split dtypes.
WORKLOADS = {
    "vision": {
        "display_name": "Vision (Keyhole)",
        "drivers": "YOLO-class CNNs, well-bounded activations",
        "verdict": "stays integer-friendly throughout",
        "cells": {
            "2026_2028": {"format": "INT8", "precision_class": "int8",
                          "rationale": "Mature, lossless via TRT PTQ",
                          "anchor_models": "SAM 3 · YOLOv8n-seg", "measured": True},
            "2028_2030": {"format": "INT8", "precision_class": "int8",
                          "rationale": "INT4 weights emerging; activations stay INT8",
                          "anchor_models": "+ INT4 weights"},
            "2030_2033": {"format": "INT8 + INT4", "precision_class": "int4",
                          "rationale": "Vision stays integer-friendly throughout",
                          "anchor_models": "SAM/YOLO-class stay INT"},
        },
    },
    "llm": {
        "display_name": "LLM",
        "drivers": "outlier activations break INT clipping",
        "verdict": "the strong migration: INT8 → FP8 → FP4",
        "cells": {
            "2026_2028": {"format": "INT8 / Q4_K_M", "precision_class": "int8",
                          "rationale": "GGUF mature; outliers tolerable at 8-bit",
                          "anchor_models": "Llama-3.1 · Qwen2.5 · Mistral", "measured": True},
            "2028_2030": {"format": "FP8 needed", "precision_class": "fp8",
                          "rationale": "Sub-8-bit deploy wants FP8 (DeepSeek-class)",
                          "anchor_models": "DeepSeek-V3 · Llama 4 (FP8)"},
            "2030_2033": {"format": "FP4 / NVFP4", "precision_class": "fp4",
                          "rationale": "Flagship LLMs ship at FP4; INT8-only locked out",
                          "anchor_models": "Qwen3 · DeepSeek-V4 (NVFP4)"},
        },
    },
    "vla": {
        "display_name": "VLA (robotics, ADAS)",
        "drivers": "mixed: vision INT, LLM FP, action head FP",
        "verdict": "already mixed; becomes more FP / layer-adaptive",
        "caveat": "Dual-loop / cached-intent architectures (NORA-1.5, TIDAL, π0.5) "
                  "shift VLA toward compute-bound rather than bandwidth-bound — but "
                  "their flow-matching/diffusion action heads still require FP. The "
                  "argument shifts, it doesn't collapse.",
        "cells": {
            "2026_2028": {"format": "Mixed INT8", "precision_class": "mixed",
                          "rationale": "OpenVLA-class works on INT8 today",
                          "anchor_models": "INT: BitVLA·OpenVLA  ↔  FP-head: π0.5·NORA-1.5",
                          "measured": True},
            "2028_2030": {"format": "Mixed INT8/FP8", "precision_class": "mixed_fp8",
                          "rationale": "30 Hz control loops want FP8 LLM path",
                          "anchor_models": "+ QuantVLA · TIDAL (FP attention)"},
            "2030_2033": {"format": "Layer-adaptive", "precision_class": "layer_adaptive",
                          "rationale": "Diffusion action heads + MoFQ per-layer dtypes",
                          "anchor_models": "MoFQ per-layer dtypes"},
        },
    },
}

SILICON_IMPLICATIONS = [
    {"horizon": "2028_2030", "claim": "NPU Mid needs FP8 (E4M3) inference path by 2028",
     "evidence": "Qualcomm Hexagon NPU6 already ships FP8 + BF16 (mobile + automotive)"},
    {"horizon": "2030_2033", "claim": "NPU High needs FP4 / NVFP4-class by 2030",
     "evidence": "NVIDIA Blackwell already ships it; flagship LLMs quantizing to 4-bit float"},
    {"horizon": "2028_2030", "claim": "Shipping INT8-only past 2028 is a competitive feature gap — not a silicon-area win",
     "evidence": "see competitive-silicon timeline below"},
]

# Adoption-curve precedent (why the ~2028 / ~2030 dates are not arbitrary).
ADOPTION_PRECEDENT = {
    "fp16_bf16": "Tensor Cores 2017 → ~50% adoption 2020 (3 yr) → default 2022 (5 yr)",
    "fp8": "H100 shipped 2022 → projection ~2028 default (5–6 yr)",
    "nvfp4": "hardware 2024 → projection ~2030 default",
}

COMPETITIVE_SILICON = [
    {"part": "NVIDIA Blackwell", "shipping": "2024", "year_num": 2024.5,
     "formats": "FP4/FP6/FP8/INT8/BF16", "market": "datacenter; RTX 5090"},
    {"part": "Qualcomm IQ9", "shipping": "2025", "year_num": 2025.3,
     "formats": "INT8+FP8+BF16 (Hexagon)", "market": "industrial · robotics (100 TOPS)"},
    {"part": "Qualcomm Hexagon NPU6", "shipping": "Oct 2025", "year_num": 2025.8,
     "formats": "INT8+FP8+BF16+INT2", "market": "Snapdragon X2 (PC, mobile)"},
    {"part": "TI TDA5 (C7 NPU)", "shipping": "2026 samples", "year_num": 2026.5,
     "formats": "5nm · transformer-class", "market": "ADAS L3 (400 single / 1200 chiplet)"},
]

# Visual bit-width descent: representative deploy weight bit-width per horizon and
# the 8-bit INT/FP threshold below which FP's exponent is needed for outliers.
BIT_WIDTH_DESCENT = {
    "int_fp_threshold_bits": 8,
    "per_horizon_bits": {"2026_2028": "≈8-bit", "2028_2030": "8→6-bit", "2030_2033": "≈4-bit"},
    "note": "At/above 8-bit INT holds; below 8-bit, FP (FP8 E4M3 / NVFP4 E2M1) preserves "
            "heavy-tail activation outliers that INT clipping loses.",
}

# Keyhole NPU tier action per horizon (the silicon ask, mapped onto the grid).
NPU_TIER_ACTIONS = {
    "2028_2030": "NPU Mid → add FP8 (E4M3)",
    "2030_2033": "NPU High → add FP4 (NVFP4)",
}

# Measured on the 5090 (Jun 2026) — what the FP4 test established and, honestly,
# what it did NOT. The 2030 LLM cell is thus partially MEASURED, not pure forecast.
MEASURED_VALIDATION_2026 = {
    "what": "Qwen3-8B NVFP4 vs Q4_K_M on RTX 5090 (sm_120a), llama.cpp build c30e012",
    "confirmed": [
        "NVFP4 executes NATIVELY on Blackwell today — the 2030 format runs now on consumer silicon",
        "FP-residual is real: the 'FP4 model' keeps embed+LM-head in BF16 (2.49 of 6.40 GB)",
        "decode is bandwidth-bound (FP4 decode tracks its byte footprint)",
    ],
    "not_yet": "no FP4 perf/memory win on llama.cpp (NVFP4 kernels new vs mature Q4_K_M; "
               "FP4 ~15-19% slower here). Published ~3x wins are NVIDIA vLLM/TensorRT-LLM, not measured.",
    "data": "data/output/precision_5090_fp4_vs_int.json",
}

# Why FP wins below 8 bits — the technical crux that defeats "INT8 is enough".
FP_RATIONALE = [
    "LLM activations have heavy-tail outliers — INT clipping loses signal, FP's exponent preserves it",
    "FP8 (E4M3) and NVFP4 (E2M1) carry exponent bits — much wider dynamic range than INT at same bit-width",
    "DeepSeek-R1: <1% accuracy loss FP8→FP4 PTQ on MMLU, GSM8K, AIME24, GPQA, MATH-500",
    "Even compute-bound VLAs (NORA, TIDAL) still need FP for flow-matching/diffusion action heads",
]

# Slide-3 citation list (keyed for cross-reference from cells/claims).
CITATIONS = {
    "vla_quant": [
        "I-ViT — Li & Gu, ICCV 2023, arxiv 2207.01405 (integer-only ViT)",
        "I-LLM — Hu et al., Houmo AI, arxiv 2405.17849 (integer-only low-bit LLM)",
        "IntAttention — Zhong et al., SUSTech Nov 2025, arxiv 2511.21513",
        "BitVLA — first 1-bit VLA, arxiv Mar 2026 (ternary + INT8 activations)",
        "NORA — Hung et al., declare-lab, arxiv 2504.19854 (Apr 2025)",
        "NORA-1.5 — same group, arxiv 2511.14659 (flow-matching action expert)",
        "TIDAL — arxiv 2601.14945 (Jan 2026), GR00T-N1.5 backbone, cached intent",
        "QuantVLA — CVPR'26, arxiv 2602.20309 (first DiT VLA PTQ; requires FP attention)",
        "EaqVLA — arxiv 2505.21567 (task failures under INT quant in dense scenes)",
        "OpenVLA — Kim et al., RSS 2024 (INT8 PTQ retains 97% success on Jetson Orin)",
        "π0 / π0.5 — Physical Intelligence (flow matching to 50 Hz; BF16 required)",
        "RDT-1B — Tsinghua RSS 2024; GR00T-N1/N1.5 — NVIDIA Mar 2025",
    ],
    "llm_precision": [
        "DeepSeek-V3 — arxiv 2412.19437 (first frontier LLM trained natively FP8)",
        "DeepSeek-R1 — arxiv 2501.12948 (<1% loss FP8→NVFP4 PTQ)",
        "NVIDIA NVFP4 — developer.nvidia.com Jun 2025 (E2M1, Blackwell native)",
        "Llama 4 Maverick — Meta 2025, arxiv 2504.20571 (BF16 + FP8 builds)",
        "Qwen3 — HF model card (BF16 trained; FP8 + Q4_K_M deploys)",
        "llama.cpp NVFP4 — release notes Apr 2026 (GGUF FP4: Qwen3, DeepSeek V4)",
        "'Give Me BF16 or Give Me Death?' — Kurtic et al., NAACL 2025, arxiv 2411.02355",
        "LLM.int8() — Dettmers et al., NeurIPS 2022 (outlier-aware INT8)",
        "MoFQ / FGMP — mixed-format & fine-grained mixed precision (per-layer dtype)",
    ],
    "silicon": [
        "Qualcomm IQ9 — Dragonwing Q-series (100 TOPS, FP8+BF16+INT8, SIL3)",
        "TI TDA5/TDA54-Q1 — CES 2026 (400 TOPS → 1200 via chiplets, 5nm C7 NPU, UCIe)",
        "Qualcomm Hexagon NPU6 — Snapdragon X2 Elite Oct 2025 (FP8+BF16+INT2 matrix unit)",
        "NVIDIA Blackwell — whitepaper 2024 (FP4/FP6/FP8/INT8/BF16; B100/B200 + RTX 5090)",
        "UCIe 2.0 — uciexpress.org (die-to-die interconnect for chiplets)",
    ],
}


def main():
    doc = {
        "__meta__": {
            "description": "7-year edge-AI precision roadmap (2026→2033) as structured "
                           "data: per-workload deploy format by horizon, with citations. "
                           "Companion to precision_composition.json (the measured 'today' anchor).",
            "schema_version": 1,
            "methodology_version": "2026-06-01-precision-migration-v1",
            "source_deck": "precision-roadmap-combined.pptx (Executive Brief, May 2026)",
            "thesis": "INT8 holds as the deployment floor through 2033, but FP support "
                      "stops being optional: FP8 by ~2028, FP4/NVFP4 by ~2030. INT-capable "
                      "@8-bit != INT-sufficient below 8-bit. INT8-only past 2028 = feature gap.",
            "horizons": HORIZONS,
            "precision_class_colors": {
                "int8": "INT8", "int4": "INT8+INT4", "fp8": "FP8",
                "fp4": "FP4/NVFP4", "mixed": "mixed INT", "mixed_fp8": "mixed INT/FP8",
                "layer_adaptive": "layer-adaptive",
            },
        },
        "workloads": WORKLOADS,
        "silicon_implications": SILICON_IMPLICATIONS,
        "npu_tier_actions": NPU_TIER_ACTIONS,
        "measured_validation_2026": MEASURED_VALIDATION_2026,
        "bit_width_descent": BIT_WIDTH_DESCENT,
        "adoption_precedent": ADOPTION_PRECEDENT,
        "competitive_silicon": COMPETITIVE_SILICON,
        "why_fp_below_8bit": FP_RATIONALE,
        "citations": CITATIONS,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
