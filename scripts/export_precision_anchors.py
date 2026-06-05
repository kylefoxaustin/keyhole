#!/usr/bin/env python3
"""
export_precision_anchors.py — export the INT4-vs-FP4 (+FP8/BF16) inference measurements as a
consumable ANCHOR BUNDLE for keyhole-sizer to ground its FP4/precision projection.

keyhole backend is the DATA SOURCE; the sizer CONSUMES via anchors (the documented boundary).
This emits the measured truth in the sizer's own shape so it drops in:
  - `measured_llm_cells`: {model_key: {quant: {decode_tok_s, prefill_tok_s}}} — same shape as
    sizer/measured.py's measured_llm; add nvfp4 / int4_awq / fp8 / bf16 cells alongside the GGUF ones.
  - `precision_axis_5090`: dtype multipliers vs BF16 (from the size-controlled Qwen3-8B quad) +
    the sm120 capability story (nvfp4 = native FP4 tensor cores; int4 = memory-format, dequant-to-bf16).
  - `int4_vs_nvfp4_breadth`: the architecture-generality evidence (12 pairs / 6 archs): decode ties,
    prefill split range/median, size trend, hybrid-Mamba caveat — the confidence/"why" panel.

Reads data/output/precision_5090_vllm_runs/*.json. Writes data/output/precision_anchors_5090.json.
"""
import json, os, statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "data/output/precision_5090_vllm_runs")
OUT = os.path.join(REPO, "data/output/precision_anchors_5090.json")

# model_key | arch | params_b | {quant: run_label}  — quant keys match the sizer's vLLM precision axis.
MODELS = {
    "qwen3-8b":      ("Qwen3", 8,  {"bf16": "bf16", "fp8": "fp8", "int4_awq": "qwen3_8b_awq_int4", "nvfp4": "nvfp4"}),
    "qwen3-14b":     ("Qwen3", 14, {"int4_awq": "qwen3_14b_awq_int4", "nvfp4": "qwen3_14b_nvfp4"}),
    "qwen3-32b":     ("Qwen3", 32, {"int4_awq": "qwen3_32b_awq_int4", "nvfp4": "qwen3_32b_nvfp4"}),
    "llama-3.1-8b":  ("Llama", 8,  {"int4_awq": "llama8b_awq_int4", "nvfp4": "llama8b_nvfp4"}),
    "llama-3.2-1b":  ("Llama", 1,  {"int4_awq": "llama32_1b_int4", "nvfp4": "llama32_1b_nvfp4"}),
    "mistral-7b-v0.3": ("Mistral", 7, {"int4_awq": "mistral7b_v03_int4", "nvfp4": "mistral7b_v03_nvfp4"}),
    "ds-r1-distill-llama-8b": ("Llama", 8, {"int4_awq": "dsllama8b_int4", "nvfp4": "dsllama8b_nvfp4"}),
    "ds-r1-distill-qwen-14b": ("Qwen2", 14, {"int4_awq": "dsqwen14b_int4", "nvfp4": "dsqwen14b_nvfp4"}),
    "ds-r1-distill-qwen-32b": ("Qwen2", 32, {"int4_awq": "dsqwen32b_int4", "nvfp4": "dsqwen32b_nvfp4"}),
    "phi-4-reasoning-plus": ("Phi-4", 15, {"int4_awq": "phi4rp_int4", "nvfp4": "phi4rp_nvfp4"}),
    "mistral-small-24b": ("Mistral", 24, {"int4_awq": "mistral24b_int4", "nvfp4": "mistral24b_nvfp4"}),
    "nemotron-nano-9b-v2": ("Nemotron-H", 9, {"int4_awq": "nemotron_9b_int4", "nvfp4": "nemotron_9b_nvfp4"}),
}


def cell(label):
    p = os.path.join(RUNS, f"{label}_vllm.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {"decode_tok_s": round(list(d["decode_tok_s"].values())[0], 1),
            "prefill_tok_s": round(max(d["prefill_tok_s"].values()), 0)}


def main():
    cells, pairs = {}, []
    for mk, (arch, params, quants) in MODELS.items():
        got = {}
        for q, lab in quants.items():
            c = cell(lab)
            if c:
                got[q] = c
        if got:
            cells[mk] = {"arch": arch, "params_b": params, **got}
        # same-base INT4-vs-NVFP4 pair stats (the asymmetry)
        if "int4_awq" in got and "nvfp4" in got:
            i, n = got["int4_awq"], got["nvfp4"]
            pairs.append({"model": mk, "arch": arch, "params_b": params,
                          "decode_ratio_nvfp4_over_int4": round(n["decode_tok_s"] / i["decode_tok_s"], 2),
                          "prefill_split_nvfp4_over_int4": round(n["prefill_tok_s"] / i["prefill_tok_s"], 2)})

    # precision multipliers vs BF16 — from the size-controlled Qwen3-8B quad (the clean anchor)
    q = cells["qwen3-8b"]
    bf = q["bf16"]
    mult = {axis: {dt: round(q[dt][f"{axis}_tok_s"] / bf[f"{axis}_tok_s"], 2)
                   for dt in ("bf16", "fp8", "int4_awq", "nvfp4") if dt in q}
            for axis in ("decode", "prefill")}

    splits = [p["prefill_split_nvfp4_over_int4"] for p in pairs]
    decs = [p["decode_ratio_nvfp4_over_int4"] for p in pairs]
    pairs.sort(key=lambda p: p["params_b"])

    doc = {
        "__meta__": {
            "description": "INT4-vs-FP4 (+FP8/BF16) inference measurements on RTX 5090 (Blackwell sm_120), "
                           "vLLM single-stream, as a consumable anchor for keyhole-sizer's precision axis.",
            "source": "keyhole backend (data source); see scripts/bench_precision_vllm.py + breadth sweeps",
            "device": "NVIDIA GeForce RTX 5090", "runtime": "vLLM 0.22, single-stream (batch=1, greedy)",
            "schema_version": 1,
            "consume_note": "measured_llm_cells matches sizer/measured.py measured_llm shape "
                            "{model: {quant: {decode_tok_s, prefill_tok_s}}}. Add nvfp4 to the sizer's "
                            "precision capability taxonomy: on sm_120 NVFP4=tensor_native (native FP4 cores), "
                            "INT4(AWQ/GPTQ)=memory-format (dequantizes to bf16 -> prefill on the bf16 floor).",
        },
        "precision_axis_5090": {
            "anchor_model": "qwen3-8b (size-controlled quad)",
            "multiplier_vs_bf16": mult,  # e.g. decode: nvfp4 2.24, int4 2.35; prefill: nvfp4 3.59, int4 1.04
            "capability_sm120": {
                "nvfp4": "tensor_native (native FP4 tensor cores) — memory + compute format",
                "int4_awq": "memory-only — wins decode (BW) but prefill on the bf16 floor (dequant-to-bf16)",
                "fp8": "tensor_native (native fp8 cores) — mid anchor ~1.6x decode / ~1.7x prefill",
            },
        },
        "int4_vs_nvfp4_breadth": {
            "n_pairs": len(pairs), "n_archs": len(set(p["arch"] for p in pairs)),
            "decode_ratio_nvfp4_over_int4": {"median": round(statistics.median(decs), 2),
                                             "range": [min(decs), max(decs)]},
            "prefill_split_nvfp4_over_int4": {"median": round(statistics.median(splits), 2),
                                              "range": [min(splits), max(splits)]},
            "size_trend": "split tracks compute-intensity: ~2.7x (1B) -> ~3.5x (8B) -> ~4.5x (24-32B plateau)",
            "caveats": [
                "decode ~ties (NVFP4/INT4 0.81-1.01): both BW-bound by ~4-bit weight bytes.",
                "INT4 prefill ~= bf16 floor (memory format, no FP4 compute path).",
                "hybrid Mamba (Nemotron-Nano-9B) splits less (~2.8x): its NVFP4 keeps Mamba layers BF16 "
                "(partial FP4) — the win attenuates with FP4 coverage.",
            ],
            "per_pair": pairs,
        },
        "measured_llm_cells": cells,
    }
    json.dump(doc, open(OUT, "w"), indent=2)
    print("wrote", OUT)
    print(f"  cells for {len(cells)} models; {len(pairs)} INT4-vs-NVFP4 pairs / "
          f"{doc['int4_vs_nvfp4_breadth']['n_archs']} archs")
    print(f"  precision multipliers vs bf16: decode {mult['decode']}  prefill {mult['prefill']}")


if __name__ == "__main__":
    main()
