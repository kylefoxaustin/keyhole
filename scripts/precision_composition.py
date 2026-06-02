#!/usr/bin/env python3
"""
precision_composition.py — INT vs FP composition of the keyhole model catalog.

Produces data/output/precision_composition.json: for every model in the
catalog (vision / LLM / VLA), how the model decomposes into INT-capable vs
floating-point work along THREE axes:

  by_params  — fraction of weights that live in quantizable matmul (linear/conv/
               attention-projection/FFN) tensors vs FP-only tensors (norm,
               embedding) and FP-MANDATORY heads (flow-matching action experts).
  by_flops   — fraction of compute (FLOPs) in INT-capable GEMMs vs the FP-residual
               tail (softmax, norm, activation, RoPE, residual). Analytic, from
               published architecture hyperparameters at a reference seq length.
  by_bytes   — fraction of stored weight bytes that are low-precision vs the
               high-precision residual a real quant keeps (GGUF effective
               bits/weight for LLMs; bf16/int8/int4 DRAM projections for VLAs).

PROVENANCE is attached to every number:
  measured            — from a 5090 measurement / real artifact (GGUF size, DRAM)
  reference_arch       — from a paper-spec architecture dump (sam3_reference_*)
  arch_analytic        — computed here from published hyperparameters + std FLOP model
  documented_constraint— a precision requirement stated in the model spec notes
                         (e.g. flow-matching head requires FP per QuantVLA)

WHY the three axes disagree (this is the point, not a bug):
  - by_params  : nearly every model is ~all-INT (weights are almost all matmul).
  - by_flops   : still mostly INT — GEMMs dominate FLOP counts; the FP tail
                 (softmax/norm/act) is a few % of FLOPs.
  - The FP tail only DOMINATES LATENCY because it is memory-/launch-bound, not
    FLOP-heavy (cf. pi0.5 denoise at 1.6% BW util). FLOP share understates the
    silicon consequence; the dtype_path constraint + bytes view carry that story.

NOT MEASURED ANYWHERE: a per-kernel "this op ran INT8 / that one ran FP8" trace.
NCU labels dtype only in the NVTX range name. So every by_flops split here is
ANALYTIC (arch_analytic), clearly flagged — not a fabricated runtime measurement.

Run:  python3 scripts/precision_composition.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "output", "precision_composition.json")

# Hardware tiers and the dtypes they can execute (from keyhole_data_bundle meta).
# This is what the composition FEEDS: a model whose FP-required fraction is 0 can
# live on an INT8-only tier; any FP-mandatory component forces an FP-capable tier.
TIER_DTYPE_SUPPORT = {
    "NPU Low / Mid (INT8-only)": ["INT8"],
    "NPU High": ["INT8", "FP16", "BF16", "FP8"],
    "RTX 5090": ["INT8", "FP16", "BF16", "FP8", "FP32"],
}

REF_SEQ = 2048  # reference prefill length; matches the prefill_tok_s_at_2k anchors


# ---------------------------------------------------------------------------
# Analytic transformer FLOP decomposition
# ---------------------------------------------------------------------------
def transformer_flops(L, D, H, KV, F, V, S, gated_ffn=True):
    """Decoder-transformer FLOP decomposition for a prefill of S tokens.

    Returns (int_capable_flops, fp_tail_breakdown) where int_capable is all the
    GEMM work (quantizable on INT8/INT4 silicon) and the tail is the FP-residual
    ops that have no efficient integer form in practice.

    Standard 2*MACs accounting. Coefficients on the tail ops are deliberately
    GENEROUS (over- not under-count the FP tail) so the INT share is a floor.
    """
    Dkv = D * KV // H  # GQA: K/V projection width
    per_layer_matmul = (
        2 * S * D * (D + 2 * Dkv)   # QKV projection
        + 2 * S * S * D             # QK^T scores  (sum over all heads = D)
        + 2 * S * S * D             # scores @ V
        + 2 * S * D * D             # output projection
        + 2 * S * (3 if gated_ffn else 2) * D * F  # FFN (SwiGLU = 3 matmuls)
    )
    # FP-residual tail per layer (generous coefficients):
    softmax = 5 * S * S * H         # exp + normalize over the score matrix
    norms = 2 * (8 * S * D)         # 2 norms/layer, ~8 flops/elem (RMS: sq,sum,rsqrt,mul,scale)
    activation = 8 * S * F          # SiLU + gate multiply
    rope = 6 * S * D                # rotate Q and K
    residual = 2 * S * D            # 2 residual adds
    per_layer_tail = softmax + norms + activation + rope + residual

    int_capable = L * per_layer_matmul + 2 * S * D * V  # + LM head GEMM (D->V)
    tail = {
        "softmax": L * softmax,
        "norm": L * norms,
        "activation": L * activation,
        "rope": L * rope,
        "residual": L * residual,
    }
    return int_capable, tail


def transformer_params(L, D, H, KV, F, V, gated_ffn=True, tie_embed=False):
    """Parameter decomposition: matmul weights (quantizable) vs norm + embedding."""
    Dkv = D * KV // H
    per_layer_matmul = (
        D * (D + 2 * Dkv)           # QKV
        + D * D                     # O proj
        + (3 if gated_ffn else 2) * D * F  # FFN
    )
    matmul = L * per_layer_matmul
    lm_head = 0 if tie_embed else D * V     # output projection (quantizable matmul)
    embedding = D * V                       # token embedding table (FP-kept by convention)
    norm = L * 2 * D + D                    # 2 RMSNorm/layer + final norm
    return {"matmul": matmul + lm_head, "embedding": embedding, "norm": norm}


def pct(part, whole):
    return round(100.0 * part / whole, 3) if whole else 0.0


# ---------------------------------------------------------------------------
# Architecture registry (published hyperparameters)
# ---------------------------------------------------------------------------
# L=layers D=d_model H=heads KV=kv_heads F=d_ff V=vocab
LLM_ARCH = {
    "llama_3_1_8b_dense":  dict(L=32, D=4096, H=32, KV=8, F=14336, V=128256),
    "mistral_7b_v03_dense": dict(L=32, D=4096, H=32, KV=8, F=14336, V=32768),
    "qwen_2_5_7b_dense":   dict(L=28, D=3584, H=28, KV=4, F=18944, V=152064),
    "qwen_2_5_32b_dense":  dict(L=64, D=5120, H=40, KV=8, F=27648, V=152064),
}
# Nominal bits/weight for the GGUF k-quants (the body weight precision).
QUANT_NOMINAL_BITS = {"Q4_K_M": 4.0, "Q5_K_M": 5.0, "Q8_0": 8.0}


def build_llm(model_key, anchor):
    out = {
        "family": "llm",
        "model_id": anchor["model_id"],
        "n_params_b": anchor["n_params_total_b"],
        "compute_dtype": anchor["compute_dtype"],
    }
    arch = LLM_ARCH.get(model_key)

    # --- by_params (arch_analytic) ---
    if arch:
        p = transformer_params(**arch)
        total = sum(p.values())
        out["by_params"] = {
            "provenance": "arch_analytic",
            "int_capable_pct": pct(p["matmul"] + p["embedding"], total),
            "fp_required_pct": pct(p["norm"], total),
            "breakdown_pct": {
                "matmul_weights": pct(p["matmul"], total),
                "embedding": pct(p["embedding"], total),
                "norm": pct(p["norm"], total),
            },
            "note": "embeddings treated as quantizable (GGUF quantizes token_embd); only norms kept FP",
        }
        # --- by_flops (arch_analytic) ---
        ic, tail = transformer_flops(S=REF_SEQ, **arch)
        tail_total = sum(tail.values())
        total_f = ic + tail_total
        out["by_flops"] = {
            "provenance": "arch_analytic",
            "seq_len": REF_SEQ,
            "int_capable_pct": pct(ic, total_f),
            "fp_tail_pct": pct(tail_total, total_f),
            "fp_tail_breakdown_pct": {k: pct(v, total_f) for k, v in tail.items()},
            "note": "GEMMs dominate FLOPs; FP tail is small in FLOP terms but is the "
                    "memory-/launch-bound part that dominates LATENCY.",
        }

    # --- by_bytes (measured: GGUF effective bits/weight per quant) ---
    quants = {}
    for qname, q in anchor["quants"].items():
        gb = q["gguf_size_gb"]
        eff_bits = round(gb * 8 * 1e9 / (anchor["n_params_total_b"] * 1e9), 3)
        nominal = QUANT_NOMINAL_BITS.get(qname)
        # Residual = bytes beyond the nominal low-precision body (norms kept fp +
        # k-quant scales/mins). Expressed as a fraction of the stored bytes.
        residual_pct = pct(eff_bits - nominal, eff_bits) if nominal else None
        quants[qname] = {
            "gguf_size_gb": round(gb, 3),
            "effective_bits_per_weight": eff_bits,
            "nominal_bits": nominal,
            "high_precision_residual_pct": residual_pct,
        }
    out["by_bytes"] = {
        "provenance": "measured",
        "note": "weights stored at the quant's effective bits/weight; KV-cache + "
                "activations run fp16 (context-dependent, not in weight bytes).",
        "quants": quants,
    }

    out["deployable_dtype"] = "INT4/INT8 weights + fp16 compute"
    out["npu_tier_fit"] = "INT8-only OK for weights; fp16 activation path needed (norms/softmax)"
    return out


# ---------------------------------------------------------------------------
# VLA catalog (model-level specs + documented precision constraints)
# ---------------------------------------------------------------------------
VLA_FILES = {
    "nora_3b": "vla_summary_nora_3b.json",
    "openvla_7b_single": "vla_summary_openvla_7b_single.json",
    "nora_1p5": "vla_summary_nora_1p5.json",
    "pi_0p5": "vla_summary_pi_0p5.json",
    "bitvla": "vla_summary_bitvla.json",
}


def build_vla(spec, dtype_label):
    total = spec["total_params_b"]
    vlm = spec.get("vlm_params_b", 0.0)
    action_m = spec.get("action_params_m", 0.0) or 0.0
    action_b = action_m / 1000.0
    path = spec["dtype_path_default"]

    # Does this model have an FP-MANDATORY component? Flow-matching / diffusion
    # action heads break under INT8 (QuantVLA) -> documented_constraint.
    fp_mandatory_head = spec["action_head_type"] in (
        "flow-matching expert", "flow-matching (Gemma-300M)",
    ) or "flow-matching" in spec["action_head_type"]
    ternary = path == "int_only"

    if ternary:
        int_pct, fp_pct = 100.0, 0.0
        head_note = "fully ternary {-1,0,+1}; no FP-mandatory component"
        tier = "INT8-only OK (pure-int; ternary needs bitblas/LUT kernels to realize compute win)"
    elif fp_mandatory_head:
        # VLM backbone (vision+LLM) is INT-quantizable; action head MUST stay FP.
        int_pct = pct(vlm, total)
        fp_pct = pct(action_b, total)
        head_note = f"flow-matching action head ({action_m:.0f}M) REQUIRES FP (BF16/FP8) per QuantVLA; INT8 path for VLM only"
        tier = "needs FP-capable tier (NPU High / dGPU) for the action expert"
    else:
        # Single-loop autoregressive: whole stack INT8-friendly.
        int_pct, fp_pct = 100.0, 0.0
        head_note = "single-loop autoregressive; INT8-friendly throughout"
        tier = "INT8-only OK"

    return {
        "family": "vla",
        "display_name": spec["display_name"],
        "architecture": spec["architecture"],
        "n_params_b": total,
        "deployable_dtype": path,
        "npu_tier_fit": tier,
        "by_params": {
            "provenance": "documented_constraint",
            "int_capable_pct": round(int_pct, 3),
            "fp_required_pct": round(fp_pct, 3),
            "breakdown_b": {
                "vlm_backbone_int_capable": vlm,
                "action_head": action_b,
            },
            "note": head_note,
        },
        "by_flops": {
            "provenance": "not_computed",
            "note": "per-component FLOP decomposition not derived; the precision "
                    "REQUIREMENT (dtype_path + FP-mandatory head) is the load-bearing "
                    "fact for these models, not a FLOP %.",
        },
        "by_bytes": {
            "provenance": "measured",
            "note": "inference DRAM projections at three weight precisions (paper/spec).",
            "inference_dram_gb": {
                "bf16": spec["inference_dram_gb_bf16"],
                "int8": spec["inference_dram_gb_int8"],
                "int4": spec["inference_dram_gb_int4"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Vision catalog
# ---------------------------------------------------------------------------
def build_sam3():
    arch = json.load(open(os.path.join(REPO, "data/output/sam3_reference_architecture.json")))
    cats = arch["category_summary"]
    # by_params: matmul (conv+linear+attention proj) + embedding (quantizable) vs
    # fp-only (norm). Embeddings treated as quantizable, consistent with the LLMs.
    matmul_p = (cats["conv"]["params"] + cats["linear"]["params"]
                + cats["attention"]["params"] + cats["embedding"]["params"])
    fp_p = cats["norm"]["params"]
    total_p = matmul_p + fp_p

    # by_flops: the "attention" category folds in QK^T/AV matmuls (INT-capable) AND
    # softmax (FP). Decompose it. Per the dump, attention MACs = matmul GEMM work;
    # softmax has 0 MACs but real FLOPs. Approximate softmax FLOPs across the 36
    # attention blocks; everything with nonzero MACs is INT-capable.
    attn = cats["attention"]
    attn_matmul_flops = 2 * attn["macs"]              # GEMM portion (2*MACs)
    attn_softmax_flops = max(attn["flops"] - attn_matmul_flops, 0)  # FP residual
    conv_flops = cats["conv"]["flops"]
    linear_flops = cats["linear"]["flops"]
    norm_flops = cats["norm"]["flops"]
    embed_flops = cats["embedding"]["flops"]

    int_capable = attn_matmul_flops + conv_flops + linear_flops
    fp_tail = {"softmax": attn_softmax_flops, "norm": norm_flops, "embedding": embed_flops}
    fp_total = sum(fp_tail.values())
    total_f = int_capable + fp_total

    return {
        "family": "vision",
        "display_name": "SAM 3 (full)",
        "model_id": "sam3_full",
        "n_params_b": round(arch["model_summary"]["total_params"] / 1e9, 3),
        "deployable_dtype": "bf16 reference; int8/fp8 quantizable encoder",
        "npu_tier_fit": "INT8-only OK for matmul body; small FP tail (softmax/norm)",
        "by_params": {
            "provenance": "reference_arch",
            "scope": "encoder layers in the reference dump (313M of 848M total profiled)",
            "int_capable_pct": pct(matmul_p, total_p),
            "fp_required_pct": pct(fp_p, total_p),
            "breakdown_pct": {
                "matmul_weights": pct(matmul_p, total_p),
                "norm": pct(cats["norm"]["params"], total_p),
                "embedding": pct(cats["embedding"]["params"], total_p),
            },
        },
        "by_flops": {
            "provenance": "reference_arch",
            "int_capable_pct": pct(int_capable, total_f),
            "fp_tail_pct": pct(fp_total, total_f),
            "fp_tail_breakdown_pct": {k: pct(v, total_f) for k, v in fp_tail.items()},
            "note": "attention category split into GEMM (2*MACs, INT-capable) vs softmax (FP).",
        },
        "by_bytes": {
            "provenance": "reference_arch",
            "note": "param bytes at bf16; int8 halves matmul-weight bytes.",
            "param_bytes_bf16_mb": round(arch["model_summary"]["total_param_bytes_bf16"] / 1e6, 2),
        },
    }


def build_vision_quantizable():
    """YOLOv8n-seg + CLIP hybrid recipe: how many linear layers actually quantize."""
    hv = json.load(open(os.path.join(REPO, "data/output/bakeoff/hybrid_v2_summary.json")))
    info = hv[next(iter(hv))]["_info"]
    int8 = info["int8"]
    n_lin, n_q = int8["n_linear"], int8["n_quantized"]
    return {
        "family": "vision",
        "display_name": "YOLOv8n-seg + CLIP (hybrid_v2 recipe)",
        "deployable_dtype": "int8/fp8 on quantized linears, bf16 elsewhere",
        "npu_tier_fit": "INT8-only OK; un-quantized layers (detect/seg heads, norms) run fp",
        "by_params": {
            "provenance": "measured",
            "note": "of the linear layers, how many the int8/fp8 recipe actually quantizes.",
            "n_linear": n_lin,
            "n_quantized": n_q,
            "linear_layers_quantized_pct": pct(n_q, n_lin),
            "linear_layers_fp_kept_pct": pct(n_lin - n_q, n_lin),
            "yolo_params_m": int8["yolo_params_m"],
            "clip_params_m": int8["clip_params_m"],
        },
        "by_flops": {"provenance": "not_computed",
                     "note": "per-layer FLOP export not available for this recipe."},
        "by_bytes": {"provenance": "not_computed"},
    }


def main():
    anchors = json.load(open(os.path.join(REPO, "data/output/llm_anchors_5090.json")))["anchors"]
    models = {}

    # LLMs (skip MoE — routing makes the dense FLOP model invalid)
    for key, anchor in anchors.items():
        if anchor.get("model_arch") == "moe":
            continue
        models[anchor["model_id"]] = build_llm(key, anchor)

    # VLAs
    for key, fname in VLA_FILES.items():
        path = os.path.join(REPO, "data/output/bakeoff", fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        models[d["model_spec"]["vla_key"]] = build_vla(d["model_spec"], d.get("dtype"))

    # Vision
    models["sam3_full"] = build_sam3()
    models["yolov8n_seg_clip_hybrid"] = build_vision_quantizable()

    doc = {
        "__meta__": {
            "description": "INT vs FP composition of the keyhole model catalog along "
                           "three axes (params / FLOPs / bytes). See module docstring "
                           "for why the axes disagree and what is/ isn't measured.",
            "schema_version": 1,
            "methodology_version": "2026-06-01-precision-composition-v1",
            "reference_seq_len": REF_SEQ,
            "axes": ["by_params", "by_flops", "by_bytes"],
            "provenance_legend": {
                "measured": "from a 5090 measurement / real artifact (GGUF size, DRAM)",
                "reference_arch": "from a paper-spec architecture dump",
                "arch_analytic": "computed from published hyperparameters + standard FLOP model",
                "documented_constraint": "precision requirement stated in the model spec",
                "not_computed": "intentionally not derived; see note",
            },
            "tier_dtype_support": TIER_DTYPE_SUPPORT,
            "key_caveat": "No per-kernel INT-vs-FP runtime trace exists (NCU labels dtype "
                          "only in range names). All by_flops splits are ANALYTIC. The FP "
                          "tail is small by FLOPs but dominates latency (memory-/launch-bound).",
        },
        "models": models,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {OUT}  ({len(models)} models)")


if __name__ == "__main__":
    main()
