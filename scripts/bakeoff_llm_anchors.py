"""LLM 5090 anchor bake-off — additional model anchors for sizer Phase 2.

Sister to bakeoff_llm.py (which characterizes Qwen3-30B-A3B MoE end-to-end
with NPU vendor projections). This script is narrower: just collect 5090
measurements for ADDITIONAL anchor cells the sizer needs, on a registry
of models. No NPU vendor projection — sizer's projection model handles
cross-tier extrapolation downstream once the 5090 anchor lands.

Models in scope:

  qwen2.5-7b-dense    Qwen 2.5 7B Instruct dense — Q4_K_M / Q5_K_M / Q8_0  (2026-05-01)
  qwen2.5-32b-dense   Qwen 2.5 32B Instruct dense — Q4_K_M / Q5_K_M (no Q8) (2026-05-01)
  llama3.1-8b-dense   Meta Llama-3.1 8B Instruct dense — Q4_K_M             (2026-05-07, cross-family)
  mistral-7b-v0.3-dense  Mistral 7B Instruct v0.3 dense — Q4_K_M             (2026-05-07, cross-family)

Q8_0 dropped from the 32B dense plan because ~32 GB weights wouldn't leave
room on 5090's 32 GB VRAM for KV cache + activations (unlike MoE Q8 where
expert routing means only active params are VRAM-resident).

Cross-family round (Llama + Mistral) accompanies [docs]'s Tier 3 #1
training-side cross-family validation. Sister-evidence on the BW side:
verifies that 7B-class dense decode tok/s on 5090 is base-family-invariant
(within tokenizer + attention-pattern noise) — letting [docs]'s accuracy
findings stand as a model-architecture call, not a perf one. Q4_K_M is the
production target (matches v2-RAG eval quant); add Q5/Q8 later if useful.

Reuses bakeoff_llm.py's measure_prefill / measure_decode / measure_rag /
make_synthetic_tokens for harness fidelity. Output structure mirrors the
existing per-quant JSON layout but namespaced under
data/output/bakeoff/llm_anchors/<model_id>/<quant>.json so it doesn't
collide with the canonical Qwen3-30B-A3B MoE results.

Outputs:
  data/output/bakeoff/llm_anchors/<model_id>/<quant>.json
  data/output/bakeoff/llm_anchors_summary.json    (cross-model summary)

Run from repo root:
    ~/.virtualenvs/keyhole/bin/python scripts/bakeoff_llm_anchors.py
    ~/.virtualenvs/keyhole/bin/python scripts/bakeoff_llm_anchors.py --model qwen2.5-7b-dense
    ~/.virtualenvs/keyhole/bin/python scripts/bakeoff_llm_anchors.py --model qwen2.5-32b-dense --quants Q4_K_M
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.bakeoff_llm import (
    make_synthetic_tokens,
    measure_prefill,
    measure_decode,
    measure_rag,
    PREFILL_LENGTHS,
    DECODE_LENGTHS,
    RAG_PROMPT_LEN,
    RAG_RESPONSE_LEN,
    N_BATCH,
    N_THREADS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("llm_anchors")

WEIGHTS = REPO_ROOT / "weights"
OUT_DIR = REPO_ROOT / "data" / "output" / "bakeoff" / "llm_anchors"
SUMMARY_PATH = REPO_ROOT / "data" / "output" / "bakeoff" / "llm_anchors_summary.json"


# ───────────────────────── Model registry ─────────────────────────

MODELS: dict[str, dict] = {
    "qwen2.5-7b-dense": {
        "label": "Qwen 2.5 7B Instruct (dense)",
        "params_total_b": 7.6,
        "params_active_b": 7.6,
        "model_class": "dense",
        "quants": {
            "Q4_K_M": {
                "gguf": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                "n_gpu_layers": -1, "n_ctx": 16384,
                "offload_note": "all layers on GPU (~4.7 GB weights, comfortable)",
            },
            "Q5_K_M": {
                "gguf": "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
                "n_gpu_layers": -1, "n_ctx": 16384,
                "offload_note": "all layers on GPU (~5.4 GB weights, comfortable)",
            },
            "Q8_0": {
                "gguf": "Qwen2.5-7B-Instruct-Q8_0.gguf",
                "n_gpu_layers": -1, "n_ctx": 16384,
                "offload_note": "all layers on GPU (~8.1 GB weights, comfortable)",
            },
        },
    },
    "qwen2.5-32b-dense": {
        "label": "Qwen 2.5 32B Instruct (dense)",
        "params_total_b": 32.5,
        "params_active_b": 32.5,
        "model_class": "dense",
        "quants": {
            "Q4_K_M": {
                "gguf": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
                "n_gpu_layers": -1, "n_ctx": 12288,
                "offload_note": "all layers on GPU (~19.7 GB weights, ~9 GB headroom for KV @ 12K + activations)",
            },
            "Q5_K_M": {
                "gguf": "Qwen2.5-32B-Instruct-Q5_K_M.gguf",
                "n_gpu_layers": -1, "n_ctx": 12288,
                "offload_note": "all layers on GPU (~23 GB weights, ~6 GB headroom for KV @ 12K + activations)",
            },
            # Q8_0 deliberately not registered — would be ~32 GB weights vs 32 GB VRAM
            # which leaves no room for KV cache. Dense models can't expert-route to dodge.
        },
    },
    "llama3.1-8b-dense": {
        "label": "Meta Llama-3.1 8B Instruct (dense, cross-family)",
        "params_total_b": 8.03,
        "params_active_b": 8.03,
        "model_class": "dense",
        "family": "llama",
        "quants": {
            "Q4_K_M": {
                # Pulled from bartowski's GGUF mirror (ungated) — Meta-Llama is
                # gated on HF. Same source [docs] used for the 56.8% baseline eval.
                "gguf": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "n_gpu_layers": -1, "n_ctx": 16384,
                "offload_note": "all layers on GPU (~4.9 GB weights, comfortable; cross-family anchor)",
            },
        },
    },
    "mistral-7b-v0.3-dense": {
        "label": "Mistral 7B Instruct v0.3 (dense, cross-family)",
        "params_total_b": 7.25,
        "params_active_b": 7.25,
        "model_class": "dense",
        "family": "mistral",
        "quants": {
            "Q4_K_M": {
                "gguf": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                "n_gpu_layers": -1, "n_ctx": 16384,
                "offload_note": "all layers on GPU (~4.4 GB weights, comfortable; cross-family anchor)",
            },
        },
    },
}


# ───────────────────────── Per-quant runner ─────────────────────────

def run_one_quant(model_id: str, quant: str) -> dict:
    """Load + bake-off a single (model, quant) cell on 5090.

    Mirrors bakeoff_llm.py's run_one_quant structure but routed through this
    script's per-model config. Output JSON is the same shape so downstream
    consumers (export_results.py, sizer ingestion) see consistent fields.
    """
    from llama_cpp import Llama

    model_cfg = MODELS[model_id]
    quant_cfg = model_cfg["quants"][quant]
    gguf_path = WEIGHTS / quant_cfg["gguf"]
    if not gguf_path.exists():
        return {"quant": quant, "error": f"GGUF not found: {gguf_path}"}

    log.info("=" * 70)
    log.info("Model: %s  /  Quant: %s", model_cfg["label"], quant)
    log.info("GGUF:  %s (%.1f GB)", gguf_path.name, gguf_path.stat().st_size / 1e9)
    log.info("Note:  %s", quant_cfg["offload_note"])

    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(gguf_path),
        n_gpu_layers=quant_cfg["n_gpu_layers"],
        n_ctx=quant_cfg["n_ctx"],
        n_batch=N_BATCH,
        n_threads=N_THREADS,
        verbose=False,
        logits_all=False,
    )
    load_s = time.perf_counter() - t0
    vram_mb = torch.cuda.memory_allocated() / 1e6
    log.info("Loaded in %.1fs (VRAM allocated %.0f MB)", load_s, vram_mb)

    # Warmup
    warm = make_synthetic_tokens(llm, 32)
    llm.reset()
    llm.eval(warm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Prefill sweep
    prefill_results = []
    for n in PREFILL_LENGTHS:
        r = measure_prefill(llm, n)
        prefill_results.append(r)
        log.info("  prefill %5d: %7.0f ms, %6.0f tok/s", n, r["prefill_ms"], r["prefill_tok_s"])

    # Decode sweep
    decode_results = []
    for n in DECODE_LENGTHS:
        r = measure_decode(llm, n)
        decode_results.append(r)
        log.info("  decode  %5d: %7.0f ms, %6.1f tok/s", n, r["decode_ms"], r["decode_tok_s"])

    # Full RAG scenario
    rag = measure_rag(llm, RAG_PROMPT_LEN, RAG_RESPONSE_LEN)
    log.info("  RAG %d+%d: prefill %.0f ms (%.0f tok/s), decode %.0f ms (%.1f tok/s), total %.0f ms",
             RAG_PROMPT_LEN, RAG_RESPONSE_LEN,
             rag["prefill_ms"], rag["prefill_tok_s"],
             rag["decode_ms"], rag["decode_tok_s"], rag["total_ms"])

    final_peak_vram = torch.cuda.max_memory_allocated() / 1e6

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        "quant": quant,
        "model_id": model_id,
        "model_label": model_cfg["label"],
        "model_class": model_cfg["model_class"],
        "params_total_b": model_cfg["params_total_b"],
        "params_active_b": model_cfg["params_active_b"],
        "gguf_file": gguf_path.name,
        "gguf_size_gb": gguf_path.stat().st_size / 1e9,
        "n_gpu_layers": quant_cfg["n_gpu_layers"],
        "n_ctx": quant_cfg["n_ctx"],
        "offload_note": quant_cfg["offload_note"],
        "load_sec": load_s,
        "model_alloc_vram_mb": vram_mb,
        "peak_vram_mb_during_bench": final_peak_vram,
        "prefill_sweep": prefill_results,
        "decode_sweep": decode_results,
        "rag": rag,
    }


# ───────────────────────── Main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS) + ["all"], default="all",
                    help="Which model to bake off (default: all)")
    ap.add_argument("--quants", default=None,
                    help="Comma-separated subset of quants for the chosen model "
                         "(default: all quants registered for that model)")
    args = ap.parse_args()

    log.info("CUDA: %s", torch.cuda.get_device_name(0))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target_models = list(MODELS) if args.model == "all" else [args.model]
    quants_filter = set(args.quants.split(",")) if args.quants else None

    summary = {
        "host": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "models": {},
    }

    for model_id in target_models:
        model_cfg = MODELS[model_id]
        log.info("\n" + "#" * 70)
        log.info("# %s", model_cfg["label"])
        log.info("#" * 70)

        model_dir = OUT_DIR / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        per_model_results = {}

        for quant in model_cfg["quants"]:
            if quants_filter and quant not in quants_filter:
                continue
            quant_path = model_dir / f"{quant}.json"
            if quant_path.exists():
                cached = json.loads(quant_path.read_text())
                if "error" not in cached:
                    log.info("Reusing cached %s", quant_path)
                    per_model_results[quant] = cached
                    continue
                log.info("Cached %s contains error — re-running", quant_path)
            try:
                result = run_one_quant(model_id, quant)
                per_model_results[quant] = result
                if "error" in result:
                    # Don't cache errors (lets re-runs retry without manual cleanup)
                    log.warning("  → error result, NOT caching: %s", result["error"])
                else:
                    quant_path.write_text(json.dumps(result, indent=2))
                    log.info("Wrote %s", quant_path)
            except Exception as e:
                log.exception("FAILED on %s/%s", model_id, quant)
                per_model_results[quant] = {"error": f"{type(e).__name__}: {e}"}

        summary["models"][model_id] = {
            "label": model_cfg["label"],
            "model_class": model_cfg["model_class"],
            "params_total_b": model_cfg["params_total_b"],
            "params_active_b": model_cfg["params_active_b"],
            "per_quant": per_model_results,
        }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    log.info("\nWrote %s", SUMMARY_PATH)

    # Pretty print
    print()
    print(f"{'Model':<30} {'Quant':<8} {'GGUF GB':>8} {'Decode tok/s':>13} {'RAG total s':>12}")
    print("-" * 75)
    for model_id, m in summary["models"].items():
        for quant, r in m["per_quant"].items():
            if "error" in r:
                print(f"{model_id:<30} {quant:<8} ERROR: {r['error']}")
                continue
            print(f"{model_id:<30} {quant:<8} {r['gguf_size_gb']:>7.1f} "
                  f"{r['rag']['decode_tok_s']:>12.1f} {r['rag']['total_ms']/1000:>11.1f}")


if __name__ == "__main__":
    main()
