"""
LLM bake-off — Qwen3-30B-A3B-Instruct-2507 across three GGUF quantizations.

This is the sister bake-off to the vision stack: characterizes the LLM stage
that currently sits in the "NLQ / LLM" box of the Keyhole pipeline as a
placeholder. Prompt-free: synthetic token streams exercise the prefill and
decode paths so we can report tok/s without designing a prompt set.

MoE specifics:
  - 30B total params / 3B active per token (top-8 of 128 experts)
  - 48 layers, hidden_size 2048, native context 262K
  - Decode is bandwidth-bound on ACTIVE-params bytes, not total-params bytes,
    which makes MoE dramatically kinder to LPDDR5X than a dense 14B would be.

Runtime: llama.cpp via llama-cpp-python 0.3.20 with CUDA offload.

Outputs:
  data/output/bakeoff/llm/{quant}.json    (per-quant raw measurements)
  data/output/bakeoff/llm_summary.json
  data/output/bakeoff/llm_edge_projection.json
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # for cuda synchronize + VRAM probes

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("llm_bakeoff")

WEIGHTS = REPO_ROOT / "weights"
BAKEOFF_DIR = REPO_ROOT / "data" / "output" / "bakeoff"
OUT_DIR = BAKEOFF_DIR / "llm"

QUANTS = {
    "Q4_K_M": "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
    "Q5_K_M": "Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf",
    "Q8_0":   "Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf",
}

# MoE params (from Qwen3-30B-A3B config)
TOTAL_PARAMS = 30_000_000_000      # ~30B
ACTIVE_PARAMS = 3_000_000_000      # ~3B active per token
# Bytes per param per quant (effective, including overhead; rough)
BYTES_PER_PARAM = {"Q4_K_M": 0.57, "Q5_K_M": 0.68, "Q8_0": 1.04}

# Per-quant runtime tuning. Q8_0 GGUF (32.5 GB) is larger than the 5090's
# 32 GB VRAM; partial offload keeps some layers on CPU so the ctx KV cache
# + compute buffers fit. Q4/Q5 can go all-GPU with the full 16K context.
QUANT_RUNTIME = {
    "Q4_K_M": {"n_gpu_layers": -1,  "n_ctx": 16384, "offload_note": "all layers on GPU"},
    "Q5_K_M": {"n_gpu_layers": -1,  "n_ctx": 16384, "offload_note": "all layers on GPU"},
    "Q8_0":   {"n_gpu_layers": 32,  "n_ctx": 12288, "offload_note": "partial offload (32/48 layers on GPU) — weights exceed 32 GB VRAM"},
}

N_BATCH = 512          # llama.cpp batch size for prefill
N_THREADS = 8

PREFILL_LENGTHS = [128, 512, 2048, 8192]
DECODE_LENGTHS = [64, 256]
RAG_PROMPT_LEN = 8192
RAG_RESPONSE_LEN = 2048


def make_synthetic_tokens(llm, n: int) -> list[int]:
    """Build a synthetic token list of length n using the model's vocab.

    We pick a deterministic repeating pattern of common Qwen tokens so prefill
    has to actually compute (no special-case zero-len shortcuts)."""
    # Use tokenized "the quick brown fox" repeated, padded to length n
    seed = llm.tokenize(b"the quick brown fox jumps over the lazy dog ", add_bos=False)
    # Ensure BOS at position 0 for realistic prefill
    bos = llm.token_bos()
    tokens = [bos]
    while len(tokens) < n:
        tokens.extend(seed)
    return tokens[:n]


def measure_prefill(llm, n_tokens: int) -> dict:
    """Time a single full prefill of n_tokens. Does NOT sample."""
    tokens = make_synthetic_tokens(llm, n_tokens)
    llm.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.eval(tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return {
        "n_prompt": n_tokens,
        "prefill_ms": ms,
        "prefill_tok_s": n_tokens / (ms / 1000) if ms > 0 else 0,
    }


def measure_decode(llm, n_decode: int, warm_prompt_len: int = 128) -> dict:
    """Time generation of n_decode tokens after a short warm prefill."""
    warm = make_synthetic_tokens(llm, warm_prompt_len)
    llm.reset()
    llm.eval(warm)
    # Now decode one token at a time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_decode):
        tok = llm.sample(temp=0.0, top_k=1)  # deterministic, minimal sampling cost
        llm.eval([tok])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return {
        "n_decode": n_decode,
        "warm_prompt_len": warm_prompt_len,
        "decode_ms": ms,
        "decode_tok_s": n_decode / (ms / 1000) if ms > 0 else 0,
    }


def measure_rag(llm, prompt_len: int, response_len: int) -> dict:
    """End-to-end: one prefill of prompt_len then decode of response_len."""
    prompt = make_synthetic_tokens(llm, prompt_len)
    llm.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.eval(prompt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(response_len):
        tok = llm.sample(temp=0.0, top_k=1)
        llm.eval([tok])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t0) * 1000

    total_ms = prefill_ms + decode_ms
    return {
        "rag_prompt_len": prompt_len,
        "rag_response_len": response_len,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "prefill_tok_s": prompt_len / (prefill_ms / 1000) if prefill_ms > 0 else 0,
        "decode_tok_s": response_len / (decode_ms / 1000) if decode_ms > 0 else 0,
    }


def run_one_quant(quant: str, gguf_path: Path) -> dict:
    from llama_cpp import Llama

    cfg = QUANT_RUNTIME[quant]
    log.info("Loading %s (%s, n_ctx=%d) ...", gguf_path.name,
             cfg["offload_note"], cfg["n_ctx"])
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(gguf_path),
        n_gpu_layers=cfg["n_gpu_layers"],
        n_ctx=cfg["n_ctx"],
        n_batch=N_BATCH,
        n_threads=N_THREADS,
        verbose=False,
        logits_all=False,
    )
    load_s = time.perf_counter() - t0
    vram_mb = (torch.cuda.memory_allocated() / 1e6) if torch.cuda.is_available() else 0
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0
    log.info("Loaded in %.1fs (VRAM allocated %.0f MB)", load_s, vram_mb)

    # Warmup single eval to compile kernels
    warm = make_synthetic_tokens(llm, 32)
    llm.reset()
    llm.eval(warm)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Prefill sweep
    prefill_results = []
    for n in PREFILL_LENGTHS:
        r = measure_prefill(llm, n)
        prefill_results.append(r)
        log.info("  prefill %d: %.0f ms, %.0f tok/s", n, r["prefill_ms"], r["prefill_tok_s"])

    # Decode sweep
    decode_results = []
    for n in DECODE_LENGTHS:
        r = measure_decode(llm, n)
        decode_results.append(r)
        log.info("  decode %d: %.0f ms, %.1f tok/s", n, r["decode_ms"], r["decode_tok_s"])

    # Full RAG scenario
    rag = measure_rag(llm, RAG_PROMPT_LEN, RAG_RESPONSE_LEN)
    log.info("  RAG %d+%d: prefill %.0f ms (%.0f tok/s), decode %.0f ms (%.1f tok/s), total %.0f ms",
             RAG_PROMPT_LEN, RAG_RESPONSE_LEN,
             rag["prefill_ms"], rag["prefill_tok_s"],
             rag["decode_ms"], rag["decode_tok_s"], rag["total_ms"])

    final_peak_vram = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0

    # Free the model
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return {
        "quant": quant,
        "gguf_file": gguf_path.name,
        "gguf_size_gb": gguf_path.stat().st_size / 1e9,
        "n_gpu_layers": cfg["n_gpu_layers"],
        "n_ctx": cfg["n_ctx"],
        "offload_note": cfg["offload_note"],
        "load_sec": load_s,
        "model_alloc_vram_mb": vram_mb,
        "peak_vram_mb_during_bench": final_peak_vram,
        "prefill_sweep": prefill_results,
        "decode_sweep": decode_results,
        "rag": rag,
    }


def project_edge(all_results: dict) -> dict:
    """Edge projection per quant, assuming the LLM gets the whole NPU.

    Decode on MoE is bandwidth-bound on ACTIVE params per token:
        decode_tok_s = effective_BW / (active_params * bytes_per_param)

    Prefill has a compute-bound component that the NPU_emulator's TOPS ratio
    handles, but for an LLM on a single NPU we use a simpler linear scale:
        prefill_tok_s_edge ≈ prefill_tok_s_5090 * (edge_BW / 5090_BW)
    (LLM prefill on inference-optimized silicon roughly tracks bandwidth.)
    """
    from src.emulate.npu_emulator import RTX_5090, EDGE_MPU_TARGET

    edge_bw = EDGE_MPU_TARGET.mem_bandwidth_gbs * EDGE_MPU_TARGET.bandwidth_efficiency * 1e9
    ref_bw = RTX_5090.mem_bandwidth_gbs * RTX_5090.bandwidth_efficiency * 1e9
    bw_ratio = edge_bw / ref_bw

    # First pass: compute each quant's raw 5090 efficiency; flag any that
    # used partial offload (those have artificially-depressed efficiency).
    fully_offloaded_efficiencies = []
    quant_meta = {}
    for quant, result in all_results.items():
        if "error" in result:
            continue
        bpp = BYTES_PER_PARAM[quant]
        model_bytes_active = ACTIVE_PARAMS * bpp
        decode_ceiling_5090 = ref_bw / model_bytes_active
        decode_5090_longest = result["decode_sweep"][-1]["decode_tok_s"]
        raw_eff = decode_5090_longest / decode_ceiling_5090 if decode_ceiling_5090 > 0 else 0
        n_gpu_layers = result.get("n_gpu_layers", -1)
        fully_offloaded = (n_gpu_layers == -1)
        quant_meta[quant] = {
            "bpp": bpp,
            "model_bytes_active": model_bytes_active,
            "decode_ceiling_5090": decode_ceiling_5090,
            "decode_5090_longest": decode_5090_longest,
            "raw_eff": raw_eff,
            "fully_offloaded": fully_offloaded,
        }
        if fully_offloaded:
            fully_offloaded_efficiencies.append(raw_eff)

    # Use the best fully-offloaded efficiency as the reference. Partially-
    # offloaded quants get this applied instead of their own depressed number
    # (a unified-memory edge NPU wouldn't carry the CPU-offload penalty).
    edge_eff_reference = max(fully_offloaded_efficiencies) if fully_offloaded_efficiencies else 0

    proj = {}
    for quant, result in all_results.items():
        if "error" in result:
            proj[quant] = {"error": result["error"]}
            continue
        meta = quant_meta[quant]
        decode_ceiling_edge = edge_bw / meta["model_bytes_active"]

        # Use empirical efficiency IF the quant was fully GPU-resident;
        # otherwise apply the reference (best of the fully-offloaded ones).
        if meta["fully_offloaded"]:
            efficiency_for_edge = meta["raw_eff"]
            eff_source = "5090 empirical (fully offloaded)"
        else:
            efficiency_for_edge = edge_eff_reference
            eff_source = (f"ref from best fully-offloaded quant "
                          f"({edge_eff_reference*100:.0f}%) — avoids "
                          f"CPU-offload penalty that doesn't apply to "
                          f"unified-memory edge silicon")
        decode_edge_est = decode_ceiling_edge * efficiency_for_edge

        # Prefill: linear bandwidth scale (rough)
        prefill_edge_sweep = []
        for r in result["prefill_sweep"]:
            prefill_edge_sweep.append({
                "n_prompt": r["n_prompt"],
                "prefill_tok_s_5090": r["prefill_tok_s"],
                "prefill_tok_s_edge": r["prefill_tok_s"] * bw_ratio,
            })

        # RAG end-to-end projection
        rag = result["rag"]
        rag_edge_prefill_tok_s = rag["prefill_tok_s"] * bw_ratio
        rag_edge_decode_tok_s = decode_edge_est  # same regime
        rag_edge_prefill_ms = (RAG_PROMPT_LEN / rag_edge_prefill_tok_s) * 1000 if rag_edge_prefill_tok_s > 0 else 0
        rag_edge_decode_ms = (RAG_RESPONSE_LEN / rag_edge_decode_tok_s) * 1000 if rag_edge_decode_tok_s > 0 else 0
        rag_edge_total_ms = rag_edge_prefill_ms + rag_edge_decode_ms

        proj[quant] = {
            "quant": quant,
            "bytes_per_param": meta["bpp"],
            "active_model_bytes": meta["model_bytes_active"],
            "decode_ceiling_5090_tok_s": meta["decode_ceiling_5090"],
            "decode_ceiling_edge_tok_s": decode_ceiling_edge,
            "decode_efficiency_used": efficiency_for_edge,
            "decode_efficiency_source": eff_source,
            "fully_offloaded_on_5090": meta["fully_offloaded"],
            "decode_5090_measured_tok_s": meta["decode_5090_longest"],
            "decode_edge_projected_tok_s": decode_edge_est,
            "prefill_sweep_edge": prefill_edge_sweep,
            "rag_edge_prefill_ms": rag_edge_prefill_ms,
            "rag_edge_decode_ms": rag_edge_decode_ms,
            "rag_edge_total_ms": rag_edge_total_ms,
            "rag_edge_total_sec": rag_edge_total_ms / 1000,
        }

    return {
        "projections": proj,
        "bw_ratio_edge_vs_5090": bw_ratio,
        "edge_effective_bw_gb_s": edge_bw / 1e9,
        "ref_effective_bw_gb_s": ref_bw / 1e9,
        "method": (
            "LLM edge projection. Decode tok/s = bandwidth / (active_params * "
            "bytes_per_param), scaled by measured 5090 efficiency. MoE active "
            "params = 3B (Qwen3-30B-A3B). Prefill tok/s linearly scaled by "
            "bandwidth ratio (approximation — true prefill has a compute "
            "component, but on inference-optimized NPUs decode dominates edge BW)."
        ),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}

    for quant, fname in QUANTS.items():
        gguf = WEIGHTS / fname
        if not gguf.exists():
            log.warning("%s missing — skipping (%s)", quant, gguf)
            all_results[quant] = {"error": f"file not found: {gguf}"}
            continue

        out_path = OUT_DIR / f"{quant}.json"
        if out_path.exists():
            log.info("Reusing cached %s", out_path)
            all_results[quant] = json.loads(out_path.read_text())
            continue

        log.info("=== %s (%s, %.1f GB) ===", quant, fname, gguf.stat().st_size / 1e9)
        try:
            r = run_one_quant(quant, gguf)
            all_results[quant] = r
            out_path.write_text(json.dumps(r, indent=2))
        except Exception as e:
            log.exception("Failed on %s", quant)
            all_results[quant] = {"error": f"{type(e).__name__}: {e}"}

    (BAKEOFF_DIR / "llm_summary.json").write_text(json.dumps(all_results, indent=2))
    log.info("Wrote llm_summary.json")

    proj = project_edge(all_results)
    (BAKEOFF_DIR / "llm_edge_projection.json").write_text(json.dumps(proj, indent=2))
    log.info("Wrote llm_edge_projection.json")

    # Pretty print
    print()
    hdr = (f"{'Quant':7s} | {'Model GB':>8s} | {'VRAM MB':>7s} | "
           f"{'Prefill @2K':>11s} {'Decode':>7s} | "
           f"{'Edge decode':>12s} {'Edge RAG sec':>13s}")
    print(hdr); print("-"*len(hdr))
    for quant, result in all_results.items():
        if "error" in result:
            print(f"{quant:7s} | ERROR: {result['error'][:70]}")
            continue
        pf2k = next((r["prefill_tok_s"] for r in result["prefill_sweep"] if r["n_prompt"] == 2048), 0)
        dec = result["decode_sweep"][-1]["decode_tok_s"]
        p = proj["projections"].get(quant, {})
        edge_dec = p.get("decode_edge_projected_tok_s", 0)
        edge_rag = p.get("rag_edge_total_sec", 0)
        print(f"{quant:7s} | {result['gguf_size_gb']:>8.1f} | "
              f"{result['peak_vram_mb_during_bench']:>7.0f} | "
              f"{pf2k:>8.0f} t/s {dec:>6.1f} t/s | "
              f"{edge_dec:>9.1f} t/s {edge_rag:>10.1f} s")


if __name__ == "__main__":
    main()
