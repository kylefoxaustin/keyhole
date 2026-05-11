"""Consolidate per-quant LLM 5090 anchors into a single canonical file.

Produces `data/output/llm_anchors_5090.json` — the canonical source of
truth for 5090 LLM measurement cells, consumed by:

  - keyhole's data bundle (this repo) — read at bundle-build time
  - keyhole-sizer's `RTX_5090_REFERENCE.measured_llm` cells (alias keys)
  - personal-ai-framework's `build_data_bundle.py` — vendors a copy at
    bundle-build time with provenance pointer

The (b)/(c) hybrid agreed with [docs] 12:21 + 12:43:
  Keyhole owns canonical (measurements live in the harness here);
  Skippy bundle reads via vendored copy with __source__ provenance link.

Schema:

    {
      "__meta__": { ... },
      "anchors": {
         "<measurement_alias>": {
            "model_id": "<directory name>",
            "model_arch": "<llama-dense | mistral-dense | qwen-dense | moe>",
            "n_params_active_b": <float>,
            "n_params_total_b":  <float>,
            "compute_dtype": "fp16",
            "quants": {
              "Q4_K_M": {
                 "gguf_size_gb": <float>,
                 "decode_tok_s_n256":   <float>,
                 "decode_tok_s_rag_8k_2k": <float>,
                 "prefill_tok_s_at_2k": <float>,
                 "ttft_1k_sec":         <float | null>,
                 "rag_total_sec":       <float>,
                 "source_path":         "data/output/bakeoff/.../Q4_K_M.json",
              },
              ...
            }
         }
      }
    }

Sources:
  - data/output/bakeoff/llm_anchors/<model>/<quant>.json (cross-family)
  - data/output/bakeoff/llm_summary.json                (MoE bake-off)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ANCHORS_DIR = ROOT / "data" / "output" / "bakeoff" / "llm_anchors"
LLM_SUMMARY = ROOT / "data" / "output" / "bakeoff" / "llm_summary.json"
OUT_JSON = ROOT / "data" / "output" / "llm_anchors_5090.json"


def _git_head() -> str:
    head_path = ROOT / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text().strip()
    if head.startswith("ref: "):
        ref = (ROOT / ".git" / head[5:]).read_text().strip()
        return ref[:12]
    return head[:12]


# Map model_id (directory name) → measurement_alias (sizer's RTX_5090_REFERENCE.measured_llm key)
# Matches the convention in [sizer]/[pai-sizer]'s catalog wiring.
MODEL_ID_TO_ALIAS = {
    "qwen2.5-7b-dense":      "qwen_2_5_7b_dense",
    "qwen2.5-32b-dense":     "qwen_2_5_32b_dense",
    "llama3.1-8b-dense":     "llama_3_1_8b_dense",
    "mistral-7b-v0.3-dense": "mistral_7b_v03_dense",
}

MODEL_METADATA = {
    "qwen2.5-7b-dense":      ("qwen-dense",    7.62, 7.62),
    "qwen2.5-32b-dense":     ("qwen-dense",   32.5, 32.5),
    "llama3.1-8b-dense":     ("llama-dense",   8.03, 8.03),
    "mistral-7b-v0.3-dense": ("mistral-dense", 7.25, 7.25),
}


def _quant_payload_from_dense(payload: dict, source_rel: str) -> dict:
    """Extract canonical fields from a dense per-quant JSON."""
    decode_sweep = payload.get("decode_sweep", [])
    decode_n256 = decode_sweep[-1].get("decode_tok_s") if decode_sweep else None
    prefill_sweep = payload.get("prefill_sweep", [])
    prefill_2k = next(
        (p.get("prefill_tok_s") for p in prefill_sweep if p.get("n_prompt") == 2048),
        None,
    )
    ttft_1k = next(
        (p.get("ttft_sec") for p in prefill_sweep if p.get("n_prompt") == 1024),
        None,
    )
    rag = payload.get("rag", {})
    rag_total_ms = rag.get("total_ms") or (rag.get("total_sec", 0) * 1000.0 if rag.get("total_sec") else None)
    rag_total_sec = (rag_total_ms / 1000.0) if rag_total_ms else None
    return {
        "gguf_size_gb":              payload.get("gguf_size_gb"),
        "decode_tok_s_n256":         decode_n256,
        "decode_tok_s_rag_8k_2k":    rag.get("decode_tok_s"),
        "prefill_tok_s_at_2k":       prefill_2k,
        "ttft_1k_sec":               ttft_1k,
        "rag_total_sec":             rag_total_sec,
        "source_path":               source_rel,
    }


def collect_dense_anchors() -> dict[str, Any]:
    """Per-model anchors from data/output/bakeoff/llm_anchors/<model>/<quant>.json"""
    out: dict[str, Any] = {}
    if not ANCHORS_DIR.exists():
        return out
    for model_dir in sorted(ANCHORS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        if model_id not in MODEL_ID_TO_ALIAS:
            continue
        alias = MODEL_ID_TO_ALIAS[model_id]
        arch, n_active, n_total = MODEL_METADATA[model_id]
        entry = {
            "model_id":          model_id,
            "model_arch":        arch,
            "n_params_active_b": n_active,
            "n_params_total_b":  n_total,
            "compute_dtype":     "fp16",
            "quants":            {},
        }
        for q in sorted(model_dir.glob("*.json")):
            payload = json.loads(q.read_text())
            rel = q.relative_to(ROOT).as_posix()
            entry["quants"][q.stem] = _quant_payload_from_dense(payload, rel)
        if entry["quants"]:
            out[alias] = entry
    return out


def collect_moe_anchor() -> dict[str, Any]:
    """The Qwen3-30B-A3B MoE anchor (canonical key: skippy_moe_30b_a3b)."""
    if not LLM_SUMMARY.exists():
        return {}
    summary = json.loads(LLM_SUMMARY.read_text())
    entry = {
        "model_id":          "qwen3-30b-a3b-moe",
        "model_arch":        "moe",
        "n_params_active_b": 3.0,
        "n_params_total_b":  30.0,
        "compute_dtype":     "fp16",
        "quants":            {},
    }
    for quant_key in ("Q4_K_M", "Q5_K_M", "Q8_0"):
        if quant_key not in summary:
            continue
        payload = summary[quant_key]
        rel = LLM_SUMMARY.relative_to(ROOT).as_posix()
        entry["quants"][quant_key] = _quant_payload_from_dense(payload, rel)
    return {"skippy_moe_30b_a3b": entry} if entry["quants"] else {}


def main() -> int:
    anchors = {}
    anchors.update(collect_dense_anchors())
    anchors.update(collect_moe_anchor())

    bundle = {
        "__meta__": {
            "generated_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repo":                 "keyhole",
            "git_head":             _git_head(),
            "schema_version":       1,
            "methodology_version":  "2026-05-11-substring-arc-closed",
            "description": (
                "Canonical 5090 LLM measurement cells. Cells keyed by "
                "measurement_alias (the same key sizer / pai-sizer use in "
                "RTX_5090_REFERENCE.measured_llm). Use this file as the "
                "authoritative source rather than walking individual "
                "per-quant JSONs; same data, easier to consume."
            ),
            "harness":              "scripts/bakeoff_llm_anchors.py + scripts/bakeoff_llm.py",
            "host":                 "RTX 5090 + i9-14900KF, llama-cpp-python 0.3.20 (CUDA, n_ctx=16384)",
            "n_anchors":            len(anchors),
        },
        "anchors": anchors,
    }

    OUT_JSON.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"Wrote {OUT_JSON.relative_to(ROOT)} ({OUT_JSON.stat().st_size // 1024} KB)")
    print(f"  anchors: {len(anchors)}")
    for alias, entry in anchors.items():
        n_q = len(entry["quants"])
        print(f"    {alias}: {entry['model_id']} ({n_q} quant{'s' if n_q != 1 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
