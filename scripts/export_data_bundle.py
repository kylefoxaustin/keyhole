"""Aggregate every measurement / summary / projection in the repo into a
single Claude-consumable bundle.

Inputs (read-only):
  data/output/bakeoff/*.json          — top-level summary + edge-projection JSONs
  data/output/bakeoff/llm_anchors/*/*.json
  data/output/ncu/sizer_bundle.json   — curated DRAM-per-forward bundle (16+ workloads)

Outputs:
  data/output/keyhole_data_bundle.json
  data/output/keyhole_data_bundle.md

Skips:
  data/output/bakeoff/<dir>/         — per-frame raw data subdirs
  data/output/runs/                  — per-video pipeline run traces (not measurements)
  data/output/bakeoff/visuals/       — image artifacts

The .md is the human/Claude-readable view. The .json is for tools that want
to cross-correlate fields programmatically.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BAKEOFF = ROOT / "data" / "output" / "bakeoff"
NCU = ROOT / "data" / "output" / "ncu"
OUT_JSON = ROOT / "data" / "output" / "keyhole_data_bundle.json"
OUT_MD = ROOT / "data" / "output" / "keyhole_data_bundle.md"

# Canonical 5090 → NPU Mid stock LPDDR5X bandwidth ratio used across the deck +
# sizer. Effective BW: 5090 = 1792 GB/s × 0.85; Mid = 134.4 GB/s × 0.70.
BW_RATIO_5090_TO_NPU_MID = (1792 * 0.85) / (134.4 * 0.70)


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - data hygiene fallback
        return {"__load_error__": str(exc), "__path__": str(path)}


def _add_quality_aliases(payload: Any) -> Any:
    """Walk a bake-off payload and add `*_vs_fp16_engine` aliases for any
    `box_recall` / `mean_matched_iou` fields. The legacy fields measure
    quantization drift vs the FP16 TRT engine on the same input frames —
    not vs ground-truth labels. The aliased field names make this explicit
    in user-facing artifacts.

    Mutates in place; returns the payload for chaining.
    """
    if isinstance(payload, dict):
        if "box_recall" in payload and "box_recall_vs_fp16_engine" not in payload:
            payload["box_recall_vs_fp16_engine"] = payload["box_recall"]
        if "mean_matched_iou" in payload and "mean_matched_iou_vs_fp16_engine" not in payload:
            payload["mean_matched_iou_vs_fp16_engine"] = payload["mean_matched_iou"]
        for v in payload.values():
            _add_quality_aliases(v)
    elif isinstance(payload, list):
        for v in payload:
            _add_quality_aliases(v)
    return payload


def collect_bakeoff_summaries() -> dict[str, Any]:
    """Walk top-level bakeoff/*.json files. Skip subdirs and run traces.

    Adds `*_vs_fp16_engine` aliases (engine-self-comparison clarification)
    to every quality field — see KH-P0-003 in REMEDIATION_PLAN.md.
    """
    out: dict[str, Any] = {}
    for p in sorted(BAKEOFF.glob("*.json")):
        out[p.stem] = _add_quality_aliases(_load_json(p))
    return out


def collect_llm_anchors() -> dict[str, Any]:
    """Walk bakeoff/llm_anchors/<model>/<quant>.json."""
    anchors_root = BAKEOFF / "llm_anchors"
    if not anchors_root.exists():
        return {}
    out: dict[str, Any] = {}
    for model_dir in sorted(anchors_root.iterdir()):
        if not model_dir.is_dir():
            continue
        out[model_dir.name] = {}
        for q in sorted(model_dir.glob("*.json")):
            out[model_dir.name][q.stem] = _load_json(q)
    return out


def collect_ncu() -> dict[str, Any]:
    """The curated DRAM-per-forward bundle is already a single tidy file."""
    bundle_path = NCU / "sizer_bundle.json"
    if not bundle_path.exists():
        return {}
    return _load_json(bundle_path) or {}


def _fmt_md_value(v: Any) -> str:
    """Cell-friendly stringification."""
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 0.01 or abs(v) >= 100_000:
            return f"{v:.3e}"
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    return str(v)


def _md_h(level: int, text: str) -> str:
    return f"{'#' * level} {text}\n\n"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_(no rows)_\n\n"
    h = "| " + " | ".join(str(x) for x in headers) + " |\n"
    sep = "|" + "|".join(["---"] * len(headers)) + "|\n"
    body = "".join(
        "| " + " | ".join(_fmt_md_value(c) for c in r) + " |\n"
        for r in rows
    )
    return h + sep + body + "\n"


def _render_ncu(ncu: dict) -> str:
    """Render the ncu sizer_bundle as a sorted DRAM-per-forward table."""
    if not ncu or "workloads" not in ncu:
        return "_(no ncu data)_\n\n"
    headers = [
        "Workload", "Source bake-off", "DRAM MB/fwd",
        "BW-bound ms @ NPU Mid", "BW-bound FPS @ NPU Mid", "n forwards",
    ]
    rows = []
    for w in ncu["workloads"]:
        per_fwd = w.get("per_forward", {})
        proj = w.get("edge_projection_npu_mid", {})
        rows.append([
            w.get("workload_id", "—"),
            w.get("source_bakeoff", "—"),
            per_fwd.get("dram_mb"),
            proj.get("bw_bound_ms_min"),
            proj.get("bw_bound_fps_max"),
            w.get("n_forwards", "—"),
        ])
    rows.sort(key=lambda r: (r[2] if isinstance(r[2], (int, float)) else 1e18))
    md = _md_h(3, "ncu-measured DRAM per forward (sorted ascending by MB/fwd)")
    md += (
        f"_NPU Mid effective BW used for FPS ceiling: "
        f"{ncu.get('workloads', [{}])[0].get('edge_projection_npu_mid', {}).get('npu_effective_gbs', '—')} "
        f"GB/s. Bundle: `{ncu.get('description', 'sizer_bundle.json')}`. "
        f"Host: `{ncu.get('measurement_host', '—')}`._\n\n"
    )
    md += _md_table(headers, rows)
    if ncu.get("known_gaps"):
        md += "**Known gaps:**\n\n"
        for g in ncu["known_gaps"]:
            md += f"- {g}\n"
        md += "\n"
    return md


def _render_llm_anchors(anchors: dict) -> str:
    """Per-model 5090 LLM anchor measurements."""
    if not anchors:
        return ""
    md = _md_h(3, "LLM 5090 anchors (llama-cpp-python)")
    headers = [
        "Model", "Quant", "GGUF GB", "Prefill@2K tok/s",
        "Decode@256 tok/s", "RAG 8K+2K decode tok/s", "RAG total s",
    ]
    rows = []
    for model_id, quants in anchors.items():
        for quant, payload in quants.items():
            if not isinstance(payload, dict):
                continue
            decode_sweep = payload.get("decode_sweep", [])
            decode_256 = decode_sweep[-1].get("decode_tok_s") if decode_sweep else None
            prefill_sweep = payload.get("prefill_sweep", [])
            prefill_2k = next(
                (p.get("prefill_tok_s") for p in prefill_sweep
                 if p.get("n_prompt") == 2048),
                None,
            )
            rag = payload.get("rag", {})
            rows.append([
                model_id, quant,
                payload.get("gguf_size_gb", "—"),
                prefill_2k,
                decode_256,
                rag.get("decode_tok_s"),
                rag.get("total_sec"),
            ])
    md += _md_table(headers, rows)
    return md


def _render_bakeoff_index(bakeoffs: dict) -> str:
    md = _md_h(3, "Bake-off summary file index")
    headers = ["File", "Top-level keys", "Brief"]
    rows = []
    briefs = {
        "edge_projection": "Run-comparison projections per video clip",
        "fp8_summary": "torchao FP8 activation quantization on ES-Small + YOLO",
        "fp8_edge_projection": "FP8 edge projection (NPU Mid stock LPDDR5X)",
        "smoothquant_summary": "SmoothQuant + plain INT8 (CONVERT blocked by torchao 0.17)",
        "hybrid_v2_summary": "YOLO-seg + CLIP open-vocab pipeline (replaces SAM 3)",
        "yolo_conv_quant_summary": "torchao Conv-1x1→Linear quant on YOLO-seg",
        "trt_yolo_summary": "TensorRT 10.16 YOLO-seg full-model FP8/INT8",
        "trt_clip_summary": "TensorRT CLIP visual tower FP8",
        "trt_yoloe26_summary": "TRT-FP8 YOLOE-26S-PF (negative result; kernel-launch-bound)",
        "yoloe26_summary": "Ultralytics YOLOE-26 PyTorch FP16 baseline",
        "efficientsam3_summary": "Community EfficientSAM3 ES-EV-S (Apache-2.0, BF16)",
        "efficientsam3p1_summary": "EfficientSAM3.1 ES-EV-S text-prompt-capable variant",
        "vit_alternatives_summary": "RT-DETR-L / DETR / OWLv2 / Grounding-DINO benchmark",
        "vit_alternatives_recall": "ViT-alternatives box recall vs YOLO11x reference",
        "concurrency_edge_projection": "yolo11s-seg multi-stream batching projection",
        "concurrency_yolov8n-seg_edge_projection": "yolov8n-seg multi-stream batching projection",
        "keyframe_debounce_summary": "CLIP @ 1Hz debounce post-processing",
        "llm_summary": "Qwen3-30B-A3B MoE Q4/Q5/Q8 sweep",
        "llm_edge_projection": "LLM edge projection across NPU tiers",
        "llm_anchors_summary": "Per-model 5090 anchors (Qwen 7B/32B + Llama + Mistral)",
        "resnet50_summary": "ResNet-50v1 INT8 TRT (5090 anchor for sizer Path C)",
    }
    for name, payload in bakeoffs.items():
        keys = (
            ", ".join(sorted(payload.keys())[:6])
            if isinstance(payload, dict) and payload
            else "—"
        )
        # Several bake-offs come in {summary, edge_projection} pairs — match either.
        brief = briefs.get(name, briefs.get(name.replace("_yolov8n-seg", "")))
        rows.append([f"`{name}.json`", keys, brief or "—"])
    md += _md_table(headers, rows)
    return md


def render_md(bundle: dict) -> str:
    md = _md_h(1, "Keyhole data bundle")
    md += (
        f"Generated: `{bundle['__meta__']['generated_at']}`  \n"
        f"Source repo: `{bundle['__meta__']['repo']}` (`{bundle['__meta__']['git_head']}`)  \n"
        f"5090→NPU Mid BW ratio: `{BW_RATIO_5090_TO_NPU_MID:.4f}` "
        f"(effective: 1792×0.85 / 134.4×0.70)\n\n"
    )
    md += (
        "This bundle aggregates every measurement summary + edge projection in "
        "the keyhole repo into one document. Companion JSON: "
        "`keyhole_data_bundle.json` has the same data in machine-parseable form. "
        "Companion document: `docs/CLAUDE_REVIEW_BRIEFING.md` is the narrative "
        "interpretation of these numbers.\n\n"
    )

    md += _md_h(2, "1. ncu DRAM-per-forward measurements")
    md += _render_ncu(bundle.get("ncu", {}))

    md += _md_h(2, "2. LLM 5090 anchors")
    md += _render_llm_anchors(bundle.get("llm_anchors", {}))

    md += _md_h(2, "3. Bake-off summary files")
    md += (
        "Each row below corresponds to a top-level `.json` in `data/output/bakeoff/`. "
        "The full payloads are in the JSON bundle (key `bakeoffs.<file_stem>`).\n\n"
    )
    md += _render_bakeoff_index(bundle.get("bakeoffs", {}))

    md += _md_h(2, "4. Methodology notes")
    md += (
        "- **5090 anchor reference.** All edge projections start from 5090 wall-time, "
        "scaled to NPU Mid by the BW ratio above. NPU Mid effective bandwidth = "
        "94.08 GB/s (134.4 × 0.70 efficiency). 5090 effective = 1523.2 GB/s "
        "(1792 × 0.85).\n"
        "- **Effective BW efficiency = 0.70** is uniform across all 4 NPU tier "
        "presets (Low-LP4 / Low-LP5X / Mid / High). Earlier deck snapshots "
        "used 0.75/0.80; reconciled to 0.70 globally on 2026-04-21.\n"
        "- **Mid is INT8-only** (200 TOPS, no FP). High is FP-capable "
        "(200 BF16/FP16 + 400 INT8/FP8). Mid + High share the same stock "
        "LPDDR5X-8.4 memory class so BW-bound ceilings match; differentiator "
        "is dtype gating.\n"
        "- **dtype_mismatch flag** in projections means a recipe (e.g. FP8 CLIP) "
        "can't deploy on Mid silicon and projects to High instead.\n"
        "- **ncu replay mode.** Most workloads measured via app-replay (fast); "
        "TRT engines + dynamic NMS use kernel-replay (~80 min per target).\n"
    )
    return md


def _git_head() -> str:
    head_path = ROOT / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text().strip()
    if head.startswith("ref: "):
        ref = (ROOT / ".git" / head[5:]).read_text().strip()
        return ref[:12]
    return head[:12]


def main() -> int:
    bakeoffs = collect_bakeoff_summaries()
    llm_anchors = collect_llm_anchors()
    ncu = collect_ncu()

    bundle = {
        "__meta__": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repo": "keyhole",
            "git_head": _git_head(),
            "bw_ratio_5090_to_npu_mid": BW_RATIO_5090_TO_NPU_MID,
            "npu_mid_effective_gbs": 94.08,
            "schema_version": 2,
            "schema_v2_changes": (
                "Bake-off quality fields gain `box_recall_vs_fp16_engine` and "
                "`mean_matched_iou_vs_fp16_engine` aliases. These measure "
                "quantization drift relative to the FP16 TRT engine on the "
                "same input frames — NOT vs ground-truth labels. Legacy "
                "field names (`box_recall`, `mean_matched_iou`) preserved "
                "as aliases for one cycle. See KH-P0-003 in REMEDIATION_PLAN.md."
            ),
            "methodology_version": "2026-05-08-post-remediation",
        },
        "ncu": ncu,
        "llm_anchors": llm_anchors,
        "bakeoffs": bakeoffs,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(bundle, indent=2, default=str))
    OUT_MD.write_text(render_md(bundle))

    json_kb = OUT_JSON.stat().st_size // 1024
    md_kb = OUT_MD.stat().st_size // 1024
    n_bake = len(bakeoffs)
    n_llm = sum(len(v) for v in llm_anchors.values())
    n_ncu = len(ncu.get("workloads", []))
    print(
        f"Wrote {OUT_JSON.relative_to(ROOT)} ({json_kb} KB) and "
        f"{OUT_MD.relative_to(ROOT)} ({md_kb} KB).\n"
        f"  bake-off summaries: {n_bake}\n"
        f"  LLM anchor cells:   {n_llm}\n"
        f"  ncu workloads:      {n_ncu}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
