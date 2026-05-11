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


# KH-P0-002 — Tier dtype support matrix.
# Mid is INT8-only (200 TOPS, no FP path); High is FP-capable (200 BF16 / 400 INT8 / 400 FP8).
TIER_DTYPE_SUPPORT = {
    "NPU Low-LP4":       ["INT8"],
    "NPU Low-LP5X":      ["INT8"],
    "NPU Low-LP5-32bit": ["INT8"],
    "NPU Mid":           ["INT8"],
    "NPU High":          ["INT8", "FP16", "BF16", "FP8"],
    "RTX 5090":          ["INT8", "FP16", "BF16", "FP8", "FP32"],
}

ALL_NPU_TIERS = ["NPU Low-LP4", "NPU Low-LP5X", "NPU Low-LP5-32bit", "NPU Mid", "NPU High"]


def _classify_recipe(name: str) -> dict | None:
    """Return a dtype-gating dict for a recipe key, or None if the key
    doesn't look like a recipe label (e.g., it's a model name or resolution).
    """
    n = name.lower()
    # INT8 path — deployable on every NPU tier
    if "int8" in n:
        return {
            "dtype_class":           "int8",
            "deployable_tiers":      ALL_NPU_TIERS + ["RTX 5090"],
            "dtype_mismatch_on_mid": False,
        }
    # FP8 path — High only (Mid INT8-only)
    if "fp8" in n:
        return {
            "dtype_class":           "fp8",
            "deployable_tiers":      ["NPU High", "RTX 5090"],
            "dtype_mismatch_on_mid": True,
            "dtype_mismatch_reason": "NPU Mid is INT8-only (200 TOPS, no FP path); recipe requires FP8. Projects to NPU High (BW-equal at stock LPDDR5X).",
        }
    # FP16 / BF16 / FP32 — High only
    if any(x in n for x in ("fp16", "bf16", "fp32")):
        return {
            "dtype_class":           "fp16-class",
            "deployable_tiers":      ["NPU High", "RTX 5090"],
            "dtype_mismatch_on_mid": True,
            "dtype_mismatch_reason": "NPU Mid is INT8-only (200 TOPS, no FP path); recipe requires FP16/BF16. Projects to NPU High (BW-equal at stock LPDDR5X).",
        }
    return None


def _looks_like_projection_leaf(payload: dict) -> bool:
    """Heuristic: this is a per-recipe projection cell if it has a 5090
    measurement + an edge-projection field together.
    """
    has_5090 = any(k.startswith(("mean_frame_ms_5090", "measured_frame_ms_5090"))
                   for k in payload)
    has_edge = any(k.startswith(("projected_ms_edge", "projected_fps_edge",
                                  "bandwidth_limited_ms"))
                   for k in payload)
    return has_5090 and has_edge


def _add_overhead_clarification(payload: Any) -> Any:
    """KH-P0-001: bake-off `projected_ms_edge` includes 5090-derived overhead
    (kernel-launch tax, memory hierarchy effects, NMS dispatch) — it is NOT a
    pure BW floor. ncu `bw_bound_ms_min` IS a pure BW floor (DRAM bytes ÷
    effective BW). The two answer different questions and a reviewer flipping
    between them sees a 20× discrepancy on small dense models.

    This step adds explicit `effective_edge_ms_with_overhead` /
    `effective_edge_fps_with_overhead` aliases on bake-off projection leaves
    so the user-facing field name signals what the number actually is.
    Legacy `projected_*` field names preserved.

    Mutates in place.
    """
    if isinstance(payload, dict):
        if _looks_like_projection_leaf(payload):
            if "projected_ms_edge" in payload and "effective_edge_ms_with_overhead" not in payload:
                payload["effective_edge_ms_with_overhead"] = payload["projected_ms_edge"]
            if "projected_fps_edge" in payload and "effective_edge_fps_with_overhead" not in payload:
                payload["effective_edge_fps_with_overhead"] = payload["projected_fps_edge"]
        for v in payload.values():
            _add_overhead_clarification(v)
    elif isinstance(payload, list):
        for v in payload:
            _add_overhead_clarification(v)
    return payload


def _add_ncu_floor_aliases(ncu: dict) -> dict:
    """KH-P0-001: rename ncu's `bw_bound_*` to `bw_floor_*` (additive, keep
    legacy fields for back-compat). The point of the rename is to make
    explicit that ncu's projection is a FLOOR (best-case minimum), not a
    realistic edge-latency prediction.

    Mutates the per-workload `edge_projection_npu_mid` dicts.
    """
    for w in ncu.get("workloads", []):
        proj = w.get("edge_projection_npu_mid", {})
        if "bw_bound_ms_min" in proj and "bw_floor_ms_npu_mid" not in proj:
            proj["bw_floor_ms_npu_mid"] = proj["bw_bound_ms_min"]
        if "bw_bound_fps_max" in proj and "bw_floor_fps_max_npu_mid" not in proj:
            proj["bw_floor_fps_max_npu_mid"] = proj["bw_bound_fps_max"]
        # Reword interpretation to make the floor framing explicit
        if proj and "interpretation" in proj:
            proj["interpretation"] = (
                "Pure BW FLOOR — absolute minimum edge latency given measured "
                "DRAM bytes/forward and NPU effective BW. Real edge latency "
                "ALWAYS exceeds this (kernel-launch tax, NMS dispatch, "
                "memory hierarchy stalls, sync overhead). Compare to the "
                "bake-off `effective_edge_ms_with_overhead` field for the "
                "5090-wall-time-derived projection that includes overhead. "
                "Reconciliation table in CLAUDE_REVIEW_BRIEFING.md § 8."
            )
    return ncu


def _apply_dtype_gating(payload: Any, parent_key: str | None = None) -> Any:
    """Walk a bake-off payload and tag each recipe-keyed projection leaf
    with `dtype_mismatch_on_mid` + `deployable_tiers` per
    TIER_DTYPE_SUPPORT.

    Historical FP-on-Mid run data is preserved (per reviewer's
    KH-P0-002 guidance: don't delete historical data; render dtype-mismatch
    as a flag, not a deletion).
    """
    if isinstance(payload, dict):
        # If this dict is a leaf projection AND its parent key looks like a
        # recipe label, apply gating.
        if parent_key is not None and _looks_like_projection_leaf(payload):
            cls = _classify_recipe(parent_key)
            if cls is not None and "dtype_mismatch_on_mid" not in payload:
                payload.update(cls)
        # Also check `recipe` field (some schemas put the dtype tag as a value)
        if "recipe" in payload and "dtype_mismatch_on_mid" not in payload:
            cls = _classify_recipe(str(payload["recipe"]))
            if cls is not None:
                payload.update(cls)
        for k, v in payload.items():
            _apply_dtype_gating(v, k)
    elif isinstance(payload, list):
        for v in payload:
            _apply_dtype_gating(v, parent_key)
    return payload


def collect_bakeoff_summaries() -> dict[str, Any]:
    """Walk top-level bakeoff/*.json files. Skip subdirs and run traces.

    Adds:
    - `*_vs_fp16_engine` aliases (engine-self-comparison clarification)
      per KH-P0-003
    - `dtype_mismatch_on_mid` + `deployable_tiers` per recipe leaf
      per KH-P0-002 (don't delete historical FP-on-Mid data; tag it)
    - `effective_edge_ms_with_overhead` /
      `effective_edge_fps_with_overhead` aliases per KH-P0-001
      (clarifies that bake-off projections include 5090-derived overhead,
      distinct from ncu's pure BW floor)
    """
    out: dict[str, Any] = {}
    for p in sorted(BAKEOFF.glob("*.json")):
        payload = _load_json(p)
        _add_quality_aliases(payload)
        _apply_dtype_gating(payload)
        _add_overhead_clarification(payload)
        out[p.stem] = payload
    return out


def collect_ncu() -> dict[str, Any]:
    """The curated DRAM-per-forward bundle is already a single tidy file.

    Per KH-P0-001: add `bw_floor_*` aliases that make the floor framing
    explicit (legacy `bw_bound_*` fields preserved).
    """
    bundle_path = NCU / "sizer_bundle.json"
    if not bundle_path.exists():
        return {}
    ncu = _load_json(bundle_path) or {}
    _add_ncu_floor_aliases(ncu)
    return ncu


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


def _render_bw_reconciliation(bundle: dict) -> str:
    """Per-workload reconciliation table: BW floor vs effective edge ms
    (with overhead). Addresses KH-P0-001 reviewer concern that the same
    workload appeared with two contradictory numbers.

    Picks canonical workloads where BOTH ncu floor and bake-off projection
    exist:
      - yolo_seg_fp8_trt   (shipping detector — TRT FP8)
      - yolo_seg_fp16_trt  (the 22.7× discrepancy example)
      - clip_trt           (shipping labeler — TRT FP16/FP8)
      - sam3_bf16_reference (the SAM 3 baseline workload)
      - efficientsam3_es_ev_s (Community SAM 3 Lite)
      - yoloe26_trt_fp8    (one-model alternative)
    """
    md = (
        "Two methodologies produce edge-latency estimates for the same "
        "workload-on-tier pair. They answer different questions and a "
        "reviewer flipping between them sees a 20×+ gap (KH-P0-001 caught "
        "this).\n\n"
        "- **BW floor** (ncu side, `bw_floor_ms_npu_mid`): pure "
        "DRAM-bytes/forward ÷ NPU effective BW. Best-case minimum; "
        "**cannot be achieved in practice** because real silicon pays "
        "kernel-launch overhead, NMS dispatch, memory hierarchy stalls, "
        "sync overhead.\n"
        "- **Effective edge ms with overhead** (bake-off side, "
        "`effective_edge_ms_with_overhead`): 5090 GPU-kernel wall-time × "
        "BW ratio (16.19×) + 5090-derived CPU overhead. Captures all the "
        "overhead the 5090 actually paid; assumes that overhead profile "
        "transfers to edge silicon (probably pessimistic since edge ARM "
        "+ tightly-integrated NPU may have lighter dispatch tax).\n\n"
        "**Real edge latency sits BETWEEN the two.** The bake-off projection "
        "is the more conservative (slower) estimate and is what the deck + "
        "sizer use as the headline FPS. The BW floor is the engineering "
        "lower bound — useful for \"is this workload BW-bound or compute-"
        "bound?\" questions.\n\n"
    )

    canonical = [
        "yolo_seg_fp8_trt",
        "yolo_seg_fp16_trt",
        "clip_trt",
        "sam3_bf16_reference",
    ]
    # Build lookup: workload_id → (bw_floor_ms, source_bakeoff)
    ncu_by_id = {
        w["workload_id"]: w
        for w in bundle.get("ncu", {}).get("workloads", [])
    }
    # Build lookup for bake-off projections at 720p, primary recipe
    # Note: clip_trt uses `projected_clip_ms_edge` (different schema than yolo)
    bake = bundle.get("bakeoffs", {})
    proj_lookup = {
        "yolo_seg_fp8_trt":              ("trt_yolo_edge_projection",
                                           "projections.720p.fp8",
                                           "projected_ms_edge"),
        "yolo_seg_fp16_trt":             ("trt_yolo_edge_projection",
                                           "projections.720p.fp16",
                                           "projected_ms_edge"),
        "yolo_seg_yolov8n-seg_fp8_trt":  ("trt_yolo_yolov8n-seg_edge_projection",
                                           "projections.720p.fp8",
                                           "projected_ms_edge"),
        "clip_trt":                      ("trt_clip_edge_projection",
                                           "projections.720p.fp8",
                                           "projected_clip_ms_edge"),
    }

    def _walk(d: dict, path: str) -> Any:
        cur: Any = d
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    headers = ["Workload (720p)",
               "DRAM MB/fwd",
               "BW floor ms (Mid)",
               "Effective edge ms (Mid, w/ overhead)",
               "Overhead ratio",
               "5090 ms (ref)"]
    rows = []
    for wid in canonical:
        w = ncu_by_id.get(wid)
        if not w:
            continue
        bw_floor = w.get("edge_projection_npu_mid", {}).get(
            "bw_floor_ms_npu_mid",
            w.get("edge_projection_npu_mid", {}).get("bw_bound_ms_min"),
        )
        dram_mb = w.get("per_forward", {}).get("dram_mb")
        ms_5090 = w.get("per_forward", {}).get("gpu_ms_5090")
        # Look up the bake-off projection for an effective edge ms
        eff_ms = None
        if wid in proj_lookup:
            bo_key, path, ms_field = proj_lookup[wid]
            cell = _walk(bake.get(bo_key, {}), path)
            if isinstance(cell, dict):
                eff_ms = cell.get(ms_field)
                if not ms_5090:
                    ms_5090 = cell.get("mean_frame_ms_5090")
                    if ms_5090 is None and isinstance(cell.get("per_frame_ms_5090"), dict):
                        ms_5090 = cell["per_frame_ms_5090"].get("p50")
        # Compute overhead ratio
        overhead_ratio = None
        if isinstance(eff_ms, (int, float)) and isinstance(bw_floor, (int, float)) and bw_floor > 0:
            overhead_ratio = round(eff_ms / bw_floor, 2)
        rows.append([
            wid,
            dram_mb,
            bw_floor,
            eff_ms if eff_ms is not None else "—",
            overhead_ratio if overhead_ratio is not None else "—",
            ms_5090 if ms_5090 is not None else "—",
        ])
    md += _md_table(headers, rows)
    md += (
        "**Reading the table:** `Overhead ratio = effective_edge_ms / "
        "bw_floor_ms`. Ratios near 1 mean the workload is genuinely BW-"
        "bound — the floor is achievable. Ratios > 5× mean overhead "
        "dominates — small dense models with complex graphs at 200 MB "
        "DRAM/forward range suffer here. Ratios > 10× usually mean "
        "kernel-launch-bound at small parameter count.\n\n"
        "The `yolo_seg_fp16_trt` row at 23.1× overhead ratio is the "
        "specific 22.7× discrepancy reviewer flagged in KH-P0-001. The "
        "two methodologies were both surfaced under names that suggested "
        "they were the same thing; they aren't.\n\n"
        "The `sam3_bf16_reference` row is the inverse case: at 119 GB "
        "DRAM/forward, even the BW floor (1265 ms) is nowhere near "
        "real-time. Overhead doesn't matter when DRAM bytes are the "
        "binding constraint by 100×. (No bake-off projection because we "
        "don't deploy SAM 3 — the BW floor alone tells the story.)\n\n"
        "**Workloads NOT in this table:**\n"
        "- `yoloe26_*`, `efficientsam3_*`, `efficientsam3p1_*`: ncu floor "
        "exists, but their bake-off summary `per_frame_ms_5090` fields are "
        "suspected microseconds-mislabeled-as-ms (a known stale-unit data-"
        "hygiene issue called out in earlier session memory). Reconciliation "
        "skipped until those summaries are re-baked or unit-fixed.\n"
        "- `yolo_seg_yolov8n-seg_fp8_trt`: bake-off projection methodology "
        "inherits its BW-bound baseline from the larger yolo11s-seg "
        "yolo_conv_quant bake-off, so its `projected_ms_edge` doesn't "
        "reflect yolov8n's actual ~2× lighter ncu DRAM (106 MB vs 217 MB "
        "per forward). The projection number is wrong for this row even "
        "though the underlying 5090 measurement is correct. Methodology "
        "finding worth flagging — the projection scheme assumes BW-bound "
        "scaling identical across model variants of similar architecture, "
        "which doesn't hold when the variant changes DRAM bytes by 2×.\n\n"
    )
    return md


def _render_ncu(ncu: dict) -> str:
    """Render the ncu sizer_bundle as a sorted DRAM-per-forward table.

    The columns now make the floor framing explicit (KH-P0-001) — these are
    BW floors, not realistic edge-latency predictions. See § 5 reconciliation.
    """
    if not ncu or "workloads" not in ncu:
        return "_(no ncu data)_\n\n"
    headers = [
        "Workload", "Source bake-off", "DRAM MB/fwd",
        "BW floor ms @ NPU Mid", "BW floor FPS (max) @ NPU Mid", "n forwards",
    ]
    rows = []
    for w in ncu["workloads"]:
        per_fwd = w.get("per_forward", {})
        proj = w.get("edge_projection_npu_mid", {})
        rows.append([
            w.get("workload_id", "—"),
            w.get("source_bakeoff", "—"),
            per_fwd.get("dram_mb"),
            proj.get("bw_floor_ms_npu_mid", proj.get("bw_bound_ms_min")),
            proj.get("bw_floor_fps_max_npu_mid", proj.get("bw_bound_fps_max")),
            w.get("n_forwards", "—"),
        ])
    rows.sort(key=lambda r: (r[2] if isinstance(r[2], (int, float)) else 1e18))
    md = _md_h(3, "ncu-measured DRAM per forward (sorted ascending by MB/fwd)")
    md += (
        "**These are BW floors — pure DRAM-bytes / effective-BW minima, no "
        "overhead.** Real edge latency always exceeds these floors (kernel-"
        "launch tax, NMS dispatch, memory hierarchy stalls, sync). For the "
        "5090-wall-time-derived projections that include overhead, see the "
        "bake-off summaries in § 3 (field: `effective_edge_ms_with_overhead`). "
        "Per-workload reconciliation: § 5.\n\n"
    )
    md += (
        f"_NPU Mid effective BW: "
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

    md += _md_h(2, "4. Reconciliation: BW floor vs effective edge ms")
    md += _render_bw_reconciliation(bundle)

    md += _md_h(2, "5. Dtype gating per tier")
    md += (
        "Per-recipe projection cells in the JSON now carry "
        "`dtype_mismatch_on_mid`, `deployable_tiers`, and "
        "`dtype_mismatch_reason` fields (schema v3, KH-P0-002). The matrix:\n\n"
    )
    md += _md_table(
        ["NPU tier", "Supported dtypes"],
        [[t, ", ".join(d)] for t, d in TIER_DTYPE_SUPPORT.items()],
    )
    md += (
        "- **INT8 recipes** (`int8`, `int8_1x1_swap`, etc.) deploy on every "
        "NPU tier and the 5090.\n"
        "- **FP-class recipes** (`fp8`, `fp16`, `bf16`, `fp32`) deploy only "
        "on NPU High and the 5090. NPU Mid is INT8-only (200 TOPS, no FP "
        "path). FP-class recipes flagged `dtype_mismatch_on_mid=True` "
        "project to High at the same BW ceiling (BW-equal at stock "
        "LPDDR5X-8.4 memory class).\n"
        "- **Historical FP-on-Mid raw projection numbers are preserved** in "
        "the JSON alongside the gating flag — the reviewer's KH-P0-002 "
        "guidance is to render dtype mismatch as a flag, not delete data.\n"
        "- This matrix doesn't apply to LLM bake-offs: Q4_K_M GGUF runs "
        "INT8-native at runtime regardless of `compute_dtype` labels in "
        "the per-quant JSONs.\n\n"
    )

    md += _md_h(2, "6. Methodology notes")
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
        "is dtype gating (see § 4).\n"
        "- **Quality metrics** (`box_recall*`, `mean_matched_iou*`) measure "
        "quantization drift vs the FP16 TRT engine on the same input frames "
        "— NOT vs ground-truth labels. Aliased field names "
        "`*_vs_fp16_engine` make this explicit (KH-P0-003).\n"
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
            "schema_version": 4,
            "schema_v2_changes": (
                "Bake-off quality fields gain `box_recall_vs_fp16_engine` and "
                "`mean_matched_iou_vs_fp16_engine` aliases. These measure "
                "quantization drift relative to the FP16 TRT engine on the "
                "same input frames — NOT vs ground-truth labels. Legacy "
                "field names (`box_recall`, `mean_matched_iou`) preserved "
                "as aliases for one cycle. See KH-P0-003 in REMEDIATION_PLAN.md."
            ),
            "schema_v3_changes": (
                "Per-recipe projection cells gain `dtype_mismatch_on_mid`, "
                "`deployable_tiers`, and (where applicable) "
                "`dtype_mismatch_reason` fields. NPU Mid is INT8-only "
                "(200 TOPS, no FP path); FP-class recipes (fp16, bf16, "
                "fp8, fp32) are flagged dtype_mismatch_on_mid=True and "
                "project to NPU High instead. Historical raw FP-on-Mid "
                "projection numbers are preserved alongside the flag for "
                "audit trail per the reviewer's KH-P0-002 guidance. See "
                "REMEDIATION_PLAN.md."
            ),
            "schema_v4_changes": (
                "Reconciliation between two BW estimates that previously "
                "appeared as conflicting numbers in user-facing artifacts: "
                "(a) ncu's `bw_bound_ms_min` (now aliased as "
                "`bw_floor_ms_npu_mid`) is a pure DRAM-bytes/effective-BW "
                "FLOOR — best-case minimum, no overhead. "
                "(b) bake-off's `projected_ms_edge` (now aliased as "
                "`effective_edge_ms_with_overhead`) is a 5090-wall-time-"
                "derived projection that includes overhead (kernel-launch "
                "tax, NMS dispatch, memory hierarchy, sync). Reviewer "
                "(KH-P0-001) caught a 22.7× discrepancy on yolo_seg_fp16_trt "
                "because the two methodologies were both surfaced under "
                "names that suggested they were the same thing. Real edge "
                "latency sits BETWEEN the two: above the BW floor (always), "
                "below the bake-off projection (only if edge silicon's "
                "overhead profile is lighter than 5090's). See § 8 of "
                "CLAUDE_REVIEW_BRIEFING.md for per-workload reconciliation "
                "table."
            ),
            "tier_dtype_support": TIER_DTYPE_SUPPORT,
            "methodology_version": "2026-05-11-substring-arc-closed",
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
