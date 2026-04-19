"""
Consolidate data/output/ncu/*.json into a single sizer-consumable JSON.

The ncu JSONs aggregate per-range metrics across a full bake-off run
(all frames × all resolutions × sometimes all recipes collapsed into
one NVTX bucket). To feed the sizer's platform-budget projections we
need PER-FORWARD numbers. This script applies documented divisors
(derived from each bake-off's WARMUP/MAX_FRAMES/recipes constants) to
produce both totals and per-forward values.

Output: data/output/ncu/sizer_bundle.json
  - One entry per NVTX range across all 6 workload JSONs
  - Each entry has: aggregate, per_forward, divisor metadata, provenance
  - Edge projections for NPU Mid (134.4 GB/s LPDDR5X) precomputed as
    illustration — the sizer can recompute for any NPU tier from the
    per_forward.dram_bytes_total alone.

Usage:
    python scripts/export_ncu_for_sizer.py
    python scripts/export_ncu_for_sizer.py --ncu-dir data/output/ncu --out sizer_bundle.json

Downstream consumer:
    keyhole-sizer/sizer/platform_budget.py can load this and substitute
    measured ss_ddr_gbs_avg / ss_tops_avg values for the currently
    approximated ones (see PROFILE_NCU.md § Sizer integration plan).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


# ───────────────────── Divisors: range → n_forwards ─────────────────────
#
# Each divisor is the number of NVTX range pushes that landed in that
# bucket during the profiled bake-off run. Derived from each bake-off
# script's constants. Keep in sync if a bake-off changes.
#
# Convention:
#   RES = number of resolutions the bake-off sweeps (typically 3: 720p,
#         1080p, 4K)
#   WARMUP = frames discarded for cuDNN/TRT autotune (NVTX still wraps
#            them because ncu sees every kernel launch inside the range)
#   TIMED  = frames used for stable p50/p95 measurement
#   FRAMES_PER_RES = WARMUP + TIMED
#   RECIPES = number of recipes per frame (e.g. FP16/FP8 for TRT)

NVTX_DIVISORS: dict[str, dict[str, int]] = {
    # ─── trt_yolo: 14 frames × 3 res = 42 per recipe range ──────────
    # No in-loop warmup discard (one-shot warmup happens outside NVTX).
    # Each of {fp16, int8, fp8} gets its own range. INT8 dropped by ncu.
    "yolo_seg_fp16_trt":   {"source": "trt_yolo",       "n_forwards": 42},
    "yolo_seg_fp8_trt":    {"source": "trt_yolo",       "n_forwards": 42},
    # ─── trt_clip: 14 frames × 3 res × 2 NVTX-wrapped recipes (FP16 + FP8) ─
    # = 84. BF16 torch path has no NVTX. Single "clip_trt" range covers
    # both quants. Each frame typically fits in one TRT batch
    # (max_bsz >= n_crops), so ~1 NVTX push per frame.
    "clip_trt":            {"source": "trt_clip",       "n_forwards": 84},
    # ─── sam_variants: 14 sampled frames × ~13 boxes each per contestant ─
    # The bake-off iterates 14 frames; per frame it runs ALL boxes
    # (median 13) for each contestant. NVTX wraps the whole box-predict
    # call per frame, so n_forwards = 14 per contestant range.
    "mobilesam":           {"source": "sam_variants",   "n_forwards": 14},
    "efficientsam_tiny":   {"source": "sam_variants",   "n_forwards": 14},
    "efficientsam_small":  {"source": "sam_variants",   "n_forwards": 14},
    "yolo_seg":            {"source": "sam_variants",   "n_forwards": 14},
    # Dropped by relaxed-name matching: sam3_bf16_reference (see § 9 of
    # NCU_EXPLAINED.md). We don't have data for it.
    # ─── efficientsam3: 3 res × (2 WARMUP + 10 MAX) = 36 frames, 1 range ─
    "efficientsam3_es_ev_s": {"source": "efficientsam3", "n_forwards": 36},
    # ─── yoloe26: 3 res × (2 WARMUP + 10 MAX) = 36 per variant ───────
    "yoloe26_text_prompt_s":  {"source": "yoloe26",     "n_forwards": 36},
    "yoloe26_prompt_free_s":  {"source": "yoloe26",     "n_forwards": 36},
    # ─── trt_yoloe26: 3 res × (3 WARMUP + 10 TIMED) = 39 per recipe ─
    "yoloe26_pytorch_fp16":   {"source": "trt_yoloe26", "n_forwards": 39},
    "yoloe26_trt_fp16":       {"source": "trt_yoloe26", "n_forwards": 39},
    "yoloe26_trt_fp8":        {"source": "trt_yoloe26", "n_forwards": 39},
}


# ─────────────────── NPU tier specs (for edge projection) ───────────────
# Kyle-supplied authoritative NPU vendor numbers. See
# memory/project_current_state.md § NPU tier actuals.

NPU_MID_EFFECTIVE_GBS = 100.8  # 134.4 theoretical × 0.75 efficiency


# ───────────────────────────── Builders ─────────────────────────────────

def _edge_projection_bw_bound(dram_bytes_per_forward: float, npu_gbs: float) -> dict:
    """Pure bandwidth-bound minimum latency on a given NPU tier.

    NOTE: this is the LOWER BOUND for latency (UPPER BOUND for FPS).
    Real edge latency = max(bw_bound_ms, compute_bound_ms). For
    compute-heavy models (large vision transformers) this projection
    is loose and the edge FPS will be lower than implied here.
    """
    bw_bound_ms = (dram_bytes_per_forward / 1e9) / npu_gbs * 1000.0
    return {
        "bw_bound_ms_min":         round(bw_bound_ms, 2),
        "bw_bound_fps_max":        round(1000.0 / bw_bound_ms, 1) if bw_bound_ms > 0 else 0.0,
        "npu_effective_gbs":       npu_gbs,
        "interpretation": (
            "Floor on edge latency from DRAM bandwidth alone. Real edge "
            "fps = min(bw_bound_fps_max, compute_bound_fps). Compute "
            "bound is NOT computed here — derive from per_forward.tc_ops_blackwell "
            "with a Blackwell-to-NPU TOPS conversion factor, or check "
            "non-profiled bake-off measured ms × edge-scaling."
        ),
    }


def _load_ncu_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _range_entry(nvtx_label: str, info: dict, divisor_meta: dict, source_path: Path) -> dict:
    m = info["metrics"]
    n_fwd = divisor_meta["n_forwards"]

    dram_total = m.get("dram__bytes.sum", 0.0)
    tc_total   = m.get("sm__inst_executed_pipe_tensor.sum", 0.0)
    sass_total = m.get("sm__sass_thread_inst_executed.sum", 0.0)
    smsp_total = m.get("smsp__inst_executed.sum", 0.0)
    gpu_ns_total = m.get("gpu__time_duration.sum", 0.0)

    dram_per_fwd = dram_total / n_fwd if n_fwd else 0.0
    tc_per_fwd   = tc_total / n_fwd   if n_fwd else 0.0
    sass_per_fwd = sass_total / n_fwd if n_fwd else 0.0
    smsp_per_fwd = smsp_total / n_fwd if n_fwd else 0.0

    return {
        "workload_id":   nvtx_label,
        "source_bakeoff": divisor_meta["source"],
        "source_json":   str(source_path),
        "n_forwards":    n_fwd,
        "n_kernels_total": info.get("n_kernel_invocations", 0),
        "aggregate": {
            "dram_bytes_total":         dram_total,
            "dram_bytes_read_total":    m.get("dram__bytes_read.sum", 0.0),
            "dram_bytes_write_total":   m.get("dram__bytes_write.sum", 0.0),
            "tc_ops_total":             tc_total,
            "sass_thread_inst_total":   sass_total,
            "smsp_inst_total":          smsp_total,
            "gpu_ns_total_ncu_inflated": gpu_ns_total,
        },
        "per_forward": {
            # HARDWARE-NEUTRAL — transfers to edge NPU projections
            "dram_bytes":              round(dram_per_fwd, 1),
            "dram_mb":                 round(dram_per_fwd / 1e6, 2),
            # 5090-SPECIFIC — use as relative comparisons only
            "tc_ops_blackwell":        round(tc_per_fwd, 1),
            "sass_thread_inst":        round(sass_per_fwd, 1),
            "smsp_inst":               round(smsp_per_fwd, 1),
        },
        "edge_projection_npu_mid": _edge_projection_bw_bound(
            dram_per_fwd, NPU_MID_EFFECTIVE_GBS),
        "notes": [
            "Divisor n_forwards derived from bake-off constants (see script).",
            "dram_bytes is workload property — transfers to any NPU.",
            "tc_ops is Blackwell HMMA count — use as relative only.",
            "gpu_ns is ncu-inflated (serialized launches); use non-profiled "
            "bake-off JSON for real wall-clock.",
        ],
    }


def build_bundle(ncu_dir: Path) -> dict:
    entries = []
    seen_labels = set()
    missing_from_ncu = []

    for json_path in sorted(ncu_dir.glob("*.json")):
        if json_path.name == "sizer_bundle.json":
            continue
        try:
            d = _load_ncu_json(json_path)
        except Exception as e:
            print(f"  skip {json_path.name}: {e}")
            continue

        for nvtx_label, info in d.get("by_range", {}).items():
            if nvtx_label == "[unattributed]":
                continue
            divisor_meta = NVTX_DIVISORS.get(nvtx_label)
            if divisor_meta is None:
                print(f"  WARN: no divisor mapping for {nvtx_label!r} in {json_path.name}")
                continue
            entries.append(_range_entry(nvtx_label, info, divisor_meta, json_path))
            seen_labels.add(nvtx_label)

    for expected_label in NVTX_DIVISORS:
        if expected_label not in seen_labels:
            missing_from_ncu.append(expected_label)

    return {
        "bundle_version":      "1",
        "description": (
            "Keyhole ncu measurements, per-workload, normalized per-forward. "
            "Consumed by keyhole-sizer/sizer/platform_budget.py as MEASURED "
            "replacement for the approximated ss_ddr_gbs_avg / ss_tops_avg "
            "columns. Each entry covers one NVTX range from a bake-off run."
        ),
        "export_timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ncu_binary":           "Nsight Compute 2026.1.1",
        "measurement_host":     "NVIDIA RTX 5090 (Blackwell sm_120), Ubuntu 24.04",
        "npu_tier_used_for_projection": {
            "name":            "NPU Mid",
            "bus":             "128-bit LPDDR5X @ 8.4 GT/s",
            "theoretical_gbs": 134.4,
            "effective_gbs":   NPU_MID_EFFECTIVE_GBS,
            "efficiency_assumed": 0.75,
        },
        "n_workloads": len(entries),
        "workloads":   entries,
        "missing_from_ncu": missing_from_ncu,
        "known_gaps": [
            "sam3_bf16_reference: dropped by ncu --app-replay-match name "
            "(SAM 3's kernel set varied across passes).",
            "efficientsam3p1: target SIGKILL'd on app-replay pass 2; "
            "retry with kernel-replay if needed.",
            "llm: skipped for this sweep (Qwen3-30B-A3B under kernel-replay "
            "would add ~4 hrs; see PROFILE_NCU.md § B-prime rationale).",
            "yolo_seg_int8_trt: ncu couldn't profile Blackwell INT8 TRT "
            "kernels cleanly; absent from JSON (not zeros).",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("--ncu-dir", type=Path, default=Path("data/output/ncu"),
                    help="Directory containing the 6 ncu JSONs")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: <ncu-dir>/sizer_bundle.json)")
    args = ap.parse_args()

    out_path = args.out or (args.ncu_dir / "sizer_bundle.json")
    bundle = build_bundle(args.ncu_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2))

    print(f"\nWrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"  workloads:       {bundle['n_workloads']}")
    if bundle["missing_from_ncu"]:
        print(f"  missing_from_ncu: {bundle['missing_from_ncu']}")

    # Human summary table
    print("\n=== Per-forward summary (BW-bound = upper-bound FPS only) ===")
    print(f"{'workload':<32s} {'DRAM/fwd':>10s} {'NPU-Mid ms':>12s} {'max FPS':>8s}")
    for w in bundle["workloads"]:
        print(f"{w['workload_id']:<32s} "
              f"{w['per_forward']['dram_mb']:>9.2f}MB "
              f"{w['edge_projection_npu_mid']['bw_bound_ms_min']:>10.2f}ms "
              f"{w['edge_projection_npu_mid']['bw_bound_fps_max']:>8.1f}")


if __name__ == "__main__":
    main()
