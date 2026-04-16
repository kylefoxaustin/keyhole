"""
Keyframe debouncing bake-off — roadmap #5.

Post-processes the existing Hybrid V2 bake-off data (FP8 variant) to ask two
independent questions:

  1. Latency math: if CLIP only runs every Nth native-frame and YOLO runs
     every frame, what is the effective edge FPS as a function of N?
  2. Stability: how often does CLIP's top-1 concept for a given detection
     still agree with what CLIP would emit on a later frame? This tells us
     how large N can be before the cached tag is stale.

No new inference — reads data/output/bakeoff/hybrid_v2_*.json (the FP8 run
gives us per-frame per-detection top-1 concepts, which is exactly the data
we need) and emits:

  data/output/bakeoff/keyframe_debounce_summary.json

Native video fps is assumed to be 30, matching the cached bake-off frames
(sampled at 1 fps i.e. every 30th native frame).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("debounce")

BAKEOFF_DIR = REPO_ROOT / "data" / "output" / "bakeoff"
NATIVE_FPS = 30
SAMPLE_FPS = 1            # cached frames are sampled every 30th native frame
SAMPLE_STRIDE = NATIVE_FPS // SAMPLE_FPS
N_VALUES = [1, 2, 5, 10, 15, 30, 60, 90]   # native-frame keyframe intervals
IOU_MATCH_THRESHOLD = 0.5
BASELINE_RECIPE = "fp8"   # the edge-deployed variant


def bbox_iou(a, b) -> float:
    """IoU of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def match_and_score(det_a, det_b) -> tuple[int, float, int]:
    """Pair det_a with closest-IoU det_b (same class_name, IoU > threshold);
    return (top1_matches, sum_top3_jaccard, pairs_considered)."""
    pairs, matches = 0, 0
    jaccard_sum = 0.0
    used_b = set()
    for i, da in enumerate(det_a):
        best, best_iou = -1, IOU_MATCH_THRESHOLD
        for j, db in enumerate(det_b):
            if j in used_b:
                continue
            if da.get("class_name") != db.get("class_name"):
                continue
            iou = bbox_iou(da["bbox"], db["bbox"])
            if iou > best_iou:
                best, best_iou = j, iou
        if best >= 0:
            used_b.add(best)
            pairs += 1
            if da.get("top_concept") == det_b[best].get("top_concept"):
                matches += 1
            s_a = {c[0] for c in da.get("top3", [])}
            s_b = {c[0] for c in det_b[best].get("top3", [])}
            if s_a or s_b:
                jaccard_sum += len(s_a & s_b) / len(s_a | s_b)
    return matches, jaccard_sum, pairs


def stability_at_gap(frames: list[dict], gap_sampled: int) -> tuple[float, float, int]:
    """Top-1 agreement + top-3 Jaccard across all sampled-frame pairs at this gap."""
    tot_matches, tot_pairs = 0, 0
    tot_jaccard = 0.0
    for i in range(len(frames) - gap_sampled):
        m, j, p = match_and_score(frames[i]["detections"],
                                  frames[i + gap_sampled]["detections"])
        tot_matches += m
        tot_jaccard += j
        tot_pairs += p
    if not tot_pairs:
        return 1.0, 1.0, 0
    return tot_matches / tot_pairs, tot_jaccard / tot_pairs, tot_pairs


def effective_latency(yolo_ms: float, clip_ms: float, N: int) -> float:
    """Per-frame wall time if CLIP runs 1/N frames and YOLO every frame.

    Real-time pipeline amortizes CLIP over N frames; this is the steady-state
    average per-frame cost seen by the downstream consumer."""
    return yolo_ms + clip_ms / N


def main():
    edge_proj = json.loads((BAKEOFF_DIR / "hybrid_v2_edge_projection.json").read_text())
    summary = json.loads((BAKEOFF_DIR / "hybrid_v2_summary.json").read_text())

    out = {
        "native_fps_assumed": NATIVE_FPS,
        "sample_stride": SAMPLE_STRIDE,
        "iou_threshold": IOU_MATCH_THRESHOLD,
        "baseline_recipe": BASELINE_RECIPE,
        "per_resolution": {},
        "method": (
            "Post-processes the FP8 Hybrid V2 bake-off. Edge FPS(N) derived "
            "analytically: YOLO runs every native frame, CLIP every Nth. "
            "Stability(N) measured empirically from CLIP top-1 concept "
            "agreement across sampled-frame pairs at native gap N; IoU-matched "
            "detections only, same class_name, IoU > 0.5."
        ),
    }

    for res, clip_stem in [
        ("720p", "720p_EW_clip"),
        ("1080p", "embedded_world_clip_1080p"),
        ("4K", "embedded_world_clip"),
    ]:
        p_bf16 = edge_proj["projections"][res]["bf16"]
        p_fp8 = edge_proj["projections"][res]["fp8"]
        yolo_5090 = p_fp8["mean_yolo_ms_5090"]
        clip_5090 = p_fp8["mean_clip_ms_5090"]
        yolo_edge = p_fp8["projected_yolo_ms_edge"]
        clip_edge = p_fp8["projected_clip_ms_edge"]
        fp8_frames = summary[res][BASELINE_RECIPE]["frames"]

        rows = []
        for N in N_VALUES:
            # Stability: closest available sampled gap to native gap N is round(N/stride),
            # with at least 1 (can't pair a frame with itself at gap 0).
            gap_sampled = max(1, round(N / SAMPLE_STRIDE))
            gap_real_sec = N / NATIVE_FPS
            stability_top1, stability_top3, n_pairs = stability_at_gap(fp8_frames, gap_sampled)

            # Effective latency and FPS
            eff_5090_ms = effective_latency(yolo_5090, clip_5090, N)
            eff_edge_ms = effective_latency(yolo_edge, clip_edge, N)
            eff_edge_fps = 1000.0 / eff_edge_ms if eff_edge_ms > 0 else 0.0

            rows.append({
                "N_native_frames": N,
                "keyframe_interval_sec": gap_real_sec,
                "gap_sampled_used": gap_sampled,
                "stability_top1": stability_top1,
                "stability_top3_jaccard": stability_top3,
                "n_pairs": n_pairs,
                "eff_5090_ms_per_frame": eff_5090_ms,
                "eff_edge_ms_per_frame": eff_edge_ms,
                "eff_edge_fps": eff_edge_fps,
            })

        # Also record the endpoints for reference
        out["per_resolution"][res] = {
            "yolo_ms_5090": yolo_5090,
            "clip_ms_5090": clip_5090,
            "yolo_ms_edge": yolo_edge,
            "clip_ms_edge": clip_edge,
            "edge_fps_N1_fp8": p_fp8["projected_fps_edge"],     # every-frame CLIP baseline
            "edge_fps_yolo_only": 1000.0 / yolo_edge if yolo_edge > 0 else 0.0,
            "rows": rows,
        }

    out_path = BAKEOFF_DIR / "keyframe_debounce_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)

    # Pretty print
    print()
    for res in ["720p", "1080p", "4K"]:
        data = out["per_resolution"][res]
        print(f"=== {res} ===  YOLO {data['yolo_ms_edge']:.1f} ms edge  |  "
              f"CLIP {data['clip_ms_edge']:.1f} ms edge  |  "
              f"N=1 baseline {data['edge_fps_N1_fp8']:.2f} FPS  |  "
              f"YOLO-only ceiling {data['edge_fps_yolo_only']:.1f} FPS")
        print(f"  {'N':>3s}  {'sec':>5s}  {'stb1':>5s}  {'stb3':>5s}  {'pairs':>5s}  "
              f"{'5090 ms':>8s}  {'edge ms':>8s}  {'edge FPS':>8s}")
        for r in data["rows"]:
            print(f"  {r['N_native_frames']:>3d}  "
                  f"{r['keyframe_interval_sec']:>5.2f}  "
                  f"{r['stability_top1']:>5.2f}  "
                  f"{r['stability_top3_jaccard']:>5.2f}  "
                  f"{r['n_pairs']:>5d}  "
                  f"{r['eff_5090_ms_per_frame']:>8.2f}  "
                  f"{r['eff_edge_ms_per_frame']:>8.1f}  "
                  f"{r['eff_edge_fps']:>8.2f}")
        print()


if __name__ == "__main__":
    main()
