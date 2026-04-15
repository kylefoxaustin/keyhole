"""
Edge-latency projection for the mask-model bake-off.

Inputs:
  data/output/bakeoff/{clip}/results/*.json  (from bakeoff_sam_variants.py)

Adds:
  - Measured GFLOPs per contestant (once, on a 720p frame with ~13 boxes)
  - Edge projection (134.4 GB/s LPDDR5X, 25 TOPS bf16) via NPUEmulator

Outputs:
  data/output/bakeoff/edge_projection.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.emulate.npu_emulator import (
    NPUEmulator, WorkloadProfile, RTX_5090, EDGE_MPU_TARGET,
)
# Reuse contestant classes
from scripts.bakeoff_sam_variants import (
    MobileSAMContestant, EfficientSAMContestant, YoloSegContestant,
    BAKEOFF_DIR, sync_cuda,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("edge_proj")

CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}

REF_FRAME_CLIP = "720p_EW_clip"  # Use for GFLOPs measurement


def load_sample_frame(clip_stem: str, idx: int = 0):
    out_dir = BAKEOFF_DIR / clip_stem
    frames = json.loads((out_dir / "frames.json").read_text())
    prompts = json.loads((out_dir / "prompts.json").read_text())
    f = frames[idx]
    img = cv2.imread(str(out_dir / f["path"]))
    boxes = [p["box"] for p in prompts[str(f["idx"])]]
    return img, boxes, len(boxes)


def build_projection():
    contestants = ["mobilesam", "efficientsam_tiny", "efficientsam_small", "yolo_seg"]

    # Skip per-contestant GFLOPs measurement — FlopCounterMode clashes with MobileSAM's
    # internal @no_grad decorator. Let the emulator use its transformer-heuristic
    # fallback (15% compute / 85% bandwidth split), consistent with the SAM 3 analysis.
    emu = NPUEmulator(reference=RTX_5090, target=EDGE_MPU_TARGET)

    out = {
        "reference": RTX_5090.to_dict(),
        "target": EDGE_MPU_TARGET.to_dict(),
        "note": "GFLOPs omitted; projection uses emulator's 15%/85% compute/bandwidth split (matches SAM 3 analysis).",
        "projections": {},
    }

    for res, clip_stem in CLIPS.items():
        clip_dir = BAKEOFF_DIR / clip_stem
        out["projections"][res] = {}
        for name in contestants:
            data = json.loads((clip_dir / "results" / f"{name}.json").read_text())
            gflops = 0.0
            # Compute mean full-frame latency (only frames with boxes)
            frame_ms = [fr["latency_ms"] for fr in data["frames"] if fr["n_boxes"] > 0]
            if not frame_ms:
                continue
            mean_frame_ms = float(np.mean(frame_ms))
            params_m = data["params_m"]
            vram_mb = data["peak_vram_mb"]
            # Model size at bf16 (2 bytes/param)
            model_bytes = int(params_m * 1e6 * 2)
            # Activation bytes: peak VRAM - model (floor at 0)
            act_bytes = max(0, int(vram_mb * 1e6 - model_bytes))

            wl = WorkloadProfile(
                stage_name=f"{name}_{res}",
                model_name=name,
                param_count=int(params_m * 1e6),
                model_size_bytes=model_bytes,
                precision="bf16",
                gflops_per_inference=gflops,
                measured_latency_ms=mean_frame_ms,
                measured_gpu_kernel_ms=mean_frame_ms,  # for small per-frame work, ≈ kernel time
                measured_gpu=RTX_5090.name,
                measured_peak_vram_bytes=int(vram_mb * 1e6),
                peak_activation_bytes=act_bytes,
            )
            r = emu.project_workload(wl)
            edge_fps = 1000.0 / r.projected_latency_ms if r.projected_latency_ms > 0 else 0
            out["projections"][res][name] = {
                "params_m": params_m,
                "gflops_per_frame": gflops,
                "measured_frame_ms_5090": mean_frame_ms,
                "measured_fps_5090": 1000.0 / mean_frame_ms,
                "projected_frame_ms_edge": r.projected_latency_ms,
                "projected_fps_edge": edge_fps,
                "compute_limited_ms": r.compute_limited_ms,
                "bandwidth_limited_ms": r.bandwidth_limited_ms,
                "bottleneck": r.bottleneck,
                "fits_in_memory": r.fits_in_memory,
                "memory_headroom_mb": r.memory_headroom_bytes / 1e6,
            }
            log.info(
                "%-20s %-5s  5090 %6.2fms (%5.1f fps)  ->  edge %6.1fms (%4.1f fps) [%s]",
                name, res, mean_frame_ms, 1000.0/mean_frame_ms,
                r.projected_latency_ms, edge_fps, r.bottleneck,
            )

    out_path = BAKEOFF_DIR / "edge_projection.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)

    # Pretty table
    print()
    print(f"{'Model':22s} | {'Res':4s} | {'5090 ms':>8s} {'5090 FPS':>9s} | {'Edge ms':>8s} {'Edge FPS':>9s} | Bottleneck")
    print("-"*95)
    for res in CLIPS:
        for name in contestants:
            if name in out["projections"][res]:
                p = out["projections"][res][name]
                print(f"{name:22s} | {res:4s} | {p['measured_frame_ms_5090']:8.2f} {p['measured_fps_5090']:9.1f} | "
                      f"{p['projected_frame_ms_edge']:8.1f} {p['projected_fps_edge']:9.1f} | {p['bottleneck']}")
        print()


if __name__ == "__main__":
    build_projection()
