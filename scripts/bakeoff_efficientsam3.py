"""
EfficientSAM3 bake-off — the community-distilled "SAM 3 Lite" (ES-EV-S variant).

The project's main bake-off harness (bakeoff_sam_variants.py) runs in the keyhole
venv (PyTorch 2.11 / Python 3.10). EfficientSAM3 requires Python ≥3.12, so this
script runs in a separate uv-managed venv at `.venv-es3/` (PyTorch 2.11+cu130 to
keep Blackwell support).

Reuses the cached frames + YOLO prompt boxes + SAM 3 reference masks the main
bake-off already produced under `data/output/bakeoff/{clip_stem}/`, so no
re-extraction is needed.

Output: `data/output/bakeoff/efficientsam3_summary.json` — per-resolution
latency + IoU-vs-SAM3 stats at 720p / 1080p / 4K on the RTX 5090.

Run from the repo root:
    .venv-es3/bin/python scripts/bakeoff_efficientsam3.py

Source:
    github.com/SimonZeng7108/efficientsam3 (Apache-2.0)
    checkpoint: stage1_all_converted/efficient_sam3_efficientvit_s.pt (~1.7 GB)
    smallest variant: EfficientViT-B0 vision backbone (~26M params), 424M total.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bakeoff_es3")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "efficientsam3"))
# Repo root on path so `from src.profiling.nvtx_helpers import ...` resolves
# when this script is run from the .venv-es3 interpreter.
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

CKPT = REPO_ROOT / "weights" / "efficientsam3" / "stage1_all_converted" / "efficient_sam3_efficientvit_s.pt"
BAKEOFF_DIR = REPO_ROOT / "data" / "output" / "bakeoff"

# Maps res label -> cache dir used by bakeoff_sam_variants.py
RESOLUTION_CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}

WARMUP_FRAMES = 2      # discard per resolution for cuDNN autotune
MAX_FRAMES = 10        # enough for a stable p50/p95


# ── metrics ──

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    u = ua + ub - inter
    return float(inter / u) if u > 0 else 0.0


@dataclass
class ResolutionReport:
    resolution: str
    clip: str
    n_frames_timed: int
    n_boxes_total: int
    per_frame_ms: list[float] = field(default_factory=list)
    per_box_ms: list[float] = field(default_factory=list)
    iou_samples: list[float] = field(default_factory=list)

    @property
    def mean_per_frame_ms(self) -> float:
        return float(np.mean(self.per_frame_ms)) if self.per_frame_ms else 0.0

    @property
    def p50_per_frame_ms(self) -> float:
        return float(np.percentile(self.per_frame_ms, 50)) if self.per_frame_ms else 0.0

    @property
    def p95_per_frame_ms(self) -> float:
        return float(np.percentile(self.per_frame_ms, 95)) if self.per_frame_ms else 0.0

    @property
    def mean_per_box_ms(self) -> float:
        return float(np.mean(self.per_box_ms)) if self.per_box_ms else 0.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.iou_samples)) if self.iou_samples else 0.0

    @property
    def median_iou(self) -> float:
        return float(np.median(self.iou_samples)) if self.iou_samples else 0.0


def load_prompts_and_refs(clip_stem: str):
    clip_dir = BAKEOFF_DIR / clip_stem
    prompts_raw = json.loads((clip_dir / "prompts.json").read_text())
    prompts = {int(k): v for k, v in prompts_raw.items()}
    refs_raw = json.loads((clip_dir / "refs_meta.json").read_text())
    refs = {int(k): v for k, v in refs_raw.items()}
    return clip_dir, prompts, refs


def run_resolution(model, processor, resolution: str, clip_stem: str) -> ResolutionReport:
    clip_dir, prompts, refs = load_prompts_and_refs(clip_stem)
    rep = ResolutionReport(resolution=resolution, clip=clip_stem,
                            n_frames_timed=0, n_boxes_total=0)
    frames_dir = clip_dir / "frames"

    ordered_frame_ids = sorted(prompts.keys())
    usable = [fid for fid in ordered_frame_ids if prompts[fid] and (frames_dir / f"frame_{fid:06d}.png").exists()]
    log.info("[%s] %d usable frames in cache (of %d with prompts)",
             resolution, len(usable), len(ordered_frame_ids))

    to_run = usable[: WARMUP_FRAMES + MAX_FRAMES]
    for i, fid in enumerate(to_run):
        is_warmup = i < WARMUP_FRAMES
        frame_path = frames_dir / f"frame_{fid:06d}.png"
        img_bgr = cv2.imread(str(frame_path))
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        ps = prompts[fid]
        boxes = np.array([p["box"] for p in ps], dtype=np.float32)

        from src.profiling.nvtx_helpers import nvtx_range
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx_range("efficientsam3_es_ev_s"):
            state = processor.set_image(img_pil)
            masks, scores, _ = model.predict_inst(
                state, point_coords=None, point_labels=None,
                box=boxes, multimask_output=False,
            )
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        tag = "WARMUP" if is_warmup else "TIMED"
        log.info("[%s][%s] frame %d: n=%d boxes, %.1f ms", resolution, tag, fid, len(boxes), ms)

        if is_warmup:
            continue

        rep.per_frame_ms.append(ms)
        rep.per_box_ms.extend([ms / max(1, len(boxes))] * len(boxes))
        rep.n_frames_timed += 1
        rep.n_boxes_total += len(boxes)

        # Normalize mask shape -> (N_prompts, H, W) bool, H×W matching image
        masks_np = masks if isinstance(masks, np.ndarray) else masks.cpu().numpy()
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]  # collapse singleton second dim
        H, W = img_bgr.shape[:2]

        # IoU vs SAM 3 references (only for prompt indices that had a reference match)
        ref_by_pi = {r["prompt_idx"]: r for r in refs.get(fid, [])}
        for pi, m in enumerate(masks_np):
            if pi not in ref_by_pi:
                continue
            m_bool = m.astype(bool)
            if m_bool.shape != (H, W):
                m_bool = cv2.resize(m_bool.astype(np.uint8), (W, H),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
            ref_path = clip_dir / ref_by_pi[pi]["mask_path"]
            ref_mask = np.load(ref_path).astype(bool)
            iou = mask_iou(m_bool, ref_mask)
            rep.iou_samples.append(iou)

    log.info(
        "[%s] %d frames timed, %d boxes, mean %.1f ms/frame (p50 %.1f, p95 %.1f), "
        "mean IoU %.3f (n=%d)",
        resolution, rep.n_frames_timed, rep.n_boxes_total,
        rep.mean_per_frame_ms, rep.p50_per_frame_ms, rep.p95_per_frame_ms,
        rep.mean_iou, len(rep.iou_samples),
    )
    return rep


def main():
    if not CKPT.exists():
        raise SystemExit(f"Checkpoint missing: {CKPT}")

    from sam3.model_builder import build_efficientsam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    log.info("Loading EfficientSAM3 (ES-EV-S)...")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    autocast_ctx.__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_efficientsam3_image_model(
        checkpoint_path=str(CKPT),
        backbone_type="efficientvit", model_name="b0",
        enable_inst_interactivity=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for n, p in model.named_parameters()
                         if "vision_backbone" in n)
    log.info("Params: total %.2fM, vision backbone %.2fM",
             total_params / 1e6, vision_params / 1e6)

    processor = Sam3Processor(model)
    torch.cuda.reset_peak_memory_stats()

    reports: dict[str, ResolutionReport] = {}
    for res_label, clip_stem in RESOLUTION_CLIPS.items():
        clip_dir = BAKEOFF_DIR / clip_stem
        if not (clip_dir / "prompts.json").exists():
            log.warning("[%s] No cache at %s — skipping", res_label, clip_dir)
            continue
        reports[res_label] = run_resolution(model, processor, res_label, clip_stem)
        gc.collect()
        torch.cuda.empty_cache()

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    out = {
        "model": "EfficientSAM3 ES-EV-S (stage1_all_converted)",
        "checkpoint": str(CKPT.relative_to(REPO_ROOT)),
        "source": "github.com/SimonZeng7108/efficientsam3 (Apache-2.0)",
        "license": "Apache-2.0",
        "backbone_type": "efficientvit",
        "model_name": "b0",
        "total_params_m": total_params / 1e6,
        "vision_backbone_params_m": vision_params / 1e6,
        "peak_vram_mb_5090": peak_vram_mb,
        "dtype": "bfloat16 (autocast)",
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        "by_resolution": {
            res: {
                "clip": r.clip,
                "n_frames_timed": r.n_frames_timed,
                "n_boxes_total": r.n_boxes_total,
                "per_frame_ms_5090": {
                    "mean": r.mean_per_frame_ms,
                    "p50":  r.p50_per_frame_ms,
                    "p95":  r.p95_per_frame_ms,
                    "all":  r.per_frame_ms,
                },
                "per_box_ms_5090": {
                    "mean": r.mean_per_box_ms,
                },
                "iou_vs_sam3": {
                    "mean":   r.mean_iou,
                    "median": r.median_iou,
                    "n":      len(r.iou_samples),
                },
            } for res, r in reports.items()
        },
        "_note": (
            "Timings measured on RTX 5090 with BF16 autocast. Edge-MPU projection "
            "follows the same bandwidth-ratio scaling used by the rest of the "
            "bake-offs — see sizer/npu_model.py::scale_edge_ms."
        ),
    }

    out_path = BAKEOFF_DIR / "efficientsam3_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)
    print("\n=== SUMMARY ===")
    print(json.dumps({
        "params_M": round(total_params / 1e6, 2),
        "vision_backbone_M": round(vision_params / 1e6, 2),
        "peak_vram_mb_5090": round(peak_vram_mb, 0),
        "per_resolution_p50_ms_5090": {
            res: round(reports[res].p50_per_frame_ms, 1) for res in reports
        },
        "per_resolution_mean_iou_vs_sam3": {
            res: round(reports[res].mean_iou, 3) for res in reports
        },
    }, indent=2))

    autocast_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
