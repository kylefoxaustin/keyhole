"""
TensorRT bake-off for YOLOE-26S-PF: does TRT close the gap to the shipping stack?

Compares:
  - PyTorch FP16 (baseline from bakeoff_yoloe26.py)
  - TRT FP16 engine
  - TRT FP8 engine (BuilderFlag.FP8 + FP16 + BF16, no explicit QDQ —
    TRT auto-selects FP8 where it helps)

Inputs:
  data/trt_engines/yoloe-26s-seg-pf.onnx       (via `YOLOE(...).export(format='onnx')`)
  data/trt_engines/yoloe-26s-seg-pf.fp16.engine
  data/trt_engines/yoloe-26s-seg-pf.fp8.engine

Output:
  data/output/bakeoff/trt_yoloe26_summary.json

Run from repo root:
    ~/.virtualenvs/keyhole/bin/python scripts/bakeoff_trt_yoloe26.py
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("trt_yoloe26")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

BAKEOFF_DIR = REPO_ROOT / "data" / "output" / "bakeoff"
TRT_DIR = REPO_ROOT / "data" / "trt_engines"

RESOLUTION_CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}

WARMUP = 3
TIMED = 10


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


def load_reference_boxes(clip_stem: str) -> dict[int, list]:
    prompts_path = BAKEOFF_DIR / clip_stem / "prompts.json"
    raw = json.loads(prompts_path.read_text())
    return {int(k): [p["box"] for p in v] for k, v in raw.items()}


@dataclass
class Report:
    recipe: str
    engine_path: str | None
    by_resolution: dict = field(default_factory=dict)
    peak_vram_mb: float = 0.0


def bench_one(model_path: str, recipe_tag: str) -> Report:
    from ultralytics import YOLOE
    log.info("=== Bench %s (%s) ===", recipe_tag, model_path)
    model = YOLOE(model_path)
    torch.cuda.reset_peak_memory_stats()

    rep = Report(recipe=recipe_tag, engine_path=model_path)

    for res_label, clip_stem in RESOLUTION_CLIPS.items():
        frames_dir = BAKEOFF_DIR / clip_stem / "frames"
        refs = load_reference_boxes(clip_stem)
        ordered = sorted(f for f in refs if refs[f] and (frames_dir / f"frame_{f:06d}.png").exists())

        times = []
        det_counts = []
        matched = 0
        total_ref = 0
        total_det = 0

        to_run = ordered[: WARMUP + TIMED]
        for i, fid in enumerate(to_run):
            is_warmup = i < WARMUP
            img = cv2.imread(str(frames_dir / f"frame_{fid:06d}.png"))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = model.predict(img, verbose=False)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000

            n_det = len(r[0].boxes) if r[0].boxes is not None else 0
            tag = "WARMUP" if is_warmup else "TIMED"
            log.info("[%s][%s][%s] frame %d: %d dets, %.2f ms",
                     recipe_tag, res_label, tag, fid, n_det, ms)
            if is_warmup:
                continue
            times.append(ms)
            det_counts.append(n_det)

            # Box recall vs YOLO11x reference
            boxes = r[0].boxes.xyxy.cpu().numpy().tolist() if r[0].boxes is not None else []
            ref_boxes = refs[fid]
            total_ref += len(ref_boxes)
            total_det += len(boxes)
            for rb in ref_boxes:
                rb_np = np.array(rb, dtype=np.float32)
                best = 0.0
                for yb in boxes:
                    iou = iou_xyxy(rb_np, np.array(yb, dtype=np.float32))
                    if iou > best:
                        best = iou
                if best >= 0.5:
                    matched += 1

        p50 = float(np.percentile(times, 50)) if times else 0.0
        p95 = float(np.percentile(times, 95)) if times else 0.0
        mean_det = float(np.mean(det_counts)) if det_counts else 0.0
        recall = (matched / total_ref) if total_ref > 0 else 0.0

        log.info("[%s][%s] %d frames: mean %.2f ms, p50 %.2f ms, p95 %.2f ms, "
                 "mean_det=%.1f, recall=%d/%d=%.3f",
                 recipe_tag, res_label, len(times), float(np.mean(times)),
                 p50, p95, mean_det, matched, total_ref, recall)

        rep.by_resolution[res_label] = {
            "clip": clip_stem,
            "n_frames_timed": len(times),
            "per_frame_ms_5090": {
                "mean": float(np.mean(times)) if times else 0.0,
                "p50": p50, "p95": p95,
                "all": times,
            },
            "mean_dets_per_frame": mean_det,
            "total_detections": total_det,
            "total_reference_boxes": total_ref,
            "matched_boxes_iou_0.5": matched,
            "box_recall": recall,
        }

    rep.peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rep


def main():
    recipes = [
        ("pytorch_fp16", "yoloe-26s-seg-pf.pt"),
        ("trt_fp16",     str(TRT_DIR / "yoloe-26s-seg-pf.fp16.engine")),
        ("trt_fp8",      str(TRT_DIR / "yoloe-26s-seg-pf.fp8.engine")),
    ]

    reports: dict = {}
    for tag, path in recipes:
        if not Path(path).exists():
            log.warning("Skipping %s — %s not found", tag, path)
            continue
        rep = bench_one(path, tag)
        reports[tag] = {
            "engine_path": rep.engine_path,
            "peak_vram_mb_5090": rep.peak_vram_mb,
            "by_resolution": rep.by_resolution,
        }

    bw_ratio = (1792.0 * 0.85) / (134.4 * 0.80)  # 14.17×

    out = {
        "model": "YOLOE-26S-PF (Ultralytics, Jan 2026 open-vocab YOLO)",
        "license": "AGPL-3.0",
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "tensorrt": __import__("tensorrt").__version__,
        "bw_ratio_5090_to_npu_mid": bw_ratio,
        "recipes": reports,
    }
    out_path = BAKEOFF_DIR / "trt_yoloe26_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)

    print("\n=== TRT YOLOE-26S-PF SUMMARY ===")
    print(f"{'Recipe':<14}{'720p p50':<12}{'1080p p50':<12}{'4K p50':<12}{'VRAM MB':<10}")
    print("-" * 60)
    for tag, r in reports.items():
        row = f"{tag:<14}"
        for res in ["720p", "1080p", "4K"]:
            p50 = r["by_resolution"].get(res, {}).get("per_frame_ms_5090", {}).get("p50", 0)
            row += f"{p50:<12.2f}"
        row += f"{r['peak_vram_mb_5090']:<10.0f}"
        print(row)

    print("\nNPU Mid projection (BW-scaled):")
    for tag, r in reports.items():
        p50_720 = r["by_resolution"].get("720p", {}).get("per_frame_ms_5090", {}).get("p50", 0)
        if p50_720 > 0:
            ms_mid = p50_720 * bw_ratio
            fps_mid = 1000.0 / ms_mid
            print(f"  {tag:<14} {ms_mid:6.1f} ms NPU Mid ({fps_mid:5.2f} FPS)")


if __name__ == "__main__":
    main()
