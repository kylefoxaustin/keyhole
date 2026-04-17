"""
YOLOE-26 bake-off — Ultralytics' Jan 2026 open-vocab YOLO.

YOLOE-26 collapses the two-stage Keyhole pipeline (YOLO-seg for detection/segmentation
+ CLIP for open-vocab tags) into a SINGLE model. Same output shape as our YOLO-seg
bake-off contestants (boxes + masks + class ids), plus optional text-prompt
class filter or prompt-free mode (built-in 4,585-class vocab).

Benches two variants against the cached 720p/1080p/4K frames from the existing
bake-off:
  - yoloe-26s-seg.pt (text-prompted with SAM3_CONCEPTS)          -> "replace YOLO+CLIP"
  - yoloe-26s-seg-pf.pt (prompt-free, built-in 4585-class vocab) -> "replace YOLO+CLIP, no manual class list"

Output: `data/output/bakeoff/yoloe26_summary.json` — per-resolution latency,
box-recall vs our shipping YOLO-seg, peak VRAM on 5090.

Run from repo root:
    ~/.virtualenvs/keyhole/bin/python scripts/bakeoff_yoloe26.py
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bakeoff_yoloe26")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

BAKEOFF_DIR = REPO_ROOT / "data" / "output" / "bakeoff"

RESOLUTION_CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}

# Same concept list used by the SAM 3 reference pass in bakeoff_sam_variants.py.
# Keeps comparability across the three SAM-3-competitor contestants (ours, community
# EfficientSAM3, and YOLOE-26 text-prompted).
SAM3_CONCEPTS = [
    "person", "vehicle", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "bird", "animal",
    "backpack", "bag", "hat", "umbrella",
    "package", "box", "suitcase", "chair", "laptop",
]

WARMUP_FRAMES = 2
MAX_FRAMES = 10


# ── Metrics ──

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
    n_frames_timed: int = 0
    n_boxes_yoloe: int = 0
    n_boxes_reference: int = 0
    n_matched_boxes: int = 0
    per_frame_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.per_frame_ms)) if self.per_frame_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.per_frame_ms, 50)) if self.per_frame_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.per_frame_ms, 95)) if self.per_frame_ms else 0.0

    @property
    def box_recall_vs_ref(self) -> float:
        """Fraction of reference YOLO-seg boxes that YOLOE-26 also detected
        (IoU ≥ 0.5). Treat as 'did the one-model open-vocab YOLO find the same
        things the two-stage pipeline found?'."""
        if self.n_boxes_reference == 0:
            return 0.0
        return self.n_matched_boxes / self.n_boxes_reference


def load_reference_boxes(clip_stem: str) -> dict[int, list[list[float]]]:
    """Reference boxes = YOLO11x prompt boxes from the existing bake-off cache.
    These were the input to SAM 3 / EfficientSAM / MobileSAM / EfficientSAM3
    in prior contestants — high-confidence, filtered 0.35+. Good ground-truth
    for 'what did the two-stage pipeline find'."""
    prompts_path = BAKEOFF_DIR / clip_stem / "prompts.json"
    raw = json.loads(prompts_path.read_text())
    return {int(k): [p["box"] for p in v] for k, v in raw.items()}


def run_variant(variant_tag: str, model_name: str, prompt_free: bool,
                 class_list: list[str] | None) -> dict:
    """Bench one YOLOE variant across all resolutions."""
    from ultralytics import YOLOE
    log.info("=== Loading %s (%s) ===", model_name, variant_tag)
    model = YOLOE(model_name)
    if not prompt_free:
        assert class_list, "Text-prompted mode needs a class list"
        model.set_classes(class_list)
        log.info("Set %d text-prompt classes: %s", len(class_list), class_list[:6] + ["..."])
    else:
        log.info("Prompt-free mode: using model's built-in ~4585-class vocabulary")

    n_params = sum(p.numel() for p in model.model.parameters())
    torch.cuda.reset_peak_memory_stats()

    reports: dict[str, ResolutionReport] = {}
    for res_label, clip_stem in RESOLUTION_CLIPS.items():
        clip_dir = BAKEOFF_DIR / clip_stem
        if not (clip_dir / "prompts.json").exists():
            continue
        refs = load_reference_boxes(clip_stem)
        frames_dir = clip_dir / "frames"
        ordered = sorted(f for f in refs if refs[f] and (frames_dir / f"frame_{f:06d}.png").exists())

        rep = ResolutionReport(resolution=res_label, clip=clip_stem)
        to_run = ordered[: WARMUP_FRAMES + MAX_FRAMES]

        for i, fid in enumerate(to_run):
            is_warmup = i < WARMUP_FRAMES
            img = cv2.imread(str(frames_dir / f"frame_{fid:06d}.png"))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = model.predict(img, verbose=False)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000

            tag = "WARMUP" if is_warmup else "TIMED"
            n_det = len(r[0].boxes) if r[0].boxes is not None else 0
            log.info("[%s][%s][%s] frame %d: %d detections, %.1f ms",
                     variant_tag, res_label, tag, fid, n_det, ms)

            if is_warmup:
                continue
            rep.per_frame_ms.append(ms)
            rep.n_frames_timed += 1

            # Box-recall vs YOLO11x prompt boxes (our reference boxes)
            ref_boxes = refs[fid]
            yoloe_boxes = []
            if r[0].boxes is not None and len(r[0].boxes) > 0:
                yoloe_boxes = r[0].boxes.xyxy.cpu().numpy().tolist()
            rep.n_boxes_yoloe += len(yoloe_boxes)
            rep.n_boxes_reference += len(ref_boxes)
            for rb in ref_boxes:
                rb_np = np.array(rb, dtype=np.float32)
                best_iou = 0.0
                for yb in yoloe_boxes:
                    iou = iou_xyxy(rb_np, np.array(yb, dtype=np.float32))
                    if iou > best_iou:
                        best_iou = iou
                if best_iou >= 0.5:
                    rep.n_matched_boxes += 1

        log.info("[%s][%s] %d frames timed, mean %.1f ms (p50 %.1f, p95 %.1f), "
                 "recall vs YOLO11x ref = %d/%d = %.3f",
                 variant_tag, res_label, rep.n_frames_timed, rep.mean_ms,
                 rep.p50_ms, rep.p95_ms, rep.n_matched_boxes, rep.n_boxes_reference,
                 rep.box_recall_vs_ref)
        reports[res_label] = rep

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "variant_tag": variant_tag,
        "weights": model_name,
        "prompt_free": prompt_free,
        "class_list": class_list if not prompt_free else None,
        "params_m": n_params / 1e6,
        "peak_vram_mb_5090": peak_vram_mb,
        "by_resolution": {
            res: {
                "clip": r.clip,
                "n_frames_timed": r.n_frames_timed,
                "n_boxes_yoloe": r.n_boxes_yoloe,
                "n_boxes_reference_yolo11x": r.n_boxes_reference,
                "n_matched_boxes_iou_ge_0.5": r.n_matched_boxes,
                "box_recall_vs_yolo11x": r.box_recall_vs_ref,
                "per_frame_ms_5090": {
                    "mean": r.mean_ms,
                    "p50":  r.p50_ms,
                    "p95":  r.p95_ms,
                    "all":  r.per_frame_ms,
                },
            } for res, r in reports.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["text_prompt_s", "prompt_free_s"],
                    help="Which variants to run")
    args = ap.parse_args()

    registry = {
        "text_prompt_s":  ("yoloe-26s-seg.pt",    False, SAM3_CONCEPTS),
        "prompt_free_s":  ("yoloe-26s-seg-pf.pt", True,  None),
        # Add more if you want a bigger sweep — weights auto-download
        "text_prompt_l":  ("yoloe-26l-seg.pt",    False, SAM3_CONCEPTS),
        "prompt_free_l":  ("yoloe-26l-seg-pf.pt", True,  None),
    }

    out = {
        "model": "YOLOE-26 (Ultralytics, Jan 2026)",
        "source": "docs.ultralytics.com/models/yoloe",
        "license": "AGPL-3.0 (+ enterprise via Ultralytics)",
        "ultralytics_version": __import__("ultralytics").__version__,
        "torch_version": torch.__version__,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dtype": "fp16 (ultralytics default on cuda)",
        "variants": {},
    }
    for tag in args.variants:
        if tag not in registry:
            log.warning("Unknown variant: %s — skipping", tag)
            continue
        weights, prompt_free, class_list = registry[tag]
        try:
            out["variants"][tag] = run_variant(tag, weights, prompt_free, class_list)
        except Exception as e:
            log.error("Variant %s failed: %s", tag, e)
            out["variants"][tag] = {"error": str(e)}

    out_path = BAKEOFF_DIR / "yoloe26_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)

    # Pretty one-liner per variant
    print("\n=== YOLOE-26 SUMMARY ===")
    for tag, v in out["variants"].items():
        if "error" in v:
            print(f"{tag}: ERROR — {v['error']}")
            continue
        parts = [f"{tag} ({v['weights']}) params={v['params_m']:.1f}M"]
        for res, rr in v["by_resolution"].items():
            parts.append(f"{res} {rr['per_frame_ms_5090']['p50']:.1f}ms recall={rr['box_recall_vs_yolo11x']:.2f}")
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()
