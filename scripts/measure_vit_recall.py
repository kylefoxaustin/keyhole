"""Box-recall measurement for the ViT-alternatives candidates.

Companion to `scripts/bakeoff_vit_alternatives.py`. The bake-off script
measures latency + DRAM under ncu; this one measures detection accuracy
on the same EW-clip cache, comparing each variant's boxes against the
canonical YOLO11x reference (the same protocol `bakeoff_yoloe26.py` and
`bakeoff_efficientsam3.py` use → keeps everything on a unified accuracy
axis with the existing deck contestants).

Decision counted as a "hit" when the variant produces a box with IoU ≥ 0.5
to a YOLO11x reference box. Box-recall is what fraction of YOLO11x
reference boxes are matched. Box-precision (variant boxes that match
something in the reference) is reported as a sanity-check axis.

Out: `data/output/bakeoff/vit_alternatives_recall.json` — per-variant /
per-resolution recall + precision, plus per-frame counts.

Run after ncu sweep completes (one model at a time, GPU-exclusive):
    ~/.virtualenvs/keyhole/bin/python scripts/measure_vit_recall.py
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("vit_recall")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

BAKEOFF_DIR = REPO_ROOT / "data" / "output" / "bakeoff"
OUT = BAKEOFF_DIR / "vit_alternatives_recall.json"

RESOLUTION_CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}

# Same concept list every prior contestant used for SAM-3-style comparison.
SAM3_CONCEPTS = [
    "person", "vehicle", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "bird", "animal",
    "backpack", "bag", "hat", "umbrella",
    "package", "box", "suitcase", "chair", "laptop",
]

IOU_THRESHOLD = 0.5


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


def load_reference_boxes(clip_stem: str) -> dict[int, list[list[float]]]:
    """YOLO11x prompt boxes from prompts.json — the canonical reference
    used by every prior bake-off contestant (yoloe26, efficientsam3, etc.).
    Filtered at 0.35+ confidence at extraction time."""
    p = BAKEOFF_DIR / clip_stem / "prompts.json"
    raw = json.loads(p.read_text())
    return {int(k): [p["box"] for p in v] for k, v in raw.items()}


def score_recall(
    candidate_boxes_per_frame: dict[int, list[list[float]]],
    reference_boxes_per_frame: dict[int, list[list[float]]],
) -> dict:
    """Return dict with n_ref / n_cand / n_matched_ref / recall / precision."""
    n_ref = 0
    n_cand = 0
    n_matched_ref = 0
    n_matched_cand = 0
    for fid, refs in reference_boxes_per_frame.items():
        cands = candidate_boxes_per_frame.get(fid, [])
        n_ref += len(refs)
        n_cand += len(cands)
        if not refs or not cands:
            continue
        cand_arr = np.array(cands, dtype=np.float32)
        for rb in refs:
            rb_np = np.array(rb, dtype=np.float32)
            best = max((iou_xyxy(rb_np, c) for c in cand_arr), default=0.0)
            if best >= IOU_THRESHOLD:
                n_matched_ref += 1
        for cb in cand_arr:
            best = max((iou_xyxy(cb, np.array(r, dtype=np.float32)) for r in refs), default=0.0)
            if best >= IOU_THRESHOLD:
                n_matched_cand += 1
    return {
        "n_reference_boxes":  n_ref,
        "n_candidate_boxes":  n_cand,
        "n_matched_ref":      n_matched_ref,
        "n_matched_cand":     n_matched_cand,
        "recall":    (n_matched_ref / n_ref) if n_ref > 0 else 0.0,
        "precision": (n_matched_cand / n_cand) if n_cand > 0 else 0.0,
    }


# ─────────────── Per-variant detectors (return boxes per frame) ───────────────

def detect_rtdetr(resolutions: list[str]) -> dict[str, dict[int, list[list[float]]]]:
    from ultralytics import RTDETR
    model = RTDETR("rtdetr-l.pt")
    out: dict[str, dict] = {}
    for res in resolutions:
        clip_dir = BAKEOFF_DIR / RESOLUTION_CLIPS[res]
        if not (clip_dir / "prompts.json").exists():
            continue
        refs = load_reference_boxes(RESOLUTION_CLIPS[res])
        frames_dir = clip_dir / "frames"
        out[res] = {}
        for fid in sorted(refs):
            fp = frames_dir / f"frame_{fid:06d}.png"
            if not fp.exists():
                continue
            r = model.predict(cv2.imread(str(fp)), verbose=False, conf=0.25)
            boxes = []
            if r[0].boxes is not None and len(r[0].boxes) > 0:
                boxes = r[0].boxes.xyxy.cpu().numpy().tolist()
            out[res][fid] = boxes
    del model
    gc.collect(); torch.cuda.empty_cache()
    return out


def detect_detr(resolutions: list[str]) -> dict[str, dict[int, list[list[float]]]]:
    from transformers import DetrImageProcessor, DetrForObjectDetection
    from PIL import Image
    pid = "facebook/detr-resnet-50"
    proc = DetrImageProcessor.from_pretrained(pid)
    model = DetrForObjectDetection.from_pretrained(pid).to("cuda").eval()
    out: dict[str, dict] = {}
    for res in resolutions:
        clip_dir = BAKEOFF_DIR / RESOLUTION_CLIPS[res]
        if not (clip_dir / "prompts.json").exists():
            continue
        refs = load_reference_boxes(RESOLUTION_CLIPS[res])
        frames_dir = clip_dir / "frames"
        out[res] = {}
        for fid in sorted(refs):
            fp = frames_dir / f"frame_{fid:06d}.png"
            if not fp.exists():
                continue
            img = Image.fromarray(cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB))
            inputs = proc(images=img, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]])
            res_d = proc.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5,
            )
            boxes = res_d[0]["boxes"].cpu().numpy().tolist() if res_d else []
            out[res][fid] = boxes
    del model, proc
    gc.collect(); torch.cuda.empty_cache()
    return out


def detect_owlv2(resolutions: list[str]) -> dict[str, dict[int, list[list[float]]]]:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from PIL import Image
    pid = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(pid)
    model = Owlv2ForObjectDetection.from_pretrained(pid).to("cuda").eval()
    text_queries = [SAM3_CONCEPTS]
    out: dict[str, dict] = {}
    for res in resolutions:
        clip_dir = BAKEOFF_DIR / RESOLUTION_CLIPS[res]
        if not (clip_dir / "prompts.json").exists():
            continue
        refs = load_reference_boxes(RESOLUTION_CLIPS[res])
        frames_dir = clip_dir / "frames"
        out[res] = {}
        for fid in sorted(refs):
            fp = frames_dir / f"frame_{fid:06d}.png"
            if not fp.exists():
                continue
            img = Image.fromarray(cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB))
            inputs = proc(images=img, text=text_queries, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]])
            try:
                res_d = proc.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_sizes,
                    text_labels=text_queries, threshold=0.1,
                )
                boxes = res_d[0]["boxes"].cpu().numpy().tolist() if res_d else []
            except Exception:
                boxes = []
            out[res][fid] = boxes
    del model, proc
    gc.collect(); torch.cuda.empty_cache()
    return out


def detect_grounding_dino(resolutions: list[str]) -> dict[str, dict[int, list[list[float]]]]:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image
    pid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(pid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(pid).to("cuda").eval()
    text_query = ". ".join(c.lower() for c in SAM3_CONCEPTS) + "."
    out: dict[str, dict] = {}
    for res in resolutions:
        clip_dir = BAKEOFF_DIR / RESOLUTION_CLIPS[res]
        if not (clip_dir / "prompts.json").exists():
            continue
        refs = load_reference_boxes(RESOLUTION_CLIPS[res])
        frames_dir = clip_dir / "frames"
        out[res] = {}
        for fid in sorted(refs):
            fp = frames_dir / f"frame_{fid:06d}.png"
            if not fp.exists():
                continue
            img = Image.fromarray(cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB))
            inputs = proc(images=img, text=text_query, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]])
            try:
                res_d = proc.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"], target_sizes=target_sizes,
                )
                boxes = res_d[0]["boxes"].cpu().numpy().tolist() if res_d else []
            except Exception:
                boxes = []
            out[res][fid] = boxes
    del model, proc
    gc.collect(); torch.cuda.empty_cache()
    return out


VARIANTS = {
    "rtdetr-l":        detect_rtdetr,
    "detr_resnet50":   detect_detr,
    "owlv2":           detect_owlv2,
    "grounding_dino":  detect_grounding_dino,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS.keys()) + ["all"], default="all")
    ap.add_argument("--resolutions", default="720p")
    args = ap.parse_args()

    resolutions = [r.strip() for r in args.resolutions.split(",") if r.strip()]
    to_run = list(VARIANTS) if args.variant == "all" else [args.variant]
    log.info("Variants: %s · Resolutions: %s", to_run, resolutions)

    summary: dict = {"iou_threshold": IOU_THRESHOLD, "variants": {}}
    if OUT.exists():
        try:
            existing = json.loads(OUT.read_text())
            summary["variants"].update(existing.get("variants", {}))
        except Exception:
            pass

    for v in to_run:
        log.info("=== %s ===", v)
        try:
            cand = VARIANTS[v](resolutions)
        except Exception as e:
            log.error("[%s] FAILED: %s", v, e)
            summary["variants"][v] = {"error": str(e)}
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(summary, indent=2))
            continue
        per_res = {}
        for res, frame_boxes in cand.items():
            refs = load_reference_boxes(RESOLUTION_CLIPS[res])
            score = score_recall(frame_boxes, refs)
            per_res[res] = score
            log.info("[%s][%s] recall %.3f (%d/%d), precision %.3f (%d/%d)",
                     v, res, score["recall"], score["n_matched_ref"], score["n_reference_boxes"],
                     score["precision"], score["n_matched_cand"], score["n_candidate_boxes"])
        summary["variants"][v] = per_res
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(summary, indent=2))
        log.info("Saved partial summary to %s", OUT)

    log.info("=== DONE ===")
    log.info("Final at %s", OUT)
    for v, vd in summary.get("variants", {}).items():
        if "error" in vd:
            log.info("  %s: ERROR (%s)", v, vd["error"])
            continue
        for res, sc in vd.items():
            log.info("  %s [%s]: recall %.3f, precision %.3f",
                     v, res, sc["recall"], sc["precision"])


if __name__ == "__main__":
    main()
