"""ViT-alternatives bake-off — what-if candidates to replace YOLO-seg + SAM 3.

Kyle's "what if" 2026-04-25: could vision transformers replace the existing
two-stage CNN-based stack (YOLO-seg + CLIP for cameras, SAM 3 for agentic
prompts)? Two roles tested:

  Camera stream (every frame, real-time):
    - rtdetr-l        Ultralytics RT-DETR-L (~32M params, ViT encoder + transformer decoder)
    - detr_resnet50   Facebook DETR ResNet-50 (~41M params) — original ViT detector

  Agentic prompt (on-demand text-to-segment / open-vocab detection):
    - owlv2           Google OWLv2-base-patch16-ensemble (~150M params) — open-vocab detection
    - grounding_dino  IDEA-Research Grounding DINO-Tiny (~172M params) — text-prompted detection

Note: Florence-2-base was the original 4th candidate but is pinned to an older
transformers version (its custom config code references `forced_bos_token_id`
which was removed in transformers 5.x). DETR + Grounding DINO are equivalent-
family substitutes that work cleanly with the current keyhole venv.

Goal: characterize the BW envelope these would impose on edge silicon. Per-
forward DRAM bytes (via ncu kernel-replay, separate pass) + 5090 latency
(this script). Compare against shipping YOLO-seg FP8 TRT (105 MB / 0.49 ms
on yolov8n; 217 MB / 0.68 ms on yolo11s) and SAM 3 (119 GB / 95 ms).

Frames: cached embedded_world clips at 720p / 1080p / 4K from prior bake-offs.
Output: data/output/bakeoff/vit_alternatives_summary.json.

Run from repo root:
    ~/.virtualenvs/keyhole/bin/python scripts/bakeoff_vit_alternatives.py
    ~/.virtualenvs/keyhole/bin/python scripts/bakeoff_vit_alternatives.py --variant rtdetr-l
    ~/.virtualenvs/keyhole/bin/python scripts/bakeoff_vit_alternatives.py --resolutions 720p
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
log = logging.getLogger("bakeoff_vit")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

BAKEOFF_DIR = REPO_ROOT / "data" / "output" / "bakeoff"
OUTPUT_PATH = BAKEOFF_DIR / "vit_alternatives_summary.json"

RESOLUTION_CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}

# Same concept list used by SAM 3 reference + YOLOE-26 contestants. Keeps
# the agentic-prompt apples-to-apples across the SAM 3 lineage.
SAM3_CONCEPTS = [
    "person", "vehicle", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "bird", "animal",
    "backpack", "bag", "hat", "umbrella",
    "package", "box", "suitcase", "chair", "laptop",
]

WARMUP_FRAMES = 2
MAX_FRAMES = 10


# ────────────────────────── Per-variant report ──────────────────────────

@dataclass
class ResolutionReport:
    resolution: str
    n_frames_timed: int = 0
    n_detections_total: int = 0
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


def _load_frames_for_resolution(res_label: str) -> tuple[Path, list[int]] | None:
    """Resolve cached frames + ordered frame ids for `res_label`. Returns None
    if the clip doesn't exist (skipped in the run)."""
    clip_stem = RESOLUTION_CLIPS[res_label]
    clip_dir = BAKEOFF_DIR / clip_stem
    prompts_path = clip_dir / "prompts.json"
    if not prompts_path.exists():
        return None
    refs = json.loads(prompts_path.read_text())
    # Use prompt-bearing frames so the input distribution matches the
    # existing bake-off cohort (YOLOE-26, EfficientSAM3, mask_variants).
    frames_dir = clip_dir / "frames"
    ordered = sorted(int(k) for k in refs if refs[k] and (frames_dir / f"frame_{int(k):06d}.png").exists())
    return frames_dir, ordered[: WARMUP_FRAMES + MAX_FRAMES]


# ────────────────────────── Variants ──────────────────────────

def run_rtdetr(model_name: str, resolutions: list[str]) -> dict:
    """Ultralytics RT-DETR. Pre-trained on COCO. Same .predict() shape as YOLO."""
    from ultralytics import RTDETR

    log.info("=== Loading %s (RT-DETR) ===", model_name)
    model = RTDETR(model_name)
    n_params = sum(p.numel() for p in model.model.parameters())
    torch.cuda.reset_peak_memory_stats()

    reports: dict[str, ResolutionReport] = {}
    for res_label in resolutions:
        loaded = _load_frames_for_resolution(res_label)
        if loaded is None:
            log.info("  skip %s (no cached frames)", res_label)
            continue
        frames_dir, frame_ids = loaded
        rep = ResolutionReport(resolution=res_label)
        from src.profiling.nvtx_helpers import nvtx_range
        for i, fid in enumerate(frame_ids):
            is_warmup = i < WARMUP_FRAMES
            img = cv2.imread(str(frames_dir / f"frame_{fid:06d}.png"))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with nvtx_range(f"rtdetr_l__{res_label}"):
                r = model.predict(img, verbose=False)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000

            n_det = len(r[0].boxes) if r[0].boxes is not None else 0
            log.info("[rtdetr-l][%s][%s] frame %d: %d det, %.2f ms",
                     res_label, "WARM" if is_warmup else "TIMED", fid, n_det, ms)
            if is_warmup:
                continue
            rep.per_frame_ms.append(ms)
            rep.n_frames_timed += 1
            rep.n_detections_total += n_det

        reports[res_label] = rep

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "variant": "rtdetr-l",
        "model": model_name,
        "role": "camera_stream",
        "n_params_M": round(n_params / 1e6, 2),
        "peak_vram_mb": round(peak_mb, 1),
        "resolutions": {k: asdict(v) | {
            "mean_ms": v.mean_ms, "p50_ms": v.p50_ms, "p95_ms": v.p95_ms
        } for k, v in reports.items()},
    }


def run_detr_resnet50(resolutions: list[str]) -> dict:
    """Facebook DETR ResNet-50 — original ViT-based detector (CNN backbone,
    transformer encoder + decoder), the architecture RT-DETR descends from.
    Pre-trained on COCO. Smaller param count than RT-DETR-L (~41M vs ~33M
    same order) but heavier per forward (no Ultralytics-style optimization).
    Camera-stream role."""
    from transformers import DetrImageProcessor, DetrForObjectDetection
    from PIL import Image

    model_id = "facebook/detr-resnet-50"
    log.info("=== Loading %s (DETR) ===", model_id)
    processor = DetrImageProcessor.from_pretrained(model_id)
    model = DetrForObjectDetection.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()

    reports: dict[str, ResolutionReport] = {}
    for res_label in resolutions:
        loaded = _load_frames_for_resolution(res_label)
        if loaded is None:
            log.info("  skip %s (no cached frames)", res_label)
            continue
        frames_dir, frame_ids = loaded
        rep = ResolutionReport(resolution=res_label)
        from src.profiling.nvtx_helpers import nvtx_range

        for i, fid in enumerate(frame_ids):
            is_warmup = i < WARMUP_FRAMES
            img_bgr = cv2.imread(str(frames_dir / f"frame_{fid:06d}.png"))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            inputs = processor(images=img, return_tensors="pt").to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with nvtx_range(f"detr__{res_label}"):
                with torch.inference_mode():
                    outputs = model(**inputs)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000

            target_sizes = torch.tensor([img.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5,
            )
            n_det = len(results[0]["scores"]) if results else 0
            log.info("[detr][%s][%s] frame %d: %d det, %.2f ms",
                     res_label, "WARM" if is_warmup else "TIMED", fid, n_det, ms)
            if is_warmup:
                continue
            rep.per_frame_ms.append(ms)
            rep.n_frames_timed += 1
            rep.n_detections_total += n_det

        reports[res_label] = rep

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "variant": "detr_resnet50",
        "model": model_id,
        "role": "camera_stream",
        "n_params_M": round(n_params / 1e6, 2),
        "peak_vram_mb": round(peak_mb, 1),
        "resolutions": {k: asdict(v) | {
            "mean_ms": v.mean_ms, "p50_ms": v.p50_ms, "p95_ms": v.p95_ms
        } for k, v in reports.items()},
    }


def run_grounding_dino(resolutions: list[str]) -> dict:
    """IDEA-Research Grounding DINO Tiny — text-prompted open-vocab
    detection (Swin-Tiny backbone + DINO transformer head + BERT text
    encoder). Same role as OWLv2 (agentic prompt → boxes). Comparable
    point on the agentic-prompt design space.

    Input is the SAM3_CONCEPTS as a single space-separated query string
    (Grounding DINO's text format expects period-separated concept list)."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image

    model_id = "IDEA-Research/grounding-dino-tiny"
    log.info("=== Loading %s (Grounding DINO) ===", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    # fp32-only: GDino's text-vision cross-attention silently produces fp32
    # tensors that mismatch fp16 weights when the model is loaded in fp16
    # (RuntimeError on F.linear). Other ViT variants in this bake-off use
    # fp16 for fairer 5090-side timing; flagging the asymmetry — GDino is
    # ~2× slower than its fp16 sibling would be at the same arch.
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()

    # Grounding DINO format: period-separated lowercase concepts
    text_query = ". ".join(c.lower() for c in SAM3_CONCEPTS) + "."

    reports: dict[str, ResolutionReport] = {}
    for res_label in resolutions:
        loaded = _load_frames_for_resolution(res_label)
        if loaded is None:
            log.info("  skip %s (no cached frames)", res_label)
            continue
        frames_dir, frame_ids = loaded
        rep = ResolutionReport(resolution=res_label)
        from src.profiling.nvtx_helpers import nvtx_range

        for i, fid in enumerate(frame_ids):
            is_warmup = i < WARMUP_FRAMES
            img_bgr = cv2.imread(str(frames_dir / f"frame_{fid:06d}.png"))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            inputs = processor(images=img, text=text_query, return_tensors="pt").to(
                "cuda"
            )

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with nvtx_range(f"grounding_dino__{res_label}"):
                with torch.inference_mode():
                    outputs = model(**inputs)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000

            # Post-process is informational only (not timed). API kwargs
            # vary across transformers versions; if it errors, just count
            # forward as having run and report 0 detections.
            try:
                target_sizes = torch.tensor([img.size[::-1]])
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"], target_sizes=target_sizes,
                )
                n_det = len(results[0].get("scores", [])) if results else 0
            except Exception:
                n_det = -1  # post-process API mismatch; forward succeeded
            log.info("[grounding_dino][%s][%s] frame %d: %d det, %.2f ms",
                     res_label, "WARM" if is_warmup else "TIMED", fid, n_det, ms)
            if is_warmup:
                continue
            rep.per_frame_ms.append(ms)
            rep.n_frames_timed += 1
            rep.n_detections_total += n_det

        reports[res_label] = rep

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "variant": "grounding_dino",
        "model": model_id,
        "role": "agentic_prompt",
        "n_params_M": round(n_params / 1e6, 2),
        "peak_vram_mb": round(peak_mb, 1),
        "n_text_queries": len(SAM3_CONCEPTS),
        "resolutions": {k: asdict(v) | {
            "mean_ms": v.mean_ms, "p50_ms": v.p50_ms, "p95_ms": v.p95_ms
        } for k, v in reports.items()},
    }


def run_owlv2(resolutions: list[str]) -> dict:
    """Google OWLv2 — open-vocab detection from text queries. Pass the
    SAM3_CONCEPTS as text queries; model returns boxes per query.
    Agentic-prompt role."""
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from PIL import Image

    model_id = "google/owlv2-base-patch16-ensemble"
    log.info("=== Loading %s (OWLv2) ===", model_id)
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()

    text_queries = [SAM3_CONCEPTS]  # batch of 1

    reports: dict[str, ResolutionReport] = {}
    for res_label in resolutions:
        loaded = _load_frames_for_resolution(res_label)
        if loaded is None:
            log.info("  skip %s (no cached frames)", res_label)
            continue
        frames_dir, frame_ids = loaded
        rep = ResolutionReport(resolution=res_label)
        from src.profiling.nvtx_helpers import nvtx_range

        for i, fid in enumerate(frame_ids):
            is_warmup = i < WARMUP_FRAMES
            img_bgr = cv2.imread(str(frames_dir / f"frame_{fid:06d}.png"))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            inputs = processor(images=img, text=text_queries, return_tensors="pt").to(
                "cuda"
            )
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with nvtx_range(f"owlv2__{res_label}"):
                with torch.inference_mode():
                    outputs = model(**inputs)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000

            # Post-process to get box count (cheap CPU step, not timed)
            target_sizes = torch.tensor([img.size[::-1]])
            results = processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes,
                text_labels=text_queries, threshold=0.1,
            )
            n_det = len(results[0]["boxes"]) if results else 0
            log.info("[owlv2][%s][%s] frame %d: %d det, %.2f ms",
                     res_label, "WARM" if is_warmup else "TIMED", fid, n_det, ms)
            if is_warmup:
                continue
            rep.per_frame_ms.append(ms)
            rep.n_frames_timed += 1
            rep.n_detections_total += n_det

        reports[res_label] = rep

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "variant": "owlv2",
        "model": model_id,
        "role": "agentic_prompt",
        "n_params_M": round(n_params / 1e6, 2),
        "peak_vram_mb": round(peak_mb, 1),
        "n_text_queries": len(SAM3_CONCEPTS),
        "resolutions": {k: asdict(v) | {
            "mean_ms": v.mean_ms, "p50_ms": v.p50_ms, "p95_ms": v.p95_ms
        } for k, v in reports.items()},
    }


VARIANTS = {
    "rtdetr-l":        lambda res: run_rtdetr("rtdetr-l.pt", res),
    "detr_resnet50":   run_detr_resnet50,
    "owlv2":           run_owlv2,
    "grounding_dino":  run_grounding_dino,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS.keys()) + ["all"], default="all")
    ap.add_argument("--resolutions", default="720p,1080p,4K",
                    help="comma-separated subset of 720p,1080p,4K")
    args = ap.parse_args()

    resolutions = [r.strip() for r in args.resolutions.split(",") if r.strip()]
    log.info("CUDA: %s", torch.cuda.get_device_name(0))
    log.info("Resolutions: %s", resolutions)

    to_run = list(VARIANTS) if args.variant == "all" else [args.variant]
    log.info("Variants: %s", to_run)

    # Merge into existing summary if present so a single-variant rerun
    # doesn't clobber sibling variant data.
    if OUTPUT_PATH.exists():
        try:
            summary = json.loads(OUTPUT_PATH.read_text())
            log.info("Merging into existing summary at %s (variants present: %s)",
                     OUTPUT_PATH, list(summary.get("variants", {}).keys()))
        except Exception:
            summary = {}
    else:
        summary = {}
    summary.setdefault("host", "RTX 5090")
    summary.setdefault("torch", torch.__version__)
    summary.setdefault("warmup_frames", WARMUP_FRAMES)
    summary.setdefault("timed_frames", MAX_FRAMES)
    summary.setdefault("variants", {})
    for v in to_run:
        log.info("============================================================")
        log.info("Variant: %s", v)
        log.info("============================================================")
        result = VARIANTS[v](resolutions)
        summary["variants"][v] = result
        # Save incrementally so a crash midway doesn't lose earlier work.
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
        log.info("Saved partial summary to %s", OUTPUT_PATH)

    log.info("=== DONE ===")
    log.info("Final summary at %s", OUTPUT_PATH)
    for v, r in summary["variants"].items():
        log.info("  %s (%s, %.1f M params, %.0f MB peak):",
                 v, r.get("role", "?"), r.get("n_params_M", 0), r.get("peak_vram_mb", 0))
        for res, rd in r.get("resolutions", {}).items():
            log.info("    %s  p50 %.2f ms  mean %.2f ms",
                     res, rd.get("p50_ms", 0), rd.get("mean_ms", 0))


if __name__ == "__main__":
    main()
