"""
Hybrid V2 bake-off — characterize YOLO-seg + CLIP and measure the effect of
FP8 / INT8 activation quantization applied to the CLIP half only.

YOLO-seg is Conv-only, so torchao can't touch it (roadmap #6 blocked). CLIP's
ViT-B-32 vision + text encoders are ~72 nn.Linear layers — those should swap
cleanly and let us actually exercise the activation-traffic halving on the
portion of the pipeline where it applies.

Per-frame latency is split YOLO vs CLIP using the detector's internal timers.
Quality is measured as top-1 concept-tag agreement vs the BF16 reference (YOLO
bboxes are deterministic, so CLIP is the only source of divergence).

Outputs:
  data/output/bakeoff/hybrid_v2/{clip_stem}/{recipe}.json
  data/output/bakeoff/hybrid_v2_summary.json
  data/output/bakeoff/hybrid_v2_edge_projection.json
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detect.hybrid_v2 import HybridV2Detector
from src.ingest.video import ExtractedFrame
from src.emulate.npu_emulator import (
    NPUEmulator, WorkloadProfile, RTX_5090, EDGE_MPU_TARGET,
)
from scripts.bakeoff_sam_variants import (
    BAKEOFF_DIR, sync_cuda, gpu_reset_peak, gpu_peak_mb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("hybrid_v2")

CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}
OUT_DIR = BAKEOFF_DIR / "hybrid_v2"
RECIPES = ["bf16", "fp8", "int8"]
YOLO_SEG_VARIANT = "yolo11s-seg.pt"
CLIP_VARIANT = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"


@dataclass
class SampledFrame:
    idx: int
    timestamp_sec: float
    path: str


def count_quantized_linears(model: torch.nn.Module) -> int:
    """Detect torchao-swapped Linears via weight tensor subclass name.
    Matches the helper in bakeoff_smoothquant.py — counts each weight once."""
    n = 0
    for m in model.modules():
        w = getattr(m, "weight", None)
        if w is None:
            continue
        tname = type(w).__name__
        if any(s in tname for s in ("LinearActivationQuantized", "QuantizedTensor", "Float8")):
            n += 1
        elif hasattr(w, "dtype") and w.dtype == torch.float8_e4m3fn:
            n += 1
    return n


def apply_fp8(clip_model: torch.nn.Module) -> dict:
    """Quantize CLIP to FP8 dynamic-act / fp8-weight via torchao."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig, quantize_, PerTensor,
    )
    n_linear = sum(1 for m in clip_model.modules() if isinstance(m, torch.nn.Linear))
    clip_model.bfloat16()
    cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
    quantize_(clip_model, cfg)
    n_quant = count_quantized_linears(clip_model)
    return {"recipe": "fp8", "n_linear": n_linear, "n_quantized": n_quant}


def apply_int8(clip_model: torch.nn.Module) -> dict:
    """Quantize CLIP to INT8 dynamic-act / int8-weight via torchao."""
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
    n_linear = sum(1 for m in clip_model.modules() if isinstance(m, torch.nn.Linear))
    clip_model.bfloat16()
    cfg = Int8DynamicActivationInt8WeightConfig()
    quantize_(clip_model, cfg)
    n_quant = count_quantized_linears(clip_model)
    return {"recipe": "int8", "n_linear": n_linear, "n_quantized": n_quant}


def wrap_clip_forwards_with_autocast(clip_model: torch.nn.Module):
    """Wrap encode_image / encode_text so inputs are cast and ops run under autocast(bf16).

    CLIP crops preprocess to fp32; we need bf16 entering quantized linears, matching
    the pattern bakeoff_fp8.py used for EfficientSAM's vision tower.
    """
    _encode_image = clip_model.encode_image
    _encode_text = clip_model.encode_text

    def _cast_image(image, *args, **kwargs):
        if torch.is_tensor(image) and image.is_floating_point() and image.dtype == torch.float32:
            image = image.bfloat16()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return _encode_image(image, *args, **kwargs)

    def _cast_text(text, *args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return _encode_text(text, *args, **kwargs)

    clip_model.encode_image = _cast_image
    clip_model.encode_text = _cast_text


def load_detector(recipe: str) -> tuple[HybridV2Detector, dict]:
    """Load Hybrid V2 and optionally quantize the CLIP half."""
    det = HybridV2Detector(
        yolo_seg_model=YOLO_SEG_VARIANT,
        clip_model=CLIP_VARIANT,
        clip_pretrained=CLIP_PRETRAINED,
        retain_masks=False,
    )
    det.load_model()

    clip_params = sum(p.numel() for p in det.clip_model.parameters())
    yolo_params = sum(p.numel() for p in det.yolo.model.parameters())
    info = {
        "recipe": recipe,
        "yolo_params_m": yolo_params / 1e6,
        "clip_params_m": clip_params / 1e6,
        "n_linear": 0,
        "n_quantized": 0,
    }

    if recipe == "bf16":
        det.clip_model.bfloat16()
        wrap_clip_forwards_with_autocast(det.clip_model)
        info["n_linear"] = sum(1 for m in det.clip_model.modules()
                               if isinstance(m, torch.nn.Linear))
    elif recipe == "fp8":
        qinfo = apply_fp8(det.clip_model)
        wrap_clip_forwards_with_autocast(det.clip_model)
        info.update(qinfo)
    elif recipe == "int8":
        qinfo = apply_int8(det.clip_model)
        wrap_clip_forwards_with_autocast(det.clip_model)
        info.update(qinfo)
    else:
        raise ValueError(f"unknown recipe: {recipe}")

    # CLIP tokenizer/text_cache is re-primed lazily on first frame.
    det._text_cache = {}
    return det, info


def run_one(detector: HybridV2Detector, clip_stem: str) -> dict:
    """Run detector on cached frames for one clip; record per-frame latencies + top-k concepts."""
    clip_dir = BAKEOFF_DIR / clip_stem
    frames = [SampledFrame(**e) for e in json.loads((clip_dir / "frames.json").read_text())]

    # Warm-up pass to fill text_cache + kernel compile
    img0 = cv2.imread(str(clip_dir / frames[0].path))
    _ = detector.detect_frame(ExtractedFrame(
        frame_number=frames[0].idx, timestamp_sec=frames[0].timestamp_sec,
        image=img0, source_video=clip_stem,
    ))
    detector._latencies.clear()
    detector._yolo_times.clear()
    detector._clip_times.clear()
    detector._det_counts.clear()
    gpu_reset_peak()

    frame_results = []
    for f in frames:
        img = cv2.imread(str(clip_dir / f.path))
        ef = ExtractedFrame(frame_number=f.idx, timestamp_sec=f.timestamp_sec,
                            image=img, source_video=clip_stem)
        enriched = detector.detect_frame(ef)
        det_summary = []
        for d in enriched.detections:
            top_concept = d.concepts[0].concept if d.concepts else None
            top_score = d.concepts[0].confidence if d.concepts else None
            top3 = [(c.concept, c.confidence) for c in d.concepts[:3]]
            det_summary.append({
                "bbox": list(d.bbox),
                "class_name": d.class_name,
                "confidence": d.confidence,
                "top_concept": top_concept,
                "top_score": top_score,
                "top3": top3,
            })
        frame_results.append({
            "frame_idx": f.idx,
            "total_ms": detector._latencies[-1],
            "yolo_ms": detector._yolo_times[-1],
            "clip_ms": detector._clip_times[-1],
            "n_det": len(enriched.detections),
            "detections": det_summary,
        })

    peak_vram = gpu_peak_mb()
    return {
        "clip_stem": clip_stem,
        "frames": frame_results,
        "mean_total_ms": float(np.mean([fr["total_ms"] for fr in frame_results])),
        "mean_yolo_ms": float(np.mean([fr["yolo_ms"] for fr in frame_results])),
        "mean_clip_ms": float(np.mean([fr["clip_ms"] for fr in frame_results])),
        "mean_det_per_frame": float(np.mean([fr["n_det"] for fr in frame_results])),
        "peak_vram_mb": peak_vram,
    }


def top1_agreement(ref: list[dict], variant: list[dict]) -> tuple[float, int, int]:
    """Top-1 concept tag agreement between two runs.

    Since YOLO-seg is deterministic (same input → same boxes), we match by index
    within each frame. If counts differ we score mismatches as disagreement.
    """
    match, total = 0, 0
    for fr_ref, fr_var in zip(ref, variant):
        d_ref = fr_ref["detections"]
        d_var = fr_var["detections"]
        n = min(len(d_ref), len(d_var))
        total += max(len(d_ref), len(d_var))
        for i in range(n):
            if (d_ref[i].get("top_concept") is None and d_var[i].get("top_concept") is None):
                match += 1
                continue
            if d_ref[i].get("top_concept") == d_var[i].get("top_concept"):
                match += 1
    return (match / total if total else 0.0), match, total


def top3_jaccard(ref: list[dict], variant: list[dict]) -> float:
    """Mean Jaccard of top-3 concept sets across matched detections."""
    sims = []
    for fr_ref, fr_var in zip(ref, variant):
        d_ref = fr_ref["detections"]
        d_var = fr_var["detections"]
        n = min(len(d_ref), len(d_var))
        for i in range(n):
            s_ref = set(c[0] for c in d_ref[i].get("top3", []))
            s_var = set(c[0] for c in d_var[i].get("top3", []))
            if not s_ref and not s_var:
                continue
            union = len(s_ref | s_var)
            if union:
                sims.append(len(s_ref & s_var) / union)
    return float(np.mean(sims)) if sims else 0.0


def project_edge(all_results: dict) -> dict:
    """Project per-half (YOLO + CLIP) then sum, per resolution, per recipe."""
    emu = NPUEmulator(reference=RTX_5090, target=EDGE_MPU_TARGET)
    projections = {}

    for res, clip_stem in CLIPS.items():
        projections[res] = {}
        bf16_run = all_results.get(res, {}).get("bf16")
        if not bf16_run:
            continue

        # -- YOLO (never quantized here) --
        yolo_params = int(all_results[res]["_info"]["bf16"]["yolo_params_m"] * 1e6)
        yolo_bytes_bf16 = yolo_params * 2
        yolo_ms_bf16 = bf16_run["mean_yolo_ms"]
        yolo_wl = WorkloadProfile(
            stage_name="yolo_seg",
            model_name="yolo11s-seg",
            param_count=yolo_params,
            model_size_bytes=yolo_bytes_bf16,
            precision="bf16",
            measured_latency_ms=yolo_ms_bf16,
            measured_gpu_kernel_ms=yolo_ms_bf16,
            measured_gpu=RTX_5090.name,
            measured_peak_vram_bytes=yolo_bytes_bf16 * 2,  # rough: weights + acts
            peak_activation_bytes=yolo_bytes_bf16,
        )
        yolo_proj = emu.project_workload(yolo_wl)

        # -- CLIP (recipe-dependent) --
        clip_params = int(all_results[res]["_info"]["bf16"]["clip_params_m"] * 1e6)
        clip_bytes_bf16 = clip_params * 2
        clip_ms_bf16 = bf16_run["mean_clip_ms"]
        clip_wl = WorkloadProfile(
            stage_name="clip_vitb32",
            model_name="clip_ViT-B-32",
            param_count=clip_params,
            model_size_bytes=clip_bytes_bf16,
            precision="bf16",
            measured_latency_ms=clip_ms_bf16,
            measured_gpu_kernel_ms=clip_ms_bf16,
            measured_gpu=RTX_5090.name,
            measured_peak_vram_bytes=clip_bytes_bf16 * 2,
            peak_activation_bytes=clip_bytes_bf16,
        )
        clip_proj_bf16 = emu.project_workload(clip_wl)

        for recipe in RECIPES:
            run = all_results[res].get(recipe)
            info = all_results[res]["_info"].get(recipe, {})
            if not run:
                continue

            actually_applied = info.get("n_quantized", 0) > 0 and recipe != "bf16"
            # Halve CLIP bandwidth portion only when CLIP linears actually swapped.
            bw_mul = 0.5 if actually_applied else 1.0
            clip_ms_edge = (clip_proj_bf16.compute_limited_ms
                            + clip_proj_bf16.bandwidth_limited_ms * bw_mul)

            total_edge_ms = yolo_proj.projected_latency_ms + clip_ms_edge
            total_edge_fps = 1000.0 / total_edge_ms if total_edge_ms > 0 else 0.0
            total_5090_ms = run["mean_total_ms"]

            quality = all_results[res].get("_quality", {}).get(recipe, {})

            projections[res][recipe] = {
                "recipe": recipe,
                "actually_applied": actually_applied,
                "n_linear": info.get("n_linear", 0),
                "n_quantized": info.get("n_quantized", 0),
                "mean_total_ms_5090": total_5090_ms,
                "mean_yolo_ms_5090": run["mean_yolo_ms"],
                "mean_clip_ms_5090": run["mean_clip_ms"],
                "projected_yolo_ms_edge": yolo_proj.projected_latency_ms,
                "projected_clip_ms_edge": clip_ms_edge,
                "projected_total_ms_edge": total_edge_ms,
                "projected_fps_edge": total_edge_fps,
                "clip_compute_ms": clip_proj_bf16.compute_limited_ms,
                "clip_bandwidth_ms_bf16": clip_proj_bf16.bandwidth_limited_ms,
                "clip_bandwidth_ms_recipe": clip_proj_bf16.bandwidth_limited_ms * bw_mul,
                **quality,
            }

    return {
        "projections": projections,
        "method": ("Hybrid V2 edge projection sums per-half workloads (YOLO-seg + CLIP). "
                   "YOLO-seg is Conv-only and can't be quantized via torchao (Linear-targeted); "
                   "CLIP's ViT-B-32 has ~72 Linear layers which DO swap. Activation bytes are "
                   "halved only on the CLIP half, only when n_quantized > 0. Desktop latency is "
                   "not predictive for quantized paths — edge silicon's native FP8/INT8 MMAs "
                   "realize the bandwidth savings."),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}

    for res, clip_stem in CLIPS.items():
        log.info("=== %s (%s) ===", res, clip_stem)
        all_results[res] = {"_info": {}, "_quality": {}}
        for recipe in RECIPES:
            out_path = OUT_DIR / clip_stem / f"{recipe}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                log.info("Reusing cached %s", out_path)
                cached = json.loads(out_path.read_text())
                all_results[res][recipe] = cached["run"]
                all_results[res]["_info"][recipe] = cached["info"]
                continue

            log.info("--- recipe=%s ---", recipe)
            t_load = time.perf_counter()
            try:
                det, info = load_detector(recipe)
            except Exception as e:
                log.error("Load failed (%s %s): %s", res, recipe, e)
                all_results[res][recipe] = {"error": str(e)}
                all_results[res]["_info"][recipe] = {"recipe": recipe, "error": str(e)}
                continue
            log.info("loaded in %.1fs: %s", time.perf_counter() - t_load, info)

            try:
                run = run_one(det, clip_stem)
            except Exception as e:
                log.exception("Run failed (%s %s): %s", res, recipe, e)
                run = {"error": str(e)}

            all_results[res][recipe] = run
            all_results[res]["_info"][recipe] = info

            out_path.write_text(json.dumps({"run": run, "info": info}, indent=2))
            log.info("%s %s: total=%.2fms (YOLO %.2f + CLIP %.2f), peak VRAM %.0f MB",
                     recipe, res,
                     run.get("mean_total_ms", 0), run.get("mean_yolo_ms", 0),
                     run.get("mean_clip_ms", 0), run.get("peak_vram_mb", 0))

            # Free before next recipe
            del det
            gc.collect()
            torch.cuda.empty_cache()
            sync_cuda()

        # Compute quality deltas for this resolution
        bf16_run = all_results[res].get("bf16")
        if bf16_run and "frames" in bf16_run:
            for recipe in RECIPES:
                var_run = all_results[res].get(recipe)
                if not var_run or "frames" not in var_run:
                    continue
                if recipe == "bf16":
                    all_results[res]["_quality"][recipe] = {
                        "top1_agreement": 1.0, "top3_jaccard": 1.0,
                        "n_agreement_total": 0,
                    }
                    continue
                agree, match, total = top1_agreement(bf16_run["frames"], var_run["frames"])
                jac = top3_jaccard(bf16_run["frames"], var_run["frames"])
                all_results[res]["_quality"][recipe] = {
                    "top1_agreement": agree,
                    "top3_jaccard": jac,
                    "n_agreement_total": total,
                }

    (BAKEOFF_DIR / "hybrid_v2_summary.json").write_text(json.dumps(all_results, indent=2))
    log.info("Wrote hybrid_v2_summary.json")

    proj = project_edge(all_results)
    (BAKEOFF_DIR / "hybrid_v2_edge_projection.json").write_text(json.dumps(proj, indent=2))
    log.info("Wrote hybrid_v2_edge_projection.json")

    # Pretty print
    print()
    hdr = (f"{'Res':6s} | {'Recipe':6s} | {'Lin':>4s} {'Q':>4s} | "
           f"{'top1 ag':>7s} {'top3 J':>7s} | "
           f"{'5090 tot':>9s} {'YOLO':>6s} {'CLIP':>6s} | "
           f"{'Edge tot':>9s} {'Edge FPS':>9s}")
    print(hdr); print("-"*len(hdr))
    for res in CLIPS:
        for recipe in RECIPES:
            p = proj["projections"].get(res, {}).get(recipe)
            if not p:
                continue
            print(f"{res:6s} | {recipe:6s} | {p['n_linear']:>4d} {p['n_quantized']:>4d} | "
                  f"{p.get('top1_agreement', 0):>7.3f} {p.get('top3_jaccard', 0):>7.3f} | "
                  f"{p['mean_total_ms_5090']:>9.2f} {p['mean_yolo_ms_5090']:>6.2f} "
                  f"{p['mean_clip_ms_5090']:>6.2f} | "
                  f"{p['projected_total_ms_edge']:>9.1f} {p['projected_fps_edge']:>9.1f}")
        print()


if __name__ == "__main__":
    main()
