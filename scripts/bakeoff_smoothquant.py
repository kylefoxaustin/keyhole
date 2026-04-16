"""
SmoothQuant + plain-INT8 activation quantization on the bake-off winners.

Applies two quant recipes to EfficientSAM-Small and YOLO-seg-s:
  - int8        : torchao Int8DynamicActivationInt8WeightConfig (no smoothing)
  - smoothquant : torchao SmoothQuantConfig(Int8...) with alpha=0.5
                  (PREPARE + calibrate + CONVERT flow)

Measures per-frame mask IoU vs SAM 3 references on the cached bake-off frames
and re-projects edge FPS under halved-activation-traffic assumption.

Outputs:
  data/output/bakeoff/smoothquant/{clip}/results/{name}_{recipe}.json
  data/output/bakeoff/smoothquant_summary.json
  data/output/bakeoff/smoothquant_edge_projection.json

Sibling to scripts/bakeoff_fp8.py — re-uses the same contestant classes.
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.emulate.npu_emulator import (
    NPUEmulator, WorkloadProfile, RTX_5090, EDGE_MPU_TARGET,
)
from scripts.bakeoff_sam_variants import (
    EfficientSAMContestant, YoloSegContestant, BAKEOFF_DIR,
    Prompt, RefMask, SampledFrame,
    mask_iou, gpu_reset_peak, gpu_peak_mb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("smoothquant")

CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}
OUT_DIR = BAKEOFF_DIR / "smoothquant"
CALIB_FRAMES = 5          # how many frames to calibrate on (SmoothQuant only)


def count_quantized_linears(model: torch.nn.Module) -> int:
    """Detect torchao-swapped Linears via weight tensor subclass name."""
    n = 0
    for m in model.modules():
        w = getattr(m, "weight", None)
        if w is None:
            continue
        tname = type(w).__name__
        if any(s in tname for s in ("LinearActivationQuantized", "QuantizedTensor", "Float8")):
            n += 1
    return n


def apply_int8(model: torch.nn.Module) -> dict:
    """Plain INT8 dynamic-act + int8-weight via torchao."""
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    n_conv = sum(1 for m in model.modules()
                 if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)))
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    return {
        "recipe": "int8",
        "n_linear": n_linear,
        "n_conv": n_conv,
        "n_quantized": count_quantized_linears(model),
    }


def apply_smoothquant(model: torch.nn.Module, calibrate_fn, alpha: float = 0.5) -> dict:
    """SmoothQuant: PREPARE + calibrate + CONVERT (INT8 base)."""
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
    from torchao.prototype.smoothquant import SmoothQuantConfig
    from torchao.quantization.quantize_.common.quantization_step import QuantizationStep

    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    n_conv = sum(1 for m in model.modules()
                 if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)))

    # Step 1: PREPARE — insert observers
    cfg = SmoothQuantConfig(
        base_config=Int8DynamicActivationInt8WeightConfig(),
        step=QuantizationStep.PREPARE,
        alpha=alpha,
    )
    quantize_(model, cfg)
    n_observers = sum(
        1 for _, m in model.named_modules()
        if type(m).__name__ == "SmoothQuantObservedLinear"
    )
    log.info("  SmoothQuant PREPARE: %d observers installed", n_observers)

    # Step 2: Calibrate (model in observer-mode runs forward passes)
    log.info("  Calibrating on %d frames...", CALIB_FRAMES)
    calibrate_fn(model)

    # Step 3: CONVERT — apply smoothing scale + replace with INT8
    cfg.step = QuantizationStep.CONVERT
    quantize_(model, cfg)

    return {
        "recipe": "smoothquant",
        "alpha": alpha,
        "n_linear": n_linear,
        "n_conv": n_conv,
        "n_observers_installed": n_observers,
        "n_quantized": count_quantized_linears(model),
    }


def run_recipe(
    contestant,
    name: str,
    recipe: str,
    clip_stem: str,
) -> dict:
    """Run one (contestant, recipe, clip) — returns dict of results + info."""
    clip_dir = BAKEOFF_DIR / clip_stem
    out_dir = OUT_DIR / clip_stem / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_{recipe}.json"
    if out_path.exists():
        log.info("Reusing cached: %s", out_path)
        return json.loads(out_path.read_text())

    frames = [SampledFrame(**e) for e in json.loads((clip_dir / "frames.json").read_text())]
    prompts_raw = json.loads((clip_dir / "prompts.json").read_text())
    prompts = {int(k): [Prompt(**p) for p in v] for k, v in prompts_raw.items()}
    refs_raw = json.loads((clip_dir / "refs_meta.json").read_text())
    refs = {int(k): [RefMask(**r) for r in v] for k, v in refs_raw.items()}

    log.info("=== %s / %s / %s ===", name, recipe, clip_stem)
    contestant.load()

    # Reach underlying nn.Module
    if name.startswith("efficientsam"):
        target = contestant.model
        target.bfloat16()

        # Cast top-level inputs to bf16
        def _cast_hook(module, inputs):
            return tuple(inp.bfloat16() if torch.is_tensor(inp) and inp.is_floating_point()
                         and inp.dtype == torch.float32 else inp
                         for inp in inputs)
        target.register_forward_pre_hook(_cast_hook)

        # Wrap forward with autocast(bf16) to coerce intermediate fp32 activations
        _orig_forward = target.forward
        def _autocast_forward(*args, **kwargs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return _orig_forward(*args, **kwargs)
        target.forward = _autocast_forward
    elif name == "yolo_seg" and hasattr(contestant.model, "model"):
        target = contestant.model.model
    else:
        raise RuntimeError(f"Don't know how to reach nn.Module for {name}")

    # Calibrator: run model on a few prompted frames
    def calibrate(_model):
        for i, f in enumerate(frames[:CALIB_FRAMES]):
            img = cv2.imread(str(clip_dir / f.path))
            boxes = [p.box for p in prompts[f.idx]]
            if not boxes:
                continue
            try:
                contestant.infer_frame(img, boxes)
            except Exception as e:
                log.warning("  Calibration on frame %d failed: %s", f.idx, e)

    # Apply recipe
    if recipe == "int8":
        info = apply_int8(target)
    elif recipe == "smoothquant":
        info = apply_smoothquant(target, calibrate, alpha=0.5)
    else:
        raise ValueError(recipe)
    log.info("Recipe info: %s", info)

    # Warm up under quantized graph
    try:
        img = cv2.imread(str(clip_dir / frames[0].path))
        boxes = [p.box for p in prompts[frames[0].idx]][:4]
        if boxes:
            contestant.infer_frame(img, boxes)
    except Exception as e:
        log.warning("Post-quant warm-up failed: %s", e)

    gpu_reset_peak()

    # Run benchmark
    frame_results = []
    all_ious = []
    all_per_box_ms = []
    total_boxes = 0
    for f in frames:
        img = cv2.imread(str(clip_dir / f.path))
        fp = prompts.get(f.idx, [])
        boxes = [p.box for p in fp]
        if not boxes:
            frame_results.append({"frame_idx": f.idx, "latency_ms": 0.0,
                                  "per_box_latency_ms": 0.0, "n_boxes": 0,
                                  "box_results": []})
            continue
        try:
            masks, latency_ms = contestant.infer_frame(img, boxes)
        except Exception as e:
            log.error("Inference failed on frame %d: %s", f.idx, e)
            result = {"error": str(e), "info": info, "name": name,
                      "recipe": recipe, "clip_stem": clip_stem}
            out_path.write_text(json.dumps(result, indent=2))
            contestant.unload()
            gc.collect(); torch.cuda.empty_cache()
            return result

        per_box_ms = latency_ms / max(1, len(boxes))
        ref_by_pidx = {r.prompt_idx: r for r in refs.get(f.idx, [])}
        box_rs = []
        for pi, m in enumerate(masks):
            iou_val = None
            if m is not None and pi in ref_by_pidx:
                ref_mask = np.load(clip_dir / ref_by_pidx[pi].mask_path)
                iou_val = mask_iou(m, ref_mask)
                all_ious.append(iou_val)
            box_rs.append({"prompt_idx": pi, "mask_present": bool(m is not None),
                           "iou_vs_ref": iou_val})
        frame_results.append({"frame_idx": f.idx, "latency_ms": latency_ms,
                              "per_box_latency_ms": per_box_ms, "n_boxes": len(boxes),
                              "box_results": box_rs})
        total_boxes += len(boxes)
        all_per_box_ms.extend([per_box_ms] * len(boxes))
        iou_str = (f"{np.mean([b['iou_vs_ref'] for b in box_rs if b['iou_vs_ref'] is not None]):.3f}"
                   if any(b["iou_vs_ref"] is not None for b in box_rs) else "n/a")
        log.info("  %s/%s frame %d: %d boxes, %.1fms (%.2fms/box), IoU=%s",
                 name, recipe, f.idx, len(boxes), latency_ms, per_box_ms, iou_str)

    peak_vram = gpu_peak_mb()
    contestant.unload()
    gc.collect(); torch.cuda.empty_cache()

    out = {
        "name": name,
        "recipe": recipe,
        "clip_stem": clip_stem,
        "info": info,
        "params_m": contestant.params() / 1e6,
        "peak_vram_mb": peak_vram,
        "frames": frame_results,
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "median_iou": float(np.median(all_ious)) if all_ious else 0.0,
        "mean_per_box_ms": float(np.mean(all_per_box_ms)) if all_per_box_ms else 0.0,
        "mean_frame_ms": float(np.mean([fr["latency_ms"] for fr in frame_results if fr["n_boxes"] > 0]))
                          if any(fr["n_boxes"] > 0 for fr in frame_results) else 0.0,
        "n_iou_samples": len(all_ious),
        "n_box_inferences": total_boxes,
    }
    out_path.write_text(json.dumps(out, indent=2))
    log.info("  → mean IoU=%.3f (n=%d), mean %.2f ms/box, peak VRAM %.0f MB, quantized=%d",
             out["mean_iou"], out["n_iou_samples"], out["mean_per_box_ms"],
             out["peak_vram_mb"], info.get("n_quantized", 0))
    return out


def project_edge(all_results: dict) -> dict:
    """Re-project edge latency with halved activation traffic (if recipe actually quantized)."""
    emu = NPUEmulator(reference=RTX_5090, target=EDGE_MPU_TARGET)
    projections = {}
    for res, clip_stem in CLIPS.items():
        clip_dir = BAKEOFF_DIR / clip_stem
        projections[res] = {}
        for name in ["efficientsam_small", "yolo_seg"]:
            for recipe in ["int8", "smoothquant"]:
                r = all_results.get(res, {}).get(name, {}).get(recipe)
                if not r or "error" in r:
                    continue
                n_quantized = r.get("info", {}).get("n_quantized", 0)
                actually_applied = n_quantized > 0

                bf16 = json.loads((clip_dir / "results" / f"{name}.json").read_text())
                bf16_frame_ms = float(np.mean(
                    [fr["latency_ms"] for fr in bf16["frames"] if fr["n_boxes"] > 0]
                ))

                params_m = r["params_m"]
                # INT8 model bytes = 1B/param
                model_bytes = int(params_m * 1e6 * (1 if actually_applied else 2))
                bf16_peak_vram = bf16["peak_vram_mb"]
                bf16_model_bytes = int(params_m * 1e6 * 2)
                bf16_act_bytes = max(0, int(bf16_peak_vram * 1e6 - bf16_model_bytes))
                int8_act_bytes = bf16_act_bytes // 2 if actually_applied else bf16_act_bytes

                wl = WorkloadProfile(
                    stage_name=f"{name}_{recipe}",
                    model_name=f"{name}_{recipe}",
                    param_count=int(params_m * 1e6),
                    model_size_bytes=model_bytes,
                    precision="int8" if actually_applied else "bf16",
                    gflops_per_inference=0.0,
                    measured_latency_ms=bf16_frame_ms,
                    measured_gpu_kernel_ms=bf16_frame_ms,
                    measured_gpu=RTX_5090.name,
                    measured_peak_vram_bytes=int(model_bytes + int8_act_bytes),
                    peak_activation_bytes=int8_act_bytes,
                )
                proj = emu.project_workload(wl)

                bw_mult = 0.5 if actually_applied else 1.0
                adj_ms = proj.compute_limited_ms + (proj.bandwidth_limited_ms * bw_mult)
                adj_fps = 1000.0 / adj_ms if adj_ms > 0 else 0.0

                projections[res].setdefault(name, {})[recipe] = {
                    "params_m": params_m,
                    "recipe_applied": actually_applied,
                    "n_quantized": n_quantized,
                    "mean_iou_recipe": r["mean_iou"],
                    "mean_iou_bf16": bf16["mean_iou"],
                    "iou_delta": r["mean_iou"] - bf16["mean_iou"],
                    "measured_frame_ms_5090_bf16": bf16_frame_ms,
                    "measured_frame_ms_5090_recipe": r.get("mean_frame_ms", 0.0),
                    "projected_ms_edge_bf16": proj.compute_limited_ms + proj.bandwidth_limited_ms,
                    "projected_ms_edge_recipe": adj_ms,
                    "projected_fps_edge_recipe": adj_fps,
                    "compute_limited_ms": proj.compute_limited_ms,
                    "bandwidth_limited_ms_bf16": proj.bandwidth_limited_ms,
                    "bandwidth_limited_ms_recipe": proj.bandwidth_limited_ms * bw_mult,
                }

    out = {
        "projections": projections,
        "method": ("SmoothQuant + plain INT8 via torchao. Edge projection halves "
                   "activation bytes ONLY for recipes that actually swapped Linears. "
                   "Desktop latency is not predictive — edge silicon's native INT8 MMA "
                   "paths realize the bandwidth savings."),
    }
    (BAKEOFF_DIR / "smoothquant_edge_projection.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}
    for res, clip_stem in CLIPS.items():
        all_results[res] = {}
        for name, builder in [
            ("efficientsam_small", lambda: EfficientSAMContestant("small")),
            ("yolo_seg",           lambda: YoloSegContestant("yolo11s-seg.pt")),
        ]:
            all_results[res][name] = {}
            for recipe in ["int8", "smoothquant"]:
                try:
                    r = run_recipe(builder(), name, recipe, clip_stem)
                    all_results[res][name][recipe] = r
                except Exception as e:
                    log.error("%s/%s/%s failed: %s", name, recipe, res, e)
                    all_results[res][name][recipe] = {"error": str(e), "name": name}

    (BAKEOFF_DIR / "smoothquant_summary.json").write_text(json.dumps(all_results, indent=2))
    projections = project_edge(all_results)

    print()
    print(f"{'Model':22s} | {'Recipe':11s} | {'Res':4s} | {'applied?':14s} | "
          f"{'IoU bf16':>8s} {'IoU q':>8s} {'Δ':>7s} | {'Edge FPS':>8s}")
    print("-"*112)
    for res in CLIPS:
        for name in ["efficientsam_small", "yolo_seg"]:
            for recipe in ["int8", "smoothquant"]:
                p = projections["projections"].get(res, {}).get(name, {}).get(recipe)
                if not p:
                    continue
                applied = "YES" if p["recipe_applied"] else "NO (Conv-only)"
                print(f"{name:22s} | {recipe:11s} | {res:4s} | {applied:14s} | "
                      f"{p['mean_iou_bf16']:8.3f} {p['mean_iou_recipe']:8.3f} {p['iou_delta']:+7.3f} | "
                      f"{p['projected_fps_edge_recipe']:8.1f}")
        print()


if __name__ == "__main__":
    main()
