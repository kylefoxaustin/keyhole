"""
YOLO-seg Conv quantization bake-off — roadmap #6.

YOLO-seg is Conv-only (100 Conv2d, 0 Linear), so torchao's Linear-targeted
FP8/INT8 configs were noop in earlier bake-offs. Here we try the one actual
torchao lever for Conv models: swap_conv2d_1x1_to_linear converts 1x1 Convs
to equivalent Linear layers, which can then be quantized.

Two paths:
  1. FP8 path (the roadmap's target): torchao 0.17's version=2
     Float8DynamicActivationFloat8WeightConfig with PerTensor granularity
     asserts "input_tensor must be 1x128 scaled" during forward — a second
     tooling gap beyond the Linear-only restriction. Not fixable without
     kernel work. Path documented as blocked.
  2. INT8 path (the reachable substitute):
     Int8DynamicActivationInt8WeightConfig captures 49 of 50 swapped
     Linears (the 1x1 Convs, ~44% of YOLO-seg conv weights), runs through
     the ultralytics predict pipeline, preserves box counts.

INT8 halves activation bytes on the quantized fraction, so the edge
bandwidth savings apply proportionally to the share of the model that
was actually quantized.

Outputs:
  data/output/bakeoff/yolo_conv_quant/{clip_stem}/{recipe}.json
  data/output/bakeoff/yolo_conv_quant_summary.json
  data/output/bakeoff/yolo_conv_quant_edge_projection.json
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.emulate.npu_emulator import (
    NPUEmulator, WorkloadProfile, RTX_5090, EDGE_MPU_TARGET,
)
from scripts.bakeoff_sam_variants import BAKEOFF_DIR, sync_cuda, gpu_reset_peak, gpu_peak_mb

warnings.filterwarnings("ignore", category=UserWarning, module="torchao")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("yolo_conv_quant")

CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}
import os as _os
_DEFAULT_YOLO = "yolo11s-seg"
_YOLO_VAR = _os.environ.get("KEYHOLE_YOLO_VARIANT", _DEFAULT_YOLO)
_VARIANT_SLUG = "" if _YOLO_VAR == _DEFAULT_YOLO else f"_{_YOLO_VAR}"

OUT_DIR = BAKEOFF_DIR / f"yolo_conv_quant{_VARIANT_SLUG}"
YOLO_MODEL = f"{_YOLO_VAR}.pt"
CONF_THRESHOLD = 0.35
IOU_MATCH_THRESHOLD = 0.5


@dataclass
class SampledFrame:
    idx: int
    timestamp_sec: float
    path: str


def bbox_iou(a, b) -> float:
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


def count_quantized_linears(model) -> int:
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


def profile_yolo_structure(yolo) -> dict:
    """Describe how the underlying YOLO model is composed."""
    convs = [m for m in yolo.model.modules() if isinstance(m, torch.nn.Conv2d)]
    n_conv = len(convs)
    n_1x1 = sum(1 for c in convs if c.kernel_size == (1, 1))
    n_3x3 = sum(1 for c in convs if c.kernel_size == (3, 3))
    p_1x1 = sum(c.weight.numel() for c in convs if c.kernel_size == (1, 1))
    p_total_conv = sum(c.weight.numel() for c in convs)
    params = sum(p.numel() for p in yolo.model.parameters())
    return {
        "n_conv2d": n_conv,
        "n_conv2d_1x1": n_1x1,
        "n_conv2d_3x3": n_3x3,
        "params_m": params / 1e6,
        "weights_1x1_m": p_1x1 / 1e6,
        "weights_conv_total_m": p_total_conv / 1e6,
        "frac_weights_in_1x1": p_1x1 / p_total_conv if p_total_conv else 0.0,
    }


def load_yolo(recipe: str) -> tuple["YOLO", dict]:  # type: ignore[name-defined]
    from ultralytics import YOLO
    yolo = YOLO(YOLO_MODEL)
    # Prime the predict pipeline so Conv+BN gets fused BEFORE any weight rewrites.
    _ = yolo.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    base_struct = profile_yolo_structure(yolo)
    info = {"recipe": recipe, **base_struct, "n_swapped_linears": 0, "n_quantized": 0}

    if recipe == "bf16":
        return yolo, info

    if recipe == "int8_1x1_swap":
        from torchao.quantization import (
            swap_conv2d_1x1_to_linear, quantize_, Int8DynamicActivationInt8WeightConfig,
        )
        yolo.model.cuda()  # keep fp32 — ultralytics predict preproc emits fp32 tensors
        swap_conv2d_1x1_to_linear(yolo.model)
        info["n_swapped_linears"] = sum(
            1 for m in yolo.model.modules() if isinstance(m, torch.nn.Linear)
        )
        quantize_(yolo.model, Int8DynamicActivationInt8WeightConfig())
        info["n_quantized"] = count_quantized_linears(yolo.model)
        return yolo, info

    if recipe == "fp8_1x1_swap":
        # Exercised to record evidence of the second tooling gap. The known
        # blocker is inside _float8_addmm_impl during forward (asserts
        # "input_tensor must be 1x128 scaled"). We probe with a direct-model
        # BF16 forward so the error record in JSON reflects the true blocker,
        # not an incidental fuse/dtype mismatch from y.predict.
        from torchao.quantization import (
            swap_conv2d_1x1_to_linear, quantize_,
            Float8DynamicActivationFloat8WeightConfig, PerTensor,
        )
        yolo.model.cuda().bfloat16()
        swap_conv2d_1x1_to_linear(yolo.model)
        info["n_swapped_linears"] = sum(
            1 for m in yolo.model.modules() if isinstance(m, torch.nn.Linear)
        )
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
        quantize_(yolo.model, cfg)
        info["n_quantized"] = count_quantized_linears(yolo.model)

        # Probe forward directly to capture the real blocker message.
        probe_err = None
        try:
            x = torch.zeros(1, 3, 640, 640, dtype=torch.bfloat16, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = yolo.model(x)
        except Exception as e:
            probe_err = f"{type(e).__name__}: {str(e)[:200]}"
        info["fp8_forward_probe_error"] = probe_err or "no error (unexpected)"
        return yolo, info

    raise ValueError(f"unknown recipe: {recipe}")


def run_one(yolo, clip_stem: str) -> dict:
    clip_dir = BAKEOFF_DIR / clip_stem
    frames = [SampledFrame(**e) for e in json.loads((clip_dir / "frames.json").read_text())]

    # Warmup run on first frame
    img0 = cv2.imread(str(clip_dir / frames[0].path))
    _ = yolo.predict(img0, verbose=False, conf=CONF_THRESHOLD)

    gpu_reset_peak()
    frame_results = []
    for f in frames:
        img = cv2.imread(str(clip_dir / f.path))
        sync_cuda()
        t0 = time.perf_counter()
        res = yolo.predict(img, verbose=False, conf=CONF_THRESHOLD)
        sync_cuda()
        ms = (time.perf_counter() - t0) * 1000

        r = res[0]
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                bbox = r.boxes.xyxy[i].cpu().numpy().tolist()
                cls_id = int(r.boxes.cls[i].cpu())
                dets.append({
                    "bbox": bbox,
                    "class_id": cls_id,
                    "class_name": r.names.get(cls_id, f"class_{cls_id}"),
                    "confidence": float(r.boxes.conf[i].cpu()),
                })
        frame_results.append({"frame_idx": f.idx, "latency_ms": ms, "detections": dets})

    return {
        "clip_stem": clip_stem,
        "frames": frame_results,
        "mean_frame_ms": float(np.mean([fr["latency_ms"] for fr in frame_results])),
        "mean_det_per_frame": float(np.mean([len(fr["detections"]) for fr in frame_results])),
        "peak_vram_mb": gpu_peak_mb(),
    }


def detection_stability(ref_frames, var_frames) -> dict:
    """Per-frame IoU-match between reference and variant detections. Reports:
    box_recall (fraction of ref boxes with an IoU>0.5 match, same class),
    mean matched IoU, false-positive rate (var boxes with no match)."""
    match_ious = []
    total_ref, total_matched = 0, 0
    total_var, total_fp = 0, 0
    for fr_ref, fr_var in zip(ref_frames, var_frames):
        dr, dv = fr_ref["detections"], fr_var["detections"]
        used_v = set()
        total_ref += len(dr)
        total_var += len(dv)
        for da in dr:
            best, best_iou = -1, IOU_MATCH_THRESHOLD
            for j, db in enumerate(dv):
                if j in used_v:
                    continue
                if da["class_name"] != db["class_name"]:
                    continue
                iou = bbox_iou(da["bbox"], db["bbox"])
                if iou > best_iou:
                    best, best_iou = j, iou
            if best >= 0:
                used_v.add(best)
                total_matched += 1
                match_ious.append(best_iou)
        total_fp += len(dv) - len(used_v)
    return {
        "box_recall": total_matched / total_ref if total_ref else 1.0,
        "mean_matched_iou": float(np.mean(match_ious)) if match_ious else 0.0,
        "n_ref_boxes": total_ref,
        "n_var_boxes": total_var,
        "n_matched": total_matched,
        "n_fp": total_fp,
    }


def project_edge(all_results: dict) -> dict:
    """Edge projection for YOLO-seg per recipe. Halves activation bytes on the
    quantized fraction only (4x of conv weights that are 1x1 and actually swapped)."""
    emu = NPUEmulator(reference=RTX_5090, target=EDGE_MPU_TARGET)
    projections = {}

    for res, clip_stem in CLIPS.items():
        projections[res] = {}
        bf16 = all_results[res].get("bf16")
        if not bf16:
            continue
        info_bf16 = all_results[res]["_info"]["bf16"]

        params = int(info_bf16["params_m"] * 1e6)
        model_bytes_bf16 = params * 2
        bf16_ms = bf16["mean_frame_ms"]

        wl = WorkloadProfile(
            stage_name="yolo_seg",
            model_name=_YOLO_VAR,
            param_count=params,
            model_size_bytes=model_bytes_bf16,
            precision="bf16",
            measured_latency_ms=bf16_ms,
            measured_gpu_kernel_ms=bf16_ms,
            measured_gpu=RTX_5090.name,
            measured_peak_vram_bytes=int(bf16["peak_vram_mb"] * 1e6),
            peak_activation_bytes=model_bytes_bf16,
        )
        proj_bf16 = emu.project_workload(wl)

        for recipe in ["bf16", "int8_1x1_swap", "fp8_1x1_swap"]:
            run = all_results[res].get(recipe)
            info = all_results[res]["_info"].get(recipe, {})
            quality = all_results[res].get("_quality", {}).get(recipe, {})
            if not run or "error" in run:
                projections[res][recipe] = {
                    "recipe": recipe,
                    "error": (run or {}).get("error", "not_run"),
                    "info": info,
                }
                continue

            # Fraction of activation traffic quantization touches =
            # (quantized linear weights) / (total conv weights before swap)
            frac = 0.0
            if info.get("n_quantized", 0) > 0 and info.get("weights_conv_total_m", 0) > 0:
                # swap_conv2d_1x1_to_linear fully swapped 1x1 Convs; the fraction
                # of conv params that got quantized is the same as the fraction
                # of conv weights that are 1x1 (times n_quantized / n_swapped).
                applied_ratio = (info["n_quantized"] /
                                 max(1, info.get("n_swapped_linears", 1)))
                frac = info["frac_weights_in_1x1"] * applied_ratio

            # Bandwidth savings apply ONLY to the quantized fraction.
            # bw_savings = 0.5 on the quantized fraction, 1.0 on the rest.
            bw_multiplier = 1.0 - 0.5 * frac
            adjusted_ms = (proj_bf16.compute_limited_ms
                           + proj_bf16.bandwidth_limited_ms * bw_multiplier)
            adjusted_fps = 1000.0 / adjusted_ms if adjusted_ms > 0 else 0.0

            projections[res][recipe] = {
                "recipe": recipe,
                "n_conv2d": info.get("n_conv2d", 0),
                "n_conv2d_1x1": info.get("n_conv2d_1x1", 0),
                "n_swapped_linears": info.get("n_swapped_linears", 0),
                "n_quantized": info.get("n_quantized", 0),
                "frac_conv_weights_quantized": frac,
                "mean_frame_ms_5090": run["mean_frame_ms"],
                "projected_ms_edge_bf16": proj_bf16.projected_latency_ms,
                "projected_ms_edge_recipe": adjusted_ms,
                "projected_fps_edge_recipe": adjusted_fps,
                "compute_limited_ms": proj_bf16.compute_limited_ms,
                "bandwidth_limited_ms_bf16": proj_bf16.bandwidth_limited_ms,
                "bandwidth_limited_ms_recipe": proj_bf16.bandwidth_limited_ms * bw_multiplier,
                **quality,
            }

    return {
        "projections": projections,
        "method": (
            "YOLO-seg edge projection. BF16 measured on 5090 → projected via "
            "NPUEmulator. For quantized recipes, bandwidth-limited ms scaled by "
            "(1 - 0.5 * frac_conv_weights_quantized) since only the quantized "
            "fraction sees halved activation bytes. 1x1 Convs carry ~44% of "
            "conv weights in yolo11s-seg; int8_1x1_swap captures ~49 of 50."
        ),
        "fp8_blocked_note": (
            "torchao 0.17 Float8DynamicActivationFloat8WeightConfig with "
            "PerTensor granularity asserts 'input_tensor must be 1x128 scaled' "
            "inside _float8_addmm_impl during forward — a second tooling gap "
            "beyond the Linear-only restriction. version=1 is rejected at "
            "runtime; PerRow fails with 'Only bf16 and fp16 high precision "
            "output types are supported for row-wise scaling'. Full FP8 on "
            "YOLO-seg still requires custom kernels or transformer_engine."
        ),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}

    for res, clip_stem in CLIPS.items():
        log.info("=== %s (%s) ===", res, clip_stem)
        all_results[res] = {"_info": {}, "_quality": {}}

        for recipe in ["bf16", "int8_1x1_swap", "fp8_1x1_swap"]:
            out_path = OUT_DIR / clip_stem / f"{recipe}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                cached = json.loads(out_path.read_text())
                all_results[res][recipe] = cached["run"]
                all_results[res]["_info"][recipe] = cached["info"]
                log.info("Reusing cached %s", out_path)
                continue

            log.info("--- recipe=%s ---", recipe)
            try:
                yolo, info = load_yolo(recipe)
            except Exception as e:
                log.exception("Load failed")
                all_results[res][recipe] = {"error": f"load: {e}"}
                all_results[res]["_info"][recipe] = {"recipe": recipe, "error": str(e)}
                continue

            log.info("load info: %s", {k: v for k, v in info.items()
                                        if k in ("recipe", "n_conv2d", "n_conv2d_1x1",
                                                 "n_swapped_linears", "n_quantized",
                                                 "frac_weights_in_1x1")})
            try:
                run = run_one(yolo, clip_stem)
            except Exception as e:
                log.exception("Run failed")
                run = {"error": f"run: {type(e).__name__}: {e}"}

            all_results[res][recipe] = run
            all_results[res]["_info"][recipe] = info
            out_path.write_text(json.dumps({"run": run, "info": info}, indent=2))
            if "error" not in run:
                log.info("%s %s: %.2f ms/frame, %.1f dets/frame",
                         recipe, res, run["mean_frame_ms"], run["mean_det_per_frame"])
            else:
                log.info("%s %s: ERROR %s", recipe, res, run["error"])

            del yolo
            gc.collect()
            torch.cuda.empty_cache()
            sync_cuda()

        # Quality deltas — variant vs bf16 per-frame detections
        bf16_run = all_results[res].get("bf16")
        if bf16_run and "frames" in bf16_run:
            for recipe in ["bf16", "int8_1x1_swap", "fp8_1x1_swap"]:
                var = all_results[res].get(recipe)
                if not var or "frames" not in var:
                    continue
                if recipe == "bf16":
                    all_results[res]["_quality"][recipe] = {
                        "box_recall": 1.0, "mean_matched_iou": 1.0,
                        "n_ref_boxes": 0, "n_var_boxes": 0,
                        "n_matched": 0, "n_fp": 0,
                    }
                    continue
                all_results[res]["_quality"][recipe] = detection_stability(
                    bf16_run["frames"], var["frames"])

    (BAKEOFF_DIR / f"yolo_conv_quant{_VARIANT_SLUG}_summary.json").write_text(
        json.dumps(all_results, indent=2))
    log.info("Wrote yolo_conv_quant_summary.json")

    proj = project_edge(all_results)
    (BAKEOFF_DIR / f"yolo_conv_quant{_VARIANT_SLUG}_edge_projection.json").write_text(
        json.dumps(proj, indent=2))
    log.info("Wrote yolo_conv_quant_edge_projection.json")

    # Pretty print
    print()
    hdr = (f"{'Res':6s} | {'Recipe':18s} | {'swap':>4s} {'Q':>3s} {'%conv':>5s} | "
           f"{'recall':>6s} {'match IoU':>9s} | "
           f"{'5090 ms':>8s} | {'Edge bf16':>9s} {'Edge Q':>7s} {'FPS':>5s}")
    print(hdr); print("-"*len(hdr))
    for res in CLIPS:
        for recipe in ["bf16", "int8_1x1_swap", "fp8_1x1_swap"]:
            p = proj["projections"].get(res, {}).get(recipe)
            if not p or "error" in p:
                err = (p or {}).get("error", "?") if p else "missing"
                print(f"{res:6s} | {recipe:18s} | {'—':>4s} {'—':>3s} {'—':>5s} | "
                      f"{'—':>6s} {'—':>9s} | {'—':>8s} | {'—':>9s} {'—':>7s} {'—':>5s}  ({err[:40]})")
                continue
            print(f"{res:6s} | {recipe:18s} | {p['n_swapped_linears']:>4d} "
                  f"{p['n_quantized']:>3d} {100*p['frac_conv_weights_quantized']:>4.1f}% | "
                  f"{p.get('box_recall', 0):>6.3f} {p.get('mean_matched_iou', 0):>9.3f} | "
                  f"{p['mean_frame_ms_5090']:>8.2f} | "
                  f"{p['projected_ms_edge_bf16']:>9.1f} "
                  f"{p['projected_ms_edge_recipe']:>7.1f} "
                  f"{p['projected_fps_edge_recipe']:>5.1f}")
        print()


if __name__ == "__main__":
    main()
