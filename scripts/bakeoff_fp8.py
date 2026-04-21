"""
FP8 activation quantization — does it preserve quality and (per edge projection)
halve the bandwidth-bound latency on LPDDR5X?

Quantizes EfficientSAM-Small and YOLO-seg-s via torchao's
Float8DynamicActivationFloat8WeightConfig, reruns on the cached bake-off frames,
records per-frame latency + mask IoU vs the SAM 3 reference masks, and re-projects
edge latency with halved activation bytes.

Outputs:
  data/output/bakeoff/fp8/{clip_stem}/results/{name}_fp8.json
  data/output/bakeoff/fp8_summary.json
  data/output/bakeoff/fp8_edge_projection.json
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
    iou_xyxy, mask_iou, sync_cuda, gpu_reset_peak, gpu_peak_mb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fp8")

CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}
import os as _os_pf
_yv = _os_pf.environ.get("KEYHOLE_YOLO_VARIANT", "yolo11s-seg")
_vs = "" if _yv == "yolo11s-seg" else f"_{_yv}"
OUT_DIR = BAKEOFF_DIR / f"fp8{_vs}"


def fp8_quantize(model: torch.nn.Module) -> dict:
    """Apply torchao FP8 dynamic-act / fp8-weight to nn.Linear layers.
    Returns info dict (how many modules were swapped)."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig, quantize_, PerRow,
    )

    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    n_conv = sum(1 for m in model.modules()
                 if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)))

    # PerTensor is more permissive w.r.t. input/output dtypes than PerRow,
    # and works with torchao 0.17 + autocast(bf16).
    from torchao.quantization import PerTensor
    cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
    quantize_(model, cfg)

    # torchao wraps Linear weights in a Float8Tensor (tensor subclass). Detect by
    # checking for the wrapping — cleanest is to look at qdata or fp8 internal data.
    n_fp8 = 0
    for m in model.modules():
        w = getattr(m, "weight", None)
        if w is None:
            continue
        # Float8Tensor subclass has an '_data' or similar attribute in some versions;
        # simplest: check repr / type name.
        tname = type(w).__name__
        if "Float8" in tname or (hasattr(w, "dtype") and w.dtype == torch.float8_e4m3fn):
            n_fp8 += 1

    return {
        "n_linear_before": n_linear,
        "n_conv_before": n_conv,
        "n_fp8_weights_after": n_fp8,
    }


def run_fp8_contestant(contestant, name: str, clip_stem: str) -> dict:
    """Run a single FP8-quantized contestant on one clip. Returns result dict."""
    clip_dir = BAKEOFF_DIR / clip_stem
    out_dir = OUT_DIR / clip_stem / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_fp8.json"
    if out_path.exists():
        log.info("Reusing cached FP8 result: %s", out_path)
        return json.loads(out_path.read_text())

    frames = [SampledFrame(**e) for e in json.loads((clip_dir / "frames.json").read_text())]
    prompts_raw = json.loads((clip_dir / "prompts.json").read_text())
    prompts = {int(k): [Prompt(**p) for p in v] for k, v in prompts_raw.items()}
    refs_raw = json.loads((clip_dir / "refs_meta.json").read_text())
    refs = {int(k): [RefMask(**r) for r in v] for k, v in refs_raw.items()}

    log.info("Loading %s (bf16) then quantizing to FP8...", name)
    contestant.load()
    # Apply FP8 to the actual underlying nn.Module. Attribute naming differs per contestant.
    if name.startswith("efficientsam"):
        target = contestant.model
        target.bfloat16()
        # Cast top-level inputs to bf16.
        def _cast_hook(module, inputs):
            return tuple(inp.bfloat16() if torch.is_tensor(inp) and inp.is_floating_point()
                         and inp.dtype == torch.float32 else inp
                         for inp in inputs)
        target.register_forward_pre_hook(_cast_hook)
        info = fp8_quantize(target)
        # Wrap forward with autocast(bf16) to coerce any fp32 intermediate activations
        # at op boundaries (catches F.linear, addmm, etc.).
        _orig_forward = target.forward
        def _autocast_forward(*args, **kwargs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return _orig_forward(*args, **kwargs)
        target.forward = _autocast_forward
    elif name == "yolo_seg" and hasattr(contestant.model, "model"):
        info = fp8_quantize(contestant.model.model)
    elif hasattr(contestant, "model") and isinstance(contestant.model, torch.nn.Module):
        info = fp8_quantize(contestant.model)
    else:
        raise RuntimeError(f"Don't know how to reach nn.Module for {name}")
    log.info("FP8 quantize info: %s", info)

    # Re-warm after swap
    try:
        img = cv2.imread(str(clip_dir / frames[0].path))
        boxes = [p.box for p in prompts[frames[0].idx]][:4]
        if boxes:
            contestant.infer_frame(img, boxes)
    except Exception as e:
        log.warning("FP8 warm-up failed: %s", e)

    gpu_reset_peak()

    # Run per frame
    frame_results = []
    all_ious = []
    all_per_box_ms = []
    total_boxes = 0
    for f in frames:
        img = cv2.imread(str(clip_dir / f.path))
        frame_prompts = prompts.get(f.idx, [])
        boxes = [p.box for p in frame_prompts]
        if not boxes:
            frame_results.append({"frame_idx": f.idx, "latency_ms": 0.0,
                                  "per_box_latency_ms": 0.0, "n_boxes": 0,
                                  "box_results": []})
            continue

        try:
            masks, latency_ms = contestant.infer_frame(img, boxes)
        except Exception as e:
            log.error("FP8 inference failed on frame %d: %s", f.idx, e)
            return {"error": str(e), "fp8_info": info, "name": name, "clip_stem": clip_stem}

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
        log.info("%s frame %d: %d boxes, %.1fms (%.2fms/box), IoU@matches=%.3f",
                 name, f.idx, len(boxes), latency_ms, per_box_ms,
                 float(np.mean([b["iou_vs_ref"] for b in box_rs if b["iou_vs_ref"] is not None]))
                 if any(b["iou_vs_ref"] is not None for b in box_rs) else float("nan"))

    peak_vram = gpu_peak_mb()
    contestant.unload()
    gc.collect()
    torch.cuda.empty_cache()

    out = {
        "name": name,
        "clip_stem": clip_stem,
        "fp8_info": info,
        "params_m": contestant.params() / 1e6,  # pre-quantize param count still valid
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
    log.info("%s FP8: mean IoU=%.3f (n=%d), mean %.2f ms/box, peak VRAM %.0f MB",
             name, out["mean_iou"], out["n_iou_samples"], out["mean_per_box_ms"],
             out["peak_vram_mb"])
    return out


def project_edge_fp8(fp8_results: dict) -> dict:
    """Re-project edge latency with halved activation bytes + halved model size."""
    emu = NPUEmulator(reference=RTX_5090, target=EDGE_MPU_TARGET)
    projections = {}

    # Also load matching bf16 baseline for side-by-side
    for res, clip_stem in CLIPS.items():
        projections[res] = {}
        clip_dir = BAKEOFF_DIR / clip_stem
        for name in ["efficientsam_small", "yolo_seg"]:
            fp8 = fp8_results[res].get(name)
            if not fp8 or "error" in fp8:
                continue
            # bf16 baseline
            bf16 = json.loads((clip_dir / "results" / f"{name}.json").read_text())
            bf16_frame_ms = float(np.mean([fr["latency_ms"] for fr in bf16["frames"] if fr["n_boxes"] > 0]))

            # If torchao applied 0 FP8 swaps (e.g. Conv-only model), don't pretend
            # FP8 helped — the "FP8 run" is just bf16.
            n_fp8_swapped = fp8.get("fp8_info", {}).get("n_fp8_weights_after", 0)
            fp8_actually_applied = n_fp8_swapped > 0

            # FP8: halve model bytes + activation bytes (only if actually quantized)
            params_m = fp8["params_m"]
            fp8_model_bytes = int(params_m * 1e6 * (1 if fp8_actually_applied else 2))
            fp8_vram = fp8["peak_vram_mb"]
            # Activation bytes under fp8 = half the bf16 activation bytes
            bf16_peak_vram = bf16["peak_vram_mb"]
            bf16_model_bytes = int(params_m * 1e6 * 2)
            bf16_act_bytes = max(0, int(bf16_peak_vram * 1e6 - bf16_model_bytes))
            # Activation traffic halved only if FP8 is actually in play
            fp8_act_bytes = bf16_act_bytes // 2 if fp8_actually_applied else bf16_act_bytes

            # Emulator workload — use FP8 characteristics; measured latency still
            # comes from a bf16 run (desktop FP8 path in torchao is not representative
            # of edge FP8 silicon), so we treat bf16 latency as the reference
            # AND scale the bandwidth-bound portion down by the smaller activation bytes.
            wl = WorkloadProfile(
                stage_name=f"{name}_fp8",
                model_name=f"{name}_fp8",
                param_count=int(params_m * 1e6),
                model_size_bytes=fp8_model_bytes,
                precision="fp8_e4m3",
                gflops_per_inference=0.0,
                measured_latency_ms=bf16_frame_ms,
                measured_gpu_kernel_ms=bf16_frame_ms,
                measured_gpu=RTX_5090.name,
                measured_peak_vram_bytes=int(fp8_model_bytes + fp8_act_bytes),
                peak_activation_bytes=fp8_act_bytes,
            )
            r = emu.project_workload(wl)

            # For an apples-to-apples "with FP8 bandwidth savings" projection:
            # the emulator scales by reference/target bandwidth ratio, but doesn't
            # know that the per-forward bytes dropped. Approximation:
            # halve the bandwidth-limited portion manually (since act traffic halves).
            # This matches the deck's ~2x reduction claim.
            # Halve the bandwidth-limited portion only when FP8 was actually
            # applied (else no bandwidth savings — torchao didn't touch the model).
            bw_multiplier = 0.5 if fp8_actually_applied else 1.0
            adjusted_ms = r.compute_limited_ms + (r.bandwidth_limited_ms * bw_multiplier)
            adjusted_fps = 1000.0 / adjusted_ms if adjusted_ms > 0 else 0.0

            projections[res][name] = {
                "params_m": params_m,
                "fp8_actually_applied": fp8_actually_applied,
                "n_fp8_weights_swapped": n_fp8_swapped,
                "mean_iou_fp8": fp8["mean_iou"],
                "mean_iou_bf16": bf16["mean_iou"],
                "iou_delta": fp8["mean_iou"] - bf16["mean_iou"],
                "measured_frame_ms_5090_bf16": bf16_frame_ms,
                "measured_frame_ms_5090_fp8": fp8.get("mean_frame_ms", 0.0),
                "projected_ms_edge_bf16": r.projected_latency_ms + r.bandwidth_limited_ms,  # nominal
                "projected_ms_edge_fp8": adjusted_ms,
                "projected_fps_edge_fp8": adjusted_fps,
                "compute_limited_ms": r.compute_limited_ms,
                "bandwidth_limited_ms_bf16": r.bandwidth_limited_ms,
                "bandwidth_limited_ms_fp8": r.bandwidth_limited_ms * 0.5,
                "memory_headroom_mb": r.memory_headroom_bytes / 1e6,
            }

    # Also attach the pre-existing bf16 projection for side-by-side
    bf16_edge_path = BAKEOFF_DIR / "edge_projection.json"
    bf16_edge = json.loads(bf16_edge_path.read_text()) if bf16_edge_path.exists() else None
    out = {
        "fp8_projections": projections,
        "bf16_projections": bf16_edge["projections"] if bf16_edge else None,
        "method": ("FP8 edge projection reuses bf16 RTX 5090 latency as reference, "
                   "halves model bytes (fp8 weights = 1B/param), halves bandwidth-limited "
                   "portion (act traffic halved). Desktop FP8 path is kernel-immature, "
                   "not predictive."),
    }
    import os as _os
    _vs = "" if _os.environ.get("KEYHOLE_YOLO_VARIANT", "yolo11s-seg") == "yolo11s-seg" else f"_{_os.environ['KEYHOLE_YOLO_VARIANT']}"
    (BAKEOFF_DIR / f"fp8{_vs}_edge_projection.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}
    for res, clip_stem in CLIPS.items():
        log.info("=== %s (%s) ===", res, clip_stem)
        all_results[res] = {}
        for name, build in [
            ("efficientsam_small", lambda: EfficientSAMContestant("small")),
            ("yolo_seg",           lambda: YoloSegContestant(
                f"{__import__('os').environ.get('KEYHOLE_YOLO_VARIANT', 'yolo11s-seg')}.pt")),
        ]:
            try:
                r = run_fp8_contestant(build(), name, clip_stem)
                all_results[res][name] = r
            except Exception as e:
                log.error("%s %s failed: %s", name, res, e)
                all_results[res][name] = {"error": str(e), "name": name}

    import os as _os2
    _vs2 = "" if _os2.environ.get("KEYHOLE_YOLO_VARIANT", "yolo11s-seg") == "yolo11s-seg" else f"_{_os2.environ['KEYHOLE_YOLO_VARIANT']}"
    (BAKEOFF_DIR / f"fp8{_vs2}_summary.json").write_text(json.dumps(all_results, indent=2))
    log.info("Wrote FP8 summary")

    projections = project_edge_fp8(all_results)

    # Pretty print
    print()
    print(f"{'Model':22s} | {'Res':4s} | {'IoU bf16':>9s} {'IoU fp8':>9s} {'Δ':>7s} | "
          f"{'5090 ms bf16':>13s} {'5090 ms fp8':>13s} | {'Edge bf16 ms':>13s} {'Edge fp8 ms':>13s} {'Edge fp8 FPS':>13s}")
    print("-"*135)
    for res in CLIPS:
        for name in ["efficientsam_small", "yolo_seg"]:
            p = projections["fp8_projections"].get(res, {}).get(name)
            if not p:
                continue
            print(f"{name:22s} | {res:4s} | {p['mean_iou_bf16']:9.3f} {p['mean_iou_fp8']:9.3f} "
                  f"{p['iou_delta']:+7.3f} | {p['measured_frame_ms_5090_bf16']:13.2f} "
                  f"{p['measured_frame_ms_5090_fp8']:13.2f} | {p['projected_ms_edge_bf16']:13.1f} "
                  f"{p['projected_ms_edge_fp8']:13.1f} {p['projected_fps_edge_fp8']:13.1f}")
        print()


if __name__ == "__main__":
    main()
