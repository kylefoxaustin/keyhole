"""
TensorRT bake-off for OpenCLIP ViT-B-32 vision tower.

Companion to bakeoff_trt_yolo.py. YOLO was Conv-dominated; CLIP visual is
a ViT — more amenable to torchao already (we did this in Hybrid V2 CLIP
quant bake-off), but we didn't push through proper TRT compile. This
script does: exports visual → ONNX (dynamic batch), builds FP16 + FP8
engines, runs on the bake-off crops, compares top-1 concept tags vs the
PyTorch BF16 reference.

Inputs:
  data/output/bakeoff/hybrid_v2_summary.json  (per-frame detections → crops)
  data/output/bakeoff/{clip_stem}/frames/*.png

Outputs:
  data/trt_engines/clip_vit_b32_visual.onnx
  data/trt_engines/clip_vit_b32_visual.fp16.engine
  data/trt_engines/clip_vit_b32_visual.fp8.engine
  data/output/bakeoff/trt_clip/{clip_stem}/{recipe}.json
  data/output/bakeoff/trt_clip_summary.json
  data/output/bakeoff/trt_clip_edge_projection.json
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.emulate.npu_emulator import (
    NPUEmulator, WorkloadProfile, RTX_5090, EDGE_MPU_TARGET,
)
from scripts.bakeoff_sam_variants import BAKEOFF_DIR, gpu_reset_peak, gpu_peak_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("trt_clip")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}
TRT_DIR = REPO_ROOT / "data" / "trt_engines"
ONNX_PATH = TRT_DIR / "clip_vit_b32_visual.onnx"
FP16_ENGINE = TRT_DIR / "clip_vit_b32_visual.fp16.engine"
FP8_ENGINE = TRT_DIR / "clip_vit_b32_visual.fp8.engine"
OUT_DIR = BAKEOFF_DIR / "trt_clip"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
INPUT_HW = 224
BATCH_MIN, BATCH_OPT, BATCH_MAX = 1, 8, 32

# Concept lists match src/detect/hybrid_v2.py PERSON/VEHICLE/GENERAL_ATTRIBUTES
PERSON_ATTRIBUTES = [
    "person wearing hat", "person wearing backpack", "person wearing jacket",
    "person wearing uniform", "person with glasses", "person carrying bag",
    "child", "adult walking", "person on phone",
]
VEHICLE_ATTRIBUTES = [
    "red vehicle", "blue vehicle", "white vehicle", "black vehicle",
    "delivery truck", "police car", "SUV", "sedan",
]
GENERAL_ATTRIBUTES = [
    "package", "box", "bag", "bicycle", "skateboard", "umbrella",
]


def prompts_for_class(cls: str) -> list[str]:
    if cls == "person":
        return PERSON_ATTRIBUTES
    if cls in ("car", "truck", "bus", "motorcycle"):
        return VEHICLE_ATTRIBUTES
    return GENERAL_ATTRIBUTES


# ---------- Export ----------

def export_onnx():
    if ONNX_PATH.exists():
        log.info("Reusing ONNX %s", ONNX_PATH)
        return
    import open_clip
    log.info("Loading CLIP %s for ONNX export...", CLIP_MODEL)
    m, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device="cuda")
    m.eval()
    visual = m.visual
    dummy = torch.randn(BATCH_OPT, 3, INPUT_HW, INPUT_HW, device="cuda")
    TRT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Exporting visual tower to ONNX ...")
    # Use the legacy TorchScript tracer (dynamo=False) so weights stay inline in
    # the .onnx file. The newer dynamo exporter splits large weights into a
    # sibling .data file which TRT's OnnxParser can't follow when we pass it
    # raw bytes.
    torch.onnx.export(
        visual, dummy, str(ONNX_PATH),
        input_names=["image"], output_names=["features"],
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    log.info("ONNX written: %s (%.1f MB)", ONNX_PATH, ONNX_PATH.stat().st_size / 1e6)


# ---------- Engine build ----------

def build_engine(out_path: Path, flags: set[int]):
    if out_path.exists():
        log.info("Reusing engine %s", out_path)
        return
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    # Embed ONNX-node-derived kernel names for readable Nsight profiling.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    for fl in flags:
        config.set_flag(fl)

    # Dynamic-batch optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape("image",
                      min=(BATCH_MIN, 3, INPUT_HW, INPUT_HW),
                      opt=(BATCH_OPT, 3, INPUT_HW, INPUT_HW),
                      max=(BATCH_MAX, 3, INPUT_HW, INPUT_HW))
    config.add_optimization_profile(profile)

    log.info("Building %s with %s ...", out_path.name, [f.name for f in flags])
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"build_serialized_network returned None for {out_path.name}")
    out_path.write_bytes(bytes(serialized))
    log.info("Built %s in %.1fs (%.1f MB)", out_path.name,
             time.perf_counter() - t0, out_path.stat().st_size / 1e6)


def load_engine(path: Path):
    runtime = trt.Runtime(TRT_LOGGER)
    raw = path.read_bytes()
    # Not ultralytics-wrapped (we built these ourselves), but be safe
    if len(raw) > 4 and raw[4:5] == b"{":
        meta_len = int.from_bytes(raw[:4], "little")
        if 4 + meta_len < len(raw):
            raw = raw[4 + meta_len:]
    return runtime.deserialize_cuda_engine(raw)


# ---------- Inference helpers ----------

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def preprocess_crops(crops_bgr: list[np.ndarray]) -> torch.Tensor:
    """Resize to 224x224, BGR→RGB, normalize. Match OpenCLIP preprocess."""
    if not crops_bgr:
        return torch.empty(0, 3, INPUT_HW, INPUT_HW, device="cuda")
    tensors = []
    for c in crops_bgr:
        if c is None or c.size == 0:
            continue
        # Resize short-side to 224, center-crop to 224x224 (matches preprocess)
        h, w = c.shape[:2]
        scale = INPUT_HW / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_CUBIC)
        dy = (nh - INPUT_HW) // 2
        dx = (nw - INPUT_HW) // 2
        cropped = resized[dy:dy + INPUT_HW, dx:dx + INPUT_HW]
        rgb = cropped[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, [0,1]
        tensors.append(torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))))
    if not tensors:
        return torch.empty(0, 3, INPUT_HW, INPUT_HW, device="cuda")
    batch = torch.stack(tensors).cuda()
    batch = (batch - _CLIP_MEAN.cuda()) / _CLIP_STD.cuda()
    return batch


def build_text_cache_torch(m, tokenizer) -> dict:
    """Compute normalized text features per class list, in torch."""
    cache = {}
    for key, prompts in [("person", PERSON_ATTRIBUTES),
                         ("vehicle", VEHICLE_ATTRIBUTES),
                         ("general", GENERAL_ATTRIBUTES)]:
        tokens = tokenizer(prompts).cuda()
        with torch.no_grad():
            feats = m.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        cache[key] = (prompts, feats.float())  # float32 for stable matmul
    return cache


def class_to_cache_key(cls: str) -> str:
    if cls == "person":
        return "person"
    if cls in ("car", "truck", "bus", "motorcycle"):
        return "vehicle"
    return "general"


def top1_per_detection(image_features: torch.Tensor, class_names: list[str],
                        text_cache: dict) -> list[str]:
    """For each detection, pick top-1 concept using its class's text cache."""
    img = image_features / image_features.norm(dim=-1, keepdim=True)
    tops = []
    for i, cls in enumerate(class_names):
        prompts, text_feats = text_cache[class_to_cache_key(cls)]
        sim = (img[i:i+1].float() @ text_feats.T).squeeze(0)
        probs = sim.softmax(dim=-1)
        tops.append(prompts[int(probs.argmax())])
    return tops


# ---------- Bake-off runner ----------

def load_crops_and_classes(clip_stem: str):
    """Use the cached hybrid_v2 FP8 detections as the source of truth for
    which crops to score on per frame."""
    h = json.loads((BAKEOFF_DIR / "hybrid_v2_summary.json").read_text())
    per_res_frames = None
    for res, stem in CLIPS.items():
        if stem == clip_stem:
            per_res_frames = h[res]["fp8"]["frames"]
            break
    assert per_res_frames is not None, clip_stem

    clip_dir = BAKEOFF_DIR / clip_stem
    out = []
    for fr in per_res_frames:
        img = cv2.imread(str(clip_dir / f"frames/frame_{fr['frame_idx']:06d}.png"))
        crops, classes, truth_tops = [], [], []
        for d in fr["detections"]:
            x1, y1, x2, y2 = map(int, d["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            h_img, w_img = img.shape[:2]
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            crops.append(img[y1:y2, x1:x2])
            classes.append(d["class_name"])
            truth_tops.append(d.get("top_concept"))
        out.append({
            "frame_idx": fr["frame_idx"],
            "crops": crops,
            "classes": classes,
            "torch_bf16_top1": truth_tops,
        })
    return out


def run_torch_bf16(frames_data: list[dict], m_bf16, tokenizer) -> dict:
    """Reference run: OpenCLIP visual in BF16 (same as hybrid_v2 FP8 CLIP variant)."""
    text_cache = build_text_cache_torch(m_bf16, tokenizer)
    torch.cuda.synchronize()
    gpu_reset_peak()
    frame_results = []
    for fd in frames_data:
        if not fd["crops"]:
            frame_results.append({"frame_idx": fd["frame_idx"], "latency_ms": 0.0,
                                   "top1": [], "n": 0})
            continue
        batch = preprocess_crops(fd["crops"]).bfloat16()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            feats = m_bf16.visual(batch)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
        tops = top1_per_detection(feats, fd["classes"], text_cache)
        frame_results.append({"frame_idx": fd["frame_idx"], "latency_ms": ms,
                               "top1": tops, "n": len(fd["crops"])})
    return {
        "frames": frame_results,
        "mean_frame_ms": float(np.mean([fr["latency_ms"] for fr in frame_results if fr["n"] > 0]))
                          if any(fr["n"] > 0 for fr in frame_results) else 0.0,
        "peak_vram_mb": gpu_peak_mb(),
    }


def run_trt(frames_data: list[dict], engine, text_cache: dict) -> dict:
    ctx = engine.create_execution_context()
    input_name = next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                      if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT)
    output_name = next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                       if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT)
    in_dtype = trt.nptype(engine.get_tensor_dtype(input_name))

    # Pre-allocate max-size buffers
    max_bsz = BATCH_MAX
    in_buf = torch.zeros(max_bsz, 3, INPUT_HW, INPUT_HW,
                          dtype=torch.float16 if in_dtype == np.float16 else torch.float32,
                          device="cuda")
    out_buf = torch.zeros(max_bsz, 512, dtype=torch.float32, device="cuda")

    # Warmup
    batch = preprocess_crops(frames_data[0]["crops"][:BATCH_OPT])
    if batch.shape[0] > 0:
        ctx.set_input_shape(input_name, batch.shape)
        in_buf[:batch.shape[0]].copy_(batch.to(in_buf.dtype))
        ctx.set_tensor_address(input_name, int(in_buf.data_ptr()))
        ctx.set_tensor_address(output_name, int(out_buf.data_ptr()))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    gpu_reset_peak()
    stream = torch.cuda.Stream()
    frame_results = []
    for fd in frames_data:
        if not fd["crops"]:
            frame_results.append({"frame_idx": fd["frame_idx"], "latency_ms": 0.0,
                                   "top1": [], "n": 0})
            continue
        # Batch across BATCH_MAX chunks if needed
        batch_all = preprocess_crops(fd["crops"])
        N = batch_all.shape[0]
        feats_all = torch.zeros(N, 512, dtype=torch.float32, device="cuda")

        torch.cuda.synchronize(); t0 = time.perf_counter()
        offset = 0
        while offset < N:
            chunk = batch_all[offset:offset + max_bsz]
            bsz = chunk.shape[0]
            ctx.set_input_shape(input_name, (bsz, 3, INPUT_HW, INPUT_HW))
            in_buf[:bsz].copy_(chunk.to(in_buf.dtype))
            ctx.set_tensor_address(input_name, int(in_buf.data_ptr()))
            ctx.set_tensor_address(output_name, int(out_buf.data_ptr()))
            from src.profiling.nvtx_helpers import nvtx_range
            with torch.cuda.stream(stream), nvtx_range("clip_trt"):
                ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            feats_all[offset:offset + bsz].copy_(out_buf[:bsz])
            offset += bsz
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000

        tops = top1_per_detection(feats_all, fd["classes"], text_cache)
        frame_results.append({"frame_idx": fd["frame_idx"], "latency_ms": ms,
                               "top1": tops, "n": N})

    del ctx
    return {
        "frames": frame_results,
        "mean_frame_ms": float(np.mean([fr["latency_ms"] for fr in frame_results if fr["n"] > 0]))
                          if any(fr["n"] > 0 for fr in frame_results) else 0.0,
        "peak_vram_mb": gpu_peak_mb(),
    }


def top1_agreement(ref_frames, var_frames) -> float:
    match, total = 0, 0
    for fr_ref, fr_var in zip(ref_frames, var_frames):
        for a, b in zip(fr_ref["top1"], fr_var["top1"]):
            total += 1
            if a == b:
                match += 1
    return match / total if total else 1.0


# ---------- Main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRT_DIR.mkdir(parents=True, exist_ok=True)

    export_onnx()
    build_engine(FP16_ENGINE, flags={trt.BuilderFlag.FP16})
    fp8_ok = False
    for flag_set in [
        {trt.BuilderFlag.FP8, trt.BuilderFlag.FP16, trt.BuilderFlag.BF16},
        {trt.BuilderFlag.FP8, trt.BuilderFlag.FP16},
        {trt.BuilderFlag.FP8},
    ]:
        try:
            build_engine(FP8_ENGINE, flags=flag_set)
            fp8_ok = True
            break
        except Exception as e:
            log.warning("FP8 build %s failed: %s",
                        [f.name for f in flag_set], str(e)[:150])
            if FP8_ENGINE.exists():
                FP8_ENGINE.unlink()

    # Load BF16 CLIP for reference + text features
    import open_clip
    m_ref, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device="cuda")
    m_ref.eval().bfloat16()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    text_cache = build_text_cache_torch(m_ref, tokenizer)

    all_results = {}
    for res, clip_stem in CLIPS.items():
        log.info("=== %s ===", res)
        all_results[res] = {}
        frames_data = load_crops_and_classes(clip_stem)

        # KEYHOLE_FORCE_RERUN=1 bypasses cached JSON so ncu can profile actual
        # GPU kernel launches (cache short-circuit would skip the work).
        import os as _os
        _force = bool(_os.environ.get("KEYHOLE_FORCE_RERUN"))

        # Torch BF16 reference
        out_path = OUT_DIR / clip_stem / "bf16_torch.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not _force:
            all_results[res]["bf16_torch"] = json.loads(out_path.read_text())
        else:
            r = run_torch_bf16(frames_data, m_ref, tokenizer)
            out_path.write_text(json.dumps(r, indent=2))
            all_results[res]["bf16_torch"] = r
        log.info("bf16_torch %s: %.2f ms/frame", res,
                 all_results[res]["bf16_torch"]["mean_frame_ms"])

        # TRT FP16
        for recipe, engine_path in [("fp16", FP16_ENGINE),
                                    ("fp8", FP8_ENGINE if fp8_ok else None)]:
            if engine_path is None:
                all_results[res][recipe] = {"error": "engine not built"}
                continue
            out_path = OUT_DIR / clip_stem / f"{recipe}.json"
            if out_path.exists() and not _force:
                all_results[res][recipe] = json.loads(out_path.read_text())
            else:
                eng = load_engine(engine_path)
                r = run_trt(frames_data, eng, text_cache)
                del eng
                torch.cuda.empty_cache()
                out_path.write_text(json.dumps(r, indent=2))
                all_results[res][recipe] = r
            log.info("%s %s: %.2f ms/frame", recipe, res,
                     all_results[res][recipe]["mean_frame_ms"])

    # Quality + edge projection
    # Reuse prior hybrid_v2 BF16/FP8 CLIP ms for reference points.
    prior = json.loads((BAKEOFF_DIR / "hybrid_v2_edge_projection.json").read_text())

    proj = {}
    emu = NPUEmulator(reference=RTX_5090, target=EDGE_MPU_TARGET)
    params_clip = 151_277_313
    model_bytes_bf16 = params_clip * 2

    for res, clip_stem in CLIPS.items():
        proj[res] = {}
        ref_frames = all_results[res]["bf16_torch"]["frames"]

        # Build the BF16 edge projection for CLIP
        bf16_ms_5090 = all_results[res]["bf16_torch"]["mean_frame_ms"]
        wl = WorkloadProfile(
            stage_name="clip_vit_b32_bf16",
            model_name="clip_ViT-B-32",
            param_count=params_clip,
            model_size_bytes=model_bytes_bf16,
            precision="bf16",
            measured_latency_ms=bf16_ms_5090,
            measured_gpu_kernel_ms=bf16_ms_5090,
            measured_gpu=RTX_5090.name,
            measured_peak_vram_bytes=model_bytes_bf16 * 2,
            peak_activation_bytes=model_bytes_bf16,
        )
        proj_bf16 = emu.project_workload(wl)

        for recipe in ["bf16_torch", "fp16", "fp8"]:
            r = all_results[res].get(recipe)
            if not r or "error" in r:
                proj[res][recipe] = {"error": (r or {}).get("error", "not_run")}
                continue
            var_frames = r["frames"]
            agree = top1_agreement(ref_frames, var_frames)
            bw_mul = {"bf16_torch": 1.0, "fp16": 1.0, "fp8": 0.5}[recipe]
            adj_ms = proj_bf16.compute_limited_ms + proj_bf16.bandwidth_limited_ms * bw_mul
            proj[res][recipe] = {
                "recipe": recipe,
                "mean_frame_ms_5090": r["mean_frame_ms"],
                "top1_agreement": agree,
                "projected_clip_ms_edge": adj_ms,
                "projected_fps_edge_clip_only": 1000.0 / adj_ms if adj_ms > 0 else 0.0,
                "bandwidth_multiplier": bw_mul,
            }

    (BAKEOFF_DIR / "trt_clip_summary.json").write_text(json.dumps(all_results, indent=2))
    (BAKEOFF_DIR / "trt_clip_edge_projection.json").write_text(
        json.dumps({"projections": proj,
                    "method": ("CLIP visual compiled via TRT 10.16 from OpenCLIP "
                               "ViT-B-32 ONNX (dynamic batch 1-32). FP8 uses "
                               "BuilderFlag.FP8 + FP16 + BF16 mixed precision. "
                               "Edge projection halves bandwidth for FP8 (activation "
                               "bytes halved full-model); FP16 and BF16 treated as "
                               "same bandwidth. Top-1 agreement measured vs BF16 "
                               "torch reference per detection."),
                    "fp8_built": fp8_ok}, indent=2))
    log.info("Wrote trt_clip_{summary,edge_projection}.json")

    # Pretty print
    print()
    hdr = (f"{'Res':6s} | {'Recipe':11s} | {'5090 ms':>8s} | "
           f"{'Top-1 agree':>11s} | {'Edge CLIP ms':>13s} {'CLIP-only FPS':>13s}")
    print(hdr); print("-"*len(hdr))
    for res in CLIPS:
        for recipe in ["bf16_torch", "fp16", "fp8"]:
            p = proj[res].get(recipe)
            if not p or "error" in p:
                print(f"{res:6s} | {recipe:11s} | {'—':>8s} | {'—':>11s} | "
                      f"{'—':>13s} {'—':>13s}  ({(p or {}).get('error','?')[:40]})")
                continue
            print(f"{res:6s} | {recipe:11s} | {p['mean_frame_ms_5090']:>8.2f} | "
                  f"{p['top1_agreement']:>11.3f} | "
                  f"{p['projected_clip_ms_edge']:>13.1f} "
                  f"{p['projected_fps_edge_clip_only']:>13.2f}")
        print()


if __name__ == "__main__":
    main()
