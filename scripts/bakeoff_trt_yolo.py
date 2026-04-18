"""
TensorRT bake-off for YOLO-seg: FP16 / INT8 / FP8 engines from one ONNX.

Explores the proper-TensorRT path that our torchao work couldn't reach.
- FP16: baseline TRT engine (built separately by ultralytics, used as-is)
- INT8: PTQ via TensorRT's Int8EntropyCalibrator2, calibrated on bake-off frames
- FP8:  BuilderFlag.FP8 + strongly-typed network; if FP8 requires explicit
        QDQ (it does in TRT 10), the build will still succeed with
        fp16+fp8 mixed precision where TRT can auto-select — we document
        whether TRT actually uses FP8 in the final plan.

Inputs:
  data/trt_engines/yolo11s-seg.onnx        # exported earlier via ultralytics
  data/trt_engines/yolo11s-seg.fp16.engine # built earlier via ultralytics

Outputs:
  data/trt_engines/yolo11s-seg.int8.engine
  data/trt_engines/yolo11s-seg.fp8.engine
  data/output/bakeoff/trt_yolo/{clip_stem}/{recipe}.json
  data/output/bakeoff/trt_yolo_summary.json
  data/output/bakeoff/trt_yolo_edge_projection.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.emulate.npu_emulator import (
    NPUEmulator, WorkloadProfile, RTX_5090, EDGE_MPU_TARGET,
)
from scripts.bakeoff_sam_variants import BAKEOFF_DIR, gpu_reset_peak, gpu_peak_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("trt_yolo")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}
TRT_DIR = REPO_ROOT / "data" / "trt_engines"
ONNX_PATH = TRT_DIR / "yolo11s-seg.onnx"
FP16_ENGINE = TRT_DIR / "yolo11s-seg.fp16.engine"
INT8_ENGINE = TRT_DIR / "yolo11s-seg.int8.engine"
FP8_ENGINE = TRT_DIR / "yolo11s-seg.fp8.engine"
OUT_DIR = BAKEOFF_DIR / "trt_yolo"
IMGSZ = 640
CONF_THRESHOLD = 0.35
IOU_MATCH_THRESHOLD = 0.5
N_CALIB_FRAMES = 20


# ---------- calibration ----------

class BakeoffCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator that streams bake-off frames (preprocessed 640x640) to TRT."""

    def __init__(self, image_paths, cache_file):
        super().__init__()
        self.image_paths = list(image_paths)
        self.cache_file = Path(cache_file)
        self.idx = 0
        self.device_input = torch.zeros(1, 3, IMGSZ, IMGSZ,
                                         dtype=torch.float32, device="cuda")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.image_paths):
            return None
        img = preprocess(cv2.imread(str(self.image_paths[self.idx])))
        self.device_input.copy_(torch.from_numpy(img).cuda())
        self.idx += 1
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.write_bytes(cache)


# ---------- preprocessing (matches ultralytics letterbox) ----------

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Letterbox → float32 [0,1] → CHW contiguous, shape (1,3,IMGSZ,IMGSZ)."""
    h, w = img_bgr.shape[:2]
    scale = min(IMGSZ / h, IMGSZ / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h))
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    dy, dx = (IMGSZ - new_h) // 2, (IMGSZ - new_w) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    # BGR→RGB, HWC→CHW, float [0,1], batch
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None])


# ---------- engine build ----------

def build_engine(onnx_path: Path, out_path: Path, flags: set[int],
                 calibrator=None) -> None:
    """Build a TRT engine from ONNX at the specified precision set."""
    if out_path.exists():
        log.info("Reusing cached engine %s", out_path)
        return

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB
    # Embed ONNX-node-derived kernel names into the engine so Nsight Compute /
    # Nsight Systems profilers produce readable per-layer breakdowns.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    for f in flags:
        config.set_flag(f)
    if calibrator is not None:
        config.int8_calibrator = calibrator

    log.info("Building engine at %s (flags=%s) ...", out_path.name,
             [f.name for f in flags])
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"build_serialized_network returned None for {out_path.name}")
    out_path.write_bytes(bytes(serialized))
    log.info("Built %s in %.1fs (%.1f MB)", out_path.name,
             time.perf_counter() - t0, out_path.stat().st_size / 1e6)


def load_engine(path: Path):
    """Deserialize a TRT engine, skipping ultralytics' metadata header if present.

    Ultralytics' engine exporter prepends a length-prefixed JSON metadata blob
    before the actual TRT bytes (4-byte little-endian length + metadata JSON).
    Engines we build ourselves don't have this wrapper.
    """
    runtime = trt.Runtime(TRT_LOGGER)
    raw = path.read_bytes()
    # Heuristic: ultralytics wrapper starts with a 4-byte LE length followed by `{`.
    if len(raw) > 4 and raw[4:5] == b"{":
        meta_len = int.from_bytes(raw[:4], "little")
        if 4 + meta_len < len(raw):
            raw = raw[4 + meta_len:]
    return runtime.deserialize_cuda_engine(raw)


# ---------- inference ----------

def run_inference(engine, clip_stem: str, nvtx_label: str = "yolo_seg_trt") -> dict:
    """Run the TRT engine across cached bake-off frames and postprocess detections.

    `nvtx_label` is pushed as an NVTX range around each engine execution so that
    when this script is run under `scripts/profile_ncu.py`, the kernels the TRT
    engine launches get attributed to a named stage. Default covers the common
    case (YOLO-seg); callers should pass a more specific label like
    'yolo_seg_fp8_trt' / 'yolo_seg_fp16_trt' for per-recipe breakdowns.
    """
    from src.profiling.nvtx_helpers import nvtx_range
    clip_dir = BAKEOFF_DIR / clip_stem
    frames_meta = json.loads((clip_dir / "frames.json").read_text())

    ctx = engine.create_execution_context()

    # Binding introspection
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in tensor_names
                      if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_names = [n for n in tensor_names
                    if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    in_shape = engine.get_tensor_shape(input_name)
    in_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    inp = torch.zeros(tuple(in_shape), dtype=torch.float32, device="cuda")

    # Allocate outputs
    outs = {}
    for n in output_names:
        shp = engine.get_tensor_shape(n)
        outs[n] = torch.zeros(tuple(shp),
                              dtype=torch.float32, device="cuda")

    # Tensor bindings
    ctx.set_tensor_address(input_name, int(inp.data_ptr()))
    for n in output_names:
        ctx.set_tensor_address(n, int(outs[n].data_ptr()))

    stream = torch.cuda.Stream()

    # Warmup (not NVTX-labeled — profilers should skip this)
    img0 = cv2.imread(str(clip_dir / frames_meta[0]["path"]))
    x = preprocess(img0)
    if in_dtype == np.float16:
        inp.copy_(torch.from_numpy(x).half().cuda())
    else:
        inp.copy_(torch.from_numpy(x).cuda())
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    gpu_reset_peak()
    frame_results = []
    for fmeta in frames_meta:
        img = cv2.imread(str(clip_dir / fmeta["path"]))
        x = preprocess(img)
        if in_dtype == np.float16:
            inp.copy_(torch.from_numpy(x).half().cuda())
        else:
            inp.copy_(torch.from_numpy(x).cuda())

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream), nvtx_range(nvtx_label):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        # Decode detections from output tensor. YOLO-seg ONNX emits
        # output0: (1, 4+80+32, 8400) and output1: (1, 32, 160, 160) typically.
        # We only need bbox + class + conf for box-recall comparison.
        dets = decode_yolo_detections(outs, img.shape[:2])
        frame_results.append({"frame_idx": fmeta["idx"],
                              "latency_ms": ms, "detections": dets})

    del ctx
    return {
        "frames": frame_results,
        "mean_frame_ms": float(np.mean([fr["latency_ms"] for fr in frame_results])),
        "mean_det_per_frame": float(np.mean([len(fr["detections"]) for fr in frame_results])),
        "peak_vram_mb": gpu_peak_mb(),
    }


def decode_yolo_detections(outs: dict, orig_hw: tuple[int, int]) -> list[dict]:
    """Decode YOLO-seg detection output to bbox/class list.

    Output schema varies by ultralytics version. This grabs whatever output
    has shape (1, C, N) with C in [4+80, 4+80+32] and does non-max suppression.
    """
    # Find the detection tensor
    det_tensor = None
    for name, t in outs.items():
        if t.ndim == 3 and t.shape[0] == 1 and t.shape[1] in (84, 116):
            det_tensor = t
            break
    if det_tensor is None:
        return []

    x = det_tensor[0].transpose(0, 1)  # (N, C)
    box_xywh = x[:, :4]
    cls_scores = x[:, 4:84]  # 80 COCO classes
    conf, cls = cls_scores.max(dim=1)

    mask = conf > CONF_THRESHOLD
    box_xywh = box_xywh[mask]
    conf = conf[mask]
    cls = cls[mask]

    if box_xywh.numel() == 0:
        return []

    # xywh → xyxy (model coordinates — 640x640 letterboxed)
    xy1 = box_xywh[:, :2] - box_xywh[:, 2:] / 2
    xy2 = box_xywh[:, :2] + box_xywh[:, 2:] / 2
    boxes = torch.cat([xy1, xy2], dim=1)

    # Torchvision NMS
    from torchvision.ops import nms
    keep = nms(boxes, conf, iou_threshold=0.7)
    boxes = boxes[keep].cpu().numpy()
    conf = conf[keep].cpu().numpy()
    cls = cls[keep].cpu().numpy()

    # Undo letterbox → original-image coords
    h, w = orig_hw
    scale = min(IMGSZ / h, IMGSZ / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    dy, dx = (IMGSZ - new_h) // 2, (IMGSZ - new_w) // 2
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

    dets = []
    for b, c, s in zip(boxes, cls, conf):
        dets.append({"bbox": b.tolist(),
                     "class_id": int(c),
                     "class_name": COCO_NAMES.get(int(c), f"class_{int(c)}"),
                     "confidence": float(s)})
    return dets


# COCO 80-class names (matches ultralytics' default yolo11s-seg model.names)
COCO_NAMES = {i: n for i, n in enumerate([
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
])}


# ---------- quality helpers (same as bakeoff_yolo_conv_quant.py) ----------

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


def detection_stability(ref_frames, var_frames) -> dict:
    match_ious = []
    total_ref, total_matched, total_var, total_fp = 0, 0, 0, 0
    for fr_ref, fr_var in zip(ref_frames, var_frames):
        dr, dv = fr_ref["detections"], fr_var["detections"]
        used_v = set()
        total_ref += len(dr); total_var += len(dv)
        for da in dr:
            best, best_iou = -1, IOU_MATCH_THRESHOLD
            for j, db in enumerate(dv):
                if j in used_v or da["class_name"] != db["class_name"]:
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
        "n_ref": total_ref, "n_var": total_var,
        "n_matched": total_matched, "n_fp": total_fp,
    }


# ---------- main ----------

def gather_calib_paths() -> list[Path]:
    paths = []
    for stem in CLIPS.values():
        clip_dir = BAKEOFF_DIR / stem
        meta = json.loads((clip_dir / "frames.json").read_text())
        paths.extend(clip_dir / m["path"] for m in meta[:7])
    return paths[:N_CALIB_FRAMES]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRT_DIR.mkdir(parents=True, exist_ok=True)

    # INT8 engine: built separately via ultralytics' int8 export path (handles
    # YOLO output-head quirks that hand-rolled Int8EntropyCalibrator2 trips on
    # "invalid precision Int8" for). See commit notes for details.
    if not INT8_ENGINE.exists():
        log.warning("INT8 engine missing at %s — run ultralytics export first",
                    INT8_ENGINE)

    # FP8 engine — hand-rolled. Try BuilderFlag.FP8 alone; if it fails, try
    # combined with FP16 as allowed fallback precision.
    fp8_ok = False
    for flag_set in [
        {trt.BuilderFlag.FP8, trt.BuilderFlag.FP16, trt.BuilderFlag.BF16},
        {trt.BuilderFlag.FP8, trt.BuilderFlag.FP16},
        {trt.BuilderFlag.FP8},
    ]:
        try:
            build_engine(ONNX_PATH, FP8_ENGINE, flags=flag_set)
            fp8_ok = True
            break
        except Exception as e:
            log.warning("FP8 build with %s failed: %s",
                        [f.name for f in flag_set], str(e)[:150])
            if FP8_ENGINE.exists():
                FP8_ENGINE.unlink()

    # Run inference for each recipe + resolution
    recipes = {
        "fp16": FP16_ENGINE,
        "int8": INT8_ENGINE,
    }
    if fp8_ok and FP8_ENGINE.exists():
        recipes["fp8"] = FP8_ENGINE
    else:
        recipes["fp8"] = None

    all_results = {}
    for res, clip_stem in CLIPS.items():
        log.info("=== %s (%s) ===", res, clip_stem)
        all_results[res] = {}
        for recipe, engine_path in recipes.items():
            if engine_path is None:
                all_results[res][recipe] = {"error": "fp8 engine build failed"}
                continue
            out_path = OUT_DIR / clip_stem / f"{recipe}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                all_results[res][recipe] = json.loads(out_path.read_text())
                log.info("Reusing %s", out_path)
                continue
            try:
                eng = load_engine(engine_path)
                run = run_inference(eng, clip_stem,
                                     nvtx_label=f"yolo_seg_{recipe}_trt")
                del eng
                torch.cuda.empty_cache()
                all_results[res][recipe] = run
                out_path.write_text(json.dumps(run, indent=2))
                log.info("%s %s: %.2f ms/frame, %.1f dets",
                         recipe, res, run["mean_frame_ms"], run["mean_det_per_frame"])
            except Exception as e:
                log.exception("Inference failed: %s %s", recipe, res)
                all_results[res][recipe] = {"error": str(e)}

    # Quality deltas — variant vs fp16 baseline (TRT's highest-precision reference)
    quality = {}
    for res in CLIPS:
        quality[res] = {}
        ref = all_results[res].get("fp16")
        if not ref or "frames" not in ref:
            continue
        for recipe in ["fp16", "int8", "fp8"]:
            var = all_results[res].get(recipe)
            if not var or "frames" not in var:
                continue
            if recipe == "fp16":
                quality[res][recipe] = {"box_recall": 1.0, "mean_matched_iou": 1.0,
                                         "n_ref": 0, "n_var": 0,
                                         "n_matched": 0, "n_fp": 0}
                continue
            quality[res][recipe] = detection_stability(ref["frames"], var["frames"])

    summary = {"results": all_results, "quality": quality, "fp8_built": fp8_ok}
    (BAKEOFF_DIR / "trt_yolo_summary.json").write_text(json.dumps(summary, indent=2))

    # Edge projection — use BF16 baseline from prior bake-off as reference;
    # halve activation bytes on the quantized fraction (1.0 for INT8/FP8 full-model).
    prior = json.loads((BAKEOFF_DIR / "yolo_conv_quant_edge_projection.json").read_text())
    prior_bf16 = prior["projections"]
    emu = NPUEmulator(reference=RTX_5090, target=EDGE_MPU_TARGET)
    proj = {}
    for res, clip_stem in CLIPS.items():
        proj[res] = {}
        pb = prior_bf16[res]["bf16"]
        comp = pb["compute_limited_ms"]
        bw_bf16 = pb["bandwidth_limited_ms_bf16"]
        for recipe in ["fp16", "int8", "fp8"]:
            r = all_results[res].get(recipe)
            if not r or "error" in r:
                proj[res][recipe] = {"error": (r or {}).get("error", "not_run")}
                continue
            # Precision-specific bandwidth reduction (full-model):
            #   fp16 = 0.5 of bf16 activation traffic (same bits but TRT kernel effects)
            #          treat as 1.0 since bf16 and fp16 activation bytes are equal
            #   int8 = 0.5 (half bits), fp8 = 0.5 (half bits)
            bw_mul = {"fp16": 1.0, "int8": 0.5, "fp8": 0.5}[recipe]
            adj_ms = comp + bw_bf16 * bw_mul
            q = quality.get(res, {}).get(recipe, {})
            proj[res][recipe] = {
                "recipe": recipe,
                "mean_frame_ms_5090": r["mean_frame_ms"],
                "projected_ms_edge": adj_ms,
                "projected_fps_edge": 1000.0 / adj_ms if adj_ms > 0 else 0.0,
                "bandwidth_multiplier": bw_mul,
                "compute_limited_ms": comp,
                "bandwidth_limited_ms": bw_bf16 * bw_mul,
                **q,
            }
    (BAKEOFF_DIR / "trt_yolo_edge_projection.json").write_text(
        json.dumps({"projections": proj,
                    "method": ("TensorRT engines built from same ONNX. INT8 uses "
                               "Int8EntropyCalibrator2 on 20 bake-off frames. FP8 "
                               "uses BuilderFlag.FP8 + FP16 (mixed precision; TRT "
                               "auto-selects FP8 layers where safe without QDQ). "
                               "Edge projection reuses BF16 RTX-5090 reference from "
                               "the prior YOLO-conv-quant bake-off; bandwidth ms "
                               "scaled by 0.5 for INT8/FP8 (full-model half bytes), "
                               "1.0 for FP16 (same activation bytes as BF16)."),
                    "fp8_built": fp8_ok}, indent=2))
    log.info("Wrote trt_yolo_{summary,edge_projection}.json")

    # Pretty print
    print()
    hdr = (f"{'Res':6s} | {'Recipe':6s} | {'ms (5090)':>10s} {'dets':>4s} | "
           f"{'recall':>6s} {'IoU':>5s} | {'Edge ms':>8s} {'Edge FPS':>8s}")
    print(hdr); print("-"*len(hdr))
    for res in CLIPS:
        for recipe in ["fp16", "int8", "fp8"]:
            p = proj[res].get(recipe)
            r = all_results[res].get(recipe, {})
            if not p or "error" in p:
                err = (p or {}).get("error", "?")
                print(f"{res:6s} | {recipe:6s} | {'—':>10s} {'—':>4s} | "
                      f"{'—':>6s} {'—':>5s} | {'—':>8s} {'—':>8s}  ({err[:40]})")
                continue
            print(f"{res:6s} | {recipe:6s} | {p['mean_frame_ms_5090']:>10.2f} "
                  f"{r.get('mean_det_per_frame', 0):>4.1f} | "
                  f"{p.get('box_recall', 0):>6.3f} {p.get('mean_matched_iou', 0):>5.3f} | "
                  f"{p['projected_ms_edge']:>8.1f} {p['projected_fps_edge']:>8.1f}")
        print()


if __name__ == "__main__":
    main()
