"""TensorRT INT8 bake-off for ResNet-50v1 — 5090 anchor for sizer Phase 2.

Per [sizer] 12:31 ask (2026-05-01): build a ResNet-50v1 INT8 TRT engine,
run sustained inference on 5090, ncu profile for per-forward DRAM bytes.
Output flows through the standard pipeline so export_ncu_for_sizer.py
picks it up into sizer_bundle.json on next refresh, validating the
projection slope from sizer's measured Low-LP5X anchor (1125 FPS, ce03030)
upward to the 5090 cell.

Standard MLPerf-style classification workload:
  Model:  torchvision ResNet-50v1 (IMAGENET1K_V1 weights, ~25.5M params)
  Input:  224×224 RGB, fixed (constant input size — ImageNet canonical)
  Output: 1000-class softmax logits
  Precision: INT8 weight-only via TRT Int8EntropyCalibrator2
  Calibration: 20 frames from cached bake-off EW clip (resized + center-cropped to 224)

Outputs:
  data/trt_engines/resnet50v1.onnx
  data/trt_engines/resnet50v1.int8.engine
  data/output/bakeoff/resnet50_summary.json

ncu sweep target: see scripts/profile_all_ncu.sh `resnet50` block.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("trt_resnet50")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

REPO = REPO_ROOT
TRT_DIR = REPO / "data" / "trt_engines"
TRT_DIR.mkdir(parents=True, exist_ok=True)
BAKEOFF_DIR = REPO / "data" / "output" / "bakeoff"
BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = TRT_DIR / "resnet50v1.onnx"
INT8_ENGINE = TRT_DIR / "resnet50v1.int8.engine"
SUMMARY_PATH = BAKEOFF_DIR / "resnet50_summary.json"

INPUT_SIZE = 224
N_CALIB_FRAMES = 14   # all available frames in 720p_EW_clip — enough for INT8 PTQ stats
N_WARMUP = 2
N_TIMED = 10   # matches ViT-alternatives convention; enough for ncu DRAM-per-forward + p50 timing
               # (full statistical run can re-bump to 100 when needed)

# Calibration frames — 720p EW clip frames are general-scene RGB (not ImageNet
# class, but fine for INT8 quantization range calibration; we're not validating
# accuracy, just measuring throughput). Mirrors bakeoff_trt_yolo's pattern.
CALIB_CLIP_DIR = BAKEOFF_DIR / "720p_EW_clip" / "frames"


# ────────────────────────── ONNX export ──────────────────────────

def export_onnx() -> None:
    """Export torchvision ResNet-50v1 (IMAGENET1K_V1 weights) to ONNX at 224×224."""
    if ONNX_PATH.exists():
        log.info("Reusing cached ONNX %s (%.1f MB)", ONNX_PATH, ONNX_PATH.stat().st_size / 1e6)
        return

    log.info("Exporting torchvision ResNet-50v1 → %s", ONNX_PATH)
    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
    )
    log.info("ONNX export complete (%.1f MB)", ONNX_PATH.stat().st_size / 1e6)


# ────────────────────────── Calibration ──────────────────────────

# Standard ImageNet preprocessing (matches torchvision's default):
#   resize shorter side to 256, center-crop 224, /255, normalize mean/std
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """ImageNet-style preprocessing → (1, 3, 224, 224) float32 normalized."""
    h, w = img_bgr.shape[:2]
    scale = 256.0 / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h))
    # Center crop 224×224
    dy = (new_h - INPUT_SIZE) // 2
    dx = (new_w - INPUT_SIZE) // 2
    cropped = resized[dy:dy + INPUT_SIZE, dx:dx + INPUT_SIZE]
    # BGR→RGB, HWC→CHW, /255, normalize
    rgb = cropped[:, :, ::-1].astype(np.float32) / 255.0
    chw = rgb.transpose(2, 0, 1)[None]  # (1, 3, 224, 224)
    chw = (chw - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(chw)


class ResNetCalibrator(trt.IInt8EntropyCalibrator2):
    """Streams 720p EW frames preprocessed to ImageNet 224×224 for TRT INT8 PTQ."""

    def __init__(self, image_paths, cache_file):
        super().__init__()
        self.image_paths = list(image_paths)
        self.cache_file = Path(cache_file)
        self.idx = 0
        self.device_input = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE,
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


# ────────────────────────── Engine build ──────────────────────────

def build_int8_engine() -> None:
    """Build the INT8 TRT engine from the ONNX with PTQ calibration."""
    if INT8_ENGINE.exists():
        log.info("Reusing cached engine %s (%.1f MB)", INT8_ENGINE, INT8_ENGINE.stat().st_size / 1e6)
        return

    if not CALIB_CLIP_DIR.exists():
        raise FileNotFoundError(
            f"Calibration frames missing: {CALIB_CLIP_DIR}. Run a frame-cache "
            "step first (any prior bake-off populates 720p_EW_clip/frames/)."
        )
    calib_paths = sorted(CALIB_CLIP_DIR.glob("frame_*.png"))[:N_CALIB_FRAMES]
    if len(calib_paths) < N_CALIB_FRAMES:
        raise RuntimeError(
            f"Need {N_CALIB_FRAMES} calib frames, found {len(calib_paths)} in {CALIB_CLIP_DIR}"
        )
    log.info("Calibration: %d frames from %s", len(calib_paths), CALIB_CLIP_DIR.name)

    calibrator = ResNetCalibrator(
        calib_paths,
        cache_file=TRT_DIR / "resnet50v1.int8.cache",
    )

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # parse_from_file resolves external-data sidecars (.onnx.data) relative to
    # the model path. torch.onnx (dynamo) emits weights externally for large
    # models, which `parser.parse(raw_bytes)` cannot resolve.
    if not parser.parse_from_file(str(ONNX_PATH)):
        for i in range(parser.num_errors):
            log.error("ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    log.info("Building INT8 engine at %s ...", INT8_ENGINE.name)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    INT8_ENGINE.write_bytes(bytes(serialized))
    log.info("Built %s in %.1fs (%.1f MB)",
             INT8_ENGINE.name, time.perf_counter() - t0, INT8_ENGINE.stat().st_size / 1e6)


def load_engine(path: Path):
    """Deserialize TRT engine. We build cleanly, no header to skip."""
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(path.read_bytes())


# ────────────────────────── Inference ──────────────────────────

def run_inference(engine) -> dict:
    """Sustained INT8 ResNet-50 inference on 5090, NVTX-wrapped per forward.

    Mirrors bakeoff_trt_yolo's nvtx pattern so profile_all_ncu.sh can
    attribute kernels to the `resnet50_int8_trt__224` range.
    """
    from src.profiling.nvtx_helpers import nvtx_range

    ctx = engine.create_execution_context()

    # I/O introspection
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in tensor_names
                      if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_names = [n for n in tensor_names
                    if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    # Allocate buffers
    inp = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32, device="cuda")
    outs = {}
    for n in output_names:
        shp = tuple(engine.get_tensor_shape(n))
        outs[n] = torch.zeros(shp, dtype=torch.float32, device="cuda")

    ctx.set_tensor_address(input_name, int(inp.data_ptr()))
    for n in output_names:
        ctx.set_tensor_address(n, int(outs[n].data_ptr()))

    stream = torch.cuda.Stream()
    torch.cuda.reset_peak_memory_stats()

    # Use a single calibration frame as the timing input — content-invariant for
    # a fixed-input-size pure-conv classifier; the 5090's perf signature is
    # determined by graph + INT8 kernels, not pixel content.
    sample_path = sorted(CALIB_CLIP_DIR.glob("frame_*.png"))[0]
    sample = preprocess(cv2.imread(str(sample_path)))
    inp.copy_(torch.from_numpy(sample).cuda())

    # Warmup (NOT NVTX-labeled — profilers should skip)
    for _ in range(N_WARMUP):
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Timed: each forward inside its own NVTX range so ncu app-replay can
    # attribute kernels per-iteration without overflowing the divisor count.
    log.info("Sustained inference: %d timed forwards @ 224×224 INT8", N_TIMED)
    per_frame_ms = []
    for i in range(N_TIMED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx_range("resnet50_int8_trt__224"):
            with torch.cuda.stream(stream):
                ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        per_frame_ms.append(ms)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    arr = np.array(per_frame_ms)
    return {
        "n_timed": N_TIMED,
        "n_warmup": N_WARMUP,
        "input_hw": [INPUT_SIZE, INPUT_SIZE],
        "precision": "INT8",
        "model": "resnet50v1",
        "weights": "torchvision IMAGENET1K_V1",
        "per_frame_ms": per_frame_ms,
        "mean_ms": float(arr.mean()),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
        "fps_p50": float(1000.0 / np.percentile(arr, 50)),
        "fps_mean": float(1000.0 / arr.mean()),
        "peak_vram_mb_5090": round(peak_mb, 1),
        "engine_path": str(INT8_ENGINE.relative_to(REPO)),
        "engine_size_mb": round(INT8_ENGINE.stat().st_size / 1e6, 1),
    }


# ────────────────────────── Main ──────────────────────────

def main():
    log.info("CUDA: %s", torch.cuda.get_device_name(0))
    log.info("TRT version: %s", trt.__version__)

    export_onnx()
    build_int8_engine()

    log.info("Loading engine ...")
    engine = load_engine(INT8_ENGINE)
    log.info("Engine loaded — running sustained inference on 5090 ...")
    result = run_inference(engine)

    log.info("=" * 60)
    log.info("ResNet-50v1 INT8 TRT @ 224×224 on RTX 5090:")
    log.info("  p50 latency:     %.3f ms", result["p50_ms"])
    log.info("  mean latency:    %.3f ms", result["mean_ms"])
    log.info("  p95 latency:     %.3f ms", result["p95_ms"])
    log.info("  FPS (p50):       %.1f inf/s", result["fps_p50"])
    log.info("  FPS (mean):      %.1f inf/s", result["fps_mean"])
    log.info("  peak VRAM:       %.1f MB", result["peak_vram_mb_5090"])
    log.info("  engine size:     %.1f MB", result["engine_size_mb"])
    log.info("=" * 60)

    # Compare against sizer's edge anchor (Kyle 2026-05-01: Low-LP5X 1125 FPS)
    edge_fps = 1125.0
    speedup_5090_vs_edge = result["fps_p50"] / edge_fps
    log.info("Edge anchor (Low-LP5X 100-TOPS, ce03030): 1125 FPS → 0.889 ms")
    log.info("5090 / edge throughput ratio: %.2f×", speedup_5090_vs_edge)

    SUMMARY_PATH.write_text(json.dumps({
        "host": "RTX 5090",
        "torch": torch.__version__,
        "trt": trt.__version__,
        "result": result,
        "edge_anchor_low_lp5x_fps": edge_fps,
        "speedup_5090_vs_edge_p50": round(speedup_5090_vs_edge, 2),
    }, indent=2))
    log.info("Wrote %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
