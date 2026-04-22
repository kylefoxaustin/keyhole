"""
Build ONNX + TensorRT engines for a YOLO-seg variant (default yolov8n-seg).

Run once per variant. Produces these files under data/trt_engines/:
  {variant}.onnx                fixed-batch ONNX
  {variant}.dynbatch.onnx       dynamic-batch ONNX (for concurrency bake-off)
  {variant}.fp16.engine         TRT FP16 (via ultralytics)
  {variant}.fp8.engine          TRT FP8 (hand-rolled)
  {variant}.dynbatch.fp16.engine TRT FP16 dynamic-batch (via trtexec)
  {variant}.dynbatch.fp8.engine TRT FP8 dynamic-batch (hand-rolled)

INT8 is skipped here — ultralytics' INT8 export needs a dataset YAML; we
can add a separate PTQ calibrator later if needed.

Usage:
  python scripts/build_yolo_seg_engines.py --variant yolov8n-seg
  python scripts/build_yolo_seg_engines.py --variant yolo11s-seg
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRT_DIR = ROOT / "data" / "trt_engines"
WEIGHTS_DIR = ROOT / "weights"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_yolo_seg")


def ensure_weights(variant: str) -> Path:
    """Download the YOLO weights file if not already cached."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out = WEIGHTS_DIR / f"{variant}.pt"
    if out.exists():
        log.info("Weights already cached at %s", out)
        return out
    from ultralytics import YOLO
    log.info("Downloading %s.pt via ultralytics…", variant)
    cwd_pt = ROOT / f"{variant}.pt"
    if cwd_pt.exists():
        shutil.move(str(cwd_pt), out)
        return out
    # Triggers ultralytics' auto-download into cwd
    _ = YOLO(f"{variant}.pt")
    if cwd_pt.exists():
        shutil.move(str(cwd_pt), out)
    elif not out.exists():
        raise RuntimeError(f"Ultralytics did not produce {variant}.pt as expected")
    return out


def export_onnx(pt_path: Path, out_path: Path, dynamic: bool):
    """Export the YOLO .pt to ONNX via ultralytics."""
    if out_path.exists():
        log.info("ONNX already present at %s", out_path)
        return
    from ultralytics import YOLO
    log.info("Exporting ONNX (dynamic=%s) from %s …", dynamic, pt_path)
    m = YOLO(str(pt_path))
    # ultralytics writes <stem>.onnx into the same directory as <pt_path>
    res = m.export(format="onnx", imgsz=640, dynamic=dynamic, opset=17, simplify=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = Path(res) if isinstance(res, str) else pt_path.with_suffix(".onnx")
    if src.resolve() != out_path.resolve():
        shutil.move(str(src), out_path)
    log.info("  ✓ %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def export_fp16_engine(pt_path: Path, out_path: Path):
    """Build FP16 TensorRT engine via ultralytics' .export(format='engine')."""
    if out_path.exists():
        log.info("FP16 engine already present at %s", out_path)
        return
    from ultralytics import YOLO
    import torch
    log.info("Building FP16 engine for %s (cuda_available=%s) …", pt_path.name, torch.cuda.is_available())
    m = YOLO(str(pt_path))
    # ultralytics writes <stem>.engine in the same dir as the .pt. Explicit
    # device=0 avoids the "cuda unavailable" false positive we saw in some
    # ultralytics 8.4.37 paths where it probed before loading the .pt.
    res = m.export(format="engine", imgsz=640, half=True, dynamic=False, device=0)
    src = Path(res) if isinstance(res, str) else pt_path.with_suffix(".engine")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), out_path)
    log.info("  ✓ %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def build_fp8_engine(onnx_path: Path, out_path: Path, dynbatch_profile: bool = False):
    """Hand-roll a TRT FP8 engine from the ONNX.

    Mirrors the build_engine() path in bakeoff_trt_yolo.py — tries progressively
    narrower flag sets if a broader one fails.
    """
    if out_path.exists():
        log.info("FP8 engine already present at %s", out_path)
        return
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    def build(flags):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errs = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parse failed: {errs}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        for f in flags:
            config.set_flag(f)
        if dynbatch_profile:
            profile = builder.create_optimization_profile()
            inp_name = network.get_input(0).name
            profile.set_shape(inp_name, min=(1, 3, 640, 640),
                               opt=(4, 3, 640, 640), max=(16, 3, 640, 640))
            config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"build_serialized_network returned None (flags={[f.name for f in flags]})")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(serialized)

    flag_sets = [
        {trt.BuilderFlag.FP8, trt.BuilderFlag.FP16, trt.BuilderFlag.BF16},
        {trt.BuilderFlag.FP8, trt.BuilderFlag.FP16},
        {trt.BuilderFlag.FP8},
    ]
    last_err = None
    for fs in flag_sets:
        try:
            log.info("Trying FP8 build with %s …", [f.name for f in fs])
            build(fs)
            log.info("  ✓ %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
            return
        except Exception as e:
            log.warning("  build failed: %s", str(e)[:200])
            last_err = e
            if out_path.exists():
                out_path.unlink()
    raise RuntimeError(f"All FP8 flag combinations failed for {onnx_path}: {last_err}")


def build_int8_engine(variant: str, onnx_path: Path, out_path: Path):
    """Build a TRT INT8 engine from the ONNX using the bake-off's
    Int8EntropyCalibrator2 over 20 bake-off frames.

    Reuses BakeoffCalibrator from scripts.bakeoff_trt_yolo so the calibration
    stream matches what the FP8/FP16 engines see during timing.
    """
    if out_path.exists():
        log.info("INT8 engine already present at %s", out_path)
        return
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from scripts.bakeoff_trt_yolo import BakeoffCalibrator, gather_calib_paths  # type: ignore
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # Gather 20 calibration frames from the cached bake-off clips
    calib_images = gather_calib_paths()
    cache_file = out_path.with_suffix(".calib_cache")
    calib = BakeoffCalibrator(calib_images, cache_file)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calib
    log.info("Calibrating + building INT8 engine for %s on %d frames …",
             variant, len(calib_images))
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"INT8 engine build returned None for {variant}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(serialized)
    log.info("  ✓ %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def build_fp16_dynbatch_engine(onnx_path: Path, out_path: Path):
    """Hand-roll FP16 dynbatch engine for the concurrency bake-off."""
    if out_path.exists():
        log.info("FP16 dynbatch engine already present at %s", out_path)
        return
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    inp_name = network.get_input(0).name
    profile.set_shape(inp_name, min=(1, 3, 640, 640),
                       opt=(4, 3, 640, 640), max=(16, 3, 640, 640))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(serialized)
    log.info("  ✓ %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="yolov8n-seg",
                     help="YOLO-seg model name (e.g. yolov8n-seg, yolo11s-seg, yolo11n-seg).")
    args = ap.parse_args()
    variant = args.variant

    TRT_DIR.mkdir(parents=True, exist_ok=True)
    pt = ensure_weights(variant)

    # Fixed-batch ONNX + engines
    onnx_fixed = TRT_DIR / f"{variant}.onnx"
    fp16_engine = TRT_DIR / f"{variant}.fp16.engine"
    int8_engine = TRT_DIR / f"{variant}.int8.engine"
    fp8_engine = TRT_DIR / f"{variant}.fp8.engine"

    export_onnx(pt, onnx_fixed, dynamic=False)
    export_fp16_engine(pt, fp16_engine)
    build_int8_engine(variant, onnx_fixed, int8_engine)
    build_fp8_engine(onnx_fixed, fp8_engine)

    # Dynbatch ONNX + engines (for concurrency bake-off)
    onnx_dyn = TRT_DIR / f"{variant}.dynbatch.onnx"
    fp16_dyn = TRT_DIR / f"{variant}.dynbatch.fp16.engine"
    fp8_dyn = TRT_DIR / f"{variant}.dynbatch.fp8.engine"

    export_onnx(pt, onnx_dyn, dynamic=True)
    build_fp16_dynbatch_engine(onnx_dyn, fp16_dyn)
    build_fp8_engine(onnx_dyn, fp8_dyn, dynbatch_profile=True)

    log.info("")
    log.info("=== Summary: engines for %s ===", variant)
    for p in [onnx_fixed, fp16_engine, int8_engine, fp8_engine, onnx_dyn, fp16_dyn, fp8_dyn]:
        if p.exists():
            log.info("  %s (%.1f MB)", p.name, p.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
