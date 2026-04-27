"""Minimal INT8 vs FP8 yolov8n-seg runner for ncu DP4A instruction-mix probe.

Answers the question [docs] raised 2026-04-23: does keyhole's INT8 TRT engine
on Blackwell SM120 route through DP4A CUDA cores (Pascal-era integer SIMD) or
through the (now-dropped-on-consumer-Blackwell) IMMA tensor-core path?

Runs both engines inside distinct NVTX ranges so ncu can attribute kernels:
  - yolo_int8_dp4a_probe  → INT8 engine
  - yolo_fp8_tc_probe     → FP8 engine (tensor-core QMMA control case)

No timing here — ncu overhead invalidates wall-clock. Intended target:

  sudo ncu --nvtx \\
      --nvtx-include 'yolo_int8_dp4a_probe' \\
      --nvtx-include 'yolo_fp8_tc_probe' \\
      --metrics <IDP/IMMA/tensor/int-pipe set> \\
      --csv --log-file <out> \\
      python scripts/ncu_dp4a_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import tensorrt as trt
import torch

from src.profiling.nvtx_helpers import nvtx_range

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(path: Path):
    """Deserialize a TRT engine, skipping ultralytics' length-prefixed JSON header."""
    runtime = trt.Runtime(TRT_LOGGER)
    raw = path.read_bytes()
    if len(raw) > 4 and raw[4:5] == b"{":
        meta_len = int.from_bytes(raw[:4], "little")
        if 4 + meta_len < len(raw):
            raw = raw[4 + meta_len:]
    return runtime.deserialize_cuda_engine(raw)


def run_engine(engine_path: Path, nvtx_name: str, n_iters: int = 5):
    print(f"  loading {engine_path.name}")
    engine = load_engine(engine_path)
    ctx = engine.create_execution_context()

    input_name = next(
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
    )
    ctx.set_input_shape(input_name, (1, 3, 640, 640))

    buffers = {}
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        shape = tuple(ctx.get_tensor_shape(nm))
        dt = trt.nptype(engine.get_tensor_dtype(nm))
        tdt = {
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.int8: torch.int8,
        }.get(dt, torch.float16)
        buf = torch.zeros(shape, dtype=tdt, device="cuda")
        buffers[nm] = buf
        ctx.set_tensor_address(nm, int(buf.data_ptr()))

    stream = torch.cuda.Stream()

    for _ in range(2):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    with nvtx_range(nvtx_name):
        for _ in range(n_iters):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
    print(f"  ran {n_iters} iters inside NVTX range '{nvtx_name}'")


def main():
    trt_dir = REPO / "data" / "trt_engines"
    print("→ INT8 engine")
    run_engine(trt_dir / "yolov8n-seg.int8.engine", "yolo_int8_dp4a_probe")
    print("→ FP8 engine")
    run_engine(trt_dir / "yolov8n-seg.fp8.engine", "yolo_fp8_tc_probe")
    print("DONE")


if __name__ == "__main__":
    main()
