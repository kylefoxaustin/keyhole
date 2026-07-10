#!/usr/bin/env python3
"""
bench_corpus_trt_py.py — 5090-side twin of bench_corpus_trt.py.

The Orin ships a `trtexec` binary; this host does not (TensorRT arrived as a pip
wheel inside the `keyhole` venv). So the same corpus is measured through the
TensorRT Python API, deliberately reproducing trtexec's semantics so the two
hosts can be diffed without an asterisk:

  * identical ONNX (static, batch=1) on both sides
  * warm up for `--warmup-ms`, then time for `--duration` seconds
  * compute_ms  = CUDA events around execute_async_v3 only  -> pure GPU kernel time
  * latency_ms  = CUDA events spanning H2D + compute + D2H  -> end-to-end
  * both reported as median/mean/p95/p99 over every timed iteration

Buffers are allocated once and reused, and the same device pointers are bound for
every iteration, matching trtexec's steady-state behaviour (it does not re-upload
per iteration when measuring compute).

INT8 is PERF-ONLY here exactly as in the trtexec path: no calibrator, so TensorRT
assigns arbitrary dynamic ranges. Kernel selection and latency are representative;
the numerics are meaningless. Do not quote accuracy off these runs.

Run:  ~/.virtualenvs/keyhole/bin/python scripts/bench_corpus_trt_py.py \
          --onnx-dir data/output/onnx_corpus_iq9 --out data/output/rtx5090_corpus_trt.json \
          --host-label rtx-5090
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

MODELS = [
    "resnet50v1",
    "yolov8n-seg",
    "efficient_sam_vitt_decoder",
    "efficient_sam_vitt_encoder",
    "yolo11s-seg",
    "yoloe-26s-seg-pf",
    "clip_vit_b32_visual",
]

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}


def set_tensor_scales(network, in_scale=2.0, out_scale=4.0):
    """Assign arbitrary dynamic ranges, exactly as trtexec does for --int8 with no
    calibrator (TensorRT samples: setTensorScales, inScales=2.0, outScales=4.0).

    Without this the builder throws 'bad optional access'. Reproducing trtexec's
    constants — rather than picking our own — is what keeps the Orin (trtexec) and
    5090 (this file) runs comparable: identical kernel selection, identically
    meaningless numerics.
    """
    def unset(tensor):
        try:                       # getter returns None when no range was assigned
            return tensor.dynamic_range is None
        except (AttributeError, RuntimeError):
            return True

    for i in range(network.num_inputs):
        t = network.get_input(i)
        if unset(t):
            t.set_dynamic_range(-in_scale, in_scale)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            t = layer.get_output(j)
            if unset(t):
                t.set_dynamic_range(-out_scale, out_scale)


def build_engine(onnx_path, precision, logger, workspace_gb=8):
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read(), path=str(onnx_path)):  # path => external data resolves
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errs[:3]}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if precision in ("fp16", "int8", "fp8"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)
        set_tensor_scales(network)
    if precision == "fp8":
        cfg.set_flag(trt.BuilderFlag.FP8)

    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise RuntimeError("engine build returned None")
    return plan


def make_io(engine, context):
    """Allocate one persistent device buffer per tensor; bind addresses once."""
    inputs, outputs = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = TRT_TO_TORCH[engine.get_tensor_dtype(name)]
        shape = tuple(context.get_tensor_shape(name))
        if dtype.is_floating_point:
            buf = torch.randn(shape, dtype=dtype, device="cuda")
        else:
            buf = torch.ones(shape, dtype=dtype, device="cuda")
        context.set_tensor_address(name, buf.data_ptr())
        (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
         else outputs)[name] = buf
    return inputs, outputs


def percentiles(vals):
    vals = sorted(vals)
    n = len(vals)
    return {
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "p95": vals[min(n - 1, int(0.95 * n))],
        "p99": vals[min(n - 1, int(0.99 * n))],
        "min": vals[0],
        "max": vals[-1],
    }


def bench(engine, context, inputs, outputs, duration_s, warmup_ms):
    stream = torch.cuda.Stream()
    host_in = {k: v.cpu() for k, v in inputs.items()}
    host_out = {k: v.cpu() for k, v in outputs.items()}

    def one_compute():
        context.execute_async_v3(stream.cuda_stream)

    with torch.cuda.stream(stream):
        t_end = time.time() + warmup_ms / 1000.0
        while time.time() < t_end:
            one_compute()
        stream.synchronize()

        compute, latency = [], []
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        t_end = time.time() + duration_s
        while time.time() < t_end:
            ev[0].record(stream)
            for k, v in inputs.items():          # H2D
                v.copy_(host_in[k], non_blocking=True)
            ev[1].record(stream)
            one_compute()                        # compute
            ev[2].record(stream)
            for k, v in outputs.items():         # D2H
                host_out[k].copy_(v, non_blocking=True)
            ev[3].record(stream)
            stream.synchronize()
            compute.append(ev[1].elapsed_time(ev[2]))
            latency.append(ev[0].elapsed_time(ev[3]))

    return {"compute_ms": percentiles(compute),
            "latency_ms": percentiles(latency),
            "n_iterations": len(compute)}


def run_one(onnx_path, precision, duration_s, warmup_ms):
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    rec = {"model": onnx_path.stem, "precision": precision, "ok": False}
    t0 = time.time()
    try:
        plan = build_engine(onnx_path, precision, logger)
        rec["engine_size_mb"] = round(plan.nbytes / 1e6, 2)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        inputs, outputs = make_io(engine, context)
        rec.update(bench(engine, context, inputs, outputs, duration_s, warmup_ms))
        rec["throughput_qps"] = 1000.0 / rec["compute_ms"]["median"]
        rec["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        rec["ok"] = True
        del context, engine, inputs, outputs
    except Exception as exc:                                   # noqa: BLE001
        rec["error_tail"] = [f"{type(exc).__name__}: {exc}"[:400]]
    finally:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rec["build_plus_run_wall_s"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host-label", required=True)
    ap.add_argument("--precisions", default="fp16,int8")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--warmup-ms", type=int, default=3000)
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    results = []
    models = [m for m in args.models.split(",") if m]
    precisions = [p for p in args.precisions.split(",") if p]
    total = len(models) * len(precisions)
    i = 0
    for model in models:
        onnx_path = onnx_dir / f"{model}.onnx"
        if not onnx_path.exists():
            print(f"!! missing {onnx_path}", flush=True)
            continue
        for precision in precisions:
            i += 1
            print(f"[{i}/{total}] {model} {precision} ...", flush=True)
            rec = run_one(onnx_path, precision, args.duration, args.warmup_ms)
            if rec["ok"]:
                print(f"    compute p50 {rec['compute_ms']['median']:.3f} ms "
                      f"({rec['throughput_qps']:.1f} qps, "
                      f"build+run {rec['build_plus_run_wall_s']}s)", flush=True)
            else:
                print(f"    FAILED: {rec['error_tail']}", flush=True)
            results.append(rec)
            Path(args.out).write_text(json.dumps({
                "__meta__": {
                    "description": "keyhole vision corpus, same ONNX, TensorRT on one host.",
                    "harness": "TensorRT Python API; semantics matched to trtexec "
                               "(see scripts/bench_corpus_trt_py.py docstring)",
                    "int8_caveat": "PERF-ONLY: no calibration cache, arbitrary dynamic "
                                   "ranges. Latency representative, numerics are NOT.",
                    "compute_ms": "pure GPU kernel time (no H2D/D2H) — the cross-platform number",
                    "latency_ms": "end-to-end incl. host transfers — flatters unified memory",
                    "schema_version": 1,
                },
                "host": {
                    "host_label": args.host_label,
                    "machine": platform.machine(),
                    "gpu": torch.cuda.get_device_name(0),
                    "tensorrt": trt.__version__,
                    "torch": torch.__version__,
                },
                "config": {"duration_s": args.duration, "warmup_ms": args.warmup_ms},
                "results": results,
            }, indent=2))
    print(f"\nwrote {args.out}  ({sum(r['ok'] for r in results)}/{len(results)} ok)")


if __name__ == "__main__":
    main()
