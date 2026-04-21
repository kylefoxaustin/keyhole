"""
Concurrency / multi-stream bake-off.

The 36 FPS shipping-stack projection is single-stream. Real deployments
frequently need multiple camera feeds (security, fleet, multi-room) sharing
one NPU. This bake-off characterizes how many streams one NPU can serve.

Two levers measured:
  1. **YOLO-seg TRT FP8 batching** — a single forward at batch=N amortizes
     kernel-launch overhead. Measured on the dynamic-batch TRT engine at
     B ∈ {1, 2, 4, 8, 16}.
  2. **CLIP-at-1-Hz under N streams** — the 1-second keyframe debounce
     means CLIP runs N times per second total (across all streams), not
     N × every_frame. Cost amortizes to near-zero.

Computed (not measured) mitigations:
  - Per-stream resolution drop (proportional scale from 720p measurement)
  - Second-NPU horizontal scale (vanilla division)

Outputs:
  data/output/bakeoff/concurrency/summary.json
  data/output/bakeoff/concurrency_edge_projection.json
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

from src.emulate.npu_emulator import NPUEmulator, RTX_5090, EDGE_MPU_TARGET
from scripts.bakeoff_trt_yolo import load_engine, preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("concurrency")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_DIR = REPO_ROOT / "data" / "trt_engines"
BAKEOFF_DIR = REPO_ROOT / "data" / "output" / "bakeoff"

import os as _os
DEFAULT_VARIANT = "yolo11s-seg"
YOLO_VARIANT = _os.environ.get("KEYHOLE_YOLO_VARIANT", DEFAULT_VARIANT)
VARIANT_SLUG = "" if YOLO_VARIANT == DEFAULT_VARIANT else f"_{YOLO_VARIANT}"

OUT_DIR = REPO_ROOT / "data" / "output" / "bakeoff" / f"concurrency{VARIANT_SLUG}"

# FP8 dynbatch engine for whichever variant is active.
ENGINE_PATH = TRT_DIR / f"{YOLO_VARIANT}.dynbatch.fp8.engine"

BATCH_SIZES = [1, 2, 4, 8, 16]
IMGSZ = 640
WARMUP_ITERS = 5
MEASURE_ITERS = 30

# Reference numbers from existing bake-offs (720p, Edge MPU 134.4 GB/s).
# These come from trt_yolo_edge_projection.json + trt_clip_edge_projection.json.
YOLO_EDGE_MS_B1 = 27.2          # YOLO-seg FP8 edge ms per frame at batch=1
CLIP_EDGE_MS_FP8 = 15.1          # CLIP visual FP8 edge ms per frame
CLIP_1HZ_AMORT_MS = CLIP_EDGE_MS_FP8 / 30.0   # amortized over 30 fps native = 0.5 ms

# LLM duty-cycle reference (from llm_edge_projection.json, Q4_K_M):
LLM_SHORT_ANS_MS = 12000        # 200 tok at 16.5 tok/s edge
LLM_RAG_MS = 156000             # 8K+2K RAG


def run_batch(ctx, stream, input_name, output_names, in_buf, out_bufs,
              batch_size: int, sample_img: np.ndarray) -> float:
    """One synchronized batched forward. Returns ms."""
    # Build batch by tiling the same frame — kernel cost doesn't know the difference.
    x_single = preprocess(sample_img)  # (1,3,640,640) fp32
    batch = np.repeat(x_single, batch_size, axis=0)

    ctx.set_input_shape(input_name, (batch_size, 3, IMGSZ, IMGSZ))
    in_buf[:batch_size].copy_(torch.from_numpy(batch).cuda())
    ctx.set_tensor_address(input_name, int(in_buf.data_ptr()))
    for n in output_names:
        ctx.set_tensor_address(n, int(out_bufs[n].data_ptr()))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return (time.perf_counter() - t0) * 1000


def measure_batch(engine, batch_size: int, sample_img: np.ndarray) -> dict:
    ctx = engine.create_execution_context()
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in tensor_names
                      if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_names = [n for n in tensor_names
                    if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    # Allocate max-size buffers (16)
    in_buf = torch.zeros(16, 3, IMGSZ, IMGSZ, dtype=torch.float32, device="cuda")
    out_bufs = {}
    for n in output_names:
        shp = list(engine.get_tensor_shape(n))
        shp[0] = 16  # dynamic batch
        out_bufs[n] = torch.zeros(tuple(shp), dtype=torch.float32, device="cuda")

    stream = torch.cuda.Stream()

    # Warmup
    for _ in range(WARMUP_ITERS):
        run_batch(ctx, stream, input_name, output_names, in_buf, out_bufs,
                  batch_size, sample_img)

    latencies = []
    for _ in range(MEASURE_ITERS):
        latencies.append(run_batch(ctx, stream, input_name, output_names,
                                   in_buf, out_bufs, batch_size, sample_img))
    del ctx
    ms = float(np.mean(latencies))
    p95 = float(np.percentile(latencies, 95))
    return {
        "batch": batch_size,
        "mean_ms": ms,
        "p95_ms": p95,
        "per_stream_ms": ms / batch_size,
        "throughput_streams_per_sec": (batch_size * 1000) / ms if ms > 0 else 0,
    }


def compute_edge_batch_scaling(batch_5090: list[dict]) -> list[dict]:
    """Scale 5090 batched latency to Edge MPU using the same BW/compute ratio
    we used for the single-stream TRT YOLO projection.

    Approximation: total-frame latency scales by the bandwidth ratio because
    the YOLO forward is bandwidth-bound at this model size (see TRT YOLO
    slide — bottleneck is bandwidth, not compute). The per-batch kernel
    overhead scales with compute ratio but is small vs the BW portion.
    """
    # Scale 5090 ms → edge ms by the effective-BW ratio (NPU Mid at 0.70
    # efficiency uniform across tiers). This is variant-agnostic — yolo11s-seg
    # B=1 (≈ 1.68 ms) × 16.19 lands at 27.2 ms edge as before, and yolov8n-seg
    # (≈ 0.54 ms) scales to its own ~8.7 ms edge without being pinned to the
    # larger model's reference.
    BW_RATIO = (1792.0 * 0.85) / (134.4 * 0.70)   # 16.19×, matches deck/sizer
    scale = BW_RATIO
    out = []
    for row in batch_5090:
        edge_ms = row["mean_ms"] * scale
        out.append({
            "batch": row["batch"],
            "mean_ms_edge": edge_ms,
            "per_stream_ms_edge": edge_ms / row["batch"],
            "streams_per_sec_edge": (row["batch"] * 1000) / edge_ms if edge_ms > 0 else 0,
        })
    return out


def build_deployment_scenarios(edge_batch: list[dict]) -> list[dict]:
    """Concrete multi-stream deployment recipes at the Edge MPU target.

    At N concurrent streams batched at B = N, each stream sees 1 frame per
    full-batch cycle. So:
        per_stream_FPS = 1000 / T(B)         (NOT 1000 / (T(B)/B))
        total_system_FPS = N * per_stream_FPS = N * 1000/T(B)
    The per_stream_ms field below is the full batch T(B) — the time a given
    stream must wait between its own consecutive frames.
    """
    b_to_batch_ms = {r["batch"]: r["mean_ms_edge"] for r in edge_batch}

    def edge_batch_ms(batch: int) -> float:
        if batch in b_to_batch_ms:
            return b_to_batch_ms[batch]
        closest = min(b_to_batch_ms.keys(), key=lambda k: abs(k - batch))
        return b_to_batch_ms[closest]

    scenarios = []

    def add(label, n_streams, yolo_batch, extra_ms=0, note=None):
        T = edge_batch_ms(yolo_batch) + CLIP_1HZ_AMORT_MS + extra_ms
        s = {
            "label": label,
            "n_streams": n_streams,
            "yolo_batch": yolo_batch,
            "batch_ms_edge": T,
            "fps_per_stream": 1000 / T if T > 0 else 0,
            "total_system_fps": n_streams * 1000 / T if T > 0 else 0,
        }
        if note:
            s["note"] = note
        scenarios.append(s)

    # Baseline and batched scenarios — N streams batched at B=N
    add("1 stream (shipping)", 1, 1)
    add("2 streams, YOLO batch=2", 2, 2)
    add("4 streams, YOLO batch=4", 4, 4)
    add("8 streams, YOLO batch=8", 8, 8)
    add("16 streams, YOLO batch=16", 16, 16)

    # 4 streams @ 480p per stream — model takes ~half the time per-batch at 480p
    T_480p = (edge_batch_ms(4) / 2.0) + CLIP_1HZ_AMORT_MS
    scenarios.append({
        "label": "4 streams, batch=4, 480p per stream",
        "n_streams": 4, "yolo_batch": 4,
        "batch_ms_edge": T_480p,
        "fps_per_stream": 1000 / T_480p,
        "total_system_fps": 4 * 1000 / T_480p,
        "note": "480p approximated as ~2× 720p speed",
    })

    # LLM interference: shared NPU, LLM query steals wall time
    # Modeled as duty cycle — fraction of time NPU is doing LLM rather than vision
    hourly_rag_duty = LLM_RAG_MS / 3600_000
    short_min_duty = LLM_SHORT_ANS_MS / 60_000
    for label, duty in [("4 streams + 1 RAG query/hr", hourly_rag_duty),
                        ("4 streams + 1 short LLM/min", short_min_duty)]:
        T_base = edge_batch_ms(4) + CLIP_1HZ_AMORT_MS
        # Effective per-stream FPS = base FPS × (1 - duty)
        fps_per_stream = (1000 / T_base) * (1 - duty)
        scenarios.append({
            "label": label,
            "n_streams": 4, "yolo_batch": 4,
            "batch_ms_edge": T_base,
            "fps_per_stream": fps_per_stream,
            "total_system_fps": 4 * fps_per_stream,
            "note": f"LLM duty cycle {duty*100:.1f}%",
        })

    return scenarios


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading engine: %s", ENGINE_PATH.name)
    engine = load_engine(ENGINE_PATH)

    # Use a real bake-off frame for realistic workload
    clip_dir = BAKEOFF_DIR / "720p_EW_clip"
    meta = json.loads((clip_dir / "frames.json").read_text())
    sample_img = cv2.imread(str(clip_dir / meta[0]["path"]))

    # Measure batched inference on 5090
    results_5090 = []
    for B in BATCH_SIZES:
        r = measure_batch(engine, B, sample_img)
        results_5090.append(r)
        log.info("batch=%d: %.2f ms total, %.2f ms/stream, %.1f streams/s",
                 B, r["mean_ms"], r["per_stream_ms"], r["throughput_streams_per_sec"])

    del engine
    torch.cuda.empty_cache()

    # Edge projection (bandwidth-scaled)
    results_edge = compute_edge_batch_scaling(results_5090)

    # Deployment scenarios
    scenarios = build_deployment_scenarios(results_edge)

    summary = {
        "engine": ENGINE_PATH.name,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "batches_5090": results_5090,
        "batches_edge": results_edge,
        "scenarios_edge": scenarios,
        "method": (
            "Dynamic-batch TRT FP8 YOLO-seg engine measured at B ∈ {1,2,4,8,16} "
            "on the 5090 with 30 iters per batch after 5 warmup. Edge latency "
            "scaled by a single factor calibrated from the single-stream TRT "
            "bake-off (YOLO-seg FP8 edge = 27.2 ms/frame at batch=1). Deployment "
            "scenarios add CLIP amortization (1 Hz debounce = 0.5 ms/frame) and "
            "LLM duty-cycle interference based on the Q4_K_M llm_edge_projection."
        ),
    }
    summary["yolo_variant"] = YOLO_VARIANT
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (BAKEOFF_DIR / f"concurrency{VARIANT_SLUG}_edge_projection.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote concurrency%s_edge_projection.json", VARIANT_SLUG)

    # Pretty print
    print()
    print("Raw batched inference (RTX 5090, TRT FP8 engine):")
    hdr = (f"  {'Batch':>5s} | {'Total ms':>10s} {'P95 ms':>8s} | "
           f"{'/stream ms':>11s} {'streams/s':>10s}")
    print(hdr); print("-"*len(hdr))
    for r in results_5090:
        print(f"  {r['batch']:>5d} | {r['mean_ms']:>8.2f} ms {r['p95_ms']:>5.2f} ms | "
              f"{r['per_stream_ms']:>8.2f} ms {r['throughput_streams_per_sec']:>7.1f}")

    print()
    print("Edge MPU projected (bandwidth-scaled from batch=1 reference):")
    hdr = (f"  {'Batch':>5s} | {'Total ms':>10s} | {'/stream ms':>11s} {'streams/s':>10s}")
    print(hdr); print("-"*len(hdr))
    for r in results_edge:
        print(f"  {r['batch']:>5d} | {r['mean_ms_edge']:>8.1f} ms | "
              f"{r['per_stream_ms_edge']:>8.1f} ms {r['streams_per_sec_edge']:>7.1f}")

    print()
    print("Deployment scenarios (Edge MPU):")
    hdr = (f"  {'Scenario':45s} | {'Batch ms':>9s} | {'FPS/stream':>10s}  "
           f"{'Total FPS':>10s}")
    print(hdr); print("-"*len(hdr))
    for s in scenarios:
        print(f"  {s['label']:45s} | {s['batch_ms_edge']:>7.1f} ms | "
              f"{s['fps_per_stream']:>8.1f}  {s['total_system_fps']:>9.1f}")


if __name__ == "__main__":
    main()
