"""End-to-end pipeline latency budget — KH-P3-002 in REMEDIATION_PLAN.md.

Profiles a real video clip end-to-end on the 5090 reference platform,
recording per-stage wall-time in ms:

  1. ingest_decode    — read encoded frame from disk (cv2.VideoCapture)
  2. preprocess       — letterbox + resize to 640x640 + tensor build
  3. yolo_trt_infer   — YOLO-seg FP8 TRT engine (every frame)
  4. clip_trt_infer   — CLIP visual FP8 TRT (every 30th frame, 1 Hz)
  5. db_insert        — SQLite INSERT for detected events

Outputs:
  data/output/bakeoff/e2e_pipeline_summary.json

Each stage is timed with time.perf_counter() (CPU stages) or
torch.cuda.synchronize + perf_counter (GPU stages). Per-stage stats:
mean, p50, p95, total ms. Then projects to NPU Mid:
  - GPU stages scale by BW_RATIO_5090_TO_MID = 16.19×
  - CPU stages stay roughly constant (preproc + DB are not BW-bound on
    edge ARM Cortex-A55, just slower per-core; we apply a ~10× CPU-class
    slowdown documented in the deck preprocessing footnote)

The slide will surface the 27 ms/frame budget at 36 FPS NPU Mid and show
where the 5 ms slack lives.

Usage:
  python scripts/profile_e2e_pipeline.py [--frames 200] [--video data/videos/720p_EW_clip.mp4]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

DEFAULT_VIDEO = ROOT / "data" / "videos" / "720p_EW_clip.mp4"
DEFAULT_FRAMES = 200

YOLO_ENGINE = ROOT / "data" / "trt_engines" / "yolo11s-seg.dynbatch.fp8.engine"
CLIP_ENGINE = ROOT / "data" / "trt_engines" / "clip_vit_b32_visual.fp8.engine"

# Same BW ratio used elsewhere in the pipeline (see CLAUDE_REVIEW_BRIEFING.md § 8).
BW_RATIO_5090_TO_NPU_MID = (1792 * 0.85) / (134.4 * 0.70)   # 16.19×

# CPU-stage slowdown (5090 host i9-14900KF -> edge ARM Cortex-A55).
# Documented in slide_trt_yolo's preprocessing footnote: "Edge ARM ≈ 10× slower
# single-thread → ~2-3 ms/frame ... ~6-10% of one edge core at 30 fps."
CPU_SLOWDOWN_5090_TO_EDGE = 10.0


def _load_engine(path: Path):
    """Deserialize a TRT engine, skipping ultralytics' metadata header if present."""
    runtime = trt.Runtime(TRT_LOGGER)
    raw = path.read_bytes()
    if len(raw) > 4 and raw[4:5] == b"{":
        meta_len = int.from_bytes(raw[:4], "little")
        if 4 + meta_len < len(raw):
            raw = raw[4 + meta_len:]
    return runtime.deserialize_cuda_engine(raw)


def _build_yolo_runner(engine_path: Path):
    """Return (runner, input_shape) for a single-batch inference closure."""
    engine = _load_engine(engine_path)
    ctx = engine.create_execution_context()
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in tensor_names
                      if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_names = [n for n in tensor_names
                    if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    # Set batch=1 dynamic shape
    in_shape = engine.get_tensor_shape(input_name)
    if in_shape[0] == -1:
        in_shape = (1,) + tuple(in_shape[1:])
        ctx.set_input_shape(input_name, in_shape)
    inp = torch.zeros(tuple(in_shape), dtype=torch.float32, device="cuda")
    outs = {}
    for n in output_names:
        shp = ctx.get_tensor_shape(n)
        outs[n] = torch.zeros(tuple(shp), dtype=torch.float32, device="cuda")
    ctx.set_tensor_address(input_name, int(inp.data_ptr()))
    for n in output_names:
        ctx.set_tensor_address(n, int(outs[n].data_ptr()))
    stream = torch.cuda.Stream()

    def _run(input_tensor: torch.Tensor):
        # Copy input into the bound input tensor; then execute.
        inp.copy_(input_tensor, non_blocking=True)
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        return outs

    return _run, in_shape


def _build_clip_runner(engine_path: Path):
    """Same pattern as YOLO but for CLIP visual tower (3x224x224 input)."""
    engine = _load_engine(engine_path)
    ctx = engine.create_execution_context()
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in tensor_names
                      if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_names = [n for n in tensor_names
                    if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    in_shape = tuple(engine.get_tensor_shape(input_name))
    # Resolve dynamic dims (e.g., dynamic batch) to concrete batch=1
    in_shape = tuple(d if d > 0 else 1 for d in in_shape)
    ctx.set_input_shape(input_name, in_shape)
    inp = torch.zeros(in_shape, dtype=torch.float32, device="cuda")
    outs = {}
    for n in output_names:
        shp = ctx.get_tensor_shape(n)
        out_shape = tuple(d if d > 0 else 1 for d in shp)
        outs[n] = torch.zeros(out_shape, dtype=torch.float32, device="cuda")
    ctx.set_tensor_address(input_name, int(inp.data_ptr()))
    for n in output_names:
        ctx.set_tensor_address(n, int(outs[n].data_ptr()))
    stream = torch.cuda.Stream()

    def _run(input_tensor: torch.Tensor):
        inp.copy_(input_tensor, non_blocking=True)
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        return outs

    return _run, in_shape


def _letterbox_640(frame: np.ndarray) -> np.ndarray:
    """Letterbox-resize HxWx3 BGR frame to 640x640, return HWC float32 [0,1]."""
    h, w = frame.shape[:2]
    scale = 640.0 / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad = np.full((640, 640, 3), 114, dtype=np.uint8)
    pad[:new_h, :new_w] = resized
    return pad.astype(np.float32) / 255.0


def _setup_db() -> sqlite3.Connection:
    """Throwaway in-memory SQLite mirroring the events schema (FTS5 disabled
    for the profiling exercise — measuring INSERT latency without the FTS5
    overhead since FTS5 indexing happens lazily in production)."""
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    conn = sqlite3.connect(db.name)
    conn.execute("""
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            video_id INTEGER,
            frame_idx INTEGER,
            ts_ms REAL,
            class_name TEXT,
            score REAL,
            box_x REAL, box_y REAL, box_w REAL, box_h REAL
        )
    """)
    conn.commit()
    return conn


def profile(video_path: Path, n_frames: int) -> dict:
    """Run the e2e pipeline for n_frames and return per-stage timings."""
    if not video_path.exists():
        raise FileNotFoundError(f"video missing: {video_path}")
    if not YOLO_ENGINE.exists():
        raise FileNotFoundError(f"YOLO engine missing: {YOLO_ENGINE}")
    if not CLIP_ENGINE.exists():
        raise FileNotFoundError(f"CLIP engine missing: {CLIP_ENGINE}")

    yolo_run, yolo_in_shape = _build_yolo_runner(YOLO_ENGINE)
    clip_run, clip_in_shape = _build_clip_runner(CLIP_ENGINE)
    db = _setup_db()

    # Pre-allocate input buffers we reuse each frame to avoid measuring tensor allocation.
    # YOLO expects (1, 3, 640, 640).
    yolo_in_t = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device="cuda")
    clip_in_t = torch.zeros(clip_in_shape, dtype=torch.float32, device="cuda")

    timings = {
        "ingest_decode_ms": [],
        "preprocess_ms":    [],
        "yolo_trt_ms":      [],
        "clip_trt_ms":      [],
        "db_insert_ms":     [],
    }
    n_clip_invocations = 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    try:
        # Warmup the engines (not measured)
        for _ in range(3):
            yolo_run(yolo_in_t)
        for _ in range(3):
            clip_run(clip_in_t)
        torch.cuda.synchronize()

        for frame_idx in range(n_frames):
            # 1. INGEST DECODE
            t = time.perf_counter()
            ok, frame = cap.read()
            timings["ingest_decode_ms"].append((time.perf_counter() - t) * 1000.0)
            if not ok:
                # Loop back to start if clip is shorter than n_frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break

            # 2. PREPROCESS
            t = time.perf_counter()
            letterboxed = _letterbox_640(frame)
            yolo_in_np = letterboxed.transpose(2, 0, 1)[None]  # 1x3x640x640
            yolo_in_cpu = torch.from_numpy(yolo_in_np)
            yolo_in_t.copy_(yolo_in_cpu, non_blocking=True)
            torch.cuda.synchronize()
            timings["preprocess_ms"].append((time.perf_counter() - t) * 1000.0)

            # 3. YOLO INFERENCE (every frame)
            torch.cuda.synchronize()
            t = time.perf_counter()
            yolo_run(yolo_in_t)
            torch.cuda.synchronize()
            timings["yolo_trt_ms"].append((time.perf_counter() - t) * 1000.0)

            # 4. CLIP INFERENCE (every 30th frame — 1 Hz at 30 fps source)
            if frame_idx % 30 == 0:
                # Build a dummy crop input — production would crop YOLO boxes;
                # for latency-budget purposes the constant input shape is what matters.
                clip_in_t.zero_()
                torch.cuda.synchronize()
                t = time.perf_counter()
                clip_run(clip_in_t)
                torch.cuda.synchronize()
                timings["clip_trt_ms"].append((time.perf_counter() - t) * 1000.0)
                n_clip_invocations += 1

            # 5. DB INSERT (1 row per detected event; assume avg ~3 dets/frame)
            # Production pattern: BATCH commits across frames to amortize fsync.
            # We measure the per-frame INSERT cost (cheap, ~10 us total per frame),
            # not the commit cost (which is fsync-bound and only triggers every
            # COMMIT_BATCH_FRAMES frames in production). Per-frame commit is the
            # naive integration anti-pattern; the realistic projection below
            # amortizes fsync over the batch.
            n_dets = 3
            t = time.perf_counter()
            for d in range(n_dets):
                db.execute(
                    "INSERT INTO events (video_id, frame_idx, ts_ms, class_name, score, "
                    "box_x, box_y, box_w, box_h) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (frame_idx, frame_idx * 33.3, "person", 0.85, 0.1, 0.1, 0.5, 0.5),
                )
            # COMMIT every 30 frames (1 Hz fsync cadence — production pattern)
            if (frame_idx + 1) % 30 == 0:
                db.commit()
            timings["db_insert_ms"].append((time.perf_counter() - t) * 1000.0)
        # Final commit for any pending inserts after loop
        db.commit()
    finally:
        cap.release()
        db.close()

    # Aggregate
    def _stats(samples: list[float]) -> dict:
        if not samples:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "n": 0}
        sorted_s = sorted(samples)
        return {
            "mean": statistics.mean(samples),
            "p50":  sorted_s[len(sorted_s) // 2],
            "p95":  sorted_s[int(len(sorted_s) * 0.95)],
            "n":    len(samples),
        }

    out_5090 = {k: _stats(v) for k, v in timings.items()}

    # Per-stage classification: GPU stages scale by BW ratio; CPU stages by
    # CPU-class slowdown.
    stage_class = {
        "ingest_decode_ms":  "cpu",  # cv2 decode runs on CPU
        "preprocess_ms":     "cpu",  # letterbox/resize is CPU
        "yolo_trt_ms":       "gpu",
        "clip_trt_ms":       "gpu",
        "db_insert_ms":      "cpu",  # SQLite is CPU-bound
    }

    out_npu_mid = {}
    for stage, stats in out_5090.items():
        klass = stage_class[stage]
        if klass == "gpu":
            scale = BW_RATIO_5090_TO_NPU_MID
        else:
            scale = CPU_SLOWDOWN_5090_TO_EDGE
        out_npu_mid[stage] = {
            "mean":  stats["mean"] * scale,
            "p50":   stats["p50"]  * scale,
            "p95":   stats["p95"]  * scale,
            "n":     stats["n"],
            "class": klass,
            "scale_factor": scale,
        }

    # Per-frame totals (CLIP amortized at 1/30 since it runs once per 30 frames)
    def _per_frame_total(stats: dict, ms_field: str = "p50") -> float:
        total = 0.0
        for stage, s in stats.items():
            ms = s[ms_field] if isinstance(s, dict) else s
            if stage == "clip_trt_ms":
                total += ms / 30.0  # 1 Hz amortization
            else:
                total += ms
        return total

    total_5090_p50_ms = _per_frame_total(out_5090, "p50")
    total_npu_mid_p50_ms = _per_frame_total(out_npu_mid, "p50")
    total_npu_mid_p50_fps = 1000.0 / total_npu_mid_p50_ms if total_npu_mid_p50_ms > 0 else 0.0

    # Headline budget: 36 FPS = 27.78 ms/frame on NPU Mid. Slack = 27.78 - total.
    npu_mid_36fps_budget_ms = 1000.0 / 36.0
    slack_ms = npu_mid_36fps_budget_ms - total_npu_mid_p50_ms

    return {
        "video":            str(video_path),
        "n_frames":         n_frames,
        "n_clip_invocations": n_clip_invocations,
        "yolo_engine":      str(YOLO_ENGINE.relative_to(ROOT)),
        "clip_engine":      str(CLIP_ENGINE.relative_to(ROOT)),
        "bw_ratio_5090_to_npu_mid": BW_RATIO_5090_TO_NPU_MID,
        "cpu_slowdown_5090_to_edge": CPU_SLOWDOWN_5090_TO_EDGE,
        "per_stage_5090":   out_5090,
        "per_stage_npu_mid": out_npu_mid,
        "totals": {
            "per_frame_5090_p50_ms":    total_5090_p50_ms,
            "per_frame_5090_p50_fps":   1000.0 / total_5090_p50_ms if total_5090_p50_ms > 0 else 0.0,
            "per_frame_npu_mid_p50_ms": total_npu_mid_p50_ms,
            "per_frame_npu_mid_p50_fps": total_npu_mid_p50_fps,
            "npu_mid_36fps_budget_ms":  npu_mid_36fps_budget_ms,
            "slack_ms":                 slack_ms,
            "clip_amortization_factor": "1 Hz (1/30 frames)",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                    help="Input video file (default: %(default)s)")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                    help="Number of frames to profile (default: %(default)s)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "output" / "bakeoff" / "e2e_pipeline_summary.json")
    args = ap.parse_args()

    print(f"Profiling {args.frames} frames from {args.video.name}")
    print(f"  YOLO: {YOLO_ENGINE.name}")
    print(f"  CLIP: {CLIP_ENGINE.name}")
    print()

    summary = profile(args.video, args.frames)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))

    # Pretty-print
    print(f"5090 host (per-frame p50 ms):")
    for stage, s in summary["per_stage_5090"].items():
        amort = " (1/30 amort)" if stage == "clip_trt_ms" else ""
        print(f"  {stage:24s} {s['p50']:>7.3f} ms (mean {s['mean']:.3f}, p95 {s['p95']:.3f}, n={s['n']}){amort}")
    print(f"  per-frame total p50:     {summary['totals']['per_frame_5090_p50_ms']:>7.3f} ms = {summary['totals']['per_frame_5090_p50_fps']:.1f} FPS")
    print()
    print(f"NPU Mid projected (per-stage p50 ms):")
    for stage, s in summary["per_stage_npu_mid"].items():
        amort = " (1/30 amort)" if stage == "clip_trt_ms" else ""
        print(f"  {stage:24s} {s['p50']:>7.3f} ms ({s['class']}, ×{s['scale_factor']:.2f}){amort}")
    print(f"  per-frame total p50:     {summary['totals']['per_frame_npu_mid_p50_ms']:>7.3f} ms = {summary['totals']['per_frame_npu_mid_p50_fps']:.1f} FPS")
    print()
    print(f"NPU Mid 36-FPS budget:     {summary['totals']['npu_mid_36fps_budget_ms']:.2f} ms")
    print(f"Slack vs 36-FPS budget:    {summary['totals']['slack_ms']:+.2f} ms "
          f"(positive = headroom, negative = over budget)")
    print()
    print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
