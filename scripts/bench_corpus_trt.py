#!/usr/bin/env python3
"""
bench_corpus_trt.py — run the keyhole vision corpus through TensorRT on any host.

Same ONNX, same measurement, on the RTX 5090 (sm_120) and the Jetson AGX Orin
(sm_87). qualcomm runs the same ONNX through QNN/HTP on the IQ-9075. That is the
whole point of the three-platform sweep: the graph is a constant, the silicon is
the variable.

Deliberately shells out to `trtexec` rather than using the TensorRT Python API —
trtexec is a C++ binary present on both hosts, which sidesteps the Orin's
torch/triton aarch64 situation entirely.

Two latencies are recorded per run and they mean different things:
  compute_ms  pure GPU kernel time, no H2D/D2H. The apples-to-apples number
              across a discrete GPU and an integrated SoC.
  latency_ms  end-to-end incl. host transfers. Flatters unified-memory parts.
Report compute_ms as the headline; carry latency_ms so nobody has to guess.

INT8 here is PERF-ONLY. Without a calibration cache trtexec assigns arbitrary
dynamic ranges: kernel selection and therefore latency are representative, the
numerics are not. Never quote accuracy from an int8 run made by this script.

Run:  python3 scripts/bench_corpus_trt.py --onnx-dir <dir> --out <json> --host-label orin-agx-64gb
"""
import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Corpus order: cheapest first. A canary ordering means a toolchain problem
# surfaces in 90 seconds on ResNet50, not 40 minutes into the ViT build.
MODELS = [
    "resnet50v1",
    "yolov8n-seg",
    "efficient_sam_vitt_decoder",
    "efficient_sam_vitt_encoder",
    "yolo11s-seg",
    "yoloe-26s-seg-pf",
    "clip_vit_b32_visual",
]

PRECISION_FLAGS = {
    "fp16": ["--fp16"],
    "int8": ["--int8", "--fp16"],  # int8 + fp16 fallback for unsupported layers
    "fp8": ["--fp8", "--fp16"],    # sm_89+/sm_120 only; absent on Orin sm_87
}


def parse_times(times_path):
    """Median/percentiles from trtexec's per-iteration export."""
    with open(times_path) as fh:
        recs = json.load(fh)
    if not recs:
        return None
    out = {}
    for key, field in (("compute_ms", "computeMs"), ("latency_ms", "latencyMs")):
        vals = sorted(r[field] for r in recs)
        n = len(vals)
        out[key] = {
            "median": statistics.median(vals),
            "mean": statistics.fmean(vals),
            "p95": vals[min(n - 1, int(0.95 * n))],
            "p99": vals[min(n - 1, int(0.99 * n))],
            "min": vals[0],
            "max": vals[-1],
        }
    out["n_iterations"] = len(recs)
    return out


def run_one(trtexec, onnx_path, precision, out_dir, duration, warmup_ms):
    stem = onnx_path.stem
    tag = f"{stem}.{precision}"
    times_json = out_dir / f"{tag}.times.json"
    engine = out_dir / f"{tag}.engine"
    log_path = out_dir / f"{tag}.log"

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        *PRECISION_FLAGS[precision],
        f"--warmUp={warmup_ms}",
        f"--duration={duration}",
        "--avgRuns=100",
        f"--exportTimes={times_json}",
        f"--saveEngine={engine}",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_s = time.time() - t0
    log_path.write_text(proc.stdout + "\n===STDERR===\n" + proc.stderr)

    ok = proc.returncode == 0 and times_json.exists()
    rec = {
        "model": stem,
        "precision": precision,
        "ok": ok,
        "build_plus_run_wall_s": round(wall_s, 1),
        "log": str(log_path),
    }
    if not ok:
        tail = [l for l in (proc.stdout + proc.stderr).splitlines()
                if re.search(r"error|Error|failed|not supported", l)]
        rec["error_tail"] = tail[-6:] or ["(no explicit error line; see log)"]
        return rec

    rec.update(parse_times(times_json) or {})
    if engine.exists():
        rec["engine_size_mb"] = round(engine.stat().st_size / 1e6, 2)
        # The engine is a build artifact, not a deliverable; the plan is the payload.
        engine.unlink()

    m = re.search(r"Throughput: ([\d.]+) qps", proc.stdout)
    if m:
        rec["throughput_qps"] = float(m.group(1))
    # Layers TRT could not run in the requested precision are the op-cliff signal.
    rec["n_fallback_warnings"] = len(re.findall(r"not supported|falling back|Falling back",
                                                proc.stdout + proc.stderr))
    return rec


def host_info(label, trtexec):
    info = {"host_label": label, "machine": platform.machine(), "node": platform.node()}
    v = subprocess.run([trtexec, "--help"], capture_output=True, text=True)
    m = re.search(r"TensorRT.trtexec \[TensorRT v(\d+)\]", v.stdout + v.stderr)
    if m:
        raw = m.group(1)
        info["tensorrt"] = f"{int(raw[:-4] or 0)}.{int(raw[-4:-2])}.{int(raw[-2:])}"
    for cmd, key in (
        (["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], "gpu"),
        (["nvpmodel", "-q"], "power_mode"),
    ):
        if shutil.which(cmd[0]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                info[key] = r.stdout.strip().splitlines()[0] if r.stdout.strip() else None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host-label", required=True)
    ap.add_argument("--precisions", default="fp16,int8")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--duration", type=int, default=10, help="seconds of timed inference")
    ap.add_argument("--warmup-ms", type=int, default=3000,
                    help="long warmup so DVFS has boosted before we time anything")
    ap.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    args = ap.parse_args()

    trtexec = args.trtexec if os.path.exists(args.trtexec) else shutil.which("trtexec")
    if not trtexec:
        sys.exit("trtexec not found; pass --trtexec")

    onnx_dir = Path(args.onnx_dir)
    out_dir = Path(args.out).parent / f"_{args.host_label}_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

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
            rec = run_one(trtexec, onnx_path, precision, out_dir,
                          args.duration, args.warmup_ms)
            if rec["ok"]:
                print(f"    compute p50 {rec['compute_ms']['median']:.3f} ms  "
                      f"({rec.get('throughput_qps', 0):.1f} qps, "
                      f"build+run {rec['build_plus_run_wall_s']}s)", flush=True)
            else:
                print(f"    FAILED: {rec['error_tail'][:2]}", flush=True)
            results.append(rec)
            # Persist after every run: a 90-minute sweep must survive a crash at 80.
            Path(args.out).write_text(json.dumps({
                "__meta__": {
                    "description": "keyhole vision corpus, same ONNX, TensorRT on one host.",
                    "int8_caveat": "PERF-ONLY: no calibration cache, arbitrary dynamic "
                                   "ranges. Latency representative, numerics are NOT.",
                    "compute_ms": "pure GPU kernel time (no H2D/D2H) — the cross-platform number",
                    "latency_ms": "end-to-end incl. host transfers — flatters unified memory",
                    "schema_version": 1,
                },
                "host": host_info(args.host_label, trtexec),
                "config": {"duration_s": args.duration, "warmup_ms": args.warmup_ms,
                           "avg_runs": 100},
                "results": results,
            }, indent=2))
    print(f"\nwrote {args.out}  ({sum(r['ok'] for r in results)}/{len(results)} ok)")


if __name__ == "__main__":
    main()
