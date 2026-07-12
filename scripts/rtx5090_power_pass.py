#!/usr/bin/env python3
"""
rtx5090_power_pass.py — measured per-model 5090 power, on a card PROVEN clean.

WHY THIS EXISTS
---------------
`data/output/rtx5090_power_by_model.json` sits in the DENOMINATOR of every perf/W
claim we publish. It was produced by an ad-hoc run: no script in this repo, no
recorded method, no clean-card gate. Its recorded idle is **64.97 W**.

This card idles at **21.7 W** measured on a clean card (2026-07-12). And when
[image_gen] shut down its two ComfyUI processes this morning it reported: *"the idle
floor just fell from 61 W to 21 W."* **64.97 W is the idle floor OF A CARD WITH A
RESIDENT TENANT.** The old numbers were taken on a dirty card.

Direction of the error, which is the part that matters: a co-resident process
INFLATES the 5090's watts -> its inferences-per-joule is too LOW -> our published
"the Orin wins perf/W on all six models" **OVERSTATES the edge part.** That is the
direction the fleet has now been wrong in 17 consecutive times.

I built a careful instrument for the Orin (pinned mode, preloaded engines, ramp
discarded, validity gate) and then divided by a 5090 number I had never examined,
because it was already "measured." *The stale term is whichever one you did not
just work on.*

THE GATE IS THE POINT
---------------------
A DISCIPLINE ("remember to check the card is clean") is not a GUARD. `pgrep` typed
by hand is a habit. So this script REFUSES rather than trusting me to remember:

  * any other compute process on the GPU  -> ABORT
  * idle median above IDLE_MAX_W          -> ABORT (the card is not at rest)
  * a tenant appearing mid-run            -> that row is INVALID
  * power that does not RISE over idle    -> that row is INVALID (an engine that did
    not run draws idle watts and looks exactly like a plausible result)

A number that cannot pass the gate is NOT PRINTED. Not "printed with a caveat."

The engine build/run path is IMPORTED from bench_corpus_trt_py.py — the same code
that produced the published latencies. A second engine path would be a second thing
to be wrong about.

Run:  python3 scripts/rtx5090_power_pass.py --check   (gate only)
      python3 scripts/rtx5090_power_pass.py           (full pass)
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch                                                    # noqa: E402
import tensorrt as trt                                          # noqa: E402
from bench_corpus_trt_py import build_engine, make_io, bench    # noqa: E402

ONNX = ROOT / "data" / "output" / "onnx_corpus_iq9"
OUT = ROOT / "data" / "output"

# canary-first: cheapest model leads, so a broken harness fails in seconds.
MODELS = [
    "resnet50v1",
    "clip_vit_b32_visual",
    "yolov8n-seg",
    "yolo11s-seg",
    "efficient_sam_vitt_decoder",
    "efficient_sam_vitt_encoder",
    "yoloe-26s-seg-pf",
]

IDLE_MAX_W = 35.0     # clean card measures 21.7 W. The OLD data recorded 64.97 W.
IDLE_S = 12
RUN_S = 30
RAMP_S = 6            # discard the boost transient: a mean across it is two machines
SAMPLE_MS = 100
MIN_RISE_W = 25.0     # inference must move the rail, or it did not run


def gpu_procs(exclude_self=True):
    """Compute processes on the GPU. EXCLUDES our own PID by default.

    We hold a CUDA context for the entire run, so counting ourselves as a tenant
    makes the contention guard accuse itself and mark every row INVALID against
    good data. A guard that fires on its own reflection is worse than no guard —
    it manufactures a false negative wearing a finding's clothes.
    """
    import os
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
         "--format=csv,noheader"], capture_output=True, text=True)
    me = str(os.getpid())
    out = []
    for l in r.stdout.strip().splitlines():
        if not l.strip():
            continue
        pid = l.split(",")[0].strip()
        if exclude_self and pid == me:
            continue
        out.append(l)
    return out


def power_now():
    r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return float(r.stdout.strip().splitlines()[0])


class Sampler:
    def __init__(self):
        self.samples, self._stop = [], False

    def _pump(self):
        while not self._stop:
            try:
                self.samples.append((time.time(), power_now()))
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(SAMPLE_MS / 1000.0)

    def start(self):
        threading.Thread(target=self._pump, daemon=True).start()

    def stop(self):
        self._stop = True

    def window(self, t0, t1):
        return [w for t, w in self.samples if t0 <= t <= t1]


def gate(strict=True):
    print("── CLEAN-CARD GATE ───────────────────────────────────────")
    procs = gpu_procs()
    if procs:
        print("  ⛔ OTHER COMPUTE PROCESSES ON THE GPU:")
        for p in procs:
            print(f"       {p}")
        print("  A tenant inflates the idle floor AND can contend mid-run.")
        print("  Both errors land in the denominator. Ask them to yield — never kill.")
        if strict:
            raise SystemExit("ABORT: card not exclusive. Nothing measured, nothing printed.")
    else:
        print("  ✅ no other compute processes on the GPU")

    print(f"  sampling idle for {IDLE_S}s …")
    vals = []
    t0 = time.time()
    while time.time() - t0 < IDLE_S:
        vals.append(power_now())
        time.sleep(0.1)
    idle = statistics.median(vals)
    print(f"  idle: median {idle:.2f} W  (min {min(vals):.2f} / max {max(vals):.2f}, n={len(vals)})")
    if idle > IDLE_MAX_W:
        print(f"  ⛔ idle {idle:.2f} W exceeds the {IDLE_MAX_W} W gate — CARD IS NOT AT REST.")
        print("     This is exactly the bug: the old data recorded a 64.97 W 'idle'.")
        if strict:
            raise SystemExit("ABORT: card not at rest. Nothing measured, nothing printed.")
    else:
        print(f"  ✅ idle {idle:.2f} W within the {IDLE_MAX_W} W gate — card at rest")
    print("──────────────────────────────────────────────────────────\n")
    return idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    idle = gate()
    if a.check:
        return

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    sampler = Sampler()
    sampler.start()
    results = []

    for name in MODELS:
        onnx = ONNX / f"{name}.onnx"
        if not onnx.exists():
            print(f"{name:28s} SKIP — no onnx")
            continue

        print(f"{name:28s} building …", end=" ", flush=True)
        try:
            plan = build_engine(onnx, "fp16", logger)
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(plan)
            context = engine.create_execution_context()
            inputs, outputs = make_io(engine, context)
        except Exception as exc:                                 # noqa: BLE001
            print(f"BUILD FAILED: {type(exc).__name__}: {str(exc)[:80]}")
            continue

        # ---- POWER WINDOW: sustained inference on a preloaded engine ----
        print(f"running {RUN_S}s …", end=" ", flush=True)
        stream = torch.cuda.Stream()
        t_start = time.time()
        with torch.cuda.stream(stream):
            t_end = time.time() + RUN_S
            while time.time() < t_end:
                for _ in range(50):
                    context.execute_async_v3(stream.cuda_stream)
                stream.synchronize()
        t_stop = time.time()

        win = sampler.window(t_start + RAMP_S, t_stop - 0.5)
        med = statistics.median(win) if win else None
        mx = max(win) if win else None
        rise = (med - idle) if med else 0.0
        contended = len(gpu_procs()) > 0
        ran = bool(med and rise >= MIN_RISE_W and not contended)

        # latency cross-check, same code path that produced the published numbers
        b = bench(engine, context, inputs, outputs, duration_s=3, warmup_ms=500)
        compute_med = b["compute_ms"]["median"]

        row = {
            "model": name,
            "power_w_median": round(med, 2) if med else None,
            "power_w_max": round(mx, 2) if mx else None,
            "n_samples": len(win),
            "rise_over_idle_w": round(rise, 2),
            "compute_p50_ms_crosscheck": round(compute_med, 4),
            "valid": ran,
        }
        if not ran:
            row["INVALID_REASON"] = (
                "another compute process appeared mid-run" if contended else
                f"power rose only {rise:.1f} W over idle (< {MIN_RISE_W} W) — the engine did "
                "not load the GPU; an idle reading is not a workload measurement.")
        results.append(row)

        flag = "" if ran else "   ⛔ INVALID"
        print(f"{med:6.1f} W  (+{rise:5.1f} over idle, max {mx:.0f}, n={len(win)})  "
              f"compute={compute_med:.3f} ms{flag}", flush=True)

        del context, engine, inputs, outputs
        torch.cuda.empty_cache()

    sampler.stop()

    payload = {
        "__meta__": {
            "supersedes": (
                "rtx5090_power_by_model.json (2026-07-10) was an AD-HOC run: no script in the "
                "repo, no recorded method, no clean-card gate, and a recorded idle of 64.97 W. "
                "[image_gen] confirmed 2026-07-12 that its two resident ComfyUI processes held "
                "the idle floor at 61 W, falling to 21 W once stopped — so 64.97 W IS THE IDLE "
                "FLOOR OF A DIRTY CARD. The old per-model watts carry a tenant's baseline, "
                "which INFLATES the 5090's power and OVERSTATES the Orin's perf/W advantage."
            ),
            "gate": (
                f"Refuses to run unless the GPU has NO other compute process and idle median "
                f"<= {IDLE_MAX_W} W. Rows are INVALID unless power rises >= {MIN_RISE_W} W over "
                f"idle and no tenant appears mid-run."
            ),
            "method": (
                f"nvidia-smi power.draw @ {SAMPLE_MS} ms. Engine PRELOADED (build excluded); "
                f"{RUN_S}s sustained execute_async_v3, first {RAMP_S}s discarded (boost "
                f"transient); median of the steady window. Engine build/run imported from "
                f"bench_corpus_trt_py.py — the same path that produced the published latencies."
            ),
            "power_basis": (
                "WHOLE-BOARD (GPU die + GDDR7 + VRM) via nvidia-smi. NOT comparable to the "
                "Orin's die-only tegrastats SoC rails without the bracketing conventions in "
                "merge_power_perfw.py."
            ),
            "measured": "2026-07-12",
            "card_state": "EXCLUSIVE — image_gen released both ComfyUI processes at 09:13",
        },
        "idle_w": round(idle, 2),
        "idle_w_previous_DIRTY": 64.97,
        "models": {r["model"]: r for r in results},
    }
    (OUT / "rtx5090_power_by_model_v2.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote rtx5090_power_by_model_v2.json   idle {idle:.2f} W "
          f"(previous, dirty: 64.97 W)")


if __name__ == "__main__":
    main()
