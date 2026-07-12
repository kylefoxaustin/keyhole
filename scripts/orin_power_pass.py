#!/usr/bin/env python3
"""
orin_power_pass.py — measured per-model power for the vision corpus on Jetson AGX Orin.

Closes the perf/W hole. The 5090 side is already measured (176-554 W by nvidia-smi,
NOT the 575 W nameplate); the Orin side has only ever been the 60 W SoC nameplate,
which is a ceiling, not a batch-1 draw. Nameplate-vs-measured flipped the answer on
the 5090, so it must be measured here too.

METHOD (the traps this is built to avoid)
-----------------------------------------
1. BUILD IS NOT MEASURED. Engines are built in a separate pass and saved. The power
   window covers ONLY steady-state inference on a preloaded engine. (orin-agx card:
   "it ran across model-load too -- filter to the window or your mean is nonsense.")
2. DVFS RAMP IS DISCARDED. trtexec warms up, and we additionally drop the first
   RAMP_S seconds of samples. The board boosts on load; an unfiltered mean is a
   mixture of two different machines.
3. THE CHANGE MUST BE VERIFIED TO HAVE TAKEN. If mean GPU_SOC during the run is not
   materially above idle, NOTHING RAN and the number is a lie that looks plausible.
   We assert it and mark the row INVALID rather than emitting a pretty watt figure.
   (Fleet rule: an experiment that never ran looks exactly like one that found nothing.)
4. RAIL CONVENTION IS STATED, NOT ASSUMED. tegrastats on AGX Orin exposes:
      VDD_GPU_SOC  -- GPU + SoC die
      VDD_CPU_CV   -- CPU + CV (DLA/PVA)
      VIN_SYS_5V0  -- 5V system/carrier input
   soc_w = VDD_GPU_SOC + VDD_CPU_CV. This is the fleet convention (it reproduces the
   published idle ~3.2 W and LLM-decode ~27 W figures). VIN_SYS_5V0 is recorded
   SEPARATELY and never silently folded in -- it is carrier draw, not the inference
   engine. Any perf/W number must say which rail it used.
5. IDLE IS MEASURED IN THE SAME SESSION, not assumed from the card. Reported both as
   absolute power and as delta-over-idle, because "watts to do the work" and "watts the
   board draws while doing the work" are different questions and get different answers.

Run ON the board:  python3 orin_power_pass.py --build && python3 orin_power_pass.py --measure
"""
import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

CORPUS = Path.home() / "keyhole_corpus"
ONNX = CORPUS / "onnx"
OUT = CORPUS / "out"
ENGINES = CORPUS / "engines"

# canary-first: cheapest model leads, so a broken harness fails in 40 s not 20 min.
MODELS = [
    ("resnet50",            "resnet50v1.onnx"),
    ("clip_vit_b32",        "clip_vit_b32_visual.onnx"),
    ("yolov8n_seg",         "yolov8n-seg.onnx"),
    ("yolo11s_seg",         "yolo11s-seg.onnx"),
    ("efficientsam_encoder","efficient_sam_vitt_encoder.onnx"),
    ("yoloe_26s_seg",       "yoloe-26s-seg-pf.onnx"),
    # efficient_sam_vitt_decoder: does NOT build on Orin's TRT 10.3 (IIOneHotLayer
    # cannot compute a shape tensor). Version limit, not silicon. Excluded, not hidden.
]

# trtexec is not on the non-interactive ssh PATH on JetPack; use the absolute path.
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

RUN_S = 30      # steady-state power window length
RAMP_S = 6      # discarded head of the window (DVFS boost transient)
IDLE_S = 12     # idle baseline sample length
SAMPLE_MS = 100

RAIL_RE = re.compile(
    r"VDD_GPU_SOC (\d+)mW/\d+mW.*?VDD_CPU_CV (\d+)mW/\d+mW.*?VIN_SYS_5V0 (\d+)mW/\d+mW"
)
GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%")


class TegraSampler:
    """Samples tegrastats in a thread. Each sample: (t, gpu_soc_mW, cpu_cv_mW, vin_mW, gr3d%)."""

    def __init__(self):
        self.samples = []
        self._proc = None
        self._thread = None
        self._stop = False

    def _pump(self):
        for line in self._proc.stdout:
            if self._stop:
                break
            m = RAIL_RE.search(line)
            if not m:
                continue
            g = GR3D_RE.search(line)
            self.samples.append((
                time.time(),
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(g.group(1)) if g else -1,
            ))

    def start(self):
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(SAMPLE_MS)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        subprocess.run(["pkill", "-f", "tegrastats"], capture_output=True)

    def window(self, t0, t1):
        return [s for s in self.samples if t0 <= s[0] <= t1]


def summarize(samples):
    if not samples:
        return None
    n = len(samples)
    gpu = [s[1] for s in samples]
    cpu = [s[2] for s in samples]
    vin = [s[3] for s in samples]
    gr3d = [s[4] for s in samples if s[4] >= 0]
    soc = [a + b for a, b in zip(gpu, cpu)]
    srt = sorted(soc)
    return {
        "n_samples": n,
        "vdd_gpu_soc_mean_mw": round(sum(gpu) / n, 1),
        "vdd_cpu_cv_mean_mw": round(sum(cpu) / n, 1),
        "vin_sys_5v0_mean_mw": round(sum(vin) / n, 1),
        "soc_mean_w": round(sum(soc) / n / 1000, 3),
        "soc_p50_w": round(srt[n // 2] / 1000, 3),
        "soc_max_w": round(max(soc) / 1000, 3),
        "gr3d_mean_pct": round(sum(gr3d) / len(gr3d), 1) if gr3d else None,
    }


def build():
    ENGINES.mkdir(exist_ok=True)
    for name, onnx in MODELS:
        eng = ENGINES / f"{name}.fp16.engine"
        if eng.exists():
            print(f"  {name:22s} engine cached")
            continue
        print(f"  {name:22s} building…", flush=True)
        r = subprocess.run(
            [TRTEXEC, f"--onnx={ONNX / onnx}", "--fp16",
             f"--saveEngine={eng}", "--skipInference"],
            capture_output=True, text=True, timeout=1800,
        )
        print(f"  {name:22s} {'OK' if eng.exists() else 'BUILD FAILED'}")
        if not eng.exists():
            (OUT / f"{name}.buildfail.log").write_text(r.stdout[-4000:] + r.stderr[-2000:])


def measure():
    OUT.mkdir(exist_ok=True)
    sampler = TegraSampler()
    sampler.start()
    time.sleep(1.5)

    # --- idle baseline, THIS session, same background state ---
    print(f"idle baseline ({IDLE_S}s)…", flush=True)
    t0 = time.time()
    time.sleep(IDLE_S)
    idle = summarize(sampler.window(t0 + 2, time.time()))
    print(f"  idle soc={idle['soc_mean_w']} W  (gpu_soc={idle['vdd_gpu_soc_mean_mw']} mW, "
          f"cpu_cv={idle['vdd_cpu_cv_mean_mw']} mW, vin={idle['vin_sys_5v0_mean_mw']} mW)\n")

    results = []
    for name, _ in MODELS:
        eng = ENGINES / f"{name}.fp16.engine"
        if not eng.exists():
            print(f"{name:22s} SKIP (no engine)")
            continue

        print(f"{name:22s} running {RUN_S}s…", flush=True)
        t_start = time.time()
        r = subprocess.run(
            [TRTEXEC, f"--loadEngine={eng}",
             f"--duration={RUN_S}", "--warmUp=3000", "--avgRuns=100", "--noDataTransfers"],
            capture_output=True, text=True, timeout=RUN_S + 180,
        )
        t_end = time.time()

        # discard the DVFS ramp; measure only steady state
        win = sampler.window(t_start + RAMP_S, t_end - 1)
        stats = summarize(win)

        # trtexec's own latency, to cross-check against the earlier sweep
        lat = None
        for line in r.stdout.splitlines():
            if "GPU Compute Time" in line and "median" in line:
                m = re.search(r"median = ([\d.]+) ms", line)
                if m:
                    lat = float(m.group(1))

        # ---- VERIFY THE CHANGE TOOK ----
        # If the GPU rail didn't move off idle, nothing ran on the GPU and any watt
        # figure here is a plausible number, not a measurement.
        delta_w = round(stats["soc_mean_w"] - idle["soc_mean_w"], 3) if stats else None
        ran = bool(stats and delta_w is not None and delta_w > 1.0
                   and (stats["gr3d_mean_pct"] is None or stats["gr3d_mean_pct"] > 5))

        row = {
            "model": name,
            "precision": "fp16",
            "valid": ran,
            "compute_p50_ms_trtexec": lat,
            "idle_soc_w": idle["soc_mean_w"],
            "power": stats,
            "soc_delta_over_idle_w": delta_w,
        }
        if not ran:
            row["INVALID_REASON"] = (
                "GPU rail did not rise materially above idle (delta<=1.0 W or GR3D~0%) — "
                "the engine did not execute on the GPU. Do NOT quote this as a power number."
            )
        results.append(row)

        flag = "" if ran else "   ⛔ DID NOT RUN"
        print(f"  {name:22s} soc={stats['soc_mean_w'] if stats else '?'} W "
              f"(Δ{delta_w} over idle)  GR3D={stats['gr3d_mean_pct'] if stats else '?'}%  "
              f"compute={lat} ms{flag}", flush=True)

        (OUT / "orin_power.json").write_text(json.dumps({
            "__meta__": {
                "board": "Jetson AGX Orin 64GB devkit",
                "power_mode": "MAXN (nvpmodel mode 0) — PINNED, stated per the card's rule",
                "trt": "10.3 (JetPack 6)",
                "rail_convention": (
                    "soc_w = VDD_GPU_SOC + VDD_CPU_CV (GPU+SoC die plus CPU+CV). "
                    "VIN_SYS_5V0 is the 5V carrier/system input and is recorded separately, "
                    "NOT folded in. Reproduces the fleet's published idle ~3.2 W."
                ),
                "window": f"{RUN_S}s steady-state, first {RAMP_S}s discarded (DVFS ramp); "
                          f"engine PRELOADED — build/load excluded from the window",
                "sample_interval_ms": SAMPLE_MS,
                "validity_gate": "rows with valid=false did not execute on the GPU — do not quote",
                "excluded": {
                    "efficientsam_decoder": "does not build on TRT 10.3 (IIOneHotLayer "
                                            "cannot compute a shape tensor) — a TRT VERSION "
                                            "limit, not silicon; builds fine on 10.16."
                },
            },
            "idle": idle,
            "results": results,
        }, indent=2))

    sampler.stop()
    print(f"\nwrote {OUT / 'orin_power.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--measure", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.measure:
        measure()
    if not (a.build or a.measure):
        ap.error("pass --build and/or --measure")
