#!/usr/bin/env python3
"""
merge_corpus_platforms.py — fold the per-host corpus sweeps into one anchor.

Consumes the per-host JSONs written by bench_corpus_trt.py (Orin, trtexec) and
bench_corpus_trt_py.py (5090, TensorRT Python API), and emits the cross-platform
anchor that the sizers and qualcomm's IQ-9075 run diff against.

The iq9 column is left as a declared hole rather than a projection. An empty cell
is honest; an extrapolated cell would get quoted back to us as a measurement.

Run: python3 scripts/merge_corpus_platforms.py \
        --host data/output/rtx5090_corpus_trt.json \
        --host data/output/orin_corpus_trt.json \
        --out  data/output/vision_corpus_three_platform.json
"""
import argparse
import json
from pathlib import Path

# Nameplate power CEILINGS. These are what a part is permitted to draw, not what it
# does draw running a batch-1 graph — and the gap is not small: the 5090's measured
# draw across this corpus spans 176-554 W against a 575 W TGP. Dividing throughput by
# nameplate silently hands the low-TDP part an efficiency win it has not earned.
# Used ONLY as an upper bound where a host has no measured power.
NAMEPLATE_CEILING_W = {
    "rtx-5090": 575.0,       # OFFICIAL TGP
    "orin-agx-64gb": 60.0,   # MAXN power-mode ceiling (nvpmodel 0), whole SoC incl. CPU
    "iq9075": 40.0,          # EST top of the ~15-40W window
}


def load_measured_power(path):
    """{model: median watts} sampled under sustained load. Absent -> {}."""
    if not path or not Path(path).exists():
        return {}
    d = json.loads(Path(path).read_text())
    return {k: v["power_w_median"] for k, v in d["models"].items()}


def load(paths):
    hosts = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        label = d["host"]["host_label"]
        hosts[label] = d
    return hosts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="append", required=True)
    ap.add_argument("--power-5090", default="data/output/rtx5090_power_by_model.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hosts = load(args.host)
    measured_power = {"rtx-5090": load_measured_power(args.power_5090)}
    models, precisions = [], []
    for d in hosts.values():
        for r in d["results"]:
            if r["model"] not in models:
                models.append(r["model"])
            if r["precision"] not in precisions:
                precisions.append(r["precision"])

    matrix = {}
    for model in models:
        matrix[model] = {}
        for precision in precisions:
            cell = {}
            for label, d in hosts.items():
                rec = next((r for r in d["results"]
                            if r["model"] == model and r["precision"] == precision), None)
                if rec is None:
                    continue
                if not rec["ok"]:
                    cell[label] = {"status": "FAILED", "error": rec.get("error_tail")}
                    continue
                p50 = rec["compute_ms"]["median"]
                fps = 1000.0 / p50
                entry = {
                    "status": "MEASURED",
                    "compute_p50_ms": round(p50, 4),
                    "compute_p99_ms": round(rec["compute_ms"]["p99"], 4),
                    "e2e_p50_ms": round(rec["latency_ms"]["median"], 4),
                    "fps": round(fps, 1),
                    "engine_size_mb": rec.get("engine_size_mb"),
                }
                watts = measured_power.get(label, {}).get(model)
                if watts:
                    entry["power_w_measured"] = round(watts, 1)
                    entry["inferences_per_joule"] = round(fps / watts, 2)
                    entry["perf_per_watt_basis"] = "MEASURED power under sustained load"
                elif NAMEPLATE_CEILING_W.get(label):
                    ceiling = NAMEPLATE_CEILING_W[label]
                    entry["power_w_nameplate_ceiling"] = ceiling
                    # True draw is at or below the ceiling, so fps/ceiling is a FLOOR on
                    # efficiency — never an estimate of it.
                    entry["inferences_per_joule_lower_bound"] = round(fps / ceiling, 2)
                    entry["perf_per_watt_basis"] = ("UNMEASURED — nameplate ceiling only; "
                                                    "this is a lower bound, not a value")
                cell[label] = entry
            cell["iq9075"] = {"status": "PENDING — qualcomm QNN/HTP, same ONNX"}

            # Cross-platform ratios only where both sides actually measured.
            a = cell.get("rtx-5090", {})
            b = cell.get("orin-agx-64gb", {})
            if a.get("status") == "MEASURED" and b.get("status") == "MEASURED":
                cell["ratios"] = {
                    "5090_faster_than_orin_x": round(
                        b["compute_p50_ms"] / a["compute_p50_ms"], 2),
                }
                # Perf/W needs BOTH sides measured. With the 5090 measured and the Orin
                # only bounded, every ratio we could write down is either unsupported or
                # a bound so loose it invites misquotation. State the hole instead.
                if "inferences_per_joule" in a and "inferences_per_joule" not in b:
                    cell["ratios"]["perf_per_watt"] = (
                        "INCONCLUSIVE — 5090 power measured, Orin power NOT measured. "
                        "Orin's batch-1 draw is well under its 60 W SoC ceiling, so the "
                        "sign of this comparison is undetermined. Needs a tegrastats pass.")
            matrix[model][precision] = cell

    # INT8 speedup over FP16, per host — the "is INT8 a real compute win here" question.
    int8_gain = {}
    for model in models:
        row = {}
        for label in hosts:
            f = matrix[model].get("fp16", {}).get(label, {})
            i = matrix[model].get("int8", {}).get(label, {})
            if f.get("status") == "MEASURED" and i.get("status") == "MEASURED":
                row[label] = round(f["compute_p50_ms"] / i["compute_p50_ms"], 3)
        if row:
            int8_gain[model] = row

    out = {
        "__meta__": {
            "description": "Three-platform vision-corpus anchor: identical ONNX, identical "
                           "measurement protocol, one row per model x precision.",
            "headline_metric": "compute_p50_ms — pure accelerator kernel time, no host transfers.",
            "int8_caveat": "PERF-ONLY (no calibration cache; trtexec dynamic-range convention). "
                           "Latency representative, numerics are NOT. Never quote accuracy from these.",
            "harness_note": "Orin measured with trtexec; 5090 with the TensorRT Python API "
                            "(no trtexec binary on that host). Semantics matched: same warmup, "
                            "same duration, CUDA events isolating compute from transfers, same "
                            "int8 dynamic-range constants. The one Orin build failure was "
                            "reproduced through BOTH harnesses, so it is a property of the "
                            "graph+TensorRT version, not of trtexec.",
            "power_note": "5090 power is MEASURED per model under sustained load (nvidia-smi, "
                          "0.1s sampling, median): 176-554 W against a 575 W nameplate TGP. "
                          "Orin power is NOT measured — only its 60 W SoC ceiling is known. "
                          "Any perf/W comparison between them is therefore INCONCLUSIVE, and "
                          "a nameplate-TDP-derived one would overstate the Orin's advantage by "
                          "up to 3.3x. Do not quote perf/W from this file.",
            "nameplate_ceiling_w": NAMEPLATE_CEILING_W,
            "schema_version": 2,
        },
        "hosts": {label: d["host"] | {"config": d["config"]} for label, d in hosts.items()},
        "int8_is_a_toolchain_artifact_not_silicon": {
            "verdict": "The INT8 column measures TensorRT's implicit-quantization path, "
                       "NOT INT8 silicon throughput. Do not compare it against a "
                       "calibrated INT8 result (e.g. QNN/HTP on the IQ-9075).",
            "evidence": [
                "TRT 10 implicit quantization (set_dynamic_range, deprecated in 10.1) with "
                "no calibration cache does not select INT8 kernels for most of this corpus; "
                "it inserts reformat / data-movement layers at every precision boundary.",
                "yolov8n-seg 193 -> 272 layers under --int8, and 0.431 -> 0.567 ms (SLOWER).",
                "efficient_sam_vitt_encoder 94 -> 139 layers, 1.343 -> 2.656 ms (2x SLOWER).",
                "clip_vit_b32_visual 103 -> 105 layers, latency identical to fp16 on BOTH "
                "hosts (5090 0.486->0.487, Orin 1.834->1.830) — i.e. nothing ran in INT8.",
                "Only resnet50v1 (pure conv, fused INT8 conv tactics exist) shows a real "
                "gain: Orin 1.037 -> 0.742 ms (1.40x).",
            ],
            "fix": "Emit QDQ-annotated ONNX from a real calibration set (explicit "
                   "quantization) and rebuild. Then INT8 latency AND numerics are both "
                   "meaningful, and cross-platform INT8 comparison becomes legitimate.",
        },
        "build_failures": {
            "conv_silu_no_int8_tactic": {
                "models": ["yolov8n-seg (Orin only)", "yolo11s-seg (both hosts)",
                           "yoloe-26s-seg-pf (both hosts)"],
                "error": "Could not find any implementation for node "
                         "<...>/conv/Conv + PWN(PWN(Sigmoid), PWN(Mul))",
                "reading": "A Conv fused with SiLU has no INT8 tactic in this TensorRT "
                           "under implicit quantization. Same root cause as the artifact "
                           "above; expected to disappear with QDQ.",
            },
            "efficientsam_decoder_onehot": {
                "models": ["efficient_sam_vitt_decoder (Orin only, fp16 AND int8)"],
                "error": "node /mask_decoder/Tile_1 (Tile): "
                         "an IIOneHotLayer cannot be used to compute a shape tensor",
                "reading": "TensorRT VERSION limitation, not silicon and not the harness. "
                           "Fails at ONNX parse on TRT 10.3 (JetPack 6) through BOTH trtexec "
                           "and the Python API; the identical graph parses and builds on TRT "
                           "10.16 (5090). Deployment fact: EfficientSAM's mask decoder does "
                           "not build on JetPack 6's stock TensorRT.",
                "control": "yolov8n-seg fp16 built successfully via the same Python-API path "
                           "on the same box, so the failure is the graph, not the script.",
            },
        },
        "int8_speedup_over_fp16": int8_gain,
        "matrix": matrix,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}: {len(models)} models x {len(precisions)} precisions "
          f"x {len(hosts)} measured hosts")


if __name__ == "__main__":
    main()
