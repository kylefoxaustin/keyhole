# Nsight Compute data — what it is, what it means, how to use it

> **Audience:** reviewers, future-Kyle, anyone reading the keyhole-sizer
> platform-budget CSV and wondering where the "measured" columns came
> from. For *how to produce* these JSONs, see [PROFILE_NCU.md](PROFILE_NCU.md);
> this doc is about *interpreting* them.

## tl;dr

Seven JSONs in `data/output/ncu/` contain per-model **DRAM byte counts**,
**tensor-core op counts**, and **instruction counts** measured via
NVIDIA Nsight Compute on an RTX 5090. The DRAM numbers are
hardware-neutral and transfer directly to edge NPU projections. The
compute numbers are 5090-specific but useful as relative comparisons.

## 1. What's in `data/output/ncu/`

| File                       | Workload                                        | Status      |
|----------------------------|-------------------------------------------------|-------------|
| `trt_yolo.json`            | YOLO11s-seg under TRT FP16 / FP8 (INT8 dropped) | ✓ complete  |
| `trt_clip.json`            | CLIP ViT-B/32 visual tower under TRT FP16 / FP8 | ✓ complete  |
| `sam_variants.json`        | SAM3 ref + MobileSAM + 2× EfficientSAM + YOLO   | ✓ complete  |
| `efficientsam3.json`       | EfficientSAM3 ES-EV-S (Option A, stage1)        | ✓ complete  |
| `yoloe26.json`             | YOLOE-26S text-prompt + prompt-free (PyTorch)   | ✓ complete  |
| `trt_yoloe26.json`         | YOLOE-26S-PF under TRT FP16 / FP8               | pending     |
| *(efficientsam3p1 skipped — SIGKILL'd on pass 2, see Caveats §9)*               |

Each JSON has the schema described in
[PROFILE_NCU.md § Output JSON schema](PROFILE_NCU.md#output-json-schema).
The load-bearing part is `by_range[<nvtx_label>].metrics`.

## 2. The metrics glossary

ncu was asked for 7 metrics per kernel, aggregated per NVTX range:

| Metric                                | Plain English                                       |
|---------------------------------------|-----------------------------------------------------|
| `dram__bytes_read.sum`                | Total bytes read from GPU DRAM                      |
| `dram__bytes_write.sum`               | Total bytes written to GPU DRAM                     |
| `dram__bytes.sum`                     | Sum of the above (total DRAM traffic)               |
| `sm__inst_executed_pipe_tensor.sum`   | Tensor Core ops (HMMA/BMMA/IMMA count on Blackwell) |
| `smsp__inst_executed.sum`             | SM-partition (warp-level) instructions executed     |
| `sm__sass_thread_inst_executed.sum`   | Thread-level SASS instructions (FLOP proxy)         |
| `gpu__time_duration.sum`              | Cumulative kernel-on-GPU time (ns, inflated by ncu) |

### Things worth reemphasizing

- **Sums** — over every kernel inside the NVTX range, across the whole
  bake-off run. Divide by `n_forwards` for a per-forward view.
- **`gpu__time_duration.sum` is inflated.** ncu's instrumentation
  serializes kernel launches and flushes counters; the number is
  ~10× the normal wall-clock. Use the non-profiled bake-off JSON for
  real latency.
- **INT8 in TRT** — ncu couldn't profile the integer quantized kernels
  cleanly on the 5090; their counters came back as error-annotation
  rows that we filter out. Every INT8 TRT range is therefore *absent*
  from the JSON. FP16 and FP8 are fine.

## 3. Reading a single JSON — worked example

Excerpt from `trt_yolo.json` (the FP8 TRT range):

```json
{
  "by_range": {
    "yolo_seg_fp8_trt": {
      "metrics": {
        "dram__bytes.sum":                 9.10e+09,
        "dram__bytes_read.sum":            6.04e+09,
        "dram__bytes_write.sum":           3.06e+09,
        "sm__inst_executed_pipe_tensor.sum": 3.65e+08,
        "smsp__inst_executed.sum":         4.36e+09,
        "sm__sass_thread_inst_executed.sum": 1.38e+11,
        "gpu__time_duration.sum":          4.15e+07
      },
      "n_kernel_invocations": 8274
    }
  }
}
```

Reading this:
- 8,274 kernel launches landed in the `yolo_seg_fp8_trt` NVTX range.
- They collectively touched **9.1 GB of DRAM** (6.0 GB read + 3.1 GB
  write) across the whole bake-off.
- Tensor-core ops totalled 365M (Blackwell HMMA count).
- Cumulative kernel time was 41.5 ms (real wall-clock would be lower —
  this is ncu-inflated).

To get a per-forward number, divide each metric by the number of
forward passes that contributed to this range — see § 4.

## 4. Per-forward normalization

Each ncu JSON aggregates across **all forwards of all recipes of all
resolutions** in a bake-off run. To get a single per-forward number you
need to know how many forwards landed in that NVTX bucket. The divisor
depends on the bake-off's structure — read the corresponding
`data/output/bakeoff/<name>_summary.json` to find it.

| Bake-off       | Range labels                                  | Forwards per range (typical) |
|----------------|-----------------------------------------------|------------------------------|
| `trt_yolo`     | `yolo_seg_fp16_trt`, `yolo_seg_fp8_trt`       | 3 res × 10 frames = **30**   |
| `trt_clip`     | `clip_trt`                                    | 3 res × 30 batches = **90**  |
| `sam_variants` | `mobilesam`, `efficientsam_*`, `yolo_seg`     | 14 frames × ~13 boxes = **~180** |
| `efficientsam3`| `efficientsam3_es_ev_s`                       | 3 res × 14 frames = **42**   |
| `yoloe26`      | `yoloe26_text_prompt_s`, `yoloe26_prompt_free_s` | 3 res × 10 frames = **30** |
| `trt_yoloe26`  | `yoloe26_pytorch_fp16/trt_fp16/trt_fp8`       | 3 res × 10 frames = **30**   |

Example: per-forward DRAM for trt_yolo FP8:
```python
import json
d = json.load(open('data/output/ncu/trt_yolo.json'))
bytes_per_fwd = d['by_range']['yolo_seg_fp8_trt']['metrics']['dram__bytes.sum'] / 30
# → 303.4 MB per forward (9.10 GB ÷ 30)
```

Cross-check the divisor against `gpu__time_duration.sum / n_forwards`
vs the non-profiled bake-off's reported per-frame ms — they should
agree within ncu's ~10× inflation.

## 5. Hardware-neutral vs 5090-specific

| Metric family                    | Transfers absolutely? | Use for edge NPU projection?               |
|----------------------------------|-----------------------|--------------------------------------------|
| `dram__bytes_*`                  | **Yes** — workload property, not device | Bandwidth-bound latency floor       |
| `sm__inst_executed_pipe_tensor`  | No — Blackwell-specific | Relative only (compare workloads)         |
| `smsp__inst_executed`            | No — NVIDIA SASS count | Relative only                             |
| `sm__sass_thread_inst_executed`  | No — NVIDIA SASS count | Relative only                             |
| `gpu__time_duration`             | No — literal 5090 wall-clock | Measurement sanity check only        |

**Rule of thumb:** if a sentence uses GB, pull from DRAM bytes. If a
sentence uses TOPS, the compute column is a ballpark, not a guarantee.

## 6. Edge NPU projection recipe

Given `bytes_per_forward` (from ncu) and an NPU tier:

```
bw_bound_ms    = (bytes_per_forward / 1e9) / npu.effective_bandwidth_gbs * 1000
compute_ms     = (flops_per_forward / 1e12) / npu.effective_tops
edge_latency_ms = max(bw_bound_ms, compute_ms)
edge_fps       = 1000 / edge_latency_ms
```

For bandwidth-bound edge workloads (the default for our model sizes and
NPU tiers), the max() collapses to just `bw_bound_ms` — so DRAM bytes
alone give you the answer.

See § **Sizer integration plan** in
[PROFILE_NCU.md](PROFILE_NCU.md#sizer-integration-plan-wip--to-land-after-first-clean-sweep)
for how this wires into `keyhole-sizer/sizer/platform_budget.py`.

## 7. Host CPU utilization — why edge CPUs don't need to be big

Observed during the 2026-04-18 sweep: **~3% overall CPU utilization**
on the 32-core host. One Python thread pegged, all 31 others idle.
This is the Python-GIL + synchronous-CUDA pattern: a single thread
drives the entire pipeline, and nothing parallelizes across cores.

Practical implication for edge SoCs: **a single modern ARM core
(Cortex-A78 class) is enough** to drive the vision + LLM pipeline
alongside the NPU. Don't size for more.

## 8. Cross-workload comparison — headline table

Aggregated across the full bake-off run (totals, not per-forward — see
§ 4 for divisors).

| NVTX range                  | Kernels | DRAM GB | Tensor ops (B) | GPU ms (ncu-inflated) |
|-----------------------------|--------:|--------:|---------------:|----------------------:|
| **CLIP — FP16+FP8 fused**   |         |         |                |                       |
| `clip_trt`                  |   9,996 |   36.41 |          2.406 |               112.0   |
| **YOLO11s-seg — TRT**       |         |         |                |                       |
| `yolo_seg_fp16_trt`         |   8,232 |    9.18 |          0.365 |                40.8   |
| `yolo_seg_fp8_trt`          |   8,274 |    9.10 |          0.365 |                41.5   |
| **SAM 3 mask alternatives** |         |         |                |                       |
| `mobilesam`                 |  36,676 |  113.37 |          0.200 |               334.1   |
| `efficientsam_tiny`         |   6,576 |  264.62 |          0.110 |               318.4   |
| `efficientsam_small`        |   6,576 |  481.03 |          0.112 |               528.5   |
| `yolo_seg` (in sam_variants)|   6,781 |    8.67 |          0.145 |                40.5   |
| `efficientsam3_es_ev_s`     |  24,161 |  321.62 |          6.914 |               621.6   |
| **YOLOE-26 (one-model)**    |         |         |                |                       |
| `yoloe26_text_prompt_s`     |  20,870 |   26.73 |          0.420 |               121.1   |
| `yoloe26_prompt_free_s`     |  20,829 |   30.21 |          0.413 |               125.7   |
| `yoloe26_pytorch_fp16`      |  22,515 |   32.79 |          0.446 |               126.3   |
| `yoloe26_trt_fp16`          |  12,174 |   19.51 |          0.448 |               165.3   |
| `yoloe26_trt_fp8`           |  12,214 |   19.42 |          0.446 |               164.6   |

**Headlines from the table:**

- **EfficientSAM-Small is the bandwidth hog** of the SAM3 alternatives
  (481 GB total — 4× MobileSAM). Its compute is small but DRAM dominates,
  i.e. classic memory-bound vision transformer.
- **YOLO11s-seg under TRT** is the lightest segmentation option at
  ~9 GB DRAM total — confirming why it's the shipping recipe.
- **CLIP under TRT FP8** is twice the DRAM of FP16-equivalent YOLO-seg
  but lands once per second (1 Hz keyframe) so amortizes to negligible
  steady-state load.
- **YOLOE-26 TRT FP8** at 19.4 GB total is **~2× more bandwidth than
  YOLO-seg TRT FP8** for the same prompt-free open-vocab task — the
  cost of one-model open-vocab vs the two-stage YOLO+CLIP pipeline.
- **TRT FP8 vs FP16** on the same model show essentially identical
  DRAM (9.10 vs 9.18 GB on YOLO-seg, 19.42 vs 19.51 on YOLOE-26).
  The FP8 win is throughput (faster matmul) not bandwidth — weights
  are small enough to live in L2 after the first frame, so steady-state
  DRAM is dominated by activation streaming, which is the same width
  in both recipes (input/output tensors are FP16 at the boundaries).

## 9. Caveats — read before citing these numbers

1. **INT8 TRT kernels are missing.** ncu 2026.1 couldn't profile the
   integer-quantized TRT kernels cleanly on Blackwell; our parser
   correctly drops them rather than report zeros. Substitute FP8
   numbers where INT8 would go; the TOPS-budget implication is
   similar (INT8 and FP8 share MAC arrays on Blackwell).
2. **EfficientSAM3.1 (Option B, 106M params) skipped.** Pass 2 of
   app-replay was SIGKILL'd (OOM or external). Use the 429M Option A
   (`efficientsam3.json`) as a same-family surrogate with appropriate
   downscaling.
3. **`sam3_bf16_reference` dropped from sam_variants.** ncu's
   `--app-replay-match name` rejected SAM 3's cross-pass kernel
   variations. Not a regression — SAM 3 is the 0.4 FPS baseline we're
   shipping *away from*, so ncu metrics for it are lower priority.
4. **Replay-mode choice affects fidelity.** Kernel-replay (used for
   TRT targets because NMS breaks app-replay) saves/restores GPU
   state per kernel — the resulting metric sums are what you'd see
   on an unperturbed run. App-replay (used for PyTorch targets) runs
   the whole app multiple times and may drop a small fraction of
   kernels that don't match across passes. Expect ≤5% undercount on
   app-replay targets.
5. **All numbers are RTX 5090 Blackwell measurements.** NPU tier
   projections in the sizer are derived, not measured on silicon.
   Real vendor NPU numbers may differ by ±30%; treat the sizer as a
   well-informed estimate, not procurement-grade spec.

## 10. How to regenerate

```
sudo -E HOME=$HOME KEYHOLE_NCU_KEEP_CSV=1 KEYHOLE_NCU_REPLAY=application \
  /usr/bin/bash /home/kyle/Documents/GitHub/keyhole/scripts/profile_all_ncu.sh

# For TRT+NMS targets (trt_yolo, trt_yoloe26), override to kernel-replay:
sudo -E HOME=$HOME KEYHOLE_NCU_KEEP_CSV=1 KEYHOLE_NCU_REPLAY=kernel \
  /usr/bin/bash /home/kyle/Documents/GitHub/keyhole/scripts/profile_all_ncu.sh \
  trt_yolo trt_yoloe26
```

Expected wall-clock: 4–6 hours for PyTorch targets (app-replay runs
the app N times for N metric groups), 10–80 min per TRT target (kernel
replay scales with kernel count). See [PROFILE_NCU.md](PROFILE_NCU.md)
for prerequisites.
