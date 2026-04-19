# Nsight Compute profiling — per-model platform-budget breakdown

This doc covers the `ncu`-based profiling path that produces MEASURED
(instead of APPROXIMATED) instruction-count, FLOP, and DRAM-byte numbers
per workload model. Downstream consumer: the platform-budget spreadsheet
and the `keyhole-sizer` CSV export, which can swap the approximated
`ss_tops_avg` / `ss_ddr_gbs_avg` columns for measured values.

## Why this exists

The normal Keyhole bake-offs measure **wall-clock ms per frame** using
`torch.cuda.synchronize()` + `time.perf_counter()`. That's great for
latency, but for a platform-level compute/DDR budget the spreadsheet
really wants **instructions executed**, **FLOPs**, and **DRAM bytes**
per forward pass. Those come from hardware performance counters, which
you get via NVIDIA Nsight Compute (`ncu`).

## How the numbers get attributed to YOLO vs SAM3 vs LLM

The bake-off scripts push NVTX ranges around each model's forward pass
(via `src.profiling.nvtx_helpers.nvtx_range("name")`). When run under
`ncu --nvtx`, every CUDA kernel that executes inside the range gets
tagged with the range's name. The wrapper parses the ncu CSV output and
sums metrics per range — so you get `yolo_seg_fp8_trt`, `sam3_bf16_ref`,
`efficientsam3_es_ev_s`, `clip_trt`, `llm_prefill_n2048`,
`llm_decode_n256` each as a separate row in the output JSON.

## Prerequisites

1. **Nsight Compute installed** (ships with CUDA toolkit). Verify:
   ```bash
   ncu --version
   ```
   Default location on this box: `/usr/local/cuda-12.6/bin/ncu`.
2. **No other CUDA process holding the GPU.** `ncu` needs exclusive
   access, and in particular a running `python3 llm_server.py` will
   block it. Check `nvidia-smi` before running.
3. **Normal bake-offs have run once.** Profiling reuses the cached
   frames / prompts / refs / TRT engines under `data/output/bakeoff/`.
   It doesn't re-extract frames.
4. **The EfficientSAM3 variants use `.venv-es3/`** (Python 3.12). The
   shell script handles both venvs — no manual switching needed.

## Quick start — profile everything

```bash
bash scripts/profile_all_ncu.sh
```

Runtime: ~30-60 min for the full sweep. Outputs to
`data/output/ncu/*.json` (one file per bake-off).

## Single-target run

```bash
python scripts/profile_ncu.py \
    --out data/output/ncu/trt_yolo.json \
    -- \
    python scripts/bakeoff_trt_yolo.py --clip data/videos/720p_EW_clip.mp4
```

Everything after the `--` is the command being profiled, verbatim.

## Filtering to a specific NVTX range

Useful when you want to profile only the LLM decode path (not prefill)
or only the YOLO-seg FP8 recipe:

```bash
python scripts/profile_ncu.py \
    --out data/output/ncu/llm_decode_only.json \
    --nvtx-include 'regex:llm_decode' \
    -- \
    python scripts/bakeoff_llm.py --quants Q4_K_M

python scripts/profile_ncu.py \
    --out data/output/ncu/yolo_fp8_only.json \
    --nvtx-include 'regex:yolo_seg_fp8_trt' \
    -- \
    python scripts/bakeoff_trt_yolo.py --clip data/videos/720p_EW_clip.mp4
```

`--nvtx-include` syntax is what `ncu --nvtx-include` accepts (see
`ncu --help | grep nvtx-include`).

## Output JSON schema

```json
{
  "profiler": "Nsight Compute (ncu)",
  "ncu_binary": "/usr/local/cuda-12.6/bin/ncu",
  "command": ["python", "scripts/bakeoff_trt_yolo.py", "..."],
  "nvtx_include_filter": null,
  "metrics_requested": ["smsp__inst_executed.sum", "...", "dram__bytes.sum"],
  "export_timestamp_iso": "2026-04-17T...",
  "by_range": {
    "yolo_seg_fp8_trt": {
      "metrics": {
        "smsp__inst_executed.sum": 123456789.0,
        "sm__inst_executed_pipe_tensor.sum": 12345.0,
        "dram__bytes_read.sum":  9876543.0,
        "dram__bytes_write.sum": 1234567.0,
        "dram__bytes.sum":       11111110.0,
        "gpu__time_duration.sum": 1234567.0
      },
      "units": { "dram__bytes.sum": "byte", "...": "..." },
      "n_kernel_invocations": 42
    },
    "yolo_seg_fp16_trt": { ... },
    ...
  }
}
```

Divide the sums by (frames_processed × n_forward_per_frame) to get
per-forward averages. Wall-clock comes from the normal (non-profiled)
bake-off JSON, not from `gpu__time_duration.sum` under ncu — the latter
includes profiling overhead.

## Metric set

The wrapper requests a targeted (not `--set detailed`) list to keep
runtime to ~2-5× the normal bake-off, not 50×:

| Metric                                   | What it measures                      |
|------------------------------------------|---------------------------------------|
| `smsp__inst_executed.sum`                | Total SM-partition instructions       |
| `sm__sass_thread_inst_executed.sum`      | Thread-level SASS (FLOP proxy)        |
| `sm__inst_executed_pipe_tensor.sum`      | Tensor Core op count                  |
| `dram__bytes_read.sum`                   | DRAM reads (bytes)                    |
| `dram__bytes_write.sum`                  | DRAM writes (bytes)                   |
| `dram__bytes.sum`                        | Total DRAM traffic                    |
| `gpu__time_duration.sum`                 | Cumulative kernel GPU time (inflated) |

Add more in `scripts/profile_ncu.py::METRICS` if the spreadsheet needs
cache-miss rate, warp efficiency, per-precision FLOPs (`*_op_hmma_*`,
`*_op_bfma_*`, `*_op_fmma_*`), etc.

## NVTX range names — current inventory

| Bake-off script                         | NVTX range(s) pushed                                                   |
|-----------------------------------------|-------------------------------------------------------------------------|
| `bakeoff_trt_yolo.py`                   | `yolo_seg_fp16_trt`, `yolo_seg_int8_trt`, `yolo_seg_fp8_trt`          |
| `bakeoff_sam_variants.py`               | `sam3_bf16_reference`, `mobilesam`, `efficientsam_tiny`, `efficientsam_small`, `yolo_seg` |
| `bakeoff_efficientsam3.py`              | `efficientsam3_es_ev_s`                                                 |
| `bakeoff_efficientsam3p1.py`            | `efficientsam3p1_es_ev_s__set_image`, `efficientsam3p1_es_ev_s__text_prompt` |
| `bakeoff_trt_clip.py`                   | `clip_trt`                                                              |
| `bakeoff_yoloe26.py`                    | `yoloe26_text_prompt_s`, `yoloe26_prompt_free_s`                       |
| `bakeoff_trt_yoloe26.py`                | `yoloe26_pytorch_fp16`, `yoloe26_trt_fp16`, `yoloe26_trt_fp8`          |
| `bakeoff_llm.py`                        | `llm_prefill_n{N}`, `llm_decode_n{N}` (per-config N values)            |

Adding a new range is a one-liner:

```python
from src.profiling.nvtx_helpers import nvtx_range
with nvtx_range("my_new_stage"):
    model.forward(...)
```

No harm if run without a profiler — the context manager is a no-op.

## TRT engines & profiling verbosity

Engines built by `bakeoff_trt_yolo.py` and `bakeoff_trt_clip.py` now set
`config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED` so the
ONNX-node names survive into the TRT engine. This makes the ncu output
much more readable: instead of `generated_kernel_<hash>` you get
`/backbone/stem/Conv_output` etc., attributed to the right NVTX range.

If you have an already-built engine from before this change, delete the
engine file (`data/trt_engines/<name>.*.engine`) and re-run the
bake-off — it'll rebuild with the new profiling verbosity.

## Host CPU utilization

Observed during the 2026-04-18 full sweep: one Python thread pegged at
~98% CPU, 31 other cores idle — overall **~3% system CPU utilization**
on a 32-core host. This is inherent to the pipeline, not a ncu artifact:
Python GIL + synchronous CUDA per thread means a single thread drives
the GPU, and the rest of the CPU sits unused. Multi-core CPU work is
limited to ffmpeg frame decode (brief) and TRT's tiny kernel-dispatch
thread pool.

**Edge takeaway:** the companion NPU doesn't need a beefy host CPU. A
single Cortex-A78-class core is enough to drive the whole vision +
(optional) LLM pipeline. All the parallelism lives on the NPU.

## Transferability to edge NPUs

The collected metrics split into two classes by how they generalize:

**Hardware-neutral — use as absolute edge projections**
- `dram__bytes_read/write/sum` — **property of the workload, not the
  device.** A CLIP ViT-B/32 forward that reads ~180 MB of weights on
  the 5090 will read approximately the same on an edge NPU (same
  model, same quant). This is the gold column: feeds the sizer's
  bandwidth-bound minimum latency: `latency_ms ≥ dram_bytes / npu_gbps`.

**5090-specific — use as relative comparisons or sanity checks only**
- `sm__inst_executed_pipe_tensor.sum` — Blackwell HMMA/BMMA/IMMA op
  count. Edge NPUs have totally different matrix engines (FP8/INT8
  MAC arrays, different widths). Use as *ratio between workloads*
  rather than as absolutes.
- `smsp__inst_executed.sum`, `sm__sass_thread_inst_executed.sum` —
  NVIDIA SASS instruction counts. Same caveat: relative only.
- `gpu__time_duration.sum` — literally 5090 wall-clock. Useless for
  edge projection, good as a measurement sanity baseline.

**How the sizer applies this:**
1. Take `dram__bytes.sum` per inference → bandwidth-bound floor.
2. Compute compute-bound floor from model FLOPs / NPU TOPS spec.
3. Real edge latency ≈ max(bandwidth-bound, compute-bound).
4. Compare against vendor-reported NPU benchmarks for cross-check.

## Sizer integration plan (WIP — to land after first clean sweep)

`keyhole-sizer/sizer/platform_budget.py` currently approximates the
additive columns:

```python
# Current (approximate) values in vision_workload_row / llm_workload_row:
ss_ddr_gbs_avg = hw.effective_bandwidth_gbs * duty_cycle_frac   # saturation
ss_tops_avg    = (hw.peak_tops_fp8 * hw.compute_efficiency) * duty_cycle_frac
```

This assumes every workload saturates the bus during its duty cycle.
Good enough for BW-bound edge models, but imprecise (e.g., YOLO-seg at
36 FPS may only pull 40 GB/s on NPU Mid, not all 100 GB/s).

**Replacement using ncu JSONs:**

1. **Map NVTX range → sizer pipeline stage.** Static dict in
   `sizer/ncu_lookup.py`:
   ```python
   NCU_RANGE_TO_PIPELINE = {
       "yolo_seg_fp8_trt":          ("hybrid_v2_fp8",  "detect"),
       "clip_trt":                  ("hybrid_v2_fp8",  "enrich"),
       "yoloe26_prompt_free_s":     ("yoloe26_pf",     "detect_enrich"),
       "efficientsam3_es_ev_s":     ("efficientsam3",  "sam_alt"),
       # etc.
   }
   ```

2. **Normalize per-forward.** Each ncu JSON reports *summed* metrics
   over the bake-off run. Divide by `n_frames × n_recipes_per_frame`
   (from the bake-off's own summary JSON) to get per-forward numbers:
   ```python
   bytes_per_forward = ncu["by_range"][name]["metrics"]["dram__bytes.sum"] / n_forwards
   tc_ops_per_forward = ncu["by_range"][name]["metrics"]["sm__inst_executed_pipe_tensor.sum"] / n_forwards
   ```

3. **Edge projection (bandwidth-bound path).** DRAM bytes are
   hardware-neutral — use them directly against NPU tier bandwidth:
   ```python
   min_latency_ms = (bytes_per_forward / 1e9) / hw.effective_bandwidth_gbs * 1000
   ss_ddr_gbs_avg_measured = bytes_per_forward * fps / 1e9   # GB/s at target rate
   ```

4. **Edge projection (compute-bound path) — approximate.** Blackwell
   HMMA ops don't map 1:1 to NPU MAC ops, but a precision-aware TOPS
   ratio works as a first-order estimate:
   ```python
   tops_per_forward = tc_ops_per_forward * 2 / 1e12   # HMMA 16×16×16 = 256 MAC ≈ 512 ops
   ss_tops_avg_measured = tops_per_forward * fps
   ```

5. **Schema addition.** Extend CSV_COLUMNS in `platform_budget.py`:
   - `ss_ddr_gbs_avg_measured` (replaces/augments the approximate one)
   - `ss_tops_avg_measured`
   - `ncu_source_file` (provenance: which trt_yolo.json fed this row)
   - `ncu_n_forwards` (how many forward passes were averaged)

6. **Fallback.** If the ncu JSON is missing or the NVTX range isn't
   mapped, fall back to the current approximation and set
   `ncu_source_file = "approximated"`. Lets the spreadsheet progress
   incrementally as we profile more targets.

**Bridge script (DONE):** `scripts/export_ncu_for_sizer.py` reads the
6 ncu JSONs and emits `data/output/ncu/sizer_bundle.json` — one entry
per NVTX range with totals, per-forward normalization (using documented
divisors), and BW-bound edge projections precomputed for NPU Mid. Run
`python scripts/export_ncu_for_sizer.py` after any new sweep. The
sizer repo can vendor this file directly into `sizer/measured/` or
fetch via raw github URL.

## Known gotchas

- **LLM server conflict.** Running `python3 llm_server.py` in another
  terminal holds 24 GB VRAM and blocks ncu. Stop it before profiling.
- **TRT engine rebuild time.** First run after deleting an engine takes
  20-60 seconds to rebuild. Cached after that.
- **Wall-clock inside ncu is garbage.** The `gpu__time_duration.sum`
  metric includes profiler overhead (serialized kernel launches,
  counter flushes). Use the normal non-profiled bake-off for latency.
- **Anonymous kernels in attribution.** Without
  `profiling_verbosity=DETAILED` on the engine, TRT kernels show up
  as `generated_kernel_<hash>`. Counts are still correct, but per-layer
  attribution is lost. Fixed by rebuilding engines — see above.
- **Large checkpoint loads dominate the first range.** Model-load
  kernels show up in the first NVTX range if the range is pushed too
  early. The instrumentation pushes ranges around forward passes only
  (after warmup), not load, so this shouldn't happen — but worth
  knowing if you see bizarre numbers.
