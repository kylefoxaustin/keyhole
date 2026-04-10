"""
NPU Emulator — Simulate Edge MPU Performance from GPU Baseline

Takes real profiling data from RTX 5090 pipeline runs and projects
what the workload would look like on a target edge NPU/MPU. Models
both compute-bound and bandwidth-bound bottlenecks with configurable
hardware specs.

Can operate in two modes:
  1. Projection Mode — post-hoc analysis of profiling data
  2. Throttle Mode — injects artificial delays during live inference
     to simulate real-time edge behavior (useful for UX validation)

Usage:
    # Projection from saved profile
    python -m src.emulate.npu_emulator project \\
        --profile data/output/profile_report.json \\
        --target-config configs/nxp_edge_mpu.json

    # Live throttled inference
    python -m src.main process --video test.mp4 --emulate-npu configs/nxp_edge_mpu.json
"""

import json
import time
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)


# ============================================================
# Hardware Spec Definitions
# ============================================================

@dataclass
class HardwareSpec:
    """Hardware specification for a compute target."""
    name: str
    tops_bf16: float         # Peak TOPS at BF16
    tops_int8: float         # Peak TOPS at INT8
    tops_int4: float         # Peak TOPS at INT4 (if supported)
    mem_bandwidth_gbs: float # Memory bandwidth in GB/s
    mem_capacity_gb: float   # Total DRAM capacity in GB
    mem_bus_width_bits: int   # Bus width (e.g., 128, 256, 384)
    mem_type: str            # LPDDR5X, GDDR7, HBM3, etc.
    mem_data_rate_gtps: float  # Data rate in GT/s

    # Efficiency factors (0.0 - 1.0)
    # Real silicon never hits peak TOPS — these model utilization
    compute_efficiency: float = 0.65   # Typical MAC utilization
    bandwidth_efficiency: float = 0.80  # Effective vs theoretical BW

    # Power envelope (for thermal projections)
    tdp_watts: float = 0.0

    @property
    def effective_tops_bf16(self) -> float:
        return self.tops_bf16 * self.compute_efficiency

    @property
    def effective_bandwidth_gbs(self) -> float:
        return self.mem_bandwidth_gbs * self.bandwidth_efficiency

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tops_bf16": self.tops_bf16,
            "tops_int8": self.tops_int8,
            "tops_int4": self.tops_int4,
            "mem_bandwidth_gbs": self.mem_bandwidth_gbs,
            "mem_capacity_gb": self.mem_capacity_gb,
            "mem_bus_width_bits": self.mem_bus_width_bits,
            "mem_type": self.mem_type,
            "mem_data_rate_gtps": self.mem_data_rate_gtps,
            "compute_efficiency": self.compute_efficiency,
            "bandwidth_efficiency": self.bandwidth_efficiency,
            "effective_tops_bf16": self.effective_tops_bf16,
            "effective_bandwidth_gbs": self.effective_bandwidth_gbs,
            "tdp_watts": self.tdp_watts,
        }


# --- Predefined Hardware Specs ---

RTX_5090 = HardwareSpec(
    name="NVIDIA RTX 5090",
    tops_bf16=209.0,
    tops_int8=419.0,
    tops_int4=838.0,
    mem_bandwidth_gbs=1792.0,
    mem_capacity_gb=32.0,
    mem_bus_width_bits=512,
    mem_type="GDDR7",
    mem_data_rate_gtps=28.0,
    compute_efficiency=0.70,
    bandwidth_efficiency=0.85,
    tdp_watts=575.0,
)

# Kyle's target edge MPU
NXP_EDGE_MPU = HardwareSpec(
    name="NXP Edge MPU (Target)",
    tops_bf16=200.0,
    tops_int8=400.0,
    tops_int4=800.0,
    mem_bandwidth_gbs=134.4,   # 128-bit LPDDR5X @ 8.4 GT/s
    mem_capacity_gb=8.0,
    mem_bus_width_bits=128,
    mem_type="LPDDR5X",
    mem_data_rate_gtps=8.4,
    compute_efficiency=0.60,   # Edge silicon typically lower utilization
    bandwidth_efficiency=0.75,
    tdp_watts=25.0,
)

# Hypothetical smaller variant for comparison
NXP_EDGE_MPU_LITE = HardwareSpec(
    name="NXP Edge MPU Lite",
    tops_bf16=50.0,
    tops_int8=100.0,
    tops_int4=200.0,
    mem_bandwidth_gbs=67.2,    # 64-bit LPDDR5X @ 8.4 GT/s
    mem_capacity_gb=4.0,
    mem_bus_width_bits=64,
    mem_type="LPDDR5X",
    mem_data_rate_gtps=8.4,
    compute_efficiency=0.55,
    bandwidth_efficiency=0.70,
    tdp_watts=12.0,
)

PRESET_TARGETS = {
    "rtx5090": RTX_5090,
    "nxp_edge": NXP_EDGE_MPU,
    "nxp_edge_lite": NXP_EDGE_MPU_LITE,
}


# ============================================================
# Workload Model
# ============================================================

@dataclass
class WorkloadProfile:
    """
    Characterization of a single pipeline stage's compute requirements.

    Captures both the measured GPU performance AND the underlying
    workload characteristics needed to project onto different hardware.
    """
    stage_name: str
    model_name: str

    # Model characteristics
    param_count: int = 0
    model_size_bytes: int = 0          # At operating precision
    precision: str = "bf16"            # bf16, fp16, int8, int4

    # Compute characteristics
    gflops_per_inference: float = 0.0  # Total GFLOPs per frame/token
    arithmetic_intensity: float = 0.0  # FLOPs per byte transferred
    is_bandwidth_bound: bool = False   # True for LLM decode, false for vision

    # Measured on reference GPU
    measured_latency_ms: float = 0.0
    measured_gpu: str = ""
    measured_gpu_utilization: float = 0.0
    measured_mem_bandwidth_util: float = 0.0

    # For LLM models
    is_autoregressive: bool = False
    avg_tokens_per_query: int = 0
    weight_load_per_token_bytes: int = 0

    # Activation memory (intermediate tensors during inference)
    peak_activation_bytes: int = 0

    def total_memory_bytes(self) -> int:
        """Total memory needed: model weights + peak activations."""
        return self.model_size_bytes + self.peak_activation_bytes


@dataclass
class EmulationResult:
    """Results of projecting a workload onto target hardware."""
    stage_name: str
    model_name: str

    # Projected latencies
    compute_limited_ms: float = 0.0    # If only compute matters
    bandwidth_limited_ms: float = 0.0  # If only bandwidth matters
    projected_latency_ms: float = 0.0  # max(compute, bandwidth)
    bottleneck: str = ""               # "compute" or "bandwidth"

    # Comparison to reference
    reference_latency_ms: float = 0.0
    slowdown_factor: float = 0.0

    # Feasibility flags
    fits_in_memory: bool = True
    memory_headroom_bytes: int = 0

    # For LLM stages
    projected_tokens_per_sec: float = 0.0

    # Throttle delay to inject for live emulation
    throttle_delay_ms: float = 0.0


# ============================================================
# NPU Emulator Engine
# ============================================================

class NPUEmulator:
    """
    Projects and emulates pipeline performance on target edge hardware.

    Two operating modes:
    1. **Project** — analyze saved profile data, produce comparison report
    2. **Throttle** — wrap inference calls with artificial delays to
       simulate edge latency in real-time

    The projection model accounts for:
    - Compute throughput scaling (TOPS ratio × efficiency)
    - Memory bandwidth scaling (GB/s ratio × efficiency)
    - DRAM capacity limits
    - Arithmetic intensity (determines compute vs bandwidth bottleneck)
    - Activation memory overhead
    """

    def __init__(
        self,
        reference: HardwareSpec = RTX_5090,
        target: HardwareSpec = NXP_EDGE_MPU,
    ):
        self.reference = reference
        self.target = target

        # Compute scaling ratios
        self.compute_ratio = (
            target.effective_tops_bf16 / reference.effective_tops_bf16
        )
        self.bandwidth_ratio = (
            target.effective_bandwidth_gbs / reference.effective_bandwidth_gbs
        )

        logger.info(
            "NPU Emulator: %s → %s  (compute: %.2fx, bandwidth: %.2fx)",
            reference.name, target.name,
            self.compute_ratio, self.bandwidth_ratio,
        )

    def project_workload(self, workload: WorkloadProfile) -> EmulationResult:
        """
        Project a workload from reference GPU to target NPU.

        For vision models (compute-bound):
            - Scale by TOPS ratio for compute time
            - Scale by bandwidth ratio for data movement
            - Bottleneck = max(compute_time, bandwidth_time)

        For LLM decode (bandwidth-bound):
            - Each token requires full weight matrix load
            - Throughput ≈ bandwidth / model_size
        """
        result = EmulationResult(
            stage_name=workload.stage_name,
            model_name=workload.model_name,
            reference_latency_ms=workload.measured_latency_ms,
        )

        # --- Memory feasibility ---
        total_mem = workload.total_memory_bytes()
        target_capacity = int(self.target.mem_capacity_gb * 1e9)
        result.fits_in_memory = total_mem < target_capacity
        result.memory_headroom_bytes = target_capacity - total_mem

        if workload.is_autoregressive:
            # --- LLM Autoregressive Decode ---
            # Bandwidth-dominated: each token loads full weight matrix
            weight_bytes = workload.weight_load_per_token_bytes or workload.model_size_bytes
            effective_bw = self.target.effective_bandwidth_gbs * 1e9  # bytes/s

            bytes_per_token = weight_bytes
            seconds_per_token = bytes_per_token / effective_bw
            ms_per_token = seconds_per_token * 1000

            result.bandwidth_limited_ms = ms_per_token
            result.compute_limited_ms = ms_per_token * 0.1  # Negligible
            result.projected_latency_ms = ms_per_token
            result.bottleneck = "bandwidth"
            result.projected_tokens_per_sec = 1000.0 / ms_per_token

            # For full query latency
            if workload.avg_tokens_per_query > 0:
                result.projected_latency_ms = (
                    ms_per_token * workload.avg_tokens_per_query
                )

        else:
            # --- Vision Model (per-frame) ---
            if workload.gflops_per_inference > 0:
                # Compute time from FLOP count
                effective_tops = self.target.effective_tops_bf16
                compute_seconds = workload.gflops_per_inference / (effective_tops * 1000)
                result.compute_limited_ms = compute_seconds * 1000
            else:
                # Scale from measured latency using compute ratio
                # This assumes the reference measurement was compute-bound
                result.compute_limited_ms = (
                    workload.measured_latency_ms / self.compute_ratio
                )

            # Bandwidth time: model weights + activations
            model_gb = workload.model_size_bytes / 1e9
            activation_gb = workload.peak_activation_bytes / 1e9
            # Vision models reuse weights across spatial positions,
            # so effective transfer is activation-dominated after warmup
            effective_transfer_gb = activation_gb + (model_gb * 0.05)
            if effective_transfer_gb > 0:
                result.bandwidth_limited_ms = (
                    effective_transfer_gb / self.target.effective_bandwidth_gbs * 1000
                )

            # Bottleneck is whichever takes longer
            result.projected_latency_ms = max(
                result.compute_limited_ms,
                result.bandwidth_limited_ms,
            )
            result.bottleneck = (
                "compute" if result.compute_limited_ms > result.bandwidth_limited_ms
                else "bandwidth"
            )

        # Slowdown vs reference
        if workload.measured_latency_ms > 0:
            result.slowdown_factor = (
                result.projected_latency_ms / workload.measured_latency_ms
            )
        else:
            result.slowdown_factor = 0.0

        # Throttle delay for live emulation
        result.throttle_delay_ms = max(
            0.0,
            result.projected_latency_ms - workload.measured_latency_ms,
        )

        return result

    def create_throttle_wrapper(
        self, workload: WorkloadProfile
    ) -> Callable:
        """
        Create a wrapper function that adds artificial delay
        to simulate edge NPU latency during live inference.

        Usage:
            throttle = emulator.create_throttle_wrapper(yolo_workload)
            # In detection loop:
            result = detector.detect_frame(frame)
            throttle(result.inference_ms)  # Adds remaining delay
        """
        projection = self.project_workload(workload)
        target_ms = projection.projected_latency_ms

        def throttle(actual_inference_ms: float):
            """Sleep for the remaining time to hit target latency."""
            remaining = target_ms - actual_inference_ms
            if remaining > 0:
                time.sleep(remaining / 1000.0)

        return throttle


# ============================================================
# Workload Profiler — Extract characteristics from live models
# ============================================================

class WorkloadProfiler:
    """
    Extracts workload characteristics from loaded models.

    Runs micro-benchmarks to measure actual compute and bandwidth
    utilization, then builds WorkloadProfile objects for projection.
    """

    @staticmethod
    def profile_yolo(detector, num_warmup: int = 5, num_measure: int = 20) -> WorkloadProfile:
        """Profile a loaded YOLO detector."""
        import torch
        import numpy as np

        model = detector.model.model
        params = sum(p.numel() for p in model.parameters())
        # Estimate model size at operating precision
        param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )

        # Measure inference latency
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        latencies = []

        for i in range(num_warmup + num_measure):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            detector.model.predict(dummy, verbose=False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000

            if i >= num_warmup:
                latencies.append(elapsed)

        avg_ms = sum(latencies) / len(latencies)

        # Estimate peak activation memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            detector.model.predict(dummy, verbose=False)
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated()
            activation_bytes = peak_mem - param_bytes
        else:
            activation_bytes = param_bytes * 2  # Rough estimate

        return WorkloadProfile(
            stage_name="yolo_detection",
            model_name=str(detector.model.model_name),
            param_count=params,
            model_size_bytes=param_bytes,
            precision="fp16",
            gflops_per_inference=196.0,  # YOLO 11x at 640x640
            arithmetic_intensity=85.0,
            is_bandwidth_bound=False,
            measured_latency_ms=avg_ms,
            measured_gpu=RTX_5090.name,
            peak_activation_bytes=max(0, activation_bytes),
        )

    @staticmethod
    def profile_sam3(enricher, num_warmup: int = 3, num_measure: int = 10) -> WorkloadProfile:
        """Profile a loaded SAM 3 enricher."""
        import torch
        import numpy as np
        from PIL import Image

        if enricher.model is None:
            return WorkloadProfile(
                stage_name="sam3_enrichment",
                model_name="sam3_not_loaded",
            )

        model = enricher.model
        params = sum(p.numel() for p in model.parameters())
        param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )

        # Measure with a realistic crop size
        dummy = Image.fromarray(
            np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        )
        latencies = []

        for i in range(num_warmup + num_measure):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            state = enricher.processor.set_image(dummy)
            enricher.processor.set_text_prompt(state=state, prompt="person")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000

            if i >= num_warmup:
                latencies.append(elapsed)

        avg_ms = sum(latencies) / len(latencies)

        # Peak activation memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            state = enricher.processor.set_image(dummy)
            enricher.processor.set_text_prompt(state=state, prompt="test")
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated()
            activation_bytes = peak_mem - param_bytes
        else:
            activation_bytes = param_bytes * 3

        return WorkloadProfile(
            stage_name="sam3_enrichment",
            model_name="sam3_full",
            param_count=params,
            model_size_bytes=param_bytes,
            precision="bf16",
            gflops_per_inference=350.0,
            arithmetic_intensity=120.0,
            is_bandwidth_bound=False,
            measured_latency_ms=avg_ms,
            measured_gpu=RTX_5090.name,
            peak_activation_bytes=max(0, activation_bytes),
        )

    @staticmethod
    def profile_llm_query(
        model_name: str = "qwen2.5:3b",
        model_size_gb: float = 1.5,
        measured_tok_per_sec: float = 0.0,
    ) -> WorkloadProfile:
        """
        Create an LLM workload profile.

        For LLMs we don't micro-benchmark here (they run on separate
        backends), but we characterize the workload for projection.
        """
        size_bytes = int(model_size_gb * 1e9)

        measured_ms = 0.0
        if measured_tok_per_sec > 0:
            measured_ms = 1000.0 / measured_tok_per_sec

        return WorkloadProfile(
            stage_name="nlq_query",
            model_name=model_name,
            param_count=0,  # Varies
            model_size_bytes=size_bytes,
            precision="int4",
            gflops_per_inference=0,
            arithmetic_intensity=1.0,
            is_bandwidth_bound=True,
            is_autoregressive=True,
            avg_tokens_per_query=80,
            weight_load_per_token_bytes=size_bytes,
            measured_latency_ms=measured_ms,
            measured_gpu=RTX_5090.name,
            peak_activation_bytes=int(0.5e9),  # KV cache estimate
        )


# ============================================================
# CLI — Projection Report
# ============================================================

@click.command()
@click.option("--profile", "profile_path", default=None,
              help="Path to profile_report.json")
@click.option("--target", "target_name", default="nxp_edge",
              type=click.Choice(list(PRESET_TARGETS.keys())),
              help="Target hardware preset")
@click.option("--target-config", default=None,
              help="Path to custom target hardware JSON config")
@click.option("--compare-all", is_flag=True,
              help="Compare across all preset targets")
def emulate(profile_path, target_name, target_config, compare_all):
    """Project pipeline performance onto edge NPU hardware."""

    # Load or build target spec
    if target_config:
        with open(target_config) as f:
            cfg = json.load(f)
        target = HardwareSpec(**cfg)
    else:
        target = PRESET_TARGETS[target_name]

    targets = list(PRESET_TARGETS.values()) if compare_all else [target]

    # Build workload profiles
    workloads = []

    if profile_path and Path(profile_path).exists():
        with open(profile_path) as f:
            data = json.load(f)

        if data.get("yolo"):
            yolo_wl = WorkloadProfile(
                stage_name="yolo_detection",
                model_name="yolo11x",
                param_count=int(data["yolo"].get("params_m", 57) * 1e6),
                model_size_bytes=int(data["yolo"].get("params_m", 57) * 1e6 * 2),
                precision="fp16",
                gflops_per_inference=196.0,
                arithmetic_intensity=85.0,
                measured_latency_ms=data["yolo"]["avg_ms"],
                measured_gpu=RTX_5090.name,
                peak_activation_bytes=int(0.2e9),
            )
            workloads.append(yolo_wl)

        if data.get("sam3", {}).get("avg_enrichment_ms"):
            sam_wl = WorkloadProfile(
                stage_name="sam3_enrichment",
                model_name="sam3_full",
                param_count=848_000_000,
                model_size_bytes=int(848e6 * 2),
                precision="bf16",
                gflops_per_inference=350.0,
                arithmetic_intensity=120.0,
                measured_latency_ms=data["sam3"]["avg_enrichment_ms"],
                measured_gpu=RTX_5090.name,
                peak_activation_bytes=int(1.0e9),
            )
            workloads.append(sam_wl)
    else:
        # Use estimated workloads if no profile data
        console.print("[yellow]No profile data — using estimated workloads[/]\n")

        workloads = [
            WorkloadProfile(
                stage_name="yolo_detection", model_name="yolo11x",
                param_count=57_000_000, model_size_bytes=int(57e6 * 2),
                precision="fp16", gflops_per_inference=196.0,
                arithmetic_intensity=85.0, measured_latency_ms=3.0,
                measured_gpu=RTX_5090.name, peak_activation_bytes=int(0.2e9),
            ),
            WorkloadProfile(
                stage_name="sam3_enrichment", model_name="sam3_full",
                param_count=848_000_000, model_size_bytes=int(848e6 * 2),
                precision="bf16", gflops_per_inference=350.0,
                arithmetic_intensity=120.0, measured_latency_ms=30.0,
                measured_gpu=RTX_5090.name, peak_activation_bytes=int(1.0e9),
            ),
            WorkloadProfile(
                stage_name="sam3_enrichment_lite", model_name="efficient_sam3_tinyvit",
                param_count=11_000_000, model_size_bytes=int(11e6 * 2),
                precision="fp16", gflops_per_inference=12.0,
                arithmetic_intensity=60.0, measured_latency_ms=5.0,
                measured_gpu=RTX_5090.name, peak_activation_bytes=int(0.05e9),
            ),
        ]

    # Add LLM workloads for comparison
    workloads.extend([
        WorkloadProfiler.profile_llm_query("qwen2.5_3b_int4", 1.5, 180.0),
        WorkloadProfiler.profile_llm_query("qwen2.5_7b_int4", 3.5, 90.0),
        WorkloadProfiler.profile_llm_query("llama3_8b_int4", 4.0, 75.0),
    ])

    # === Generate Report ===
    for tgt in targets:
        emulator = NPUEmulator(reference=RTX_5090, target=tgt)

        console.print(Panel.fit(
            f"[bold]{tgt.name}[/]\n"
            f"Compute: {tgt.tops_bf16} TOPS BF16 "
            f"(eff: {tgt.effective_tops_bf16:.0f} TOPS)\n"
            f"Bandwidth: {tgt.mem_bandwidth_gbs} GB/s "
            f"(eff: {tgt.effective_bandwidth_gbs:.0f} GB/s)\n"
            f"DRAM: {tgt.mem_capacity_gb} GB {tgt.mem_type} "
            f"({tgt.mem_bus_width_bits}-bit @ {tgt.mem_data_rate_gtps} GT/s)\n"
            f"TDP: {tgt.tdp_watts}W",
            border_style="cyan",
        ))

        # Vision stages
        console.print("\n[bold]Vision Pipeline Stages[/]")
        vtable = Table(show_header=True, header_style="bold")
        vtable.add_column("Stage", min_width=22)
        vtable.add_column("RTX 5090", justify="right", width=10)
        vtable.add_column("Projected", justify="right", width=10)
        vtable.add_column("Slowdown", justify="right", width=9)
        vtable.add_column("Bottleneck", width=11)
        vtable.add_column("Max FPS", justify="right", width=8)
        vtable.add_column("DRAM", justify="right", width=10)
        vtable.add_column("Fits?", width=6)

        for wl in workloads:
            if wl.is_autoregressive:
                continue
            result = emulator.project_workload(wl)
            fits_style = "green" if result.fits_in_memory else "red"
            fits_str = "YES" if result.fits_in_memory else "NO"
            mem_mb = wl.total_memory_bytes() / 1e6
            max_fps = 1000.0 / result.projected_latency_ms if result.projected_latency_ms > 0 else 999

            vtable.add_row(
                f"{wl.model_name}",
                f"{wl.measured_latency_ms:.1f}ms",
                f"{result.projected_latency_ms:.1f}ms",
                f"{result.slowdown_factor:.1f}x",
                result.bottleneck,
                f"{max_fps:.0f}",
                f"{mem_mb:.0f} MB",
                f"[{fits_style}]{fits_str}[/]",
            )

        console.print(vtable)

        # LLM stages
        console.print("\n[bold]LLM Query Stages[/]")
        ltable = Table(show_header=True, header_style="bold")
        ltable.add_column("Model", min_width=20)
        ltable.add_column("tok/sec", justify="right", width=10)
        ltable.add_column("Query (80tok)", justify="right", width=12)
        ltable.add_column("DRAM", justify="right", width=10)
        ltable.add_column("Fits?", width=6)

        for wl in workloads:
            if not wl.is_autoregressive:
                continue
            result = emulator.project_workload(wl)
            fits_style = "green" if result.fits_in_memory else "red"
            fits_str = "YES" if result.fits_in_memory else "NO"
            mem_mb = wl.total_memory_bytes() / 1e6

            ltable.add_row(
                wl.model_name,
                f"{result.projected_tokens_per_sec:.0f}",
                f"{result.projected_latency_ms:.0f}ms",
                f"{mem_mb:.0f} MB",
                f"[{fits_style}]{fits_str}[/]",
            )

        console.print(ltable)

        # Combined pipeline estimate
        vision_results = [
            emulator.project_workload(wl)
            for wl in workloads if not wl.is_autoregressive
        ]
        if vision_results:
            # Find the SAM 3 full + YOLO combo
            yolo_r = next((r for r in vision_results if "yolo" in r.model_name), None)
            sam_r = next((r for r in vision_results if "sam3_full" in r.model_name), None)

            if yolo_r and sam_r:
                combined_ms = yolo_r.projected_latency_ms + sam_r.projected_latency_ms
                combined_fps = 1000.0 / combined_ms if combined_ms > 0 else 0

                console.print(f"\n[bold]Combined Pipeline (YOLO + SAM 3)[/]")
                console.print(
                    f"  Per-frame: {combined_ms:.1f}ms → "
                    f"{combined_fps:.0f} FPS\n"
                    f"  At 1 FPS extraction: "
                    f"{'[green]FEASIBLE' if combined_ms < 1000 else '[red]TOO SLOW'}[/] "
                    f"({combined_ms/1000:.2f}s per frame, budget 1.0s)\n"
                    f"  At 5 FPS extraction: "
                    f"{'[green]FEASIBLE' if combined_ms < 200 else '[red]TOO SLOW'}[/] "
                    f"({combined_ms:.0f}ms per frame, budget 200ms)"
                )

        console.print()


def load_target_from_json(path: str) -> HardwareSpec:
    """Load a custom hardware target from JSON file."""
    with open(path) as f:
        cfg = json.load(f)
    return HardwareSpec(**cfg)


if __name__ == "__main__":
    emulate()
