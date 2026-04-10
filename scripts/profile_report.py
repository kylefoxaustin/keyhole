#!/usr/bin/env python3
"""
GPU Profiling Report Generator

Takes profiling data from a pipeline run on the RTX 5090 and maps
compute/bandwidth requirements to a target edge MPU, estimating
feasibility for each pipeline stage.

Usage:
    python scripts/profile_report.py --target-tops 200 --target-bw 134.4
    python scripts/profile_report.py --profile data/output/profile_report.json
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# --- Reference GPU specs ---
@dataclass
class GPUSpec:
    name: str
    tops_fp16: float    # TOPS at FP16/BF16
    bandwidth_gbs: float  # Memory bandwidth in GB/s
    vram_gb: float


RTX_5090 = GPUSpec(
    name="NVIDIA RTX 5090",
    tops_fp16=209.0,   # FP16 Tensor TOPS
    bandwidth_gbs=1792.0,
    vram_gb=32.0,
)

# --- Known model profiles ---
# These are estimated based on architecture analysis.
# Real values get filled in from --profile data.

MODEL_PROFILES = {
    "yolo11x": {
        "params_m": 56.9,
        "model_size_bf16_gb": 0.114,  # 56.9M * 2 bytes
        "gflops_per_frame": 196.0,    # At 640x640 input
        "arithmetic_intensity": 85.0,  # FLOPs/byte — compute bound
        "category": "compute-bound",
    },
    "sam3_full": {
        "params_m": 848.0,
        "model_size_bf16_gb": 1.696,
        "gflops_per_frame": 350.0,    # Estimated for 1024x1024
        "arithmetic_intensity": 120.0,
        "category": "compute-bound",
    },
    "sam3_efficient_tinyvit": {
        "params_m": 11.0,
        "model_size_bf16_gb": 0.022,
        "gflops_per_frame": 12.0,
        "arithmetic_intensity": 60.0,
        "category": "compute-bound",
    },
    "sam3_efficient_repvit": {
        "params_m": 25.0,
        "model_size_bf16_gb": 0.050,
        "gflops_per_frame": 28.0,
        "arithmetic_intensity": 70.0,
        "category": "compute-bound",
    },
    "llama3_8b_int4": {
        "params_m": 8000.0,
        "model_size_bf16_gb": 4.0,    # INT4 quantized
        "gflops_per_frame": 0,        # N/A — token-based
        "arithmetic_intensity": 1.0,   # Extremely bandwidth-bound
        "category": "bandwidth-bound",
        "tokens_per_weight_load_gb": 4.0,  # Full model load per token
    },
    "qwen25_3b_int4": {
        "params_m": 3000.0,
        "model_size_bf16_gb": 1.5,
        "gflops_per_frame": 0,
        "arithmetic_intensity": 1.0,
        "category": "bandwidth-bound",
        "tokens_per_weight_load_gb": 1.5,
    },
    "qwen25_7b_int4": {
        "params_m": 7000.0,
        "model_size_bf16_gb": 3.5,
        "gflops_per_frame": 0,
        "arithmetic_intensity": 1.0,
        "category": "bandwidth-bound",
        "tokens_per_weight_load_gb": 3.5,
    },
}


def estimate_vision_latency(
    model: dict,
    target_tops: float,
    target_bw: float,
) -> dict:
    """Estimate per-frame latency for a vision model on target hardware."""
    gflops = model["gflops_per_frame"]
    model_gb = model["model_size_bf16_gb"]

    # Compute time: GFLOPs / TOPS → milliseconds
    compute_ms = (gflops / target_tops) * 1000

    # Bandwidth time (activation + weight loading)
    # For vision models, activation bandwidth is typically 10-20% of weight size
    effective_transfer_gb = model_gb * 0.15  # Activation-dominated after warmup
    bandwidth_ms = (effective_transfer_gb / target_bw) * 1000

    # Actual latency is max of compute and bandwidth (whichever bottlenecks)
    latency_ms = max(compute_ms, bandwidth_ms)
    bottleneck = "compute" if compute_ms > bandwidth_ms else "bandwidth"

    return {
        "compute_ms": compute_ms,
        "bandwidth_ms": bandwidth_ms,
        "estimated_ms": latency_ms,
        "max_fps": 1000.0 / latency_ms if latency_ms > 0 else 999,
        "bottleneck": bottleneck,
    }


def estimate_llm_throughput(
    model: dict,
    target_bw: float,
) -> dict:
    """Estimate token throughput for an LLM on target hardware."""
    weight_gb = model.get("tokens_per_weight_load_gb", model["model_size_bf16_gb"])

    # LLM autoregressive decoding is almost purely bandwidth-bound
    # Each token requires loading the full weight matrix
    ms_per_token = (weight_gb / target_bw) * 1000
    tokens_per_sec = 1000.0 / ms_per_token if ms_per_token > 0 else 0

    return {
        "ms_per_token": ms_per_token,
        "tokens_per_sec": tokens_per_sec,
        "weight_load_gb": weight_gb,
        "bottleneck": "bandwidth",
    }


@click.command()
@click.option("--target-tops", default=200.0, help="Target MPU TOPS (BF16)")
@click.option("--target-bw", default=134.4, help="Target memory bandwidth (GB/s)")
@click.option("--target-vram", default=8.0, help="Target DRAM capacity (GB)")
@click.option(
    "--profile", "profile_path", default=None,
    help="Path to profile_report.json from pipeline run",
)
def generate_report(
    target_tops: float,
    target_bw: float,
    target_vram: float,
    profile_path: str,
):
    """Generate edge MPU feasibility report."""

    console.print(Panel.fit(
        "[bold]Keyhole — Edge MPU Feasibility Report[/]\n"
        f"Target: {target_tops} TOPS BF16, {target_bw} GB/s bandwidth, "
        f"{target_vram} GB DRAM",
        border_style="blue",
    ))

    # Load measured profile data if available
    measured = {}
    if profile_path and Path(profile_path).exists():
        with open(profile_path) as f:
            measured = json.load(f)
        console.print(f"\n[dim]Loaded measured profile from: {profile_path}[/]")

    # === Vision Models Table ===
    console.print("\n[bold]Vision Model Estimates (per frame)[/]")

    vtable = Table(show_header=True, header_style="bold cyan")
    vtable.add_column("Model", min_width=20)
    vtable.add_column("Params", justify="right", width=10)
    vtable.add_column("Size (BF16)", justify="right", width=10)
    vtable.add_column("GFLOPs", justify="right", width=8)
    vtable.add_column("Compute ms", justify="right", width=10)
    vtable.add_column("BW ms", justify="right", width=8)
    vtable.add_column("Est. ms", justify="right", width=8)
    vtable.add_column("Max FPS", justify="right", width=8)
    vtable.add_column("Bottleneck", width=10)
    vtable.add_column("Fits DRAM?", width=10)

    vision_models = [
        "yolo11x", "sam3_full",
        "sam3_efficient_tinyvit", "sam3_efficient_repvit",
    ]

    for name in vision_models:
        model = MODEL_PROFILES[name]
        est = estimate_vision_latency(model, target_tops, target_bw)
        fits = "YES" if model["model_size_bf16_gb"] < target_vram else "NO"
        fits_style = "green" if fits == "YES" else "red"

        vtable.add_row(
            name,
            f"{model['params_m']:.0f}M",
            f"{model['model_size_bf16_gb']:.3f} GB",
            f"{model['gflops_per_frame']:.0f}",
            f"{est['compute_ms']:.2f}",
            f"{est['bandwidth_ms']:.2f}",
            f"{est['estimated_ms']:.2f}",
            f"{est['max_fps']:.0f}",
            est["bottleneck"],
            f"[{fits_style}]{fits}[/]",
        )

    console.print(vtable)

    # === LLM Models Table ===
    console.print("\n[bold]LLM Estimates (NLQ query engine)[/]")

    ltable = Table(show_header=True, header_style="bold cyan")
    ltable.add_column("Model", min_width=20)
    ltable.add_column("Size (INT4)", justify="right", width=10)
    ltable.add_column("ms/token", justify="right", width=10)
    ltable.add_column("tok/sec", justify="right", width=10)
    ltable.add_column("Fits DRAM?", width=10)

    llm_models = ["qwen25_3b_int4", "qwen25_7b_int4", "llama3_8b_int4"]

    for name in llm_models:
        model = MODEL_PROFILES[name]
        est = estimate_llm_throughput(model, target_bw)
        fits = "YES" if model["model_size_bf16_gb"] < target_vram else "NO"
        fits_style = "green" if fits == "YES" else "red"

        ltable.add_row(
            name,
            f"{model['model_size_bf16_gb']:.1f} GB",
            f"{est['ms_per_token']:.1f}",
            f"{est['tokens_per_sec']:.0f}",
            f"[{fits_style}]{fits}[/]",
        )

    console.print(ltable)

    # === Measured vs Estimated (if profile data available) ===
    if measured.get("yolo"):
        console.print("\n[bold]Measured on RTX 5090 vs Edge MPU Estimate[/]")

        ctable = Table(show_header=True, header_style="bold cyan")
        ctable.add_column("Stage", min_width=15)
        ctable.add_column("RTX 5090 (measured)", justify="right", width=18)
        ctable.add_column(f"Edge MPU (estimated)", justify="right", width=18)
        ctable.add_column("Slowdown", justify="right", width=10)

        yolo_measured = measured["yolo"]["avg_ms"]
        yolo_est = estimate_vision_latency(
            MODEL_PROFILES["yolo11x"], target_tops, target_bw
        )["estimated_ms"]

        ctable.add_row(
            "YOLO 11x",
            f"{yolo_measured:.1f} ms",
            f"{yolo_est:.1f} ms",
            f"{yolo_est / yolo_measured:.1f}x",
        )

        if measured.get("sam3", {}).get("avg_enrichment_ms"):
            sam_measured = measured["sam3"]["avg_enrichment_ms"]
            sam_est = estimate_vision_latency(
                MODEL_PROFILES["sam3_full"], target_tops, target_bw
            )["estimated_ms"]
            ctable.add_row(
                "SAM 3",
                f"{sam_measured:.1f} ms",
                f"{sam_est:.1f} ms",
                f"{sam_est / sam_measured:.1f}x" if sam_measured > 0 else "N/A",
            )

        console.print(ctable)

    # === Recommendations ===
    console.print("\n[bold]Recommendations for Edge MPU[/]")
    console.print(Panel(
        "[bold]Detection Tier:[/]\n"
        "  YOLO 11x fits comfortably. At 200 TOPS, expect <1ms compute.\n"
        "  Even YOLO 11x is overkill — YOLO 11n/s would free compute headroom.\n"
        "\n"
        "[bold]Enrichment Tier:[/]\n"
        "  SAM 3 full (848M) fits in DRAM and is compute-bound, making it a\n"
        "  natural fit for high-TOPS / moderate-bandwidth silicon.\n"
        "  EfficientSAM3 (TinyViT/RepViT) variants run at <5ms — massive margin.\n"
        "\n"
        "[bold]NLQ Tier:[/]\n"
        "  LLM inference is bandwidth-bound. Qwen 2.5 3B at INT4 gives ~89 tok/s.\n"
        "  This is sufficient for query translation (responses are <100 tokens).\n"
        "  7B+ models work but slower — consider offloading NLQ to a companion AP.\n"
        "\n"
        "[bold]Combined Pipeline:[/]\n"
        "  YOLO + SAM 3 full can coexist in 8 GB DRAM with room for an INT4 3B LLM.\n"
        "  Total model footprint: ~0.11 + 1.7 + 1.5 = ~3.3 GB, leaving 4.7 GB\n"
        "  for activations, KV cache, and OS overhead.\n",
        border_style="green",
        title="Edge MPU Strategy",
    ))


if __name__ == "__main__":
    generate_report()
