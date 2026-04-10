"""
Workload Characterization Export — Per-Layer Profiling for Hardware Simulation

Hooks into PyTorch model execution to capture detailed per-layer metrics:
- FLOPs, MACs, parameter counts
- Input/output tensor shapes and sizes
- Memory read/write bytes per layer
- Arithmetic intensity (FLOPs per byte transferred)
- Operator type classification (conv, attention, linear, norm, activation)

Exports in formats consumable by hardware performance models:
- JSON  — machine-readable for custom simulators
- CSV   — for Excel/spreadsheet analysis
- YAML  — for hardware description language tooling

Usage:
    # Profile YOLO model and export
    python -m src.emulate.layer_profiler --model yolo --output workload_yolo.json

    # Profile SAM 3 and export CSV for spreadsheet analysis
    python -m src.emulate.layer_profiler --model sam3 --format csv --output workload_sam3.csv

    # Profile with a real input frame
    python -m src.emulate.layer_profiler --model yolo --input frame.jpg --output yolo_layers.json
"""

import csv
import json
import time
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Layer Classification
# ============================================================

# Map PyTorch module types to hardware-meaningful operator categories
OP_CATEGORY_MAP = {
    # Convolutions
    nn.Conv1d: "conv",
    nn.Conv2d: "conv",
    nn.Conv3d: "conv",
    nn.ConvTranspose2d: "conv_transpose",
    # Linear / Dense
    nn.Linear: "linear",
    # Normalization
    nn.BatchNorm1d: "norm",
    nn.BatchNorm2d: "norm",
    nn.LayerNorm: "norm",
    nn.GroupNorm: "norm",
    nn.InstanceNorm2d: "norm",
    # Activation
    nn.ReLU: "activation",
    nn.GELU: "activation",
    nn.SiLU: "activation",
    nn.Sigmoid: "activation",
    nn.Softmax: "activation",
    nn.Hardswish: "activation",
    nn.Hardsigmoid: "activation",
    # Pooling
    nn.AdaptiveAvgPool2d: "pool",
    nn.AvgPool2d: "pool",
    nn.MaxPool2d: "pool",
    # Attention (detected by name pattern, not type)
    nn.MultiheadAttention: "attention",
    # Embedding
    nn.Embedding: "embedding",
    # Dropout (no-op at inference)
    nn.Dropout: "dropout",
    nn.Dropout2d: "dropout",
}


def classify_layer(module: nn.Module, name: str) -> str:
    """Classify a layer into a hardware-meaningful category."""
    # Check direct type match
    for mod_type, category in OP_CATEGORY_MAP.items():
        if isinstance(module, mod_type):
            return category

    # Pattern-based detection for custom modules
    name_lower = name.lower()
    if "attn" in name_lower or "attention" in name_lower:
        return "attention"
    if "conv" in name_lower:
        return "conv"
    if "norm" in name_lower or "bn" in name_lower:
        return "norm"
    if "linear" in name_lower or "fc" in name_lower or "proj" in name_lower:
        return "linear"
    if "pool" in name_lower:
        return "pool"
    if "act" in name_lower or "relu" in name_lower or "gelu" in name_lower:
        return "activation"
    if "embed" in name_lower:
        return "embedding"
    if "detr" in name_lower or "decoder" in name_lower:
        return "decoder"

    return "other"


# ============================================================
# FLOP Estimation per Layer
# ============================================================

def estimate_conv2d_flops(module: nn.Conv2d, input_shape: tuple) -> int:
    """Estimate FLOPs for Conv2d layer."""
    batch, c_in, h_in, w_in = input_shape
    c_out = module.out_channels
    k_h, k_w = module.kernel_size
    stride_h, stride_w = module.stride
    groups = module.groups

    h_out = (h_in + 2 * module.padding[0] - k_h) // stride_h + 1
    w_out = (w_in + 2 * module.padding[1] - k_w) // stride_w + 1

    # MACs = output_elements × kernel_size × (c_in / groups)
    macs = batch * c_out * h_out * w_out * k_h * k_w * (c_in // groups)
    flops = 2 * macs  # multiply + accumulate
    if module.bias is not None:
        flops += batch * c_out * h_out * w_out
    return flops


def estimate_linear_flops(module: nn.Linear, input_shape: tuple) -> int:
    """Estimate FLOPs for Linear layer."""
    # input_shape could be (batch, features) or (batch, seq, features)
    batch_elements = 1
    for dim in input_shape[:-1]:
        batch_elements *= dim

    macs = batch_elements * module.in_features * module.out_features
    flops = 2 * macs
    if module.bias is not None:
        flops += batch_elements * module.out_features
    return flops


def estimate_norm_flops(module: nn.Module, input_shape: tuple) -> int:
    """Estimate FLOPs for normalization layers."""
    elements = 1
    for dim in input_shape:
        elements *= dim
    # Mean, variance, normalize, scale, shift ≈ 5 ops per element
    return 5 * elements


def estimate_attention_flops(input_shape: tuple, head_dim: int = 64, num_heads: int = 8) -> int:
    """Estimate FLOPs for self-attention (Q×K^T + softmax + ×V)."""
    if len(input_shape) >= 3:
        batch, seq_len = input_shape[0], input_shape[1]
    else:
        batch, seq_len = input_shape[0], 1

    # Q×K^T: batch × heads × seq × seq × head_dim
    qk_flops = 2 * batch * num_heads * seq_len * seq_len * head_dim
    # softmax: ~5 ops per element
    softmax_flops = 5 * batch * num_heads * seq_len * seq_len
    # attn×V: batch × heads × seq × head_dim × seq
    av_flops = 2 * batch * num_heads * seq_len * head_dim * seq_len
    return qk_flops + softmax_flops + av_flops


def estimate_layer_flops(module: nn.Module, name: str, input_shape: tuple) -> int:
    """Estimate FLOPs for any layer type."""
    if isinstance(module, nn.Conv2d):
        return estimate_conv2d_flops(module, input_shape)
    elif isinstance(module, nn.Linear):
        return estimate_linear_flops(module, input_shape)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        return estimate_norm_flops(module, input_shape)
    elif isinstance(module, nn.MultiheadAttention):
        return estimate_attention_flops(input_shape)
    elif "attn" in name.lower() or "attention" in name.lower():
        return estimate_attention_flops(input_shape)
    else:
        # For activations, pooling, etc.: ~1 op per element
        elements = 1
        for dim in input_shape:
            elements *= dim
        return elements


# ============================================================
# Layer Profile Data Structure
# ============================================================

@dataclass
class LayerProfile:
    """Complete characterization of a single model layer."""
    # Identity
    layer_index: int
    layer_name: str
    module_type: str        # e.g., "Conv2d", "Linear", "MultiheadAttention"
    op_category: str        # e.g., "conv", "linear", "attention", "norm"

    # Tensor shapes
    input_shape: list       # e.g., [1, 256, 64, 64]
    output_shape: list      # e.g., [1, 512, 32, 32]

    # Parameters
    param_count: int = 0
    param_bytes: int = 0    # At operating precision
    param_precision: str = "fp32"

    # Compute
    flops: int = 0          # Floating point operations
    macs: int = 0           # Multiply-accumulate ops (flops / 2)

    # Memory access
    input_bytes: int = 0    # Bytes read from memory
    output_bytes: int = 0   # Bytes written to memory
    weight_bytes: int = 0   # Weight tensor bytes accessed
    total_memory_bytes: int = 0  # Total bytes transferred

    # Derived metrics
    arithmetic_intensity: float = 0.0   # FLOPs / total_memory_bytes
    compute_fraction: float = 0.0       # This layer's FLOPs / total model FLOPs

    # Measured timing (filled in during profiling)
    measured_ms: float = 0.0
    cuda_time_ms: float = 0.0

    # Hardware mapping hints
    can_fuse_with_next: bool = False  # Conv+BN+ReLU fusion candidate
    is_depthwise: bool = False        # Depthwise separable conv
    is_pointwise: bool = False        # 1x1 conv


@dataclass
class ModelWorkloadExport:
    """Complete workload characterization for hardware simulation."""
    # Model identity
    model_name: str
    model_family: str       # e.g., "yolo", "sam3", "llm"
    framework: str = "pytorch"
    export_version: str = "1.0"

    # Global model stats
    total_params: int = 0
    total_param_bytes: int = 0
    total_flops: int = 0
    total_macs: int = 0
    total_memory_bytes: int = 0
    operating_precision: str = "fp32"

    # Input specification
    input_shape: list = field(default_factory=list)
    input_dtype: str = "float32"

    # Per-layer profiles
    layers: list[LayerProfile] = field(default_factory=list)

    # Aggregate by category
    category_summary: dict = field(default_factory=dict)

    # Measured reference timing
    reference_hardware: str = ""
    reference_latency_ms: float = 0.0
    reference_throughput_fps: float = 0.0

    # Memory bandwidth analysis
    peak_activation_bytes: int = 0
    min_memory_required_bytes: int = 0


# ============================================================
# Model Profiler
# ============================================================

class LayerProfiler:
    """
    Hooks into PyTorch model execution to capture per-layer metrics.

    Usage:
        profiler = LayerProfiler("yolo11x", "yolo")
        export = profiler.profile_model(model, input_tensor)
        profiler.export_json(export, "yolo_workload.json")
        profiler.export_csv(export, "yolo_workload.csv")
    """

    def __init__(self, model_name: str, model_family: str):
        self.model_name = model_name
        self.model_family = model_family
        self._hooks = []
        self._layer_data = OrderedDict()
        self._layer_index = 0

    def _create_hook(self, name: str, module: nn.Module):
        """Create a forward hook that captures layer I/O shapes and timing."""
        def hook_fn(mod, inp, out):
            # Get input shape
            if isinstance(inp, tuple) and len(inp) > 0:
                if isinstance(inp[0], torch.Tensor):
                    input_shape = list(inp[0].shape)
                    input_bytes = inp[0].nelement() * inp[0].element_size()
                else:
                    input_shape = []
                    input_bytes = 0
            elif isinstance(inp, torch.Tensor):
                input_shape = list(inp.shape)
                input_bytes = inp.nelement() * inp.element_size()
            else:
                input_shape = []
                input_bytes = 0

            # Get output shape
            if isinstance(out, tuple):
                if len(out) > 0 and isinstance(out[0], torch.Tensor):
                    output_shape = list(out[0].shape)
                    output_bytes = out[0].nelement() * out[0].element_size()
                else:
                    output_shape = []
                    output_bytes = 0
            elif isinstance(out, torch.Tensor):
                output_shape = list(out.shape)
                output_bytes = out.nelement() * out.element_size()
            else:
                output_shape = []
                output_bytes = 0

            # Parameter stats
            param_count = sum(p.numel() for p in mod.parameters(recurse=False))
            param_bytes = sum(
                p.numel() * p.element_size() for p in mod.parameters(recurse=False)
            )
            precision = "fp32"
            if param_count > 0:
                first_param = next(mod.parameters(recurse=False))
                if first_param.dtype == torch.float16:
                    precision = "fp16"
                elif first_param.dtype == torch.bfloat16:
                    precision = "bf16"

            # Estimate FLOPs
            flops = estimate_layer_flops(mod, name, tuple(input_shape)) if input_shape else 0
            macs = flops // 2

            # Memory access
            weight_bytes = param_bytes
            total_mem = input_bytes + output_bytes + weight_bytes

            # Arithmetic intensity
            arith_intensity = flops / total_mem if total_mem > 0 else 0.0

            # Classification
            op_category = classify_layer(mod, name)

            # Detect special conv types
            is_depthwise = (
                isinstance(mod, nn.Conv2d) and
                mod.groups == mod.in_channels == mod.out_channels
            )
            is_pointwise = (
                isinstance(mod, nn.Conv2d) and
                mod.kernel_size == (1, 1)
            )

            self._layer_data[name] = LayerProfile(
                layer_index=self._layer_index,
                layer_name=name,
                module_type=type(mod).__name__,
                op_category=op_category,
                input_shape=input_shape,
                output_shape=output_shape,
                param_count=param_count,
                param_bytes=param_bytes,
                param_precision=precision,
                flops=flops,
                macs=macs,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                weight_bytes=weight_bytes,
                total_memory_bytes=total_mem,
                arithmetic_intensity=arith_intensity,
                is_depthwise=is_depthwise,
                is_pointwise=is_pointwise,
            )
            self._layer_index += 1

        return hook_fn

    def profile_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        warmup_runs: int = 3,
        measure_runs: int = 5,
    ) -> ModelWorkloadExport:
        """
        Run a full profiling pass on the model.

        Registers hooks on all leaf modules, runs inference, and
        captures per-layer metrics.
        """
        self._layer_data.clear()
        self._layer_index = 0
        self._hooks.clear()

        model.eval()

        # Register hooks on leaf modules (modules with no children)
        for name, module in model.named_modules():
            # Only hook leaf modules (actual compute) and named containers
            # that are interesting (attention blocks, etc.)
            children = list(module.children())
            if len(children) == 0 or isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(
                    self._create_hook(name, module)
                )
                self._hooks.append(hook)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                model(sample_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        # Clear and measure
        self._layer_data.clear()
        self._layer_index = 0

        latencies = []
        with torch.no_grad():
            for _ in range(measure_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                model(sample_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        # Remove hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        # Build export
        layers = list(self._layer_data.values())
        total_flops = sum(l.flops for l in layers)
        total_macs = sum(l.macs for l in layers)
        total_params = sum(p.numel() for p in model.parameters())
        total_param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        total_mem = sum(l.total_memory_bytes for l in layers)

        # Compute fractions
        for layer in layers:
            layer.compute_fraction = (
                layer.flops / total_flops if total_flops > 0 else 0
            )

        # Detect fusion candidates (Conv + BN + ReLU patterns)
        for i, layer in enumerate(layers[:-1]):
            next_layer = layers[i + 1]
            if (layer.op_category == "conv" and
                    next_layer.op_category in ("norm", "activation")):
                layer.can_fuse_with_next = True

        # Category summary
        categories = {}
        for layer in layers:
            cat = layer.op_category
            if cat not in categories:
                categories[cat] = {
                    "count": 0, "flops": 0, "macs": 0,
                    "params": 0, "memory_bytes": 0,
                    "flops_pct": 0.0,
                }
            categories[cat]["count"] += 1
            categories[cat]["flops"] += layer.flops
            categories[cat]["macs"] += layer.macs
            categories[cat]["params"] += layer.param_count
            categories[cat]["memory_bytes"] += layer.total_memory_bytes

        for cat in categories:
            categories[cat]["flops_pct"] = (
                categories[cat]["flops"] / total_flops * 100
                if total_flops > 0 else 0
            )

        # Peak activation memory (max of all output tensors)
        peak_act = max((l.output_bytes for l in layers), default=0)

        # Determine operating precision
        precision = "fp32"
        if total_param_bytes > 0 and total_params > 0:
            bytes_per_param = total_param_bytes / total_params
            if bytes_per_param <= 2.1:
                precision = "fp16/bf16"
            elif bytes_per_param <= 1.1:
                precision = "int8"

        export = ModelWorkloadExport(
            model_name=self.model_name,
            model_family=self.model_family,
            total_params=total_params,
            total_param_bytes=total_param_bytes,
            total_flops=total_flops,
            total_macs=total_macs,
            total_memory_bytes=total_mem,
            operating_precision=precision,
            input_shape=list(sample_input.shape),
            input_dtype=str(sample_input.dtype),
            layers=layers,
            category_summary=categories,
            reference_hardware="NVIDIA RTX 5090",
            reference_latency_ms=avg_latency,
            reference_throughput_fps=1000.0 / avg_latency if avg_latency > 0 else 0,
            peak_activation_bytes=peak_act,
            min_memory_required_bytes=total_param_bytes + peak_act,
        )

        logger.info(
            "Profiled %s: %d layers, %.1fM params, %.1f GFLOPs, %.1fms",
            self.model_name, len(layers), total_params / 1e6,
            total_flops / 1e9, avg_latency,
        )

        return export

    # ============================================================
    # Export Formats
    # ============================================================

    @staticmethod
    def export_json(export: ModelWorkloadExport, output_path: str):
        """
        Export full workload characterization as JSON.

        This is the primary format for consumption by hardware
        performance simulators and custom analysis tools.
        """
        data = {
            "metadata": {
                "model_name": export.model_name,
                "model_family": export.model_family,
                "framework": export.framework,
                "export_version": export.export_version,
                "reference_hardware": export.reference_hardware,
                "reference_latency_ms": export.reference_latency_ms,
                "reference_throughput_fps": export.reference_throughput_fps,
            },
            "model_summary": {
                "total_params": export.total_params,
                "total_param_bytes": export.total_param_bytes,
                "total_flops": export.total_flops,
                "total_macs": export.total_macs,
                "total_gflops": export.total_flops / 1e9,
                "total_memory_bytes": export.total_memory_bytes,
                "operating_precision": export.operating_precision,
                "input_shape": export.input_shape,
                "input_dtype": export.input_dtype,
                "peak_activation_bytes": export.peak_activation_bytes,
                "min_memory_required_bytes": export.min_memory_required_bytes,
                "min_memory_required_mb": export.min_memory_required_bytes / 1e6,
            },
            "category_summary": export.category_summary,
            "layers": [asdict(layer) for layer in export.layers],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Exported JSON workload: %s (%d layers)", output_path, len(export.layers))

    @staticmethod
    def export_csv(export: ModelWorkloadExport, output_path: str):
        """
        Export per-layer data as CSV for spreadsheet analysis.

        Columns designed for direct import into Excel pivot tables
        or hardware team analysis workflows.
        """
        fieldnames = [
            "layer_index", "layer_name", "module_type", "op_category",
            "input_shape", "output_shape",
            "param_count", "param_bytes_kb", "param_precision",
            "flops", "flops_giga", "macs", "macs_giga",
            "input_bytes_kb", "output_bytes_kb", "weight_bytes_kb",
            "total_memory_bytes_kb",
            "arithmetic_intensity",
            "compute_fraction_pct",
            "is_depthwise", "is_pointwise", "can_fuse_with_next",
            "measured_ms",
        ]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for layer in export.layers:
                writer.writerow({
                    "layer_index": layer.layer_index,
                    "layer_name": layer.layer_name,
                    "module_type": layer.module_type,
                    "op_category": layer.op_category,
                    "input_shape": str(layer.input_shape),
                    "output_shape": str(layer.output_shape),
                    "param_count": layer.param_count,
                    "param_bytes_kb": f"{layer.param_bytes / 1024:.2f}",
                    "param_precision": layer.param_precision,
                    "flops": layer.flops,
                    "flops_giga": f"{layer.flops / 1e9:.4f}",
                    "macs": layer.macs,
                    "macs_giga": f"{layer.macs / 1e9:.4f}",
                    "input_bytes_kb": f"{layer.input_bytes / 1024:.2f}",
                    "output_bytes_kb": f"{layer.output_bytes / 1024:.2f}",
                    "weight_bytes_kb": f"{layer.weight_bytes / 1024:.2f}",
                    "total_memory_bytes_kb": f"{layer.total_memory_bytes / 1024:.2f}",
                    "arithmetic_intensity": f"{layer.arithmetic_intensity:.2f}",
                    "compute_fraction_pct": f"{layer.compute_fraction * 100:.3f}",
                    "is_depthwise": layer.is_depthwise,
                    "is_pointwise": layer.is_pointwise,
                    "can_fuse_with_next": layer.can_fuse_with_next,
                    "measured_ms": f"{layer.measured_ms:.3f}",
                })

        # Also write a summary sheet
        summary_path = output_path.replace(".csv", "_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Model", export.model_name])
            writer.writerow(["Total Parameters", export.total_params])
            writer.writerow(["Total Param MB", f"{export.total_param_bytes / 1e6:.2f}"])
            writer.writerow(["Total GFLOPs", f"{export.total_flops / 1e9:.2f}"])
            writer.writerow(["Total GMACs", f"{export.total_macs / 1e9:.2f}"])
            writer.writerow(["Operating Precision", export.operating_precision])
            writer.writerow(["Input Shape", str(export.input_shape)])
            writer.writerow(["Peak Activation MB", f"{export.peak_activation_bytes / 1e6:.2f}"])
            writer.writerow(["Min Memory Required MB", f"{export.min_memory_required_bytes / 1e6:.2f}"])
            writer.writerow(["Reference Hardware", export.reference_hardware])
            writer.writerow(["Reference Latency ms", f"{export.reference_latency_ms:.2f}"])
            writer.writerow([])
            writer.writerow(["Category", "Count", "GFLOPs", "FLOPs %", "Params", "Memory KB"])
            for cat, stats in sorted(
                export.category_summary.items(),
                key=lambda x: x[1]["flops"],
                reverse=True,
            ):
                writer.writerow([
                    cat,
                    stats["count"],
                    f"{stats['flops'] / 1e9:.2f}",
                    f"{stats['flops_pct']:.1f}",
                    stats["params"],
                    f"{stats['memory_bytes'] / 1024:.0f}",
                ])

        logger.info(
            "Exported CSV workload: %s + %s (%d layers)",
            output_path, summary_path, len(export.layers),
        )

    @staticmethod
    def export_hardware_sim(export: ModelWorkloadExport, output_path: str):
        """
        Export in a hardware-simulator-friendly format.

        Produces a flat JSON array where each entry represents one
        compute kernel dispatch, with fields directly mappable to
        NPU pipeline stages:

        - op_type: kernel type for dispatch scheduling
        - compute_ops: total FLOPs for the kernel
        - read_bytes: bytes read from DRAM/SRAM
        - write_bytes: bytes written to DRAM/SRAM
        - weight_bytes: weight tensor bytes (cacheable)
        - arithmetic_intensity: ops/byte ratio
        - can_fuse: whether this kernel can fuse with successor

        Hardware simulators can iterate this list to model:
        - Pipeline scheduling and utilization
        - SRAM vs DRAM traffic based on tensor sizes
        - Kernel fusion opportunities
        - Compute vs memory bottleneck per stage
        """
        kernels = []

        for layer in export.layers:
            if layer.op_category == "dropout":
                continue  # No-op at inference

            kernel = {
                "kernel_id": layer.layer_index,
                "name": layer.layer_name,
                "op_type": layer.op_category,
                "module_type": layer.module_type,

                # Compute
                "compute_ops": layer.flops,
                "mac_ops": layer.macs,

                # Memory access pattern
                "read_bytes": layer.input_bytes + layer.weight_bytes,
                "write_bytes": layer.output_bytes,
                "weight_bytes": layer.weight_bytes,
                "activation_read_bytes": layer.input_bytes,
                "activation_write_bytes": layer.output_bytes,

                # Tensor dimensions (for SRAM tiling decisions)
                "input_shape": layer.input_shape,
                "output_shape": layer.output_shape,
                "param_count": layer.param_count,

                # Derived
                "arithmetic_intensity": round(layer.arithmetic_intensity, 2),
                "compute_fraction": round(layer.compute_fraction, 4),

                # Scheduling hints
                "can_fuse_with_next": layer.can_fuse_with_next,
                "is_depthwise_conv": layer.is_depthwise,
                "is_pointwise_conv": layer.is_pointwise,

                # Reference timing
                "reference_ms": layer.measured_ms,
            }
            kernels.append(kernel)

        output = {
            "format": "ai_sentinel_hw_sim_v1",
            "model": export.model_name,
            "total_compute_ops": export.total_flops,
            "total_read_bytes": sum(k["read_bytes"] for k in kernels),
            "total_write_bytes": sum(k["write_bytes"] for k in kernels),
            "total_weight_bytes": export.total_param_bytes,
            "reference_hardware": export.reference_hardware,
            "reference_latency_ms": export.reference_latency_ms,
            "kernel_count": len(kernels),
            "kernels": kernels,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(
            "Exported HW sim workload: %s (%d kernels, %.1f GFLOPs)",
            output_path, len(kernels), export.total_flops / 1e9,
        )


# ============================================================
# CLI Entry Point
# ============================================================

import click


@click.command()
@click.option("--model", type=click.Choice(["yolo", "sam3"]),
              required=True, help="Model to profile")
@click.option("--format", "fmt", type=click.Choice(["json", "csv", "hwsim", "all"]),
              default="all", help="Export format")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--input-size", default=640, type=int,
              help="Input image size (square)")
def profile_cli(model, fmt, output, input_size):
    """Profile a model and export layer-level workload characterization."""
    from rich.console import Console
    console = Console()

    console.print(f"\n[bold]Keyhole — Layer Profiler[/]")
    console.print(f"  Model: {model}, Input: {input_size}x{input_size}\n")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if model == "yolo":
        from ultralytics import YOLO
        yolo = YOLO("yolo11x.pt")
        yolo.to(device)
        # Get the underlying torch model
        torch_model = yolo.model
        sample = torch.randn(1, 3, input_size, input_size).to(device)

        profiler = LayerProfiler("yolo11x", "yolo")
        export = profiler.profile_model(torch_model, sample)

    elif model == "sam3":
        try:
            from sam3.model_builder import build_sam3_image_model
            sam3 = build_sam3_image_model()
            # Profile the image encoder portion
            torch_model = sam3  # Will hook all submodules
            sample = torch.randn(1, 3, input_size, input_size).to(device)

            profiler = LayerProfiler("sam3_full", "sam3")
            export = profiler.profile_model(torch_model, sample)
        except ImportError:
            console.print("[red]SAM 3 not installed. Install from source first.[/]")
            return

    # Determine output paths
    base = output or f"data/output/workload_{model}"

    if fmt in ("json", "all"):
        path = f"{base}.json" if not base.endswith(".json") else base
        LayerProfiler.export_json(export, path)
        console.print(f"  [green]JSON:[/] {path}")

    if fmt in ("csv", "all"):
        path = f"{base}.csv" if not base.endswith(".csv") else base
        LayerProfiler.export_csv(export, path)
        console.print(f"  [green]CSV:[/]  {path}")

    if fmt in ("hwsim", "all"):
        path = f"{base}_hwsim.json" if not base.endswith(".json") else base.replace(".json", "_hwsim.json")
        LayerProfiler.export_hardware_sim(export, path)
        console.print(f"  [green]HW Sim:[/] {path}")

    # Print summary table
    console.print(f"\n  Total: {export.total_params/1e6:.1f}M params, "
                  f"{export.total_flops/1e9:.1f} GFLOPs, "
                  f"{export.reference_latency_ms:.1f}ms on {export.reference_hardware}")

    from rich.table import Table
    table = Table(title="Category Breakdown", show_header=True, header_style="bold")
    table.add_column("Category", min_width=12)
    table.add_column("Layers", justify="right", width=8)
    table.add_column("GFLOPs", justify="right", width=10)
    table.add_column("FLOPs %", justify="right", width=8)
    table.add_column("Params", justify="right", width=12)

    for cat, stats in sorted(
        export.category_summary.items(),
        key=lambda x: x[1]["flops"],
        reverse=True,
    ):
        table.add_row(
            cat,
            str(stats["count"]),
            f"{stats['flops']/1e9:.2f}",
            f"{stats['flops_pct']:.1f}%",
            f"{stats['params']:,}",
        )
    console.print(table)


if __name__ == "__main__":
    profile_cli()
