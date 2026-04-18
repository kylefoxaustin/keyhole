"""
NVTX helpers — tiny context manager for labeling kernels by model/stage.

The bake-off scripts wrap their model forward passes with these so that when
the scripts are run under `ncu` (Nsight Compute) or `nsys` (Nsight Systems),
every kernel inside the range gets attributed to the named stage. This gives
us per-model instruction / FLOP / DRAM-byte counts for the platform budget
(YOLO vs SAM3 vs LLM breakdowns the platform-engineer wants).

Safe to call in any environment — if torch.cuda.nvtx isn't importable (CPU-only
install, non-CUDA runtime) the context manager is a no-op. No profiler is
required to be attached; the NVTX ranges sit in the app until something reads
them.

Usage:
    from src.profiling.nvtx_helpers import nvtx_range

    with nvtx_range("yolo_seg_fp8_trt"):
        ctx.execute_async_v3(stream.cuda_stream)
"""
from __future__ import annotations

import contextlib

try:
    import torch
    _HAS_NVTX = torch.cuda.is_available() and hasattr(torch.cuda, "nvtx")
except Exception:
    _HAS_NVTX = False


@contextlib.contextmanager
def nvtx_range(name: str):
    """Mark a block of code with an NVTX range so Nsight profilers attribute
    kernels to `name`. Safe to use even without a profiler attached."""
    if _HAS_NVTX:
        try:
            torch.cuda.nvtx.range_push(name)
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def push(name: str) -> None:
    """Imperative API — push an NVTX range. Must be paired with pop()."""
    if _HAS_NVTX:
        torch.cuda.nvtx.range_push(name)


def pop() -> None:
    """Imperative API — pop the current NVTX range."""
    if _HAS_NVTX:
        torch.cuda.nvtx.range_pop()
