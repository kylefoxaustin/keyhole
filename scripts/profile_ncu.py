"""
Nsight Compute wrapper for the Keyhole bake-offs.

Runs any command under `ncu` with a targeted metric set, then parses the CSV
output and emits a per-NVTX-range JSON summary. The JSON schema matches what
`keyhole-sizer/sizer/platform_budget.py` can consume to replace the approximated
TOPS / DDR-GB/s columns with MEASURED values.

Key design decisions:
  - Targeted metric list (not `--set detailed`) keeps overhead to ~2-5x (not 50x).
  - `--nvtx` + `--nvtx-include` captures the NVTX ranges that the bake-offs push,
    so per-model (YOLO vs SAM3 vs LLM) attribution works without kernel-name
    mangling.
  - Aggregates ALL kernels within each NVTX range — we don't care about per-kernel
    detail, the platform budget just needs per-stage totals.

Metric set (platform-budget relevant only):
  smsp__inst_executed.sum         — total SM-partition instructions executed
  sm__sass_thread_inst_executed.sum — SASS thread instruction count (FLOP proxy)
  sm__inst_executed_pipe_tensor.sum — Tensor Core ops (counts hmma/bmma/fmma)
  dram__bytes_read.sum            — DRAM reads (bytes)
  dram__bytes_write.sum           — DRAM writes (bytes)
  dram__bytes.sum                 — total DRAM traffic (bytes)
  gpu__time_duration.sum          — cumulative kernel GPU time (NOT wall-clock!)

Usage:
    # Profile a trt_yolo bake-off run:
    python scripts/profile_ncu.py \
        --out data/output/ncu/trt_yolo.json \
        -- \
        python scripts/bakeoff_trt_yolo.py --clip data/videos/720p_EW_clip.mp4

    # Just the decode step of the LLM bake-off:
    python scripts/profile_ncu.py \
        --out data/output/ncu/llm_decode.json \
        --nvtx-include 'regex:llm_decode' \
        -- \
        python scripts/bakeoff_llm.py --quants Q4_K_M

Everything after the `--` is the command to profile.

Notes:
  - The command being profiled MUST have NVTX ranges pushed via
    `src.profiling.nvtx_helpers.nvtx_range(name)` for per-stage attribution.
  - Wall-clock timings recorded DURING ncu are invalid (inflated by profiling
    overhead). Use the bake-off's normal non-profiled output for latency.
  - ncu needs exclusive GPU access. Stop any other CUDA workloads first
    (the LLM server often holds 24 GB — check `nvidia-smi`).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ncu 2026+ emits the NVTX Push/Pop payload as a colon-delimited structure
# prefixed by the PID and wrapped in its own quotes, e.g.
#   '97844  "<default domain>:warmup:none:none:none:none:none:none"'
# Extract the label sitting between the domain and the first PL_Type field.
_NVTX_LABEL_RE = re.compile(r"<[^>]+>:([^:]+):")

# ncu metric names look like 'smsp__inst_executed.sum' or
# 'sm__inst_executed_pipe_tensor.sum' — snake_case with dots, no whitespace
# or quotes. Reject anything else (e.g. diagnostic JSON fragments).
_VALID_METRIC_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


def _extract_nvtx_label(cell: str) -> str:
    """Pull the NVTX range label out of an ncu CSV Push/Pop_Range cell."""
    if not cell:
        return ""
    m = _NVTX_LABEL_RE.search(cell)
    return m.group(1).strip() if m else ""


# Targeted metric list — keeps ncu overhead to ~2-5x. Enough for the
# platform-budget spreadsheet; skip the warp-efficiency / cache-miss stuff.
METRICS = [
    "smsp__inst_executed.sum",
    "sm__sass_thread_inst_executed.sum",
    "sm__inst_executed_pipe_tensor.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__bytes.sum",
    "gpu__time_duration.sum",
]


def _find_ncu() -> str:
    """Locate the ncu binary. Prefer newest available; fall back to older ones.

    The CUDA 13 driver (580.x) requires Nsight Compute 2025+ — older ncu builds
    fail with 'Failed to load Nsight Compute CUDA modules'. Install newer ncu
    via `sudo apt-get install nsight-compute-2026.1.1` (or newer).
    """
    candidates = [
        # 2026+ standalone installs (preferred on CUDA 13 driver)
        "/opt/nvidia/nsight-compute/2026.1.1/ncu",
        "/opt/nvidia/nsight-compute/2026.1.0/ncu",
        "/opt/nvidia/nsight-compute/2025.4.1/ncu",
        "/opt/nvidia/nsight-compute/2025.4.0/ncu",
        "/opt/nvidia/nsight-compute/2025.3.1/ncu",
        # Whatever's on PATH
        "ncu",
        # CUDA-toolkit-bundled fallbacks (only work on matching driver)
        "/usr/local/cuda/bin/ncu",
        "/usr/local/cuda-13.0/bin/ncu",
        "/usr/local/cuda-12.6/bin/ncu",
    ]
    for candidate in candidates:
        try:
            r = subprocess.run([candidate, "--version"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise SystemExit(
        "ncu (Nsight Compute) not found. Install with "
        "`sudo apt-get install -y nsight-compute-2026.1.1` (or newer)."
    )


def _run_ncu(cmd: list[str], csv_path: Path, nvtx_include: str | None = None) -> None:
    ncu = _find_ncu()
    # Replay strategy:
    #   - 'application' (default) with --app-replay-mode relaxed: ncu re-runs
    #     the whole app once per metric group and matches kernels by name
    #     across passes, tolerating variation in count/order (TRT autotuning,
    #     NMS dynamic sizes). Fast (~app wall-clock × N_metric_groups).
    #   - 'kernel' (override via KEYHOLE_NCU_REPLAY=kernel): save/restore GPU
    #     state per-kernel, immune to non-determinism but 10-20x slower.
    replay_mode = os.environ.get("KEYHOLE_NCU_REPLAY", "application")
    ncu_cmd = [
        ncu,
        "--csv",
        "--log-file", str(csv_path),
        "--target-processes", "application-only",
        "--nvtx",
        "--metrics", ",".join(METRICS),
        "--replay-mode", replay_mode,
        "--kernel-name-base", "function",  # use demangled function names
        "--print-summary", "none",         # we parse the CSV ourselves
    ]
    if replay_mode == "application":
        # 'relaxed' drops unmatched kernels across passes; 'name' matches by
        # kernel name alone (loosest — tolerates grid/block variation from
        # NMS dynamic output sizes, TRT autotune, etc).
        ncu_cmd += ["--app-replay-mode", "relaxed", "--app-replay-match", "name"]
    if nvtx_include:
        ncu_cmd += ["--nvtx-include", nvtx_include]
    ncu_cmd += cmd

    print(f"[profile_ncu] Launching: {' '.join(shlex.quote(c) for c in ncu_cmd)}",
          file=sys.stderr)
    r = subprocess.run(ncu_cmd)
    if r.returncode != 0:
        raise SystemExit(
            f"ncu exited with code {r.returncode}. Check that the target command "
            f"runs cleanly outside ncu first."
        )


def _parse_csv(csv_path: Path) -> dict:
    """Parse ncu CSV output. Aggregates metrics by NVTX range name (summed
    across all kernels inside that range)."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise SystemExit(f"ncu produced no CSV at {csv_path}")

    # ncu CSV format (as of Nsight Compute 2024+):
    # skip any leading comment lines starting with '==', then a header row,
    # then one row per (kernel invocation × metric).
    with csv_path.open() as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("==")]

    if not lines:
        raise SystemExit("CSV body empty after header-strip.")

    reader = csv.DictReader(lines)
    # Different ncu versions use slightly different column names. Probe:
    cols = reader.fieldnames or []
    # NVTX columns in ncu 2026+ look like
    #   'thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg'
    #   'Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg'
    # Prefer Push/Pop (torch.cuda.nvtx.range_push/pop uses it); fall back to
    # Start/Stop. Older ncu versions used a column starting with 'NVTX'.
    def _find_nvtx(pattern: str) -> str | None:
        for c in cols:
            if pattern.lower() in c.lower():
                return c
        return None
    nvtx_pp_col = _find_nvtx("push/pop_range")
    nvtx_ss_col = _find_nvtx("start/stop_range")
    nvtx_legacy_col = next((c for c in cols if c.lower().startswith("nvtx")), None)
    kernel_col = next((c for c in cols if "kernel" in c.lower() and "name" in c.lower()), None)
    metric_name_col = next((c for c in cols if c.lower() in ("metric name", "metricname")), None)
    metric_value_col = next((c for c in cols if c.lower() in ("metric value", "metricvalue")), None)
    metric_unit_col = next((c for c in cols if c.lower() in ("metric unit", "metricunit")), None)

    if not (metric_name_col and metric_value_col):
        raise SystemExit(
            f"Couldn't find Metric Name/Value columns in ncu CSV. Got: {cols}"
        )

    # by_range[nvtx_label][metric_name] = summed value
    by_range: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_range_units: dict[str, dict[str, str]] = defaultdict(dict)
    kernel_count: dict[str, int] = defaultdict(int)
    prev_kernel_key = None

    for row in reader:
        label = ""
        if nvtx_pp_col:
            label = _extract_nvtx_label(row.get(nvtx_pp_col, "") or "")
        if not label and nvtx_ss_col:
            label = _extract_nvtx_label(row.get(nvtx_ss_col, "") or "")
        if not label and nvtx_legacy_col:
            label = (row.get(nvtx_legacy_col, "") or "").strip()
        if not label:
            label = "[unattributed]"
        metric = row[metric_name_col].strip()
        # ncu interleaves per-kernel error/diagnostic rows (e.g. INT8 TRT
        # kernels that hit instrumentation errors) and the CSV parser sees
        # their embedded JSON fragments as metric names. Skip any metric
        # name that isn't a plausible ncu metric identifier.
        if not _VALID_METRIC_RE.match(metric):
            continue
        raw_val = (row[metric_value_col] or "0").replace(",", "").strip()
        try:
            value = float(raw_val)
        except ValueError:
            continue
        by_range[label][metric] += value
        if metric_unit_col:
            by_range_units[label][metric] = (row.get(metric_unit_col, "") or "").strip()
        kernel_key = (label, row.get(kernel_col or "", ""))
        if kernel_key != prev_kernel_key:
            kernel_count[label] += 1
            prev_kernel_key = kernel_key

    return {
        "by_range": {
            label: {
                "metrics": dict(metrics),
                "units": by_range_units.get(label, {}),
                "n_kernel_invocations": kernel_count.get(label, 0),
            } for label, metrics in by_range.items()
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run a command under ncu with targeted metrics; emit per-NVTX-range JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path")
    ap.add_argument("--nvtx-include", default=None,
                    help="Only profile kernels inside matching NVTX ranges "
                         "(see ncu --nvtx-include docs; e.g. 'regex:llm_decode')")
    ap.add_argument("--keep-csv", action="store_true",
                    help="Don't delete the intermediate CSV (debugging)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="Command to profile (precede with `--`)")
    args = ap.parse_args()

    if not args.cmd:
        ap.error("Missing command to profile. Precede with `--`: "
                 "profile_ncu.py --out X.json -- python scripts/bakeoff_XXX.py")
    if args.cmd[0] == "--":
        args.cmd = args.cmd[1:]

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        csv_path = Path(tmp.name)

    try:
        _run_ncu(args.cmd, csv_path, nvtx_include=args.nvtx_include)
        parsed = _parse_csv(csv_path)
    finally:
        if not args.keep_csv:
            csv_path.unlink(missing_ok=True)
        else:
            print(f"[profile_ncu] Kept CSV at {csv_path}", file=sys.stderr)

    # Wrap with provenance
    out = {
        "profiler": "Nsight Compute (ncu)",
        "ncu_binary": _find_ncu(),
        "command": args.cmd,
        "nvtx_include_filter": args.nvtx_include,
        "metrics_requested": METRICS,
        "export_timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **parsed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    # Under sudo, the process runs as root and would leave files root-owned,
    # blocking subsequent user-owned rewrites. Hand ownership back to the
    # invoking user.
    sudo_user = os.environ.get("SUDO_USER")
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if sudo_uid and sudo_gid and os.geteuid() == 0:
        try:
            os.chown(args.out, int(sudo_uid), int(sudo_gid))
        except (OSError, ValueError):
            pass
    print(f"[profile_ncu] Wrote {args.out}", file=sys.stderr)

    # Human summary
    print("\n=== ncu summary (sum across kernels per NVTX range) ===", file=sys.stderr)
    print(f"{'Range':<40} {'n_kern':>8} {'inst':>14} {'tensor':>14} {'DRAM GB':>10}",
          file=sys.stderr)
    for label, info in sorted(parsed["by_range"].items()):
        m = info["metrics"]
        n_kern = info["n_kernel_invocations"]
        inst = m.get("smsp__inst_executed.sum", 0)
        tensor = m.get("sm__inst_executed_pipe_tensor.sum", 0)
        dram = (m.get("dram__bytes.sum", 0)) / 1e9
        print(f"{label[:40]:<40} {n_kern:>8d} {inst:>14.3e} {tensor:>14.3e} {dram:>10.3f}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
