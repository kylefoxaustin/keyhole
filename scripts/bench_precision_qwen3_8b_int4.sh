#!/usr/bin/env bash
# bench_precision_qwen3_8b_int4.sh — fill the ONE gap in the same-base INT4-vs-FP4 quad.
#
# Tier-1 already measured Qwen3-8B in NVFP4 / FP8 / BF16 on vLLM (single-stream prefill+
# decode). The missing leg is the INT4 *memory format* on the SAME base so slide-7's
# asymmetry is reproduced on identical weights:
#   INT4 (AWQ) decode should be high (BW-bound win) BUT prefill should sit near the
#   ~13k tok/s BF16 floor (weights dequantize to bf16 → no compute win), whereas NVFP4
#   wins BOTH (decode 217 / prefill ~48k) via native sm_120 FP4 tensor cores.
#
# DENSE model on purpose: MoE NVFP4 hits the broken CUTLASS FP4-MoE path on sm_120
# (vllm#31085). Single small (6.1 GB) checkpoint — no heavyweight download, no crash risk.
# Failure-isolated (no set -e); logged; STATUS marker on completion.
set -uo pipefail

export CUDA_HOME=/home/kyle/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
VP="$HOME/.virtualenvs/vllm_fp4/bin"
REPO="$(cd "$(dirname "$0")/.."; pwd)"
export OUTDIR="$REPO/data/output/precision_5090_vllm_runs"
LOG="$OUTDIR/qwen3_8b_int4.log"
mkdir -p "$OUTDIR"; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

AWQ_REPO="Qwen/Qwen3-8B-AWQ"
LABEL="qwen3_8b_awq_int4"

say "=== Qwen3-8B AWQ-INT4 bench starting (completes the same-base INT4-vs-FP4 quad) ==="
say "GPU at start:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG" || true

say "downloading $AWQ_REPO ..."
if "$VP/hf" download "$AWQ_REPO" >>"$LOG" 2>&1; then
  say "  download ok"
else
  say "  DOWNLOAD FAILED — aborting"; echo "FAIL download" > "$OUTDIR/STATUS_int4"; exit 1
fi

# AWQ checkpoint auto-detects awq_marlin in vLLM; float16 is the AWQ-native compute dtype.
say "--- bench: $LABEL ($AWQ_REPO, awq_marlin auto, dtype=float16) ---"
if timeout 1800 "$VP/python" "$REPO/scripts/bench_precision_vllm.py" \
      --label "$LABEL" --model "$AWQ_REPO" --dtype float16 >>"$LOG" 2>&1; then
  say "  ok: $LABEL"
  echo "DONE $LABEL" > "$OUTDIR/STATUS_int4"
else
  say "  BENCH FAILED: $LABEL (rc=$?) — see qwen3_8b_int4.log"
  echo "FAIL bench" > "$OUTDIR/STATUS_int4"; exit 1
fi

say "=== DONE — compare against nvfp4/fp8/bf16_vllm.json for the four-way ==="
