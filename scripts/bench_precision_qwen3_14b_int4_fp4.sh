#!/usr/bin/env bash
# bench_precision_qwen3_14b_int4_fp4.sh — scale-up check of the INT4-vs-FP4 asymmetry.
#
# At 8B, on identical weights: INT4 decode ~tied with NVFP4 (both ~2.3x BF16, BW-bound)
# but INT4 prefill pinned to the BF16 floor (~13k) while NVFP4 prefill is ~3.5x that.
# This confirms it holds at 14B with the same controlled same-base head-to-head.
#
# Both checkpoints are pre-quantized (~10 GB each) and DENSE, so they load safely on a
# 32 GB 5090 and the sm_120 native FP4 path engages (no broken CUTLASS FP4-MoE, vllm#31085).
# We run only the two 4-bit formats: the BF16 floor reference is already established at 8B,
# and a 28 GB BF16-14B load would crowd the KV pool. Failure-isolated; STATUS marker.
set -uo pipefail

export CUDA_HOME=/home/kyle/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
VP="$HOME/.virtualenvs/vllm_fp4/bin"
REPO="$(cd "$(dirname "$0")/.."; pwd)"
export OUTDIR="$REPO/data/output/precision_5090_vllm_runs"
LOG="$OUTDIR/qwen3_14b_int4_fp4.log"
mkdir -p "$OUTDIR"; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# label | repo | quant (- = auto-detect from checkpoint config) | dtype
# INT4 (AWQ) is the lighter download -> runs first as the canary; NVFP4 second.
MODELS=(
  "qwen3_14b_awq_int4|Qwen/Qwen3-14B-AWQ|-|float16"
  "qwen3_14b_nvfp4|RedHatAI/Qwen3-14B-NVFP4|-|bfloat16"
)

say "=== Qwen3-14B INT4-vs-FP4 scale-up bench starting (${#MODELS[@]} runs) ==="
say "GPU at start:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG" || true

ok=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r label repo quant dtype <<<"$spec"
  say "--- $label ($repo, quant=${quant}, dtype=$dtype) ---"
  say "  downloading $repo ..."
  if ! "$VP/hf" download "$repo" >>"$LOG" 2>&1; then
    say "  DOWNLOAD FAILED ($repo) — skipping $label"; continue
  fi
  args=(--label "$label" --model "$repo" --dtype "$dtype")
  [ "$quant" != "-" ] && args+=(--quantization "$quant")
  if timeout 2400 "$VP/python" "$REPO/scripts/bench_precision_vllm.py" "${args[@]}" >>"$LOG" 2>&1; then
    say "  ok $label"; ok=$((ok+1))
  else
    say "  BENCH FAILED $label (rc=$?) — see qwen3_14b_int4_fp4.log"
  fi
done

say "=== DONE: $ok/${#MODELS[@]} produced JSON ==="
echo "DONE $ok/${#MODELS[@]}" > "$OUTDIR/STATUS_14b"
