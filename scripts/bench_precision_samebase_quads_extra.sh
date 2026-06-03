#!/usr/bin/env bash
# bench_precision_samebase_quads_extra.sh — two new same-base INT4-vs-FP4 quads:
#   A. Qwen3-32B  -> extends the "prefill split widens with scale" curve to a 3rd point
#                    (8B 3.5x -> 14B 4.4x -> 32B ?).
#   B. Llama-3.1-8B -> proves the asymmetry is ARCHITECTURE-general, not a Qwen artifact
#                    (we already have Llama-8B BF16/FP8; this adds the INT4 + NVFP4 corners).
#
# Canary order (crash lesson): the small Llama-8B pair (~5 GB each) runs FIRST so any
# harness/GPU problem surfaces cheap; the heavy Qwen3-32B pair (~18-19 GB each) runs LAST.
# All checkpoints are DENSE (sm_120 native FP4 path engages; no broken CUTLASS FP4-MoE).
# NVFP4 loader differs by source: nvidia/ modelopt checkpoints need --quantization
# modelopt_fp4; RedHatAI/ compressed-tensors NVFP4 auto-detects (same as the 14B run).
# 32B loads one-at-a-time (separate processes) so only ~19 GB is ever resident -> fits 32 GB.
# Failure-isolated (no set -e); per-model download-then-bench; STATUS marker at the end.
set -uo pipefail

export CUDA_HOME=/home/kyle/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
VP="$HOME/.virtualenvs/vllm_fp4/bin"
REPO="$(cd "$(dirname "$0")/.."; pwd)"
export OUTDIR="$REPO/data/output/precision_5090_vllm_runs"
LOG="$OUTDIR/samebase_quads_extra.log"
mkdir -p "$OUTDIR"; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# label | repo | quant (- = auto-detect from checkpoint config) | dtype
MODELS=(
  # B. Llama-3.1-8B quad (canary first — small).
  "llama8b_awq_int4|hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4|-|float16"
  "llama8b_nvfp4|nvidia/Llama-3.1-8B-Instruct-NVFP4|modelopt_fp4|bfloat16"
  # A. Qwen3-32B quad (heavy — last).
  "qwen3_32b_awq_int4|Qwen/Qwen3-32B-AWQ|-|float16"
  "qwen3_32b_nvfp4|RedHatAI/Qwen3-32B-NVFP4|-|bfloat16"
)

say "=== A+B same-base quads bench starting (${#MODELS[@]} runs) ==="
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
  if timeout 2700 "$VP/python" "$REPO/scripts/bench_precision_vllm.py" "${args[@]}" >>"$LOG" 2>&1; then
    say "  ok $label"; ok=$((ok+1))
  else
    say "  BENCH FAILED $label (rc=$?) — see samebase_quads_extra.log"
  fi
done

say "=== DONE: $ok/${#MODELS[@]} produced JSON ==="
echo "DONE $ok/${#MODELS[@]}" > "$OUTDIR/STATUS_quads_extra"
