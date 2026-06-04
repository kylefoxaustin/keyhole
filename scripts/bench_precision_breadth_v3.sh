#!/usr/bin/env bash
# bench_precision_breadth_v3.sh — extend the INT4-vs-FP4 breadth: small end (1-3B, low
# arithmetic intensity -> split should SHRINK, extending the scale curve down), a new arch
# (GLM-4), and a 2nd 32B. Same harness as breadth_v2, canary-first, failure-isolated.
set -uo pipefail
export CUDA_HOME=/home/kyle/cuda-12.9; export PATH="$CUDA_HOME/bin:$PATH"
VP="$HOME/.virtualenvs/vllm_fp4/bin"
REPO="$(cd "$(dirname "$0")/.."; pwd)"
export OUTDIR="$REPO/data/output/precision_5090_vllm_runs"
LOG="$OUTDIR/breadth_v3.log"; mkdir -p "$OUTDIR"; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
MODELS=(
  "llama32_1b_int4|AMead10/Llama-3.2-1B-Instruct-AWQ|float16"
  "llama32_1b_nvfp4|Yi30/Llama-3.2-1B-Instruct-NVFP4-OFFLINE|bfloat16"
  "dsqwen1p5b_int4|jakiAJK/DeepSeek-R1-Distill-Qwen-1.5B_GPTQ-int4|float16"
  "dsqwen1p5b_nvfp4|wefstratis-nv/DeepSeek-R1-Distill-Qwen-1.5B-NVFP4|bfloat16"
  "smollm3_3b_int4|drawais/SmolLM3-3B-AWQ-INT4|float16"
  "smollm3_3b_nvfp4|Firworks/SmolLM3-3B-nvfp4|bfloat16"
  "glm4_9b_int4|model-scope/glm-4-9b-chat-GPTQ-Int4|float16"
  "glm4_9b_nvfp4|GizzmoShifu/glm-4-9b-chat-hf-nvfp4|bfloat16"
  "dsqwen32b_int4|casperhansen/deepseek-r1-distill-qwen-32b-awq|float16"
  "dsqwen32b_nvfp4|vipertsniper/DeepSeek-R1-Distill-Qwen-32B-NVFP4|bfloat16"
)
say "=== breadth-v3 starting (${#MODELS[@]} runs, 5 models) ==="
ok=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r label repo dtype <<<"$spec"
  out="$OUTDIR/${label}_vllm.json"
  [ -f "$out" ] && { say "  skip $label (exists)"; ok=$((ok+1)); continue; }
  say "--- $label ($repo, dtype=$dtype) ---"
  if ! "$VP/hf" download "$repo" >>"$LOG" 2>&1; then say "  DOWNLOAD FAILED $repo"; continue; fi
  if timeout 2400 "$VP/python" "$REPO/scripts/bench_precision_vllm.py" \
        --label "$label" --model "$repo" --dtype "$dtype" >>"$LOG" 2>&1; then
    say "  ok $label"; ok=$((ok+1))
  else say "  BENCH FAILED $label (rc=$?)"; fi
done
say "=== DONE: $ok/${#MODELS[@]} ==="; echo "DONE $ok/${#MODELS[@]}" > "$OUTDIR/STATUS_breadth_v3"
