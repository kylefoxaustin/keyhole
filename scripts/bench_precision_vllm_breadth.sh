#!/usr/bin/env bash
# bench_precision_vllm_breadth.sh — harden the precision-migration transitions with more
# models/formats on the vLLM harness (single-stream prefill+decode, same methodology as Tier-1).
#
# Batches: A gpt-oss MXFP4 (2nd FP4 format) · B FP8 arch breadth · C INT4(AWQ/GPTQ) vs FP4 ·
#          D DeepSeek-V2-Lite FP8 (the 2028 driver).
# Download-then-bench per model so early results land while later models download.
# Failure-isolated (no set -e); CUDA_HOME=12.9 set globally for any FP4/MXFP4 JIT.
set -uo pipefail

export CUDA_HOME=/home/kyle/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
VP="$HOME/.virtualenvs/vllm_fp4/bin"
REPO="$(cd "$(dirname "$0")/.."; pwd)"
export OUTDIR="$REPO/data/output/precision_5090_vllm_runs"
LOG="$OUTDIR/breadth.log"
mkdir -p "$OUTDIR"; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# label | hf_repo | quantization (- = auto-detect from checkpoint) | dtype
# Ordered small/fast -> heavy: llama8b_fp8 is the canary (validates harness + GPU
# stability cheaply). The two heavyweights — gpt-oss-20b and the DeepSeek-V2-Lite
# MoE — run LAST so a crash surfaces early on a cheap model, not after a 20B download.
# Note: same-repo runs (llama bf16/fp8, mistral bf16/fp8) reuse the cached download.
MODELS=(
  "llama8b_fp8|meta-llama/Llama-3.1-8B-Instruct|fp8|bfloat16"
  "llama8b_bf16|meta-llama/Llama-3.1-8B-Instruct|-|bfloat16"
  "mistral7b_fp8|mistralai/Mistral-7B-Instruct-v0.3|fp8|bfloat16"
  "mistral7b_bf16|mistralai/Mistral-7B-Instruct-v0.3|-|bfloat16"
  "phi4_fp8|microsoft/phi-4|fp8|bfloat16"
  "qwen7b_awq_int4|Qwen/Qwen2.5-7B-Instruct-AWQ|-|float16"
  "qwen7b_gptq_int4|Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4|-|float16"
  "gptoss20b_mxfp4|openai/gpt-oss-20b|-|bfloat16"
  "deepseekv2lite_fp8|deepseek-ai/DeepSeek-V2-Lite-Chat|fp8|bfloat16"
)

say "=== precision breadth run starting (${#MODELS[@]} runs) ==="
declare -A DOWNLOADED   # cache downloads across runs that share a repo

for spec in "${MODELS[@]}"; do
  IFS='|' read -r label repo quant dtype <<<"$spec"
  say "--- $label  ($repo, quant=${quant}, dtype=$dtype) ---"

  if [ -z "${DOWNLOADED[$repo]:-}" ]; then
    say "  downloading $repo ..."
    if "$VP/hf" download "$repo" >>"$LOG" 2>&1; then DOWNLOADED[$repo]=1; say "    download ok";
    else say "    DOWNLOAD FAILED ($repo) — skipping $label and any sibling runs"; DOWNLOADED[$repo]=0; fi
  fi
  if [ "${DOWNLOADED[$repo]}" = "0" ]; then say "  skip $label (download failed)"; continue; fi

  args=(--label "$label" --model "$repo" --dtype "$dtype")
  [ "$quant" != "-" ] && args+=(--quantization "$quant")
  if timeout 2400 "$VP/python" "$REPO/scripts/bench_precision_vllm.py" "${args[@]}" >>"$LOG" 2>&1; then
    say "  ok $label"
  else
    say "  BENCH FAILED $label (rc=$?) — see breadth.log"
  fi
done

n=$(ls "$OUTDIR"/{gptoss20b_mxfp4,llama8b_bf16,llama8b_fp8,mistral7b_bf16,mistral7b_fp8,phi4_fp8,qwen7b_awq_int4,qwen7b_gptq_int4,deepseekv2lite_fp8}_vllm.json 2>/dev/null | wc -l)
say "=== DONE: $n/${#MODELS[@]} breadth runs produced JSON ==="
echo "DONE $n/${#MODELS[@]}" > "$OUTDIR/BREADTH_STATUS"
