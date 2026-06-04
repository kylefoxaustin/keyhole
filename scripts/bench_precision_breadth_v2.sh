#!/usr/bin/env bash
# bench_precision_breadth_v2.sh — BREADTH expansion of the INT4-vs-FP4 asymmetry across
# more architectures/sizes (breadth IS the result: more same-regime points harden the
# "architecture-general" + scale claims). 7 models x {INT4, NVFP4}, same-base head-to-head.
#
# Each model: a downloadable INT4 (AWQ/GPTQ) + NVFP4 build of the SAME base, dense, fits a
# 32GB 5090. Auto-detect quant from each checkpoint's config (awq/gptq marlin; compressed-
# tensors/modelopt NVFP4). The prefill number reveals real W4A4 (clears bf16 floor = FP4
# compute) vs weight-only W4A16 (stuck on floor like INT4).
#
# Canary-first ordering (crash lesson): smallest model FIRST, heaviest (Mistral-24B) LAST,
# download-then-bench per model so a crash surfaces cheap. Failure-isolated (no set -e):
# a bad community quant just fails + is skipped. STATUS marker at the end.
set -uo pipefail

export CUDA_HOME=/home/kyle/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
VP="$HOME/.virtualenvs/vllm_fp4/bin"
REPO="$(cd "$(dirname "$0")/.."; pwd)"
export OUTDIR="$REPO/data/output/precision_5090_vllm_runs"
LOG="$OUTDIR/breadth_v2.log"
mkdir -p "$OUTDIR"; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# label | repo | dtype   (quant auto-detected from checkpoint config). Small -> large.
MODELS=(
  "gemma3_4b_int4|gaunernst/gemma-3-4b-it-int4-awq|float16"
  "gemma3_4b_nvfp4|holtmann/gemma-3-4b-it-NVFP4|bfloat16"
  "mistral7b_v03_int4|solidrust/Mistral-7B-Instruct-v0.3-AWQ|float16"
  "mistral7b_v03_nvfp4|llmat/Mistral-7B-Instruct-v0.3-NVFP4|bfloat16"
  "dsllama8b_int4|casperhansen/deepseek-r1-distill-llama-8b-awq|float16"
  "dsllama8b_nvfp4|DevQuasar/DeepSeek-R1-Distill-Llama-8B_nvfp4|bfloat16"
  "gemma3_12b_int4|gaunernst/gemma-3-12b-it-int4-awq|float16"
  "gemma3_12b_nvfp4|holtmann/gemma-3-12b-it-NVFP4|bfloat16"
  "dsqwen14b_int4|casperhansen/deepseek-r1-distill-qwen-14b-awq|float16"
  "dsqwen14b_nvfp4|vipertsniper/DeepSeek-R1-Distill-Qwen-14B-NVFP4|bfloat16"
  "phi4rp_int4|steadyflow/Phi-4-reasoning-plus-AWQ|float16"
  "phi4rp_nvfp4|nvidia/Phi-4-reasoning-plus-NVFP4|bfloat16"
  "mistral24b_int4|jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym|float16"
  "mistral24b_nvfp4|RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4|bfloat16"
)

say "=== breadth-v2 INT4-vs-NVFP4 sweep starting (${#MODELS[@]} runs, 7 models) ==="
say "GPU:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG" || true

ok=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r label repo dtype <<<"$spec"
  out="$OUTDIR/${label}_vllm.json"
  if [ -f "$out" ]; then say "  skip $label (json exists)"; ok=$((ok+1)); continue; fi
  say "--- $label ($repo, dtype=$dtype, quant=auto) ---"
  if ! "$VP/hf" download "$repo" >>"$LOG" 2>&1; then
    say "  DOWNLOAD FAILED ($repo) — skipping $label"; continue
  fi
  if timeout 2400 "$VP/python" "$REPO/scripts/bench_precision_vllm.py" \
        --label "$label" --model "$repo" --dtype "$dtype" >>"$LOG" 2>&1; then
    say "  ok $label"; ok=$((ok+1))
  else
    say "  BENCH FAILED $label (rc=$?) — see breadth_v2.log"
  fi
done

say "=== DONE: $ok/${#MODELS[@]} produced JSON ==="
echo "DONE $ok/${#MODELS[@]}" > "$OUTDIR/STATUS_breadth_v2"
