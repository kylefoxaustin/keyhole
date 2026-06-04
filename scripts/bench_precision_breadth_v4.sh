#!/usr/bin/env bash
# bench_precision_breadth_v4.sh — NEW ARCHITECTURES: complete GLM-4-9B + Gemma-3-12B (their
# NVFP4 already measured in v3/v2; just need loadable INT4) and add Yi-1.5-34B + Nemotron-Nano-9B.
# Gemma-3 needs GPTQ (not AWQ: fp16-vs-bf16 trap) + --hf-overrides to force the text path.
set -uo pipefail
export CUDA_HOME=/home/kyle/cuda-12.9; export PATH="$CUDA_HOME/bin:$PATH"
VP="$HOME/.virtualenvs/vllm_fp4/bin"
REPO="$(cd "$(dirname "$0")/.."; pwd)"
export OUTDIR="$REPO/data/output/precision_5090_vllm_runs"
LOG="$OUTDIR/breadth_v4.log"; mkdir -p "$OUTDIR"; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
# label | repo | dtype | hf_overrides_json(optional)
MODELS=(
  "glm4_9b_int4|bean980310/glm-4-9b-chat-hf-int4|float16|"
  "nemotron_9b_int4|cyankiwi/NVIDIA-Nemotron-Nano-9B-v2-AWQ-4bit|bfloat16|"
  "nemotron_9b_nvfp4|nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4|bfloat16|"
  "gemma3_12b_int4|ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g|bfloat16|{\"architectures\": [\"Gemma3ForCausalLM\"]}"
  "yi15_34b_int4|modelscope/Yi-1.5-34B-Chat-AWQ|float16|"
  "yi15_34b_nvfp4|Firworks/dolphin-2.9.1-yi-1.5-34b-nvfp4|bfloat16|"
)
say "=== breadth-v4 starting (${#MODELS[@]} runs) ==="
ok=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r label repo dtype override <<<"$spec"
  out="$OUTDIR/${label}_vllm.json"
  [ -f "$out" ] && { say "  skip $label (exists)"; ok=$((ok+1)); continue; }
  say "--- $label ($repo, dtype=$dtype${override:+, override=$override}) ---"
  if ! "$VP/hf" download "$repo" >>"$LOG" 2>&1; then say "  DOWNLOAD FAILED $repo"; continue; fi
  args=(--label "$label" --model "$repo" --dtype "$dtype")
  [ -n "$override" ] && args+=(--hf-overrides "$override")
  if timeout 2700 "$VP/python" "$REPO/scripts/bench_precision_vllm.py" "${args[@]}" >>"$LOG" 2>&1; then
    say "  ok $label"; ok=$((ok+1))
  else say "  BENCH FAILED $label (rc=$?)"; fi
done
say "=== DONE: $ok/${#MODELS[@]} ==="; echo "DONE $ok/${#MODELS[@]}" > "$OUTDIR/STATUS_breadth_v4"
