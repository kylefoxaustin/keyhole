#!/usr/bin/env bash
# probe_w4a8_5090.sh — empirically test whether W4A8 (4-bit weight + 8-bit activation,
# the textbook MIXED-INT/FP "escape the prefill floor" scheme) actually runs on the
# RTX 5090 (sm_120) via vLLM. Research says NO usable sm_120 kernel exists (QQQ removed
# after v0.10; compressed-tensors W4A8-int finds no Blackwell kernel; W4A8-fp8 is
# Hopper/SM90-only CUTLASS). This converts that claim into a MEASURED fact on this box:
# for each downloadable W4A8 checkpoint, capture whether vLLM (a) refuses with a
# no-kernel/capability error, or (b) loads but falls back to the W4A16 bf16 prefill floor.
#
# TINY models first (1B -> 8B) so there is zero OOM/crash risk (crash lesson: heavyweight
# downloads can take the box down). Failure-isolated (no set -e); each outcome -> a
# .result file with the error signature; JSON written only if a model actually loads+runs.
set -uo pipefail

export CUDA_HOME=/home/kyle/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
VP="$HOME/.virtualenvs/vllm_fp4/bin"
REPO="$(cd "$(dirname "$0")/.."; pwd)"
export OUTDIR="$REPO/data/output/precision_5090_vllm_runs"
LOG="$OUTDIR/w4a8_probe.log"
mkdir -p "$OUTDIR"; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# label | repo | scheme-note   (compressed-tensors auto-detects; pass it explicitly)
MODELS=(
  "w4a8int_llama1b|zera09/Llama-3.2-1B-Instruct-W4A8-GPTQ|W4A8-int (int4 w + int8 act)"
  "w4a8fp8_tiny1b|czhu-cohere/TinyLlama-1.1B-Chat-v1.0-W4A8-e2e|W4A8-fp8 (int4 w + fp8 act)"
  "w4a8int_gemma4b|nm-testing/gemma-3-4b-it-s_q-W4A8-G512|W4A8-int (int4 w + int8 act)"
  "w4a8fp8_llama8b|czhu-cohere/Meta-Llama-3-8B-Instruct-W4A8-compressed-tensors-test|W4A8-fp8 (int4 w + fp8 act)"
)

say "=== W4A8 sm_120 probe starting (${#MODELS[@]} checkpoints, tiny->small) ==="
say "vLLM:"; "$VP/python" -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,'cap',torch.cuda.get_device_capability(0))" 2>&1 | tee -a "$LOG"

loaded=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r label repo note <<<"$spec"
  RES="$OUTDIR/${label}.result"
  say "--- $label ($repo) :: $note ---"
  say "  downloading $repo ..."
  if ! "$VP/hf" download "$repo" >>"$LOG" 2>&1; then
    say "  DOWNLOAD FAILED ($repo)"; echo "DOWNLOAD_FAILED $repo" > "$RES"; continue
  fi
  # Per-model log so we can extract its exact error signature.
  MLOG="$OUTDIR/${label}_run.log"; : > "$MLOG"
  say "  loading+benching via bench_precision_vllm.py (quant=compressed-tensors) ..."
  if timeout 1200 "$VP/python" "$REPO/scripts/bench_precision_vllm.py" \
        --label "$label" --model "$repo" --quantization compressed-tensors --dtype auto \
        >>"$MLOG" 2>&1; then
    loaded=$((loaded+1))
    say "  LOADED+RAN ok -> ${label}_vllm.json"
    echo "LOADED_AND_RAN $repo" > "$RES"
  else
    rc=$?
    # Extract the most telling error line(s): kernel/capability/support failures.
    sig=$(grep -hiE "failed to find a kernel|no kernel image|compute capability|min_capability|not supported|requires compute|unknown quantization|ValueError|RuntimeError|Marlin.*will be used|degrade performance|does not support" "$MLOG" | tail -6)
    say "  FAILED (rc=$rc). signature:"; echo "$sig" | tee -a "$LOG"
    { echo "FAILED rc=$rc $repo"; echo "--- signature ---"; echo "$sig"; } > "$RES"
  fi
done

say "=== DONE: $loaded/${#MODELS[@]} W4A8 checkpoints actually loaded+ran on sm_120 ==="
echo "DONE loaded=$loaded/${#MODELS[@]}" > "$OUTDIR/STATUS_w4a8"
