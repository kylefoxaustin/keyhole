#!/usr/bin/env bash
# bench_precision_5090.sh — INT (Q4_K_M) vs FP4 (NVFP4) on the RTX 5090 via llama-bench.
#
# For each GGUF: prefill (pp2048) + decode (tg256) tok/s via llama-bench (JSON),
# plus peak VRAM sampled from nvidia-smi during a short generation. Emits one JSON
# row per (model,quant) to data/output/precision_5090_runs/.
#
# Usage: scripts/bench_precision_5090.sh <label> <gguf_path> [<label2> <gguf2> ...]
set -euo pipefail

LLAMA_BENCH="$HOME/.personal-ai/llama.cpp/build-cuda/bin/llama-bench"
LLAMA_CLI="$HOME/.personal-ai/llama.cpp/build-cuda/bin/llama-cli"
OUTDIR="$(cd "$(dirname "$0")/.."; pwd)/data/output/precision_5090_runs"
mkdir -p "$OUTDIR"

bench_one() {
  local label="$1" gguf="$2"
  echo "=== $label : $gguf ==="
  [ -f "$gguf" ] || { echo "MISSING: $gguf"; return 1; }

  # Sample VRAM ONLY during the (bounded) llama-bench run — no open-ended llama-cli.
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.2; done ) \
    > "$OUTDIR/${label}_vram.samples" &
  local sampler=$!

  # llama-bench: prefill (pp 2048) and decode (tg 256), 3 reps, all layers on GPU.
  # Hard timeout guard so a misbehaving run can never wedge the harness.
  timeout 900 "$LLAMA_BENCH" -m "$gguf" -ngl 99 -p 2048 -n 256 -r 3 -o json \
    > "$OUTDIR/${label}_bench.json" 2> "$OUTDIR/${label}_bench.err"
  local rc=$?

  kill "$sampler" 2>/dev/null || true
  local peak
  peak=$(sort -n "$OUTDIR/${label}_vram.samples" | tail -1)
  echo "$peak" > "$OUTDIR/${label}_peak_vram_mib.txt"
  [ "$rc" -eq 0 ] || { echo "  llama-bench rc=$rc; see ${label}_bench.err"; return 1; }
  echo "  peak VRAM during bench: ${peak} MiB  ·  file: $(du -h "$gguf" | cut -f1)"
}

while [ $# -ge 2 ]; do
  bench_one "$1" "$2"
  shift 2
done
echo "done -> $OUTDIR"
