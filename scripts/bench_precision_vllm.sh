#!/usr/bin/env bash
# bench_precision_vllm.sh — overnight Tier-1 run: NVFP4 vs FP8 vs BF16 for Qwen3-8B on
# the RTX 5090 via vLLM (the mature-FP4-kernel runtime the llama.cpp finding flagged as
# missing). Downloads the two needed checkpoints, then benches all three precisions.
#
# Isolated per-model (no `set -e`): one precision failing does NOT abort the others.
# Everything logged to data/output/precision_5090_vllm_runs/run.log; DONE/FAIL marker
# written at the end so the orchestrator knows the outcome on re-invocation.
set -uo pipefail

VP="$HOME/.virtualenvs/vllm_fp4/bin"
REPO="$(cd "$(dirname "$0")/.."; pwd)"
export OUTDIR="$REPO/data/output/precision_5090_vllm_runs"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
: > "$LOG"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

BF16_REPO="Qwen/Qwen3-8B"
NVFP4_REPO="nvidia/Qwen3-8B-NVFP4"

say "=== Tier-1 vLLM precision bench starting ==="
say "GPU at start:"; "$VP/../bin/python" - <<'PY' 2>&1 | tee -a "$LOG" || true
import subprocess; print(subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
PY

# --- downloads (idempotent; hf download is a no-op if already complete) ---
for repo in "$NVFP4_REPO" "$BF16_REPO"; do
  say "downloading $repo ..."
  if "$VP/hf" download "$repo" >>"$LOG" 2>&1; then
    say "  ok: $repo"
  else
    say "  DOWNLOAD FAILED: $repo (see run.log) — continuing; dependent benches will be skipped"
  fi
done

# record weight footprint from disk (vLLM runtime VRAM is not a footprint signal)
say "checkpoint footprints:"
"$VP/python" - <<'PY' 2>&1 | tee -a "$LOG" || true
import os, glob, json
HUB=os.path.expanduser("~/.cache/huggingface/hub")
def sz(repo):
    d=os.path.join(HUB,"models--"+repo.replace("/","--"),"snapshots")
    tot=0
    for f in glob.glob(d+"/*/*"):
        try: tot+=os.path.getsize(os.path.realpath(f))
        except OSError: pass
    return round(tot/1e9,2)
out={r: sz(r) for r in ["nvidia/Qwen3-8B-NVFP4","Qwen/Qwen3-8B"]}
print(json.dumps(out))
json.dump(out, open(os.path.join(os.environ["OUTDIR"],"footprints_gb.json"),"w"), indent=2)
PY

run_one() {
  local label="$1"; shift
  say "--- bench: $label ($*) ---"
  if timeout 1800 "$VP/python" "$REPO/scripts/bench_precision_vllm.py" --label "$label" "$@" >>"$LOG" 2>&1; then
    say "  ok: $label"
  else
    say "  BENCH FAILED: $label (rc=$?) — see run.log"
  fi
}

run_one nvfp4 --model "$NVFP4_REPO" --quantization modelopt_fp4
run_one fp8   --model "$BF16_REPO"  --quantization fp8
run_one bf16  --model "$BF16_REPO"

# tally
n=$(ls "$OUTDIR"/*_vllm.json 2>/dev/null | wc -l)
say "=== DONE: $n/3 precisions produced JSON ==="
if [ "$n" -ge 1 ]; then echo "DONE $n/3" > "$OUTDIR/STATUS"; else echo "FAIL 0/3" > "$OUTDIR/STATUS"; fi
