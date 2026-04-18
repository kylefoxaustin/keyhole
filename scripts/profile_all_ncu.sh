#!/usr/bin/env bash
# Canonical Nsight Compute runs for the platform-budget breakdown.
#
# Emits JSON files under data/output/ncu/ with per-NVTX-range metrics
# (instruction count, tensor-core ops, DRAM bytes). The sizer's
# platform_budget.py can then replace its approximated ss_tops_avg /
# ss_ddr_gbs_avg columns with measured values.
#
# Before running:
#   1. Make sure no other CUDA process is holding the GPU. Check `nvidia-smi`.
#      In particular, stop the LLM server (python3 llm_server.py) if it's
#      running — ncu needs exclusive access and the server holds 24 GB.
#   2. The main bake-offs (normal, non-profiled) should have run once so their
#      cached data (frames/prompts/refs/engines) exists. Profiling reuses
#      these caches — it doesn't re-extract.
#   3. Set KEYHOLE_VENV or activate the venv with the bake-off deps:
#        source ~/.virtualenvs/keyhole/bin/activate
#      (EfficientSAM3 variants use .venv-es3/ — separate invocations below.)
#
# Runtime estimate: ~2-5× the normal bake-off wall-clock per run. Total for
# the full sweep: probably 30-60 minutes.
#
# Usage:
#   bash scripts/profile_all_ncu.sh              # run everything
#   bash scripts/profile_all_ncu.sh trt_yolo     # just the named target(s)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="$ROOT/data/output/ncu"
mkdir -p "$OUT_DIR"

KEYHOLE_PY="${KEYHOLE_PY:-$HOME/.virtualenvs/keyhole/bin/python}"
ES3_PY="$ROOT/.venv-es3/bin/python"
CLIP_ENTRY="data/videos/720p_EW_clip.mp4"

# Pick which targets to run. Default = all. Pass names as args to filter.
TARGETS=("${@:-trt_yolo sam_variants efficientsam3 efficientsam3p1 trt_clip trt_yoloe26 yoloe26 llm}")
# shellcheck disable=SC2206
TARGETS=(${TARGETS[@]})

has() { for t in "${TARGETS[@]}"; do [[ "$t" == "$1" ]] && return 0; done; return 1; }

echo "Output dir: $OUT_DIR"
echo "Targets: ${TARGETS[*]}"
echo

if has trt_yolo; then
    echo "==== [trt_yolo] YOLO-seg FP16 / INT8 / FP8 TensorRT ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/trt_yolo.json" \
        -- "$KEYHOLE_PY" scripts/bakeoff_trt_yolo.py --clip "$CLIP_ENTRY"
fi

if has sam_variants; then
    echo "==== [sam_variants] SAM 3 ref + MobileSAM / EfficientSAM / YOLO-seg ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/sam_variants.json" \
        -- "$KEYHOLE_PY" scripts/bakeoff_sam_variants.py --clip "$CLIP_ENTRY"
fi

if has efficientsam3; then
    echo "==== [efficientsam3] EfficientSAM3 ES-EV-S (Option A, stage1_all_converted) ===="
    if [[ ! -x "$ES3_PY" ]]; then
        echo "  SKIP: .venv-es3/ not present. Create it first (see REPRODUCE.md)."
    else
        "$KEYHOLE_PY" scripts/profile_ncu.py \
            --out "$OUT_DIR/efficientsam3.json" \
            -- "$ES3_PY" scripts/bakeoff_efficientsam3.py
    fi
fi

if has efficientsam3p1; then
    echo "==== [efficientsam3p1] EfficientSAM3.1 ES-EV-S (SAM 3.1 student) ===="
    if [[ ! -x "$ES3_PY" ]]; then
        echo "  SKIP: .venv-es3/ not present."
    else
        "$KEYHOLE_PY" scripts/profile_ncu.py \
            --out "$OUT_DIR/efficientsam3p1.json" \
            -- "$ES3_PY" scripts/bakeoff_efficientsam3p1.py
    fi
fi

if has trt_clip; then
    echo "==== [trt_clip] CLIP visual tower (TRT FP16/FP8) ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/trt_clip.json" \
        -- "$KEYHOLE_PY" scripts/bakeoff_trt_clip.py
fi

if has trt_yoloe26; then
    echo "==== [trt_yoloe26] YOLOE-26S-PF FP16 / FP8 TRT ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/trt_yoloe26.json" \
        -- "$KEYHOLE_PY" scripts/bakeoff_trt_yoloe26.py
fi

if has yoloe26; then
    echo "==== [yoloe26] YOLOE-26S (PyTorch, text-prompt + prompt-free) ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/yoloe26.json" \
        -- "$KEYHOLE_PY" scripts/bakeoff_yoloe26.py
fi

if has llm; then
    echo "==== [llm] Qwen3-30B-A3B Q4/Q5/Q8 — prefill + decode ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/llm.json" \
        -- "$KEYHOLE_PY" scripts/bakeoff_llm.py --quants Q4_K_M
fi

echo
echo "All requested targets complete."
echo "Output JSONs:"
ls -1 "$OUT_DIR"/*.json 2>/dev/null || echo "  (none)"
