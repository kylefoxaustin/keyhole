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

# Force fresh GPU runs so ncu actually sees kernel launches. Without this,
# bakeoff_trt_yolo.py / bakeoff_sam_variants.py / bakeoff_trt_clip.py short-
# circuit on cached JSON and the profiler gets nothing to attribute.
export KEYHOLE_FORCE_RERUN=1

# When invoked via `sudo` (required for ncu GPU perf counters unless
# NVreg_RestrictProfilingToAdminUsers=0 is set), $HOME becomes /root. Resolve
# the user's venv relative to SUDO_USER's home so the script works unchanged
# under `sudo bash scripts/profile_all_ncu.sh`.
if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
    USER_HOME="$HOME"
fi
KEYHOLE_PY="${KEYHOLE_PY:-$USER_HOME/.virtualenvs/keyhole/bin/python}"
ES3_PY="$ROOT/.venv-es3/bin/python"
CLIP_ENTRY="data/videos/720p_EW_clip.mp4"

# Under sudo, HOME becomes /root and huggingface_hub fails to find Kyle's
# cached token and gated-repo snapshots (facebook/sam3 etc.). Re-point HOME
# + HF cache dirs at the invoking user's home so gated repo loads succeed
# from cache without re-authenticating.
export HOME="$USER_HOME"
export HF_HOME="${HF_HOME:-$USER_HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

# PyTorch 2.11 + CUDA 13 wheel in .venv-es3/ ships libnvrtc-builtins.so.13.0
# under nvidia/cu13/lib/, but that path isn't on the loader's search list by
# default. Without it, any nvrtc JIT compile (e.g. torch.compile fused ops
# inside EfficientSAM3's encoder) fails with "failed to open libnvrtc-
# builtins.so.13.0". Prepend the cu13 lib dir so the loader finds it.
ES3_CU13_LIB="$ROOT/.venv-es3/lib/python3.12/site-packages/nvidia/cu13/lib"
KEYHOLE_CU13_LIB="$USER_HOME/.virtualenvs/keyhole/lib/python3.10/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$ES3_CU13_LIB:$KEYHOLE_CU13_LIB:${LD_LIBRARY_PATH:-}"

# Pick which targets to run. Default = all. Pass names as args to filter.
TARGETS=("${@:-trt_yolo sam_variants efficientsam3 efficientsam3p1 trt_clip trt_yoloe26 yoloe26 llm sam3_refs}")
# shellcheck disable=SC2206
TARGETS=(${TARGETS[@]})

has() { for t in "${TARGETS[@]}"; do [[ "$t" == "$1" ]] && return 0; done; return 1; }

# Expand to `--keep-csv` when KEYHOLE_NCU_KEEP_CSV is set. Used on first-run
# sweeps so we can re-parse the raw ncu CSVs without re-profiling if the JSON
# comes back wrong.
KEEP_CSV_FLAG="${KEYHOLE_NCU_KEEP_CSV:+--keep-csv}"

echo "Output dir: $OUT_DIR"
echo "Targets: ${TARGETS[*]}"
[[ -n "$KEEP_CSV_FLAG" ]] && echo "Keeping intermediate ncu CSVs (KEYHOLE_NCU_KEEP_CSV is set)"
echo

if has trt_yolo; then
    # Default variant (yolo11s-seg) → trt_yolo.json; other variants → trt_yolo_{variant}.json.
    # Honors KEYHOLE_YOLO_VARIANT so caller can switch without editing this file.
    VARIANT="${KEYHOLE_YOLO_VARIANT:-yolo11s-seg}"
    if [[ "$VARIANT" == "yolo11s-seg" ]]; then
        NCU_OUT="$OUT_DIR/trt_yolo.json"
    else
        NCU_OUT="$OUT_DIR/trt_yolo_${VARIANT}.json"
    fi
    echo "==== [trt_yolo] YOLO-seg (${VARIANT}) FP16 / INT8 / FP8 TensorRT → $(basename "$NCU_OUT") ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$NCU_OUT" $KEEP_CSV_FLAG \
        -- "$KEYHOLE_PY" scripts/bakeoff_trt_yolo.py --clip "$CLIP_ENTRY"
fi

if has sam_variants; then
    echo "==== [sam_variants] SAM 3 ref + MobileSAM / EfficientSAM / YOLO-seg ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/sam_variants.json" $KEEP_CSV_FLAG \
        -- "$KEYHOLE_PY" scripts/bakeoff_sam_variants.py --clip "$CLIP_ENTRY"
fi

if has sam3_refs; then
    # Surgical: capture ONLY the SAM 3 BF16 reference inference range.
    # Under app-replay-match=name the SAM 3 kernel set varied across passes
    # and was dropped — kernel-replay bypasses that. To avoid re-measuring
    # the other 4 contestants (expensive under kernel-replay), we clear
    # the refs cache (so SAM 3 runs) and lean on existing contestant
    # JSONs (which short-circuit when KEYHOLE_FORCE_RERUN is NOT set).
    echo "==== [sam3_refs] SAM 3 BF16 reference only (surgical kernel-replay) ===="
    REFS_DIR="$ROOT/data/output/bakeoff/720p_EW_clip/refs"
    REFS_META="$ROOT/data/output/bakeoff/720p_EW_clip/refs_meta.json"
    if [[ -d "$REFS_DIR" ]] || [[ -f "$REFS_META" ]]; then
        echo "  Clearing refs cache at $REFS_DIR / $REFS_META"
        rm -rf "$REFS_DIR" "$REFS_META"
    fi
    # Guard: --contestants mobilesam rewrites summary.json with only one contestant; back up and restore.
    SUMMARY_720P="$ROOT/data/output/bakeoff/720p_EW_clip/summary.json"
    [[ -f "$SUMMARY_720P" ]] && cp -a "$SUMMARY_720P" "$SUMMARY_720P.bak"
    (unset KEYHOLE_FORCE_RERUN
     "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/sam3_bf16_refs.json" $KEEP_CSV_FLAG \
        -- "$KEYHOLE_PY" scripts/bakeoff_sam_variants.py \
           --clip "$CLIP_ENTRY" --contestants mobilesam)
    [[ -f "$SUMMARY_720P.bak" ]] && mv -f "$SUMMARY_720P.bak" "$SUMMARY_720P"
fi

if has efficientsam3; then
    echo "==== [efficientsam3] EfficientSAM3 ES-EV-S (Option A, stage1_all_converted) ===="
    if [[ ! -x "$ES3_PY" ]]; then
        echo "  SKIP: .venv-es3/ not present. Create it first (see REPRODUCE.md)."
    else
        "$KEYHOLE_PY" scripts/profile_ncu.py \
            --out "$OUT_DIR/efficientsam3.json" $KEEP_CSV_FLAG \
            -- "$ES3_PY" scripts/bakeoff_efficientsam3.py
    fi
fi

if has efficientsam3p1; then
    # Kernel-replay on EfficientSAM3.1 is exceptionally slow (~36 min/frame
    # × 30 frames = 18 hrs for the full 3-resolution sweep). The sizer only
    # needs ONE bandwidth measurement per pipeline, so restrict to 720p —
    # saves ~12 hrs while producing the same DRAM-per-forward number.
    echo "==== [efficientsam3p1] EfficientSAM3.1 ES-EV-S (720p only, kernel-replay) ===="
    if [[ ! -x "$ES3_PY" ]]; then
        echo "  SKIP: .venv-es3/ not present."
    else
        "$KEYHOLE_PY" scripts/profile_ncu.py \
            --out "$OUT_DIR/efficientsam3p1.json" $KEEP_CSV_FLAG \
            -- "$ES3_PY" scripts/bakeoff_efficientsam3p1.py --resolutions 720p
    fi
fi

if has trt_clip; then
    echo "==== [trt_clip] CLIP visual tower (TRT FP16/FP8) ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/trt_clip.json" $KEEP_CSV_FLAG \
        -- "$KEYHOLE_PY" scripts/bakeoff_trt_clip.py
fi

if has trt_yoloe26; then
    echo "==== [trt_yoloe26] YOLOE-26S-PF FP16 / FP8 TRT ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/trt_yoloe26.json" $KEEP_CSV_FLAG \
        -- "$KEYHOLE_PY" scripts/bakeoff_trt_yoloe26.py
fi

if has yoloe26; then
    echo "==== [yoloe26] YOLOE-26S (PyTorch, text-prompt + prompt-free) ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/yoloe26.json" $KEEP_CSV_FLAG \
        -- "$KEYHOLE_PY" scripts/bakeoff_yoloe26.py
fi

if has llm; then
    echo "==== [llm] Qwen3-30B-A3B Q4/Q5/Q8 — prefill + decode ===="
    "$KEYHOLE_PY" scripts/profile_ncu.py \
        --out "$OUT_DIR/llm.json" $KEEP_CSV_FLAG \
        -- "$KEYHOLE_PY" scripts/bakeoff_llm.py --quants Q4_K_M
fi

echo
echo "All requested targets complete."
echo "Output JSONs:"
ls -1 "$OUT_DIR"/*.json 2>/dev/null || echo "  (none)"
