# Reproducing the Keyhole Bake-off Results

This document is for a software engineer who has read the Keyhole
deck (`data/output/keyhole_results.pptx`) and wants to reproduce every
number in it from first principles. If you just want to kick the
tires on the video-analysis UI, see `keyhole-UI` instead — that's the
demo repo. This one is for regenerating the measurement data that
backs the deck's claims.

Every number you see in the deck's 49 slides was produced by one of the
scripts below, running against the weights and test clips in this
repo. Nothing is hand-assembled.

---

## What you'll reproduce

Running everything in this guide regenerates:

- All 17 bake-off JSON files under `data/output/bakeoff/`
- A consolidated `data/output/keyhole_results.xlsx` with one sheet per bake-off (raw data + summary tables)
- The deck itself (`data/output/keyhole_results.pptx`, 49 slides) built from those JSONs

The bake-offs characterize the full Keyhole pipeline evolution:

| # | Bake-off | Script | Headline finding |
|---|----------|--------|------------------|
| 1 | Mask-model comparison | `bakeoff_sam_variants.py` | ES-Tiny dominates MobileSAM; ES-Small tops quality |
| 2 | FP8 activation quant | `bakeoff_fp8.py` | 94/95 Linears, ΔIoU < 0.003, edge FPS 2.5 → 4.9 |
| 3 | INT8 + SmoothQuant | `bakeoff_smoothquant.py` | Plain INT8 matches FP8; SmoothQuant CONVERT blocked by torchao 0.17 |
| 4 | Hybrid V2 CLIP quant | `bakeoff_hybrid_v2.py` | 48/72 CLIP Linears, FP8 better than INT8 on top-1 tags |
| 5 | CLIP keyframe debounce | `bakeoff_keyframe_debounce.py` | 1 Hz debounce → 16 FPS edge, 93% of YOLO ceiling |
| 6 | YOLO Conv INT8 (torchao) | `bakeoff_yolo_conv_quant.py` | 1×1 swap + INT8 → 23.8 FPS edge (partial unblock) |
| 7 | TensorRT YOLO FP8 | `bakeoff_trt_yolo.py` | Full Conv-FP8 works on Blackwell; 36.8 FPS edge, recall 1.00 |
| 8 | TensorRT CLIP FP8 | `bakeoff_trt_clip.py` | CLIP FP8 edge halves (29.8 → 15.1 ms) |
| 9 | LLM Qwen3-30B-A3B MoE | `bakeoff_llm.py` | 250 tok/s 5090 decode; NPU Mid actuals 37.85 tok/s |
| 10 | Multi-stream concurrency | `bakeoff_concurrency.py` | 4 streams at batch=4 → 26 FPS each (not 9) |

Total runtime on an RTX 5090: approximately **25-35 minutes** for the full set
once the weights and Qwen GGUFs are in place (the Qwen downloads
themselves add another 30-60 min depending on HF bandwidth).

---

## Hardware prerequisites

This entire set of bake-offs was measured on the platform detailed on
slide 3 of the deck. To reproduce the **full** pipeline (including TRT
FP8 and the LLM) you need:

- **GPU:** NVIDIA RTX 5090 or other Blackwell-class card (SM 12.0+)
  with native FP8 tensor cores. Ada/Hopper (SM 8.9 / 9.0) will work for
  most bake-offs but **TRT FP8 requires SM 12.0+**. CUDA 13 builds
  assumed throughout.
- **VRAM:** 32 GB minimum. Q8_0 Qwen GGUF (32.5 GB) needs partial CPU
  offload. Everything else fits.
- **System RAM:** 64 GB recommended (94 GB on the reference machine).
- **Disk:** ~100 GB free for the three Qwen GGUFs (72 GB) + SAM 3
  weights + test clips + TensorRT engines.
- **CPU:** any modern x86_64; i9-14900KF on reference.

Lower-tier cards can still reproduce most bake-offs:

- Vision-only (bake-offs #1-#8) needs ~16 GB VRAM minimum
- LLM (#9) can drop to Q4_K_M only on a 24 GB card
- Concurrency (#10) works on any CUDA GPU

---

## Software prerequisites

```bash
# OS
#   Reference: Ubuntu 22.04.5 LTS, kernel 6.8
#   Other modern Linuxes with CUDA 13 + driver 570+ should also work

# NVIDIA driver
#   Reference: 580.126.09 (or any driver that ships CUDA 13)

# Python
#   3.10+ (reference: 3.10 via venv)

# Heavy native deps that get pulled in by requirements.txt:
#   torch 2.11.0 (cu130 build)
#   tensorrt 10.16.1.11 (CUDA 13 build)
#   torchao 0.17
#   llama-cpp-python 0.3.20 (CUDA build)
#   ultralytics
#   open-clip-torch
#   flash-attn-3
#   SAM 3 (third_party/sam3, see below)
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/kylefoxaustin/keyhole.git
cd keyhole

# 2. Python venv
python3.10 -m venv ~/.virtualenvs/keyhole
source ~/.virtualenvs/keyhole/bin/activate

# 3. Install deps (heavy — downloads several GB of CUDA wheels)
pip install -U pip
pip install -r requirements.txt

# 4. Install llama-cpp-python with CUDA (not in requirements — needs custom index)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125

# 5. SAM 3 install (proprietary — requires Meta access grant)
#    Clone to third_party/sam3 and follow the vendor instructions.
#    See memory/project_sam3_install.md for the exact install steps
#    we used including the torch.inference_mode() / bf16 context gotchas.
#    If you can't get SAM 3 access, skip bake-off #1's reference masks;
#    the other nine bake-offs don't require SAM 3 at runtime (they use
#    cached reference masks in data/output/bakeoff/{clip}/refs/).

# 6. Vision weights (small, ~400 MB total)
./scripts/download_models.sh
#    Downloads: yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt, yolo11x-seg.pt,
#    yolo11x.pt, FastSAM-s.pt, FastSAM-x.pt, MobileSAM (weights/mobile_sam.pt),
#    EfficientSAM Tiny + Small.

# 7. Test clips (already in repo at data/videos/)
#    720p_EW_clip.mp4, 1080p_EW_clip.mp4, embedded_world_clip.mp4 (4K),
#    embedded_world_clip_1080p.mp4.
#    These are canonical "embedded world" conference-hall footage used
#    for every bake-off.

# 8. Cached bake-off inputs (sampled frames, YOLO prompt boxes, SAM 3
#    reference masks) are in data/output/bakeoff/{720p_EW_clip,
#    embedded_world_clip_1080p, embedded_world_clip}/. These are
#    version-controlled so you don't need SAM 3 to regenerate them.

# 9. Qwen LLM weights (large — 73 GB total, only needed for bake-off #9)
mkdir -p weights
cd weights
hf download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
    Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
    Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf \
    Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf \
    --local-dir .
cd ..
#    This is the exact Kyle-merged base the personal-ai-framework
#    (Skippy) project also uses — see docs/cloud-training-runbook.md
#    in that repo if you want to reproduce their QLoRA fine-tune too.

# 10. Verify install
python -c "import torch, tensorrt, torchao, llama_cpp, ultralytics; \
           print('torch', torch.__version__); \
           print('tensorrt', tensorrt.__version__); \
           print('GPU', torch.cuda.get_device_name(0))"
```

---

## Run the bake-offs (in dependency order)

Each script caches its own intermediate results to `data/output/bakeoff/`
so you can re-run any one of them without redoing the others. Delete the
corresponding `<name>.json` or `<name>/` subdir to force a rebuild.

```bash
# 1. Mask-model bake-off (prerequisite for #2, #3)
#    ~5 min — runs 4 contestants (MobileSAM, ES-Tiny, ES-Small, YOLO-seg)
#    on 3 clip resolutions against cached SAM 3 references.
python scripts/bakeoff_sam_variants.py

# 2. FP8 activation quantization on ES-Small + YOLO-seg
#    ~3 min
python scripts/bakeoff_fp8.py

# 3. INT8 + SmoothQuant on the same winners
#    ~5 min — SmoothQuant CONVERT will log an error on ES-Small (torchao
#    0.17 API gap), that's expected and documented.
python scripts/bakeoff_smoothquant.py

# 4. Hybrid V2 CLIP quantization (swaps architecture to YOLO-seg + CLIP)
#    ~5 min — reloads CLIP once per quant so it takes a bit.
python scripts/bakeoff_hybrid_v2.py

# 5. CLIP keyframe debounce — pure post-processing of #4's data
#    ~seconds. No new inference.
python scripts/bakeoff_keyframe_debounce.py

# 6. YOLO Conv INT8 via torchao 1×1 swap
#    ~3 min. Note: FP8 via torchao on YOLO is tool-chain-blocked
#    (documented on its slide); the script records the blocker explicitly.
python scripts/bakeoff_yolo_conv_quant.py

# 7. TensorRT YOLO (FP16 / INT8 / FP8)
#    ~2 min inference + ~2 min engine builds. Engines cached to
#    data/trt_engines/ on first run.
python scripts/bakeoff_trt_yolo.py

# 8. TensorRT CLIP visual (FP16 / FP8)
#    ~2 min. Requires onnxscript (already in requirements.txt).
python scripts/bakeoff_trt_clip.py

# 9. LLM bake-off (Qwen3-30B-A3B MoE)
#    ~5 min per quant × 3 quants = ~15 min. Q8_0 uses partial CPU
#    offload automatically (weights exceed 32 GB VRAM).
python scripts/bakeoff_llm.py

# 10. Multi-stream concurrency
#     ~30 s. Uses the FP8 TRT engine built in #7 (dynamic batch 1-16).
#     If your YOLO TRT engine was built static-batch, you'll need to
#     rebuild with dynamic batch first — see the notes in
#     memory/project_trt_gotchas.md.
python scripts/bakeoff_concurrency.py
```

---

## Export everything to XLSX

After all bake-offs have run, produce the consolidated spreadsheet:

```bash
python scripts/export_results.py
#    Writes data/output/keyhole_results.xlsx
#    One sheet per bake-off (raw data + per-sheet summary tables).
```

This is useful for:

- Comparing your reproduced numbers against the reference numbers
  shipped in this repo's `data/output/bakeoff/*.json`
- Pivoting the data for your own deck or analysis
- Sharing a single file with non-technical stakeholders

---

## Rebuild the deck

```bash
python scripts/build_deck.py
#    Writes data/output/keyhole_results.pptx (49 slides)
#    Reads all the JSONs + reads host specs live from /proc, /sys,
#    nvidia-smi, torch.cuda.get_device_properties().
```

If any bake-off's JSON is missing, the corresponding slide is skipped
silently and the deck still builds (just shorter). This is how the deck
degrades gracefully while you work through reproduction one bake-off at
a time.

---

## Known gotchas we already hit

See `memory/project_trt_gotchas.md` in this repo for the full list, but
the three most common reproduction snags:

1. **Ultralytics engine files** have a length-prefixed JSON metadata
   header before the raw TRT bytes. If you're deserializing engines
   yourself (not via ultralytics' own loader), you must skip 4-byte LE
   length + JSON before calling `Runtime.deserialize_cuda_engine`.
   `scripts/bakeoff_trt_yolo.py::load_engine` shows the sniff.

2. **CLIP ONNX export** needs `dynamo=False` in torch 2.11 —
   otherwise large weights get split into a sibling `.data` file that
   TRT's `OnnxParser` can't follow from raw bytes.

3. **torchao FP8 on Conv-only models** is blocked at the kernel level
   (`input_tensor must be 1x128 scaled`). The 1×1-swap INT8 path (#6)
   works as a partial unblock; full Conv-FP8 needs TensorRT (#7).

---

## Cross-validation with Skippy

The LLM bake-off's Qwen3-30B-A3B numbers were independently measured on
the same 5090 host by the personal-ai-framework / Skippy project via
its production Prometheus pipeline (QLoRA-merged Q4_K_M, real prod
traffic). Our synthetic 159 tok/s RAG decode number matches their
155 tok/s sustained production number within 3%. See the LLM bake-off
slide in the deck for details.

If you're reproducing on a different host and want to validate against
Skippy's numbers, ping the personal-ai-framework project — they have
/metrics histograms from a production deployment and can rerun any
specific benchmark in ~5 min of work.

---

## Questions, bugs, contributions

- Bug in a bake-off? Open an issue on
  [github.com/kylefoxaustin/keyhole](https://github.com/kylefoxaustin/keyhole)
- Want to port a bake-off to different silicon (Hopper, MTIA, future
  NPUs)? The `src/emulate/npu_emulator.py` spec is the central
  abstraction — add a new `HardwareSpec`, reuse the rest of the
  measurement pipeline.
- Want to play with the assumptions interactively rather than editing
  scripts? See
  [`kylefoxaustin/keyhole-sizer`](https://github.com/kylefoxaustin/keyhole-sizer)
  — a separate Streamlit app that wraps the measured bake-off numbers
  in tunable sliders (NPU preset or custom bus/BW/TOPS, pipeline
  choice, concurrent stream count, LLM co-exist toggle). Install with
  `pip install -r requirements.txt && streamlit run app.py`; no GPU
  required (pure projection math on top of the already-measured
  numbers).
