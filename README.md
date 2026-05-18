# Keyhole

**v1.0.0** · Open-source edge AI video analytics platform. Cameras detect; a
shared NPU enriches detections with vision + open-vocabulary labeling;
operators query events in natural language.

Inspired by the Ubiquiti UniFi AI Key architecture: lightweight detection on
the camera, deeper enrichment on a compute node, semantic + literal search
over the resulting event store.

---

## Headline result

**SAM 3 → Hybrid V2 architectural pivot delivers 515× lower DRAM bandwidth
per shipping pipeline frame**, measured by Nsight Compute (not projected).
On NPU Mid stock LPDDR5X, the recommended pipeline (TRT FP8 YOLO-seg + TRT
FP8 CLIP @ 1 Hz keyframe debounce) reaches **36 FPS at 720p** — first
real-time edge result in the campaign.

The SAM 3 baseline this replaces was bandwidth-bound everywhere: 119 GB
DRAM per forward, 0.4 FPS edge, no escape via quantization, prompt
reduction, or resolution cut. The lever turned out to be architectural
replacement, not optimization.

| Stage | What it does | RTX 5090 ms | NPU Mid edge ms | NPU Mid FPS |
|---|---|---|---|---|
| SAM 3 baseline (BF16) | open-vocab segmentation, monolithic | ~120 | ~1700 | **0.4** |
| Hybrid V2 BF16 (YOLO-seg + CLIP) | two-stage, FP-baseline | ~62 | ~890 | ~17 |
| **Hybrid V2 TRT FP8** + 1 Hz CLIP | shipping recipe | ~28 | ~28 | **36** |

`±15%` sensitivity band on the 36 FPS number reflects the `0.70` bandwidth
efficiency assumption used to scale from 5090 reference to NPU Mid.

## Three operational modes

The deployment surface acknowledges three modes (the `keyhole-sizer`
companion app surfaces them as a UI toggle):

1. **Vision-only** — vision pipeline runs; LLM off. Default video-analytics
   deployment.
2. **Vision + LLM** — both running on a shared NPU. The LLM acts on vision
   data (NLQ over the event store, agentic scene queries). The engineering
   question is NPU coexistence — duty cycle, bandwidth share, multi-stream.
3. **LLM-only** — vision pipeline off; LLM standalone. Equivalent to the
   Skippy product running inside `keyhole-sizer`. Perf matches the Skippy
   deck exactly.

## LLM identity

The LLM in Keyhole is the **Skippy product artifact, unmodified** —
Qwen3-30B-A3B base, Q4_K_M quantization, identical shipping recipe. Keyhole
*uses* the artifact; the training story (recipe taxonomy, fine-tuning
campaign coverage, headline-erosion methodology arc, cross-family
base-selection) is documented in the [Skippy / personal-AI-framework
deck](https://github.com/kylefoxaustin/personal-ai-framework) and is not
reproduced here.

## NPU tier model (canonical)

PAI deck slide 11 is the golden source for NPU tier specs; both
`keyhole-sizer` and the Keyhole deck pull from the same model.

| Tier | Memory bus | BW theo | BW eff (70%) | TOPS | Note |
|---|---|---|---|---|---|
| NPU Low-LP5-32bit / i.MX 95 | 32-bit LPDDR5 @ 6.4 GT/s | 25.6 | 17.92 | 2 INT8 | Neutron-class |
| NPU Low-LP5-64bit | 64-bit LPDDR5 @ 6.4 GT/s | 51.2 | 35.84 | 2 INT8 | INT8-only |
| NPU Low-LP5X | 64-bit LPDDR5X @ 8.4 GT/s | 67.2 | 47.04 | 50 / 100 / 100 BF/INT8/FP8 | FP-capable lite |
| **NPU Mid** | 128-bit LPDDR5X @ 8.4 GT/s | 134.4 | 94.08 | **200 INT8 (no FP)** | Recommended target — INT8-only |
| **NPU High** | **same bus as Mid** | 134.4 | 94.08 | 200 / 400 / 400 BF/INT8/FP8 | FP-capable — full Hybrid V2 |
| RTX 5090 (reference) | 512-bit GDDR7 @ 28 GT/s | 1792 | 1523.2 (85%) | ~105 / ~210 / DP4A | Measurement anchor |

Mid is INT8-only; High differentiates on compute + capacity + TDP, **not
bandwidth** (shared 8.4 GT/s class). FP recipes (CLIP, ViT alternatives,
EfficientSAM3 variants) pin to NPU High. Memory upgrades (LPDDR5T-11.2,
LPDDR6) lift decode on both tiers in lockstep.

## What's in this release

### Bake-off catalog

- **TensorRT YOLO-seg** at INT8 + FP8 on Blackwell — full-model Conv quant
  via TRT 10.16 (torchao's 1×128 block-size constraint had blocked this).
  Recall = 1.000, matched IoU = 0.998 vs FP16 engine.
- **TensorRT CLIP visual tower** at FP16 + FP8 — 3× speedup, top-1
  concept-tag agreement 0.964 vs BF16.
- **Mask-model bake-off** — MobileSAM, EfficientSAM-tiny/small, YOLO-seg
  scored against SAM 3 reference; YOLO-seg wins on combined detection +
  segmentation footprint.
- **EfficientSAM3 community variants** (Apr 2026) — bench against the
  shipping pipeline; community SAM 3 Lite is real but ~13× slower than the
  recommended Hybrid V2 stack at edge.
- **YOLOE-26 one-model open-vocab** — alternative to Hybrid V2; TRT FP8
  gives ~17% speedup vs 3× on YOLO-seg (kernel-launch-bound at small
  parameter count). Real but doesn't displace the two-stage pipeline.
- **ViT alternatives** (RT-DETR-L, DETR-ResNet50, OWLv2, Grounding DINO) —
  what-if analysis; camera-side ViTs don't fit NPU Mid stock memory.
  OWLv2 fits a 1 Hz agentic-query budget.

### Measurement validation

- **Nsight Compute pipeline** — measured DRAM per forward for 23 workloads;
  log-scale chart + full table in the deck. The 515× SAM 3 → Hybrid V2 win
  is measured (118,975 MB / 231 MB), not projected.
- **End-to-end pipeline latency budget** — every stage (FFmpeg decode,
  preprocess, YOLO, CLIP, SQLite INSERT) on 5090 reference + projected to
  NPU Mid. On pure-NPU edge boards (no ISP / 2D-GPU offload), CPU stages
  crowd out the 36 FPS budget; production SoCs with offloads sustain it.
- **i.MX 95 anchor** — NXP eIQ Neutron NPU measurement on yolov8n-seg INT8
  @ 1080p (29.2 FPS measured vs 18.3 FPS BW-projected — 1.6× delta
  documents pure-BW projection limits on weak silicon).

### Companion artifacts

- **Plain deck** (63 slides, dark-bg) and **NXP-branded deck** (63 slides,
  corporate template via `pptx_template_converter` theme swap).
- **`PRESENTER_SCRIPT.md`** — speaker text for 45–60 min technical-
  management walkthrough.
- **`docs/CHANGES_2026-05-17.md`** — reviewer-orientation doc.
- **`docs/ALIGNMENT_PLAN.md`** — conceptual-frame alignment history;
  ALL PHASES DONE.
- **`docs/PRIVATE_DECK.md`** — operator-discipline reference for the
  `--include-private` build that surfaces measured silicon anchor values
  from a gitignored `.streamlit/secrets.toml` file.

### Companion repo

- [**`keyhole-sizer`**](https://github.com/kylefoxaustin/keyhole-sizer) —
  Streamlit app for what-if NPU sizing across all five tiers + the three
  operational modes. Self-contained; pulls from the same NPU model the
  deck does.

---

## Quick Start

### Prerequisites

- NVIDIA GPU with ≥16 GB VRAM (RTX 5090 reference)
- Python 3.10+ (3.12+ recommended for SAM 3 / TRT 10.16)
- PyTorch 2.7+ with CUDA support
- FFmpeg
- HuggingFace account with SAM 3 access (only for SAM 3 baseline pipelines)
- TensorRT 10.16 (Blackwell) for FP8 + INT8 compile path

### Installation

```bash
git clone https://github.com/kylefoxaustin/keyhole.git
cd keyhole

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# SAM 3 (only needed for the baseline)
mkdir -p third_party && cd third_party
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e ".[notebooks]"
pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
cd ../..

# Lightweight SAM + CLIP
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install open-clip-torch
mkdir -p weights
wget -O weights/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

cp .env.example .env
hf auth login
```

### Running

```bash
# SAM 3 baseline (the slow one — for reference / re-measurement)
python -m src.main process --video data/videos/clip.mp4 --single-pass --render --profile

# Hybrid V2 BF16 (the architectural pivot, pre-TRT)
python -m src.main process --video data/videos/clip.mp4 \
  --hybrid-v2 yolo11n-seg.pt --render --profile

# Bake-offs
python scripts/bakeoff_trt_yolo.py        # TRT YOLO INT8 + FP8 full-model
python scripts/bakeoff_trt_clip.py        # TRT CLIP visual tower FP16 + FP8
python scripts/bakeoff_llm_anchors.py     # 5090 LLM anchor catalog
python scripts/profile_ncu.py             # Nsight Compute DRAM measurements

# Edge projection + comparison
python -m src.main emulate --compare-all
python scripts/compare_models.py --video data/videos/clip.mp4 --max-frames 10

# Decks
python scripts/build_deck.py                       # plain (dark-bg)
KEYHOLE_DECK_MERGE_TARGET=1 python scripts/build_deck.py  # merge-ready (for branded conversion)

# HTTP API server (for keyhole-UI or any HTTP client)
python -m src.main serve
```

### Branded deck rebuild (Phase E workflow)

```bash
# 1. Build merge-ready plain deck
KEYHOLE_DECK_MERGE_TARGET=1 python scripts/build_deck.py
cp data/output/keyhole_results.pptx \
  ~/Documents/GitHub/pptx_template_converter/input/keyhole_merge_ready.pptx

# 2. Apply NXP corporate template + color map
cd ~/Documents/GitHub/pptx_template_converter
python convert.py \
  --input    input/keyhole_merge_ready.pptx \
  --template template/corporate_template.pptx \
  --output   output/keyhole_deck_branded.pptx \
  --color-map mappings/keyhole_to_corporate.json

# 3. Copy back + restore plain (dark) variant
cp output/keyhole_deck_branded.pptx \
  ~/Documents/GitHub/keyhole/data/output/keyhole_deck_branded.pptx
cd ~/Documents/GitHub/keyhole
python scripts/build_deck.py
```

`scripts/update_branded_deck.py` is obsolete — was an in-place patcher for
a static 58-slide variant; superseded by the full-rebuild flow above.

---

## HTTP API

Keyhole exposes a FastAPI server (default `http://localhost:8777`) consumed
by the [keyhole-UI](https://github.com/kylefoxaustin/keyhole-UI) Next.js
frontend but usable by any HTTP client. Start it with `python -m src.main
serve`.

Key endpoints (full contract in [`API.md`](./API.md)):

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Server + GPU status |
| `GET /api/videos` | List videos with per-item status |
| `POST /api/videos` | Upload a video |
| `GET /api/videos/{id}/thumbnail` | First-frame JPEG |
| `GET /api/videos/{id}/annotated` | Annotated MP4 stream |
| `GET /api/videos/{id}/density?buckets=N` | Timeline heatmap |
| `DELETE /api/videos/{id}` | Remove a video and its events |
| `GET /api/events?q=...` | Hybrid semantic + literal search |
| `GET /api/events/{id}/frame` | Annotated frame JPEG |
| `GET /api/events/{id}/clip?before=5&after=5&format=mp4\|gif` | Event clip |
| `GET /api/concepts`, `GET /api/classes` | Autocomplete vocabularies |
| `WS /api/ws/processing` | Live processing-worker events |

OpenAPI schema at `/api/openapi.json`. Uploads are processed by a single
async worker; the WS emits `queue_position` for client UIs.

---

## Project Structure

```
keyhole/
├── src/
│   ├── main.py              # CLI entry point
│   ├── ingest/video.py      # FFmpeg frame extraction
│   ├── detect/
│   │   ├── yolo.py          # YOLO 11 detection
│   │   ├── sam3_detect.py   # SAM 3 single-pass (baseline)
│   │   └── hybrid_v2.py     # YOLO-seg + CLIP (recommended)
│   ├── enrich/sam3.py       # SAM 3 concept enrichment (sequential reference)
│   ├── render/video.py      # Annotated video + GIF output
│   ├── emulate/
│   │   ├── npu_emulator.py  # Bandwidth-aware edge projection
│   │   ├── sam3_reference.py
│   │   └── layer_profiler.py
│   ├── anchors/             # Private NPU + CNN measured-silicon anchor loaders
│   ├── store/               # SQLite + SQLAlchemy metadata store
│   ├── query/nlq.py         # LLM-backed natural language query
│   └── api/server.py        # FastAPI + web UI
├── scripts/
│   ├── build_deck.py        # 63-slide deck generator (KEYHOLE_DECK_MERGE_TARGET=1 for branded)
│   ├── bakeoff_*.py         # Bake-off harnesses (TRT YOLO, TRT CLIP, LLM anchors, ncu, ...)
│   ├── profile_ncu.py       # Nsight Compute pipeline
│   └── compare_models.py    # Multi-model benchmark
├── docs/
│   ├── PRESENTER_SCRIPT.md  # 45–60 min walkthrough script
│   ├── CHANGES_2026-05-17.md
│   ├── ALIGNMENT_PLAN.md    # Conceptual-frame alignment history
│   └── PRIVATE_DECK.md      # --include-private operator discipline
├── data/output/             # Decks + bake-off artifacts (gitignored)
└── third_party/             # SAM 3 source install (gitignored)
```

---

## Anchor secrets discipline

Measured silicon anchor values (NPU Mid/High decode rates, CNN ms/forward)
live in a gitignored `.streamlit/secrets.toml` file Kyle populates locally.
The `--include-private` build flag surfaces them on an extra deck slide at
runtime; values are **never** committed to source, logged, surfaced in
exception messages, or quoted on the cross-session bus. Refer by KEY, not
VALUE. See [`docs/PRIVATE_DECK.md`](./docs/PRIVATE_DECK.md) for the full
operator-discipline reference.

The `keyhole_results_PRIVATE.pptx` output of `--include-private` is
gitignored and must not be pushed to public surfaces (my-stuff,
gdrive:skippy_files). NXP-internal-only destinations.

---

## Version history

| Version | Date | Notes |
|---|---|---|
| **v1.0.0** | 2026-05-18 | First tagged release. Recovery point ahead of cross-repo engine-extraction work. 63-slide deck (plain + NXP-branded). PAI golden NPU tier framing held. Skippy training content removed (cross-referenced in deck slide 48). Three-operational-modes framing landed (deck slide 6). Phase E branded rebuild via `pptx_template_converter`. Reciprocal cross-reference live with personal-ai-framework deck slide 18. |

---

## Companion / cross-repo links

- [**`personal-ai-framework`**](https://github.com/kylefoxaustin/personal-ai-framework) —
  Skippy product (LLM artifact + training story). The Keyhole LLM layer is
  this project's `v4` Qwen3-30B-A3B Q4_K_M unmodified. Cross-reference slide:
  PAI deck slide 18 ↔ Keyhole deck slide 48.
- [**`keyhole-sizer`**](https://github.com/kylefoxaustin/keyhole-sizer) —
  Streamlit what-if sizer for the three operational modes across all five
  NPU tiers.
- [**`personal-ai-assistant-sizer`**](https://github.com/kylefoxaustin/personal-ai-assistant-sizer) —
  Sister sizer focused on the LLM-only mode (Skippy product deployments).
- [**`pptx_template_converter`**](https://github.com/kylefoxaustin/pptx_template_converter) —
  Local theme-swap tool used for the NXP-branded deck rebuild.

---

## Maintainer

**Kyle Fox** ([@kylefoxaustin](https://github.com/kylefoxaustin))

## License

MIT License — see LICENSE file.

## TTA — Trust the Awesomeness
