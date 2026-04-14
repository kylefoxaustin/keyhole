# Keyhole

**Open-source AI Key prototype** — an edge AI video intelligence pipeline that processes
camera detections through tiered AI models, profiles them against real edge hardware
bandwidth constraints, and enables natural language querying of surveillance events.

Inspired by the Ubiquiti UniFi AI Key architecture: cameras do lightweight detection,
a separate compute node enriches those detections with advanced AI, and users query
events in plain English.

## Key Finding

**SAM 3 is memory-bandwidth-bound, not compute-bound.** On an RTX 5090 (1,792 GB/s GDDR7),
98% of GPU time is spent waiting for memory, not computing. On a typical edge NPU
(134.4 GB/s LPDDR5X), this translates to ~1,700ms per frame — **0.6 FPS**, not real-time.

Keyhole systematically tests every optimization (quantization, resolution reduction,
prompt count scaling), identifies which actually help, and implements a **hybrid pipeline**
that replaces SAM 3 with three lightweight models (YOLO-seg + CLIP) that achieve
**20 FPS projected on edge hardware** while producing visually indistinguishable output.

## Pipeline Variants

Keyhole ships with four distinct pipelines for benchmarking and comparison:

| Pipeline | Params | RTX 5090 | Edge Projected | Flag |
|----------|--------|----------|---------------|------|
| YOLO + SAM 3 sequential (baseline) | 897M | 19,672ms | N/A | *(default)* |
| SAM 3 single-pass (batched API) | 840M | 121ms | ~1,700ms (0.6 FPS) | `--single-pass` |
| Hybrid V1 (YOLO + MobileSAM + CLIP) | 218M | 142ms | ~200ms (5 FPS) | `--hybrid` |
| **Hybrid V2 (YOLO-seg + CLIP)** | **155M** | **39ms** | **~51ms (20 FPS)** | `--hybrid-v2` |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Canned      │     │  Tier 1:     │     │  Tier 2:     │     │  Metadata    │
│  Video       │────▶│  Detection   │────▶│  Enrichment  │────▶│  Store       │
│  (FFmpeg)    │     │  (variants)  │     │  (variants)  │     │  (SQLite)    │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                     │
                                                               ┌─────▼───────┐
                                                               │  NLQ Engine │
                                                               │  (LLM API)  │
                                                               └─────┬───────┘
                                                                     │
                                                               ┌─────▼───────┐
                                                               │  FastAPI     │
                                                               │  + Web UI    │
                                                               └─────────────┘
```

### Pipeline Stages

1. **Ingest** — FFmpeg extracts frames from canned video at configurable FPS
2. **Detect + Enrich** — depends on pipeline mode:
   - Sequential: YOLO 11 detects → SAM 3 enriches each crop with concept prompts
   - Single-pass: SAM 3 processes full frame with batched concept prompts
   - Hybrid V1: YOLO detects → MobileSAM segments → CLIP classifies attributes
   - Hybrid V2: YOLO-seg does detection+segmentation in one pass → CLIP classifies
3. **Store** — Structured metadata written to SQLite with full-text search
4. **Query** — Natural language queries translated to structured searches via LLM

### Edge Hardware Projection

The prototype includes a bandwidth-aware NPU emulator that projects GPU measurements
onto target edge hardware. The projection model decomposes measured GPU kernel time
into compute-bound vs bandwidth-bound components and scales each independently.

Current target: **200 TOPS BF16, 134.4 GB/s LPDDR5X (128-bit @ 8.4 GT/s), 8 GB, 25W TDP**

## Quick Start

### Prerequisites
- NVIDIA GPU with ≥16 GB VRAM (tested on RTX 5090)
- Python 3.10+ (3.12+ recommended for SAM 3)
- PyTorch 2.7+ with CUDA support
- FFmpeg
- HuggingFace account with SAM 3 access (for SAM 3 pipelines)

### Installation

```bash
git clone https://github.com/kylefoxaustin/keyhole.git
cd keyhole

# Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Install SAM 3 from source (requires HuggingFace access)
mkdir -p third_party && cd third_party
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e ".[notebooks]"
pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
cd ../..

# Install lightweight SAM variants (for hybrid pipelines)
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install open-clip-torch

# Download MobileSAM checkpoint
mkdir -p weights
wget -O weights/mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Authenticate with HuggingFace (needed for SAM 3)
hf auth login
```

### Running the Pipeline

Place a test video in `data/videos/` then run any of these:

```bash
# Sequential (original SAM 3 approach — very slow)
python -m src.main process --video data/videos/clip.mp4

# SAM 3 single-pass (100x faster than sequential)
python -m src.main process --video data/videos/clip.mp4 --single-pass --render --profile

# Hybrid V1 (YOLO + MobileSAM + CLIP)
python -m src.main process --video data/videos/clip.mp4 --hybrid --render --profile

# Hybrid V2 (YOLO-seg + CLIP) — the fastest
python -m src.main process --video data/videos/clip.mp4 --hybrid-v2 yolo11n-seg.pt --render --profile

# Project GPU measurements to edge hardware
python -m src.main emulate --compare-all

# Compare multiple models on the same frames
python scripts/compare_models.py --video data/videos/clip.mp4 --max-frames 10

# Generate results deck (PowerPoint)
python scripts/build_deck.py
```

### Rendering

With `--render`, the pipeline produces **both** an annotated MP4 and an optimized GIF
in `data/output/`. The GIF is PowerPoint-friendly (no codec issues) and uses palette
optimization for reasonable file size. Default GIF width: 800px.

### Profiling

With `--profile`, GPU kernel time, wall-clock time, VRAM usage, per-frame latency
distributions (P50/P95/P99), and per-stage timing breakdowns are captured. Every run
is timestamped and archived to `data/output/runs/` for historical comparison.

## Key Results

### The Bandwidth Wall

SAM 3 on RTX 5090 (measured):
- GPU kernel time: **102ms** (98% of wall clock)
- Theoretical compute floor: **2.4ms** (from 350 GFLOPs / 146 effective TOPS)
- Gap: **42x** — workload is deeply memory-bandwidth-bound
- Activation memory: 3.71 GB (overwhelms 72 MB L2 cache)

Projection to 134.4 GB/s edge target:
- Bandwidth ratio: 15x less memory BW
- Projected latency: **~1,700ms/frame** (0.6 FPS)
- Memory fit: 7.07 GB peak vs 8 GB capacity (no headroom)

### Optimizations Tested

| Optimization | Result |
|-------------|--------|
| Lower input resolution (4K → 720p) | Negligible — SAM 3 resizes internally to 1008x1008 |
| Reduce internal resolution | Blocked — RoPE positional embeddings locked to 63×63 grid |
| Weight-only INT8 quantization | No speedup; saves 2 GB VRAM but slower GPU kernel |
| Activation quantization | Research-grade effort, best case ~1.2 FPS on edge |
| Reduce concept prompts (9→1) | Helps on desktop (72ms); 70ms encoder floor on edge |

### The Hybrid Breakthrough

**Hybrid V2 (YOLO-seg + CLIP)** replaces both YOLO and MobileSAM with a single YOLO
segmentation model. YOLO-seg does detection + instance segmentation in 3-8ms. CLIP
provides open-vocabulary concept classification via batched zero-shot matching.

Measured on RTX 5090 (720p, 30 frames, yolo11n-seg):
- YOLO-seg: **7ms**
- CLIP (batched): **26ms**
- Total: **37ms** per frame
- Throughput: **27 FPS** on desktop

Edge projection (134.4 GB/s LPDDR5X):
- Total: **~51ms per frame**
- Throughput: **~20 FPS**
- 33x faster than SAM 3 on the same hardware

Visual output quality is indistinguishable from SAM 3 for typical surveillance scenes.

## Project Structure

```
keyhole/
├── src/
│   ├── main.py              # CLI entry point
│   ├── ingest/video.py      # FFmpeg frame extraction
│   ├── detect/
│   │   ├── yolo.py          # YOLO 11 detection
│   │   ├── sam3_detect.py   # SAM 3 single-pass detection
│   │   ├── hybrid.py        # Hybrid V1 (YOLO + MobileSAM + CLIP)
│   │   └── hybrid_v2.py     # Hybrid V2 (YOLO-seg + CLIP) ← fastest
│   ├── enrich/sam3.py       # SAM 3 concept enrichment (sequential)
│   ├── render/video.py      # Annotated video + GIF output
│   ├── emulate/
│   │   ├── npu_emulator.py  # Bandwidth-aware edge hardware projection
│   │   ├── sam3_reference.py  # Paper-spec layer breakdown
│   │   └── layer_profiler.py  # PyTorch hook-based layer profiler
│   ├── store/               # SQLite + SQLAlchemy metadata store
│   ├── query/nlq.py         # LLM-backed natural language query
│   └── api/server.py        # FastAPI + web UI
├── scripts/
│   ├── build_deck.py        # Regenerates PowerPoint results deck
│   ├── compare_models.py    # Multi-model speed/accuracy benchmark
│   └── download_models.sh   # Model weight downloader
├── configs/
│   └── edge_mpu.json        # Edge NPU target specification
├── data/
│   ├── videos/              # Input videos (gitignored)
│   └── output/              # Annotated videos, GIFs, profiles, runs/ (gitignored)
└── third_party/             # SAM 3 source install (gitignored)
```

## Configuration

See `config/settings.py` and `.env.example`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `EXTRACT_FPS` | 1.0 | Frames per second to extract from video |
| `YOLO_MODEL` | `yolo11x.pt` | YOLO detection model |
| `YOLO_CONFIDENCE` | 0.35 | Minimum detection confidence |
| `SAM3_ENABLED` | `true` | Enable SAM 3 (sequential mode) |
| `SAM3_CONCEPTS` | *see config* | Default concept prompts |
| `LLM_BACKEND` | `anthropic` | NLQ backend: anthropic, ollama, skippy |
| `PROFILE_GPU` | `false` | Enable GPU profiling |

## Maintainer

**Kyle Fox** ([@kylefoxaustin](https://github.com/kylefoxaustin))

## License

MIT License — see LICENSE file.

## TTA — Trust the Awesomeness
