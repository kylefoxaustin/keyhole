# Keyhole

**Open-source AI Key prototype** — an edge AI video intelligence pipeline that processes
camera detections through tiered AI models and enables natural language querying of
surveillance events.

Inspired by the Ubiquiti UniFi AI Key architecture: cameras do lightweight detection,
a separate compute node enriches those detections with advanced AI, and users query
events in plain English.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Canned      │     │  Tier 1:     │     │  Tier 2:     │     │  Metadata    │
│  Video       │────▶│  YOLO 11     │────▶│  SAM 3       │────▶│  Store       │
│  (FFmpeg)    │     │  Detection   │     │  Enrichment  │     │  (SQLite)    │
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
2. **Detect** — YOLO 11 runs Tier-1 detection: person, vehicle, animal, package, etc.
3. **Enrich** — SAM 3 processes each detection crop with concept prompts, extracting
   attributes (color, clothing, object type, pose, carrying items)
4. **Store** — Structured metadata written to SQLite with full-text search + optional
   vector embeddings for semantic search
5. **Query** — Natural language queries translated to structured searches via LLM
   (Claude API, local Ollama, or Skippy)

### Edge MPU Profiling

The prototype includes GPU profiling hooks to measure:
- Per-stage FLOP counts and memory bandwidth utilization
- Inference latency distributions (P50/P95/P99)
- Peak VRAM usage per model
- Arithmetic intensity (FLOPs/byte) for each stage

This data maps directly to edge MPU feasibility analysis for targets like:
- 200 TOPS BF16 with 128-bit LPDDR5X @ 8.4 GT/s (134.4 GB/s)
- Compare compute-bound (SAM 3) vs bandwidth-bound (LLM) workloads

## Quick Start

### Prerequisites
- NVIDIA GPU with ≥16 GB VRAM (tested on RTX 5090)
- Docker + NVIDIA Container Toolkit
- Python 3.11+

### Option A: Docker (Recommended)
```bash
# Clone and configure
git clone https://github.com/kylefoxaustin/keyhole.git
cd keyhole
cp .env.example .env
# Edit .env with your API keys and model preferences

# Download model weights
./scripts/download_models.sh

# Build and run
docker compose up --build
```

### Option B: Local Install
```bash
# Create venv
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
./scripts/download_models.sh

# Process a video
python -m src.main process --video data/videos/sample.mp4

# Start the query server
python -m src.api.server
```

### Process a Video
```bash
# Process with default settings (1 FPS extraction, all enrichment)
python -m src.main process --video data/videos/your_video.mp4

# Process at 5 FPS with profiling enabled
python -m src.main process --video data/videos/your_video.mp4 --fps 5 --profile

# Detect-only mode (skip SAM 3 enrichment for quick testing)
python -m src.main process --video data/videos/your_video.mp4 --detect-only
```

### Query Your Detections
```bash
# Interactive CLI query mode
python -m src.main query

# Single query
python -m src.main query --q "person wearing a red hat near the front door"

# Start web UI
python -m src.api.server
# Open http://localhost:8777
```

## Configuration

See `config/settings.py` for all configurable parameters. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `EXTRACT_FPS` | 1.0 | Frames per second to extract from video |
| `YOLO_MODEL` | `yolo11x.pt` | YOLO model variant |
| `YOLO_CONFIDENCE` | 0.35 | Minimum detection confidence |
| `SAM3_ENABLED` | `true` | Enable SAM 3 enrichment |
| `SAM3_CONCEPTS` | see config | Default concept prompts |
| `LLM_BACKEND` | `anthropic` | LLM for NLQ: anthropic, ollama, or skippy |
| `PROFILE_GPU` | `false` | Enable GPU profiling |

## Project Structure

```
keyhole/
├── src/
│   ├── main.py              # CLI entry point & pipeline orchestrator
│   ├── ingest/video.py      # FFmpeg frame extraction
│   ├── detect/yolo.py       # YOLO 11 detection tier
│   ├── enrich/sam3.py       # SAM 3 concept segmentation
│   ├── store/
│   │   ├── models.py        # SQLAlchemy/DB models
│   │   └── db.py            # Database operations
│   ├── query/nlq.py         # Natural language query engine
│   └── api/server.py        # FastAPI + web UI
├── config/settings.py       # Centralized configuration
├── scripts/
│   ├── download_models.sh   # Model weight downloader
│   └── profile_report.py    # GPU profiling report generator
├── data/
│   ├── videos/              # Input videos
│   └── output/              # Processing output
└── tests/
```

## Edge MPU Analysis

After processing videos with `--profile`, generate a mapping report:

```bash
python scripts/profile_report.py --target-tops 200 --target-bw 134.4
```

This produces a breakdown showing which pipeline stages are compute-bound vs
bandwidth-bound on your target silicon, with estimated per-frame latency.

## License

MIT License — see LICENSE file.

## TTA — Trust the Awesomeness
