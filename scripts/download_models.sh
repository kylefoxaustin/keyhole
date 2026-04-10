#!/usr/bin/env bash
# Keyhole — Model Weight Downloader
# Downloads YOLO 11 and SAM 3 model weights

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEIGHTS_DIR="$PROJECT_DIR/weights"

mkdir -p "$WEIGHTS_DIR"

echo "============================================"
echo "  Keyhole — Model Download"
echo "============================================"
echo ""

# --- YOLO 11 ---
echo "[1/3] Downloading YOLO 11x..."
if [ -f "$WEIGHTS_DIR/yolo11x.pt" ]; then
    echo "  Already exists, skipping."
else
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11x.pt')
print('  YOLO 11x downloaded successfully')
"
    # Ultralytics downloads to current dir or ~/.config/Ultralytics
    # Move to weights dir if needed
    if [ -f "yolo11x.pt" ]; then
        mv yolo11x.pt "$WEIGHTS_DIR/"
    fi
fi

# --- SAM 3 ---
echo ""
echo "[2/3] Installing SAM 3 from source..."
SAM3_DIR="$PROJECT_DIR/third_party/sam3"

if [ -d "$SAM3_DIR" ]; then
    echo "  SAM 3 repo already cloned, pulling latest..."
    cd "$SAM3_DIR" && git pull
else
    echo "  Cloning facebookresearch/sam3..."
    mkdir -p "$PROJECT_DIR/third_party"
    cd "$PROJECT_DIR/third_party"
    git clone https://github.com/facebookresearch/sam3.git
fi

echo "  Installing SAM 3 package..."
cd "$SAM3_DIR"
pip install -e ".[notebooks]" 2>/dev/null || {
    echo "  [WARN] SAM 3 install failed — you may need to install manually."
    echo "  See: https://github.com/facebookresearch/sam3#installation"
}

echo ""
echo "[3/3] Downloading SAM 3 checkpoint..."
echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║  SAM 3 requires HuggingFace authentication.            ║"
echo "  ║                                                         ║"
echo "  ║  1. Request access at:                                  ║"
echo "  ║     https://huggingface.co/facebook/sam3                ║"
echo "  ║                                                         ║"
echo "  ║  2. Once approved, run:                                 ║"
echo "  ║     huggingface-cli login                               ║"
echo "  ║                                                         ║"
echo "  ║  3. The checkpoint (~3.2 GB) will auto-download on      ║"
echo "  ║     first use via build_sam3_image_model()              ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

# Try to verify HF auth
python3 -c "
try:
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.whoami()
    print(f'  HuggingFace authenticated as: {info[\"name\"]}')
except Exception:
    print('  [WARN] Not authenticated with HuggingFace.')
    print('  Run: huggingface-cli login')
" 2>/dev/null || true

echo ""
echo "============================================"
echo "  Download complete!"
echo ""
echo "  YOLO 11:  $WEIGHTS_DIR/yolo11x.pt"
echo "  SAM 3:    Auto-downloads on first run"
echo ""
echo "  Next steps:"
echo "    1. Place test videos in: data/videos/"
echo "    2. Run: python -m src.main process --video data/videos/test.mp4"
echo "============================================"
