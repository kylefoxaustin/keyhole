# Keyhole — GPU-accelerated Docker image
# Based on NVIDIA PyTorch container for CUDA + cuDNN + PyTorch

FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attn for SAM 3 (requires CUDA build)
RUN pip install einops ninja && \
    pip install flash-attn --no-build-isolation 2>/dev/null || \
    echo "flash-attn build skipped — install manually if needed"

# Copy project
COPY . .

# Create data directories
RUN mkdir -p data/videos data/output weights

# Default: run the API server
EXPOSE 8777
CMD ["python", "-m", "src.api.server"]
