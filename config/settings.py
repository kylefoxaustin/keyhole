"""
Keyhole — Centralized Configuration

All settings loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure output dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoConfig:
    extract_fps: float = float(os.getenv("EXTRACT_FPS", "1.0"))
    max_frames: int = int(os.getenv("MAX_FRAMES", "0"))


@dataclass
class YOLOConfig:
    model: str = os.getenv("YOLO_MODEL", "yolo11x.pt")
    confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
    iou_threshold: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    device: str = os.getenv("YOLO_DEVICE", "cuda:0")
    classes: list[int] = field(default_factory=list)

    def __post_init__(self):
        classes_str = os.getenv("YOLO_CLASSES", "")
        if classes_str.strip():
            self.classes = [int(c) for c in classes_str.split(",")]


@dataclass
class SAM3Config:
    enabled: bool = os.getenv("SAM3_ENABLED", "true").lower() == "true"
    device: str = os.getenv("SAM3_DEVICE", "cuda:0")
    concepts: list[str] = field(default_factory=list)
    confidence: float = float(os.getenv("SAM3_CONFIDENCE", "0.3"))

    def __post_init__(self):
        concepts_str = os.getenv(
            "SAM3_CONCEPTS",
            "person,vehicle,animal,package,bag,hat,uniform,bicycle,skateboard"
        )
        self.concepts = [c.strip() for c in concepts_str.split(",")]


@dataclass
class LLMConfig:
    backend: str = os.getenv("LLM_BACKEND", "anthropic")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    skippy_host: str = os.getenv("SKIPPY_HOST", "http://localhost:8000")
    skippy_model: str = os.getenv("SKIPPY_MODEL", "mixtral-8x7b")


@dataclass
class DatabaseConfig:
    url: str = os.getenv("DATABASE_URL", f"sqlite:///{OUTPUT_DIR / 'sentinel.db'}")


@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8777"))


@dataclass
class ProfileConfig:
    enabled: bool = os.getenv("PROFILE_GPU", "false").lower() == "true"
    output: str = os.getenv("PROFILE_OUTPUT", str(OUTPUT_DIR / "profile_report.json"))


@dataclass
class Settings:
    video: VideoConfig = field(default_factory=VideoConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)


# Global settings instance
settings = Settings()
