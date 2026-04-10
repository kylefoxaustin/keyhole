"""
Video Ingestion — FFmpeg-based frame extraction.

Extracts frames from video files at configurable FPS, preserving timestamps
for correlation with detection events.
"""

import subprocess
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    """Metadata extracted from video file."""
    path: str
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    codec: str


@dataclass
class ExtractedFrame:
    """A single extracted frame with timestamp info."""
    frame_number: int
    timestamp_sec: float
    image: np.ndarray  # BGR numpy array
    source_video: str


def probe_video(video_path: str | Path) -> VideoMeta:
    """Extract video metadata using ffprobe."""
    video_path = str(video_path)

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to probe video: {video_path}") from e

    # Find the video stream
    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise RuntimeError(f"No video stream found in: {video_path}")

    # Parse FPS from r_frame_rate (e.g., "30/1" or "30000/1001")
    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

    duration = float(info.get("format", {}).get("duration", 0))

    return VideoMeta(
        path=video_path,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        duration=duration,
        total_frames=int(video_stream.get("nb_frames", int(fps * duration))),
        codec=video_stream.get("codec_name", "unknown"),
    )


def extract_frames(
    video_path: str | Path,
    target_fps: float = 1.0,
    max_frames: int = 0,
) -> Generator[ExtractedFrame, None, None]:
    """
    Extract frames from video at target FPS using OpenCV.

    Yields ExtractedFrame objects with BGR images and timestamps.

    Args:
        video_path: Path to video file
        target_fps: Desired extraction rate (frames per second)
        max_frames: Maximum frames to extract (0 = no limit)

    Yields:
        ExtractedFrame for each extracted frame
    """
    video_path = str(video_path)
    meta = probe_video(video_path)

    logger.info(
        "Ingesting video: %s (%dx%d, %.1f FPS, %.1fs duration)",
        video_path, meta.width, meta.height, meta.fps, meta.duration
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Calculate frame interval based on source FPS and target FPS
    source_fps = cap.get(cv2.CAP_PROP_FPS) or meta.fps
    frame_interval = max(1, int(source_fps / target_fps))

    frame_idx = 0
    extracted_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / source_fps

                yield ExtractedFrame(
                    frame_number=frame_idx,
                    timestamp_sec=timestamp,
                    image=frame,
                    source_video=video_path,
                )

                extracted_count += 1
                if max_frames > 0 and extracted_count >= max_frames:
                    logger.info("Reached max_frames limit: %d", max_frames)
                    break

            frame_idx += 1

    finally:
        cap.release()

    logger.info(
        "Extracted %d frames from %d total (target %.1f FPS)",
        extracted_count, frame_idx, target_fps
    )
