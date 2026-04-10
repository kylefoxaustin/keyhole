"""
Tier 1 Detection — YOLO 11 object detection.

Runs YOLO inference on extracted frames, producing bounding boxes with
class labels and confidence scores. Includes GPU profiling hooks for
edge MPU feasibility analysis.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from ultralytics import YOLO

from src.ingest.video import ExtractedFrame

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection from YOLO."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    class_id: int
    class_name: str
    confidence: float
    crop: Optional[np.ndarray] = None  # Cropped image region (BGR)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    frame_number: int
    timestamp_sec: float
    source_video: str
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    gpu_mem_mb: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)


@dataclass
class ProfileMetrics:
    """GPU profiling metrics for a detection run."""
    total_frames: int = 0
    total_detections: int = 0
    avg_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    p99_inference_ms: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    model_params: int = 0
    model_flops: float = 0.0  # GFLOPs per frame


class YOLODetector:
    """
    YOLO 11 object detector with GPU profiling.

    Wraps Ultralytics YOLO with structured output and optional
    GPU utilization tracking for edge MPU analysis.
    """

    def __init__(
        self,
        model_name: str = "yolo11x.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cuda:0",
        classes: Optional[list[int]] = None,
        profile: bool = False,
    ):
        self.device = device
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.classes = classes if classes else None
        self.profile = profile
        self._latencies: list[float] = []

        logger.info("Loading YOLO model: %s on %s", model_name, device)
        self.model = YOLO(model_name)
        self.model.to(device)

        # Warm up the model
        logger.info("Warming up YOLO model...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)

        # Log model info
        params = sum(p.numel() for p in self.model.model.parameters())
        logger.info("YOLO model loaded: %.1fM parameters", params / 1e6)

    def detect_frame(self, frame: ExtractedFrame) -> FrameDetections:
        """
        Run YOLO detection on a single frame.

        Returns FrameDetections with all detected objects and their crops.
        """
        # Track GPU memory before inference
        gpu_mem_before = 0.0
        if self.profile and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_before = torch.cuda.memory_allocated() / 1e6

        # Run inference with timing
        t_start = time.perf_counter()

        results = self.model.predict(
            frame.image,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_ms = (time.perf_counter() - t_start) * 1000

        # Track GPU memory after inference
        gpu_mem_after = 0.0
        if self.profile and torch.cuda.is_available():
            gpu_mem_after = torch.cuda.max_memory_allocated() / 1e6

        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu())
                    conf = float(boxes.conf[i].cpu())
                    class_name = result.names.get(class_id, f"class_{class_id}")

                    # Extract crop from original frame
                    x1, y1, x2, y2 = map(int, bbox)
                    # Clamp to frame dimensions
                    h, w = frame.image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = frame.image[y1:y2, x1:x2].copy()

                    detections.append(Detection(
                        bbox=(float(bbox[0]), float(bbox[1]),
                              float(bbox[2]), float(bbox[3])),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        crop=crop,
                    ))

        # Track profiling
        self._latencies.append(inference_ms)

        frame_result = FrameDetections(
            frame_number=frame.frame_number,
            timestamp_sec=frame.timestamp_sec,
            source_video=frame.source_video,
            detections=detections,
            inference_ms=inference_ms,
            gpu_mem_mb=gpu_mem_after,
        )

        logger.debug(
            "Frame %d: %d detections in %.1fms",
            frame.frame_number, len(detections), inference_ms
        )

        return frame_result

    def get_profile_metrics(self) -> ProfileMetrics:
        """Get aggregate profiling metrics from all processed frames."""
        if not self._latencies:
            return ProfileMetrics()

        import numpy as np
        latencies = np.array(self._latencies)
        params = sum(p.numel() for p in self.model.model.parameters())

        return ProfileMetrics(
            total_frames=len(self._latencies),
            avg_inference_ms=float(np.mean(latencies)),
            p95_inference_ms=float(np.percentile(latencies, 95)),
            p99_inference_ms=float(np.percentile(latencies, 99)),
            peak_gpu_mem_mb=max(self._latencies) if self.profile else 0,
            model_params=params,
        )

    def reset_profile(self):
        """Reset profiling metrics."""
        self._latencies.clear()
