"""
Single-Pass SAM 3 Detector — Full-Frame Detection + Segmentation + Concepts

Feeds full frames directly into SAM 3's batched inference API, using the
DETR detector head to find all objects and classify all concepts in a
single forward pass. This is how SAM 3 was designed to be used.

Replaces the YOLO→SAM3 sequential pipeline with:
    Frame → SAM 3 (one forward pass) → all detections + masks + concept labels

Reference: SAM 3 paper reports 30ms per image on H200 with 100+ objects.

Usage:
    detector = SAM3SinglePassDetector(concepts=["person", "car", "dog", ...])
    detector.load_model()
    enriched = detector.detect_frame(frame)
"""

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

from src.ingest.video import ExtractedFrame
from src.enrich.sam3 import EnrichedFrame, EnrichedDetection, ConceptMatch

logger = logging.getLogger(__name__)


@dataclass
class SinglePassMetrics:
    """Profiling metrics for single-pass detection."""
    total_frames: int = 0
    avg_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    p99_inference_ms: float = 0.0
    model_params_m: float = 0.0
    avg_detections_per_frame: float = 0.0


class SAM3SinglePassDetector:
    """
    Single-pass detector using SAM 3's batched inference API.

    Sends all concept prompts for a full frame in one forward pass
    through the DETR detector head. The vision backbone runs once,
    text encoder batches all prompts, and the detector produces all
    bounding boxes + masks + concept classifications simultaneously.
    """

    def __init__(
        self,
        concepts: Optional[list[str]] = None,
        detection_threshold: float = 0.3,
        device: str = "cuda:0",
        profile: bool = False,
        retain_masks: bool = False,
    ):
        self.concepts = concepts or [
            "person", "vehicle", "car", "truck", "bus", "motorcycle", "bicycle",
            "dog", "cat", "bird", "animal",
            "backpack", "bag", "hat", "umbrella",
            "package", "box", "suitcase",
        ]
        self.detection_threshold = detection_threshold
        self.device = device
        self.profile = profile
        self.retain_masks = retain_masks
        self._latencies: list[float] = []
        self._det_counts: list[int] = []

        self.model = None
        self.transform = None
        self.postprocessor = None

    def load_model(self):
        """Load SAM 3 model, transforms, and postprocessor for batched inference."""
        from sam3 import build_sam3_image_model
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI,
        )
        from sam3.eval.postprocessors import PostProcessImage
        from sam3.train.data.collator import collate_fn_api

        logger.info("Loading SAM 3 single-pass detector on %s...", self.device)

        # Enable bf16 autocast and inference mode (per official examples)
        # These must persist for the lifetime of the detector
        self._autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        self._autocast_ctx.__enter__()
        self._inference_ctx = torch.inference_mode()
        self._inference_ctx.__enter__()

        self.model = build_sam3_image_model()

        self.transform = ComposeAPI(transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=self.detection_threshold,
            to_cpu=True,
        )

        # Store collate function
        self._collate = collate_fn_api

        # Warmup
        logger.info("Warming up SAM 3 single-pass (first run includes compilation)...")
        dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        self._run_single_pass(dummy, ["test"])

        params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "SAM 3 single-pass ready: %.1fM params, %d concepts",
            params / 1e6, len(self.concepts),
        )

    def _build_datapoint(self, pil_image: Image.Image, concepts: list[str]):
        """Build a SAM 3 datapoint with all concept prompts for one image."""
        from sam3.train.data.sam3_image_dataset import (
            InferenceMetadata, FindQueryLoaded,
            Image as SAMImage, Datapoint,
        )

        w, h = pil_image.size
        datapoint = Datapoint(find_queries=[], images=[])
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

        query_ids = {}
        for i, concept in enumerate(concepts):
            query_id = i + 1
            query_ids[query_id] = concept
            datapoint.find_queries.append(
                FindQueryLoaded(
                    query_text=concept,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=query_id,
                        original_image_id=query_id,
                        original_category_id=1,
                        original_size=[w, h],
                        object_id=0,
                        frame_index=0,
                    ),
                )
            )

        return datapoint, query_ids

    def _run_single_pass(
        self, pil_image: Image.Image, concepts: list[str],
    ) -> dict:
        """
        Run SAM 3 single-pass inference on one image with all concepts.

        Returns dict mapping query_id → {scores, boxes, masks, labels}.
        """
        from sam3.model.utils.misc import copy_data_to_device

        datapoint, query_ids = self._build_datapoint(pil_image, concepts)
        datapoint = self.transform(datapoint)

        batch = self._collate([datapoint], dict_key="d")["d"]
        batch = copy_data_to_device(batch, torch.device(self.device), non_blocking=True)

        output = self.model(batch)
        results = self.postprocessor.process_results(output, batch.find_metadatas)

        return results, query_ids

    def detect_frame(self, frame: ExtractedFrame) -> EnrichedFrame:
        """
        Detect and enrich all objects in a frame with a single SAM 3 forward pass.

        Returns an EnrichedFrame with all detections, masks, and concept labels.
        """
        # Convert BGR (OpenCV) to RGB PIL
        pil_image = Image.fromarray(frame.image[:, :, ::-1])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        results, query_ids = self._run_single_pass(pil_image, self.concepts)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - t_start) * 1000

        # Collect all detections across all concept queries
        # Each concept query produces its own set of detections
        all_detections = []

        for query_id, concept_name in query_ids.items():
            if query_id not in results:
                continue

            result = results[query_id]
            scores = result.get("scores", torch.tensor([]))
            boxes = result.get("boxes", torch.tensor([]))
            masks = result.get("masks", None)

            if len(scores) == 0:
                continue

            for det_idx in range(len(scores)):
                score = float(scores[det_idx])
                box = boxes[det_idx].tolist()  # [x1, y1, x2, y2]

                mask_arr = None
                mask_area_pct = 0.0
                if masks is not None and det_idx < len(masks):
                    m = masks[det_idx]
                    if hasattr(m, 'numpy'):
                        m = m.numpy()
                    # Ensure 2D bool mask (H, W)
                    m_np = np.array(m).squeeze()
                    if m_np.ndim > 2:
                        m_np = m_np[0]  # Take first channel if multi-channel
                    mask_area_pct = float(m_np.sum()) / max(m_np.size, 1) * 100
                    mask_arr = m_np.astype(bool) if self.retain_masks else None

                concept_match = ConceptMatch(
                    concept=concept_name,
                    confidence=score,
                    mask_area_pct=mask_area_pct,
                    mask=mask_arr,
                )

                all_detections.append(EnrichedDetection(
                    bbox=(box[0], box[1], box[2], box[3]),
                    class_id=0,
                    class_name=concept_name,
                    confidence=score,
                    concepts=[concept_match],
                    description=f"{concept_name} ({score:.0%})",
                    enrichment_ms=inference_ms / max(len(query_ids), 1),
                ))

        # Deduplicate overlapping detections (same object found by multiple concepts)
        all_detections = self._merge_overlapping(all_detections)

        self._latencies.append(inference_ms)
        self._det_counts.append(len(all_detections))

        enriched = EnrichedFrame(
            frame_number=frame.frame_number,
            timestamp_sec=frame.timestamp_sec,
            source_video=frame.source_video,
            detections=all_detections,
            total_enrichment_ms=inference_ms,
        )

        logger.debug(
            "Frame %d: %d detections in %.1fms (single-pass)",
            frame.frame_number, len(all_detections), inference_ms,
        )

        return enriched

    def _merge_overlapping(
        self, detections: list[EnrichedDetection], iou_threshold: float = 0.7,
    ) -> list[EnrichedDetection]:
        """
        Merge detections that overlap significantly (same object, different concepts).

        When multiple concepts detect the same bounding box region, merge them
        into a single detection with multiple concept tags.
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        merged = []
        used = set()

        for i, det_i in enumerate(detections):
            if i in used:
                continue

            # Find all overlapping detections
            group_concepts = list(det_i.concepts)
            best_mask = det_i.concepts[0].mask if det_i.concepts else None

            for j, det_j in enumerate(detections[i + 1:], start=i + 1):
                if j in used:
                    continue

                iou = self._compute_iou(det_i.bbox, det_j.bbox)
                if iou >= iou_threshold:
                    # Same object, different concept — merge
                    group_concepts.extend(det_j.concepts)
                    used.add(j)

            # Build merged detection
            # Use the primary concept (highest confidence) as class name
            primary = group_concepts[0]
            description_parts = [
                f"{c.concept} ({c.confidence:.0%})"
                for c in sorted(group_concepts, key=lambda c: c.confidence, reverse=True)[:5]
            ]

            merged.append(EnrichedDetection(
                bbox=det_i.bbox,
                class_id=det_i.class_id,
                class_name=primary.concept,
                confidence=det_i.confidence,
                concepts=group_concepts,
                description=f"{primary.concept}: {', '.join(c.concept for c in group_concepts[:5])}",
                enrichment_ms=det_i.enrichment_ms,
            ))
            used.add(i)

        return merged

    @staticmethod
    def _compute_iou(box_a, box_b) -> float:
        """Compute IoU between two (x1,y1,x2,y2) boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def get_profile_metrics(self) -> dict:
        """Get profiling metrics."""
        if not self._latencies:
            return {"total_frames": 0, "mode": "single-pass"}

        latencies = np.array(self._latencies)
        params = sum(p.numel() for p in self.model.parameters()) if self.model else 0

        return {
            "total_frames": len(self._latencies),
            "avg_inference_ms": float(np.mean(latencies)),
            "p95_inference_ms": float(np.percentile(latencies, 95)),
            "p99_inference_ms": float(np.percentile(latencies, 99)),
            "model_params_m": params / 1e6,
            "avg_detections_per_frame": float(np.mean(self._det_counts)),
            "num_concepts": len(self.concepts),
            "mode": "single-pass",
            "model": "SAM 3 (single-pass batched)",
        }
