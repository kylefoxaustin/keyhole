"""
Tier 2 Enrichment — SAM 3 Concept Segmentation.

Takes YOLO detection crops and runs SAM 3's Promptable Concept Segmentation
to extract rich semantic attributes: clothing color, object type, carrying
items, vehicle make, animal breed, etc.

Also supports a fallback mode using SAM 2 or EfficientSAM3 for edge
deployment testing.

GPU profiling included for edge MPU feasibility analysis.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from PIL import Image

from src.detect.yolo import Detection, FrameDetections

logger = logging.getLogger(__name__)


@dataclass
class ConceptMatch:
    """A single concept detected by SAM 3 within a detection crop."""
    concept: str          # The text prompt concept (e.g., "red hat")
    confidence: float     # SAM 3 presence score
    mask_area_pct: float  # Percentage of crop covered by this concept's mask
    bbox: Optional[tuple[float, float, float, float]] = None  # Relative bbox within crop


@dataclass
class EnrichedDetection:
    """A YOLO detection enriched with SAM 3 concept analysis."""
    # Original YOLO data
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float

    # SAM 3 enrichment
    concepts: list[ConceptMatch] = field(default_factory=list)
    description: str = ""  # Generated natural language description
    enrichment_ms: float = 0.0

    @property
    def concept_tags(self) -> list[str]:
        """Flat list of detected concept names."""
        return [c.concept for c in self.concepts]


@dataclass
class EnrichedFrame:
    """All enriched detections for a single frame."""
    frame_number: int
    timestamp_sec: float
    source_video: str
    detections: list[EnrichedDetection] = field(default_factory=list)
    total_enrichment_ms: float = 0.0


# --- Concept Prompt Sets ---
# These define what SAM 3 looks for within each detection category.
# Organized by YOLO class to focus prompts on relevant attributes.

PERSON_CONCEPTS = [
    "hat", "cap", "helmet", "hood",
    "red shirt", "blue shirt", "white shirt", "black shirt",
    "jacket", "coat", "hoodie", "vest",
    "shorts", "pants", "skirt", "dress",
    "backpack", "bag", "purse", "briefcase",
    "sunglasses", "glasses", "mask",
    "uniform", "high-visibility vest",
    "phone", "umbrella", "walking stick",
    "child", "adult",
]

VEHICLE_CONCEPTS = [
    "car", "truck", "van", "SUV", "sedan",
    "motorcycle", "scooter", "bicycle",
    "red vehicle", "blue vehicle", "white vehicle", "black vehicle",
    "silver vehicle",
    "license plate",
    "delivery truck", "police car", "ambulance", "fire truck",
]

ANIMAL_CONCEPTS = [
    "dog", "cat", "bird", "rabbit", "deer",
    "large dog", "small dog",
    "leash", "collar",
]

GENERAL_CONCEPTS = [
    "package", "box", "bag",
    "weapon", "tool",
    "sign", "banner",
]

CLASS_CONCEPT_MAP = {
    "person": PERSON_CONCEPTS,
    "car": VEHICLE_CONCEPTS,
    "truck": VEHICLE_CONCEPTS,
    "bus": VEHICLE_CONCEPTS,
    "motorcycle": VEHICLE_CONCEPTS,
    "bicycle": VEHICLE_CONCEPTS,
    "dog": ANIMAL_CONCEPTS,
    "cat": ANIMAL_CONCEPTS,
    "bird": ANIMAL_CONCEPTS,
}


class SAM3Enricher:
    """
    SAM 3 concept segmentation enricher.

    Runs SAM 3's text-prompted detection on each YOLO crop to extract
    detailed attribute information.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        concepts: Optional[list[str]] = None,
        confidence_threshold: float = 0.3,
        profile: bool = False,
    ):
        self.device = device
        self.extra_concepts = concepts or []
        self.confidence_threshold = confidence_threshold
        self.profile = profile
        self._latencies: list[float] = []
        self.model = None
        self.processor = None

    def load_model(self):
        """
        Load SAM 3 model and processor.

        Attempts to load the full SAM 3 model. Falls back to a
        description-only mode if SAM 3 is not installed.
        """
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info("Loading SAM 3 model on %s...", self.device)
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)

            # Warm up
            dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            state = self.processor.set_image(dummy)
            self.processor.set_text_prompt(state=state, prompt="test")

            params = sum(p.numel() for p in self.model.parameters())
            logger.info("SAM 3 loaded: %.1fM parameters", params / 1e6)

        except ImportError:
            logger.warning(
                "SAM 3 not installed. Install from: "
                "https://github.com/facebookresearch/sam3\n"
                "Falling back to YOLO-only attribute extraction."
            )
            self.model = None
            self.processor = None

    def _get_concepts_for_class(self, class_name: str) -> list[str]:
        """Get relevant concept prompts for a given YOLO class."""
        concepts = CLASS_CONCEPT_MAP.get(class_name, GENERAL_CONCEPTS).copy()
        concepts.extend(self.extra_concepts)
        return concepts

    def _enrich_with_sam3(
        self, crop: np.ndarray, class_name: str
    ) -> tuple[list[ConceptMatch], float]:
        """
        Run SAM 3 concept segmentation on a detection crop.

        Returns list of matched concepts and inference time in ms.
        """
        if self.model is None or self.processor is None:
            return [], 0.0

        # Convert BGR crop to PIL
        pil_image = Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB

        concepts = self._get_concepts_for_class(class_name)
        matches = []

        t_start = time.perf_counter()

        # Set the image once, then probe with multiple concept prompts
        inference_state = self.processor.set_image(pil_image)

        for concept in concepts:
            try:
                output = self.processor.set_text_prompt(
                    state=inference_state, prompt=concept
                )

                masks = output.get("masks")
                scores = output.get("scores")

                if masks is not None and scores is not None and len(scores) > 0:
                    # Take best match for this concept
                    best_idx = scores.argmax()
                    best_score = float(scores[best_idx])

                    if best_score >= self.confidence_threshold:
                        mask = masks[best_idx]
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu().numpy()
                        mask_area_pct = float(mask.sum()) / mask.size * 100

                        # Only keep if mask covers meaningful area
                        if mask_area_pct > 0.5:
                            matches.append(ConceptMatch(
                                concept=concept,
                                confidence=best_score,
                                mask_area_pct=mask_area_pct,
                            ))

            except Exception as e:
                logger.debug("SAM 3 concept '%s' failed: %s", concept, e)
                continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_ms = (time.perf_counter() - t_start) * 1000

        # Sort by confidence, keep top matches
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:10], inference_ms

    def _generate_description(
        self, class_name: str, concepts: list[ConceptMatch]
    ) -> str:
        """Generate a natural language description from detected concepts."""
        if not concepts:
            return class_name

        concept_strs = [c.concept for c in concepts[:5]]
        if class_name == "person":
            return f"Person with: {', '.join(concept_strs)}"
        elif class_name in ("car", "truck", "bus", "motorcycle"):
            return f"Vehicle ({class_name}): {', '.join(concept_strs)}"
        elif class_name in ("dog", "cat", "bird"):
            return f"Animal ({class_name}): {', '.join(concept_strs)}"
        else:
            return f"{class_name}: {', '.join(concept_strs)}"

    def enrich_frame(self, frame_detections: FrameDetections) -> EnrichedFrame:
        """
        Enrich all detections in a frame with SAM 3 concept analysis.
        """
        enriched_dets = []
        total_ms = 0.0

        for det in frame_detections.detections:
            concepts = []
            enrich_ms = 0.0

            if det.crop is not None and det.crop.size > 0:
                concepts, enrich_ms = self._enrich_with_sam3(
                    det.crop, det.class_name
                )
                total_ms += enrich_ms

            description = self._generate_description(det.class_name, concepts)

            enriched_dets.append(EnrichedDetection(
                bbox=det.bbox,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                concepts=concepts,
                description=description,
                enrichment_ms=enrich_ms,
            ))

        self._latencies.append(total_ms)

        result = EnrichedFrame(
            frame_number=frame_detections.frame_number,
            timestamp_sec=frame_detections.timestamp_sec,
            source_video=frame_detections.source_video,
            detections=enriched_dets,
            total_enrichment_ms=total_ms,
        )

        logger.debug(
            "Frame %d: enriched %d detections in %.1fms",
            frame_detections.frame_number, len(enriched_dets), total_ms
        )

        return result

    def get_profile_metrics(self) -> dict:
        """Get aggregate profiling metrics."""
        if not self._latencies:
            return {"total_frames": 0}

        latencies = np.array(self._latencies)
        return {
            "total_frames": len(self._latencies),
            "avg_enrichment_ms": float(np.mean(latencies)),
            "p95_enrichment_ms": float(np.percentile(latencies, 95)),
            "p99_enrichment_ms": float(np.percentile(latencies, 99)),
            "model": "SAM 3" if self.model else "fallback",
            "model_params_m": (
                sum(p.numel() for p in self.model.parameters()) / 1e6
                if self.model else 0
            ),
        }
