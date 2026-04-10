"""
Hybrid Pipeline — YOLO Detection + MobileSAM Segmentation + CLIP Classification

Combines three lightweight models to approximate SAM 3's capabilities:
  1. YOLO 11x: Fast object detection (bounding boxes + class labels)
  2. MobileSAM: Lightweight segmentation (masks from YOLO boxes)
  3. OpenCLIP: Zero-shot concept classification (attributes from crops)

Each model plays to its strength:
  - YOLO: 12ms detection, 80 COCO classes, proven reliability
  - MobileSAM: 5-10ms segmentation per box, 9.7M params vs SAM 3's 840M
  - CLIP: ~2ms per crop for open-vocabulary classification

Total projected: ~25-40ms per frame on RTX 5090 (25-40 FPS)
vs SAM 3 single-pass: ~130ms per frame (7.6 FPS)

Usage:
    python -m src.main process --video input.mp4 --hybrid --render --profile
"""

import time
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

from src.ingest.video import ExtractedFrame
from src.enrich.sam3 import EnrichedFrame, EnrichedDetection, ConceptMatch

logger = logging.getLogger(__name__)

# CLIP concept prompts for classification
PERSON_ATTRIBUTES = [
    "person wearing hat", "person wearing backpack", "person wearing jacket",
    "person wearing uniform", "person with glasses", "person carrying bag",
    "child", "adult walking", "person on phone",
]
VEHICLE_ATTRIBUTES = [
    "red vehicle", "blue vehicle", "white vehicle", "black vehicle",
    "silver vehicle", "delivery truck", "police car", "SUV", "sedan",
]
GENERAL_ATTRIBUTES = [
    "package", "box", "bag", "bicycle", "skateboard", "umbrella",
]


class HybridDetector:
    """
    Three-model hybrid: YOLO + MobileSAM + CLIP.

    Significantly faster than SAM 3 while providing detection,
    segmentation, and open-vocabulary concept classification.
    """

    def __init__(
        self,
        yolo_model: str = "yolo11x.pt",
        mobile_sam_checkpoint: str = "weights/mobile_sam.pt",
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        yolo_confidence: float = 0.35,
        clip_top_k: int = 3,
        device: str = "cuda:0",
        profile: bool = False,
        retain_masks: bool = False,
    ):
        self.yolo_model_name = yolo_model
        self.mobile_sam_checkpoint = mobile_sam_checkpoint
        self.clip_model_name = clip_model
        self.clip_pretrained = clip_pretrained
        self.yolo_confidence = yolo_confidence
        self.clip_top_k = clip_top_k
        self.device = device
        self.profile = profile
        self.retain_masks = retain_masks

        self._latencies: list[float] = []
        self._yolo_times: list[float] = []
        self._sam_times: list[float] = []
        self._clip_times: list[float] = []
        self._det_counts: list[int] = []

        self.yolo = None
        self.sam_predictor = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None

    def load_model(self):
        """Load all three models."""
        import open_clip
        from ultralytics import YOLO
        from mobile_sam import sam_model_registry, SamPredictor

        # --- YOLO ---
        logger.info("Loading YOLO %s...", self.yolo_model_name)
        self.yolo = YOLO(self.yolo_model_name)
        self.yolo.to(self.device)
        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo.predict(dummy, verbose=False)

        yolo_params = sum(p.numel() for p in self.yolo.model.parameters())
        logger.info("YOLO loaded: %.1fM params", yolo_params / 1e6)

        # --- MobileSAM ---
        logger.info("Loading MobileSAM from %s...", self.mobile_sam_checkpoint)
        sam = sam_model_registry["vit_t"](checkpoint=self.mobile_sam_checkpoint)
        sam.to(device=self.device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)

        sam_params = sum(p.numel() for p in sam.parameters())
        logger.info("MobileSAM loaded: %.1fM params", sam_params / 1e6)

        # --- OpenCLIP ---
        logger.info("Loading CLIP %s...", self.clip_model_name)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained, device=self.device,
        )
        self.clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        self.clip_model.eval()

        clip_params = sum(p.numel() for p in self.clip_model.parameters())
        logger.info("CLIP loaded: %.1fM params", clip_params / 1e6)

        total_params = yolo_params + sam_params + clip_params
        logger.info(
            "Hybrid pipeline ready: %.1fM total params (YOLO %.1fM + MobileSAM %.1fM + CLIP %.1fM)",
            total_params / 1e6, yolo_params / 1e6, sam_params / 1e6, clip_params / 1e6,
        )

    def _get_clip_prompts(self, class_name: str) -> list[str]:
        """Get relevant CLIP prompts for a YOLO class."""
        if class_name == "person":
            return PERSON_ATTRIBUTES
        elif class_name in ("car", "truck", "bus", "motorcycle"):
            return VEHICLE_ATTRIBUTES
        return GENERAL_ATTRIBUTES

    def _classify_crops_batched(
        self, crops_pil: list[Optional[Image.Image]], class_names: list[str],
    ) -> list[list[tuple[str, float]]]:
        """
        Batch CLIP classification across all crops in a frame.

        Text features are cached per class (same prompts for all "person" crops).
        Image features are batched into a single encode_image call.
        """
        if not crops_pil:
            return []

        # Cache text features per class
        if not hasattr(self, "_text_cache"):
            self._text_cache = {}

        # Pre-encode text for each unique class
        for cls in set(class_names):
            if cls not in self._text_cache:
                prompts = self._get_clip_prompts(cls)
                if prompts:
                    text_tokens = self.clip_tokenizer(prompts).to(self.device)
                    with torch.no_grad():
                        text_features = self.clip_model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                    self._text_cache[cls] = (prompts, text_features)

        # Batch all valid crops into one image tensor
        valid_indices = []
        batch_tensors = []
        for i, crop in enumerate(crops_pil):
            if crop is not None:
                batch_tensors.append(self.clip_preprocess(crop))
                valid_indices.append(i)

        if not batch_tensors:
            return [[] for _ in crops_pil]

        image_batch = torch.stack(batch_tensors).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # Match each crop's features against its class text features
        all_results = [[] for _ in crops_pil]

        for batch_idx, orig_idx in enumerate(valid_indices):
            cls = class_names[orig_idx]
            if cls not in self._text_cache:
                continue

            prompts, text_features = self._text_cache[cls]
            img_feat = image_features[batch_idx:batch_idx + 1]

            similarity = (img_feat @ text_features.T).squeeze(0)
            probs = similarity.softmax(dim=-1)

            top_k = min(self.clip_top_k, len(prompts))
            top_probs, top_indices = probs.topk(top_k)

            results = []
            for prob, idx in zip(top_probs, top_indices):
                results.append((prompts[idx], float(prob)))
            all_results[orig_idx] = results

        return all_results

    def detect_frame(self, frame: ExtractedFrame) -> EnrichedFrame:
        """Run the full hybrid pipeline on one frame."""
        t_total_start = time.perf_counter()

        image_rgb = frame.image[:, :, ::-1]  # BGR→RGB
        h, w = frame.image.shape[:2]

        # === Stage 1: YOLO Detection ===
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_yolo = time.perf_counter()

        yolo_results = self.yolo.predict(
            frame.image, conf=self.yolo_confidence, verbose=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        yolo_ms = (time.perf_counter() - t_yolo) * 1000

        # Parse YOLO detections
        boxes = []
        class_names = []
        confidences = []
        crops_pil = []

        if yolo_results and len(yolo_results) > 0:
            r = yolo_results[0]
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    bbox = r.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    cls_id = int(r.boxes.cls[i].cpu())
                    cls_name = r.names.get(cls_id, f"class_{cls_id}")
                    conf = float(r.boxes.conf[i].cpu())

                    boxes.append([float(bbox[0]), float(bbox[1]),
                                  float(bbox[2]), float(bbox[3])])
                    class_names.append(cls_name)
                    confidences.append(conf)

                    # Crop for CLIP
                    crop = frame.image[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops_pil.append(Image.fromarray(crop[:, :, ::-1]))
                    else:
                        crops_pil.append(None)

        # === Stage 2: MobileSAM Segmentation ===
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_sam = time.perf_counter()

        masks = []
        # MobileSAM requires FP32 — disable any active autocast
        with torch.amp.autocast("cuda", enabled=False):
            self.sam_predictor.set_image(image_rgb)

            for box in boxes:
                input_box = np.array(box)
                sam_masks, sam_scores, _ = self.sam_predictor.predict(
                    point_coords=None, point_labels=None,
                    box=input_box[None, :], multimask_output=False,
                )
                if sam_masks is not None and len(sam_masks) > 0:
                    mask = sam_masks[0].astype(bool)
                    masks.append(mask if self.retain_masks else None)
                else:
                    masks.append(None)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sam_ms = (time.perf_counter() - t_sam) * 1000

        # === Stage 3: CLIP Classification (batched) ===
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_clip = time.perf_counter()

        clip_results = self._classify_crops_batched(crops_pil, class_names)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        clip_ms = (time.perf_counter() - t_clip) * 1000

        total_ms = (time.perf_counter() - t_total_start) * 1000

        # === Build enriched output ===
        detections = []
        for i in range(len(boxes)):
            concepts = []
            for attr_name, attr_score in clip_results[i]:
                mask_area = 0.0
                if masks[i] is not None:
                    mask_area = float(masks[i].sum()) / masks[i].size * 100
                concepts.append(ConceptMatch(
                    concept=attr_name,
                    confidence=attr_score,
                    mask_area_pct=mask_area,
                    mask=masks[i] if concepts == [] else None,  # Only first concept gets mask
                ))

            # Build description
            attr_strs = [c.concept for c in concepts[:3]]
            desc = f"{class_names[i]}: {', '.join(attr_strs)}" if attr_strs else class_names[i]

            detections.append(EnrichedDetection(
                bbox=tuple(boxes[i]),
                class_id=0,
                class_name=class_names[i],
                confidence=confidences[i],
                concepts=concepts,
                description=desc,
                enrichment_ms=sam_ms / max(len(boxes), 1) + clip_ms / max(len(boxes), 1),
            ))

        # Track profiling
        self._latencies.append(total_ms)
        self._yolo_times.append(yolo_ms)
        self._sam_times.append(sam_ms)
        self._clip_times.append(clip_ms)
        self._det_counts.append(len(detections))

        enriched = EnrichedFrame(
            frame_number=frame.frame_number,
            timestamp_sec=frame.timestamp_sec,
            source_video=frame.source_video,
            detections=detections,
            total_enrichment_ms=total_ms,
        )

        logger.debug(
            "Frame %d: %d dets in %.1fms (YOLO=%.1f + SAM=%.1f + CLIP=%.1f)",
            frame.frame_number, len(detections), total_ms,
            yolo_ms, sam_ms, clip_ms,
        )

        return enriched

    def get_profile_metrics(self) -> dict:
        """Get profiling metrics."""
        if not self._latencies:
            return {"total_frames": 0, "mode": "hybrid"}

        latencies = np.array(self._latencies)

        return {
            "total_frames": len(self._latencies),
            "avg_inference_ms": float(np.mean(latencies)),
            "p95_inference_ms": float(np.percentile(latencies, 95)),
            "p99_inference_ms": float(np.percentile(latencies, 99)),
            "avg_yolo_ms": float(np.mean(self._yolo_times)),
            "avg_mobilesam_ms": float(np.mean(self._sam_times)),
            "avg_clip_ms": float(np.mean(self._clip_times)),
            "avg_detections_per_frame": float(np.mean(self._det_counts)),
            "mode": "hybrid",
            "model": "YOLO 11x + MobileSAM + OpenCLIP",
            "models": {
                "yolo": {"name": self.yolo_model_name, "role": "detection"},
                "mobilesam": {"name": "MobileSAM (vit_t)", "role": "segmentation"},
                "clip": {"name": f"OpenCLIP {self.clip_model_name}", "role": "classification"},
            },
        }
