"""
Hybrid V2 — YOLO-Seg + CLIP (Two-Model Pipeline)

Eliminates MobileSAM entirely by using YOLO's built-in segmentation.
YOLO-seg does detection + instance segmentation in one pass (~3-8ms),
then CLIP classifies attributes on crops (~30ms batched).

Target: 15+ FPS on edge hardware.

Variants tested:
  yolo11n-seg (2.9M):  3.3ms — nano, fastest
  yolo11s-seg (10.1M): 3.4ms — small, good balance
  yolo11m-seg (22.4M): 4.3ms — medium, more accurate
  yolo11x-seg (62.1M): 7.7ms — xlarge, best accuracy
"""

import time
import logging
from typing import Optional

import torch
import numpy as np
from PIL import Image

from src.ingest.video import ExtractedFrame
from src.enrich.sam3 import EnrichedFrame, EnrichedDetection, ConceptMatch

logger = logging.getLogger(__name__)

PERSON_ATTRIBUTES = [
    "person wearing hat", "person wearing backpack", "person wearing jacket",
    "person wearing uniform", "person with glasses", "person carrying bag",
    "child", "adult walking", "person on phone",
]
VEHICLE_ATTRIBUTES = [
    "red vehicle", "blue vehicle", "white vehicle", "black vehicle",
    "delivery truck", "police car", "SUV", "sedan",
]
GENERAL_ATTRIBUTES = [
    "package", "box", "bag", "bicycle", "skateboard", "umbrella",
]


class HybridV2Detector:
    """
    Two-model hybrid: YOLO-seg + OpenCLIP.

    YOLO-seg replaces both YOLO detection and MobileSAM segmentation
    in a single model pass. CLIP provides open-vocabulary classification.
    """

    def __init__(
        self,
        yolo_seg_model: str = "yolo11s-seg.pt",
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        yolo_confidence: float = 0.35,
        clip_top_k: int = 3,
        device: str = "cuda:0",
        profile: bool = False,
        retain_masks: bool = False,
        skip_clip_classes: Optional[list[str]] = None,
    ):
        self.yolo_seg_model_name = yolo_seg_model
        self.clip_model_name = clip_model
        self.clip_pretrained = clip_pretrained
        self.yolo_confidence = yolo_confidence
        self.clip_top_k = clip_top_k
        self.device = device
        self.profile = profile
        self.retain_masks = retain_masks
        # Classes to skip CLIP for (just use YOLO label)
        self.skip_clip_classes = set(skip_clip_classes or [])

        self._latencies = []
        self._yolo_times = []
        self._clip_times = []
        self._det_counts = []
        self._text_cache = {}

        self.yolo = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None

    def load_model(self):
        """Load YOLO-seg and CLIP."""
        import open_clip
        from ultralytics import YOLO

        logger.info("Loading YOLO-seg %s...", self.yolo_seg_model_name)
        self.yolo = YOLO(self.yolo_seg_model_name)
        self.yolo.to(self.device)

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo.predict(dummy, verbose=False)

        yolo_params = sum(p.numel() for p in self.yolo.model.parameters())
        logger.info("YOLO-seg loaded: %.1fM params", yolo_params / 1e6)

        logger.info("Loading CLIP %s...", self.clip_model_name)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained, device=self.device,
        )
        self.clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        self.clip_model.eval()

        clip_params = sum(p.numel() for p in self.clip_model.parameters())
        logger.info("CLIP loaded: %.1fM params", clip_params / 1e6)

        total = yolo_params + clip_params
        logger.info("Hybrid V2 ready: %.1fM total (YOLO-seg %.1fM + CLIP %.1fM)",
                     total / 1e6, yolo_params / 1e6, clip_params / 1e6)

    def _get_clip_prompts(self, class_name: str) -> list[str]:
        if class_name == "person":
            return PERSON_ATTRIBUTES
        elif class_name in ("car", "truck", "bus", "motorcycle"):
            return VEHICLE_ATTRIBUTES
        return GENERAL_ATTRIBUTES

    def _classify_crops_batched(self, crops_pil, class_names):
        """Batch CLIP with text caching."""
        if not crops_pil:
            return []

        for cls in set(class_names):
            if cls not in self._text_cache and cls not in self.skip_clip_classes:
                prompts = self._get_clip_prompts(cls)
                if prompts:
                    text_tokens = self.clip_tokenizer(prompts).to(self.device)
                    with torch.no_grad():
                        text_features = self.clip_model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                    self._text_cache[cls] = (prompts, text_features)

        valid_indices = []
        batch_tensors = []
        for i, (crop, cls) in enumerate(zip(crops_pil, class_names)):
            if crop is not None and cls not in self.skip_clip_classes:
                batch_tensors.append(self.clip_preprocess(crop))
                valid_indices.append(i)

        all_results = [[] for _ in crops_pil]

        if not batch_tensors:
            return all_results

        image_batch = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        for batch_idx, orig_idx in enumerate(valid_indices):
            cls = class_names[orig_idx]
            if cls not in self._text_cache:
                continue
            prompts, text_features = self._text_cache[cls]
            similarity = (image_features[batch_idx:batch_idx + 1] @ text_features.T).squeeze(0)
            probs = similarity.softmax(dim=-1)
            top_k = min(self.clip_top_k, len(prompts))
            top_probs, top_indices = probs.topk(top_k)
            all_results[orig_idx] = [(prompts[idx], float(prob)) for prob, idx in zip(top_probs, top_indices)]

        return all_results

    def detect_frame(self, frame: ExtractedFrame) -> EnrichedFrame:
        """Run YOLO-seg + CLIP on one frame."""
        t_total = time.perf_counter()
        h, w = frame.image.shape[:2]

        # === YOLO-seg: detection + segmentation in one pass ===
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_yolo = time.perf_counter()

        results = self.yolo.predict(frame.image, conf=self.yolo_confidence, verbose=False)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        yolo_ms = (time.perf_counter() - t_yolo) * 1000

        boxes, class_names, confidences, masks_list, crops_pil = [], [], [], [], []

        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    bbox = r.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    cls_id = int(r.boxes.cls[i].cpu())
                    cls_name = r.names.get(cls_id, f"class_{cls_id}")

                    boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
                    class_names.append(cls_name)
                    confidences.append(float(r.boxes.conf[i].cpu()))

                    # Mask from YOLO-seg
                    if r.masks is not None and i < len(r.masks.data):
                        mask = r.masks.data[i].cpu().numpy().astype(bool)
                        masks_list.append(mask if self.retain_masks else None)
                    else:
                        masks_list.append(None)

                    # Crop for CLIP
                    crop = frame.image[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops_pil.append(Image.fromarray(crop[:, :, ::-1]))
                    else:
                        crops_pil.append(None)

        # === CLIP classification (batched) ===
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_clip = time.perf_counter()

        clip_results = self._classify_crops_batched(crops_pil, class_names)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        clip_ms = (time.perf_counter() - t_clip) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        # Build enriched output
        detections = []
        for i in range(len(boxes)):
            concepts = []
            for attr_name, attr_score in clip_results[i]:
                concepts.append(ConceptMatch(
                    concept=attr_name, confidence=attr_score,
                    mask_area_pct=0.0,
                    mask=masks_list[i] if not concepts else None,
                ))

            attr_strs = [c.concept for c in concepts[:3]]
            desc = f"{class_names[i]}: {', '.join(attr_strs)}" if attr_strs else class_names[i]

            detections.append(EnrichedDetection(
                bbox=tuple(boxes[i]), class_id=0,
                class_name=class_names[i], confidence=confidences[i],
                concepts=concepts, description=desc,
                enrichment_ms=clip_ms / max(len(boxes), 1),
            ))

        self._latencies.append(total_ms)
        self._yolo_times.append(yolo_ms)
        self._clip_times.append(clip_ms)
        self._det_counts.append(len(detections))

        return EnrichedFrame(
            frame_number=frame.frame_number,
            timestamp_sec=frame.timestamp_sec,
            source_video=frame.source_video,
            detections=detections,
            total_enrichment_ms=total_ms,
        )

    def get_profile_metrics(self) -> dict:
        if not self._latencies:
            return {"total_frames": 0, "mode": "hybrid-v2"}
        latencies = np.array(self._latencies)
        return {
            "total_frames": len(self._latencies),
            "avg_inference_ms": float(np.mean(latencies)),
            "p95_inference_ms": float(np.percentile(latencies, 95)),
            "p99_inference_ms": float(np.percentile(latencies, 99)),
            "avg_yolo_seg_ms": float(np.mean(self._yolo_times)),
            "avg_clip_ms": float(np.mean(self._clip_times)),
            "avg_detections_per_frame": float(np.mean(self._det_counts)),
            "mode": "hybrid-v2",
            "model": f"YOLO-seg ({self.yolo_seg_model_name}) + OpenCLIP",
        }
