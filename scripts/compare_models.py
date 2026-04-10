"""
Keyhole — Model Comparison Framework

Runs multiple SAM variants on the same video frames and compares:
- Latency (wall clock + GPU kernel)
- Detection count and recall vs SAM 3 baseline
- Segmentation mask IoU vs SAM 3 baseline
- Concept/class coverage
- Memory usage
- Edge NPU projections

Usage:
    python scripts/compare_models.py --video data/videos/720p_EW_clip.mp4 --max-frames 5
"""

import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import torch
import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Results from running one model on one frame."""
    boxes: list  # List of [x1, y1, x2, y2]
    masks: list  # List of HxW bool arrays (or None)
    scores: list  # List of confidence floats
    labels: list  # List of class/concept strings
    inference_ms: float = 0.0
    gpu_kernel_ms: float = 0.0
    peak_vram_bytes: int = 0


@dataclass
class ModelProfile:
    """Aggregate profile for a model across all frames."""
    name: str
    param_count_m: float
    supports_text_prompts: bool
    concept_vocabulary: str  # "4M+" or "80 COCO" or "CLIP open-vocab"

    avg_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    avg_gpu_kernel_ms: float = 0.0
    peak_vram_gb: float = 0.0

    avg_detections: float = 0.0
    recall_vs_sam3: float = 0.0  # % of SAM 3 detections matched
    avg_mask_iou: float = 0.0  # Average IoU with SAM 3 masks
    false_positive_rate: float = 0.0

    edge_projected_ms: float = 0.0
    edge_fps: float = 0.0


def compute_iou_box(box_a, box_b) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def compute_iou_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary masks."""
    if mask_a is None or mask_b is None:
        return 0.0
    # Resize to same shape if needed
    if mask_a.shape != mask_b.shape:
        from PIL import Image as PILImage
        mask_b_resized = np.array(PILImage.fromarray(mask_b.astype(np.uint8)).resize(
            (mask_a.shape[1], mask_a.shape[0]), PILImage.NEAREST)).astype(bool)
        mask_b = mask_b_resized
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def match_detections(ref_boxes, test_boxes, iou_threshold=0.5):
    """Match test detections to reference detections by IoU."""
    matched = []  # (ref_idx, test_idx, iou)
    used_test = set()

    for ref_idx, ref_box in enumerate(ref_boxes):
        best_iou = 0.0
        best_test_idx = -1
        for test_idx, test_box in enumerate(test_boxes):
            if test_idx in used_test:
                continue
            iou = compute_iou_box(ref_box, test_box)
            if iou > best_iou:
                best_iou = iou
                best_test_idx = test_idx

        if best_iou >= iou_threshold and best_test_idx >= 0:
            matched.append((ref_idx, best_test_idx, best_iou))
            used_test.add(best_test_idx)

    return matched


def project_to_edge(gpu_kernel_ms: float) -> tuple:
    """Project GPU kernel time to NXP Edge NPU. Returns (projected_ms, fps)."""
    if gpu_kernel_ms <= 0:
        return 0, 0
    compute_frac = 0.023
    bw_frac = 0.977
    compute_ms = gpu_kernel_ms * compute_frac * (146 / 120)
    bw_ms = gpu_kernel_ms * bw_frac * (1523 / 101)
    total = compute_ms + bw_ms + 5
    return total, 1000 / total


# ============================================================
# Model Runners
# ============================================================

def run_sam3_single_pass(frames: list, concepts: list) -> list[ModelResult]:
    """Run SAM 3 single-pass on frames."""
    from sam3 import build_sam3_image_model
    from sam3.train.data.sam3_image_dataset import (
        InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint,
    )
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI,
    )
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.train.data.collator import collate_fn_api
    from sam3.model.utils.misc import copy_data_to_device

    console.print("  Loading SAM 3...")
    # Must enter autocast here — click destroys module-level context managers
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam3_image_model()

    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postprocessor = PostProcessImage(
        max_dets_per_img=-1, iou_type="segm",
        use_original_sizes_box=True, use_original_sizes_mask=True,
        convert_mask_to_rle=False, detection_threshold=0.3, to_cpu=True,
    )

    # Skip separate warmup — first frame will be warmup
    # (saves ~7 GB VRAM from not having warmup data in flight alongside real data)

    results = []
    for frame_img in frames:
        pil = Image.fromarray(frame_img[:, :, ::-1])  # BGR→RGB
        w, h = pil.size

        dp = Datapoint(find_queries=[], images=[])
        dp.images = [SAMImage(data=pil, objects=[], size=[h, w])]
        query_ids = {}
        for i, concept in enumerate(concepts):
            qid = i + 1
            query_ids[qid] = concept
            dp.find_queries.append(FindQueryLoaded(
                query_text=concept, image_id=0, object_ids_output=[],
                is_exhaustive=True, query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=qid, original_image_id=qid,
                    original_category_id=1, original_size=[w, h],
                    object_id=0, frame_index=0)))

        dp = transform(dp)
        batch = collate_fn_api([dp], dict_key="d")["d"]
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        t0 = time.perf_counter()
        start_evt.record()
        output = model(batch)
        end_evt.record()
        torch.cuda.synchronize()

        wall_ms = (time.perf_counter() - t0) * 1000
        gpu_ms = start_evt.elapsed_time(end_evt)
        peak_vram = torch.cuda.max_memory_allocated()

        proc_results = postprocessor.process_results(output, batch.find_metadatas)

        boxes, masks, scores, labels = [], [], [], []
        for qid, concept in query_ids.items():
            if qid not in proc_results:
                continue
            r = proc_results[qid]
            for det_idx in range(len(r.get("scores", []))):
                boxes.append(r["boxes"][det_idx].tolist())
                scores.append(float(r["scores"][det_idx]))
                labels.append(concept)
                if "masks" in r and det_idx < len(r["masks"]):
                    m = r["masks"][det_idx]
                    if hasattr(m, "numpy"):
                        m = m.numpy()
                    masks.append(np.squeeze(m).astype(bool))
                else:
                    masks.append(None)

        results.append(ModelResult(
            boxes=boxes, masks=masks, scores=scores, labels=labels,
            inference_ms=wall_ms, gpu_kernel_ms=gpu_ms, peak_vram_bytes=peak_vram,
        ))

    # Free SAM 3 from GPU before next model loads
    del model, postprocessor
    torch.cuda.empty_cache()
    import gc; gc.collect()
    console.print(f"  SAM 3 unloaded, GPU freed")

    return results


def run_fastsam(frames: list, model_name: str = "FastSAM-x.pt") -> list[ModelResult]:
    """Run FastSAM on frames."""
    from ultralytics import FastSAM

    console.print(f"  Loading {model_name}...")
    model = FastSAM(model_name)

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, device="cuda", retina_masks=True, conf=0.3, verbose=False)

    results = []
    for frame_img in frames:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        t0 = time.perf_counter()
        start_evt.record()
        preds = model(frame_img, device="cuda", retina_masks=True, conf=0.3, iou=0.5, verbose=False)
        end_evt.record()
        torch.cuda.synchronize()

        wall_ms = (time.perf_counter() - t0) * 1000
        gpu_ms = start_evt.elapsed_time(end_evt)
        peak_vram = torch.cuda.max_memory_allocated()

        boxes, masks, scores, labels = [], [], [], []
        if preds and len(preds) > 0:
            r = preds[0]
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    boxes.append(r.boxes.xyxy[i].cpu().tolist())
                    scores.append(float(r.boxes.conf[i].cpu()))
                    cls_id = int(r.boxes.cls[i].cpu())
                    labels.append(r.names.get(cls_id, f"class_{cls_id}"))
                    if r.masks is not None and i < len(r.masks.data):
                        m = r.masks.data[i].cpu().numpy().astype(bool)
                        masks.append(m)
                    else:
                        masks.append(None)

        results.append(ModelResult(
            boxes=boxes, masks=masks, scores=scores, labels=labels,
            inference_ms=wall_ms, gpu_kernel_ms=gpu_ms, peak_vram_bytes=peak_vram,
        ))

    return results


def run_yolo_only(frames: list) -> list[ModelResult]:
    """Run YOLO 11x detection only (no segmentation)."""
    from ultralytics import YOLO

    console.print("  Loading YOLO 11x...")
    model = YOLO("yolo11x.pt")
    model.to("cuda")

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)

    results = []
    for frame_img in frames:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        t0 = time.perf_counter()
        start_evt.record()
        preds = model.predict(frame_img, conf=0.35, verbose=False)
        end_evt.record()
        torch.cuda.synchronize()

        wall_ms = (time.perf_counter() - t0) * 1000
        gpu_ms = start_evt.elapsed_time(end_evt)
        peak_vram = torch.cuda.max_memory_allocated()

        boxes, scores, labels = [], [], []
        if preds and len(preds) > 0:
            r = preds[0]
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    boxes.append(r.boxes.xyxy[i].cpu().tolist())
                    scores.append(float(r.boxes.conf[i].cpu()))
                    cls_id = int(r.boxes.cls[i].cpu())
                    labels.append(r.names.get(cls_id, f"class_{cls_id}"))

        results.append(ModelResult(
            boxes=boxes, masks=[None] * len(boxes), scores=scores, labels=labels,
            inference_ms=wall_ms, gpu_kernel_ms=gpu_ms, peak_vram_bytes=peak_vram,
        ))

    return results


# ============================================================
# Comparison Engine
# ============================================================

def compare_to_baseline(
    baseline_results: list[ModelResult],
    test_results: list[ModelResult],
    iou_threshold: float = 0.5,
) -> dict:
    """Compare test model results against SAM 3 baseline."""
    total_baseline_dets = 0
    total_test_dets = 0
    total_matched = 0
    total_mask_iou = 0.0
    mask_iou_count = 0

    for base_frame, test_frame in zip(baseline_results, test_results):
        total_baseline_dets += len(base_frame.boxes)
        total_test_dets += len(test_frame.boxes)

        matches = match_detections(base_frame.boxes, test_frame.boxes, iou_threshold)
        total_matched += len(matches)

        for ref_idx, test_idx, box_iou in matches:
            ref_mask = base_frame.masks[ref_idx] if ref_idx < len(base_frame.masks) else None
            test_mask = test_frame.masks[test_idx] if test_idx < len(test_frame.masks) else None
            if ref_mask is not None and test_mask is not None:
                miou = compute_iou_mask(ref_mask, test_mask)
                total_mask_iou += miou
                mask_iou_count += 1

    recall = total_matched / total_baseline_dets if total_baseline_dets > 0 else 0
    false_positives = max(0, total_test_dets - total_matched)
    fp_rate = false_positives / total_test_dets if total_test_dets > 0 else 0
    avg_mask_iou = total_mask_iou / mask_iou_count if mask_iou_count > 0 else 0

    return {
        "recall": recall,
        "false_positive_rate": fp_rate,
        "avg_mask_iou": avg_mask_iou,
        "total_baseline_dets": total_baseline_dets,
        "total_test_dets": total_test_dets,
        "total_matched": total_matched,
    }


# ============================================================
# CLI
# ============================================================

@click.command()
@click.option("--video", "-v", required=True, help="Path to video file")
@click.option("--max-frames", "-m", default=5, type=int, help="Max frames to compare")
@click.option("--fps", "-f", default=2.0, type=float, help="Frame extraction FPS")
@click.option("--output", "-o", default="data/output/model_comparison.json", help="Output JSON")
def compare(video, max_frames, fps, output):
    """Compare SAM model variants on the same video frames."""
    from src.ingest.video import extract_frames

    console.print(f"\n[bold]Keyhole — Model Comparison[/]\n")
    console.print(f"  Video: {video}")
    console.print(f"  Frames: {max_frames} @ {fps} FPS\n")

    # Extract frames
    console.print("[bold]Extracting frames...[/]")
    video_path = Path(video)
    frame_images = []
    for frame in extract_frames(video_path, target_fps=fps, max_frames=max_frames):
        frame_images.append(frame.image.copy())
    console.print(f"  Extracted {len(frame_images)} frames\n")

    concepts = ["person", "vehicle", "car", "truck", "bus", "motorcycle",
                "bicycle", "dog", "cat"]

    # Run each model, freeing GPU memory between runs
    model_results = {}

    console.print("[bold]Running SAM 3 (840M, single-pass baseline)...[/]")
    # Use our pipeline's SAM3SinglePassDetector which handles inference_mode correctly
    from src.detect.sam3_detect import SAM3SinglePassDetector
    from src.ingest.video import ExtractedFrame
    sam3_det = SAM3SinglePassDetector(
        concepts=concepts, detection_threshold=0.3,
        device="cuda:0", retain_masks=True,
    )
    sam3_det.load_model()

    sam3_results = []
    for i, frame_img in enumerate(frame_images):
        fake_frame = ExtractedFrame(
            image=frame_img, frame_number=i,
            timestamp_sec=i / fps, source_video=str(video_path),
        )
        torch.cuda.reset_peak_memory_stats()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start_evt.record()
        enriched = sam3_det.detect_frame(fake_frame)
        end_evt.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000
        gpu_ms = start_evt.elapsed_time(end_evt)
        peak_vram = torch.cuda.max_memory_allocated()

        sam3_results.append(ModelResult(
            boxes=[list(d.bbox) for d in enriched.detections],
            masks=[d.concepts[0].mask if d.concepts and d.concepts[0].mask is not None else None
                   for d in enriched.detections],
            scores=[d.confidence for d in enriched.detections],
            labels=[d.class_name for d in enriched.detections],
            inference_ms=wall_ms, gpu_kernel_ms=gpu_ms, peak_vram_bytes=peak_vram,
        ))
    model_results["sam3"] = sam3_results

    # Free SAM 3 from GPU
    del sam3_det
    import gc
    torch.cuda.empty_cache(); gc.collect()
    console.print(f"  [dim]GPU freed: {torch.cuda.memory_allocated()/1e9:.1f} GB used[/]")

    console.print("[bold]Running FastSAM-x (68M, YOLO-based)...[/]")
    model_results["fastsam_x"] = run_fastsam(frame_images, "FastSAM-x.pt")
    torch.cuda.empty_cache(); gc.collect()

    console.print("[bold]Running FastSAM-s (11M, YOLO-based lightweight)...[/]")
    model_results["fastsam_s"] = run_fastsam(frame_images, "FastSAM-s.pt")
    torch.cuda.empty_cache(); gc.collect()

    console.print("[bold]Running YOLO 11x (57M, detection only)...[/]")
    model_results["yolo11x"] = run_yolo_only(frame_images)

    # Build profiles
    profiles = {}

    model_meta = {
        "sam3":      ("SAM 3 (single-pass)", 840.5, True, "4M+ concepts"),
        "fastsam_x": ("FastSAM-x", 68.0, False, "80 COCO classes"),
        "fastsam_s": ("FastSAM-s", 11.0, False, "80 COCO classes"),
        "yolo11x":   ("YOLO 11x (detect only)", 56.9, False, "80 COCO classes"),
    }

    for key, results_list in model_results.items():
        name, params, text_prompts, vocab = model_meta[key]

        latencies = [r.inference_ms for r in results_list]
        gpu_times = [r.gpu_kernel_ms for r in results_list]
        det_counts = [len(r.boxes) for r in results_list]
        peak_vram = max(r.peak_vram_bytes for r in results_list)
        avg_gpu = sum(gpu_times) / len(gpu_times)
        edge_ms, edge_fps = project_to_edge(avg_gpu)

        # Compare to SAM 3 baseline
        if key == "sam3":
            recall = 1.0
            fp_rate = 0.0
            mask_iou = 1.0
        else:
            comp = compare_to_baseline(model_results["sam3"], results_list)
            recall = comp["recall"]
            fp_rate = comp["false_positive_rate"]
            mask_iou = comp["avg_mask_iou"]

        profiles[key] = ModelProfile(
            name=name, param_count_m=params,
            supports_text_prompts=text_prompts, concept_vocabulary=vocab,
            avg_inference_ms=sum(latencies) / len(latencies),
            p95_inference_ms=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
            avg_gpu_kernel_ms=avg_gpu,
            peak_vram_gb=peak_vram / 1e9,
            avg_detections=sum(det_counts) / len(det_counts),
            recall_vs_sam3=recall,
            avg_mask_iou=mask_iou,
            false_positive_rate=fp_rate,
            edge_projected_ms=edge_ms,
            edge_fps=edge_fps,
        )

    # Display results
    console.print("\n")
    table = Table(title="Model Comparison Results", show_header=True, header_style="bold")
    table.add_column("Model", min_width=22)
    table.add_column("Params", justify="right", width=8)
    table.add_column("5090 ms", justify="right", width=9)
    table.add_column("5090 FPS", justify="right", width=9)
    table.add_column("GPU ms", justify="right", width=8)
    table.add_column("VRAM", justify="right", width=7)
    table.add_column("Dets/frm", justify="right", width=9)
    table.add_column("Recall", justify="right", width=8)
    table.add_column("Mask IoU", justify="right", width=9)
    table.add_column("Edge ms", justify="right", width=9)
    table.add_column("Edge FPS", justify="right", width=9)
    table.add_column("Concepts", width=12)

    for key in ["sam3", "fastsam_x", "fastsam_s", "yolo11x"]:
        p = profiles[key]
        table.add_row(
            p.name,
            f"{p.param_count_m:.0f}M",
            f"{p.avg_inference_ms:.0f}",
            f"{1000/p.avg_inference_ms:.1f}",
            f"{p.avg_gpu_kernel_ms:.0f}",
            f"{p.peak_vram_gb:.1f}GB",
            f"{p.avg_detections:.0f}",
            f"{p.recall_vs_sam3:.0%}",
            f"{p.avg_mask_iou:.0%}" if p.avg_mask_iou > 0 else "N/A",
            f"{p.edge_projected_ms:.0f}",
            f"{p.edge_fps:.1f}",
            p.concept_vocabulary[:12],
        )

    console.print(table)

    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "video": video,
        "num_frames": len(frame_images),
        "concepts": concepts,
        "profiles": {k: asdict(v) for k, v in profiles.items()},
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    console.print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Patch SAM 3's fused kernel to work without inference_mode
    # (inference_mode context doesn't survive click decorator boundaries)
    import sam3.perflib.fused as _fused
    import sam3.model.vitdet as _vitdet
    _original_addmm = _fused.addmm_act
    def _patched_addmm(activation, linear, mat1):
        with torch.no_grad():
            self = linear.bias.detach()
            mat2 = linear.weight.detach()
            self = self.to(torch.bfloat16)
            mat1 = mat1.to(torch.bfloat16)
            mat2 = mat2.to(torch.bfloat16)
            mat1_flat = mat1.view(-1, mat1.shape[-1])
            if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
                y = _fused.addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
                return y.view(mat1.shape[:-1] + (y.shape[-1],))
            if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
                y = _fused.addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
                return y.view(mat1.shape[:-1] + (y.shape[-1],))
            raise ValueError(f"Unexpected activation {activation}")
    _fused.addmm_act = _patched_addmm
    _vitdet.addmm_act = _patched_addmm

    compare()
