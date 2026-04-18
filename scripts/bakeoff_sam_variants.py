"""
Mask-model bake-off: MobileSAM vs EfficientSAM-Tiny vs EfficientSAM-Small vs YOLO-seg.
Reference: SAM 3 single-pass (concept-prompted, matched to YOLO prompt boxes by IoU).

Pipeline (staged, disk-cached for reruns):
  1. Sample frames at --fps from clip
  2. yolo11x.pt -> shared prompt boxes per frame
  3. SAM 3 single-pass -> reference masks, IoU-matched to prompt boxes
  4. Per contestant: load, infer all (frame, box), measure latency + VRAM + IoU vs ref
  5. Serialize aggregate results

Outputs under data/output/bakeoff/{clip_stem}/
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Make src/ importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bakeoff")

BAKEOFF_DIR = REPO_ROOT / "data" / "output" / "bakeoff"
EFFICIENTSAM_DIR = REPO_ROOT / "third_party" / "efficient_sam"

SAM3_CONCEPTS = [
    "person", "vehicle", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "bird", "animal",
    "backpack", "bag", "hat", "umbrella",
    "package", "box", "suitcase", "chair", "laptop",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e6


# -----------------------------------------------------------------------------
# Stage 1: frame sampling
# -----------------------------------------------------------------------------

@dataclass
class SampledFrame:
    idx: int
    timestamp_sec: float
    path: str  # relative to BAKEOFF_DIR/{clip_stem}


def sample_frames(clip_path: Path, out_dir: Path, fps: float) -> list[SampledFrame]:
    manifest_path = out_dir / "frames.json"
    frames_dir = out_dir / "frames"
    if manifest_path.exists() and frames_dir.exists():
        entries = json.loads(manifest_path.read_text())
        log.info("Reusing %d cached sampled frames", len(entries))
        return [SampledFrame(**e) for e in entries]

    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {clip_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(source_fps / fps)))
    log.info("Source %.2f fps, sampling every %d frames (target %.1f fps)", source_fps, interval, fps)

    sampled: list[SampledFrame] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            name = f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frames_dir / name), frame)
            sampled.append(SampledFrame(
                idx=frame_idx,
                timestamp_sec=frame_idx / source_fps,
                path=f"frames/{name}",
            ))
        frame_idx += 1
    cap.release()

    manifest_path.write_text(json.dumps([asdict(s) for s in sampled], indent=2))
    log.info("Sampled %d frames from %d total -> %s", len(sampled), frame_idx, frames_dir)
    return sampled


# -----------------------------------------------------------------------------
# Stage 2: YOLO prompt boxes (yolo11x)
# -----------------------------------------------------------------------------

@dataclass
class Prompt:
    box: list[float]  # xyxy
    class_id: int
    class_name: str
    confidence: float


def extract_prompts(out_dir: Path, frames: list[SampledFrame]) -> dict[int, list[Prompt]]:
    prompts_path = out_dir / "prompts.json"
    if prompts_path.exists():
        raw = json.loads(prompts_path.read_text())
        log.info("Reusing cached prompts for %d frames", len(raw))
        return {int(k): [Prompt(**p) for p in v] for k, v in raw.items()}

    from ultralytics import YOLO
    log.info("Loading yolo11x.pt for prompt extraction...")
    yolo = YOLO("yolo11x.pt")
    yolo.to("cuda:0")
    # warm up
    yolo.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    result: dict[int, list[Prompt]] = {}
    for f in frames:
        img = cv2.imread(str(out_dir / f.path))
        res = yolo.predict(img, conf=0.35, iou=0.45, verbose=False)
        prompts: list[Prompt] = []
        if res and len(res) > 0:
            r = res[0]
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    bb = boxes.xyxy[i].cpu().numpy()
                    cid = int(boxes.cls[i].cpu())
                    conf = float(boxes.conf[i].cpu())
                    prompts.append(Prompt(
                        box=[float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                        class_id=cid, class_name=r.names.get(cid, f"class_{cid}"),
                        confidence=conf,
                    ))
        result[f.idx] = prompts
        log.info("Frame %d: %d prompts", f.idx, len(prompts))

    prompts_path.write_text(json.dumps(
        {str(k): [asdict(p) for p in v] for k, v in result.items()}, indent=2,
    ))
    del yolo
    gc.collect()
    torch.cuda.empty_cache()
    return result


# -----------------------------------------------------------------------------
# Stage 3: SAM 3 reference masks
# -----------------------------------------------------------------------------

@dataclass
class RefMask:
    """Reference mask aligned to a prompt box."""
    prompt_idx: int          # index into prompts[frame]
    sam3_box: list[float]    # xyxy box SAM 3 returned
    match_iou: float         # bbox IoU between prompt and sam3_box
    mask_path: str           # relative .npy path


def generate_refs(
    out_dir: Path,
    frames: list[SampledFrame],
    prompts: dict[int, list[Prompt]],
    iou_match_threshold: float = 0.5,
) -> dict[int, list[RefMask]]:
    refs_meta_path = out_dir / "refs_meta.json"
    refs_dir = out_dir / "refs"
    if refs_meta_path.exists() and refs_dir.exists():
        raw = json.loads(refs_meta_path.read_text())
        log.info("Reusing cached SAM 3 reference masks for %d frames", len(raw))
        return {int(k): [RefMask(**r) for r in v] for k, v in raw.items()}

    refs_dir.mkdir(parents=True, exist_ok=True)
    from src.detect.sam3_detect import SAM3SinglePassDetector
    from src.ingest.video import ExtractedFrame

    log.info("Loading SAM 3 single-pass detector for references...")
    det = SAM3SinglePassDetector(
        concepts=SAM3_CONCEPTS, detection_threshold=0.3,
        retain_masks=True, profile=False,
    )
    det.load_model()

    result: dict[int, list[RefMask]] = {}
    from src.profiling.nvtx_helpers import nvtx_range
    for f in frames:
        img = cv2.imread(str(out_dir / f.path))
        ef = ExtractedFrame(
            frame_number=f.idx, timestamp_sec=f.timestamp_sec,
            image=img, source_video=str(out_dir),
        )
        with nvtx_range("sam3_bf16_reference"):
            enriched = det.detect_frame(ef)

        # Collect all SAM 3 detections that have masks
        sam3_dets: list[tuple[list[float], np.ndarray]] = []
        for d in enriched.detections:
            if d.concepts and d.concepts[0].mask is not None:
                m = d.concepts[0].mask
                if hasattr(m, "cpu"):
                    m = m.cpu().numpy()
                m = np.asarray(m).squeeze().astype(bool)
                sam3_dets.append((list(d.bbox), m))

        # Match each prompt to best SAM 3 box by IoU
        frame_prompts = prompts.get(f.idx, [])
        refs: list[RefMask] = []
        for pi, p in enumerate(frame_prompts):
            best_iou = 0.0
            best_j = -1
            for j, (sbox, _) in enumerate(sam3_dets):
                iou = iou_xyxy(np.array(p.box), np.array(sbox))
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= iou_match_threshold:
                mask = sam3_dets[best_j][1]
                mask_name = f"frame_{f.idx:06d}_p{pi:03d}.npy"
                np.save(refs_dir / mask_name, mask)
                refs.append(RefMask(
                    prompt_idx=pi,
                    sam3_box=sam3_dets[best_j][0],
                    match_iou=best_iou,
                    mask_path=f"refs/{mask_name}",
                ))
        result[f.idx] = refs
        log.info("Frame %d: %d/%d prompts matched to SAM3 refs",
                 f.idx, len(refs), len(frame_prompts))

    refs_meta_path.write_text(json.dumps(
        {str(k): [asdict(r) for r in v] for k, v in result.items()}, indent=2,
    ))

    # SAM 3 enters bf16 autocast + inference_mode at load() and never exits them,
    # which poisons subsequent contestants (MobileSAM crashes on bf16). Exit both.
    if hasattr(det, "_inference_ctx"):
        det._inference_ctx.__exit__(None, None, None)
    if hasattr(det, "_autocast_ctx"):
        det._autocast_ctx.__exit__(None, None, None)
    del det
    gc.collect()
    torch.cuda.empty_cache()
    return result


# -----------------------------------------------------------------------------
# Stage 4: contestants (common interface)
# -----------------------------------------------------------------------------

@dataclass
class BoxResult:
    prompt_idx: int
    mask_present: bool
    iou_vs_ref: Optional[float]  # None if no ref for this box


@dataclass
class FrameResult:
    frame_idx: int
    latency_ms: float            # wall-clock for all boxes in this frame
    per_box_latency_ms: float    # amortized per-box
    n_boxes: int
    box_results: list[BoxResult]


@dataclass
class ContestantReport:
    name: str
    params_m: float
    peak_vram_mb: float
    frames: list[FrameResult]
    mean_iou: float
    median_iou: float
    mean_per_box_ms: float
    p95_per_box_ms: float
    n_box_inferences: int
    n_iou_samples: int


class Contestant:
    name = "base"

    def load(self): ...
    def unload(self):
        gc.collect()
        torch.cuda.empty_cache()
    def params(self) -> int: return 0
    def infer_frame(self, image_bgr: np.ndarray, boxes: list[list[float]]) -> tuple[list[Optional[np.ndarray]], float]:
        """Returns (masks, total_latency_ms). masks[i] is bool array same H,W as image or None."""
        raise NotImplementedError


class MobileSAMContestant(Contestant):
    name = "mobilesam"

    def load(self):
        from mobile_sam import sam_model_registry, SamPredictor
        log.info("Loading MobileSAM (vit_t)...")
        sam = sam_model_registry["vit_t"](checkpoint=str(REPO_ROOT / "weights" / "mobile_sam.pt"))
        sam.to("cuda:0")
        sam.eval()
        self.predictor = SamPredictor(sam)
        self._params = sum(p.numel() for p in sam.parameters())
        # warm up — MobileSAM requires FP32, explicitly disable any ambient autocast
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        with torch.amp.autocast("cuda", enabled=False):
            self.predictor.set_image(dummy)
            self.predictor.predict(box=np.array([10.0, 10.0, 100.0, 100.0]), multimask_output=False)

    def params(self) -> int: return self._params

    def infer_frame(self, image_bgr, boxes):
        rgb = image_bgr[:, :, ::-1]
        masks: list[Optional[np.ndarray]] = []
        sync_cuda()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=False):
            self.predictor.set_image(rgb)
            for b in boxes:
                m, _, _ = self.predictor.predict(
                    point_coords=None, point_labels=None,
                    box=np.array(b, dtype=np.float32)[None, :],
                    multimask_output=False,
                )
                masks.append(m[0].astype(bool) if m is not None and len(m) > 0 else None)
        sync_cuda()
        return masks, (time.perf_counter() - t0) * 1000


class EfficientSAMContestant(Contestant):
    """Tiny or Small variant."""

    def __init__(self, variant: str):
        assert variant in ("tiny", "small")
        self.variant = variant
        self.name = f"efficientsam_{variant}"

    def load(self):
        # EfficientSAM's build fn uses relative paths — cd into its dir first
        import os
        cwd = os.getcwd()
        os.chdir(str(EFFICIENTSAM_DIR))
        try:
            sys.path.insert(0, str(EFFICIENTSAM_DIR))
            from efficient_sam.build_efficient_sam import (
                build_efficient_sam_vitt, build_efficient_sam_vits,
            )
            if self.variant == "tiny":
                self.model = build_efficient_sam_vitt()
            else:
                self.model = build_efficient_sam_vits()
        finally:
            os.chdir(cwd)
            if str(EFFICIENTSAM_DIR) in sys.path:
                sys.path.remove(str(EFFICIENTSAM_DIR))
        self.model = self.model.to("cuda:0").eval()
        self._params = sum(p.numel() for p in self.model.parameters())
        # warm up
        dummy = torch.zeros(1, 3, 512, 512, device="cuda:0")
        pts = torch.tensor([[[[10.0, 10.0], [100.0, 100.0]]]], device="cuda:0")
        lbls = torch.tensor([[[2, 3]]], device="cuda:0")
        with torch.no_grad():
            self.model(dummy, pts, lbls)

    def params(self) -> int: return self._params

    def infer_frame(self, image_bgr, boxes):
        H, W = image_bgr.shape[:2]
        rgb = image_bgr[:, :, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        t = t.unsqueeze(0).to("cuda:0")  # [1, 3, H, W]

        n = len(boxes)
        if n == 0:
            return [], 0.0

        # Build box prompts: 2 points per box (TL, BR) with labels (2, 3)
        pts = torch.zeros(1, n, 2, 2, device="cuda:0")
        lbls = torch.zeros(1, n, 2, dtype=torch.long, device="cuda:0")
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            pts[0, i, 0] = torch.tensor([x1, y1])
            pts[0, i, 1] = torch.tensor([x2, y2])
            lbls[0, i, 0] = 2
            lbls[0, i, 1] = 3

        sync_cuda()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, iou = self.model(t, pts, lbls)
        sync_cuda()
        latency_ms = (time.perf_counter() - t0) * 1000

        # logits: [B, Q, num_masks, H, W]. Take mask 0 (highest iou-sorted index? docs say sorted already)
        # Sort by iou like the example does, take top-1
        sorted_ids = torch.argsort(iou, dim=-1, descending=True)
        logits_sorted = torch.take_along_dim(logits, sorted_ids[..., None, None], dim=2)
        # Binarize mask 0 for each query
        masks: list[Optional[np.ndarray]] = []
        for i in range(n):
            m = (logits_sorted[0, i, 0] >= 0).cpu().numpy()
            # Spatial size should already equal HxW of input. If not, resize.
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            masks.append(m.astype(bool))
        return masks, latency_ms


class YoloSegContestant(Contestant):
    name = "yolo_seg"

    def __init__(self, weights: str = "yolo11s-seg.pt", match_iou: float = 0.5):
        self.weights = weights
        self.match_iou = match_iou

    def load(self):
        from ultralytics import YOLO
        log.info("Loading %s...", self.weights)
        self.model = YOLO(self.weights)
        self.model.to("cuda:0")
        self._params = sum(p.numel() for p in self.model.model.parameters())
        # warm up
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    def params(self) -> int: return self._params

    def infer_frame(self, image_bgr, boxes):
        H, W = image_bgr.shape[:2]
        sync_cuda()
        t0 = time.perf_counter()
        res = self.model.predict(image_bgr, conf=0.25, iou=0.45, verbose=False)
        sync_cuda()
        latency_ms = (time.perf_counter() - t0) * 1000

        # Gather detections with masks
        seg_boxes: list[np.ndarray] = []
        seg_masks: list[np.ndarray] = []
        if res and len(res) > 0:
            r = res[0]
            if r.boxes is not None and r.masks is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                # masks.data: [N, h, w] at model res — use masks.xy + draw, or interpolate
                mdata = r.masks.data.cpu().numpy()  # [N, h, w] float/bool
                for i in range(len(xyxy)):
                    seg_boxes.append(xyxy[i])
                    m = mdata[i]
                    if m.shape != (H, W):
                        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    seg_masks.append(m.astype(bool))

        # For each prompt box, pick highest-IoU seg detection
        masks: list[Optional[np.ndarray]] = []
        for b in boxes:
            best_iou = 0.0
            best_j = -1
            pb = np.array(b)
            for j, sb in enumerate(seg_boxes):
                iou = iou_xyxy(pb, sb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= self.match_iou:
                masks.append(seg_masks[best_j])
            else:
                masks.append(None)
        return masks, latency_ms


# -----------------------------------------------------------------------------
# Stage 4 driver: run one contestant across all frames
# -----------------------------------------------------------------------------

def run_contestant(
    contestant: Contestant,
    out_dir: Path,
    frames: list[SampledFrame],
    prompts: dict[int, list[Prompt]],
    refs: dict[int, list[RefMask]],
) -> ContestantReport:
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{contestant.name}.json"
    if out_path.exists():
        log.info("Reusing cached results for %s", contestant.name)
        data = json.loads(out_path.read_text())
        return ContestantReport(
            name=data["name"],
            params_m=data["params_m"],
            peak_vram_mb=data["peak_vram_mb"],
            frames=[FrameResult(
                frame_idx=fr["frame_idx"], latency_ms=fr["latency_ms"],
                per_box_latency_ms=fr["per_box_latency_ms"], n_boxes=fr["n_boxes"],
                box_results=[BoxResult(**br) for br in fr["box_results"]],
            ) for fr in data["frames"]],
            mean_iou=data["mean_iou"],
            median_iou=data["median_iou"],
            mean_per_box_ms=data["mean_per_box_ms"],
            p95_per_box_ms=data["p95_per_box_ms"],
            n_box_inferences=data["n_box_inferences"],
            n_iou_samples=data["n_iou_samples"],
        )

    log.info("=== Running contestant: %s ===", contestant.name)
    contestant.load()
    gpu_reset_peak()

    frame_results: list[FrameResult] = []
    all_ious: list[float] = []
    all_per_box_ms: list[float] = []
    total_boxes = 0

    for f in frames:
        img = cv2.imread(str(out_dir / f.path))
        frame_prompts = prompts.get(f.idx, [])
        boxes = [p.box for p in frame_prompts]
        if not boxes:
            frame_results.append(FrameResult(
                frame_idx=f.idx, latency_ms=0.0, per_box_latency_ms=0.0,
                n_boxes=0, box_results=[],
            ))
            continue

        from src.profiling.nvtx_helpers import nvtx_range
        with nvtx_range(contestant.name):
            masks, latency_ms = contestant.infer_frame(img, boxes)
        per_box_ms = latency_ms / max(1, len(boxes))

        # Build ref lookup by prompt_idx
        ref_by_pidx = {r.prompt_idx: r for r in refs.get(f.idx, [])}

        box_results: list[BoxResult] = []
        for pi, m in enumerate(masks):
            iou_val: Optional[float] = None
            if m is not None and pi in ref_by_pidx:
                ref_mask = np.load(out_dir / ref_by_pidx[pi].mask_path)
                iou_val = mask_iou(m, ref_mask)
                all_ious.append(iou_val)
            box_results.append(BoxResult(
                prompt_idx=pi, mask_present=(m is not None), iou_vs_ref=iou_val,
            ))

        frame_results.append(FrameResult(
            frame_idx=f.idx, latency_ms=latency_ms,
            per_box_latency_ms=per_box_ms, n_boxes=len(boxes),
            box_results=box_results,
        ))
        total_boxes += len(boxes)
        all_per_box_ms.extend([per_box_ms] * len(boxes))
        log.info("%s frame %d: %d boxes, %.1fms (%.2fms/box), IoU@matches=%s",
                 contestant.name, f.idx, len(boxes), latency_ms, per_box_ms,
                 f"{np.mean([b.iou_vs_ref for b in box_results if b.iou_vs_ref is not None]):.3f}"
                 if any(b.iou_vs_ref is not None for b in box_results) else "n/a")

    peak_vram = gpu_peak_mb()
    contestant.unload()

    report = ContestantReport(
        name=contestant.name,
        params_m=contestant.params() / 1e6,
        peak_vram_mb=peak_vram,
        frames=frame_results,
        mean_iou=float(np.mean(all_ious)) if all_ious else 0.0,
        median_iou=float(np.median(all_ious)) if all_ious else 0.0,
        mean_per_box_ms=float(np.mean(all_per_box_ms)) if all_per_box_ms else 0.0,
        p95_per_box_ms=float(np.percentile(all_per_box_ms, 95)) if all_per_box_ms else 0.0,
        n_box_inferences=total_boxes,
        n_iou_samples=len(all_ious),
    )

    out_path.write_text(json.dumps(asdict(report), indent=2))
    log.info("%s: params=%.2fM, VRAM=%.0f MB, mean IoU=%.3f (n=%d), mean %.2f ms/box",
             report.name, report.params_m, report.peak_vram_mb, report.mean_iou,
             report.n_iou_samples, report.mean_per_box_ms)
    return report


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

CONTESTANT_REGISTRY = {
    "mobilesam": lambda: MobileSAMContestant(),
    "efficientsam_tiny": lambda: EfficientSAMContestant("tiny"),
    "efficientsam_small": lambda: EfficientSAMContestant("small"),
    "yolo_seg": lambda: YoloSegContestant("yolo11s-seg.pt"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="Path to clip (relative or absolute)")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--contestants", nargs="+",
                    default=["mobilesam", "efficientsam_tiny", "efficientsam_small", "yolo_seg"])
    ap.add_argument("--skip-refs", action="store_true",
                    help="Skip SAM 3 reference generation (IoU will be unavailable)")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.is_absolute():
        clip_path = REPO_ROOT / clip_path
    if not clip_path.exists():
        raise SystemExit(f"Clip not found: {clip_path}")

    out_dir = BAKEOFF_DIR / clip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    frames = sample_frames(clip_path, out_dir, fps=args.fps)
    prompts = extract_prompts(out_dir, frames)
    if args.skip_refs:
        refs = {f.idx: [] for f in frames}
        log.warning("--skip-refs set; IoU metrics will be unavailable")
    else:
        refs = generate_refs(out_dir, frames, prompts)

    reports: dict[str, ContestantReport] = {}
    for name in args.contestants:
        if name not in CONTESTANT_REGISTRY:
            log.warning("Unknown contestant: %s", name)
            continue
        rep = run_contestant(CONTESTANT_REGISTRY[name](), out_dir, frames, prompts, refs)
        reports[name] = rep

    summary = {
        "clip": str(clip_path),
        "fps": args.fps,
        "n_frames": len(frames),
        "n_prompts": sum(len(v) for v in prompts.values()),
        "n_refs": sum(len(v) for v in refs.values()),
        "contestants": {
            name: {
                "params_m": r.params_m,
                "peak_vram_mb": r.peak_vram_mb,
                "mean_iou": r.mean_iou,
                "median_iou": r.median_iou,
                "mean_per_box_ms": r.mean_per_box_ms,
                "p95_per_box_ms": r.p95_per_box_ms,
                "n_box_inferences": r.n_box_inferences,
                "n_iou_samples": r.n_iou_samples,
            } for name, r in reports.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote summary -> %s", out_dir / "summary.json")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
