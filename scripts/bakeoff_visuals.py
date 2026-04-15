"""
Generate side-by-side mask comparison visuals for the bake-off.

For a representative frame from each resolution, renders a grid:
  | original + YOLO boxes | SAM 3 ref | MobileSAM | ES-Tiny | ES-Small | YOLO-seg |

Also rebuilds contestant masks on-the-fly (not cached to disk in the harness
to keep that output lean). Fast — one frame per resolution.

Output: data/output/bakeoff/visuals/{res}_sidebyside.png
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.bakeoff_sam_variants import (
    MobileSAMContestant, EfficientSAMContestant, YoloSegContestant,
    BAKEOFF_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("visuals")

CLIPS = {
    "720p":  ("720p_EW_clip", 0),                  # (stem, frame index to render)
    "1080p": ("embedded_world_clip_1080p", 0),
    "4K":    ("embedded_world_clip", 0),
}

COLORS = [
    (255, 99,  71),   # coral
    (100, 200, 255),  # sky
    (255, 215, 0),    # gold
    (144, 238, 144),  # light green
    (255, 105, 180),  # pink
    (173, 216, 230),  # light blue
    (255, 165, 0),    # orange
    (221, 160, 221),  # plum
]


def overlay_masks(image_bgr: np.ndarray, masks: list, alpha: float = 0.55) -> np.ndarray:
    out = image_bgr.copy()
    for i, m in enumerate(masks):
        if m is None:
            continue
        color = np.array(COLORS[i % len(COLORS)], dtype=np.uint8)
        mask_bool = m.astype(bool)
        out[mask_bool] = (out[mask_bool] * (1 - alpha) + color * alpha).astype(np.uint8)
    # Draw mask outlines for contrast
    for i, m in enumerate(masks):
        if m is None:
            continue
        mu = m.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(int(c) for c in COLORS[i % len(COLORS)])
        cv2.drawContours(out, contours, -1, color, 2)
    return out


def draw_boxes(image_bgr: np.ndarray, boxes: list, labels: list[str] = None) -> np.ndarray:
    out = image_bgr.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b]
        color = tuple(int(c) for c in COLORS[i % len(COLORS)])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if labels:
            cv2.putText(out, labels[i], (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def add_panel_label(img: np.ndarray, text: str) -> np.ndarray:
    """Strip a label bar on top of the panel."""
    h, w = img.shape[:2]
    bar_h = max(28, h // 22)
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[:bar_h] = (30, 30, 30)
    out[bar_h:] = img
    cv2.putText(out, text, (10, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def hconcat_panels(panels: list[np.ndarray]) -> np.ndarray:
    # All must have same height; pad as needed
    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < h:
            pad = np.zeros((h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)
    return np.hstack(padded)


def build_visual_for_clip(clip_stem: str, frame_idx_in_list: int, contestants):
    out_dir = BAKEOFF_DIR / clip_stem
    frames = json.loads((out_dir / "frames.json").read_text())
    prompts = json.loads((out_dir / "prompts.json").read_text())
    refs_meta = json.loads((out_dir / "refs_meta.json").read_text())

    f = frames[frame_idx_in_list]
    img = cv2.imread(str(out_dir / f["path"]))
    H, W = img.shape[:2]
    frame_prompts = prompts[str(f["idx"])]
    boxes = [p["box"] for p in frame_prompts]
    labels = [f"{p['class_name']}" for p in frame_prompts]

    # SAM 3 ref masks for this frame (only for matched boxes)
    ref_entries = refs_meta[str(f["idx"])]
    ref_masks = [None] * len(frame_prompts)
    for r in ref_entries:
        m = np.load(out_dir / r["mask_path"])
        ref_masks[r["prompt_idx"]] = m

    # Panel 1: original + YOLO boxes
    boxes_panel = draw_boxes(img, boxes, labels)

    # Panel 2: SAM 3 ref masks
    sam3_panel = overlay_masks(img, ref_masks)

    # Panels 3-N: each contestant
    panels = [
        add_panel_label(boxes_panel, f"YOLO prompts ({len(boxes)} boxes)"),
        add_panel_label(sam3_panel,
                        f"SAM 3 reference ({sum(1 for m in ref_masks if m is not None)} matched)"),
    ]

    for name, c in contestants.items():
        log.info("  %s...", name)
        c.load()
        masks, _ = c.infer_frame(img, boxes)
        c.unload()
        panel = overlay_masks(img, masks)
        panels.append(add_panel_label(panel, name))

    # Shrink each panel horizontally for display (keep aspect)
    target_w = 640
    scaled = []
    for p in panels:
        h, w = p.shape[:2]
        new_h = int(h * target_w / w)
        scaled.append(cv2.resize(p, (target_w, new_h), interpolation=cv2.INTER_AREA))

    # Two rows of 3 panels for readability
    row1 = hconcat_panels(scaled[:3])
    row2 = hconcat_panels(scaled[3:])
    return np.vstack([row1, row2])


def main():
    vis_dir = BAKEOFF_DIR / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Build one instance of each — we'll load/unload per clip to keep VRAM low
    def fresh():
        return {
            "mobilesam":         MobileSAMContestant(),
            "efficientsam_tiny": EfficientSAMContestant("tiny"),
            "efficientsam_small": EfficientSAMContestant("small"),
            "yolo_seg":          YoloSegContestant("yolo11s-seg.pt"),
        }

    for res, (stem, idx) in CLIPS.items():
        log.info("Building %s visual...", res)
        combined = build_visual_for_clip(stem, idx, fresh())
        out_path = vis_dir / f"{res}_sidebyside.png"
        cv2.imwrite(str(out_path), combined)
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
