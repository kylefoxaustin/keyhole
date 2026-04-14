"""
Annotated Video Renderer — Visualize Pipeline Output

Takes enriched frame data (YOLO bounding boxes + SAM 3 segmentation masks
and concept labels) and composites them onto the original video frames,
producing an annotated output video.

Usage:
    Integrated into the pipeline via --render flag:
        python -m src.main process --video input.mp4 --render

    The renderer produces:
        data/output/{video_stem}_annotated.mp4
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.enrich.sam3 import EnrichedFrame, EnrichedDetection

logger = logging.getLogger(__name__)

# Color palette for different object classes (BGR for OpenCV)
CLASS_COLORS = {
    "person":     (0xFF, 0x88, 0x00),   # Orange
    "car":        (0xFF, 0xD4, 0x00),   # Cyan
    "truck":      (0x00, 0xD4, 0xFF),   # Yellow
    "bus":        (0x88, 0xFF, 0x00),   # Green
    "motorcycle": (0xFC, 0x86, 0xBB),   # Purple
    "bicycle":    (0x00, 0xFF, 0x88),   # Green
    "dog":        (0x44, 0x44, 0xFF),   # Red
    "cat":        (0xFF, 0x44, 0x44),   # Blue
    "bird":       (0x88, 0xFF, 0xFF),   # Yellow
}
DEFAULT_COLOR = (0xFF, 0xD4, 0x00)

# Distinct mask overlay colors (BGR, cycled per detection)
MASK_PALETTE = [
    (255, 136, 0),    # Orange
    (0, 212, 255),    # Cyan
    (0, 255, 136),    # Green
    (187, 134, 252),  # Purple
    (255, 68, 68),    # Red
    (136, 255, 255),  # Yellow
    (255, 0, 110),    # Pink
    (68, 255, 68),    # Lime
    (255, 200, 68),   # Gold
    (68, 136, 255),   # Blue
]


@dataclass
class RenderConfig:
    """Configuration for video rendering."""
    box_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1
    mask_alpha: float = 0.35       # Opacity for mask overlays
    label_bg_alpha: float = 0.7    # Opacity for label background
    max_concepts_shown: int = 3    # Max concept labels per detection
    show_confidence: bool = True
    show_concepts: bool = True
    show_masks: bool = True
    output_fps: float = 2.0        # Matches extraction FPS
    codec: str = "libx264"
    crf: int = 20

    # GIF generation (for PowerPoint-friendly outputs)
    generate_gif: bool = True      # Auto-generate optimized GIF alongside MP4
    gif_width: int = 800           # Max width for GIF (keeps aspect ratio)
    gif_fps: float = 0.0           # 0 = use output_fps, else override


class VideoRenderer:
    """
    Renders annotated video frames with detection overlays.

    Composites YOLO bounding boxes, SAM 3 segmentation masks,
    and concept labels onto original video frames.
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self._frames: list[np.ndarray] = []
        self._frame_size: Optional[tuple[int, int]] = None  # (width, height)

    def _get_color(self, class_name: str, det_index: int) -> tuple:
        """Get color for a detection, falling back to palette cycling."""
        return CLASS_COLORS.get(class_name, MASK_PALETTE[det_index % len(MASK_PALETTE)])

    def _draw_mask_overlay(
        self, frame: np.ndarray, mask: np.ndarray,
        x1: int, y1: int, color: tuple,
    ) -> np.ndarray:
        """Overlay a segmentation mask onto the frame with transparency."""
        fh, fw = frame.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        overlay = frame.copy()

        # Determine if mask is full-frame or crop-relative
        # Full-frame: mask covers the entire image
        # Crop-relative: mask is sized to the detection crop
        is_full_frame = (mask_h >= fh * 0.5 and mask_w >= fw * 0.5)

        if is_full_frame:
            # Full-frame mask (from single-pass mode) — resize if needed
            if mask_h != fh or mask_w != fw:
                import cv2 as _cv2
                mask = _cv2.resize(mask.astype(np.uint8), (fw, fh),
                                   interpolation=_cv2.INTER_NEAREST).astype(bool)
            overlay[mask] = color
        else:
            # Crop-relative mask (from sequential YOLO+SAM3 mode)
            x2 = min(x1 + mask_w, fw)
            y2 = min(y1 + mask_h, fh)
            mask_crop = mask[:y2 - y1, :x2 - x1]
            if mask_crop.size > 0:
                region = overlay[y1:y2, x1:x2]
                region[mask_crop] = color

        frame = cv2.addWeighted(overlay, self.config.mask_alpha,
                                frame, 1 - self.config.mask_alpha, 0)
        return frame

    def _draw_box_and_label(
        self, frame: np.ndarray, det: EnrichedDetection,
        color: tuple, det_index: int,
    ) -> np.ndarray:
        """Draw bounding box and label text on frame."""
        x1, y1, x2, y2 = map(int, det.bbox)
        cfg = self.config

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.box_thickness)

        # Build label text
        label_parts = [det.class_name]
        if cfg.show_confidence:
            label_parts.append(f"{det.confidence:.0%}")
        label = " ".join(label_parts)

        # Concept tags below the main label
        concept_lines = []
        if cfg.show_concepts and det.concepts:
            tags = [c.concept for c in det.concepts[:cfg.max_concepts_shown]]
            concept_lines.append(", ".join(tags))

        # Measure text sizes
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, cfg.font_scale, cfg.font_thickness)

        # Label background
        label_y = max(y1 - th - 10, 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, label_y), (x1 + tw + 8, label_y + th + 8),
                       color, -1)
        frame = cv2.addWeighted(overlay, cfg.label_bg_alpha,
                                frame, 1 - cfg.label_bg_alpha, 0)

        # Label text
        cv2.putText(frame, label, (x1 + 4, label_y + th + 4),
                     font, cfg.font_scale, (255, 255, 255), cfg.font_thickness,
                     cv2.LINE_AA)

        # Concept tags (smaller, below the box)
        if concept_lines:
            concept_font_scale = cfg.font_scale * 0.8
            for i, line in enumerate(concept_lines):
                cy = y2 + 15 + i * 18
                if cy < frame.shape[0] - 5:
                    (cw, ch), _ = cv2.getTextSize(line, font, concept_font_scale, 1)
                    # Small background
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, cy - ch - 2), (x1 + cw + 6, cy + 4),
                                   (30, 30, 30), -1)
                    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                    cv2.putText(frame, line, (x1 + 3, cy),
                                 font, concept_font_scale, color, 1, cv2.LINE_AA)

        return frame

    def render_frame(
        self, image: np.ndarray, enriched: EnrichedFrame,
    ) -> np.ndarray:
        """
        Render all annotations onto a single frame.

        Returns the annotated frame (BGR numpy array).
        """
        frame = image.copy()

        for i, det in enumerate(enriched.detections):
            color = self._get_color(det.class_name, i)
            x1, y1 = int(det.bbox[0]), int(det.bbox[1])

            # SAM 3 mask overlays
            if self.config.show_masks and det.concepts:
                for concept in det.concepts:
                    if concept.mask is not None:
                        frame = self._draw_mask_overlay(
                            frame, concept.mask, x1, y1, color,
                        )

            # Bounding box + labels
            frame = self._draw_box_and_label(frame, det, color, i)

        # Frame info watermark
        info = (
            f"Frame {enriched.frame_number} | "
            f"t={enriched.timestamp_sec:.1f}s | "
            f"{len(enriched.detections)} detections"
        )
        fh, fw = frame.shape[:2]
        cv2.putText(frame, info, (10, fh - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                     cv2.LINE_AA)

        return frame

    def add_frame(self, annotated_frame: np.ndarray):
        """Add a rendered frame to the output buffer."""
        if self._frame_size is None:
            h, w = annotated_frame.shape[:2]
            self._frame_size = (w, h)
        self._frames.append(annotated_frame)

    def write_video(self, output_path: str) -> str:
        """
        Write all buffered frames to an MP4 video file.

        Uses OpenCV VideoWriter with H.264 encoding.
        Falls back to FFmpeg subprocess if OpenCV codec isn't available.
        """
        if not self._frames:
            logger.warning("No frames to write")
            return ""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        w, h = self._frame_size
        fps = self.config.output_fps

        # Try OpenCV VideoWriter first
        temp_path = str(output_path.with_suffix(".tmp.avi"))
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            logger.warning("OpenCV VideoWriter failed, using raw frame write")
            writer.release()
            # Fallback: write frames directly with FFmpeg
            return self._write_with_ffmpeg(str(output_path), w, h, fps)

        for frame in self._frames:
            writer.write(frame)
        writer.release()

        # Re-encode to H.264 MP4 with FFmpeg
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", temp_path,
                "-c:v", self.config.codec,
                "-crf", str(self.config.crf),
                "-pix_fmt", "yuv420p",
                "-r", str(fps),
                str(output_path),
            ], check=True, capture_output=True)
            Path(temp_path).unlink(missing_ok=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If FFmpeg fails, keep the AVI
            logger.warning("FFmpeg re-encode failed, keeping AVI output")
            Path(temp_path).rename(output_path.with_suffix(".avi"))
            return str(output_path.with_suffix(".avi"))

        logger.info("Rendered video: %s (%d frames, %.1f FPS)",
                     output_path, len(self._frames), fps)

        # Optionally generate a GIF alongside the MP4 for PowerPoint compatibility
        if self.config.generate_gif:
            gif_path = output_path.with_suffix(".gif")
            self.mp4_to_gif(str(output_path), str(gif_path))

        return str(output_path)

    def mp4_to_gif(self, mp4_path: str, gif_path: str) -> Optional[str]:
        """
        Convert an MP4 to an optimized animated GIF using a two-pass
        palette approach for good quality at reasonable file size.

        PowerPoint embeds GIFs reliably (unlike video, which often fails
        to play on different machines due to codec issues).
        """
        gif_fps = self.config.gif_fps if self.config.gif_fps > 0 else self.config.output_fps
        gif_width = self.config.gif_width

        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", mp4_path,
                "-vf", (
                    f"fps={gif_fps},"
                    f"scale={gif_width}:-1:flags=lanczos,"
                    "split[s0][s1];"
                    "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
                    "[s1][p]paletteuse=dither=bayer:bayer_scale=4"
                ),
                "-loop", "0",
                gif_path,
            ], check=True, capture_output=True)

            gif_size_mb = Path(gif_path).stat().st_size / 1e6
            logger.info("Rendered GIF: %s (%.1f MB, %.0fpx wide, %.1f FPS)",
                        gif_path, gif_size_mb, gif_width, gif_fps)
            return gif_path

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("GIF generation failed: %s", e)
            return None

    def _write_with_ffmpeg(self, output_path: str, w: int, h: int, fps: float) -> str:
        """Write frames directly to FFmpeg via pipe."""
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", self.config.codec,
            "-crf", str(self.config.crf),
            "-pix_fmt", "yuv420p",
            output_path,
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in self._frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            logger.error("FFmpeg pipe write failed: %s", proc.stderr.read().decode())
            return ""

        logger.info("Rendered video (ffmpeg pipe): %s (%d frames)", output_path, len(self._frames))
        return output_path

    def reset(self):
        """Clear the frame buffer."""
        self._frames.clear()
        self._frame_size = None
