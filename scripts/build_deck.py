"""
Keyhole — Results Deck Generator

Reads profiling data from data/output/ and generates a PowerPoint
presentation with architecture diagrams, profiling results, NPU
projections, and run-over-run comparison.

Usage:
    python scripts/build_deck.py
    python scripts/build_deck.py --output my_deck.pptx
    python scripts/build_deck.py --runs-dir data/output/runs
"""

import json
import math
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ============================================================
# Color Palette
# ============================================================

class C:
    """Consistent color palette across all slides."""
    BG_DARK = RGBColor(0x1A, 0x1A, 0x2E)
    BG_SLIDE = RGBColor(0x16, 0x21, 0x3E)
    ACCENT_BLUE = RGBColor(0x00, 0xD4, 0xFF)
    ACCENT_GREEN = RGBColor(0x00, 0xFF, 0x88)
    ACCENT_ORANGE = RGBColor(0xFF, 0x8C, 0x00)
    ACCENT_RED = RGBColor(0xFF, 0x44, 0x44)
    ACCENT_PURPLE = RGBColor(0xBB, 0x86, 0xFC)
    TEXT_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    TEXT_DIM = RGBColor(0xAA, 0xAA, 0xCC)
    TEXT_BRIGHT = RGBColor(0xE0, 0xE0, 0xFF)
    TABLE_HEADER = RGBColor(0x0D, 0x47, 0xA1)
    TABLE_ROW_1 = RGBColor(0x1A, 0x23, 0x40)
    TABLE_ROW_2 = RGBColor(0x22, 0x2B, 0x4A)
    FEASIBLE = RGBColor(0x00, 0xE6, 0x76)
    NOT_FEASIBLE = RGBColor(0xFF, 0x44, 0x44)

# Matplotlib equivalents
MPL_COLORS = {
    "bg": "#1A1A2E",
    "bg_slide": "#16213E",
    "blue": "#00D4FF",
    "green": "#00FF88",
    "orange": "#FF8C00",
    "red": "#FF4444",
    "purple": "#BB86FC",
    "text": "#E0E0FF",
    "dim": "#AAAACC",
    "grid": "#333355",
}


# ============================================================
# Helpers
# ============================================================

def set_slide_bg(slide, color=C.BG_SLIDE):
    """Set slide background to a solid color."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text,
                 font_size=18, color=C.TEXT_WHITE, bold=False,
                 alignment=PP_ALIGN.LEFT, font_name="Segoe UI"):
    """Add a styled text box."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_title_bar(slide, title, subtitle=None):
    """Add a consistent title bar across the top of a slide."""
    # Title
    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(9), Inches(0.6),
                 title, font_size=28, color=C.ACCENT_BLUE, bold=True)
    # Accent line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.5), Inches(0.85), Inches(2), Pt(3),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = C.ACCENT_BLUE
    line.line.fill.background()

    if subtitle:
        add_text_box(slide, Inches(0.5), Inches(0.95), Inches(9), Inches(0.4),
                     subtitle, font_size=14, color=C.TEXT_DIM)


def add_styled_table(slide, left, top, width, height,
                     headers, rows, col_widths=None):
    """Add a table with dark themed styling."""
    num_rows = len(rows) + 1
    num_cols = len(headers)

    table_shape = slide.shapes.add_table(
        num_rows, num_cols, left, top, width, height
    )
    table = table_shape.table

    # Set column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w

    # Style header row
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = C.TABLE_HEADER
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(11)
            paragraph.font.color.rgb = C.TEXT_WHITE
            paragraph.font.bold = True
            paragraph.font.name = "Segoe UI"
            paragraph.alignment = PP_ALIGN.CENTER

    # Style data rows
    for r_idx, row in enumerate(rows):
        bg = C.TABLE_ROW_1 if r_idx % 2 == 0 else C.TABLE_ROW_2
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(10)
                paragraph.font.color.rgb = C.TEXT_BRIGHT
                paragraph.font.name = "Segoe UI"
                paragraph.alignment = PP_ALIGN.CENTER

    return table_shape


def fig_to_image_stream(fig) -> io.BytesIO:
    """Convert a matplotlib figure to a BytesIO PNG stream."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================
# Data Loaders
# ============================================================

def load_runs(runs_dir: Path) -> list[dict]:
    """Load all run JSON files, sorted by timestamp."""
    runs = []
    if runs_dir.exists():
        for f in sorted(runs_dir.glob("run_*.json")):
            with open(f) as fh:
                runs.append(json.load(fh))
    return runs


def load_sam3_reference(output_dir: Path) -> Optional[dict]:
    """Load SAM 3 reference architecture data."""
    ref_path = output_dir / "sam3_reference_architecture.json"
    if ref_path.exists():
        with open(ref_path) as f:
            return json.load(f)
    return None


def load_npu_targets() -> list[dict]:
    """Load NPU hardware target definitions."""
    from src.emulate.npu_emulator import PRESET_TARGETS
    return {name: spec.to_dict() for name, spec in PRESET_TARGETS.items()}


# ============================================================
# Chart Builders
# ============================================================

def build_architecture_diagram():
    """Build the Keyhole pipeline architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5), facecolor=MPL_COLORS["bg_slide"])
    ax.set_facecolor(MPL_COLORS["bg_slide"])
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 3)
    ax.axis("off")

    boxes = [
        (0.2, 1.0, "Ingest\nFFmpeg", MPL_COLORS["blue"]),
        (2.3, 1.0, "Tier 1\nYOLO 11", MPL_COLORS["green"]),
        (4.4, 1.0, "Tier 2\nSAM 3", MPL_COLORS["orange"]),
        (6.5, 1.0, "Store\nSQLite", MPL_COLORS["purple"]),
        (8.6, 1.0, "Query\nNLQ/LLM", MPL_COLORS["blue"]),
    ]

    for x, y, label, color in boxes:
        rect = FancyBboxPatch(
            (x, y), 1.6, 1.2, boxstyle="round,pad=0.1",
            facecolor=color + "22", edgecolor=color, linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x + 0.8, y + 0.6, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color, family="monospace")

    # Arrows
    for i in range(4):
        x_start = boxes[i][0] + 1.6
        x_end = boxes[i + 1][0]
        y_mid = 1.6
        ax.annotate("", xy=(x_end, y_mid), xytext=(x_start, y_mid),
                     arrowprops=dict(arrowstyle="->", color=MPL_COLORS["dim"],
                                     lw=2, connectionstyle="arc3,rad=0"))

    # Subtitle labels
    subtitles = [
        (1.0, 0.65, "Frame\nextraction"),
        (3.1, 0.65, "Object\ndetection"),
        (5.2, 0.65, "Concept\nenrichment"),
        (7.3, 0.65, "Metadata\npersistence"),
        (9.4, 0.65, "Natural\nlanguage"),
    ]
    for x, y, text in subtitles:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=7, color=MPL_COLORS["dim"], family="monospace")

    fig.tight_layout(pad=0.5)
    return fig


def build_sam3_flop_breakdown(ref_data: dict):
    """Build SAM 3 compute distribution pie/bar chart."""
    categories = ref_data.get("category_summary", {})
    if not categories:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5),
                                     facecolor=MPL_COLORS["bg_slide"])

    # Pie chart — FLOP distribution
    ax1.set_facecolor(MPL_COLORS["bg_slide"])
    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["flops"], reverse=True)
    labels = [c[0] for c in sorted_cats]
    sizes = [c[1]["flops"] for c in sorted_cats]
    colors = [MPL_COLORS["blue"], MPL_COLORS["green"], MPL_COLORS["orange"],
              MPL_COLORS["purple"], MPL_COLORS["red"]][:len(labels)]

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        textprops={"color": MPL_COLORS["text"], "fontsize": 9},
        pctdistance=0.75, startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
    ax1.set_title("SAM 3 — FLOP Distribution", color=MPL_COLORS["text"],
                   fontsize=12, fontweight="bold", pad=10)

    # Bar chart — GFLOPs by category
    ax2.set_facecolor(MPL_COLORS["bg_slide"])
    gflops = [c[1]["flops"] / 1e9 for c in sorted_cats]
    bars = ax2.barh(labels[::-1], gflops[::-1], color=colors[::-1], height=0.6)
    ax2.set_xlabel("GFLOPs", color=MPL_COLORS["text"], fontsize=10)
    ax2.set_title("SAM 3 — GFLOPs by Op Type", color=MPL_COLORS["text"],
                   fontsize=12, fontweight="bold", pad=10)
    ax2.tick_params(colors=MPL_COLORS["dim"], labelsize=9)
    ax2.spines[:].set_color(MPL_COLORS["grid"])
    ax2.xaxis.grid(True, color=MPL_COLORS["grid"], alpha=0.3)

    for bar, val in zip(bars, gflops[::-1]):
        ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}", va="center", color=MPL_COLORS["text"], fontsize=8)

    fig.tight_layout(pad=1.5)
    return fig


def build_roofline_chart(targets: dict, sam3_ref: Optional[dict]):
    """Build a roofline model chart for edge hardware targets."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), facecolor=MPL_COLORS["bg_slide"])
    ax.set_facecolor(MPL_COLORS["bg_slide"])

    target_styles = {
        "rtx5090": (MPL_COLORS["blue"], "o", "RTX 5090"),
        "nxp_edge": (MPL_COLORS["green"], "s", "NXP Edge MPU"),
        "nxp_edge_lite": (MPL_COLORS["orange"], "^", "NXP Edge Lite"),
    }

    ai_range = np.logspace(-1, 3, 200)

    for name, spec in targets.items():
        color, marker, label = target_styles.get(name, (MPL_COLORS["dim"], "x", name))
        peak_tops = spec["tops_bf16"] * spec.get("compute_efficiency", 0.65)
        peak_bw = spec["mem_bandwidth_gbs"] * spec.get("bandwidth_efficiency", 0.8)

        # Roofline: performance = min(peak_tops, AI * peak_bw)
        # peak_bw is GB/s, AI is FLOPs/byte, so AI * peak_bw = GFLOP/s when AI in FLOP/byte
        # peak_tops is TOPS = 1000 GFLOP/s
        perf = np.minimum(peak_tops * 1000, ai_range * peak_bw)
        ax.loglog(ai_range, perf, color=color, linewidth=2, label=label)

        # Ridge point
        ridge_ai = (peak_tops * 1000) / peak_bw
        ax.axvline(x=ridge_ai, color=color, linestyle=":", alpha=0.4)

    # Plot workload points
    workloads = [
        ("YOLO 11x", 85.0, 196.0, MPL_COLORS["green"]),
        ("SAM 3 PE", 120.0, 3500.0, MPL_COLORS["orange"]),
        ("SAM 3 DETR", 15.0, 200.0, MPL_COLORS["purple"]),
        ("LLM decode", 1.0, 5.0, MPL_COLORS["red"]),
    ]

    for wl_name, ai, gflops, color in workloads:
        ax.plot(ai, gflops, "D", color=color, markersize=10, markeredgecolor="white",
                markeredgewidth=1.5, zorder=5)
        ax.annotate(wl_name, (ai, gflops), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)", color=MPL_COLORS["text"], fontsize=11)
    ax.set_ylabel("Performance (GFLOP/s)", color=MPL_COLORS["text"], fontsize=11)
    ax.set_title("Roofline Model — Workloads vs Hardware Targets",
                  color=MPL_COLORS["text"], fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, facecolor=MPL_COLORS["bg"],
              edgecolor=MPL_COLORS["grid"], labelcolor=MPL_COLORS["text"])
    ax.tick_params(colors=MPL_COLORS["dim"], labelsize=9)
    ax.grid(True, which="both", color=MPL_COLORS["grid"], alpha=0.2)
    ax.spines[:].set_color(MPL_COLORS["grid"])

    fig.tight_layout(pad=1.0)
    return fig


def build_latency_comparison_chart(runs: list[dict], targets: dict):
    """Build a bar chart comparing per-frame latency across runs and targets."""
    if not runs:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor=MPL_COLORS["bg_slide"])
    ax.set_facecolor(MPL_COLORS["bg_slide"])

    run_labels = []
    yolo_times = []
    sam3_times = []

    for run in runs[-8:]:  # Show last 8 runs
        label = run.get("video", {}).get("name", run.get("run_id", "unknown"))
        label = label.replace(".mp4", "")
        if len(label) > 15:
            label = label[:12] + "..."
        run_labels.append(label)
        yolo_times.append(run.get("yolo", {}).get("avg_ms", 0))
        sam3_times.append(run.get("sam3", {}).get("avg_enrichment_ms", 0))

    x = np.arange(len(run_labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, yolo_times, width, label="YOLO 11x",
                    color=MPL_COLORS["green"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, sam3_times, width, label="SAM 3",
                    color=MPL_COLORS["orange"], alpha=0.85)

    ax.set_ylabel("Latency (ms)", color=MPL_COLORS["text"], fontsize=11)
    ax.set_title("Per-Frame Inference Latency by Run",
                  color=MPL_COLORS["text"], fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=30, ha="right")
    ax.legend(facecolor=MPL_COLORS["bg"], edgecolor=MPL_COLORS["grid"],
              labelcolor=MPL_COLORS["text"])
    ax.tick_params(colors=MPL_COLORS["dim"], labelsize=9)
    ax.spines[:].set_color(MPL_COLORS["grid"])
    ax.yaxis.grid(True, color=MPL_COLORS["grid"], alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    color=MPL_COLORS["green"], fontsize=8)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    color=MPL_COLORS["orange"], fontsize=8)

    fig.tight_layout(pad=1.0)
    return fig


# ============================================================
# NPU Projection Engine (mirrors npu_emulator.py logic)
# ============================================================

def project_for_deck(run: dict, targets: dict) -> list[dict]:
    """Project a run's workload onto all hardware targets."""
    from src.emulate.npu_emulator import (
        NPUEmulator, RTX_5090, HardwareSpec, WorkloadProfile,
    )

    results = []
    for name, spec_dict in targets.items():
        spec = HardwareSpec(**{k: v for k, v in spec_dict.items()
                              if k in HardwareSpec.__dataclass_fields__})
        emulator = NPUEmulator(reference=RTX_5090, target=spec)

        yolo_data = run.get("yolo", {})
        sam3_data = run.get("sam3", {})

        yolo_wl = WorkloadProfile(
            stage_name="yolo_detection", model_name="yolo11x",
            param_count=int(yolo_data.get("params_m", 57) * 1e6),
            model_size_bytes=int(yolo_data.get("params_m", 57) * 1e6 * 2),
            precision="fp16", gflops_per_inference=196.0,
            arithmetic_intensity=85.0,
            measured_latency_ms=yolo_data.get("avg_ms", 10.0),
            measured_gpu=RTX_5090.name,
            peak_activation_bytes=int(0.2e9),
        )

        sam3_wl = WorkloadProfile(
            stage_name="sam3_enrichment", model_name="sam3_full",
            param_count=848_000_000,
            model_size_bytes=int(848e6 * 2),
            precision="bf16", gflops_per_inference=350.0,
            arithmetic_intensity=120.0,
            measured_latency_ms=sam3_data.get("avg_enrichment_ms", 30.0),
            measured_gpu=RTX_5090.name,
            peak_activation_bytes=int(1.0e9),
        )

        yolo_r = emulator.project_workload(yolo_wl)
        sam3_r = emulator.project_workload(sam3_wl)

        combined_ms = yolo_r.projected_latency_ms + sam3_r.projected_latency_ms
        combined_fps = 1000.0 / combined_ms if combined_ms > 0 else 0

        results.append({
            "target": spec.name,
            "target_key": name,
            "yolo_projected_ms": yolo_r.projected_latency_ms,
            "yolo_bottleneck": yolo_r.bottleneck,
            "sam3_projected_ms": sam3_r.projected_latency_ms,
            "sam3_bottleneck": sam3_r.bottleneck,
            "combined_ms": combined_ms,
            "combined_fps": combined_fps,
            "feasible_1fps": combined_ms < 1000,
            "feasible_5fps": combined_ms < 200,
            "yolo_fits": yolo_r.fits_in_memory,
            "sam3_fits": sam3_r.fits_in_memory,
            "tops": spec.tops_bf16,
            "bw_gbs": spec.mem_bandwidth_gbs,
            "mem_gb": spec.mem_capacity_gb,
            "tdp_w": spec.tdp_watts,
        })

    return results


# ============================================================
# Slide Builders
# ============================================================

def slide_title(prs: Presentation):
    """Slide 1: Title slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    set_slide_bg(slide, C.BG_DARK)

    add_text_box(slide, Inches(1), Inches(1.5), Inches(8), Inches(1),
                 "KEYHOLE", font_size=48, color=C.ACCENT_BLUE, bold=True,
                 alignment=PP_ALIGN.CENTER, font_name="Segoe UI")

    add_text_box(slide, Inches(1), Inches(2.5), Inches(8), Inches(0.6),
                 "Open-Source AI Key Prototype", font_size=24,
                 color=C.TEXT_WHITE, alignment=PP_ALIGN.CENTER)

    add_text_box(slide, Inches(1), Inches(3.2), Inches(8), Inches(0.5),
                 "Edge AI Video Intelligence Pipeline  |  Workload Characterization & NPU Feasibility",
                 font_size=14, color=C.TEXT_DIM, alignment=PP_ALIGN.CENTER)

    # Accent line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(3), Inches(3.8), Inches(4), Pt(2),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = C.ACCENT_BLUE
    line.line.fill.background()

    timestamp = datetime.now().strftime("%B %d, %Y")
    add_text_box(slide, Inches(1), Inches(4.2), Inches(8), Inches(0.5),
                 f"Generated {timestamp}  |  github.com/kylefoxaustin/keyhole",
                 font_size=11, color=C.TEXT_DIM, alignment=PP_ALIGN.CENTER)


def slide_architecture(prs: Presentation):
    """Slide 2: Pipeline architecture diagram."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title_bar(slide, "Pipeline Architecture",
                  "5-stage edge AI video intelligence pipeline")

    fig = build_architecture_diagram()
    img_stream = fig_to_image_stream(fig)
    slide.shapes.add_picture(img_stream, Inches(0.3), Inches(1.5),
                              width=Inches(9.4))

    # Key specs box
    specs = [
        "Tier 1: YOLO 11x  (56.9M params, ~196 GFLOPs/frame)",
        "Tier 2: SAM 3 Concept Segmentation  (840.5M params, ~4,175 GFLOPs/frame)",
        "NLQ: Claude API / Ollama / Skippy  (3B-8B int4 models)",
        "Store: SQLite + FTS5 + optional vector embeddings",
    ]
    for i, spec in enumerate(specs):
        add_text_box(slide, Inches(0.8), Inches(4.0 + i * 0.35),
                     Inches(8.5), Inches(0.35),
                     spec, font_size=10, color=C.TEXT_DIM)


def slide_sam3_reference(prs: Presentation, ref_data: dict):
    """Slide 3: SAM 3 reference architecture breakdown."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    summary = ref_data.get("model_summary", {})
    total_gflops = summary.get("total_gflops", 0)
    total_params = summary.get("total_params", 0)
    num_layers = len(ref_data.get("layers", []))

    add_title_bar(slide, "SAM 3 Reference Architecture",
                  f"{total_params/1e6:.0f}M params  |  {total_gflops:.0f} GFLOPs  |  {num_layers} layers")

    fig = build_sam3_flop_breakdown(ref_data)
    if fig:
        img_stream = fig_to_image_stream(fig)
        slide.shapes.add_picture(img_stream, Inches(0.2), Inches(1.4),
                                  width=Inches(9.6))

    # Component table
    components = ref_data.get("model_summary", {}).get("components", {})
    if components:
        headers = ["Component", "Params", "Role"]
        rows = []
        for name, info in components.items():
            rows.append([
                name.replace("_", " ").title(),
                f"{info['params_m']}M",
                info["role"],
            ])
        add_styled_table(slide, Inches(0.5), Inches(4.5), Inches(9), Inches(1.2),
                         headers, rows,
                         col_widths=[Inches(2), Inches(1), Inches(6)])


def slide_roofline(prs: Presentation, targets: dict, sam3_ref: Optional[dict]):
    """Slide 4: Roofline model."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title_bar(slide, "Roofline Model — Compute vs Bandwidth",
                  "Workload placement determines bottleneck on each hardware target")

    fig = build_roofline_chart(targets, sam3_ref)
    img_stream = fig_to_image_stream(fig)
    slide.shapes.add_picture(img_stream, Inches(0.1), Inches(1.3),
                              width=Inches(9.8))


def slide_run_results(prs: Presentation, run: dict, run_index: int):
    """Per-run result slide with profiling data."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    video = run.get("video", {})
    video_name = video.get("name", "unknown")
    run_id = run.get("run_id", "unknown")

    add_title_bar(slide, f"Test Run: {video_name}",
                  f"Run ID: {run_id}  |  {video.get('width', '?')}x{video.get('height', '?')} @ {video.get('extract_fps', '?')} FPS extraction")

    yolo = run.get("yolo", {})
    sam3 = run.get("sam3", {})
    pipeline = run.get("pipeline", {})

    # YOLO results
    add_text_box(slide, Inches(0.5), Inches(1.4), Inches(4), Inches(0.4),
                 "YOLO 11x Detection", font_size=16, color=C.ACCENT_GREEN, bold=True)

    yolo_rows = [
        ["Model", yolo.get("model", "yolo11x.pt")],
        ["Avg Inference", f"{yolo.get('avg_ms', 0):.1f} ms"],
        ["P95 Latency", f"{yolo.get('p95_ms', 0):.1f} ms"],
        ["P99 Latency", f"{yolo.get('p99_ms', 0):.1f} ms"],
        ["Parameters", f"{yolo.get('params_m', 0):.1f}M"],
    ]
    add_styled_table(slide, Inches(0.5), Inches(1.9), Inches(4), Inches(1.8),
                     ["Metric", "Value"], yolo_rows,
                     col_widths=[Inches(2), Inches(2)])

    # SAM 3 results
    add_text_box(slide, Inches(5.2), Inches(1.4), Inches(4), Inches(0.4),
                 "SAM 3 Enrichment", font_size=16, color=C.ACCENT_ORANGE, bold=True)

    sam3_model = sam3.get("model", "not loaded")
    sam3_rows = [
        ["Model", sam3_model],
        ["Avg Enrichment", f"{sam3.get('avg_enrichment_ms', 0):.0f} ms"],
        ["P95 Latency", f"{sam3.get('p95_enrichment_ms', 0):.0f} ms"],
        ["Parameters", f"{sam3.get('model_params_m', 0):.1f}M"],
        ["Frames Profiled", str(sam3.get("total_frames", 0))],
    ]
    add_styled_table(slide, Inches(5.2), Inches(1.9), Inches(4.3), Inches(1.8),
                     ["Metric", "Value"], sam3_rows,
                     col_widths=[Inches(2.2), Inches(2.1)])

    # Pipeline summary
    add_text_box(slide, Inches(0.5), Inches(4.0), Inches(9), Inches(0.4),
                 "Pipeline Summary", font_size=16, color=C.ACCENT_BLUE, bold=True)

    pipe_rows = [
        ["Total Frames", str(pipeline.get("total_frames", 0))],
        ["Total Detections", str(pipeline.get("total_detections", 0))],
        ["Pipeline Time", f"{pipeline.get('total_seconds', 0):.1f}s"],
        ["Throughput", f"{pipeline.get('fps', 0):.2f} FPS"],
        ["Detect Only", str(pipeline.get("detect_only", False))],
    ]
    add_styled_table(slide, Inches(0.5), Inches(4.5), Inches(9), Inches(1.3),
                     ["Metric", "Value", "Metric", "Value", "Metric"],
                     # Flatten to single row table
                     [],
    )
    # Actually use a simpler approach — two-column table
    slide.shapes[-1]._element.getparent().remove(slide.shapes[-1]._element)

    add_styled_table(slide, Inches(0.5), Inches(4.5), Inches(9), Inches(1.2),
                     ["Frames", "Detections", "Pipeline Time", "Throughput", "Mode"],
                     [[
                         str(pipeline.get("total_frames", 0)),
                         str(pipeline.get("total_detections", 0)),
                         f"{pipeline.get('total_seconds', 0):.1f}s",
                         f"{pipeline.get('fps', 0):.2f} FPS",
                         "Detect Only" if pipeline.get("detect_only") else "Full (YOLO+SAM3)",
                     ]])


def slide_npu_projections(prs: Presentation, run: dict, targets: dict):
    """NPU projection slide for a given run."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    video_name = run.get("video", {}).get("name", "unknown")
    add_title_bar(slide, "Edge NPU Projections",
                  f"Based on: {video_name}  |  Measured on RTX 5090, projected to edge targets")

    projections = project_for_deck(run, targets)

    headers = ["Target", "TOPS", "BW (GB/s)", "YOLO", "SAM 3",
               "Combined", "FPS", "1 FPS?", "5 FPS?", "TDP"]
    rows = []
    for p in projections:
        rows.append([
            p["target"],
            f"{p['tops']:.0f}",
            f"{p['bw_gbs']:.0f}",
            f"{p['yolo_projected_ms']:.1f}ms",
            f"{p['sam3_projected_ms']:.1f}ms",
            f"{p['combined_ms']:.1f}ms",
            f"{p['combined_fps']:.0f}",
            "YES" if p["feasible_1fps"] else "NO",
            "YES" if p["feasible_5fps"] else "NO",
            f"{p['tdp_w']:.0f}W",
        ])

    add_styled_table(slide, Inches(0.3), Inches(1.5), Inches(9.4), Inches(1.5),
                     headers, rows)

    # Key insight callout
    nxp = next((p for p in projections if p["target_key"] == "nxp_edge"), None)
    if nxp:
        bottleneck = nxp["sam3_bottleneck"]
        insight = (
            f"NXP Edge MPU: {nxp['combined_ms']:.1f}ms combined "
            f"({nxp['combined_fps']:.0f} FPS)  |  "
            f"SAM 3 is {bottleneck}-bound  |  "
            f"{'FEASIBLE' if nxp['feasible_1fps'] else 'NOT FEASIBLE'} at 1 FPS extraction  |  "
            f"{'FEASIBLE' if nxp['feasible_5fps'] else 'NOT FEASIBLE'} at 5 FPS extraction"
        )

        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.5), Inches(3.5), Inches(9), Inches(1.0),
        )
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0x0D, 0x2B, 0x0D)
        box.line.color.rgb = C.ACCENT_GREEN
        box.line.width = Pt(2)

        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = "KEY FINDING"
        p.font.size = Pt(12)
        p.font.color.rgb = C.ACCENT_GREEN
        p.font.bold = True
        p.font.name = "Segoe UI"

        p2 = tf.add_paragraph()
        p2.text = insight
        p2.font.size = Pt(11)
        p2.font.color.rgb = C.TEXT_BRIGHT
        p2.font.name = "Segoe UI"


def slide_run_comparison(prs: Presentation, runs: list[dict], targets: dict):
    """Comparison chart across all runs."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title_bar(slide, "Run Comparison",
                  f"{len(runs)} test runs  |  Inference latency on RTX 5090")

    fig = build_latency_comparison_chart(runs, targets)
    if fig:
        img_stream = fig_to_image_stream(fig)
        slide.shapes.add_picture(img_stream, Inches(0.1), Inches(1.3),
                                  width=Inches(9.8))


def slide_bandwidth_wall(prs: Presentation):
    """Slide: Why SAM 3 hits a bandwidth wall on edge hardware."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C.BG_DARK)

    add_title_bar(slide, "The Bandwidth Wall",
                  "SAM 3 is deeply memory-bandwidth-bound — TOPS don't matter")

    lines = [
        ("MEASURED: RTX 5090 (209 TOPS, 1792 GB/s, 72 MB L2 cache)", C.ACCENT_BLUE, True),
        ("  GPU kernel time: 102ms   |   Wall clock: 107ms   |   CPU overhead: only 5ms", C.TEXT_BRIGHT, False),
        ("  Theoretical compute floor (350 GFLOPs / 146 TOPS): 2.4ms", C.TEXT_DIM, False),
        ("  Actual GPU time: 102ms  =  42x longer than compute-only", C.ACCENT_ORANGE, True),
        ("  The GPU spends 98% of its time waiting for memory, not computing", C.TEXT_DIM, False),
        ("", C.TEXT_DIM, False),
        ("WHY: Transformer activations stream through VRAM every layer", C.ACCENT_BLUE, True),
        ("  840M params  |  3.71 GB peak activations  |  ~147 GB total memory traffic per frame", C.TEXT_BRIGHT, False),
        ("  Arithmetic intensity: ~2 FLOPs/byte (ridge point: 117 FLOPs/byte)", C.TEXT_DIM, False),
        ("  Even the 5090's 72 MB L2 cache can't absorb this — activations are too large", C.ACCENT_ORANGE, False),
        ("", C.TEXT_DIM, False),
        ("EDGE PROJECTION: NXP Edge MPU (200 TOPS, 134.4 GB/s, ~4 MB SRAM)", C.ACCENT_BLUE, True),
        ("  Bandwidth ratio: 1523 / 101 = 15.1x less bandwidth than RTX 5090", C.TEXT_BRIGHT, False),
        ("  Projected: ~2,400ms per frame (0.4 FPS)  |  14x slowdown", C.NOT_FEASIBLE, True),
        ("  200 TOPS is irrelevant — compute is only 2% of total time", C.TEXT_DIM, False),
        ("  Memory capacity: 7.07 GB peak vs 8 GB total = no headroom", C.ACCENT_ORANGE, False),
    ]

    for i, (text, color, bold) in enumerate(lines):
        if text:
            add_text_box(slide, Inches(0.6), Inches(1.5 + i * 0.33),
                         Inches(8.8), Inches(0.33),
                         text, font_size=11, color=color, bold=bold)


def slide_bandwidth_requirements(prs: Presentation):
    """Slide: Required bandwidth for target framerates."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_title_bar(slide, "Required Memory Bandwidth for Real-Time SAM 3",
                  "~147 GB memory traffic per frame at 1080p, 9 concept prompts")

    headers = ["Target FPS", "Time Budget", "Required BW (eff)", "Required BW (raw)",
               "Memory Tech", "Feasible at 25W?"]
    rows = [
        ["1 FPS",  "1000ms", "147 GB/s",   "196 GB/s",   "256-bit LPDDR5X", "Possible"],
        ["5 FPS",  "200ms",  "735 GB/s",   "980 GB/s",   "HBM2e or 512-bit", "Difficult"],
        ["10 FPS", "100ms",  "1,470 GB/s", "1,960 GB/s", "HBM3 (desktop-class)", "No"],
        ["24 FPS", "42ms",   "3,528 GB/s", "4,704 GB/s", "Beyond HBM3", "No"],
        ["30 FPS", "33ms",   "4,414 GB/s", "5,885 GB/s", "Multi-die HBM3e", "No"],
    ]

    add_styled_table(slide, Inches(0.3), Inches(1.5), Inches(9.4), Inches(2.2),
                     headers, rows)

    # Current hardware reference
    ref_lines = [
        ("Current Hardware Reference:", C.ACCENT_BLUE, True),
        ("  NXP Edge MPU:  134.4 GB/s  (128-bit LPDDR5X)  →  0.4 FPS", C.NOT_FEASIBLE, False),
        ("  RTX 5090:      1,792 GB/s  (512-bit GDDR7)    →  6.0 FPS", C.TEXT_BRIGHT, False),
        ("  NVIDIA H200:   4,800 GB/s  (HBM3e)            →  ~33 FPS (Meta's paper: 30ms)", C.ACCENT_GREEN, False),
        ("", C.TEXT_DIM, False),
        ("Conclusion: Real-time SAM 3 requires HBM-class bandwidth.", C.ACCENT_ORANGE, True),
        ("For edge at 25W, the model must change — not the hardware.", C.TEXT_WHITE, True),
    ]

    for i, (text, color, bold) in enumerate(ref_lines):
        if text:
            add_text_box(slide, Inches(0.6), Inches(4.0 + i * 0.35),
                         Inches(8.8), Inches(0.35),
                         text, font_size=11, color=color, bold=bold)


def slide_prompt_scaling(prs: Presentation):
    """Slide: How concept prompt count affects performance."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_title_bar(slide, "Prompt Count Scaling — Decoder Cost Is Linear",
                  "Vision encoder is fixed cost (~70ms); each concept prompt adds ~6ms in decoder")

    # Measured data table
    headers = ["Concepts", "RTX 5090", "FPS", "Edge Projected", "Edge FPS", "Example Prompts"]
    rows = [
        ["1", "72ms", "13.8", "~1,068ms", "0.9", "person"],
        ["3", "90ms", "11.1", "~1,333ms", "0.7", "person, vehicle, dog"],
        ["9 (current)", "121ms", "8.3", "~1,791ms", "0.6", "person, vehicle, car, truck, ..."],
        ["18", "177ms", "5.7", "~2,618ms", "0.4", "full concept set + accessories"],
    ]
    add_styled_table(slide, Inches(0.3), Inches(1.4), Inches(9.4), Inches(1.7),
                     headers, rows)

    # Cost breakdown chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), facecolor=MPL_COLORS["bg_slide"])

    # Left: Latency vs prompt count
    ax1.set_facecolor(MPL_COLORS["bg_slide"])
    prompts = [1, 3, 9, 18]
    latencies = [72, 90, 121, 177]
    ax1.plot(prompts, latencies, "o-", color=MPL_COLORS["blue"], linewidth=2, markersize=8)
    ax1.axhline(y=72, color=MPL_COLORS["orange"], linestyle="--", alpha=0.7, label="Vision encoder floor (72ms)")
    ax1.fill_between(prompts, 72, latencies, alpha=0.2, color=MPL_COLORS["green"], label="Decoder cost")
    ax1.set_xlabel("Number of Concept Prompts", color=MPL_COLORS["text"], fontsize=10)
    ax1.set_ylabel("Latency (ms)", color=MPL_COLORS["text"], fontsize=10)
    ax1.set_title("Latency vs Prompt Count (RTX 5090)", color=MPL_COLORS["text"], fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, facecolor=MPL_COLORS["bg"], edgecolor=MPL_COLORS["grid"],
               labelcolor=MPL_COLORS["text"])
    ax1.tick_params(colors=MPL_COLORS["dim"], labelsize=9)
    ax1.spines[:].set_color(MPL_COLORS["grid"])
    ax1.grid(True, alpha=0.2, color=MPL_COLORS["grid"])

    # Right: Stacked bar showing encoder vs decoder split
    ax2.set_facecolor(MPL_COLORS["bg_slide"])
    encoder_ms = [70, 70, 70, 70]
    decoder_ms = [2, 20, 51, 107]
    labels = ["1", "3", "9", "18"]
    x = np.arange(len(labels))
    ax2.bar(x, encoder_ms, 0.5, label="Vision Encoder (fixed)", color=MPL_COLORS["orange"])
    ax2.bar(x, decoder_ms, 0.5, bottom=encoder_ms, label="Text + DETR Decoder", color=MPL_COLORS["green"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Prompts", color=MPL_COLORS["text"], fontsize=10)
    ax2.set_ylabel("Latency (ms)", color=MPL_COLORS["text"], fontsize=10)
    ax2.set_title("Encoder vs Decoder Split", color=MPL_COLORS["text"], fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, facecolor=MPL_COLORS["bg"], edgecolor=MPL_COLORS["grid"],
               labelcolor=MPL_COLORS["text"])
    ax2.tick_params(colors=MPL_COLORS["dim"], labelsize=9)
    ax2.spines[:].set_color(MPL_COLORS["grid"])

    fig.tight_layout(pad=1.5)
    img_stream = fig_to_image_stream(fig)
    slide.shapes.add_picture(img_stream, Inches(0.2), Inches(3.3), width=Inches(9.6))

    # Key insight
    add_text_box(slide, Inches(0.5), Inches(6.5), Inches(9), Inches(0.5),
                 "Vision encoder (~70ms) is the hard floor. Even 1 prompt → 1,068ms on edge. "
                 "Prompt tuning helps on desktop (14 FPS) but cannot fix the edge bandwidth gap.",
                 font_size=10, color=C.ACCENT_ORANGE, bold=True)


def slide_quantization_tested(prs: Presentation):
    """Slide: Weight-only INT8 quantization results — doesn't help."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C.BG_DARK)

    add_title_bar(slide, "Quantization Tested — Weight-Only INT8 Doesn't Help",
                  "Measured with torchao Int8WeightOnlyConfig on RTX 5090 (9 concepts, 720p)")

    headers = ["Metric", "BF16 (baseline)", "INT8 Weight-Only", "Delta"]
    rows = [
        ["Wall clock", "121ms", "121ms", "0% (no change)"],
        ["GPU kernel time", "102ms", "117ms", "15% SLOWER"],
        ["Peak VRAM", "7.07 GB", "5.11 GB", "2 GB saved"],
        ["Edge projection", "1,791ms (0.6 FPS)", "1,731ms (0.6 FPS)", "Negligible"],
    ]
    add_styled_table(slide, Inches(0.3), Inches(1.4), Inches(9.4), Inches(1.5),
                     headers, rows)

    lines = [
        ("WHY IT DOESN'T HELP:", C.ACCENT_BLUE, True),
        ("  Weight-only quantization shrinks model params (3.36 GB → 704 MB)", C.TEXT_BRIGHT, False),
        ("  But activations stay in BF16 — and activations are 98% of bandwidth traffic", C.ACCENT_ORANGE, True),
        ("  Dequantization overhead (INT8→BF16 per matmul) adds latency", C.TEXT_DIM, False),
        ("  Lost Meta's fused addmm_act kernel → unfused path is slower", C.TEXT_DIM, False),
        ("", C.TEXT_DIM, False),
        ("WHAT WOULD HELP: Activation quantization (INT8 or FP8 activations)", C.ACCENT_GREEN, True),
        ("  Would halve the dominant memory traffic between layers", C.TEXT_BRIGHT, False),
        ("  Edge projection: ~1,700ms → ~850ms (1.2 FPS) — still not real-time", C.ACCENT_ORANGE, False),
    ]

    for i, (text, color, bold) in enumerate(lines):
        if text:
            add_text_box(slide, Inches(0.6), Inches(3.2 + i * 0.30),
                         Inches(8.8), Inches(0.30),
                         text, font_size=11, color=color, bold=bold)


def slide_activation_quant_challenges(prs: Presentation):
    """Slide: Why activation quantization is hard for SAM 3."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_title_bar(slide, "Activation Quantization — Why It's Hard for SAM 3",
                  "The one lever that could halve edge latency, but requires research-grade effort")

    headers = ["Challenge", "Impact", "Mitigation"]
    rows = [
        ["Attention score clipping",
         "INT8 clips outlier scores that encode\n'attend strongly to this location'",
         "FP8 (E4M3) preserves dynamic range"],
        ["Text-vision cross-attention",
         "Quant errors → false positives, missed\ndetections, concept misclassification",
         "Per-layer sensitivity analysis,\nmixed-precision (keep critical layers BF16)"],
        ["Calibration data dependency",
         "Scale factors derived from cal data;\nmismatch → degraded accuracy",
         "Diverse calibration set matching\ndeployment distribution"],
        ["Flash Attention 3 incompatibility",
         "No INT8 flash attention kernel;\nfallback to standard attention = slower",
         "FP8 flash attention (future),\nor accept unfused penalty"],
        ["Not all layers are equal",
         "LayerNorm outputs, residuals, first/last\nlayers are quantization-sensitive",
         "Mixed-precision: INT8 bulk matmuls,\nBF16 for sensitive layers"],
    ]

    add_styled_table(slide, Inches(0.2), Inches(1.4), Inches(9.6), Inches(3.0),
                     headers, rows,
                     col_widths=[Inches(2.5), Inches(3.5), Inches(3.6)])

    # Projection box
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(4.7), Inches(9), Inches(2.3),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0x0D, 0x1A, 0x2B)
    box.line.color.rgb = C.ACCENT_BLUE
    box.line.width = Pt(2)

    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "EVEN WITH PERFECT ACTIVATION QUANTIZATION"
    p.font.size = Pt(12)
    p.font.color.rgb = C.ACCENT_BLUE
    p.font.bold = True
    p.font.name = "Segoe UI"

    projections = [
        "INT8 activations → ~2x traffic reduction → edge: ~850ms (1.2 FPS)",
        "FP8 activations  → ~2x traffic reduction → edge: ~850ms (1.2 FPS)",
        "INT4 activations  → ~4x traffic reduction → edge: ~425ms (2.4 FPS) — significant accuracy risk",
        "",
        "None of these reach 5 FPS (200ms budget) on 134.4 GB/s LPDDR5X.",
        "",
        "Viable paths to activation quantization today:",
        "  • SmoothQuant — shifts quant difficulty from activations to weights (proven on LLMs)",
        "  • FP8 (E4M3) — RTX 5090 supports natively, best accuracy/speed tradeoff",
        "  • Wait for Meta — official quantized SAM 3 checkpoint would bypass all issues",
    ]
    for line in projections:
        p2 = tf.add_paragraph()
        p2.text = line
        p2.font.size = Pt(10)
        p2.font.name = "Segoe UI"
        if "None of these" in line:
            p2.font.color.rgb = C.NOT_FEASIBLE
            p2.font.bold = True
        elif line.startswith("  •"):
            p2.font.color.rgb = C.ACCENT_GREEN
        elif "→" in line:
            p2.font.color.rgb = C.TEXT_BRIGHT
        else:
            p2.font.color.rgb = C.TEXT_DIM


def slide_resolution_lock(prs: Presentation):
    """Slide: Why reducing input resolution doesn't help."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C.BG_DARK)

    add_title_bar(slide, "Resolution Is Locked — Input Size Doesn't Matter",
                  "SAM 3 internally processes 1008x1008 regardless of input resolution")

    # Resolution comparison table
    headers = ["Input Resolution", "Internal Resolution", "ViT Tokens", "Avg Latency (5090)", "Detections/frame"]
    rows = [
        ["4K (3840x2160)", "1008x1008", "3,969", "196ms", "37.2"],
        ["1080p (1920x1080)", "1008x1008", "3,969", "139ms", "35.0"],
        ["720p (1280x720)", "1008x1008", "3,969", "117ms", "32.1"],
    ]
    add_styled_table(slide, Inches(0.3), Inches(1.5), Inches(9.4), Inches(1.3),
                     headers, rows)

    lines = [
        ("WHY: Rotary Position Embeddings (RoPE) are resolution-locked", C.ACCENT_BLUE, True),
        ("  The ViT uses 2D RoPE pre-computed for a 63x63 token grid (1008/16 = 63)", C.TEXT_BRIGHT, False),
        ("  Feeding a different resolution → shape mismatch → assertion failure", C.TEXT_DIM, False),
        ("  The model has rope_interp support but it requires rebuild + retraining", C.TEXT_DIM, False),
        ("", C.TEXT_DIM, False),
        ("WHAT THIS MEANS:", C.ACCENT_BLUE, True),
        ("  The 16ms savings from 4K→720p is just pre/post processing overhead", C.TEXT_BRIGHT, False),
        ("  Model compute + memory traffic is IDENTICAL at every input resolution", C.ACCENT_ORANGE, True),
        ("  Token count (3,969), FLOP count, and activation memory are all fixed", C.TEXT_DIM, False),
        ("  Only detection accuracy changes (fewer small objects at 720p)", C.TEXT_DIM, False),
        ("", C.TEXT_DIM, False),
        ("IMPLICATION FOR EDGE: Resolution reduction is NOT a viable optimization lever.", C.NOT_FEASIBLE, True),
        ("  The remaining options are: quantization, fewer params, or a different model.", C.TEXT_WHITE, False),
    ]

    for i, (text, color, bold) in enumerate(lines):
        if text:
            add_text_box(slide, Inches(0.6), Inches(3.1 + i * 0.30),
                         Inches(8.8), Inches(0.30),
                         text, font_size=11, color=color, bold=bold)


def slide_optimization_roadmap(prs: Presentation):
    """Slide: Path to real-time on edge hardware."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_title_bar(slide, "Optimization Roadmap — Path to Edge Real-Time",
                  "Model changes required to fit within 134.4 GB/s bandwidth budget")

    headers = ["Optimization", "Traffic Reduction", "Est. Edge FPS", "Status"]
    rows = [
        ["SAM 3 BF16, 9 prompts (baseline)", "1x (~147 GB)", "0.4 FPS", "MEASURED"],
        ["Lower input resolution (720p)", "~1x (no change)", "0.6 FPS", "TESTED — not viable"],
        ["Reduce internal resolution", "N/A", "N/A", "BLOCKED — RoPE locked"],
        ["INT8 weight-only quantization", "Weights only (not traffic)", "0.6 FPS", "TESTED — no speedup"],
        ["Fewer prompts (1 vs 9)", "~0.6x (decoder only)", "0.9 FPS", "TESTED — helps on desktop"],
        ["INT8 activation quantization", "~2x (halve act traffic)", "~1.2 FPS", "Research-grade effort"],
        ["FP8 activation (E4M3)", "~2x (halve act traffic)", "~1.2 FPS", "RTX 5090 native, not in SAM 3"],
        ["INT4 activation quantization", "~4x", "~2.4 FPS", "Significant accuracy risk"],
        ["EfficientSAM / MobileSAM", "~50-100x (5-50M params)", "~15-30 FPS", "Different model entirely"],
    ]

    add_styled_table(slide, Inches(0.3), Inches(1.4), Inches(9.4), Inches(2.8),
                     headers, rows)

    # Key insight
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(4.5), Inches(9), Inches(2.2),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0x1A, 0x0D, 0x2B)
    box.line.color.rgb = C.ACCENT_PURPLE
    box.line.width = Pt(2)

    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "NEXT STEPS (UPDATED)"
    p.font.size = Pt(13)
    p.font.color.rgb = C.ACCENT_PURPLE
    p.font.bold = True
    p.font.name = "Segoe UI"

    steps = [
        "1. Evaluate EfficientSAM / MobileSAM — only viable path to real-time on edge",
        "2. Investigate FP8 (E4M3) activation quantization on RTX 5090",
        "3. Test SmoothQuant for activation-safe INT8 on SAM 3 ViT backbone",
        "4. Consider hybrid: YOLO for detection + lightweight model for enrichment",
        "5. Monitor Meta for official quantized SAM 3 / SAM 3 Lite release",
    ]
    for step in steps:
        p2 = tf.add_paragraph()
        p2.text = step
        p2.font.size = Pt(10)
        p2.font.color.rgb = C.TEXT_BRIGHT
        p2.font.name = "Segoe UI"


def slide_summary(prs: Presentation, runs: list[dict], targets: dict):
    """Final summary slide with key findings."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C.BG_DARK)

    add_text_box(slide, Inches(0.5), Inches(0.4), Inches(9), Inches(0.6),
                 "Summary & Key Findings", font_size=28,
                 color=C.ACCENT_BLUE, bold=True, alignment=PP_ALIGN.CENTER)

    latest_run = runs[-1] if runs else {}
    projections = project_for_deck(latest_run, targets) if latest_run else []
    nxp = next((p for p in projections if p["target_key"] == "nxp_edge"), None)

    findings = []

    if latest_run:
        sam3_data = latest_run.get("sam3", {})
        sam3_ms = sam3_data.get("avg_inference_ms") or sam3_data.get("avg_enrichment_ms", 0)
        mode = sam3_data.get("mode", "sequential")
        if sam3_ms:
            findings.append(
                f"RTX 5090 Measured: SAM 3 {mode} = {sam3_ms:.0f}ms/frame "
                f"({1000/sam3_ms:.1f} FPS)"
            )

    findings.extend([
        "",
        "KEY FINDING: SAM 3 is deeply memory-bandwidth-bound",
        "  GPU kernel time: 102ms  |  Compute-only floor: 2.4ms  |  42x gap",
        "  98% of GPU time is waiting for memory, not computing",
        "  RTX 5090's 72 MB L2 cache cannot absorb 3.71 GB of activations",
        "",
        "NXP Edge MPU (200 TOPS / 134.4 GB/s / 25W):",
        "  Projected: ~2,400ms per frame (0.4 FPS)  |  14x slower than 5090",
        "  200 TOPS is sufficient — 134.4 GB/s bandwidth is the bottleneck",
        "  Memory: 7.07 GB peak vs 8 GB capacity = no headroom",
        "  VERDICT: SAM 3 at BF16/1080p is NOT FEASIBLE on LPDDR5X",
        "",
        "Path forward: model optimization (INT4 + lower res + smaller backbone)",
        "  or accept sub-1 FPS for non-real-time batch analysis",
    ])

    for i, finding in enumerate(findings):
        color = C.TEXT_WHITE if finding.strip() else C.TEXT_DIM
        if finding.strip().startswith("NXP"):
            color = C.ACCENT_GREEN
        elif "FEASIBLE" in finding and "NOT" not in finding:
            color = C.FEASIBLE
        elif "NOT FEASIBLE" in finding:
            color = C.NOT_FEASIBLE

        add_text_box(slide, Inches(1), Inches(1.5 + i * 0.38),
                     Inches(8), Inches(0.38),
                     finding, font_size=13, color=color)

    # Footer
    add_text_box(slide, Inches(1), Inches(6.5), Inches(8), Inches(0.3),
                 "Keyhole  |  TTA — Trust the Awesomeness",
                 font_size=10, color=C.TEXT_DIM, alignment=PP_ALIGN.CENTER)


# ============================================================
# Main Build
# ============================================================

@click.command()
@click.option("--output", "-o", default="data/output/keyhole_results.pptx",
              help="Output PPTX path")
@click.option("--runs-dir", default="data/output/runs",
              help="Directory containing run JSON files")
@click.option("--data-dir", default="data/output",
              help="Directory containing reference architecture exports")
def build_deck(output, runs_dir, data_dir):
    """Generate the Keyhole results PowerPoint deck."""
    from rich.console import Console
    console = Console()

    console.print("\n[bold]Keyhole — Building Results Deck[/]\n")

    runs_dir = Path(runs_dir)
    data_dir = Path(data_dir)
    output = Path(output)

    # Load data
    runs = load_runs(runs_dir)
    sam3_ref = load_sam3_reference(data_dir)
    targets = load_npu_targets()

    console.print(f"  Runs found:     {len(runs)}")
    console.print(f"  SAM 3 ref:      {'YES' if sam3_ref else 'NO'}")
    console.print(f"  NPU targets:    {len(targets)}")

    # Build presentation
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # Slide 1: Title
    console.print("  Building: Title slide")
    slide_title(prs)

    # Slide 2: Architecture
    console.print("  Building: Architecture diagram")
    slide_architecture(prs)

    # Slide 3: SAM 3 Reference
    if sam3_ref:
        console.print("  Building: SAM 3 reference breakdown")
        slide_sam3_reference(prs, sam3_ref)

    # Slide 4: Roofline model
    console.print("  Building: Roofline model")
    slide_roofline(prs, targets, sam3_ref)

    # Per-run slides
    for i, run in enumerate(runs):
        video_name = run.get("video", {}).get("name", f"Run {i+1}")
        console.print(f"  Building: Run results — {video_name}")
        slide_run_results(prs, run, i)
        slide_npu_projections(prs, run, targets)

    # Comparison slide (if multiple runs)
    if len(runs) >= 1:
        console.print("  Building: Run comparison chart")
        slide_run_comparison(prs, runs, targets)

    # Bandwidth wall analysis
    console.print("  Building: Bandwidth wall analysis")
    slide_bandwidth_wall(prs)

    # Bandwidth requirements
    console.print("  Building: Bandwidth requirements")
    slide_bandwidth_requirements(prs)

    # Quantization results
    console.print("  Building: Quantization tested (weight-only INT8)")
    slide_quantization_tested(prs)

    # Activation quantization challenges
    console.print("  Building: Activation quantization challenges")
    slide_activation_quant_challenges(prs)

    # Prompt scaling analysis
    console.print("  Building: Prompt count scaling")
    slide_prompt_scaling(prs)

    # Resolution lock finding
    console.print("  Building: Resolution lock analysis")
    slide_resolution_lock(prs)

    # Optimization roadmap
    console.print("  Building: Optimization roadmap")
    slide_optimization_roadmap(prs)

    # Summary slide
    console.print("  Building: Summary & findings")
    slide_summary(prs, runs, targets)

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))

    total_slides = len(prs.slides)
    console.print(f"\n  [bold green]Deck generated: {output}[/]")
    console.print(f"  Total slides: {total_slides}")
    console.print(f"  Runs included: {len(runs)}\n")


if __name__ == "__main__":
    build_deck()
