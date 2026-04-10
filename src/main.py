"""
Keyhole — Main Pipeline Orchestrator

CLI entry point for processing videos and querying detections.

Usage:
    python -m src.main process --video path/to/video.mp4
    python -m src.main query --q "find person wearing red hat"
    python -m src.main serve
    python -m src.main stats
"""

import sys
import time
import json
import asyncio
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.logging import RichHandler

from config.settings import settings

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("sentinel")


@click.group()
def cli():
    """Keyhole — Edge AI Video Intelligence Pipeline"""
    pass


@cli.command()
@click.option("--video", "-v", required=True, help="Path to video file")
@click.option("--fps", "-f", default=None, type=float, help="Frame extraction FPS")
@click.option("--max-frames", "-m", default=0, type=int, help="Max frames to process")
@click.option("--detect-only", is_flag=True, help="Skip SAM 3 enrichment")
@click.option("--profile", is_flag=True, help="Enable GPU profiling")
@click.option("--emulate-npu", default=None, help="Emulate target NPU (preset name or JSON path)")
def process(video: str, fps: float, max_frames: int, detect_only: bool, profile: bool, emulate_npu: str):
    """Process a video through the detection pipeline."""

    from src.ingest.video import extract_frames, probe_video
    from src.detect.yolo import YOLODetector
    from src.enrich.sam3 import SAM3Enricher, EnrichedFrame, EnrichedDetection
    from src.store.db import DetectionStore

    video_path = Path(video)
    if not video_path.exists():
        console.print(f"[red]Video not found: {video_path}[/]")
        sys.exit(1)

    extract_fps = fps or settings.video.extract_fps
    max_f = max_frames or settings.video.max_frames
    do_profile = profile or settings.profile.enabled

    # --- NPU Emulation ---
    npu_throttle_yolo = None
    npu_throttle_sam3 = None
    if emulate_npu:
        do_profile = True  # Force profiling when emulating
        from src.emulate.npu_emulator import (
            NPUEmulator, PRESET_TARGETS, load_target_from_json,
            RTX_5090, WorkloadProfile,
        )
        if Path(emulate_npu).exists():
            target = load_target_from_json(emulate_npu)
        elif emulate_npu in PRESET_TARGETS:
            target = PRESET_TARGETS[emulate_npu]
        else:
            console.print(f"[red]Unknown NPU target: {emulate_npu}[/]")
            console.print(f"  Presets: {list(PRESET_TARGETS.keys())}")
            sys.exit(1)

        emulator = NPUEmulator(reference=RTX_5090, target=target)
        console.print(f"  [yellow]NPU Emulation: {target.name}[/]")
        console.print(
            f"  Throttling inference to simulate "
            f"{target.tops_bf16} TOPS / {target.mem_bandwidth_gbs} GB/s\n"
        )

        # Create throttle wrappers with estimated workloads
        # (will be refined after first few frames of measurement)
        yolo_wl = WorkloadProfile(
            stage_name="yolo_detection", model_name="yolo11x",
            param_count=57_000_000, model_size_bytes=int(57e6 * 2),
            gflops_per_inference=196.0, arithmetic_intensity=85.0,
            measured_latency_ms=3.0, measured_gpu=RTX_5090.name,
            peak_activation_bytes=int(0.2e9),
        )
        sam3_wl = WorkloadProfile(
            stage_name="sam3_enrichment", model_name="sam3_full",
            param_count=848_000_000, model_size_bytes=int(848e6 * 2),
            gflops_per_inference=350.0, arithmetic_intensity=120.0,
            measured_latency_ms=30.0, measured_gpu=RTX_5090.name,
            peak_activation_bytes=int(1.0e9),
        )
        npu_throttle_yolo = emulator.create_throttle_wrapper(yolo_wl)
        npu_throttle_sam3 = emulator.create_throttle_wrapper(sam3_wl)

    # --- Probe video ---
    console.print(f"\n[bold]Keyhole[/] — Processing: {video_path.name}")
    meta = probe_video(video_path)
    console.print(
        f"  Source: {meta.width}x{meta.height} @ {meta.fps:.1f} FPS, "
        f"{meta.duration:.1f}s, {meta.total_frames} frames"
    )
    console.print(f"  Extract: {extract_fps} FPS → ~{int(meta.duration * extract_fps)} frames\n")

    # --- Initialize components ---
    store = DetectionStore(settings.database.url)
    video_id = store.register_video(meta)

    detector = YOLODetector(
        model_name=settings.yolo.model,
        confidence=settings.yolo.confidence,
        iou_threshold=settings.yolo.iou_threshold,
        device=settings.yolo.device,
        classes=settings.yolo.classes or None,
        profile=do_profile,
    )

    enricher = None
    if not detect_only and settings.sam3.enabled:
        enricher = SAM3Enricher(
            device=settings.sam3.device,
            concepts=settings.sam3.concepts,
            confidence_threshold=settings.sam3.confidence,
            profile=do_profile,
        )
        enricher.load_model()

    # --- Process frames ---
    total_detections = 0
    total_frames = 0
    t_pipeline_start = time.perf_counter()

    estimated_frames = int(meta.duration * extract_fps)
    if max_f > 0:
        estimated_frames = min(estimated_frames, max_f)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing", total=estimated_frames)

        for frame in extract_frames(video_path, target_fps=extract_fps, max_frames=max_f):
            # Tier 1: YOLO detection
            frame_dets = detector.detect_frame(frame)

            # NPU emulation: throttle YOLO to target latency
            if npu_throttle_yolo:
                npu_throttle_yolo(frame_dets.inference_ms)

            # Tier 2: SAM 3 enrichment (if enabled)
            if enricher and frame_dets.count > 0:
                enriched = enricher.enrich_frame(frame_dets)
                # NPU emulation: throttle SAM 3 to target latency
                if npu_throttle_sam3:
                    npu_throttle_sam3(enriched.total_enrichment_ms)
            else:
                # Pass through without enrichment
                enriched = EnrichedFrame(
                    frame_number=frame_dets.frame_number,
                    timestamp_sec=frame_dets.timestamp_sec,
                    source_video=frame_dets.source_video,
                    detections=[
                        EnrichedDetection(
                            bbox=d.bbox,
                            class_id=d.class_id,
                            class_name=d.class_name,
                            confidence=d.confidence,
                            description=d.class_name,
                        )
                        for d in frame_dets.detections
                    ],
                )

            # Store results
            store.store_enriched_frame(
                video_id=video_id,
                enriched=enriched,
                yolo_inference_ms=frame_dets.inference_ms,
            )

            total_detections += frame_dets.count
            total_frames += 1
            progress.update(task, advance=1)

    pipeline_sec = time.perf_counter() - t_pipeline_start

    # --- Summary ---
    console.print(f"\n[bold green]Processing Complete[/]")
    console.print(f"  Frames processed: {total_frames}")
    console.print(f"  Total detections: {total_detections}")
    console.print(f"  Pipeline time: {pipeline_sec:.1f}s")
    console.print(
        f"  Throughput: {total_frames / pipeline_sec:.1f} frames/sec"
    )

    # --- Profiling report ---
    if do_profile:
        console.print("\n[bold]GPU Profile — YOLO[/]")
        yolo_metrics = detector.get_profile_metrics()
        console.print(f"  Avg inference:  {yolo_metrics.avg_inference_ms:.1f}ms")
        console.print(f"  P95 inference:  {yolo_metrics.p95_inference_ms:.1f}ms")
        console.print(f"  P99 inference:  {yolo_metrics.p99_inference_ms:.1f}ms")
        console.print(f"  Model params:   {yolo_metrics.model_params / 1e6:.1f}M")

        if enricher:
            console.print("\n[bold]GPU Profile — SAM 3[/]")
            sam3_metrics = enricher.get_profile_metrics()
            for k, v in sam3_metrics.items():
                console.print(f"  {k}: {v}")

        # Save profile data
        profile_data = {
            "yolo": {
                "avg_ms": yolo_metrics.avg_inference_ms,
                "p95_ms": yolo_metrics.p95_inference_ms,
                "p99_ms": yolo_metrics.p99_inference_ms,
                "params_m": yolo_metrics.model_params / 1e6,
            },
            "sam3": enricher.get_profile_metrics() if enricher else {},
            "pipeline": {
                "total_frames": total_frames,
                "total_detections": total_detections,
                "total_seconds": pipeline_sec,
                "fps": total_frames / pipeline_sec,
            },
        }
        profile_path = Path(settings.profile.output)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w") as f:
            json.dump(profile_data, f, indent=2)
        console.print(f"\n  Profile saved: {profile_path}")


@cli.command()
@click.option("--q", "query_text", default=None, help="Query string")
def query(query_text: str):
    """Query detections with natural language."""

    from src.store.db import DetectionStore
    from src.query.nlq import NLQueryEngine

    store = DetectionStore(settings.database.url)
    nlq = NLQueryEngine(
        store=store,
        backend=settings.llm.backend,
        api_key=settings.llm.anthropic_api_key,
        ollama_host=settings.llm.ollama_host,
        ollama_model=settings.llm.ollama_model,
        skippy_host=settings.llm.skippy_host,
        skippy_model=settings.llm.skippy_model,
    )

    if query_text:
        # Single query mode
        result = asyncio.run(nlq.query(query_text))
        _display_results(result)
    else:
        # Interactive mode
        console.print("[bold]Keyhole — Interactive Query Mode[/]")
        console.print("Type your question, or 'quit' to exit.\n")

        while True:
            try:
                q = console.input("[blue]> [/]")
                if q.lower() in ("quit", "exit", "q"):
                    break
                if not q.strip():
                    continue

                result = asyncio.run(nlq.query(q))
                _display_results(result)
                console.print()

            except (KeyboardInterrupt, EOFError):
                break

        console.print("\nGoodbye!")


def _display_results(result: dict):
    """Pretty-print query results."""
    if result["result_count"] == 0:
        console.print("[dim]No detections match your query.[/]")
        return

    console.print(f"[green]{result['result_count']} results found[/]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Time", width=8)
    table.add_column("Class", width=12)
    table.add_column("Conf", width=6)
    table.add_column("Description", min_width=30)
    table.add_column("Tags", min_width=20)

    for r in result["results"][:20]:
        timestamp = r.get("timestamp_sec", 0)
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)

        tags = r.get("concept_tags", [])
        tag_str = ", ".join(tags[:5]) if tags else ""

        table.add_row(
            f"{mins}:{secs:02d}",
            r.get("class_name", ""),
            f"{r.get('confidence', 0) * 100:.0f}%",
            r.get("description", ""),
            tag_str,
        )

    console.print(table)


@cli.command()
def serve():
    """Start the web query interface."""
    console.print("[bold]Keyhole[/] — Starting web server...")
    console.print(f"  URL: http://localhost:{settings.api.port}")

    from src.api.server import run_server
    run_server()


@cli.command()
def stats():
    """Show database statistics."""
    from src.store.db import DetectionStore

    store = DetectionStore(settings.database.url)
    data = store.get_stats()

    console.print("\n[bold]Keyhole — Database Stats[/]")
    console.print(f"  Videos:     {data['total_videos']}")
    console.print(f"  Frames:     {data['total_frames']}")
    console.print(f"  Detections: {data['total_events']}")

    if data["class_distribution"]:
        console.print("\n  [bold]Class Distribution:[/]")
        for cls, count in sorted(
            data["class_distribution"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            console.print(f"    {cls}: {count}")


@cli.command(name="emulate")
@click.option("--profile", "profile_path", default="data/output/profile_report.json",
              help="Path to profile_report.json")
@click.option("--target", default="nxp_edge", help="Target preset or JSON path")
@click.option("--compare-all", is_flag=True, help="Compare all preset targets")
def emulate_cmd(profile_path, target, compare_all):
    """Project pipeline performance onto edge NPU hardware."""
    import sys
    sys.argv = [
        "npu_emulator",
        *(["--profile", profile_path] if Path(profile_path).exists() else []),
        "--target", target,
        *(["--compare-all"] if compare_all else []),
    ]
    from src.emulate.npu_emulator import emulate as run_emulate
    run_emulate(standalone_mode=False)


@cli.command(name="layer-profile")
@click.option("--model", type=click.Choice(["yolo", "sam3"]),
              required=True, help="Model to profile")
@click.option("--format", "fmt", type=click.Choice(["json", "csv", "hwsim", "all"]),
              default="all", help="Export format")
@click.option("--output", "-o", default=None, help="Output path (without extension)")
@click.option("--input-size", default=640, type=int, help="Input image size")
def layer_profile_cmd(model, fmt, output, input_size):
    """Export per-layer workload characterization for hardware simulation."""
    import sys
    sys.argv = [
        "layer_profiler",
        "--model", model,
        "--format", fmt,
        *(["--output", output] if output else []),
        "--input-size", str(input_size),
    ]
    from src.emulate.layer_profiler import profile_cli
    profile_cli(standalone_mode=False)


if __name__ == "__main__":
    cli()
