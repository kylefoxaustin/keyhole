"""
Backfill Embeddings — one-time migration for existing detection events.

Walks the DetectionEvent table, composes a text blob per row, batch-encodes
with sentence-transformers, and inserts into the DetectionEmbedding table.

Idempotent: rows that already have an embedding are skipped. Re-runnable
after the model changes (pass --force to re-embed everything).

Usage:
    python scripts/backfill_embeddings.py
    python scripts/backfill_embeddings.py --force
    python scripts/backfill_embeddings.py --batch-size 128 --limit 1000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import settings
from src.store.db import DetectionStore
from src.store.models import DetectionEvent, DetectionEmbedding
from src.store.embeddings import EmbeddingEngine, text_for_detection, MODEL_NAME

console = Console()


@click.command()
@click.option("--batch-size", default=64, type=int, help="Encode batch size")
@click.option("--limit", default=0, type=int, help="Max rows to process (0=all)")
@click.option("--force", is_flag=True, help="Re-embed rows that already have embeddings")
@click.option("--device", default="cpu", help="Inference device (cpu or cuda:0)")
def backfill(batch_size, limit, force, device):
    """Embed existing detections for semantic search."""
    console.print("\n[bold]Keyhole — Embedding Backfill[/]\n")

    store = DetectionStore(settings.database.url)
    engine = EmbeddingEngine(device=device)
    engine.load()

    with store.get_session() as session:
        # Find detections needing embeddings
        if force:
            q = session.query(DetectionEvent)
        else:
            existing = select(DetectionEmbedding.detection_id)
            q = session.query(DetectionEvent).filter(
                ~DetectionEvent.id.in_(existing)
            )
        if limit > 0:
            q = q.limit(limit)

        total_needed = q.count()
        console.print(f"  Detections to embed: [cyan]{total_needed}[/]")

        if total_needed == 0:
            console.print("  [green]Nothing to do — all detections already embedded.[/]\n")
            return

        console.print(f"  Model: {MODEL_NAME}")
        console.print(f"  Batch size: {batch_size}")
        console.print(f"  Device: {device}\n")

        # Stream in chunks to cap memory
        processed = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding", total=total_needed)

            # Iterate in pages of batch_size
            offset = 0
            while offset < total_needed:
                # Re-fetch chunk (reusing the filtered query)
                if force:
                    chunk_q = session.query(DetectionEvent).order_by(DetectionEvent.id)
                else:
                    existing = select(DetectionEmbedding.detection_id)
                    chunk_q = (
                        session.query(DetectionEvent)
                        .filter(~DetectionEvent.id.in_(existing))
                        .order_by(DetectionEvent.id)
                    )
                chunk_q = chunk_q.limit(batch_size)
                rows = chunk_q.all()
                if not rows:
                    break

                texts = [
                    text_for_detection(
                        class_name=r.class_name,
                        description=r.description or "",
                        concept_tags=r.concept_tags or [],
                    )
                    for r in rows
                ]
                vecs = engine.encode(texts, batch_size=batch_size)

                # Upsert: delete any existing row then insert
                if force:
                    ids = [r.id for r in rows]
                    session.query(DetectionEmbedding).filter(
                        DetectionEmbedding.detection_id.in_(ids)
                    ).delete(synchronize_session=False)

                for row, vec in zip(rows, vecs):
                    session.add(DetectionEmbedding(
                        detection_id=row.id,
                        embedding=vec.tobytes(),
                        dim=vec.shape[0],
                        model="all-MiniLM-L6-v2",
                    ))
                session.commit()

                processed += len(rows)
                offset += batch_size
                progress.update(task, completed=processed)

        console.print(f"\n  [bold green]Done.[/] Embedded {processed} detections.\n")


if __name__ == "__main__":
    backfill()
