"""
Database Operations — Store and query detection events.

Provides high-level functions for writing enriched detections to the
database and querying them with structured or text-based filters.
"""

import logging
from typing import Optional
from pathlib import Path

from sqlalchemy.orm import Session

from src.store.models import (
    VideoSource, ProcessedFrame, DetectionEvent, DetectionEmbedding,
    ProcessingRun, create_db,
)
from src.ingest.video import VideoMeta
from src.enrich.sam3 import EnrichedFrame, EnrichedDetection

logger = logging.getLogger(__name__)

# Lazy-loaded embedding engine shared across ingest calls.
# Loading the model is expensive (~1s), so we cache it at module level.
_embed_engine = None


def _get_embed_engine():
    global _embed_engine
    if _embed_engine is None:
        try:
            from src.store.embeddings import EmbeddingEngine
            _embed_engine = EmbeddingEngine(device="cpu")
            _embed_engine.load()
        except Exception as e:
            logger.warning("Embedding engine unavailable, skipping ingest embeddings: %s", e)
            _embed_engine = False  # Sentinel — don't retry
    return _embed_engine if _embed_engine else None


class DetectionStore:
    """
    Storage layer for detection events.

    Handles writing enriched detections and querying them
    for the NLQ engine.
    """

    def __init__(self, database_url: str):
        self.SessionFactory = create_db(database_url)
        self._migrate_video_status()
        logger.info("Database initialized: %s", database_url)

    def _migrate_video_status(self) -> None:
        """
        Idempotent migration: add video_sources.status if missing, then
        backfill existing rows (processed if they have frames, failed if
        they don't — any pre-existing row without frames is a failed run
        from before the status column existed).
        """
        from sqlalchemy import text
        with self.SessionFactory() as session:
            cols = session.execute(text("PRAGMA table_info(video_sources)")).all()
            names = {row[1] for row in cols}
            if "status" in names:
                return
            session.execute(text(
                "ALTER TABLE video_sources ADD COLUMN status VARCHAR(16) "
                "NOT NULL DEFAULT 'queued'"
            ))
            session.execute(text("""
                UPDATE video_sources SET status = CASE
                    WHEN id IN (SELECT DISTINCT video_id FROM processed_frames)
                        THEN 'processed'
                    ELSE 'failed'
                END
            """))
            session.commit()
            logger.info("Migrated video_sources: added status column + backfilled")

    def set_video_status(self, video_id: int, status: str) -> None:
        """Persist a lifecycle transition for a video."""
        with self.get_session() as session:
            video = session.query(VideoSource).filter_by(id=video_id).first()
            if video is None:
                return
            video.status = status
            session.commit()

    def get_session(self) -> Session:
        return self.SessionFactory()

    def register_video(self, meta: VideoMeta) -> int:
        """Register a video source, return its ID."""
        with self.get_session() as session:
            # Check if already registered
            existing = session.query(VideoSource).filter_by(path=meta.path).first()
            if existing:
                return existing.id

            video = VideoSource(
                path=meta.path,
                width=meta.width,
                height=meta.height,
                fps=meta.fps,
                duration_sec=meta.duration,
                total_frames=meta.total_frames,
            )
            session.add(video)
            session.commit()
            logger.info("Registered video source: %s (ID: %d)", meta.path, video.id)
            return video.id

    def store_enriched_frame(
        self,
        video_id: int,
        enriched: EnrichedFrame,
        yolo_inference_ms: float = 0.0,
    ):
        """
        Store all enriched detections from a single frame, and compute
        semantic embeddings for each (so semantic search just works on
        freshly ingested data without a separate backfill).
        """
        with self.get_session() as session:
            frame = ProcessedFrame(
                video_id=video_id,
                frame_number=enriched.frame_number,
                timestamp_sec=enriched.timestamp_sec,
                detection_count=len(enriched.detections),
                yolo_inference_ms=yolo_inference_ms,
                sam3_inference_ms=enriched.total_enrichment_ms,
            )
            session.add(frame)
            session.flush()  # Get frame.id

            events_added = []
            for det in enriched.detections:
                event = DetectionEvent(
                    frame_id=frame.id,
                    class_name=det.class_name,
                    class_id=det.class_id,
                    confidence=det.confidence,
                    bbox_x1=det.bbox[0],
                    bbox_y1=det.bbox[1],
                    bbox_x2=det.bbox[2],
                    bbox_y2=det.bbox[3],
                    description=det.description,
                    concept_tags=det.concept_tags,
                    concept_scores=[c.confidence for c in det.concepts],
                    timestamp_sec=enriched.timestamp_sec,
                    source_video=enriched.source_video,
                )
                session.add(event)
                events_added.append((event, det))

            session.flush()  # Get event IDs

            # Embed the batch in one model call (fast) then persist
            engine = _get_embed_engine()
            if engine is not None and events_added:
                from src.store.embeddings import text_for_detection
                texts = [
                    text_for_detection(
                        class_name=det.class_name,
                        description=det.description or "",
                        concept_tags=det.concept_tags or [],
                    )
                    for _, det in events_added
                ]
                vecs = engine.encode(texts)
                for (event, _), vec in zip(events_added, vecs):
                    session.add(DetectionEmbedding(
                        detection_id=event.id,
                        embedding=vec.tobytes(),
                        dim=vec.shape[0],
                        model="all-MiniLM-L6-v2",
                    ))

            session.commit()

    def search_detections(
        self,
        query_text: Optional[str] = None,
        class_name: Optional[str] = None,
        concept_tag: Optional[str] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        video_path: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """
        Search detections with flexible filters.

        This is the structured query layer that the NLQ engine
        translates natural language into.
        """
        with self.get_session() as session:
            query = session.query(DetectionEvent)

            if class_name:
                query = query.filter(
                    DetectionEvent.class_name.ilike(f"%{class_name}%")
                )

            if query_text:
                # Full-text search on description field
                query = query.filter(
                    DetectionEvent.description.ilike(f"%{query_text}%")
                )

            if concept_tag:
                # Search within JSON concept_tags array
                # SQLite JSON support via LIKE on serialized field
                query = query.filter(
                    DetectionEvent.concept_tags.cast(str).ilike(
                        f"%{concept_tag}%"
                    )
                )

            if time_start is not None:
                query = query.filter(DetectionEvent.timestamp_sec >= time_start)

            if time_end is not None:
                query = query.filter(DetectionEvent.timestamp_sec <= time_end)

            if video_path:
                query = query.filter(
                    DetectionEvent.source_video.ilike(f"%{video_path}%")
                )

            if min_confidence > 0:
                query = query.filter(DetectionEvent.confidence >= min_confidence)

            query = query.order_by(DetectionEvent.timestamp_sec.desc())
            query = query.limit(limit)

            results = []
            for event in query.all():
                results.append({
                    "id": event.id,
                    "class_name": event.class_name,
                    "confidence": event.confidence,
                    "description": event.description,
                    "concept_tags": event.concept_tags,
                    "timestamp_sec": event.timestamp_sec,
                    "source_video": event.source_video,
                    "bbox": [
                        event.bbox_x1, event.bbox_y1,
                        event.bbox_x2, event.bbox_y2,
                    ],
                })

            return results

    def get_stats(self) -> dict:
        """Get summary statistics from the database."""
        with self.get_session() as session:
            total_events = session.query(DetectionEvent).count()
            total_frames = session.query(ProcessedFrame).count()
            total_videos = session.query(VideoSource).count()

            # Class distribution
            from sqlalchemy import func
            class_counts = (
                session.query(
                    DetectionEvent.class_name,
                    func.count(DetectionEvent.id),
                )
                .group_by(DetectionEvent.class_name)
                .all()
            )

            return {
                "total_events": total_events,
                "total_frames": total_frames,
                "total_videos": total_videos,
                "class_distribution": {
                    name: count for name, count in class_counts
                },
            }
