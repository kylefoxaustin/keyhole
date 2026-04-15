"""
Database Models — SQLAlchemy models for detection metadata storage.

Stores structured detection events with full-text search capability
for natural language querying.
"""

import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime,
    ForeignKey, Index, create_engine, JSON, LargeBinary,
)
from sqlalchemy.orm import (
    DeclarativeBase, relationship, Session, sessionmaker,
)


class Base(DeclarativeBase):
    pass


class VideoSource(Base):
    """A processed video file."""
    __tablename__ = "video_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String(512), nullable=False, unique=True)
    width = Column(Integer)
    height = Column(Integer)
    fps = Column(Float)
    duration_sec = Column(Float)
    total_frames = Column(Integer)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)
    # Lifecycle: queued → processing → processed | failed. Written by the
    # API upload handler and the processing watcher; reads in _video_to_dict
    # trust this as the source of truth (no more inferring from frame_count).
    status = Column(String(16), nullable=False, default="queued")

    # Relationships
    frames = relationship("ProcessedFrame", back_populates="video", cascade="all, delete-orphan")


class ProcessedFrame(Base):
    """A single processed frame from a video."""
    __tablename__ = "processed_frames"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, ForeignKey("video_sources.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp_sec = Column(Float, nullable=False)
    detection_count = Column(Integer, default=0)
    yolo_inference_ms = Column(Float, default=0.0)
    sam3_inference_ms = Column(Float, default=0.0)

    # Relationships
    video = relationship("VideoSource", back_populates="frames")
    detections = relationship("DetectionEvent", back_populates="frame", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_frame_timestamp", "video_id", "timestamp_sec"),
    )


class DetectionEvent(Base):
    """
    A single detection event — the core queryable record.

    Combines YOLO detection data with SAM 3 enrichment into a
    single searchable record.
    """
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_id = Column(Integer, ForeignKey("processed_frames.id"), nullable=False)

    # YOLO detection data
    class_name = Column(String(64), nullable=False, index=True)
    class_id = Column(Integer)
    confidence = Column(Float)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)

    # SAM 3 enrichment data
    description = Column(Text, default="")  # NL description for FTS
    concept_tags = Column(JSON, default=list)  # List of matched concept strings
    concept_scores = Column(JSON, default=list)  # Parallel list of scores

    # Computed fields for search
    timestamp_sec = Column(Float, index=True)  # Denormalized for fast queries
    source_video = Column(String(512))  # Denormalized

    # Relationships
    frame = relationship("ProcessedFrame", back_populates="detections")

    __table_args__ = (
        Index("idx_detection_class_time", "class_name", "timestamp_sec"),
        Index("idx_detection_description", "description"),  # For LIKE queries
    )


class DetectionEmbedding(Base):
    """
    Semantic embedding for a detection event.

    One row per DetectionEvent. The embedding is a 384-dim float32 vector
    from sentence-transformers/all-MiniLM-L6-v2, stored as raw bytes.
    Queried in memory via numpy cosine similarity — no vector extension
    required; portable across SQLite/Postgres without config.
    """
    __tablename__ = "detection_embeddings"

    detection_id = Column(
        Integer,
        ForeignKey("detection_events.id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding = Column(LargeBinary, nullable=False)  # numpy float32 bytes
    dim = Column(Integer, nullable=False, default=384)
    model = Column(String(128), nullable=False, default="all-MiniLM-L6-v2")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ProcessingRun(Base):
    """Metadata about a processing run for profiling."""
    __tablename__ = "processing_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_path = Column(String(512))
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime)
    total_frames = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    avg_yolo_ms = Column(Float, default=0.0)
    avg_sam3_ms = Column(Float, default=0.0)
    config_snapshot = Column(JSON, default=dict)  # Settings used for this run
    profile_data = Column(JSON, default=dict)  # GPU profiling metrics


def create_db(database_url: str) -> sessionmaker:
    """Create database engine and tables, return session factory."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
