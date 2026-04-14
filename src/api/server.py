"""
Keyhole API Server — FastAPI endpoints for UI clients.

Implements the HTTP/WebSocket contract documented in API.md at the repo root.
Designed to be consumed by the separate keyhole-UI Next.js frontend over HTTP,
typically exposed via Cloudflare Tunnel from the GPU-backed desktop to a
public URL the Vercel-deployed frontend can reach.
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
import uvicorn

from sqlalchemy import func, or_, String, cast

from config.settings import settings
from src.store.db import DetectionStore
from src.store.models import (
    VideoSource, ProcessedFrame, DetectionEvent, ProcessingRun,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Keyhole API",
    description="Edge AI Video Intelligence — backend for keyhole-UI",
    version="0.2.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# Allow the UI frontend (Next.js dev server or Vercel deployment) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store: Optional[DetectionStore] = None


@app.on_event("startup")
async def startup():
    global store
    store = DetectionStore(settings.database.url)
    logger.info("Keyhole API started at %s:%d", settings.api.host, settings.api.port)


# ============================================================
# System
# ============================================================

@app.get("/api/health")
async def health():
    """Health check + backend info. Used by UI to verify connectivity."""
    import torch
    gpu_info = {"available": False}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        gpu_info = {
            "available": True,
            "device": props.name,
            "memory_free_gb": round(free / 1e9, 2),
            "memory_total_gb": round(total / 1e9, 2),
        }

    return {
        "status": "ok",
        "version": "0.2.0",
        "gpu": gpu_info,
        "pipelines_available": ["sequential", "single_pass", "hybrid", "hybrid_v2"],
    }


# ============================================================
# Videos
# ============================================================

def _video_to_dict(video: VideoSource, session) -> dict:
    """Serialize a VideoSource row + computed fields."""
    detection_count = (
        session.query(func.count(DetectionEvent.id))
        .join(ProcessedFrame)
        .filter(ProcessedFrame.video_id == video.id)
        .scalar()
    ) or 0

    frame_count = (
        session.query(func.count(ProcessedFrame.id))
        .filter(ProcessedFrame.video_id == video.id)
        .scalar()
    ) or 0

    # Check if an annotated version exists on disk
    video_stem = Path(video.path).stem
    output_dir = Path(video.path).parent.parent / "output"
    annotated_candidates = list(output_dir.glob(f"{video_stem}*annotated*.mp4"))
    annotated_available = len(annotated_candidates) > 0

    return {
        "id": video.id,
        "name": Path(video.path).name,
        "path": video.path,
        "width": video.width,
        "height": video.height,
        "source_fps": video.fps,
        "duration_sec": video.duration_sec,
        "total_frames": video.total_frames,
        "registered_at": video.processed_at.isoformat() if video.processed_at else None,
        "status": "processed" if frame_count > 0 else "queued",
        "detection_count": detection_count,
        "frame_count": frame_count,
        "thumbnail_url": f"/api/videos/{video.id}/thumbnail",
        "annotated_available": annotated_available,
    }


@app.get("/api/videos")
async def list_videos():
    """List all videos in the library."""
    with store.get_session() as session:
        videos = session.query(VideoSource).order_by(
            VideoSource.processed_at.desc()
        ).all()
        return {"videos": [_video_to_dict(v, session) for v in videos]}


@app.get("/api/videos/{video_id}")
async def get_video(video_id: int):
    """Video metadata + current processing status."""
    with store.get_session() as session:
        video = session.query(VideoSource).filter_by(id=video_id).first()
        if not video:
            raise HTTPException(404, f"Video {video_id} not found")
        return _video_to_dict(video, session)


@app.get("/api/videos/{video_id}/thumbnail")
async def get_video_thumbnail(video_id: int):
    """
    Returns a thumbnail image for the video.

    Uses the first annotated frame if available, otherwise extracts
    the first frame from the source video with FFmpeg.
    """
    with store.get_session() as session:
        video = session.query(VideoSource).filter_by(id=video_id).first()
        if not video:
            raise HTTPException(404, f"Video {video_id} not found")

        # Try to find an annotated output to grab a thumbnail from
        video_stem = Path(video.path).stem
        output_dir = Path(video.path).parent.parent / "output"
        annotated = next(iter(output_dir.glob(f"{video_stem}*annotated*.mp4")), None)
        source = Path(annotated) if annotated else Path(video.path)

        if not source.exists():
            raise HTTPException(404, "Video file not on disk")

        # Extract frame with FFmpeg into memory
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-i", str(source), "-ss", "00:00:00.5",
             "-vframes", "1", "-f", "image2pipe",
             "-vcodec", "mjpeg", "-"],
            capture_output=True, check=False,
        )
        if result.returncode != 0 or not result.stdout:
            raise HTTPException(500, "Thumbnail extraction failed")

        return Response(content=result.stdout, media_type="image/jpeg")


@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: int):
    """Serve the original video file."""
    with store.get_session() as session:
        video = session.query(VideoSource).filter_by(id=video_id).first()
        if not video:
            raise HTTPException(404, f"Video {video_id} not found")
        path = Path(video.path)
        if not path.exists():
            raise HTTPException(404, "Video file not on disk")
        return FileResponse(str(path), media_type="video/mp4")


@app.get("/api/videos/{video_id}/annotated")
async def stream_annotated(video_id: int):
    """Serve the annotated (masks + boxes rendered) video file."""
    with store.get_session() as session:
        video = session.query(VideoSource).filter_by(id=video_id).first()
        if not video:
            raise HTTPException(404, f"Video {video_id} not found")

        video_stem = Path(video.path).stem
        output_dir = Path(video.path).parent.parent / "output"
        annotated = next(iter(output_dir.glob(f"{video_stem}*annotated*.mp4")), None)
        if not annotated:
            raise HTTPException(404, "No annotated version available for this video")
        return FileResponse(str(annotated), media_type="video/mp4")


@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: int):
    """Remove a video and all its associated events."""
    with store.get_session() as session:
        video = session.query(VideoSource).filter_by(id=video_id).first()
        if not video:
            raise HTTPException(404, f"Video {video_id} not found")
        session.delete(video)
        session.commit()
        return {"deleted": video_id}


# ============================================================
# Events (Detections)
# ============================================================

def _event_to_dict(event: DetectionEvent, session) -> dict:
    """Serialize a DetectionEvent row with joined frame/video info."""
    frame = event.frame
    video = frame.video if frame else None

    # Reconstruct concepts list from parallel tags/scores arrays
    concepts = []
    tags = event.concept_tags or []
    scores = event.concept_scores or []
    for i, tag in enumerate(tags):
        score = scores[i] if i < len(scores) else 0.0
        concepts.append({"concept": tag, "confidence": float(score)})

    return {
        "id": event.id,
        "video_id": video.id if video else None,
        "video_name": Path(video.path).name if video else None,
        "frame_number": frame.frame_number if frame else None,
        "timestamp_sec": event.timestamp_sec,
        "wall_time": None,  # Not yet tracked per-event
        "class_name": event.class_name,
        "class_id": event.class_id,
        "confidence": event.confidence,
        "bbox": [event.bbox_x1, event.bbox_y1, event.bbox_x2, event.bbox_y2],
        "description": event.description,
        "concept_tags": tags,
        "concepts": concepts,
        "thumbnail_url": f"/api/events/{event.id}/frame",
    }


@app.get("/api/events")
async def query_events(
    q: Optional[str] = Query(None, description="Natural language or text query"),
    video_id: Optional[int] = Query(None, description="Filter to specific video"),
    tags: Optional[str] = Query(None, description="Comma-separated concept tags"),
    class_name: Optional[str] = Query(None, alias="class", description="YOLO class filter"),
    start: Optional[float] = Query(None, description="Start timestamp (seconds)"),
    end: Optional[float] = Query(None, description="End timestamp (seconds)"),
    min_confidence: float = Query(0.0, description="Minimum detection confidence"),
    limit: int = Query(50, le=500, description="Max results"),
    offset: int = Query(0, description="Pagination offset"),
):
    """
    Query events with flexible filters.

    Supports natural language (`q`), structured filters (tags, class, time range),
    and pagination.
    """
    with store.get_session() as session:
        query = session.query(DetectionEvent).join(ProcessedFrame)

        if video_id is not None:
            query = query.filter(ProcessedFrame.video_id == video_id)

        if class_name:
            query = query.filter(DetectionEvent.class_name.ilike(f"%{class_name}%"))

        if q:
            # Text search across description, class name, and concept tags.
            # SQLite stores JSON as text, so cast via SQLAlchemy String type
            # to get a LIKE-able representation.
            pattern = f"%{q}%"
            query = query.filter(or_(
                DetectionEvent.description.ilike(pattern),
                DetectionEvent.class_name.ilike(pattern),
                cast(DetectionEvent.concept_tags, String).ilike(pattern),
            ))

        if tags:
            for tag in tags.split(","):
                tag = tag.strip()
                if tag:
                    query = query.filter(
                        cast(DetectionEvent.concept_tags, String).ilike(f"%{tag}%")
                    )

        if start is not None:
            query = query.filter(DetectionEvent.timestamp_sec >= start)
        if end is not None:
            query = query.filter(DetectionEvent.timestamp_sec <= end)

        if min_confidence > 0:
            query = query.filter(DetectionEvent.confidence >= min_confidence)

        total = query.count()
        events = (
            query.order_by(DetectionEvent.timestamp_sec.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return {
            "query": q,
            "result_count": len(events),
            "total": total,
            "offset": offset,
            "results": [_event_to_dict(e, session) for e in events],
        }


@app.get("/api/events/{event_id}")
async def get_event(event_id: int):
    """Single event with full details."""
    with store.get_session() as session:
        event = session.query(DetectionEvent).filter_by(id=event_id).first()
        if not event:
            raise HTTPException(404, f"Event {event_id} not found")
        return _event_to_dict(event, session)


@app.get("/api/events/{event_id}/frame")
async def get_event_frame(event_id: int):
    """
    Return an annotated frame image showing the bounding box for this event.

    Extracts the specific frame from the annotated video if available,
    otherwise from the source video, then draws the event's bbox on it.
    """
    import subprocess
    import cv2
    import numpy as np

    with store.get_session() as session:
        event = session.query(DetectionEvent).filter_by(id=event_id).first()
        if not event:
            raise HTTPException(404, f"Event {event_id} not found")

        frame = event.frame
        video = frame.video if frame else None
        if not video:
            raise HTTPException(404, "Event has no associated video")

        # Prefer annotated video for nicer output
        video_stem = Path(video.path).stem
        output_dir = Path(video.path).parent.parent / "output"
        annotated = next(iter(output_dir.glob(f"{video_stem}*annotated*.mp4")), None)
        source = Path(annotated) if annotated else Path(video.path)

        if not source.exists():
            raise HTTPException(404, "Video file not on disk")

        # Seek to the event's timestamp and grab one frame as JPEG
        result = subprocess.run(
            ["ffmpeg", "-ss", f"{event.timestamp_sec:.3f}",
             "-i", str(source), "-vframes", "1",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True, check=False,
        )
        if result.returncode != 0 or not result.stdout:
            raise HTTPException(500, "Frame extraction failed")

        img_bytes = result.stdout

        # If we used the source (un-annotated), draw the bbox ourselves
        if not annotated:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            x1, y1, x2, y2 = map(int, [
                event.bbox_x1, event.bbox_y1, event.bbox_x2, event.bbox_y2,
            ])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 212, 255), 3)
            label = f"{event.class_name} {event.confidence:.0%}"
            cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
            _, encoded = cv2.imencode(".jpg", img)
            img_bytes = encoded.tobytes()

        return Response(content=img_bytes, media_type="image/jpeg")


# ============================================================
# Concepts / Autocomplete
# ============================================================

@app.get("/api/concepts")
async def list_concepts():
    """
    Return the concept vocabulary present in the user's corpus.

    Powers autocomplete and filter pill UI. Aggregates across all
    detection events' concept_tags JSON arrays.
    """
    with store.get_session() as session:
        # Fetch all concept_tags arrays and count occurrences in Python
        # (SQLite JSON functions vary; this is portable and fast enough for <10k events)
        events = session.query(DetectionEvent.concept_tags).all()
        counts: dict[str, int] = {}
        for (tags,) in events:
            if not tags:
                continue
            for tag in tags:
                counts[tag] = counts.get(tag, 0) + 1

        concepts = [
            {"name": name, "event_count": count}
            for name, count in sorted(counts.items(), key=lambda x: -x[1])
        ]
        return {"concepts": concepts}


@app.get("/api/classes")
async def list_classes():
    """Return YOLO class names present in the corpus with counts."""
    with store.get_session() as session:
        rows = (
            session.query(
                DetectionEvent.class_name, func.count(DetectionEvent.id)
            )
            .group_by(DetectionEvent.class_name)
            .order_by(func.count(DetectionEvent.id).desc())
            .all()
        )
        return {
            "classes": [{"name": name, "count": count} for name, count in rows],
        }


# ============================================================
# Stats (legacy — keep for existing CLI)
# ============================================================

@app.get("/api/stats")
async def get_stats():
    """Database statistics."""
    return store.get_stats()


# ============================================================
# Runner
# ============================================================

def run_server():
    """Run the API server."""
    uvicorn.run(
        "src.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
