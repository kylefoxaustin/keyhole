"""
Keyhole API Server — FastAPI endpoints for UI clients.

Implements the HTTP/WebSocket contract documented in API.md at the repo root.
Designed to be consumed by the separate keyhole-UI Next.js frontend over HTTP,
typically exposed via Cloudflare Tunnel from the GPU-backed desktop to a
public URL the Vercel-deployed frontend can reach.
"""

import asyncio
import io
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI, File, Form, HTTPException, Query, Request,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
import uvicorn

from sqlalchemy import func, or_, String, cast

from config.settings import settings
from src.store.db import DetectionStore
from src.store.models import (
    VideoSource, ProcessedFrame, DetectionEvent, ProcessingRun,
    DetectionEmbedding,
)
from src.store.embeddings import EmbeddingEngine

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
embed_engine: Optional[EmbeddingEngine] = None

# Classes known to YOLO — we'll fast-path single-token exact matches to
# literal ILIKE instead of semantic rerank. Refreshed at startup.
_KNOWN_CLASS_NAMES: set[str] = set()

# Background processing state: video_id -> {popen, pipeline, fps, started_at,
# last_frames, stderr_path}. Populated by POST /api/videos, drained by the
# watcher task, observed by _video_to_dict for status and by the WS endpoint
# for broadcasts.
_active_processing: dict[int, dict] = {}
_ws_clients: set[WebSocket] = set()
_watcher_task: Optional[asyncio.Task] = None


@app.on_event("startup")
async def startup():
    global store, embed_engine, _KNOWN_CLASS_NAMES, _watcher_task
    store = DetectionStore(settings.database.url)

    # Load embedding model + warm the in-memory matrix from existing rows
    embed_engine = EmbeddingEngine(device="cpu")
    embed_engine.load()
    with store.get_session() as session:
        rows = session.query(
            DetectionEmbedding.detection_id,
            DetectionEmbedding.embedding,
        ).all()
        embed_engine.populate_cache([(rid, blob) for rid, blob in rows])

        # Cache known class names for the hybrid fast path
        class_rows = session.query(DetectionEvent.class_name).distinct().all()
        _KNOWN_CLASS_NAMES = {c[0].lower() for c in class_rows if c[0]}

    _watcher_task = asyncio.create_task(_processing_watcher())

    logger.info(
        "Keyhole API started at %s:%d — %d embeddings cached, %d known classes",
        settings.api.host, settings.api.port,
        embed_engine.count, len(_KNOWN_CLASS_NAMES),
    )


@app.on_event("shutdown")
async def shutdown():
    if _watcher_task is not None:
        _watcher_task.cancel()
        try:
            await _watcher_task
        except (asyncio.CancelledError, Exception):
            pass


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

def _video_status(video: VideoSource, frame_count: int) -> str:
    """Resolve current status from in-process state + on-disk evidence."""
    info = _active_processing.get(video.id)
    if info is not None:
        rc = info["popen"].poll()
        if rc is None:
            return "processing"
        # Subprocess has exited — watcher will remove the entry shortly.
        # Return terminal status immediately so the current request is accurate.
        return "processed" if rc == 0 else "failed"
    return "processed" if frame_count > 0 else "queued"


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
        "status": _video_status(video, frame_count),
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


# Map the API's pipeline tokens to the CLI flags that `src.main process` expects.
_PIPELINE_ARGS = {
    "hybrid_v2": ["--hybrid-v2", "yolo11s-seg.pt", "--render"],
    "hybrid": ["--hybrid", "--render"],
    "single_pass": ["--single-pass", "--render"],
    "sequential": ["--render"],  # default YOLO + SAM3
}


@app.post("/api/videos")
async def upload_video(
    file: UploadFile = File(...),
    pipeline: str = Form("hybrid_v2"),
    fps: float = Form(5.0),
):
    """
    Accept a video upload, persist it to data/videos/, register it in the DB,
    and spawn a background processing subprocess. Returns the video_id
    immediately so the UI can subscribe to the WS for progress.
    """
    from src.ingest.video import probe_video

    if pipeline not in _PIPELINE_ARGS:
        raise HTTPException(400, f"Unknown pipeline '{pipeline}'. "
                                  f"Valid: {list(_PIPELINE_ARGS)}")

    videos_dir = Path("data/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the client-provided basename; prefix with timestamp on collision
    # so we never silently overwrite existing footage.
    safe_name = Path(file.filename or "upload.mp4").name
    dest = videos_dir / safe_name
    if dest.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = videos_dir / f"{ts}_{safe_name}"

    # Stream-write so we don't buffer large files entirely in memory
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    try:
        meta = probe_video(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to probe video: {exc}") from exc

    video_id = store.register_video(meta)

    log_dir = Path("data/processing")
    log_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = log_dir / f"video_{video_id}.stderr.log"
    stdout_path = log_dir / f"video_{video_id}.stdout.log"

    cmd = [
        sys.executable, "-m", "src.main", "process",
        "--video", str(dest),
        "--fps", str(fps),
        *_PIPELINE_ARGS[pipeline],
    ]
    # Unbuffered so the log files update live rather than in 4KB chunks
    popen = subprocess.Popen(
        cmd,
        stdout=open(stdout_path, "wb"),
        stderr=open(stderr_path, "wb"),
        cwd=str(Path(__file__).resolve().parents[2]),
    )

    estimated_frames = int(meta.duration * fps) if meta.duration else 0
    _active_processing[video_id] = {
        "popen": popen,
        "pipeline": pipeline,
        "fps": fps,
        "started_at": time.time(),
        "last_frames": 0,
        "stderr_path": stderr_path,
        "estimated_frames": estimated_frames,
    }

    await _broadcast({
        "type": "status",
        "video_id": video_id,
        "status": "processing",
        "progress": {
            "frames_done": 0,
            "frames_total": estimated_frames,
            "current_fps": 0.0,
            "eta_seconds": None,
        },
    })

    return {"video_id": video_id, "status": "processing"}


# ============================================================
# Events (Detections)
# ============================================================

def _event_to_dict(
    event: DetectionEvent, session, similarity: Optional[float] = None,
) -> dict:
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
        # Semantic similarity score (0-1) when q was resolved via embeddings.
        # null for literal / no-query results.
        "similarity": similarity,
    }


def _is_literal_query(q: str) -> bool:
    """
    Heuristic: single-token query matching a known YOLO class exactly
    goes through literal ILIKE, not semantic rerank. Users typing
    "person" or "bag" want class-level filtering, not embedding noise.
    """
    q_clean = q.strip().lower()
    if not q_clean or " " in q_clean:
        return False
    return q_clean in _KNOWN_CLASS_NAMES


@app.get("/api/events")
async def query_events(
    q: Optional[str] = Query(None, description="Natural language or text query"),
    video_id: Optional[int] = Query(None, description="Filter to specific video"),
    tags: Optional[str] = Query(None, description="Comma-separated concept tags"),
    class_name: Optional[str] = Query(None, alias="class", description="YOLO class filter"),
    start: Optional[float] = Query(None, description="Start timestamp (seconds)"),
    end: Optional[float] = Query(None, description="End timestamp (seconds)"),
    min_confidence: float = Query(0.0, description="Minimum detection confidence"),
    min_similarity: float = Query(0.2, description="Min cosine similarity for semantic queries"),
    limit: int = Query(50, le=500, description="Max results"),
    offset: int = Query(0, description="Pagination offset"),
):
    """
    Query events with flexible filters.

    Natural-language q resolves via semantic search over detection embeddings
    (cosine similarity, top-K by relevance). Single-token queries matching a
    known YOLO class go through a literal ILIKE fast path. Structured filters
    (tags, class, time range, video_id) compose with either path.
    """
    # Route to semantic or literal path
    use_semantic = bool(q) and not _is_literal_query(q)

    with store.get_session() as session:
        query = session.query(DetectionEvent).join(ProcessedFrame)

        # Structured filters always apply regardless of search strategy
        if video_id is not None:
            query = query.filter(ProcessedFrame.video_id == video_id)

        if class_name:
            query = query.filter(DetectionEvent.class_name.ilike(f"%{class_name}%"))

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

        # === Semantic path ===
        if use_semantic and embed_engine is not None and embed_engine.count > 0:
            # Pull top-K semantic matches, then intersect with structured filters
            hits = embed_engine.search(
                q, top_k=limit + offset + 200, min_similarity=min_similarity,
            )
            if not hits:
                return {
                    "query": q,
                    "search_mode": "semantic",
                    "result_count": 0,
                    "total": 0,
                    "offset": offset,
                    "results": [],
                }

            sim_by_id = {did: score for did, score in hits}
            candidate_ids = list(sim_by_id.keys())

            # Apply structured filters to the candidate set
            filtered = (
                query.filter(DetectionEvent.id.in_(candidate_ids)).all()
            )
            # Reorder by semantic score (DB returned rows in arbitrary order)
            filtered.sort(key=lambda e: -sim_by_id[e.id])

            total = len(filtered)
            page = filtered[offset : offset + limit]
            return {
                "query": q,
                "search_mode": "semantic",
                "result_count": len(page),
                "total": total,
                "offset": offset,
                "results": [
                    _event_to_dict(e, session, similarity=sim_by_id[e.id])
                    for e in page
                ],
            }

        # === Literal / no-query path ===
        if q:  # Literal ILIKE across description, class_name, tags
            pattern = f"%{q}%"
            query = query.filter(or_(
                DetectionEvent.description.ilike(pattern),
                DetectionEvent.class_name.ilike(pattern),
                cast(DetectionEvent.concept_tags, String).ilike(pattern),
            ))

        total = query.count()
        events = (
            query.order_by(DetectionEvent.timestamp_sec.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return {
            "query": q,
            "search_mode": "literal" if q else "recent",
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


@app.get("/api/events/{event_id}/clip")
async def get_event_clip(
    event_id: int,
    before: float = Query(5.0, ge=0, le=60, description="Seconds before event"),
    after: float = Query(5.0, ge=0, le=60, description="Seconds after event"),
    fmt: str = Query("mp4", alias="format", pattern="^(mp4|gif)$"),
):
    """
    Return a short clip (default ±5s) around the event's timestamp.

    Prefers the annotated video when available so masks and boxes are visible.
    Re-encodes on the fly via ffmpeg and streams the bytes back — no tempfiles.
    """
    import subprocess

    with store.get_session() as session:
        event = session.query(DetectionEvent).filter_by(id=event_id).first()
        if not event:
            raise HTTPException(404, f"Event {event_id} not found")

        video = event.frame.video if event.frame else None
        if not video:
            raise HTTPException(404, "Event has no associated video")

        video_stem = Path(video.path).stem
        output_dir = Path(video.path).parent.parent / "output"
        annotated = next(iter(output_dir.glob(f"{video_stem}*annotated*.mp4")), None)
        source = Path(annotated) if annotated else Path(video.path)

        if not source.exists():
            raise HTTPException(404, "Video file not on disk")

        start = max(0.0, event.timestamp_sec - before)
        duration = before + after

        if fmt == "gif":
            # Two-pass palette for decent quality; downscale for size
            cmd = [
                "ffmpeg", "-ss", f"{start:.3f}", "-i", str(source),
                "-t", f"{duration:.3f}",
                "-vf", "fps=12,scale=640:-1:flags=lanczos,"
                       "split[a][b];[a]palettegen[p];[b][p]paletteuse",
                "-f", "gif", "-",
            ]
            media_type = "image/gif"
        else:
            # Fragmented MP4 so it streams cleanly over a pipe
            cmd = [
                "ffmpeg", "-ss", f"{start:.3f}", "-i", str(source),
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "frag_keyframe+empty_moov+faststart",
                "-an",  # strip audio — pipelines don't produce it and it's faster
                "-f", "mp4", "-",
            ]
            media_type = "video/mp4"

        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout:
            logger.warning("ffmpeg clip failed: %s", result.stderr[-500:].decode(errors="replace"))
            raise HTTPException(500, "Clip extraction failed")

        return Response(content=result.stdout, media_type=media_type)


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
# Processing Status (WebSocket + watcher)
# ============================================================

async def _broadcast(message: dict) -> None:
    """Send a JSON message to every connected WS client. Drops dead ones."""
    dead: list[WebSocket] = []
    for ws in list(_ws_clients):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


async def _processing_watcher():
    """
    Poll each active processing subprocess every second:
      - If still running, compute frames_done from the DB and broadcast a
        status update when it changes.
      - On exit, broadcast a 'complete' or 'error' message and drop the entry.
    """
    while True:
        try:
            await asyncio.sleep(1.0)
            if not _active_processing:
                continue

            for video_id, info in list(_active_processing.items()):
                popen = info["popen"]
                rc = popen.poll()

                # Count frames in the DB to estimate progress
                with store.get_session() as session:
                    frames_done = (
                        session.query(func.count(ProcessedFrame.id))
                        .filter(ProcessedFrame.video_id == video_id)
                        .scalar()
                    ) or 0

                if rc is None:
                    # Still running — broadcast a status frame if progress changed
                    if frames_done != info["last_frames"]:
                        elapsed = time.time() - info["started_at"]
                        current_fps = frames_done / elapsed if elapsed > 0 else 0.0
                        eta = None
                        total = info["estimated_frames"] or 0
                        if current_fps > 0 and total > frames_done:
                            eta = int((total - frames_done) / current_fps)
                        await _broadcast({
                            "type": "status",
                            "video_id": video_id,
                            "status": "processing",
                            "progress": {
                                "frames_done": frames_done,
                                "frames_total": total,
                                "current_fps": round(current_fps, 2),
                                "eta_seconds": eta,
                            },
                        })
                        info["last_frames"] = frames_done
                    continue

                # Subprocess has exited — emit terminal message and clean up
                elapsed = time.time() - info["started_at"]
                if rc == 0:
                    with store.get_session() as session:
                        detection_count = (
                            session.query(func.count(DetectionEvent.id))
                            .join(ProcessedFrame)
                            .filter(ProcessedFrame.video_id == video_id)
                            .scalar()
                        ) or 0
                    await _broadcast({
                        "type": "complete",
                        "video_id": video_id,
                        "detection_count": detection_count,
                        "total_time_sec": round(elapsed, 1),
                    })
                else:
                    # Grab the tail of stderr so the UI can show a real reason
                    tail = ""
                    try:
                        tail = Path(info["stderr_path"]).read_text(errors="replace")[-500:]
                    except Exception:
                        pass
                    await _broadcast({
                        "type": "error",
                        "video_id": video_id,
                        "message": f"Processing exited with code {rc}",
                        "details": tail,
                    })

                _active_processing.pop(video_id, None)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("processing watcher loop error")


@app.websocket("/api/ws/processing")
async def ws_processing(websocket: WebSocket):
    """
    Push processing status updates to the UI. On connect, sends a snapshot of
    every active job so a late-joining client doesn't have to wait for the
    next tick to see what's in flight.
    """
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        # Snapshot for late joiners
        for video_id, info in list(_active_processing.items()):
            if info["popen"].poll() is not None:
                continue
            with store.get_session() as session:
                frames_done = (
                    session.query(func.count(ProcessedFrame.id))
                    .filter(ProcessedFrame.video_id == video_id)
                    .scalar()
                ) or 0
            await websocket.send_json({
                "type": "status",
                "video_id": video_id,
                "status": "processing",
                "progress": {
                    "frames_done": frames_done,
                    "frames_total": info["estimated_frames"] or 0,
                    "current_fps": 0.0,
                    "eta_seconds": None,
                },
            })

        # Keep-alive: await client messages (we don't consume them, but this
        # keeps the connection open until the client disconnects).
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("websocket handler error")
    finally:
        _ws_clients.discard(websocket)


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
