# Keyhole API

HTTP/WebSocket API exposed by the Keyhole backend for UI clients (primarily
[keyhole-UI](https://github.com/kylefoxaustin/keyhole-UI)) but usable by anyone.

Backend: `src/api/server.py` (FastAPI). Default host: `http://localhost:8000`

## Authentication

**MVP**: None. Assumes self-hosted, trusted network.
**Production**: Bearer token via `Authorization: Bearer <token>` header. Token issued
by external auth service (not part of Keyhole).

## Endpoints

### Videos

#### `GET /api/videos`
List all videos in the library.

**Response:**
```json
{
  "videos": [
    {
      "id": 1,
      "name": "embedded_world_clip.mp4",
      "path": "data/videos/embedded_world_clip.mp4",
      "width": 3840,
      "height": 2160,
      "source_fps": 30.0,
      "duration_sec": 13.5,
      "total_frames": 405,
      "registered_at": "2026-04-10T17:42:45Z",
      "status": "processed",
      "pipeline": "hybrid_v2",
      "detection_count": 307,
      "thumbnail_url": "/api/videos/1/thumbnail"
    }
  ]
}
```

**Status values:** `queued`, `processing`, `processed`, `failed`

#### `POST /api/videos`
Upload a new video. Returns the video ID and starts processing.

**Request:** `multipart/form-data` with:
- `file`: the video file
- `pipeline`: `single_pass` | `hybrid` | `hybrid_v2` (default: `hybrid_v2`)
- `fps`: frame extraction rate (default: 5)

**Response:**
```json
{ "video_id": 2, "status": "queued" }
```

#### `GET /api/videos/{id}`
Video metadata + current processing status.

**Response:** Same shape as list entries.

#### `GET /api/videos/{id}/thumbnail`
Returns a JPEG thumbnail (first annotated frame).

#### `GET /api/videos/{id}/stream`
Returns the original video file. Supports HTTP range requests.

#### `GET /api/videos/{id}/annotated`
Returns the annotated video (masks + boxes rendered). Supports HTTP range requests.

#### `DELETE /api/videos/{id}`
Remove a video and all its associated events from the store.

### Events (Detections)

#### `GET /api/events`
Query events. Supports natural language queries and structured filters.

**Query parameters:**
- `q`: natural language query (e.g., `"person wearing red backpack"`)
- `video_id`: filter to a specific video
- `tags`: comma-separated concept tags (e.g., `"person,backpack"`)
- `class`: YOLO class name filter (e.g., `"person"`)
- `start`: ISO timestamp filter
- `end`: ISO timestamp filter
- `limit`: max results (default: 50, max: 500)
- `offset`: pagination offset

**Response:**
```json
{
  "query": "person wearing red backpack",
  "result_count": 12,
  "results": [
    {
      "id": 42,
      "video_id": 1,
      "video_name": "embedded_world_clip.mp4",
      "frame_number": 87,
      "timestamp_sec": 2.9,
      "wall_time": "2026-04-14T14:32:15Z",
      "class_name": "person",
      "confidence": 0.94,
      "bbox": [320, 180, 450, 640],
      "description": "person: hat, backpack, red jacket",
      "concept_tags": ["hat", "backpack", "red jacket"],
      "concepts": [
        {"concept": "person wearing hat", "confidence": 0.91},
        {"concept": "person wearing backpack", "confidence": 0.89},
        {"concept": "red jacket", "confidence": 0.76}
      ],
      "thumbnail_url": "/api/events/42/frame",
      "mask_area_pct": 8.3
    }
  ]
}
```

#### `GET /api/events/{id}`
Single event with full details.

#### `GET /api/events/{id}/frame`
Annotated frame image (JPEG) showing just the bounding box + mask for this event.

#### `GET /api/events/{id}/clip`
Short video clip around the event (default: ±5 seconds).

**Query parameters:**
- `before`: seconds before event (default: 5)
- `after`: seconds after event (default: 5)
- `format`: `mp4` | `gif` (default: `mp4`)

### Concepts (Autocomplete)

#### `GET /api/concepts`
Returns the concept vocabulary available across the user's corpus.

**Response:**
```json
{
  "concepts": [
    {"name": "person wearing hat", "event_count": 47},
    {"name": "red jacket", "event_count": 12},
    {"name": "delivery truck", "event_count": 8}
  ]
}
```

Used for query autocomplete and filter suggestions.

#### `GET /api/classes`
Returns YOLO class names present in the corpus with counts.

**Response:**
```json
{
  "classes": [
    {"name": "person", "count": 1203},
    {"name": "car", "count": 87}
  ]
}
```

### Processing Status (WebSocket)

#### `WS /api/ws/processing`
Live processing status updates. Pushed whenever a video's status changes or processing
progresses.

**Messages from server:**
```json
{
  "type": "status",
  "video_id": 2,
  "status": "processing",
  "progress": {
    "frames_done": 430,
    "frames_total": 1215,
    "current_fps": 7.2,
    "eta_seconds": 109
  }
}
```

```json
{
  "type": "complete",
  "video_id": 2,
  "detection_count": 1847,
  "total_time_sec": 152
}
```

```json
{
  "type": "error",
  "video_id": 2,
  "message": "GPU out of memory"
}
```

### System

#### `GET /api/health`
Health check + backend info.

**Response:**
```json
{
  "status": "ok",
  "version": "0.2.0",
  "gpu": {
    "available": true,
    "device": "NVIDIA GeForce RTX 5090",
    "memory_free_gb": 28.4,
    "memory_total_gb": 32.0
  },
  "pipelines_available": ["single_pass", "hybrid", "hybrid_v2"]
}
```

## Error Format

All errors return JSON with this shape:

```json
{
  "error": {
    "code": "video_not_found",
    "message": "No video with id 999",
    "details": {}
  }
}
```

**HTTP status codes used:**
- `200` success
- `400` invalid request
- `404` resource not found
- `409` conflict (e.g., re-upload same video)
- `500` server error
- `503` GPU unavailable / model not loaded

## OpenAPI Spec

Once the endpoints are implemented, FastAPI auto-generates an OpenAPI schema at
`/api/docs` (Swagger UI) and `/api/openapi.json` (raw spec).

The UI repo can auto-generate TypeScript types from this spec using
[openapi-typescript](https://github.com/drwpow/openapi-typescript):

```bash
npx openapi-typescript http://localhost:8000/api/openapi.json \
  --output src/lib/api-types.ts
```

## Implementation Status

**Not yet implemented.** `src/api/server.py` currently contains only a minimal FastAPI
stub. Endpoints need to be built out to match this contract.

**Priority order for implementation:**
1. `GET /api/videos` + `GET /api/videos/{id}` (library listing)
2. `GET /api/events` with query support (the core experience)
3. `GET /api/events/{id}/frame` (result card thumbnails)
4. `GET /api/videos/{id}/annotated` (video playback)
5. `GET /api/concepts` (autocomplete)
6. `WS /api/ws/processing` (live status)
7. Everything else (upload, delete, clip export)

## Sister Repo

UI implementation: https://github.com/kylefoxaustin/keyhole-UI
