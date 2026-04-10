"""
API Server — FastAPI endpoints + embedded web UI.

Provides REST endpoints for querying detections and a simple
web interface for natural language search.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from config.settings import settings
from src.store.db import DetectionStore
from src.query.nlq import NLQueryEngine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Keyhole",
    description="Edge AI Video Intelligence — Natural Language Query Interface",
    version="0.1.0",
)

# Initialize store and query engine at startup
store: DetectionStore = None
nlq: NLQueryEngine = None


@app.on_event("startup")
async def startup():
    global store, nlq
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


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the embedded web UI."""
    return QUERY_UI_HTML


@app.post("/api/query")
async def query_detections(request: Request):
    """Execute a natural language query."""
    body = await request.json()
    user_query = body.get("query", "")

    if not user_query.strip():
        return JSONResponse(
            {"error": "Empty query"}, status_code=400
        )

    result = await nlq.query(user_query)
    return result


@app.get("/api/stats")
async def get_stats():
    """Get database statistics."""
    return store.get_stats()


@app.get("/api/detections")
async def list_detections(
    class_name: str = None,
    limit: int = 50,
    min_confidence: float = 0.0,
):
    """List detections with optional filters."""
    return store.search_detections(
        class_name=class_name,
        min_confidence=min_confidence,
        limit=limit,
    )


# --- Embedded Web UI ---
QUERY_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keyhole — Query</title>
    <style>
        :root {
            --bg: #0f1117;
            --surface: #1a1d27;
            --border: #2a2d3a;
            --text: #e4e4e7;
            --dim: #71717a;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --green: #22c55e;
            --orange: #f59e0b;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }
        h1 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: var(--dim);
            font-size: 0.875rem;
            margin-bottom: 2rem;
        }
        .search-box {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }
        .search-box input {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--surface);
            color: var(--text);
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }
        .search-box input:focus {
            border-color: var(--accent);
        }
        .search-box input::placeholder {
            color: var(--dim);
        }
        .search-box button {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            background: var(--accent);
            color: white;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        .search-box button:hover {
            background: var(--accent-hover);
        }
        .search-box button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .stats {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }
        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            min-width: 120px;
        }
        .stat-card .label { color: var(--dim); font-size: 0.75rem; }
        .stat-card .value { font-size: 1.25rem; font-weight: 600; }
        .results-header {
            color: var(--dim);
            font-size: 0.875rem;
            margin-bottom: 0.75rem;
        }
        .result-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .result-card .class-badge {
            display: inline-block;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            background: var(--accent);
            color: white;
            margin-right: 0.5rem;
        }
        .result-card .confidence {
            color: var(--green);
            font-size: 0.75rem;
        }
        .result-card .description {
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }
        .result-card .tags {
            margin-top: 0.5rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.25rem;
        }
        .result-card .tag {
            display: inline-block;
            padding: 0.125rem 0.375rem;
            border-radius: 3px;
            font-size: 0.7rem;
            background: #2a2d3a;
            color: var(--orange);
        }
        .result-card .timestamp {
            color: var(--dim);
            font-size: 0.75rem;
            margin-top: 0.5rem;
        }
        .loading { text-align: center; color: var(--dim); padding: 2rem; }
        .empty { text-align: center; color: var(--dim); padding: 2rem; }
        .examples {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }
        .example-btn {
            padding: 0.375rem 0.75rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--surface);
            color: var(--dim);
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .example-btn:hover {
            border-color: var(--accent);
            color: var(--text);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Keyhole</h1>
        <p class="subtitle">Natural Language Video Intelligence Query</p>

        <div class="stats" id="stats"></div>

        <div class="search-box">
            <input type="text" id="query" placeholder="Find me any person wearing a red hat..."
                   onkeydown="if(event.key==='Enter') search()">
            <button onclick="search()" id="searchBtn">Search</button>
        </div>

        <div class="examples">
            <button class="example-btn" onclick="setQuery('show all people')">All people</button>
            <button class="example-btn" onclick="setQuery('person with a backpack')">With backpack</button>
            <button class="example-btn" onclick="setQuery('any vehicles detected')">Vehicles</button>
            <button class="example-btn" onclick="setQuery('person wearing a hat')">Wearing hat</button>
            <button class="example-btn" onclick="setQuery('dogs or cats')">Animals</button>
        </div>

        <div id="results"></div>
    </div>

    <script>
        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('stats').innerHTML = `
                    <div class="stat-card">
                        <div class="label">Total Events</div>
                        <div class="value">${data.total_events.toLocaleString()}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Frames</div>
                        <div class="value">${data.total_frames.toLocaleString()}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Videos</div>
                        <div class="value">${data.total_videos}</div>
                    </div>
                `;
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }

        function setQuery(q) {
            document.getElementById('query').value = q;
            search();
        }

        async function search() {
            const query = document.getElementById('query').value.trim();
            if (!query) return;

            const btn = document.getElementById('searchBtn');
            const resultsDiv = document.getElementById('results');

            btn.disabled = true;
            btn.textContent = '...';
            resultsDiv.innerHTML = '<div class="loading">Searching...</div>';

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query }),
                });
                const data = await res.json();

                if (data.results && data.results.length > 0) {
                    resultsDiv.innerHTML = `
                        <div class="results-header">${data.result_count} results found</div>
                        ${data.results.map(r => `
                            <div class="result-card">
                                <span class="class-badge">${r.class_name}</span>
                                <span class="confidence">${(r.confidence * 100).toFixed(0)}%</span>
                                <div class="description">${r.description || r.class_name}</div>
                                <div class="tags">
                                    ${(r.concept_tags || []).map(t =>
                                        `<span class="tag">${t}</span>`
                                    ).join('')}
                                </div>
                                <div class="timestamp">
                                    ${formatTimestamp(r.timestamp_sec)}
                                    &mdash; ${r.source_video?.split('/').pop() || ''}
                                </div>
                            </div>
                        `).join('')}
                    `;
                } else {
                    resultsDiv.innerHTML = '<div class="empty">No detections match your query.</div>';
                }
            } catch (e) {
                resultsDiv.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
            }

            btn.disabled = false;
            btn.textContent = 'Search';
        }

        function formatTimestamp(sec) {
            if (sec == null) return '';
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60);
            return `${m}:${s.toString().padStart(2, '0')}`;
        }

        loadStats();
    </script>
</body>
</html>
"""


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
