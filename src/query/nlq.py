"""
Natural Language Query Engine — LLM-powered search interface.

Translates natural language questions like "find me any person wearing
a yellow hat" into structured database queries. Supports multiple
LLM backends: Claude API, Ollama (local), and Skippy.
"""

import json
import logging
from typing import Optional

import httpx

from src.store.db import DetectionStore

logger = logging.getLogger(__name__)

# System prompt for the query translation LLM
QUERY_SYSTEM_PROMPT = """You are a video surveillance query translator. Your job is to convert
natural language questions about surveillance footage into structured JSON queries.

The database contains detection events with these searchable fields:
- class_name: object class (person, car, truck, bus, motorcycle, bicycle, dog, cat, bird, etc.)
- description: natural language description of the detection (e.g., "Person with: red hat, backpack, sunglasses")
- concept_tags: list of detected attributes (e.g., ["red hat", "backpack", "sunglasses"])
- timestamp_sec: seconds into the video when detected
- confidence: detection confidence (0.0 to 1.0)
- source_video: path to source video file

Given a user question, respond with ONLY a JSON object (no markdown, no explanation) with these fields:
{
    "query_text": "text to search in descriptions (or null)",
    "class_name": "object class filter (or null)",
    "concept_tag": "specific attribute to search for (or null)",
    "time_start": null,
    "time_end": null,
    "min_confidence": 0.0,
    "summary_request": "what the user wants to know, for formatting the response"
}

Examples:
- "find anyone wearing a red hat" -> {"query_text": "red hat", "class_name": "person", "concept_tag": "red hat", "min_confidence": 0.0, "summary_request": "List all people detected wearing a red hat"}
- "show me all vehicles in the last 5 minutes" -> {"query_text": null, "class_name": "car", "concept_tag": null, "time_start": null, "time_end": null, "min_confidence": 0.0, "summary_request": "List all vehicles detected"}
- "was there a dog?" -> {"query_text": null, "class_name": "dog", "concept_tag": null, "min_confidence": 0.0, "summary_request": "Check if any dogs were detected"}
- "person with a backpack near the entrance" -> {"query_text": "backpack", "class_name": "person", "concept_tag": "backpack", "min_confidence": 0.0, "summary_request": "Find people carrying backpacks"}
"""


class NLQueryEngine:
    """
    Natural language query engine.

    Uses an LLM to translate user questions into structured database
    queries, executes them, and formats results.
    """

    def __init__(
        self,
        store: DetectionStore,
        backend: str = "anthropic",
        api_key: str = "",
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b",
        skippy_host: str = "http://localhost:8000",
        skippy_model: str = "mixtral-8x7b",
    ):
        self.store = store
        self.backend = backend
        self.api_key = api_key
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.skippy_host = skippy_host
        self.skippy_model = skippy_model

        logger.info("NLQ Engine initialized with backend: %s", backend)

    async def _call_anthropic(self, user_query: str) -> str:
        """Call Claude API for query translation."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "system": QUERY_SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_query}
                    ],
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]

    async def _call_ollama(self, user_query: str) -> str:
        """Call local Ollama for query translation."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "system": QUERY_SYSTEM_PROMPT,
                    "prompt": user_query,
                    "stream": False,
                    "format": "json",
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]

    async def _call_skippy(self, user_query: str) -> str:
        """Call Skippy local LLM for query translation."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.skippy_host}/v1/chat/completions",
                json={
                    "model": self.skippy_model,
                    "messages": [
                        {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_query},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _translate_query(self, user_query: str) -> dict:
        """Translate natural language query to structured search params."""
        try:
            if self.backend == "anthropic":
                raw = await self._call_anthropic(user_query)
            elif self.backend == "ollama":
                raw = await self._call_ollama(user_query)
            elif self.backend == "skippy":
                raw = await self._call_skippy(user_query)
            else:
                raise ValueError(f"Unknown LLM backend: {self.backend}")

            # Clean up response — strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            return json.loads(raw)

        except (json.JSONDecodeError, httpx.HTTPError) as e:
            logger.error("Query translation failed: %s", e)
            # Fallback: use the raw query as a text search
            return {
                "query_text": user_query,
                "class_name": None,
                "concept_tag": None,
                "time_start": None,
                "time_end": None,
                "min_confidence": 0.0,
                "summary_request": user_query,
            }

    async def query(self, user_query: str) -> dict:
        """
        Execute a natural language query against the detection store.

        Returns structured results with the original query context.
        """
        logger.info("NLQ query: %s", user_query)

        # Step 1: Translate to structured query
        params = await self._translate_query(user_query)
        logger.debug("Translated query params: %s", params)

        # Step 2: Execute structured search
        results = self.store.search_detections(
            query_text=params.get("query_text"),
            class_name=params.get("class_name"),
            concept_tag=params.get("concept_tag"),
            time_start=params.get("time_start"),
            time_end=params.get("time_end"),
            min_confidence=params.get("min_confidence", 0.0),
            limit=50,
        )

        return {
            "user_query": user_query,
            "translated_params": params,
            "result_count": len(results),
            "results": results,
        }

    def query_sync(self, user_query: str) -> dict:
        """Synchronous wrapper for query()."""
        import asyncio
        return asyncio.run(self.query(user_query))
