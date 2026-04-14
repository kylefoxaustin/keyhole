"""
Semantic Embedding Engine — sentence-transformers wrapper for detection search.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~80MB, MIT) to embed
detection descriptions and queries for cosine-similarity search.

At the current scale (8k-100k detections) we keep the embedding matrix
in memory and run numpy cosine similarity — no vector extension needed.
If we need to scale to millions of rows, swap in sqlite-vec or pgvector
without changing the API layer.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


def text_for_detection(
    class_name: str,
    description: str,
    concept_tags: Optional[list[str]] = None,
) -> str:
    """
    Compose the text blob for a single detection that we embed.

    Pattern matches the brief:
        "{class_name}: {description}. tags: {', '.join(concept_tags)}"
    """
    parts = [f"{class_name}: {description}"]
    if concept_tags:
        parts.append(f"tags: {', '.join(concept_tags)}")
    return ". ".join(parts)


class EmbeddingEngine:
    """
    Wraps the sentence-transformer model + in-memory embedding cache.

    Loaded once at API startup. Thread-safe because sentence-transformers
    acquires its own locks around GPU/CPU inference.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        # Cache: detection_id → embedding vector (float32, L2-normalized)
        self._cache: dict[int, np.ndarray] = {}
        # Flattened matrix + id list for fast cosine sim
        self._matrix: Optional[np.ndarray] = None
        self._ids: list[int] = []

    def load(self):
        """Load the model (first call downloads ~80MB from HuggingFace)."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Embedding model ready (%d dim)", EMBED_DIM)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a batch of texts. Returns a (N, 384) float32 matrix,
        L2-normalized so cosine similarity = dot product.
        """
        self.load()
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return vecs

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single query. Returns (384,) float32 vector."""
        return self.encode([text])[0]

    def populate_cache(self, pairs: list[tuple[int, bytes]]) -> int:
        """
        Load existing embeddings from storage into the in-memory cache.

        pairs: list of (detection_id, embedding_bytes).
        Returns the number of vectors cached.
        """
        self._cache.clear()
        for det_id, blob in pairs:
            self._cache[det_id] = np.frombuffer(blob, dtype=np.float32)
        self._rebuild_matrix()
        logger.info("Loaded %d embeddings into cache", len(self._cache))
        return len(self._cache)

    def add(self, detection_id: int, vec: np.ndarray):
        """Add or replace a single embedding in the cache."""
        self._cache[detection_id] = vec.astype(np.float32)
        # Invalidate matrix — will rebuild lazily on next search.
        # For small incremental updates this is fine; for bulk use
        # populate_cache() or _rebuild_matrix() directly.
        self._matrix = None

    def _rebuild_matrix(self):
        """Flatten the cache into a dense matrix for fast cosine sim."""
        if not self._cache:
            self._matrix = None
            self._ids = []
            return
        self._ids = sorted(self._cache.keys())
        self._matrix = np.stack([self._cache[i] for i in self._ids])

    def search(
        self, query_text: str, top_k: int = 50, min_similarity: float = 0.2,
    ) -> list[tuple[int, float]]:
        """
        Semantic search against the cached embedding matrix.

        Returns list of (detection_id, similarity_score) tuples, sorted
        by similarity descending. Only results with similarity >=
        min_similarity are returned.
        """
        if self._matrix is None:
            self._rebuild_matrix()
        if self._matrix is None or len(self._ids) == 0:
            return []

        q_vec = self.encode_one(query_text)  # (384,), already normalized
        # Cosine sim = dot product (both sides L2-normalized)
        scores = self._matrix @ q_vec  # (N,)

        # Top-K by descending score
        k = min(top_k, len(scores))
        # argpartition is O(N), faster than full sort for large N
        if k < len(scores):
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        else:
            top_idx = np.argsort(-scores)

        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < min_similarity:
                break  # Sorted, so remaining are all below threshold
            results.append((self._ids[idx], score))
        return results

    @property
    def count(self) -> int:
        """Number of detections currently embedded in the cache."""
        return len(self._cache)
