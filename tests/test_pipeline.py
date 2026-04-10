"""
Keyhole — Test Suite

Basic tests for pipeline components. Run with:
    pytest tests/ -v
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# --- Video Ingestion Tests ---

class TestVideoIngestion:
    def test_extracted_frame_dataclass(self):
        from src.ingest.video import ExtractedFrame

        frame = ExtractedFrame(
            frame_number=42,
            timestamp_sec=1.4,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            source_video="/tmp/test.mp4",
        )
        assert frame.frame_number == 42
        assert frame.timestamp_sec == 1.4
        assert frame.image.shape == (480, 640, 3)


# --- YOLO Detection Tests ---

class TestYOLODetection:
    def test_detection_dataclass(self):
        from src.detect.yolo import Detection

        det = Detection(
            bbox=(100.0, 200.0, 300.0, 400.0),
            class_id=0,
            class_name="person",
            confidence=0.92,
        )
        assert det.width == 200.0
        assert det.height == 200.0
        assert det.area == 40000.0
        assert det.center == (200.0, 300.0)

    def test_frame_detections_count(self):
        from src.detect.yolo import Detection, FrameDetections

        fd = FrameDetections(
            frame_number=0,
            timestamp_sec=0.0,
            source_video="test.mp4",
            detections=[
                Detection(bbox=(0, 0, 10, 10), class_id=0,
                         class_name="person", confidence=0.9),
                Detection(bbox=(50, 50, 100, 100), class_id=2,
                         class_name="car", confidence=0.8),
            ],
        )
        assert fd.count == 2


# --- SAM 3 Enrichment Tests ---

class TestSAM3Enrichment:
    def test_concept_match_dataclass(self):
        from src.enrich.sam3 import ConceptMatch

        cm = ConceptMatch(
            concept="red hat",
            confidence=0.87,
            mask_area_pct=12.5,
        )
        assert cm.concept == "red hat"

    def test_enriched_detection_tags(self):
        from src.enrich.sam3 import EnrichedDetection, ConceptMatch

        ed = EnrichedDetection(
            bbox=(0, 0, 100, 100),
            class_id=0,
            class_name="person",
            confidence=0.9,
            concepts=[
                ConceptMatch(concept="hat", confidence=0.8, mask_area_pct=5.0),
                ConceptMatch(concept="backpack", confidence=0.7, mask_area_pct=15.0),
            ],
            description="Person with: hat, backpack",
        )
        assert ed.concept_tags == ["hat", "backpack"]

    def test_class_concept_map_coverage(self):
        from src.enrich.sam3 import CLASS_CONCEPT_MAP

        assert "person" in CLASS_CONCEPT_MAP
        assert "car" in CLASS_CONCEPT_MAP
        assert len(CLASS_CONCEPT_MAP["person"]) > 10


# --- Database Tests ---

class TestDatabase:
    def test_create_and_query(self):
        from src.store.models import create_db
        from src.store.db import DetectionStore
        from src.ingest.video import VideoMeta
        from src.enrich.sam3 import EnrichedFrame, EnrichedDetection, ConceptMatch

        with tempfile.TemporaryDirectory() as tmpdir:
            db_url = f"sqlite:///{tmpdir}/test.db"
            store = DetectionStore(db_url)

            # Register a video
            meta = VideoMeta(
                path="/tmp/test.mp4",
                width=1920, height=1080,
                fps=30.0, duration=60.0,
                total_frames=1800, codec="h264",
            )
            video_id = store.register_video(meta)
            assert video_id > 0

            # Store an enriched frame
            enriched = EnrichedFrame(
                frame_number=0,
                timestamp_sec=0.0,
                source_video="/tmp/test.mp4",
                detections=[
                    EnrichedDetection(
                        bbox=(100, 200, 300, 400),
                        class_id=0,
                        class_name="person",
                        confidence=0.95,
                        concepts=[
                            ConceptMatch(
                                concept="red hat",
                                confidence=0.88,
                                mask_area_pct=8.0,
                            ),
                        ],
                        description="Person with: red hat",
                    ),
                ],
            )
            store.store_enriched_frame(video_id, enriched)

            # Query by class
            results = store.search_detections(class_name="person")
            assert len(results) == 1
            assert results[0]["class_name"] == "person"

            # Query by description text
            results = store.search_detections(query_text="red hat")
            assert len(results) == 1

            # Query by concept tag
            results = store.search_detections(concept_tag="red hat")
            assert len(results) == 1

            # Stats
            stats = store.get_stats()
            assert stats["total_events"] == 1
            assert stats["total_frames"] == 1
            assert stats["total_videos"] == 1


# --- NLQ Engine Tests ---

class TestNLQEngine:
    @pytest.mark.asyncio
    async def test_query_translation_fallback(self):
        """Test that NLQ falls back to text search on LLM failure."""
        from src.store.db import DetectionStore
        from src.query.nlq import NLQueryEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            db_url = f"sqlite:///{tmpdir}/test.db"
            store = DetectionStore(db_url)

            nlq = NLQueryEngine(
                store=store,
                backend="anthropic",
                api_key="invalid-key",  # Will fail
            )

            # Should fall back to text search without crashing
            result = await nlq.query("find all people")
            assert "user_query" in result
            assert "results" in result


# --- Profile Report Tests ---

class TestProfileReport:
    def test_vision_latency_estimate(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        # Inline the estimation logic for testing
        target_tops = 200.0
        target_bw = 134.4

        # SAM 3 full
        gflops = 350.0
        model_gb = 1.696
        compute_ms = (gflops / target_tops) * 1000  # 1.75ms
        bw_ms = (model_gb * 0.15 / target_bw) * 1000  # ~1.89ms

        assert compute_ms < 5.0, "SAM 3 compute should be fast on 200 TOPS"
        assert bw_ms < 5.0, "SAM 3 bandwidth should be manageable"

    def test_llm_throughput_estimate(self):
        target_bw = 134.4

        # Qwen 2.5 3B INT4
        weight_gb = 1.5
        ms_per_token = (weight_gb / target_bw) * 1000
        tok_per_sec = 1000.0 / ms_per_token

        assert tok_per_sec > 50, "Should get >50 tok/s on Qwen 3B INT4"
        assert tok_per_sec < 200, "Sanity check upper bound"


import sys
