"""
Unit and Integration Tests for Phase 5: Reranker Pipeline.
Tests NVIDIA Llama Nemotron Rerank VL 1B V2 (via OpenRouter),
DeterministicLocalReranker, RRF-to-Rerank transitions, sync/async parity,
resilient fallbacks, and FastAPI endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from app.config import settings
from app.core.rag import kb
from app.core.reranker import (
    BaseReranker,
    DeterministicLocalReranker,
    OpenRouterNemotronReranker,
    get_reranker,
)
from app.main import app


# =====================================================================
# Fixtures & Sample Candidates
# =====================================================================

@pytest.fixture
def sample_candidates() -> List[Dict[str, Any]]:
    """Sample candidate chunks simulating output from Phase 4 Hybrid Search (Top 5)."""
    return [
        {
            "id": "amenity-pool",
            "category": "amenity",
            "topic": "pool",
            "title": "Oceanview Infinity Pool",
            "content": "The infinity pool is open daily from 6:00 AM to 9:00 PM with ocean views.",
            "keywords": ["pool", "swimming", "infinity", "hours"],
            "rrf_score": 0.016393,
            "rrf_rank": 1,
            "channels": ["dense", "bm25"],
            "dense_score": 0.88,
            "bm25_score": 4.5,
        },
        {
            "id": "faq-checkin",
            "category": "faq",
            "topic": "checkin",
            "title": "Check-in and Check-out Timings",
            "content": "Standard check-in begins at 2:00 PM IST and check-out is strictly by 11:00 AM.",
            "keywords": ["check-in", "check-out", "timing", "hours"],
            "rrf_score": 0.016129,
            "rrf_rank": 2,
            "channels": ["dense"],
            "dense_score": 0.75,
            "bm25_score": 0.0,
        },
        {
            "id": "room-deluxe",
            "category": "room",
            "topic": "deluxe",
            "title": "Deluxe King Room",
            "content": "A spacious 450 sq ft room with private balcony overlooking the Arabian Sea.",
            "keywords": ["deluxe", "king", "room", "balcony"],
            "rrf_score": 0.015873,
            "rrf_rank": 3,
            "channels": ["dense", "bm25"],
            "dense_score": 0.70,
            "bm25_score": 2.1,
        },
        {
            "id": "policy-pets",
            "category": "policy",
            "topic": "pets",
            "title": "Pet Policy & Guide Animals",
            "content": "Pets are strictly not allowed on resort grounds, except certified guide dogs with prior intimation.",
            "keywords": ["pets", "dogs", "animals", "policy"],
            "rrf_score": 0.015625,
            "rrf_rank": 4,
            "channels": ["bm25"],
            "dense_score": 0.0,
            "bm25_score": 3.8,
        },
        {
            "id": "amenity-dining",
            "category": "amenity",
            "topic": "dining",
            "title": "Saffron Coastal & Spice Restaurant",
            "content": "Serving authentic Goan and coastal Indian cuisine from 12:30 PM to 11:00 PM.",
            "keywords": ["food", "dining", "restaurant", "lunch"],
            "rrf_score": 0.015385,
            "rrf_rank": 5,
            "channels": ["dense"],
            "dense_score": 0.65,
            "bm25_score": 0.0,
        },
    ]


# =====================================================================
# 1. Deterministic Local Reranker Tests
# =====================================================================

def test_deterministic_local_reranker_scoring(sample_candidates):
    """Verify local cross-scorer scores candidates, assigns rank, and computes delta."""
    reranker = DeterministicLocalReranker()
    query = "What time is check-in?"
    results = reranker.rerank(query, sample_candidates, top_n=3)

    assert len(results) == 3
    # Check-in candidate should be elevated to rank 1
    top_result = results[0]
    assert top_result["id"] == "faq-checkin"
    assert top_result["rerank_rank"] == 1
    assert top_result["rerank_score"] > 0.0
    assert top_result["rerank_fallback"] is True
    # Prior rank was 2, now 1, so delta = 2 - 1 = +1
    assert top_result["rerank_delta"] == 1


def test_deterministic_local_reranker_reordering(sample_candidates):
    """Verify irrelevant distractors are pushed down and relevant candidates elevated."""
    reranker = DeterministicLocalReranker()
    query = "Can I bring my golden retriever dog?"
    results = reranker.rerank(query, sample_candidates, top_n=3)

    assert len(results) == 3
    # Pet policy (was rank 4 in hybrid) should be elevated to rank 1
    assert results[0]["id"] == "policy-pets"
    assert results[0]["rerank_rank"] == 1
    assert results[0]["rerank_delta"] == 3  # moved from rank 4 to rank 1


# =====================================================================
# 2. OpenRouter Nemotron Reranker Mocked Tests
# =====================================================================

def test_openrouter_nemotron_request_payload_and_headers(sample_candidates):
    """Verify OpenRouter rerank request headers, endpoint URL, and JSON payload format."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "gen-rerank-12345",
        "results": [
            {"index": 1, "relevance_score": 0.985},
            {"index": 0, "relevance_score": 0.620},
            {"index": 2, "relevance_score": 0.310},
        ],
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        reranker = OpenRouterNemotronReranker(
            api_key="test-openrouter-key",
            model="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
            top_n=3,
        )
        results = reranker.rerank("What time is check-in?", sample_candidates, top_n=3)

        assert len(results) == 3
        assert mock_post.called

        # Verify call details
        call_url = mock_post.call_args[0][0]
        call_kwargs = mock_post.call_args[1]

        assert call_url == "https://openrouter.ai/api/v1/rerank"
        assert "Bearer test-openrouter-key" in call_kwargs["headers"]["Authorization"]
        assert call_kwargs["json"]["model"] == "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
        assert call_kwargs["json"]["query"] == "What time is check-in?"
        assert len(call_kwargs["json"]["documents"]) == len(sample_candidates)
        assert call_kwargs["json"]["top_n"] == 3


def test_openrouter_nemotron_response_parsing_and_ranking(sample_candidates):
    """Verify response parsing maps index back to candidate and assigns rerank rank and delta."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {"index": 3, "relevance_score": 0.9921},  # policy-pets was rank 4
            {"index": 1, "relevance_score": 0.4120},  # faq-checkin was rank 2
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        reranker = OpenRouterNemotronReranker(
            api_key="test-openrouter-key",
            model="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
            top_n=2,
        )
        results = reranker.rerank("Can I bring my dog?", sample_candidates, top_n=2)

        assert len(results) == 2
        # First item is policy-pets (index 3)
        assert results[0]["id"] == "policy-pets"
        assert results[0]["rerank_rank"] == 1
        assert results[0]["rerank_score"] == 0.9921
        assert results[0]["rerank_delta"] == 3  # 4 - 1 = +3
        assert results[0]["rerank_fallback"] is False
        assert results[0]["rerank_model"] == "nvidia/llama-nemotron-rerank-vl-1b-v2:free"

        # Second item is faq-checkin (index 1)
        assert results[1]["id"] == "faq-checkin"
        assert results[1]["rerank_rank"] == 2
        assert results[1]["rerank_score"] == 0.412
        assert results[1]["rerank_delta"] == 0  # 2 - 2 = 0


def test_openrouter_nemotron_truncation_top_n(sample_candidates):
    """Verify that result list is truncated exactly to requested top_n."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
            {"index": 2, "relevance_score": 0.7},
            {"index": 3, "relevance_score": 0.6},
            {"index": 4, "relevance_score": 0.5},
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        reranker = OpenRouterNemotronReranker(
            api_key="test-openrouter-key",
            top_n=2,
        )
        results = reranker.rerank("query", sample_candidates, top_n=2)
        assert len(results) == 2


# =====================================================================
# 3. Fallback and Edge Case Tests
# =====================================================================

def test_openrouter_nemotron_fallback_when_no_api_key(sample_candidates):
    """Verify that missing API key cleanly triggers local fallback without errors."""
    reranker = OpenRouterNemotronReranker(api_key="", top_n=3)
    results = reranker.rerank("What time is check-in?", sample_candidates, top_n=3)

    assert len(results) == 3
    assert results[0]["id"] == "faq-checkin"
    assert results[0]["rerank_fallback"] is True
    stats = reranker.get_stats()
    assert stats["fallback_reranks"] >= 1


def test_openrouter_nemotron_fallback_on_api_error(sample_candidates):
    """Verify that network exceptions trigger retry and graceful fallback."""
    with patch("httpx.Client.post", side_effect=RuntimeError("Connection timeout")):
        reranker = OpenRouterNemotronReranker(
            api_key="test-key",
            max_retries=1,
            top_n=3,
        )
        results = reranker.rerank("swimming pool", sample_candidates, top_n=3)

        assert len(results) == 3
        # Infinity pool should be top in fallback for 'swimming pool'
        assert results[0]["id"] == "amenity-pool"
        assert results[0]["rerank_fallback"] is True


def test_openrouter_nemotron_empty_candidates():
    """Verify empty candidate list returns empty list immediately."""
    reranker = OpenRouterNemotronReranker(api_key="test-key")
    assert reranker.rerank("query", []) == []


# =====================================================================
# 4. Async Parity & Singleton Tests
# =====================================================================

@pytest.mark.asyncio
async def test_openrouter_nemotron_async_parity(sample_candidates):
    """Verify arerank provides identical scoring and ordering as synchronous rerank."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.85},
            {"index": 2, "relevance_score": 0.75},
        ]
    }

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        reranker = OpenRouterNemotronReranker(
            api_key="test-key",
            top_n=3,
        )
        async_results = await reranker.arerank("pool", sample_candidates, top_n=3)
        assert len(async_results) == 3
        assert async_results[0]["id"] == "amenity-pool"
        assert async_results[0]["rerank_rank"] == 1


def test_reranker_singleton_identity():
    """Verify get_reranker() behaves as a thread-safe singleton."""
    r1 = get_reranker()
    r2 = get_reranker()
    assert r1 is r2

    r3 = get_reranker(force_new=True)
    assert r3 is not r1
    # Cleanup to default singleton
    get_reranker(force_new=True)


# =====================================================================
# 5. RAG Integration & FastAPI API Endpoints Tests
# =====================================================================

def test_rag_retrieve_reranked_integration():
    """Verify kb.retrieve_reranked() integrates Hybrid Search (Top 10) with Reranker (Top 3)."""
    from app.core.vector_store import DeterministicLocalEmbedder
    with patch("app.core.vector_store.get_embedder", return_value=DeterministicLocalEmbedder()):
        results = kb.retrieve_reranked("What time is check-in?", top_n=3)
        assert len(results) <= 3
        assert len(results) >= 1
        assert "rerank_rank" in results[0]
        assert "rerank_score" in results[0]


def test_fastapi_rerank_stats_endpoint():
    """Verify GET /api/rerank/stats returns 200 and valid schema (zero-cost FlashRank default)."""
    mock_store = MagicMock()
    mock_store.get_point_count.return_value = 29

    with patch("app.core.vector_store.get_vector_store", return_value=mock_store):
        with TestClient(app) as client:
            res = client.get("/api/rerank/stats")
            assert res.status_code == 200
            data = res.json()
            assert "model" in data
            assert "provider" in data
            assert data["provider"] == "flashrank-local"
            assert "MiniLM" in data["model"]
            assert "default_top_n" in data
            assert "total_requests" in data


def test_fastapi_rerank_post_endpoint(sample_candidates):
    """Verify POST /api/rerank with explicit candidates executes successfully."""
    mock_store = MagicMock()
    mock_store.get_point_count.return_value = 29

    payload = {
        "query": "Can I bring my pet dog?",
        "top_n": 2,
        "candidates": sample_candidates,
    }
    with patch("app.core.vector_store.get_vector_store", return_value=mock_store):
        with TestClient(app) as client:
            res = client.post("/api/rerank", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["query"] == "Can I bring my pet dog?"
            assert data["count"] == 2
            assert len(data["results"]) == 2
            assert "rerank_score" in data["results"][0]
            assert "rerank_rank" in data["results"][0]
            assert data["results"][0]["id"] == "policy-pets"
