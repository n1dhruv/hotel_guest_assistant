"""Zero-cost local providers: FastEmbed embeddings + FlashRank reranking.

Verifies the quota-free pipeline stages work without any external API calls:
- FastEmbedAdapter (BAAI/bge-small-en-v1.5, 384 dim, local CPU).
- FlashRankReranker (MiniLM ONNX cross-encoder, Top 10 -> Top 3).
- get_reranker() provider routing (flashrank / local / openrouter).
- Graceful fallback to DeterministicLocalReranker when FlashRank is unavailable.
"""
import asyncio

import pytest

from app.core.reranker import (
    DeterministicLocalReranker,
    FlashRankReranker,
    OpenRouterNemotronReranker,
    get_reranker,
)
from app.core.vector_store import FastEmbedAdapter, get_embedder


def _candidates():
    return [
        {
            "id": "pool",
            "category": "amenity",
            "title": "Amenity: Oceanview Infinity Pool",
            "content": "Heated oceanview infinity pool open daily 6 AM to 9 PM.",
            "rrf_score": 0.02,
            "rrf_rank": 2,
        },
        {
            "id": "cancel",
            "category": "policy",
            "title": "Policy: Cancellation",
            "content": "Free cancellation up to 24 hours before check-in.",
            "rrf_score": 0.03,
            "rrf_rank": 1,
        },
        {
            "id": "spa",
            "category": "amenity",
            "title": "Amenity: Spa",
            "content": "Ayurvedic spa with massage therapies.",
            "rrf_score": 0.01,
            "rrf_rank": 3,
        },
    ]


# --- FastEmbed ---

def test_fastembed_factory_resolution_and_dimension():
    adapter = get_embedder("fastembed")
    assert isinstance(adapter, FastEmbedAdapter)
    assert adapter.dimension == 384
    assert "fastembed" in adapter.provider_name


def test_fastembed_produces_384d_vectors():
    adapter = get_embedder("fastembed")
    vec = adapter.embed_query("infinity pool timings")
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)
    batch = adapter.embed_documents(["pool", "spa"])
    assert len(batch) == 2 and all(len(v) == 384 for v in batch)


# --- FlashRank ---

def test_flashrank_reranks_pool_query_to_top():
    reranker = FlashRankReranker(top_n=3)
    results = reranker.rerank("When is the swimming pool open?", _candidates())
    assert len(results) == 3
    assert results[0]["id"] == "pool"  # cross-encoder beats RRF order (cancel was rank 1)
    for i, r in enumerate(results, start=1):
        assert r["rerank_rank"] == i
        assert "rerank_score" in r and "rerank_delta" in r
        assert r["rerank_fallback"] is False
        assert r["rerank_model"] == reranker.model_name


def test_flashrank_top_n_truncation():
    reranker = FlashRankReranker(top_n=1)
    results = reranker.rerank("pool", _candidates())
    assert len(results) == 1
    assert results[0]["rerank_rank"] == 1


def test_flashrank_empty_candidates():
    assert FlashRankReranker().rerank("pool", []) == []


def test_flashrank_async_parity():
    reranker = FlashRankReranker(top_n=2)
    sync_res = reranker.rerank("pool timings?", _candidates())
    async_res = asyncio.run(reranker.arerank("pool timings?", _candidates(), top_n=2))
    assert [r["id"] for r in sync_res] == [r["id"] for r in async_res]


def test_flashrank_stats_shape():
    reranker = FlashRankReranker(top_n=3)
    reranker.rerank("pool", _candidates())
    stats = reranker.get_stats()
    assert stats["provider"] == "flashrank-local"
    assert stats["total_requests"] == 1
    assert stats["successful_reranks"] == 1
    assert stats["model_loaded"] is True


def test_flashrank_falls_back_when_model_unavailable(monkeypatch):
    class _BoomRanker:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no weights available")

    monkeypatch.setattr("flashrank.Ranker", _BoomRanker)
    reranker = FlashRankReranker(top_n=2)
    results = reranker.rerank("pool timings?", _candidates())
    assert len(results) == 2  # deterministic fallback still answers, never crashes
    assert all(r["rerank_fallback"] is True for r in results)


# --- Provider routing ---

def test_get_reranker_routes_flashrank_by_default(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "RERANKER_PROVIDER", "flashrank")
    rr = get_reranker(force_new=True)
    assert isinstance(rr, FlashRankReranker)


def test_get_reranker_routes_local_and_openrouter(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "RERANKER_PROVIDER", "local")
    assert isinstance(get_reranker(force_new=True), DeterministicLocalReranker)
    monkeypatch.setattr(settings, "RERANKER_PROVIDER", "openrouter")
    rr = get_reranker(force_new=True, api_key="test-key")
    assert isinstance(rr, OpenRouterNemotronReranker)
    # Restore zero-cost default singleton for other tests
    monkeypatch.setattr(settings, "RERANKER_PROVIDER", "flashrank")
    get_reranker(force_new=True)
