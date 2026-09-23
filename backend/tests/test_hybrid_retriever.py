"""
Unit & Integration Tests for Phase 4: Hybrid Search Engine (Vector + BM25 + Reciprocal Rank Fusion).

Verifies:
1. BM25Index tokenization, stopwords, synonyms, and title boosting
2. Exact keyword / acronym / numerical recall in BM25 (e.g. 500 Mbps, Aadhaar, Tata Power)
3. Mathematical precision of Reciprocal Rank Fusion (RRF formula with k=60)
4. Candidate deduplication and multi-channel score fusion
5. Dual-channel retrieval execution (Dense + BM25)
6. Synchronous and asynchronous concurrent retrieval parity
7. Top 10 fused candidate chunk extraction
8. Category filtering in hybrid retrieval
9. Edge cases: empty/whitespace queries, single-channel hits
10. FastAPI endpoints: /api/hybrid-search/stats and /api/hybrid-search
11. Singleton thread safety
"""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.chunker import SemanticChunk
from app.core.hybrid_retriever import (
    BM25Index,
    HybridRetriever,
    get_hybrid_retriever,
    reciprocal_rank_fusion,
)
from app.core.vector_store import DeterministicLocalEmbedder, QdrantVectorStore
from app.main import app
import app.core.hybrid_retriever as hr


@pytest.fixture(autouse=True)
def isolated_test_hybrid_retriever():
    """Ensure tests run with a deterministic in-memory vector store without requiring external API keys."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    vstore = QdrantVectorStore(
        collection_name="test_fixture_hybrid",
        embedder=embedder,
        in_memory=True,
        auto_index=True,
    )
    retriever = HybridRetriever(
        vector_store=vstore,
        bm25_index=BM25Index(),
        use_hyde=False,
    )
    original = hr._hybrid_instance
    hr._hybrid_instance = retriever
    yield retriever
    hr._hybrid_instance = original
    vstore.close()


# =====================================================================
# 1. BM25Index Unit Tests
# =====================================================================

def test_bm25_tokenization_and_normalization():
    """Verify BM25 tokenization handles synonyms, phrase normalization, and stopwords."""
    index = BM25Index(chunks=[])
    tokens = index.tokenize("What is the Wi-Fi speed and check-in time for my dogs?")

    assert "the" not in tokens
    assert "is" not in tokens
    assert "what" not in tokens

    # Normalized compounds
    assert "wifi" in tokens
    assert "checkin" in tokens

    # Synonyms
    assert "pets" in tokens


def test_bm25_exact_match_retrieval():
    """Verify BM25 excels at exact technical terms and numerical attributes."""
    test_chunks = [
        {
            "chunk_id": "wifi-chunk",
            "title": "High-Speed Internet",
            "category": "amenity",
            "topic": "wifi",
            "content": "Complimentary high-speed fiber internet with up to 500 Mbps bandwidth resort-wide.",
            "keywords": ["wifi", "internet", "500 mbps"],
        },
        {
            "chunk_id": "ev-chunk",
            "title": "EV Charging Station",
            "category": "amenity",
            "topic": "ev",
            "content": "Tata Power 60kW fast DC electric vehicle charging stations available 24/7.",
            "keywords": ["ev", "tata power", "charging"],
        },
        {
            "chunk_id": "pool-chunk",
            "title": "Infinity Pool",
            "category": "amenity",
            "topic": "pool",
            "content": "Oceanview infinity pool open from 7:00 AM to 8:00 PM.",
            "keywords": ["pool", "swimming"],
        },
    ]

    index = BM25Index(chunks=test_chunks)

    # Search for specific number and technical term
    res_wifi = index.search("500 Mbps internet", top_k=2)
    assert len(res_wifi) >= 1
    assert res_wifi[0]["chunk_id"] == "wifi-chunk"
    assert res_wifi[0]["bm25_score"] > 0.0
    assert res_wifi[0]["bm25_rank"] == 1

    # Search for specific brand / acronym
    res_ev = index.search("Tata Power fast charger", top_k=2)
    assert len(res_ev) >= 1
    assert res_ev[0]["chunk_id"] == "ev-chunk"
    assert res_ev[0]["bm25_score"] > 0.0


def test_bm25_title_match_boost():
    """Verify that chunks with matching tokens in the title receive higher BM25 scores."""
    chunks = [
        {
            "chunk_id": "c1",
            "title": "Oceanview Swimming Pool",
            "category": "amenity",
            "topic": "pool",
            "content": "Relax at our resort.",
            "keywords": [],
        },
        {
            "chunk_id": "c2",
            "title": "Resort Overview",
            "category": "property",
            "topic": "overview",
            "content": "Our property features an oceanview swimming pool.",
            "keywords": [],
        },
    ]

    index = BM25Index(chunks=chunks)
    results = index.search("swimming pool", top_k=2)

    assert len(results) == 2
    # c1 has 'swimming pool' in the title and should score higher due to the 3x title boost
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["bm25_score"] > results[1]["bm25_score"]


# =====================================================================
# 2. Reciprocal Rank Fusion (RRF) Algorithm Tests
# =====================================================================

def test_rrf_mathematical_precision():
    """Verify RRF formula calculation: sum(1 / (k + rank))."""
    dense_results = [
        {"id": "doc-A", "title": "Doc A", "score": 0.95},
        {"id": "doc-B", "title": "Doc B", "score": 0.85},
    ]
    bm25_results = [
        {"id": "doc-B", "title": "Doc B", "bm25_score": 4.5},
        {"id": "doc-C", "title": "Doc C", "bm25_score": 3.0},
    ]

    # k = 60
    # doc-A: Dense Rank 1 -> 1 / (60 + 1) = 1/61 =~ 0.016393
    # doc-B: Dense Rank 2 (1/62) + BM25 Rank 1 (1/61) = 1/62 + 1/61 =~ 0.016129 + 0.016393 = 0.032522
    # doc-C: BM25 Rank 2 -> 1 / (60 + 2) = 1/62 =~ 0.016129
    fused = reciprocal_rank_fusion(dense_results, bm25_results, rrf_k=60, top_k=10)

    assert len(fused) == 3

    # doc-B was retrieved by both channels -> MUST rank 1st
    assert fused[0]["id"] == "doc-B"
    assert fused[0]["channels"] == ["dense", "bm25"]
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1
    assert pytest.approx(fused[0]["rrf_score"], abs=1e-5) == (1 / 62 + 1 / 61)

    # doc-A was rank 1 in Dense -> MUST rank 2nd
    assert fused[1]["id"] == "doc-A"
    assert fused[1]["channels"] == ["dense"]
    assert fused[1]["dense_rank"] == 1
    assert fused[1]["bm25_rank"] is None
    assert pytest.approx(fused[1]["rrf_score"], abs=1e-5) == (1 / 61)

    # doc-C was rank 2 in BM25 -> MUST rank 3rd
    assert fused[2]["id"] == "doc-C"
    assert fused[2]["channels"] == ["bm25"]
    assert fused[2]["dense_rank"] is None
    assert fused[2]["bm25_rank"] == 2
    assert pytest.approx(fused[2]["rrf_score"], abs=1e-5) == (1 / 62)


def test_rrf_deduplication_and_top_k_truncation():
    """Verify RRF deduplicates overlapping items and respects top_k limit."""
    dense = [{"id": f"chunk-{i}", "title": f"Chunk {i}"} for i in range(15)]
    bm25 = [{"id": f"chunk-{i}", "title": f"Chunk {i}"} for i in range(10, 25)]

    fused = reciprocal_rank_fusion(dense, bm25, rrf_k=60, top_k=10)

    assert len(fused) == 10
    unique_ids = {item["id"] for item in fused}
    assert len(unique_ids) == 10

    # Overlapping chunks (10, 11, 12, 13, 14) receive fusion boosts and appear in top ranks
    for item in fused[:5]:
        assert "dense" in item["channels"]
        assert "bm25" in item["channels"]


# =====================================================================
# 3. Dual-Channel Hybrid Search Engine Tests
# =====================================================================

def test_hybrid_retriever_sync_and_top_10():
    """Verify HybridRetriever executes both channels and returns Top 10 fused candidates."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    vector_store = QdrantVectorStore(
        collection_name="test_hybrid_sync",
        embedder=embedder,
        in_memory=True,
        auto_index=True,
    )
    bm25_index = BM25Index()

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        rrf_k=60,
        default_top_k=10,
        use_hyde=False,
    )

    results = retriever.retrieve_candidates("Can I bring my pet dog?", top_k=10)

    assert len(results) == 10
    for r in results:
        assert "rrf_score" in r
        assert "channels" in r
        assert "title" in r
        assert "content" in r

    # Top result should be pet policy or pet FAQ
    top_chunk = results[0]
    assert "pet" in (top_chunk["title"] + " " + top_chunk["content"]).lower()

    vector_store.close()


@pytest.mark.asyncio
async def test_hybrid_retriever_async_concurrency():
    """Verify asynchronous aretrieve_candidates runs concurrently with matching results."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    vector_store = QdrantVectorStore(
        collection_name="test_hybrid_async",
        embedder=embedder,
        in_memory=True,
        auto_index=True,
    )
    bm25_index = BM25Index()

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        rrf_k=60,
        default_top_k=10,
        use_hyde=False,
    )

    query = "What time is check-in and check-out?"
    sync_results = retriever.retrieve_candidates(query, top_k=5)
    async_results = await retriever.aretrieve_candidates(query, top_k=5)

    assert len(sync_results) == 5
    assert len(async_results) == 5

    # Rank order and IDs match identically
    for s, a in zip(sync_results, async_results):
        assert s["id"] == a["id"]
        assert s["rrf_score"] == a["rrf_score"]

    vector_store.close()


def test_hybrid_retriever_category_filter():
    """Verify category filtering constrains candidates across both channels."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    vector_store = QdrantVectorStore(
        collection_name="test_hybrid_filter",
        embedder=embedder,
        in_memory=True,
        auto_index=True,
    )
    bm25_index = BM25Index()

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        use_hyde=False,
    )

    room_results = retriever.retrieve_candidates("ocean view", top_k=5, category_filter="room")
    assert len(room_results) >= 1
    for r in room_results:
        assert r["category"] == "room"

    amenity_results = retriever.retrieve_candidates("ocean view", top_k=5, category_filter="amenity")
    assert len(amenity_results) >= 1
    for r in amenity_results:
        assert r["category"] == "amenity"

    vector_store.close()


def test_hybrid_retriever_empty_query():
    """Verify empty or whitespace query returns baseline chunks without crashing."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    vector_store = QdrantVectorStore(
        collection_name="test_hybrid_empty",
        embedder=embedder,
        in_memory=True,
        auto_index=True,
    )
    bm25_index = BM25Index()

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        use_hyde=False,
    )

    res = retriever.retrieve_candidates("", top_k=4)
    assert len(res) == 4

    vector_store.close()


# =====================================================================
# 4. Recall Coverage on Edge-Case Queries
# =====================================================================

def test_hybrid_recall_exact_and_conceptual():
    """
    Verify hybrid retrieval recall on:
    1. Exact terms/numbers (where BM25 excels): e.g. "500 Mbps", "Aadhaar", "Tata Power"
    2. Conceptual/implicit queries (where Dense excels): e.g. "somewhere for my kids to play"
    """
    embedder = DeterministicLocalEmbedder(dimension=384)
    vector_store = QdrantVectorStore(
        collection_name="test_hybrid_edge_cases",
        embedder=embedder,
        in_memory=True,
        auto_index=True,
    )
    bm25_index = BM25Index()

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        use_hyde=False,
    )

    # 1. Exact acronym / number: "500 Mbps"
    wifi_results = retriever.retrieve_candidates("500 Mbps fiber", top_k=3)
    assert any("wi-fi" in (r["title"] + " " + r["content"]).lower() or "wifi" in (r["title"] + " " + r["content"]).lower() for r in wifi_results)

    # 2. Government ID: "Aadhaar"
    id_results = retriever.retrieve_candidates("Can I show my Aadhaar card at check-in?", top_k=3)
    assert any("aadhaar" in (r["title"] + r["content"]).lower() or "id" in r["title"].lower() for r in id_results)

    # 3. Conceptual query: "somewhere for kids to play"
    kids_results = retriever.retrieve_candidates("somewhere for my children and kids", top_k=5)
    assert any("child" in (r["title"] + r["content"]).lower() or "pool" in r["title"].lower() for r in kids_results)

    vector_store.close()


# =====================================================================
# 5. FastAPI Endpoints Tests
# =====================================================================

def test_fastapi_hybrid_search_stats_endpoint():
    """Verify GET /api/hybrid-search/stats returns 200 and schema."""
    with TestClient(app) as client:
        res = client.get("/api/hybrid-search/stats")
        assert res.status_code == 200
        data = res.json()
        assert "rrf_k" in data
        assert "default_top_k" in data
        assert "dense_candidates" in data
        assert "bm25_candidates" in data
        assert "bm25_indexed_chunks" in data
        assert data["bm25_indexed_chunks"] >= 29


def test_fastapi_hybrid_search_post_endpoint():
    """Verify POST /api/hybrid-search returns Top 10 fused candidate chunks."""
    with TestClient(app) as client:
        payload = {
            "query": "Does the hotel have high speed internet and wifi?",
            "top_k": 5,
            "use_hyde": False,
        }
        res = client.post("/api/hybrid-search", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == payload["query"]
        assert data["count"] == 5
        assert len(data["candidates"]) == 5

        top_cand = data["candidates"][0]
        assert "rrf_score" in top_cand
        assert "channels" in top_cand
        assert "dense_rank" in top_cand
        assert "bm25_rank" in top_cand


# =====================================================================
# 6. Singleton Thread Safety
# =====================================================================

def test_hybrid_retriever_singleton_identity():
    """Verify get_hybrid_retriever returns singleton instance."""
    retriever1 = get_hybrid_retriever()
    retriever2 = get_hybrid_retriever()
    assert retriever1 is retriever2
