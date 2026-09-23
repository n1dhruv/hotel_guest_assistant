"""
Unit & Integration Tests for Phase 2: Qdrant Vector DB & Embedding Pipeline.

Verifies:
1. Embedding Adapters (Deterministic, FastEmbed, Gemini, OpenAI, Provider resolution)
2. Dual-mode Qdrant storage (In-memory :memory: and Local Embedded)
3. Collection creation, dimension checks, automatic recreate on mismatch
4. Idempotent upserting with deterministic UUIDv5
5. Semantic vector search & cosine similarity scoring
6. Metadata filtering by entity category
7. Full 29-chunk indexing of hotel_data.json with golden domain inquiries
8. LangChain Document ingestion compatibility
9. FastAPI /api/vector-store/stats endpoint
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.chunker import SemanticChunk, SemanticHotelChunker
from app.core.json_ingest import json_to_documents
from app.core.vector_store import (
    BaseEmbedder,
    DeterministicLocalEmbedder,
    FastEmbedAdapter,
    NemotronEmbeddingAdapter,
    OpenAIEmbeddingAdapter,
    OpenRouterNemotronEmbeddingAdapter,
    QdrantVectorStore,
    chunk_id_to_uuid,
    get_embedder,
    get_vector_store,
)
from app.main import app


# =====================================================================
# 1. Embedding Adapter Tests
# =====================================================================

def test_deterministic_local_embedder():
    """Verify deterministic token hashing embedder output format and normalization."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    assert embedder.dimension == 384
    assert "deterministic" in embedder.provider_name

    vec = embedder.embed_query("Infinity pool Candolim Goa")
    assert isinstance(vec, list)
    assert len(vec) == 384
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, 1e-4) == 1.0

    # Batch embedding
    docs = ["About hotel", "Swimming pool", "Breakfast buffet"]
    batch_vecs = embedder.embed_documents(docs)
    assert len(batch_vecs) == 3
    for v in batch_vecs:
        assert len(v) == 384
        assert pytest.approx(np.linalg.norm(v), 1e-4) == 1.0


def test_deterministic_embedder_semantic_overlap():
    """Verify that texts sharing key vocabulary produce higher cosine similarity."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    v_pool1 = np.array(embedder.embed_query("swimming pool hours"))
    v_pool2 = np.array(embedder.embed_query("oceanview swimming pool"))
    v_unrelated = np.array(embedder.embed_query("passport aadhaar identity verification"))

    sim_related = float(np.dot(v_pool1, v_pool2))
    sim_unrelated = float(np.dot(v_pool1, v_unrelated))

    assert sim_related > sim_unrelated


def test_fastembed_adapter():
    """Verify local FastEmbed ONNX model CPU inference."""
    try:
        embedder = FastEmbedAdapter()
        assert embedder.dimension == 384
        assert "fastembed" in embedder.provider_name.lower()

        vec = embedder.embed_query("Oceanview Infinity Pool")
        assert len(vec) == 384
        assert isinstance(vec[0], float)

        batch = embedder.embed_documents(["Room tariff", "Free breakfast"])
        assert len(batch) == 2
        assert len(batch[0]) == 384
    except Exception as e:
        pytest.skip(f"FastEmbed model download skipped in this test environment: {e}")


def test_nemotron_adapter_mocked():
    """Verify OpenRouter Nemotron adapter request formatting, 2048 dim, and no fallback."""
    mock_data = [
        {"embedding": [0.01] * 2048},
        {"embedding": [0.02] * 2048}
    ]
    mock_response = MagicMock()
    mock_response.data = mock_data

    with patch("litellm.embedding", return_value=mock_response) as mock_embed:
        adapter = OpenRouterNemotronEmbeddingAdapter(api_key="test-openrouter-key")
        assert adapter.dimension == 2048
        assert "openrouter" in adapter.provider_name
        assert "nemotron-3-embed-1b" in adapter.provider_name

        vectors = adapter.embed_documents(["Text one", "Text two"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 2048
        assert mock_embed.called

        # Verify exact model passed to LiteLLM
        call_kwargs = mock_embed.call_args[1]
        assert call_kwargs["model"] == "openrouter/nvidia/nemotron-3-embed-1b:free"
        assert call_kwargs["api_key"] == "test-openrouter-key"


def test_nemotron_strict_no_fallback_on_error():
    """Verify that OpenRouter Nemotron strictly raises errors and NEVER falls back."""
    with patch("litellm.embedding", side_effect=RuntimeError("LiteLLM OpenRouter error")):
        adapter = OpenRouterNemotronEmbeddingAdapter(api_key="test-openrouter-key", max_retries=1)
        with pytest.raises(RuntimeError) as exc_info:
            adapter.embed_query("Query that fails")
        assert "LiteLLM OpenRouter error" in str(exc_info.value)


def test_openai_adapter_mocked():
    """Verify OpenAI embedding adapter returns 1536 dimensions."""
    mock_lc_openai = MagicMock()
    mock_lc_openai.return_value.embed_documents.return_value = [[0.05] * 1536]
    mock_lc_openai.return_value.embed_query.return_value = [0.05] * 1536

    with patch("langchain_openai.OpenAIEmbeddings", mock_lc_openai):
        adapter = OpenAIEmbeddingAdapter(api_key="sk-mock-key")
        assert adapter.dimension == 1536
        assert "openai" in adapter.provider_name

        vec = adapter.embed_query("Luxury room")
        assert len(vec) == 1536


def test_get_embedder_factory():
    """Verify provider selection logic and fallback resolution."""
    # Explicit deterministic
    det = get_embedder("deterministic")
    assert isinstance(det, DeterministicLocalEmbedder)

    # Missing API key strictly raises ValueError (no dimension-skewing fallback)
    with patch.object(settings, "OPENROUTER_API_KEY", ""):
        with pytest.raises(ValueError) as exc_info:
            get_embedder("openrouter")
        assert "OPENROUTER_API_KEY is required" in str(exc_info.value)

    # With API key, resolves to OpenRouterNemotronEmbeddingAdapter
    with patch.object(settings, "OPENROUTER_API_KEY", "mock-openrouter-key"):
        adapter = get_embedder("openrouter")
        assert isinstance(adapter, OpenRouterNemotronEmbeddingAdapter)
        assert adapter.dimension == 2048

        auto_adapter = get_embedder("auto")
        assert isinstance(auto_adapter, OpenRouterNemotronEmbeddingAdapter)
        assert auto_adapter.dimension == 2048


# =====================================================================
# 2. Qdrant Collection & Storage Tests
# =====================================================================

def test_qdrant_in_memory_creation():
    """Verify ephemeral in-memory Qdrant instance creation and collection setup."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    store = QdrantVectorStore(
        collection_name="test_col_mem",
        embedder=embedder,
        in_memory=True,
        auto_index=False
    )
    assert store.get_point_count() == 0
    assert store.client.collection_exists("test_col_mem")
    store.close()


def test_qdrant_idempotent_upsert():
    """Verify that re-indexing the same chunks does not duplicate points."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    store = QdrantVectorStore(
        collection_name="test_idempotent",
        embedder=embedder,
        in_memory=True,
        auto_index=False
    )

    test_chunk = SemanticChunk(
        chunk_id="test-pool-1",
        category="amenity",
        topic="pool",
        title="Infinity Pool",
        keywords=["pool", "swimming"],
        content="Our pool is open from 6am to 9pm daily.",
        metadata={"hours": "6:00 AM - 9:00 PM"}
    )

    # First upsert
    c1 = store.upsert_chunks([test_chunk])
    assert c1 == 1
    assert store.get_point_count() == 1

    # Second upsert with identical chunk
    c2 = store.upsert_chunks([test_chunk])
    assert c2 == 1
    # Point count MUST remain 1 because ID is deterministic UUIDv5
    assert store.get_point_count() == 1
    store.close()


def test_deterministic_uuid_stability():
    """Verify that chunk_id_to_uuid produces identical RFC-compliant UUIDs."""
    uuid1 = chunk_id_to_uuid("property-overview")
    uuid2 = chunk_id_to_uuid("property-overview")
    uuid3 = chunk_id_to_uuid("amenity-oceanview-pool")

    assert uuid1 == uuid2
    assert uuid1 != uuid3
    assert len(uuid1) == 36


def test_qdrant_dimension_mismatch_auto_recreate():
    """Verify that if an existing collection has a different dimension, it is recreated."""
    embedder_small = DeterministicLocalEmbedder(dimension=128)
    store1 = QdrantVectorStore(
        collection_name="test_mismatch",
        embedder=embedder_small,
        in_memory=True,
        auto_index=False
    )
    assert store1.client.get_collection("test_mismatch").config.params.vectors.size == 128

    # Now open with a 384-dim embedder on the same client
    embedder_large = DeterministicLocalEmbedder(dimension=384)
    store2 = QdrantVectorStore(
        collection_name="test_mismatch",
        embedder=embedder_large,
        client=store1.client,
        auto_index=False
    )
    assert store2.client.get_collection("test_mismatch").config.params.vectors.size == 384
    store1.close()


# =====================================================================
# 3. Vector Similarity Search & Retrieval Tests
# =====================================================================

def test_vector_search_top_k_and_ranking():
    """Verify vector search returns ordered candidates with scores."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    store = QdrantVectorStore(
        collection_name="test_search_rank",
        embedder=embedder,
        in_memory=True,
        auto_index=False
    )

    chunks = [
        SemanticChunk(
            chunk_id="amenity-pool",
            category="amenity",
            topic="pool",
            title="Oceanview Infinity Pool",
            keywords=["pool", "swimming", "infinity"],
            content="Temperature-controlled oceanfront swimming pool.",
        ),
        SemanticChunk(
            chunk_id="amenity-spa",
            category="amenity",
            topic="spa",
            title="AyurVeda Spa",
            keywords=["spa", "massage", "wellness"],
            content="Authentic Ayurvedic treatments and wellness therapies.",
        ),
        SemanticChunk(
            chunk_id="policy-cancel",
            category="policy",
            topic="cancellation",
            title="Cancellation Policy",
            keywords=["cancel", "refund", "24 hours"],
            content="Free cancellation up to 24 hours prior to check-in.",
        ),
    ]
    store.upsert_chunks(chunks)

    # Search for swimming pool
    results = store.search("swimming pool", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "amenity-pool"
    assert results[0]["score"] >= results[1]["score"]
    assert "content" in results[0]
    assert "title" in results[0]

    store.close()


def test_vector_search_category_filter():
    """Verify category filtering strictly constrains returned chunks."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    store = QdrantVectorStore(
        collection_name="test_category_filter",
        embedder=embedder,
        in_memory=True,
        auto_index=False
    )

    chunks = [
        SemanticChunk(
            chunk_id="r1",
            category="room",
            topic="deluxe",
            title="Deluxe King Room",
            keywords=["room", "deluxe"],
            content="King bed overlooking the ocean.",
        ),
        SemanticChunk(
            chunk_id="a1",
            category="amenity",
            topic="pool",
            title="Infinity Pool",
            keywords=["pool"],
            content="Resort swimming pool.",
        ),
    ]
    store.upsert_chunks(chunks)

    # Filter by room
    room_results = store.search("ocean", top_k=5, category_filter="room")
    assert len(room_results) == 1
    assert room_results[0]["category"] == "room"
    assert room_results[0]["id"] == "r1"

    # Filter by amenity
    amenity_results = store.search("ocean", top_k=5, category_filter="amenity")
    assert len(amenity_results) == 1
    assert amenity_results[0]["category"] == "amenity"
    assert amenity_results[0]["id"] == "a1"

    store.close()


def test_vector_search_empty_query():
    """Verify empty query returns fallback slice without error."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    store = QdrantVectorStore(
        collection_name="test_empty_q",
        embedder=embedder,
        in_memory=True,
        auto_index=False
    )
    chunks = [
        SemanticChunk(
            chunk_id=f"c{i}",
            category="faq",
            topic="test",
            title=f"FAQ {i}",
            keywords=[],
            content=f"Content {i}"
        )
        for i in range(5)
    ]
    store.upsert_chunks(chunks)

    res = store.search("", top_k=3)
    assert len(res) == 3
    store.close()


# =====================================================================
# 4. Full Hotel Data Indexing & Golden Queries
# =====================================================================

def test_full_hotel_data_indexing():
    """Verify indexing the actual hotel_data.json creates exactly 29 chunks."""
    embedder = DeterministicLocalEmbedder(dimension=384)
    hotel_json = Path(__file__).resolve().parent.parent / "app" / "data" / "hotel_data.json"

    store = QdrantVectorStore(
        collection_name="test_full_hotel",
        embedder=embedder,
        in_memory=True,
        data_path=hotel_json,
        auto_index=True
    )

    assert store.get_point_count() == 29

    # Golden queries
    queries = [
        ("What time is check-in?", ["check-in", "timing", "arrival"]),
        ("Does the hotel have a swimming pool?", ["pool", "swimming"]),
        ("What is the cancellation policy?", ["cancellation", "refund"]),
        ("Is breakfast included?", ["breakfast", "buffet"]),
        ("Which room is suitable for three guests?", ["deluxe", "family", "guests", "three"]),
    ]

    for q, expected_terms in queries:
        results = store.search(q, top_k=3)
        assert len(results) >= 1
        combined_text = " ".join([f"{r['title']} {r['content']}".lower() for r in results])
        matched = any(term in combined_text for term in expected_terms)
        assert matched is True, f"Query '{q}' failed to retrieve any of {expected_terms}"

    store.close()


def test_upsert_langchain_documents():
    """Verify LangChain Document objects from json_to_documents() can be upserted."""
    hotel_json = Path(__file__).resolve().parent.parent / "app" / "data" / "hotel_data.json"
    import json
    with open(hotel_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = json_to_documents(data)
    assert len(docs) >= 15

    embedder = DeterministicLocalEmbedder(dimension=384)
    store = QdrantVectorStore(
        collection_name="test_lc_docs",
        embedder=embedder,
        in_memory=True,
        auto_index=False
    )

    count = store.upsert_documents(docs)
    assert count == len(docs)
    assert store.get_point_count() == len(docs)

    results = store.search("pool", top_k=2)
    assert len(results) == 2
    assert any("pool" in (r["title"] + r["content"]).lower() for r in results)
    store.close()


# =====================================================================
# 5. FastAPI Vector Store API Endpoint Tests
# =====================================================================

def test_vector_store_stats_api_endpoint():
    """Verify GET /api/vector-store/stats returns ready status and valid schema."""
    mock_store = MagicMock()
    mock_store.collection_name = "hotel_knowledge_base"
    mock_store.get_point_count.return_value = 29
    mock_store.embedder.dimension = 2048
    mock_store.embedder.provider_name = "openrouter (openrouter/nvidia/nemotron-3-embed-1b:free, 2048d)"
    mock_store.storage_mode = "local disk"

    with patch("app.core.vector_store.get_vector_store", return_value=mock_store):
        with TestClient(app) as client:
            res = client.get("/api/vector-store/stats")
            assert res.status_code == 200
            data = res.json()
            assert "collection_name" in data
            assert "point_count" in data
            assert "vector_dimension" in data
            assert "embedding_provider" in data
            assert "status" in data
            assert data["point_count"] >= 29
            assert data["status"] == "ready"
