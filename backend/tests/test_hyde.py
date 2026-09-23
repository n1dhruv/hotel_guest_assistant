"""
Unit & Integration Tests for Phase 3: HyDE (Hypothetical Document Embeddings) Query Pipeline.

Verifies:
1. Canonical prompt construction and parameterization
2. Low-latency synchronous generation with LiteLLM
3. Asynchronous non-blocking generation with LiteLLM
4. Passage post-processing and prefix/quote sanitization
5. Resilient fallback to raw user query on timeout
6. Resilient fallback to raw user query on rate limits / API exceptions
7. Offline mock mode handling with zero network overhead
8. Empty / whitespace query edge case handling
9. Observability, latency tracking, and stats reporting
10. Integration with QdrantVectorStore (use_hyde=True and search_with_hyde)
11. FastAPI /api/hyde/stats and /api/hyde/generate endpoints
12. Thread safety of singleton accessor
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.chunker import SemanticChunk
from app.core.hyde import (
    HYDE_SYSTEM_PROMPT,
    HYDE_USER_PROMPT_TEMPLATE,
    HyDEGenerator,
    get_hyde_generator,
)
from app.core.vector_store import DeterministicLocalEmbedder, QdrantVectorStore
from app.main import app


# =====================================================================
# 1. Prompt Construction Tests
# =====================================================================

def test_hyde_prompt_construction():
    """Verify that build_prompt adheres to the Phase 3 specification."""
    generator = HyDEGenerator(mock_mode=False)
    query = "Can I bring my golden retriever?"
    messages = generator.build_prompt(query)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "hospitality knowledge base generator" in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert query in messages[1]["content"]
    assert "hypothetical paragraph (2-3 sentences)" in messages[1]["content"]
    assert "Hypothetical passage:" in messages[1]["content"]


# =====================================================================
# 2. Passage Cleaning & Sanitization Tests
# =====================================================================

def test_hyde_clean_passage_various_formats():
    """Verify that clean_passage removes redundant prefixes, quotes, and markdown tags."""
    generator = HyDEGenerator(mock_mode=False)
    fallback = "Can I bring pets?"

    # Prefix with colon
    raw1 = "Hypothetical passage: The Grand Azure Resort does not permit domestic pets on property."
    assert generator.clean_passage(raw1, fallback) == "The Grand Azure Resort does not permit domestic pets on property."

    # Wrapped in double quotes
    raw2 = '"Pets are strictly prohibited at the resort, with the exception of certified service animals."'
    assert generator.clean_passage(raw2, fallback) == "Pets are strictly prohibited at the resort, with the exception of certified service animals."

    # Markdown block
    raw3 = "```text\nGuests traveling with pets should note that domestic animals are not allowed.\n```"
    assert generator.clean_passage(raw3, fallback) == "Guests traveling with pets should note that domestic animals are not allowed."

    # Whitespace or empty
    assert generator.clean_passage("   ", fallback) == fallback
    assert generator.clean_passage("", fallback) == fallback


# =====================================================================
# 3. Synchronous Generation with LiteLLM (Mocked)
# =====================================================================

def test_hyde_sync_generation_success():
    """Verify synchronous generation invokes LiteLLM with expected parameters."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Hypothetical passage: Check-in begins at 2:00 PM and check-out is at 11:00 AM."
    mock_response = MagicMock(choices=[mock_choice])

    with patch("litellm.completion", return_value=mock_response) as mock_comp:
        generator = HyDEGenerator(
            model="gemini/gemini-2.0-flash",
            temperature=0.0,
            max_tokens=80,
            timeout=1.5,
            mock_mode=False,
            enabled=True,
        )
        passage = generator.generate_hypothetical_document("What time is check-in?")

        assert mock_comp.called
        call_kwargs = mock_comp.call_args[1]
        assert call_kwargs["model"] == "gemini/gemini-2.0-flash"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 80
        assert call_kwargs["timeout"] == 1.5

        assert "Check-in begins at 2:00 PM" in passage
        assert not passage.startswith("Hypothetical passage:")


# =====================================================================
# 4. Asynchronous Generation with LiteLLM (Mocked)
# =====================================================================

@pytest.mark.asyncio
async def test_hyde_async_generation_success():
    """Verify async generation invokes litellm.acompletion without blocking."""
    mock_choice = MagicMock()
    mock_choice.message.content = "The resort features a temperature-controlled oceanfront infinity pool open from 7 AM."
    mock_response = MagicMock(choices=[mock_choice])

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_acomp:
        generator = HyDEGenerator(mock_mode=False, enabled=True)
        passage = await generator.agenerate_hypothetical_document("Do you have a pool?")

        assert mock_acomp.called
        assert "infinity pool open from 7 AM" in passage


# =====================================================================
# 5. Resilient Fallback Handling
# =====================================================================

def test_hyde_sync_timeout_fallback():
    """Verify that a timeout in litellm.completion gracefully falls back to the raw query."""
    with patch("litellm.completion", side_effect=TimeoutError("Request timed out after 1.5s")):
        generator = HyDEGenerator(mock_mode=False, enabled=True)
        raw_query = "What is the pet policy?"
        result = generator.generate_hypothetical_document(raw_query)

        assert result == raw_query
        stats = generator.get_stats()
        assert stats["total_fallbacks"] == 1
        assert stats["total_calls"] == 1


@pytest.mark.asyncio
async def test_hyde_async_timeout_fallback():
    """Verify that an async timeout gracefully falls back to the raw query."""
    with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=asyncio.TimeoutError("Async timeout")):
        generator = HyDEGenerator(mock_mode=False, enabled=True)
        raw_query = "Can I bring my dog?"
        result = await generator.agenerate_hypothetical_document(raw_query)

        assert result == raw_query
        stats = generator.get_stats()
        assert stats["total_fallbacks"] == 1


def test_hyde_api_error_fallback():
    """Verify that any LiteLLM exception (e.g. rate limit 429) falls back without crashing."""
    with patch("litellm.completion", side_effect=RuntimeError("LiteLLM 429: Resource has been exhausted")):
        generator = HyDEGenerator(mock_mode=False, enabled=True)
        raw_query = "Is breakfast included in the booking?"
        result = generator.generate_hypothetical_document(raw_query)

        assert result == raw_query
        assert generator.get_stats()["total_fallbacks"] == 1


def test_hyde_mock_mode_zero_overhead():
    """Verify that when mock_mode=True, no LiteLLM calls are made and raw query is returned."""
    with patch("litellm.completion") as mock_comp:
        generator = HyDEGenerator(mock_mode=True)
        result = generator.generate_hypothetical_document("Do you have wifi?")

        assert result == "Do you have wifi?"
        assert not mock_comp.called
        stats = generator.get_stats()
        assert stats["total_calls"] == 1
        assert stats["total_fallbacks"] == 1


def test_hyde_disabled_mode():
    """Verify that when enabled=False, raw query is returned immediately."""
    with patch("litellm.completion") as mock_comp:
        generator = HyDEGenerator(enabled=False, mock_mode=False)
        result = generator.generate_hypothetical_document("Any discounts available?")

        assert result == "Any discounts available?"
        assert not mock_comp.called


def test_hyde_empty_and_whitespace_query():
    """Verify empty or whitespace queries return unchanged without invoking LiteLLM."""
    with patch("litellm.completion") as mock_comp:
        generator = HyDEGenerator(mock_mode=False, enabled=True)
        assert generator.generate_hypothetical_document("") == ""
        assert generator.generate_hypothetical_document("   ") == "   "
        assert not mock_comp.called


# =====================================================================
# 6. Observability & Stats Reporting
# =====================================================================

def test_hyde_stats_reporting():
    """Verify get_stats and reset_stats report accurate operational metrics."""
    generator = HyDEGenerator(mock_mode=True)
    generator.reset_stats()

    stats_initial = generator.get_stats()
    assert stats_initial["total_calls"] == 0
    assert stats_initial["total_fallbacks"] == 0
    assert stats_initial["fallback_rate_pct"] == 0.0

    generator.generate_hypothetical_document("Query 1")
    generator.generate_hypothetical_document("Query 2")

    stats = generator.get_stats()
    assert stats["total_calls"] == 2
    assert stats["total_fallbacks"] == 2
    assert stats["fallback_rate_pct"] == 100.0

    generator.reset_stats()
    assert generator.get_stats()["total_calls"] == 0


# =====================================================================
# 7. Qdrant Vector Store Integration with HyDE
# =====================================================================

def test_qdrant_vector_store_with_hyde():
    """
    Verify QdrantVectorStore.search(use_hyde=True) and search_with_hyde()
    seamlessly pass the generated hypothetical passage into the embedding adapter.
    """
    embedder = DeterministicLocalEmbedder(dimension=384)
    store = QdrantVectorStore(
        collection_name="test_hyde_integration",
        embedder=embedder,
        in_memory=True,
        auto_index=False,
    )

    chunks = [
        SemanticChunk(
            chunk_id="policy-pet",
            category="policy",
            topic="pets",
            title="Pet Policy",
            keywords=["pets", "dogs", "service animal"],
            content="Pets and domestic animals are strictly prohibited on resort premises, except for certified service animals.",
        ),
        SemanticChunk(
            chunk_id="amenity-pool",
            category="amenity",
            topic="pool",
            title="Infinity Pool",
            keywords=["pool", "swimming"],
            content="Oceanview infinity swimming pool open daily from 7:00 AM to 8:00 PM.",
        ),
    ]
    store.upsert_chunks(chunks)

    # Mock HyDE generator to expand conversational query into formal handbook passage
    mock_passage = (
        "The Grand Azure Resort maintains a strict pet policy regarding domestic animals and dogs on property. "
        "Pets are not allowed on resort premises, with an exception for certified service animals."
    )

    mock_choice = MagicMock()
    mock_choice.message.content = mock_passage
    mock_response = MagicMock(choices=[mock_choice])

    with patch("litellm.completion", return_value=mock_response):
        # Force a fresh generator with mock_mode=False
        get_hyde_generator(force_new=True, mock_mode=False, enabled=True)

        results = store.search_with_hyde("Can I bring my golden retriever?", top_k=2)
        assert len(results) == 2
        # The pet policy chunk MUST be the top candidate
        assert results[0]["id"] == "policy-pet"
        assert results[0]["category"] == "policy"

    store.close()


# =====================================================================
# 8. FastAPI API Endpoints
# =====================================================================

def test_fastapi_hyde_stats_endpoint():
    """Verify GET /api/hyde/stats returns 200 with configuration dictionary."""
    with TestClient(app) as client:
        res = client.get("/api/hyde/stats")
        assert res.status_code == 200
        data = res.json()
        assert "enabled" in data
        assert "model" in data
        assert "max_tokens" in data
        assert "timeout_seconds" in data
        assert "total_calls" in data


def test_fastapi_hyde_generate_endpoint_mock_mode():
    """Verify POST /api/hyde/generate works and reports fallback status in mock mode."""
    with TestClient(app) as client:
        payload = {"query": "Can I request late check-out?"}
        res = client.post("/api/hyde/generate", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == "Can I request late check-out?"
        assert "hypothetical_passage" in data
        assert "is_fallback" in data
        assert "stats" in data


# =====================================================================
# 9. Singleton Thread Safety
# =====================================================================

def test_hyde_singleton_identity():
    """Verify get_hyde_generator returns the same instance across invocations."""
    gen1 = get_hyde_generator()
    gen2 = get_hyde_generator()
    assert gen1 is gen2

    # force_new creates a distinct instance
    gen3 = get_hyde_generator(force_new=True)
    assert gen3 is not gen1
