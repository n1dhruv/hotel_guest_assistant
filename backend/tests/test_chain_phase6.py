"""Phase 6: end-to-end LLM generation & frontend integration tests.

Verifies the full RAG loop wiring with all external I/O mocked:
HyDE -> Hybrid Top 10 -> Rerank Top 3 -> Gemini (LiteLLM) + tool calling,
latency breakdown, source citations, and SSE streaming events.
"""
import asyncio
from types import SimpleNamespace

import pytest

import app.core.chain as chain_mod
from app.core.chain import (
    AssistantOrchestrator,
    build_system_prompt,
    detect_booking_intent,
    retrieve_full_pipeline,
)
from app.config import settings


def _chunk(i, category="policy", title=None):
    return {
        "id": f"chunk-{i}",
        "category": category,
        "title": title or f"Policy: chunk {i}",
        "content": f"Resort fact content for chunk {i}.",
        "score": 10.0 - i,
        "rrf_score": 0.03 - i * 0.001,
        "rrf_rank": i + 1,
    }


class _FakeRetriever:
    def __init__(self, n=10):
        self.calls = []
        self.n = n

    async def aretrieve_candidates(self, query, top_k=None, use_hyde=None, **kwargs):
        self.calls.append({"query": query, "top_k": top_k, "use_hyde": use_hyde})
        return [_chunk(i) for i in range(self.n)]

    def retrieve_candidates(self, query, top_k=None, use_hyde=None, **kwargs):
        return [_chunk(i) for i in range(self.n or 10)]


class _FakeReranker:
    def __init__(self):
        self.calls = []

    async def arerank(self, query, candidates, top_n=None):
        self.calls.append({"query": query, "n_candidates": len(candidates), "top_n": top_n})
        out = []
        for rank, c in enumerate(list(candidates)[: (top_n or 3)], start=1):
            d = dict(c)
            d["rerank_score"] = 0.9 - rank * 0.01
            d["rerank_rank"] = rank
            d["rerank_delta"] = d.get("rrf_rank", rank) - rank
            out.append(d)
        return out


def _fake_text_completion(text):
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_system_prompt_persona_grounding_and_fallback():
    chunks = [_chunk(0), _chunk(1, category="amenity", title="Amenity: Infinity Pool")]
    prompt = build_system_prompt(chunks)
    assert "Grand Azure" in prompt
    assert "concierge" in prompt.lower()
    assert "+91 832 249 8000" in prompt  # out-of-domain fallback contact
    assert "chunk 0" in prompt  # reranked context injected


def test_detect_booking_intent_extracts_dates_and_adults():
    intent = detect_booking_intent("Are there rooms available from 2026-10-15 to 2026-10-18 for 2 adults?")
    assert intent["is_intent"] is True
    assert intent["dates"] == ["2026-10-15", "2026-10-18"]
    assert intent["adults"] == 2


def test_detect_booking_intent_no_dates_no_tool():
    intent = detect_booking_intent("Does the resort have an infinity pool?")
    assert intent["dates"] == []


def test_full_pipeline_hybrid_top10_rerank_top3(monkeypatch):
    fake_retriever, fake_reranker = _FakeRetriever(n=10), _FakeReranker()
    monkeypatch.setattr("app.core.hybrid_retriever.get_hybrid_retriever", lambda: fake_retriever)
    monkeypatch.setattr("app.core.reranker.get_reranker", lambda: fake_reranker)

    reranked, breakdown, pipeline = asyncio.run(
        retrieve_full_pipeline("What is the cancellation policy?")
    )
    # Hybrid asked for Top 10 candidates...
    assert fake_retriever.calls and fake_retriever.calls[0]["top_k"] == 10
    # ...reranker received all 10 and returned Top 3
    assert fake_reranker.calls[0]["n_candidates"] == 10
    assert len(reranked) == 3
    assert all("rerank_score" in c and "rerank_rank" in c for c in reranked)
    # Latency breakdown covers every stage
    assert set(breakdown) == {"hyde_ms", "search_ms", "rerank_ms"}
    assert pipeline["llm_model"] == settings.LLM_MODEL  # Gemini model from .env


def test_execute_uses_gemini_model_and_citations(monkeypatch):
    fake_retriever, fake_reranker = _FakeRetriever(n=10), _FakeReranker()
    monkeypatch.setattr("app.core.hybrid_retriever.get_hybrid_retriever", lambda: fake_retriever)
    monkeypatch.setattr("app.core.reranker.get_reranker", lambda: fake_reranker)
    monkeypatch.setattr(settings, "MOCK_LLM", False)

    seen = {}

    async def _fake_acompletion(model, messages, **kwargs):
        seen["model"] = model
        assert any("Grand Azure" in m.get("content", "") for m in messages)  # grounded prompt
        return _fake_text_completion("Namaste! Our infinity pool is open 6 AM to 9 PM.")

    monkeypatch.setattr(chain_mod, "acompletion", _fake_acompletion)

    orch = AssistantOrchestrator()
    result = asyncio.run(orch.execute([{"role": "user", "content": "Pool timings?"}]))
    assert seen["model"] == settings.LLM_MODEL  # Gemini model from .env, not hardcoded
    assert "pool" in result["reply"].lower()
    assert result["tool_called"] is False
    assert len(result["retrieved_sources"]) == 3  # reranked Top 3 citations
    assert len(result["sources"]) == 3
    assert set(result["latency_breakdown"]) == {"hyde_ms", "search_ms", "rerank_ms", "generation_ms"}


def test_execute_availability_tool_path(monkeypatch):
    fake_retriever, fake_reranker = _FakeRetriever(n=10), _FakeReranker()
    monkeypatch.setattr("app.core.hybrid_retriever.get_hybrid_retriever", lambda: fake_retriever)
    monkeypatch.setattr("app.core.reranker.get_reranker", lambda: fake_reranker)
    monkeypatch.setattr(settings, "MOCK_LLM", False)

    tool_msg = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(id="call_1", function=SimpleNamespace(
            name="check_room_availability",
            arguments='{"checkIn": "2026-10-15", "checkOut": "2026-10-18", "adults": 2}',
        ))],
    )
    calls = {"n": 0}

    async def _fake_acompletion(model, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            assert kwargs.get("tools")  # tool definition offered to Gemini
            return SimpleNamespace(choices=[SimpleNamespace(message=tool_msg)])
        return _fake_text_completion("Namaste! 4 room tiers are available for your dates.")

    monkeypatch.setattr(chain_mod, "acompletion", _fake_acompletion)

    orch = AssistantOrchestrator()
    result = asyncio.run(orch.execute([
        {"role": "user", "content": "Are there rooms available from 2026-10-15 to 2026-10-18 for 2 adults?"}
    ]))
    assert result["tool_called"] is True
    assert result["availability"] is not None
    assert result["availability"]["available"] is True
    assert result["availability"]["nights"] == 3
    assert len(result["availability"]["rooms"]) >= 1  # room cards payload for frontend


def test_execute_stream_token_then_metadata(monkeypatch):
    fake_retriever, fake_reranker = _FakeRetriever(n=10), _FakeReranker()
    monkeypatch.setattr("app.core.hybrid_retriever.get_hybrid_retriever", lambda: fake_retriever)
    monkeypatch.setattr("app.core.reranker.get_reranker", lambda: fake_reranker)
    monkeypatch.setattr(settings, "MOCK_LLM", False)

    async def _fake_acompletion(model, messages, **kwargs):
        assert kwargs.get("stream") is True

        async def _gen():
            for tok in ["Namaste! ", "Pool open 6 AM-9 PM."]:
                yield SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=tok, tool_calls=None))])

        return _gen()

    monkeypatch.setattr(chain_mod, "acompletion", _fake_acompletion)

    async def _collect():
        orch = AssistantOrchestrator()
        events = []
        async for ev in orch.execute_stream([{"role": "user", "content": "Pool timings?"}]):
            events.append(ev)
        return events

    events = asyncio.run(_collect())
    tokens = [e for e in events if e["type"] == "token"]
    metas = [e for e in events if e["type"] == "metadata"]
    assert len(tokens) == 2 and "".join(t["token"] for t in tokens) == "Namaste! Pool open 6 AM-9 PM."
    assert len(metas) == 1
    assert len(metas[0]["retrieved_sources"]) == 3
    assert set(metas[0]["latency_breakdown"]) == {"hyde_ms", "search_ms", "rerank_ms", "generation_ms"}
