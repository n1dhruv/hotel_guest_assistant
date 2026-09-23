import pytest
from app.core.rag import kb

def test_kb_chunk_counts():
    assert len(kb.chunks) >= 20
    categories = {c["category"] for c in kb.chunks}
    assert "property" in categories
    assert "amenity" in categories
    assert "room" in categories
    assert "policy" in categories
    assert "faq" in categories

def test_rag_pool_query():
    results = kb.retrieve("Do you have an infinity swimming pool?", k=3)
    assert len(results) == 3
    found_pool = any("pool" in (r["title"] + r["content"]).lower() for r in results)
    assert found_pool is True

def test_rag_cancellation_query():
    results = kb.retrieve("What is the cancellation policy?", k=3)
    assert len(results) == 3
    found_cancellation = any("cancellation" in (r["title"] + r["content"]).lower() for r in results)
    assert found_cancellation is True

def test_rag_breakfast_query():
    results = kb.retrieve("Is complimentary breakfast included?", k=3)
    assert len(results) == 3
    found_breakfast = any("breakfast" in (r["title"] + r["content"]).lower() for r in results)
    assert found_breakfast is True

def test_rag_id_verification_query():
    results = kb.retrieve("Which government ID is required at check-in? Can I use Aadhaar?", k=3)
    assert len(results) == 3
    found_id = any("aadhaar" in (r["title"] + r["content"]).lower() or "id" in r["title"].lower() for r in results)
    assert found_id is True

def test_rag_empty_query_fallback():
    results = kb.retrieve("", k=4)
    assert len(results) == 4
