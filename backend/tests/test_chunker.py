import pytest
from pathlib import Path
from app.core.chunker import SemanticHotelChunker, SemanticChunk
from app.config import settings

def test_semantic_chunker_loads_all_categories():
    chunker = SemanticHotelChunker(settings.HOTEL_DATA_PATH)
    chunks = chunker.build_chunks()

    assert len(chunks) > 0
    categories = {c.category for c in chunks}
    assert "property" in categories
    assert "amenity" in categories
    assert "room" in categories
    assert "policy" in categories
    assert "faq" in categories


def test_semantic_chunker_schema_integrity():
    chunker = SemanticHotelChunker(settings.HOTEL_DATA_PATH)
    chunks = chunker.build_chunks()

    for c in chunks:
        assert isinstance(c, SemanticChunk)
        assert c.chunk_id and isinstance(c.chunk_id, str)
        assert c.category in {"property", "amenity", "room", "policy", "faq"}
        assert c.topic and isinstance(c.topic, str)
        assert c.title and isinstance(c.title, str)
        assert isinstance(c.keywords, list) and len(c.keywords) > 0
        assert isinstance(c.content, str) and len(c.content.strip()) > 20
        assert isinstance(c.metadata, dict)
        assert "section" in c.metadata

        # Dict conversion compatibility
        d = c.to_dict()
        assert d["id"] == c.chunk_id
        assert d["title"] == c.title
        assert d["content"] == c.content


def test_semantic_chunker_document_text():
    chunker = SemanticHotelChunker(settings.HOTEL_DATA_PATH)
    chunks = chunker.build_chunks()

    for c in chunks:
        doc_text = c.document_text
        assert c.title in doc_text
        assert c.content in doc_text
        assert "Keywords:" in doc_text


def test_semantic_chunker_specific_policies():
    chunker = SemanticHotelChunker(settings.HOTEL_DATA_PATH)
    chunks = chunker.build_chunks()

    policy_map = {c.topic: c for c in chunks if c.category == "policy"}

    # Verify cancellation policy details
    assert "cancellation" in policy_map
    canc_content = policy_map["cancellation"].content.lower()
    assert "24 hours" in canc_content
    assert "first night" in canc_content

    # Verify pet policy details
    assert "pets" in policy_map
    pet_content = policy_map["pets"].content.lower()
    assert "not allowed" in pet_content
    assert "service animal" in pet_content

    # Verify ID verification details
    assert "id_verification" in policy_map
    id_content = policy_map["id_verification"].content.lower()
    assert "pan" in id_content
    assert "aadhaar" in id_content


def test_semantic_chunker_rooms():
    chunker = SemanticHotelChunker(settings.HOTEL_DATA_PATH)
    chunks = chunker.build_chunks()

    rooms = [c for c in chunks if c.category == "room"]
    assert len(rooms) == 4

    room_ids = {r.metadata.get("room_id") for r in rooms}
    assert {"standard-queen", "deluxe-king", "family-suite", "presidential-suite"}.issubset(room_ids)

    for r in rooms:
        assert r.metadata["base_price"] > 0
        assert r.metadata["max_guests"] >= 2
        assert "Base tariff:" in r.content
