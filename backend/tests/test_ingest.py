"""
Unit and integration tests for Structure-Aware JSON Ingestion Pipeline.
"""

import json
from pathlib import Path
import pytest
from langchain_core.documents import Document

from app.core.json_ingest import json_to_documents, maybe_split, LONG_TEXT_THRESHOLD


@pytest.fixture
def sample_hotel_data():
    data_file = Path(__file__).resolve().parent.parent / "app" / "data" / "hotel_data.json"
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_json_to_documents_structure_and_count(sample_hotel_data):
    docs = json_to_documents(sample_hotel_data)

    assert len(docs) == 29
    categories = {d.metadata.get("category") for d in docs}
    assert categories == {"property_info", "amenity", "room", "policy", "faq"}


def test_property_document(sample_hotel_data):
    docs = json_to_documents(sample_hotel_data)
    prop_docs = [d for d in docs if d.metadata.get("category") == "property_info"]

    assert len(prop_docs) == 1
    doc = prop_docs[0]
    assert "The Grand Azure Heritage Resort & Spa" in doc.page_content
    assert "Candolim Beach Road" in doc.page_content
    assert doc.metadata["property"] == "The Grand Azure Heritage Resort & Spa"
    assert doc.metadata["checkIn"] == "2:00 PM"
    assert doc.metadata["checkOut"] == "11:00 AM"


def test_amenities_documents(sample_hotel_data):
    docs = json_to_documents(sample_hotel_data)
    amenity_docs = [d for d in docs if d.metadata.get("category") == "amenity"]

    assert len(amenity_docs) == 6
    names = {d.metadata.get("name") for d in amenity_docs}
    assert "Oceanview Infinity Pool" in names
    assert "High-Speed Wi-Fi" in names
    assert "Saffron Coastal & Spice Restaurant" in names

    for doc in amenity_docs:
        assert doc.page_content.startswith("Amenity:")
        assert "Hours:" in doc.page_content
        assert len(doc.metadata.get("hours", "")) > 0


def test_rooms_documents_with_numeric_metadata(sample_hotel_data):
    docs = json_to_documents(sample_hotel_data)
    room_docs = [d for d in docs if d.metadata.get("category") == "room"]

    assert len(room_docs) == 4
    room_ids = {d.metadata.get("room_id") for d in room_docs}
    assert {"standard-queen", "deluxe-king", "family-suite", "presidential-suite"} == room_ids

    for doc in room_docs:
        # Check filterable numeric fields
        assert isinstance(doc.metadata["price"], (int, float))
        assert doc.metadata["price"] >= 4500
        assert isinstance(doc.metadata["max_guests"], int)
        assert doc.metadata["max_guests"] >= 2
        assert "Room type:" in doc.page_content
        assert "Base price:" in doc.page_content


def test_policies_documents(sample_hotel_data):
    docs = json_to_documents(sample_hotel_data)
    policy_docs = [d for d in docs if d.metadata.get("category") == "policy"]

    assert len(policy_docs) == 7
    policy_names = {d.metadata.get("policy_name") for d in policy_docs}
    expected_policies = {"cancellation", "checkInCheckOut", "idVerification", "pets", "children", "smoking", "payment"}
    assert expected_policies == policy_names

    for doc in policy_docs:
        assert doc.page_content.startswith("Policy —")
        assert "The Grand Azure" in doc.metadata["property"]


def test_faqs_documents(sample_hotel_data):
    docs = json_to_documents(sample_hotel_data)
    faq_docs = [d for d in docs if d.metadata.get("category") == "faq"]

    assert len(faq_docs) == 11
    for doc in faq_docs:
        assert doc.page_content.startswith("Q: ")
        assert "\nA: " in doc.page_content
        assert doc.metadata.get("faq_id") is not None
        assert doc.metadata.get("topic") is not None


def test_maybe_split_short_text():
    short_text = "Standard check-in begins at 2:00 PM and check-out is 11:00 AM."
    chunks = maybe_split(short_text, threshold=600)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_maybe_split_long_text_fallback():
    para1 = ("Paragraph 1: " + "This is a detailed description of the luxury spa. " * 15).strip()
    para2 = ("Paragraph 2: " + "Here are the specific Ayurvedic treatments offered. " * 15).strip()
    long_text = f"{para1}\n\n{para2}"

    assert len(long_text) > 600
    chunks = maybe_split(long_text, semantic_splitter=None, threshold=600)
    assert len(chunks) == 2
    assert chunks[0] == para1
    assert chunks[1] == para2


def test_injectable_embedding_model_support(sample_hotel_data):
    class MockEmbedder:
        def embed_documents(self, texts):
            return [[0.1] * 384 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 384

    mock_model = MockEmbedder()
    docs = json_to_documents(sample_hotel_data, embedding_model=mock_model)
    assert len(docs) == 29
    assert all(isinstance(d, Document) for d in docs)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
