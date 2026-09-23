"""
Structure-Aware JSON Ingestion Pipeline for RAG.

Converts structured hotel/resort JSON data into LangChain Documents
suitable for vector embedding and retrieval, preserving semantic coherence per entity
and attaching queryable metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional
import warnings

from langchain_core.documents import Document

# Suppress deprecation warning from langchain-experimental
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_experimental")

LONG_TEXT_THRESHOLD = 600  # chars — below this, do not sub-chunk


def maybe_split(
    text: str,
    semantic_splitter: Optional[Any] = None,
    threshold: int = LONG_TEXT_THRESHOLD
) -> List[str]:
    """
    Conditionally split text using SemanticChunker only if it exceeds the length threshold.
    Avoids over-chunking short, semantically atomic fields.
    """
    if len(text) < threshold:
        return [text]

    if semantic_splitter is not None:
        try:
            chunks = semantic_splitter.create_documents([text])
            return [d.page_content for d in chunks if d.page_content.strip()]
        except Exception:
            pass

    # Fallback to paragraph or sentence boundary splitting if no semantic splitter is active
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    return [text]


def get_default_semantic_splitter(embedding_model: Optional[Any] = None) -> Optional[Any]:
    """
    Initializes a SemanticChunker with an injectable embedding model.
    If no model is provided, attempts to load OpenAI or Gemini if configured in settings.
    """
    if embedding_model is not None:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            return SemanticChunker(
                embeddings=embedding_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95
            )
        except Exception:
            return None

    # Try loading from app settings if available
    try:
        from app.config import settings
        if settings.OPENAI_API_KEY and not settings.MOCK_LLM:
            from langchain_openai import OpenAIEmbeddings
            from langchain_experimental.text_splitter import SemanticChunker
            embedder = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model="text-embedding-3-small"
            )
            return SemanticChunker(
                embeddings=embedder,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95
            )
    except Exception:
        pass

    return None


def json_to_documents(
    data: dict[str, Any],
    embedding_model: Optional[Any] = None,
    long_text_threshold: int = LONG_TEXT_THRESHOLD
) -> List[Document]:
    """
    Converts structured hotel JSON data into a list of LangChain Document objects.
    - Flattens entity fields into natural prose (not raw JSON dumps).
    - Preserves logical entity boundaries (property, amenity, room, policy, FAQ).
    - Attaches rich, filterable metadata.
    - Applies conditional semantic sub-chunking only for fields exceeding long_text_threshold.
    """
    docs: List[Document] = []
    prop_data = data.get("property", {})
    property_name = prop_data.get("name", "The Grand Azure Heritage Resort & Spa")

    # Initialize semantic splitter if embedding model is supplied or available
    splitter = get_default_semantic_splitter(embedding_model)

    # 1. Property Overview
    contact = prop_data.get("contact", {})
    prop_text = (
        f"{property_name} — {prop_data.get('tagline', '')}. "
        f"Located at {prop_data.get('address', '')}. "
        f"Check-in: {prop_data.get('checkIn', '')}, Check-out: {prop_data.get('checkOut', '')}. "
        f"Contact: Phone {contact.get('phone', '')} / Mobile {contact.get('mobile', '')} / Email {contact.get('email', '')}."
    )
    docs.append(Document(
        page_content=prop_text,
        metadata={
            "category": "property_info",
            "property": property_name,
            "address": prop_data.get("address", ""),
            "checkIn": prop_data.get("checkIn", ""),
            "checkOut": prop_data.get("checkOut", ""),
            "phone": contact.get("phone", ""),
            "email": contact.get("email", "")
        }
    ))

    # 2. Amenities: One Document per amenity (sub-split only if exceeding threshold)
    for a in data.get("amenities", []):
        name = a.get("name", "")
        hours = a.get("hours", "")
        desc = a.get("description", "")
        text = f"Amenity: {name}. Hours: {hours}. {desc}"

        chunks = maybe_split(text, splitter, threshold=long_text_threshold)
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "category": "amenity",
                    "name": name,
                    "hours": hours,
                    "property": property_name
                }
            ))

    # 3. Rooms: One Document per room type with filterable numeric metadata
    for r in data.get("rooms", []):
        rtype = r.get("type", "")
        rid = r.get("id", "")
        price = r.get("basePricePerNight", 0)
        max_guests = r.get("maxGuests", 2)
        bed_type = r.get("bedType", "")
        desc = r.get("description", "")

        text = (
            f"Room type: {rtype} (ID: {rid}). Max guests: {max_guests}. "
            f"Base price: ₹{price:,}/night. Bed type: {bed_type}. {desc}"
        )

        chunks = maybe_split(text, splitter, threshold=long_text_threshold)
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "category": "room",
                    "room_id": rid,
                    "room_type": rtype,
                    "price": price,
                    "max_guests": max_guests,
                    "bed_type": bed_type,
                    "property": property_name
                }
            ))

    # 4. Policies: One Document per policy rule
    policy_display_names = {
        "cancellation": "Cancellation & Refund",
        "checkInCheckOut": "Check-in and Check-out Timings",
        "idVerification": "ID Verification & Documentation",
        "pets": "Pet Policy",
        "children": "Child Occupancy & Extra Bed",
        "smoking": "Smoking Policy",
        "payment": "Payment & Invoicing"
    }

    for policy_key, policy_text in data.get("policies", {}).items():
        display_name = policy_display_names.get(policy_key, policy_key.capitalize())
        chunks = maybe_split(policy_text, splitter, threshold=long_text_threshold)
        for chunk in chunks:
            docs.append(Document(
                page_content=f"Policy — {display_name}: {chunk}",
                metadata={
                    "category": "policy",
                    "policy_name": policy_key,
                    "display_name": display_name,
                    "property": property_name
                }
            ))

    # 5. FAQs: One Document per FAQ (already concise topic/answer units)
    for faq in data.get("faqs", []):
        fid = faq.get("id", "")
        topic = faq.get("topic", "")
        answer = faq.get("answer", faq.get("a", ""))
        text = f"Q: {topic}\nA: {answer}"

        docs.append(Document(
            page_content=text,
            metadata={
                "category": "faq",
                "faq_id": fid,
                "topic": topic,
                "property": property_name
            }
        ))

    return docs


if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / "data" / "hotel_data.json"

    if not data_file.exists():
        data_file = Path("backend/app/data/hotel_data.json")

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    documents = json_to_documents(raw_data)
    print(f"Successfully created {len(documents)} structured Documents.\n")
    for idx, d in enumerate(documents[:5], start=1):
        print(f"--- Document #{idx} [{d.metadata.get('category')}] ---")
        print(f"Metadata: {d.metadata}")
        print(f"Content: {d.page_content}\n")
