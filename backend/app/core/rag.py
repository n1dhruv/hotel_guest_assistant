import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np

from app.config import settings

BASE_DIR = Path(__file__).resolve().parent.parent
HOTEL_DATA_FILE = BASE_DIR / "data" / "hotel_data.json"

STOPWORDS = {
    "the", "is", "at", "which", "on", "a", "an", "this", "that", "to", "of",
    "for", "with", "does", "do", "you", "have", "can", "what", "where", "how",
    "are", "about", "hotel", "resort", "grand", "azure", "tell", "me", "any",
    "please", "i", "we", "my", "our", "would", "like", "if", "yes", "no", "so",
    "when", "why", "who", "want", "need", "could", "should", "there", "also"
}

SYNONYMS = {
    "cancel": ["cancellation", "cancellations", "cancelling", "cancelled", "refund"],
    "cancellation": ["cancel", "cancelling", "cancelled", "refund"],
    "cancelling": ["cancel", "cancellation"],
    "reservation": ["booking", "bookings", "reserve", "reservations"],
    "reserve": ["reservation", "booking", "reservations"],
    "booking": ["reservation", "reserve", "bookings"],
    "bookings": ["reservation", "booking"],
    "dog": ["pets"], "dogs": ["pets"], "cat": ["pets"], "cats": ["pets"], "pet": ["pets"],
    "smoke": ["smoking"], "smoking": ["smoke"], "cig": ["smoking"], "cigarettes": ["smoking"], "cigarette": ["smoking"],
    "wifi": ["wi-fi", "internet", "fiber"], "internet": ["wifi", "wi-fi", "fiber"],
    "breakfast": ["buffet", "morning", "dining", "food", "royal"],
    "food": ["breakfast", "restaurant", "dining", "coastal", "spice", "vegetarian", "jain"],
    "dinner": ["restaurant", "dining", "food"], "lunch": ["restaurant", "dining", "food"],
    "vegetarian": ["veg", "pure", "jain", "saffron", "restaurant", "food"],
    "jain": ["vegetarian", "veg", "saffron", "restaurant", "food"],
    "location": ["address", "candolim", "located"], "located": ["location", "address", "candolim"],
    "address": ["location", "candolim", "located"], "reach": ["location", "address", "candolim"],
    "pool": ["swimming", "infinity", "pool", "temperature"], "swimming": ["pool", "infinity"],
    "gym": ["fitness", "spa", "workout"], "workout": ["fitness", "gym", "cardio"],
    "spa": ["massage", "wellness", "fitness", "ayurveda"],
    "id": ["aadhaar", "passport", "identity", "verification", "voter"],
    "aadhaar": ["id", "identity", "verification", "passport"],
    "document": ["id", "aadhaar", "passport", "verification"],
    "documents": ["id", "aadhaar", "passport", "verification"],
    "parking": ["valet", "car", "vehicle", "ev", "charging"],
    "car": ["parking", "valet", "ev"], "vehicle": ["parking", "valet", "ev"],
    "ev": ["parking", "valet", "charging", "tata"],
    "pay": ["payment", "upi", "card", "billing", "rupay", "visa"],
    "payment": ["pay", "upi", "card", "rupay", "visa", "gst"],
    "checkin": ["arrival", "arrive", "time", "timings", "timing", "hours"],
    "checkout": ["departure", "leave", "time", "timings", "timing", "hours"],
    "timing": ["timings", "hours", "time"],
    "timings": ["timing", "hours", "time"],
    "people": ["guests", "persons", "adults", "capacity"],
    "person": ["guest", "adult"],
    "children": ["kids", "child", "bedding"],
    "kids": ["children", "child", "bedding"],
}

def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\bcheck[\s-]+in\b", "checkin", t)
    t = re.sub(r"\bcheck[\s-]+out\b", "checkout", t)
    t = re.sub(r"\bwi[\s-]+fi\b", "wifi", t)
    return t

class HotelKnowledgeBase:
    """
    RAG-Lite In-Memory Vector Store and BM25/Cosine Retriever.
    
    Architecture Design:
    - At startup, loads and segments hotel ground truth into semantic chunks.
    - Dual-mode embedding support:
        1. Dense OpenAI Embeddings (text-embedding-3-small) if API key is present.
        2. High-precision BM25 lexical retriever with stopword elimination, compound phrase
           normalization, morphological suffix stripping, and title weighting.
    - Ensures that regardless of LLM credentials, semantic retrieval yields accurate, grounded facts.
    """
    def __init__(self, data_path: Path = HOTEL_DATA_FILE):
        self.data_path = data_path
        self.chunks: list[dict[str, Any]] = []
        self.dense_embeddings: list[list[float]] | None = None
        self._load_and_chunk()
        self._init_retriever()

    def _load_and_chunk(self):
        """Chunks hotel data into fine-grained, self-contained semantic units."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Canonical topic maps
        amenity_topics = ["pool", "spa_gym", "breakfast", "wifi", "parking", "dining"]
        policy_topics = {
            "cancellation": "cancellation",
            "checkInCheckOut": "checkincheckout",
            "idVerification": "id_verification",
            "pets": "pets",
            "children": "children",
            "smoking": "smoking",
            "payment": "payment"
        }

        # 1. Property Overview Chunk
        prop = data.get("property", {})
        self.chunks.append({
            "id": "property-overview",
            "category": "property",
            "topic": "property",
            "title": f"About {prop.get('name')}",
            "content": (
                f"{prop.get('name')} - {prop.get('tagline')}. Located at {prop.get('address')}. "
                f"Location is Candolim Beach, North Goa. Standard checkin time is {prop.get('checkIn')} "
                f"and checkout time is {prop.get('checkOut')}. "
                f"Contact numbers: {prop.get('contact', {}).get('phone')}, Mobile: {prop.get('contact', {}).get('mobile')}, "
                f"Email: {prop.get('contact', {}).get('email')}."
            )
        })

        # 2. Amenity Chunks
        for i, a in enumerate(data.get("amenities", [])):
            topic = amenity_topics[i] if i < len(amenity_topics) else f"amenity_{i}"
            self.chunks.append({
                "id": f"amenity-{i}-{a.get('name', '').lower().replace(' ', '-')}",
                "category": "amenity",
                "topic": topic,
                "title": f"Amenity: {a.get('name')}",
                "content": f"{a.get('name')} | Timings: {a.get('hours')}. Description: {a.get('description')}"
            })

        # 3. Room Tier Chunks
        for r in data.get("rooms", []):
            self.chunks.append({
                "id": f"room-{r.get('id')}",
                "category": "room",
                "topic": f"room_{r.get('id')}",
                "title": f"Room: {r.get('type')}",
                "content": (
                    f"{r.get('type')} (ID: {r.get('id')}) | Maximum Capacity: {r.get('maxGuests')} guests. "
                    f"Base Tariff: ₹{r.get('basePricePerNight')} per night. "
                    f"Bedding: {r.get('bedType')}. Features: {r.get('description')}"
                )
            })

        # 4. Policy Chunks
        policies = data.get("policies", {})
        policy_display_names = {
            "cancellation": "Cancellation & Refund",
            "checkInCheckOut": "Check-in and Check-out Timings",
            "idVerification": "ID Verification & Documentation",
            "pets": "Pet Policy",
            "children": "Child Occupancy & Extra Bed",
            "smoking": "Smoking Policy",
            "payment": "Payment & Invoicing"
        }
        for pol_key, pol_val in policies.items():
            topic = policy_topics.get(pol_key, f"policy_{pol_key}")
            display_name = policy_display_names.get(pol_key, pol_key.capitalize())
            self.chunks.append({
                "id": f"policy-{pol_key}",
                "category": "policy",
                "topic": topic,
                "title": f"Policy: {display_name}",
                "content": f"The Grand Azure Resort {display_name} Policy: {pol_val}"
            })

        # 5. Curated FAQ Chunks (Stored as topics and factual answers, without questions)
        faq_topic_keys = {
            "faq-1": "checkincheckout",
            "faq-2": "pool",
            "faq-3": "breakfast",
            "faq-4": "room_recommendation",
            "faq-5": "cancellation",
            "faq-6": "parking",
            "faq-7": "pets",
            "faq-8": "id_verification",
            "faq-9": "dining",
            "faq-10": "highlights",
            "faq-11": "amenities_summary"
        }
        for faq in data.get("faqs", []):
            faq_id = faq.get("id", "")
            topic_str = faq.get("topic") or faq.get("title", "")
            topic_key = faq_topic_keys.get(faq_id, f"faq_{faq_id}")
            answer_text = faq.get("answer") or faq.get("a", "")
            self.chunks.append({
                "id": faq_id,
                "category": "faq",
                "topic": topic_key,
                "title": f"FAQ: {topic_str}",
                "content": f"{topic_str}: {answer_text}"
            })

    def _init_retriever(self):
        """Attempts OpenAI dense embeddings if key is present; otherwise initializes BM25 index."""
        if settings.OPENAI_API_KEY and not settings.MOCK_LLM:
            try:
                from langchain_openai import OpenAIEmbeddings
                embedder = OpenAIEmbeddings(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model="text-embedding-3-small"
                )
                texts = [f"{c['title']}: {c['content']}" for c in self.chunks]
                self.dense_embeddings = embedder.embed_documents(texts)
                return
            except Exception as e:
                print(f"[RAG] OpenAI dense embeddings unavailable ({e}), defaulting to BM25 index.")

        self._init_bm25()

    def _tokenize(self, text: str) -> list[str]:
        text = normalize_text(text)
        words = [w for w in re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text)]
        expanded = []
        for w in words:
            if w not in STOPWORDS:
                expanded.append(w)
                if w.endswith("ation"):
                    expanded.append(w[:-5])
                if w.endswith("ations"):
                    expanded.append(w[:-6])
                stemmed = re.sub(r"(?:ing|ed|es|s)$", "", w)
                if stemmed and len(stemmed) >= 3 and stemmed != w:
                    expanded.append(stemmed)
                if w in SYNONYMS:
                    expanded.extend(SYNONYMS[w])
        return expanded

    def _init_bm25(self):
        """Constructs an in-memory BM25 index with document frequency weights."""
        self.doc_tokens = [self._tokenize(f"{c['title']} {c['content']}") for c in self.chunks]
        self.df = Counter()
        for dt in self.doc_tokens:
            for t in set(dt):
                self.df[t] += 1

        self.N = len(self.chunks)

    def _bm25_score(self, query_tokens: list[str], idx: int) -> float:
        c = self.chunks[idx]
        title_tokens = set(self._tokenize(c["title"]))
        content_tokens = Counter(self.doc_tokens[idx])
        s = 0.0
        for t in query_tokens:
            if t in content_tokens:
                n = self.df.get(t, 0)
                # Standard BM25 IDF
                idf = math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
                # Title presence bonus
                boost = 3.0 if t in title_tokens else 1.0
                tf = content_tokens[t]
                s += idf * (tf / (tf + 1.2)) * boost
        return s

    def retrieve(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """
        Retrieves top-k relevant knowledge chunks using dense embeddings or BM25.
        """
        if not query or not query.strip():
            return self.chunks[:k]

        # 1. Use dense OpenAI embeddings if active
        if self.dense_embeddings is not None and settings.OPENAI_API_KEY and not settings.MOCK_LLM:
            try:
                from langchain_openai import OpenAIEmbeddings
                embedder = OpenAIEmbeddings(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model="text-embedding-3-small"
                )
                q_emb = embedder.embed_query(query)
                scores = []
                for i, doc_emb in enumerate(self.dense_embeddings):
                    dot = np.dot(q_emb, doc_emb)
                    norm = (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb)) or 1.0
                    sim = float(dot / norm)
                    chunk_copy = dict(self.chunks[i])
                    chunk_copy["score"] = round(sim, 4)
                    scores.append((sim, chunk_copy))
                scores.sort(key=lambda x: x[0], reverse=True)
                return [s[1] for s in scores[:k]]
            except Exception:
                pass

        # 2. Local BM25 Retrieval
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return self.chunks[:k]

        scored_chunks = []
        for i, c in enumerate(self.chunks):
            s = self._bm25_score(q_tokens, i)
            chunk_copy = dict(c)
            chunk_copy["score"] = round(s, 4)
            scored_chunks.append((s, chunk_copy))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_chunks[:k]]

# Singleton knowledge base instance
kb = HotelKnowledgeBase()
