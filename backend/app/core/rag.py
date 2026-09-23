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
    "cancel": ["cancellation", "cancellations", "cancelling", "cancelled"],
    "cancellation": ["cancel", "cancelling", "cancelled"],
    "cancelling": ["cancel", "cancellation"],
    "reservation": ["booking", "bookings", "reserve", "reservations"],
    "reserve": ["reservation", "booking", "reservations"],
    "booking": ["reservation", "reserve", "bookings"],
    "bookings": ["reservation", "booking"],
    "dog": ["pets"], "dogs": ["pets"], "cat": ["pets"], "cats": ["pets"],
    "smoke": ["smoking"], "smoking": ["smoke"], "cig": ["smoking"], "cigarettes": ["smoking"],
    "wifi": ["wi-fi", "internet"], "internet": ["wifi"],
    "food": ["breakfast", "restaurant", "dining"],
    "dinner": ["restaurant", "dining"], "lunch": ["restaurant", "dining"],
    "location": ["address", "candolim", "located"], "located": ["location", "address", "candolim"],
    "address": ["location", "candolim", "located"], "reach": ["location", "address", "candolim"],
    "pool": ["swimming", "infinity", "pool"], "swimming": ["pool", "infinity"],
    "gym": ["fitness", "spa"], "workout": ["fitness", "gym"], "spa": ["massage", "wellness"],
    "id": ["aadhaar", "passport", "identity", "verification"],
    "aadhaar": ["id", "identity", "verification"],
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

        # 1. Property Overview Chunk
        prop = data.get("property", {})
        self.chunks.append({
            "id": "property-overview",
            "category": "property",
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
            self.chunks.append({
                "id": f"amenity-{i}-{a.get('name', '').lower().replace(' ', '-')}",
                "category": "amenity",
                "title": f"Amenity: {a.get('name')}",
                "content": f"{a.get('name')} | Timings: {a.get('hours')}. Description: {a.get('description')}"
            })

        # 3. Room Tier Chunks
        for r in data.get("rooms", []):
            self.chunks.append({
                "id": f"room-{r.get('id')}",
                "category": "room",
                "title": f"Room: {r.get('type')}",
                "content": (
                    f"{r.get('type')} (ID: {r.get('id')}) | Maximum Capacity: {r.get('maxGuests')} guests. "
                    f"Base Tariff: ₹{r.get('basePricePerNight')} per night. "
                    f"Bedding: {r.get('bedType')}. Features: {r.get('description')}"
                )
            })

        # 4. Policy Chunks
        policies = data.get("policies", {})
        for pol_key, pol_val in policies.items():
            self.chunks.append({
                "id": f"policy-{pol_key}",
                "category": "policy",
                "title": f"Policy: {pol_key.capitalize()}",
                "content": f"The Grand Azure Resort {pol_key.capitalize()} Policy: {pol_val}"
            })

        # 5. Curated FAQ Chunks
        for faq in data.get("faqs", []):
            self.chunks.append({
                "id": faq.get("id"),
                "category": "faq",
                "title": f"FAQ: {faq.get('q')}",
                "content": f"Guest Question: {faq.get('q')} | Verified Answer: {faq.get('a')}"
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
