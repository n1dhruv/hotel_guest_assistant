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

class HotelKnowledgeBase:
    """
    RAG-Lite In-Memory Vector Store and Retriever.
    
    Architecture Design:
    - At startup, loads and segments hotel ground truth into semantic chunks.
    - Dual-mode embedding support:
        1. Dense OpenAI Embeddings (text-embedding-3-small) if API key is present.
        2. Zero-dependency TF-IDF cosine-similarity retriever for local / offline / test execution.
    - Demonstrates true RAG retrieval without the operational overhead of an external vector DB.
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
                f"Standard check-in time is {prop.get('checkIn')} and check-out time is {prop.get('checkOut')}. "
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
        """Attempts OpenAI dense embeddings if key is present; otherwise initializes TF-IDF vectors."""
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
                print(f"[RAG] OpenAI dense embeddings unavailable ({e}), defaulting to lexical TF-IDF index.")

        self._init_local_tfidf()

    def _tokenize(self, text: str) -> list[str]:
        return [w.lower() for w in re.findall(r"\b\w{2,}\b", text)]

    def _init_local_tfidf(self):
        """Constructs an in-memory TF-IDF index with cosine-normalized vectors."""
        doc_tokens = [self._tokenize(f"{c['title']} {c['content']}") for c in self.chunks]
        self.doc_freqs = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                self.doc_freqs[term] += 1

        self.N = len(self.chunks)
        self.doc_vectors = []
        for tokens in doc_tokens:
            tf = Counter(tokens)
            vec = {}
            for term, count in tf.items():
                idf = math.log((self.N + 1) / (self.doc_freqs[term] + 0.5)) + 1.0
                vec[term] = count * idf
            # L2 vector normalization
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_vectors.append({k: v / norm for k, v in vec.items()})

    def retrieve(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """
        Retrieves top-k relevant knowledge chunks using cosine similarity.
        """
        if not query or not query.strip():
            return self.chunks[:k]

        # 1. Use dense OpenAI embeddings if available
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

        # 2. Local TF-IDF Cosine Retrieval (deterministic, offline-ready)
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return self.chunks[:k]

        q_tf = Counter(q_tokens)
        q_vec = {}
        for term, count in q_tf.items():
            if term in self.doc_freqs:
                idf = math.log((self.N + 1) / (self.doc_freqs[term] + 0.5)) + 1.0
                q_vec[term] = count * idf

        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {k: v / q_norm for k, v in q_vec.items()}

        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            dot = sum(q_vec.get(term, 0.0) * weight for term, weight in doc_vec.items())
            chunk_copy = dict(self.chunks[i])
            chunk_copy["score"] = round(dot, 4)
            scores.append((dot, chunk_copy))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scores[:k]]

# Singleton knowledge base instance
kb = HotelKnowledgeBase()
