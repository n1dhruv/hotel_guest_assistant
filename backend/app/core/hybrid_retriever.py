"""
Hybrid Search Engine combining Dense Qdrant Vector Retrieval and Sparse BM25 Lexical Search.
Merges candidate pools using Reciprocal Rank Fusion (RRF) to extract the Top 10 fused candidate chunks.

Architecture:
    User Query
        │
        ├───> HyDE Generator ───> Dense Vector Embedding ───> Qdrant Vector Search (Top 10)
        │                                                            │
        └───> Tokenizer & Normalizer ───> BM25 Index Search (Top 10) │
                                                                     ▼
                                                      Reciprocal Rank Fusion (RRF)
                                                                     │
                                                                     ▼
                                                        Merged Top 10 Candidate Chunks

Reciprocal Rank Fusion (RRF) Formulation:
    RRF_score(d) = sum_{m in {Dense, BM25}} 1 / (k + rank_m(d))
    where k = 60 by default.
"""

from __future__ import annotations

import asyncio
from collections import Counter
import logging
import math
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Set

from app.config import settings
from app.core.chunker import SemanticChunk, SemanticHotelChunker
from app.core.vector_store import QdrantVectorStore, get_vector_store

logger = logging.getLogger("hotel_assistant.hybrid_retriever")

BASE_DIR = Path(__file__).resolve().parent.parent
HOTEL_DATA_FILE = BASE_DIR / "data" / "hotel_data.json"

STOPWORDS: Set[str] = {
    "the", "is", "at", "which", "on", "a", "an", "this", "that", "to", "of",
    "for", "with", "does", "do", "you", "have", "can", "what", "where", "how",
    "are", "about", "hotel", "resort", "grand", "azure", "tell", "me", "any",
    "please", "i", "we", "my", "our", "would", "like", "if", "yes", "no", "so",
    "when", "why", "who", "want", "need", "could", "should", "there", "also"
}

SYNONYMS: Dict[str, List[str]] = {
    "cancel": ["cancellation", "cancellations", "cancelling", "cancelled", "refund"],
    "cancellation": ["cancel", "cancelling", "cancelled", "refund"],
    "cancelling": ["cancel", "cancellation"],
    "reservation": ["booking", "bookings", "reserve", "reservations"],
    "reserve": ["reservation", "booking", "reservations"],
    "booking": ["reservation", "reserve", "bookings"],
    "bookings": ["reservation", "booking"],
    "dog": ["pets"], "dogs": ["pets"], "cat": ["pets"], "cats": ["pets"], "pet": ["pets"],
    "smoke": ["smoking"], "smoking": ["smoke"], "cig": ["smoking"], "cigarettes": ["smoking"], "cigarette": ["smoking"],
    "wifi": ["wi-fi", "internet", "fiber", "speed", "mbps"], "internet": ["wifi", "wi-fi", "fiber"],
    "breakfast": ["buffet", "morning", "dining", "food", "royal"],
    "food": ["breakfast", "restaurant", "dining", "coastal", "spice", "vegetarian", "jain"],
    "dinner": ["restaurant", "dining", "food"], "lunch": ["restaurant", "dining", "food"],
    "vegetarian": ["veg", "pure", "jain", "saffron", "restaurant", "food"],
    "jain": ["vegetarian", "veg", "saffron", "restaurant", "food"],
    "location": ["address", "candolim", "located", "reach"], "located": ["location", "address", "candolim"],
    "address": ["location", "candolim", "located"], "reach": ["location", "address", "candolim"],
    "pool": ["swimming", "infinity", "pool", "temperature"], "swimming": ["pool", "infinity"],
    "gym": ["fitness", "spa", "workout"], "workout": ["fitness", "gym", "cardio"],
    "spa": ["massage", "wellness", "fitness", "ayurveda"],
    "id": ["aadhaar", "passport", "identity", "verification", "voter", "pan"],
    "aadhaar": ["id", "identity", "verification", "passport"],
    "passport": ["id", "identity", "verification", "aadhaar"],
    "pan": ["id", "identity", "verification", "card"],
    "document": ["id", "aadhaar", "passport", "verification"],
    "documents": ["id", "aadhaar", "passport", "verification"],
    "parking": ["valet", "car", "vehicle", "ev", "charging"],
    "car": ["parking", "valet", "ev"], "vehicle": ["parking", "valet", "ev"],
    "ev": ["parking", "valet", "charging", "tata"],
    "tata": ["ev", "charging", "power"],
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
    """Normalizes compound phrases and common hotel terminology."""
    t = text.lower()
    t = re.sub(r"\bcheck[\s-]+in\b", "checkin", t)
    t = re.sub(r"\bcheck[\s-]+out\b", "checkout", t)
    t = re.sub(r"\bwi[\s-]+fi\b", "wifi", t)
    t = re.sub(r"\bpan[\s-]+card\b", "pan", t)
    return t


# =====================================================================
# 1. Sparse BM25 Index
# =====================================================================

class BM25Index:
    """
    High-precision BM25 Lexical Search Index.
    Excels at exact keyword matches, numerical values, acronyms (e.g. Aadhaar, PAN, 500 Mbps).
    """

    def __init__(
        self,
        chunks: Optional[Sequence[Dict[str, Any]]] = None,
        data_path: Optional[Path] = None,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Counter = Counter()
        self.N: int = 0

        if chunks is not None:
            self.chunks = [dict(c) for c in chunks]
        else:
            path = data_path or getattr(settings, "HOTEL_DATA_PATH", HOTEL_DATA_FILE)
            chunker = SemanticHotelChunker(Path(path))
            self.chunks = [c.to_dict() for c in chunker.build_chunks()]

        self._build_index()

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text with domain stopword removal, stemming, and synonym expansion."""
        text = normalize_text(text)
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text)
        expanded: List[str] = []
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

    def _build_index(self) -> None:
        """Constructs inverted frequency tables and document statistics."""
        self.N = len(self.chunks)
        self.doc_tokens = []
        self.doc_lengths = []
        self.df = Counter()

        for c in self.chunks:
            title_text = c.get("title", "")
            content_text = c.get("content", "")
            keywords_text = " ".join(c.get("keywords", []))
            combined = f"{title_text} {content_text} {keywords_text}"
            tokens = self.tokenize(combined)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            for t in set(tokens):
                self.df[t] += 1

        self.avg_doc_len = (
            sum(self.doc_lengths) / self.N if self.N > 0 else 1.0
        )

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Computes BM25-Okapi score with title boosting."""
        chunk = self.chunks[doc_idx]
        title_tokens = set(self.tokenize(chunk.get("title", "")))
        content_counter = Counter(self.doc_tokens[doc_idx])
        doc_len = self.doc_lengths[doc_idx]

        score = 0.0
        for t in query_tokens:
            if t in content_counter:
                n = self.df.get(t, 0)
                # Standard BM25 Robertson-Spärck Jones IDF
                idf = math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

                tf = content_counter[t]
                # Length normalization
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                tf_norm = (tf * (self.k1 + 1.0)) / denom if denom > 0 else 0.0

                # Title match presence boost (3x)
                boost = 3.0 if t in title_tokens else 1.0
                score += idf * tf_norm * boost

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sparse lexical search returning ranked chunks with BM25 scores.
        """
        if not query or not query.strip():
            results = []
            for i, c in enumerate(self.chunks[:top_k], 1):
                item = dict(c)
                item["bm25_score"] = 0.0
                item["bm25_rank"] = i
                results.append(item)
            return results

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        scored_items: List[tuple[float, Dict[str, Any]]] = []
        for i, c in enumerate(self.chunks):
            if category_filter and c.get("category") != category_filter:
                continue

            score = self._bm25_score(q_tokens, i)
            if score > 0.0:
                item = dict(c)
                item["bm25_score"] = round(score, 4)
                scored_items.append((score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, (_, item) in enumerate(scored_items[:top_k], 1):
            item["bm25_rank"] = rank
            results.append(item)

        return results


# =====================================================================
# 2. Reciprocal Rank Fusion (RRF)
# =====================================================================

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fuses rankings from Dense Vector Search and Sparse BM25 Search using RRF.

    Formula:
        RRF_score(d) = sum_{m in {dense, bm25}} 1 / (rrf_k + rank_m(d))

    Args:
        dense_results: Ordered candidates from dense vector search.
        bm25_results: Ordered candidates from sparse BM25 search.
        rrf_k: Smoothing constant mitigating top-rank bias (standard default 60).
        top_k: Maximum number of fused candidate chunks to return (default 10).

    Returns:
        Deduplicated list of top candidate chunks sorted descending by RRF score.
    """
    fused_candidates: Dict[str, Dict[str, Any]] = {}

    # 1. Process Dense Channel
    for rank_idx, item in enumerate(dense_results, 1):
        cid = item.get("id") or item.get("chunk_id") or str(rank_idx)
        score_component = 1.0 / (rrf_k + rank_idx)

        fused_candidates[cid] = {
            "id": cid,
            "chunk_id": item.get("chunk_id", cid),
            "title": item.get("title", ""),
            "category": item.get("category", "general"),
            "topic": item.get("topic", "general"),
            "content": item.get("content", ""),
            "keywords": item.get("keywords", []),
            "metadata": item.get("metadata", {}),
            "rrf_score": score_component,
            "dense_rank": rank_idx,
            "dense_score": item.get("score"),
            "bm25_rank": None,
            "bm25_score": None,
            "channels": ["dense"],
        }

    # 2. Process BM25 Channel
    for rank_idx, item in enumerate(bm25_results, 1):
        cid = item.get("id") or item.get("chunk_id") or str(rank_idx)
        score_component = 1.0 / (rrf_k + rank_idx)

        if cid in fused_candidates:
            # Document retrieved by both channels -> Fuse scores
            entry = fused_candidates[cid]
            entry["rrf_score"] += score_component
            entry["bm25_rank"] = rank_idx
            entry["bm25_score"] = item.get("bm25_score")
            if "bm25" not in entry["channels"]:
                entry["channels"].append("bm25")
        else:
            # Document retrieved exclusively by BM25
            fused_candidates[cid] = {
                "id": cid,
                "chunk_id": item.get("chunk_id", cid),
                "title": item.get("title", ""),
                "category": item.get("category", "general"),
                "topic": item.get("topic", "general"),
                "content": item.get("content", ""),
                "keywords": item.get("keywords", []),
                "metadata": item.get("metadata", {}),
                "rrf_score": score_component,
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": rank_idx,
                "bm25_score": item.get("bm25_score"),
                "channels": ["bm25"],
            }

    # 3. Sort by RRF score descending
    sorted_candidates = sorted(
        fused_candidates.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    # 4. Format scores and assign final rank
    results: List[Dict[str, Any]] = []
    for rank, cand in enumerate(sorted_candidates[:top_k], 1):
        cand["rrf_rank"] = rank
        cand["rrf_score"] = round(cand["rrf_score"], 6)
        cand["score"] = cand["rrf_score"]  # Backward-compatible score alias
        results.append(cand)

    return results


# =====================================================================
# 3. Hybrid Retriever Engine
# =====================================================================

class HybridRetriever:
    """
    Production-grade Dual-Channel Hybrid Search Engine.
    Executes dense Qdrant vector retrieval (optionally with HyDE) and sparse BM25
    retrieval concurrently, fusing candidates via Reciprocal Rank Fusion into Top 10.
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        rrf_k: Optional[int] = None,
        default_top_k: Optional[int] = None,
        dense_candidates: Optional[int] = None,
        bm25_candidates: Optional[int] = None,
        use_hyde: Optional[bool] = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.bm25_index = bm25_index or BM25Index()
        self.rrf_k = rrf_k if rrf_k is not None else getattr(settings, "HYBRID_RRF_K", 60)
        self.default_top_k = (
            default_top_k if default_top_k is not None else getattr(settings, "HYBRID_TOP_K", 10)
        )
        self.dense_candidates = (
            dense_candidates
            if dense_candidates is not None
            else getattr(settings, "HYBRID_DENSE_CANDIDATES", 10)
        )
        self.bm25_candidates = (
            bm25_candidates
            if bm25_candidates is not None
            else getattr(settings, "HYBRID_BM25_CANDIDATES", 10)
        )
        self.use_hyde = (
            use_hyde if use_hyde is not None else getattr(settings, "HYDE_ENABLED", True)
        )

        self._lock = threading.Lock()
        self._total_queries = 0

    def retrieve_dense(
        self,
        query: str,
        top_k: int = 10,
        use_hyde: Optional[bool] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense semantic search channel via Qdrant."""
        hyde_flag = self.use_hyde if use_hyde is None else use_hyde
        try:
            return self.vector_store.search(
                query=query,
                top_k=top_k,
                category_filter=category_filter,
                use_hyde=hyde_flag,
            )
        except Exception as e:
            logger.warning(f"Dense vector retrieval error in hybrid search: {e}")
            return []

    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Sparse lexical search channel via BM25."""
        try:
            return self.bm25_index.search(
                query=query,
                top_k=top_k,
                category_filter=category_filter,
            )
        except Exception as e:
            logger.warning(f"BM25 retrieval error in hybrid search: {e}")
            return []

    def retrieve_candidates(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_hyde: Optional[bool] = None,
        category_filter: Optional[str] = None,
        dense_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronously executes dense and sparse searches and merges via Reciprocal Rank Fusion.
        Extracts Top 10 fused candidate chunks.
        """
        if not query or not query.strip():
            # Return baseline chunks if query is empty
            all_chunks = self.vector_store.get_all_chunks()
            return all_chunks[: (top_k or self.default_top_k)]

        with self._lock:
            self._total_queries += 1

        k_final = top_k or self.default_top_k
        k_dense = dense_k or self.dense_candidates
        k_bm25 = bm25_k or self.bm25_candidates

        # 1. Retrieve Dense Candidates (Top 10)
        dense_results = self.retrieve_dense(
            query=query,
            top_k=k_dense,
            use_hyde=use_hyde,
            category_filter=category_filter,
        )

        # 2. Retrieve BM25 Candidates (Top 10)
        bm25_results = self.retrieve_sparse(
            query=query,
            top_k=k_bm25,
            category_filter=category_filter,
        )

        # 3. Fuse via Reciprocal Rank Fusion (RRF)
        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_k=self.rrf_k,
            top_k=k_final,
        )

        return fused

    async def aretrieve_candidates(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_hyde: Optional[bool] = None,
        category_filter: Optional[str] = None,
        dense_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously executes dense and sparse searches in parallel and fuses via RRF.
        Maximizes retrieval throughput and minimizes endpoint latency.
        """
        if not query or not query.strip():
            all_chunks = self.vector_store.get_all_chunks()
            return all_chunks[: (top_k or self.default_top_k)]

        with self._lock:
            self._total_queries += 1

        k_final = top_k or self.default_top_k
        k_dense = dense_k or self.dense_candidates
        k_bm25 = bm25_k or self.bm25_candidates

        # Execute Dense and BM25 channels concurrently in thread pool
        dense_task = asyncio.to_thread(
            self.retrieve_dense,
            query,
            k_dense,
            use_hyde,
            category_filter,
        )
        bm25_task = asyncio.to_thread(
            self.retrieve_sparse,
            query,
            k_bm25,
            category_filter,
        )

        dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)

        # Fuse rankings via RRF
        return reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_k=self.rrf_k,
            top_k=k_final,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Returns hybrid retriever configuration and operational statistics."""
        with self._lock:
            return {
                "rrf_k": self.rrf_k,
                "default_top_k": self.default_top_k,
                "dense_candidates": self.dense_candidates,
                "bm25_candidates": self.bm25_candidates,
                "use_hyde": self.use_hyde,
                "total_queries": self._total_queries,
                "bm25_indexed_chunks": self.bm25_index.N,
                "vector_store_points": self.vector_store.get_point_count(),
            }


# =====================================================================
# Singleton Management
# =====================================================================

_hybrid_lock = threading.Lock()
_hybrid_instance: Optional[HybridRetriever] = None


def get_hybrid_retriever(force_new: bool = False, **kwargs: Any) -> HybridRetriever:
    """
    Returns application-wide singleton instance of HybridRetriever.
    Thread-safe initialization.
    """
    global _hybrid_instance
    with _hybrid_lock:
        if _hybrid_instance is None or force_new:
            _hybrid_instance = HybridRetriever(**kwargs)
        return _hybrid_instance
