"""
Vector Database Management & Embedding Pipeline using Qdrant.

Supports dual-mode deployment:
1. Embedded Local Mode ($0.00 Cost, Zero External Setup): Runs in-process with
   FastAPI using local storage (app/data/qdrant_storage) or in-memory (:memory:).
2. Managed Cloud Mode: Permanent free tier connection to Qdrant Cloud via
   QDRANT_URL and QDRANT_API_KEY.

Embedding Providers:
- OpenRouter Nemotron (nvidia/nemotron-3-embed-1b:free, 2048 dim): Free tier via OpenRouter & LiteLLM.
- FastEmbed (BAAI/bge-small-en-v1.5, 384 dim): 100% free CPU-based local embeddings.
- OpenAI (text-embedding-3-small, 1536 dim): Supported when OPENAI_API_KEY is configured.
- DeterministicLocalEmbedder (384 dim): Normalized token hashing for offline tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.core.chunker import SemanticChunk, SemanticHotelChunker

logger = logging.getLogger("hotel_assistant.vector_store")

# Deterministic UUIDv5 namespace for hotel knowledge base chunks
HOTEL_NAMESPACE = uuid.UUID("a3b8c874-569d-472e-9d2a-89a302685934")


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID string from a chunk identifier."""
    return str(uuid.uuid5(HOTEL_NAMESPACE, str(chunk_id)))


# =====================================================================
# Embedding Adapters
# =====================================================================

class BaseEmbedder(ABC):
    """Abstract interface for dense embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the embedding provider and model."""
        pass

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a list of document strings into vectors."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string into a vector."""
        pass


class FastEmbedAdapter(BaseEmbedder):
    """
    Local CPU-based embedding adapter using FastEmbed and BAAI/bge-small-en-v1.5.
    Runs 100% offline with zero external network calls or subscription costs.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self.model_name = model_name
        self._dim = 384
        self._model = TextEmbedding(model_name=self.model_name)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return f"fastembed ({self.model_name})"

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        text_list = list(texts)
        embeddings = list(self._model.embed(text_list))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        if not text:
            text = " "
        return next(self._model.embed([text])).tolist()


class OpenRouterNemotronEmbeddingAdapter(BaseEmbedder):
    """
    NVIDIA Nemotron-3-Embed-1B embedding adapter via OpenRouter and LiteLLM.
    Generates 2048-dimensional embeddings. Strictly disallows fallback to avoid dimension mismatch.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "openrouter/nvidia/nemotron-3-embed-1b:free",
        dimension: int = 2048,
        max_retries: int = 4
    ):
        self.api_key = api_key or getattr(settings, "OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be provided for OpenRouter Nemotron embedding. "
                "No fallback is allowed to prevent vector dimension mismatch in Qdrant."
            )
        self.model_name = model_name if model_name.startswith("openrouter/") else f"openrouter/{model_name}"
        self._dim = dimension
        self.max_retries = max_retries

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return f"openrouter ({self.model_name}, {self._dim}d)"

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        import time
        from litellm import embedding

        if not texts:
            return []

        text_list = list(texts)
        all_embeddings: List[List[float]] = []
        batch_size = 10

        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            for attempt in range(self.max_retries):
                try:
                    res = embedding(
                        model=self.model_name,
                        input=batch,
                        api_key=self.api_key
                    )
                    batch_vectors = [item["embedding"] for item in res.data]
                    all_embeddings.extend(batch_vectors)
                    break
                except Exception as e:
                    is_rate_limit = "429" in str(e) or "RateLimitError" in type(e).__name__
                    if is_rate_limit and attempt < self.max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        logger.warning(
                            f"OpenRouter rate limit on batch for {self.model_name}. Retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        logger.error(f"OpenRouter Nemotron embedding error on batch: {e}")
                        raise e

            # Micro-pause between batches to respect rate limits
            if i + batch_size < len(text_list):
                time.sleep(0.2)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        import time
        from litellm import embedding

        query_text = text if text and text.strip() else "hotel"
        for attempt in range(self.max_retries):
            try:
                res = embedding(
                    model=self.model_name,
                    input=[query_text],
                    api_key=self.api_key
                )
                return res.data[0]["embedding"]
            except Exception as e:
                is_rate_limit = "429" in str(e) or "RateLimitError" in type(e).__name__
                if is_rate_limit and attempt < self.max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"OpenRouter rate limit on query for {self.model_name}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"OpenRouter Nemotron query embedding error: {e}")
                    raise e

        raise RuntimeError(f"Exceeded max retries embedding query with {self.model_name}")


# Alias
NemotronEmbeddingAdapter = OpenRouterNemotronEmbeddingAdapter
LiteLLMGeminiEmbeddingAdapter = OpenRouterNemotronEmbeddingAdapter
GeminiEmbeddingAdapter = OpenRouterNemotronEmbeddingAdapter


class OpenAIEmbeddingAdapter(BaseEmbedder):
    """
    OpenAI embeddings adapter using text-embedding-3-small (1536 dim).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small"
    ):
        from langchain_openai import OpenAIEmbeddings
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be provided for OpenAIEmbeddingAdapter")
        self.model = model
        self._dim = 1536
        self._client = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=self.model
        )

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return f"openai ({self.model})"

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)


class DeterministicLocalEmbedder(BaseEmbedder):
    """
    Deterministic pseudo-semantic embedding adapter based on normalized word hashes.
    Useful for offline testing without network access or downloading ONNX models.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return f"deterministic-local ({self._dim}d)"

    _STOPWORDS = {
        "is", "the", "a", "an", "what", "does", "do", "for", "to", "of", "in",
        "at", "which", "are", "have", "you", "hotel", "resort", "grand", "azure",
        "tell", "me", "this", "that", "there", "and", "or", "on", "with"
    }

    def _embed_text(self, text: str) -> List[float]:
        import re
        vec = np.zeros(self._dim, dtype=np.float32)
        raw_words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        meaningful_words = [w for w in raw_words if w not in self._STOPWORDS]
        tokens = meaningful_words if meaningful_words else raw_words
        if not tokens:
            vec[0] = 1.0
            return vec.tolist()

        for w in tokens:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self._dim
            vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embedder(provider: Optional[str] = None) -> BaseEmbedder:
    """
    Factory to retrieve an embedding adapter.
    Uses OpenRouterNemotronEmbeddingAdapter (openrouter/nvidia/nemotron-3-embed-1b:free, 2048 dim) strictly.
    NO fallback to other providers/models is permitted to prevent Qdrant vector
    dimension mismatch and collection corruption.
    """
    req_provider = (provider or getattr(settings, "EMBEDDING_PROVIDER", "openrouter")).lower().strip()

    if (
        req_provider in ("openrouter", "nemotron", "nvidia", "auto")
        or req_provider.startswith("openrouter/")
        or "nemotron" in req_provider
    ):
        api_key = settings.OPENROUTER_API_KEY
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for OpenRouter Nemotron embedding. "
                "Fallback is disabled to avoid vector dimension mismatch in Qdrant."
            )
        model_name = getattr(settings, "EMBEDDING_MODEL", "openrouter/nvidia/nemotron-3-embed-1b:free")
        return OpenRouterNemotronEmbeddingAdapter(
            api_key=api_key,
            model_name=model_name,
            dimension=getattr(settings, "EMBEDDING_DIMENSION", 2048)
        )

    if req_provider in ("deterministic", "mock", "local"):
        return DeterministicLocalEmbedder()

    if req_provider == "fastembed":
        return FastEmbedAdapter()

    if req_provider == "openai":
        if settings.OPENAI_API_KEY:
            return OpenAIEmbeddingAdapter()
        raise ValueError("OPENAI_API_KEY is required for openai embedding provider")

    raise ValueError(f"Unknown or unsupported embedding provider: '{req_provider}'")


# =====================================================================
# Qdrant Vector Store Manager
# =====================================================================

class QdrantVectorStore:
    """
    Production-grade Vector Store wrapper for Qdrant.
    Supports local embedded storage ($0 cost, local file persistence), in-memory
    testing mode, and free managed Qdrant Cloud.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedder: Optional[BaseEmbedder] = None,
        client: Optional[QdrantClient] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[Union[Path, str]] = None,
        in_memory: bool = False,
        data_path: Optional[Union[Path, str]] = None,
        force_reindex: bool = False,
        auto_index: bool = True
    ):
        self.collection_name = collection_name or getattr(settings, "QDRANT_COLLECTION", "hotel_knowledge_base")
        self.embedder = embedder or get_embedder()
        self.data_path = Path(data_path or settings.HOTEL_DATA_PATH)
        self._owns_client = client is None

        # 1. Initialize client
        if client is not None:
            self.client = client
        elif in_memory or path == ":memory:" or getattr(settings, "QDRANT_IN_MEMORY", False) or getattr(settings, "QDRANT_URL", "") == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif (url or getattr(settings, "QDRANT_URL", None)) and (api_key or getattr(settings, "QDRANT_API_KEY", None)):
            q_url = url or settings.QDRANT_URL
            q_key = api_key or settings.QDRANT_API_KEY
            self.client = QdrantClient(url=q_url, api_key=q_key)
        else:
            storage_path = Path(path or settings.QDRANT_STORAGE_PATH)
            storage_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_path))

        # 2. Ensure collection schema matches vector dimension
        self.ensure_collection(force_recreate=force_reindex)

        # 3. Auto-index if collection is empty
        if auto_index and self.get_point_count() == 0 and self.data_path.exists():
            self.index_from_file(self.data_path)

    def ensure_collection(self, force_recreate: bool = False) -> None:
        """
        Verify collection exists with correct vector dimensions and distance metric.
        Recreates collection if force_recreate is True or if vector dimensions differ.
        """
        exists = self.client.collection_exists(self.collection_name)
        if exists:
            if force_recreate:
                self.client.delete_collection(self.collection_name)
                exists = False
            else:
                # Check vector dimensionality compatibility
                info = self.client.get_collection(self.collection_name)
                existing_dim = None
                try:
                    existing_dim = info.config.params.vectors.size
                except Exception:
                    pass

                if existing_dim is not None and existing_dim != self.embedder.dimension:
                    logger.warning(
                        f"Collection {self.collection_name} dimension ({existing_dim}) does not match "
                        f"active embedder dimension ({self.embedder.dimension}). Recreating collection."
                    )
                    self.client.delete_collection(self.collection_name)
                    exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}' ({self.embedder.dimension}d, Cosine).")

    def upsert_chunks(self, chunks: Sequence[Union[SemanticChunk, Dict[str, Any], Any]]) -> int:
        """
        Embeds and upserts semantic chunks or LangChain Document objects into Qdrant.
        Strictly idempotent: each chunk produces a deterministic UUIDv5 point ID.
        """
        if not chunks:
            return 0

        prepared_items: List[Dict[str, Any]] = []

        for c in chunks:
            if isinstance(c, SemanticChunk):
                prepared_items.append({
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "category": c.category,
                    "topic": c.topic,
                    "content": c.content,
                    "keywords": c.keywords,
                    "embed_text": c.document_text,
                    "metadata": c.metadata
                })
            elif isinstance(c, dict):
                cid = c.get("chunk_id") or c.get("id") or str(uuid.uuid4())
                title = c.get("title", "")
                content = c.get("content", "")
                kws = c.get("keywords", [])
                kw_str = f" (Keywords: {', '.join(kws)})" if kws else ""
                embed_text = c.get("document_text") or f"{title}: {content}{kw_str}"
                prepared_items.append({
                    "chunk_id": cid,
                    "title": title,
                    "category": c.get("category", "general"),
                    "topic": c.get("topic", "general"),
                    "content": content,
                    "keywords": kws,
                    "embed_text": embed_text,
                    "metadata": c.get("metadata", {})
                })
            elif hasattr(c, "page_content"):
                # LangChain Document
                meta = getattr(c, "metadata", {})
                cid = meta.get("chunk_id") or meta.get("faq_id") or meta.get("room_id") or str(uuid.uuid4())
                cat = meta.get("category", "general")
                title = meta.get("title") or meta.get("display_name") or meta.get("name") or f"{cat.title()} Info"
                prepared_items.append({
                    "chunk_id": cid,
                    "title": title,
                    "category": cat,
                    "topic": meta.get("topic", cat),
                    "content": c.page_content,
                    "keywords": meta.get("keywords", []),
                    "embed_text": f"{title}: {c.page_content}",
                    "metadata": meta
                })

        texts_to_embed = [item["embed_text"] for item in prepared_items]
        embeddings = self.embedder.embed_documents(texts_to_embed)

        points: List[PointStruct] = []
        for item, emb in zip(prepared_items, embeddings):
            pid = chunk_id_to_uuid(item["chunk_id"])
            payload = {
                "id": item["chunk_id"],
                "chunk_id": item["chunk_id"],
                "title": item["title"],
                "category": item["category"],
                "topic": item["topic"],
                "content": item["content"],
                "keywords": item["keywords"],
                "document_text": item["embed_text"],
                "metadata": item["metadata"],
            }
            # Add top-level flat metadata for query filtering
            for k, v in item["metadata"].items():
                if k not in payload and isinstance(v, (str, int, float, bool)):
                    payload[k] = v

            points.append(
                PointStruct(
                    id=pid,
                    vector=emb,
                    payload=payload
                )
            )

        # Batch upsert points
        batch_size = 64
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        return len(points)

    def upsert_documents(self, documents: Sequence[Any]) -> int:
        """Alias to index LangChain Document objects."""
        return self.upsert_chunks(documents)

    def index_from_file(self, data_path: Optional[Union[Path, str]] = None, force: bool = False) -> int:
        """
        Parses hotel JSON file using SemanticHotelChunker and upserts chunks into Qdrant.
        """
        path = Path(data_path or self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Hotel ground truth JSON not found at: {path}")

        if not force and self.get_point_count() > 0:
            return self.get_point_count()

        chunker = SemanticHotelChunker(path)
        chunks = chunker.build_chunks()
        count = self.upsert_chunks(chunks)
        logger.info(f"Indexed {count} chunks into collection '{self.collection_name}' from {path.name}.")
        return count

    def index_if_empty(self) -> int:
        """Indexes ground truth hotel JSON if collection currently has 0 vectors."""
        if self.get_point_count() == 0:
            return self.index_from_file()
        return self.get_point_count()

    def search(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None,
        category_filter: Optional[str] = None,
        use_hyde: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Dense vector similarity search against the hotel knowledge base.
        Supports HyDE (Hypothetical Document Embeddings) query expansion.
        Returns candidate chunks with cosine similarity score descending.
        """
        if not query or not query.strip():
            # Return all available chunks up to top_k
            all_chunks = self.get_all_chunks()
            return all_chunks[:top_k]

        search_text = query
        if use_hyde and query and query.strip():
            try:
                from app.core.hyde import get_hyde_generator
                generator = get_hyde_generator()
                hypo_doc = generator.generate_hypothetical_document(query)
                if hypo_doc and hypo_doc.strip():
                    search_text = hypo_doc
            except Exception as e:
                logger.warning(f"HyDE expansion notice in vector search ({e}). Using raw query.")
                search_text = query

        q_emb = self.embedder.embed_query(search_text)

        query_filter: Optional[Filter] = None
        if category_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category_filter)
                    )
                ]
            )

        try:
            # Modern Qdrant API
            search_response = self.client.query_points(
                collection_name=self.collection_name,
                query=q_emb,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold
            )
            points = search_response.points
        except Exception:
            # Fallback to search() method for backwards compatibility
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=q_emb,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold
            )

        results: List[Dict[str, Any]] = []
        for p in points:
            payload = p.payload or {}
            chunk_dict = {
                "id": payload.get("chunk_id") or str(p.id),
                "chunk_id": payload.get("chunk_id", ""),
                "title": payload.get("title", ""),
                "category": payload.get("category", "general"),
                "topic": payload.get("topic", "general"),
                "content": payload.get("content", ""),
                "keywords": payload.get("keywords", []),
                "score": round(float(p.score), 4),
                "metadata": payload.get("metadata", {}),
            }
            results.append(chunk_dict)

        return results

    def search_with_hyde(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search using HyDE (Hypothetical Document Embeddings) query expansion."""
        return self.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            category_filter=category_filter,
            use_hyde=True,
        )

    @property
    def storage_mode(self) -> str:
        """Indicates whether storage is in-memory (RAM), local disk, or cloud."""
        if hasattr(self.client, "_client") and getattr(self.client._client, "location", None) == ":memory:":
            return "in-memory (RAM)"
        if getattr(settings, "QDRANT_URL", None) and getattr(settings, "QDRANT_API_KEY", None):
            return "cloud"
        return "local disk"

    def get_point_count(self) -> int:
        """Returns the number of points in the active collection."""
        try:
            if not self.client.collection_exists(self.collection_name):
                return 0
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Scrolls and retrieves all indexed chunks from Qdrant."""
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            chunks: List[Dict[str, Any]] = []
            for p in points:
                payload = p.payload or {}
                chunks.append({
                    "id": payload.get("chunk_id") or str(p.id),
                    "chunk_id": payload.get("chunk_id", ""),
                    "title": payload.get("title", ""),
                    "category": payload.get("category", "general"),
                    "topic": payload.get("topic", "general"),
                    "content": payload.get("content", ""),
                    "keywords": payload.get("keywords", []),
                    "metadata": payload.get("metadata", {}),
                    "score": 1.0
                })
            return chunks
        except Exception:
            return []

    def delete_collection(self) -> bool:
        """Deletes the collection."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            return True
        return False

    def close(self) -> None:
        """Closes the underlying client connection if owned."""
        global _vector_store_instance
        if self._owns_client and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception:
                pass
        with _instance_lock:
            if _vector_store_instance is self:
                _vector_store_instance = None


# =====================================================================
# Thread-safe Singleton Management
# =====================================================================

_instance_lock = threading.Lock()
_vector_store_instance: Optional[QdrantVectorStore] = None


def get_vector_store(force_new: bool = False, **kwargs: Any) -> QdrantVectorStore:
    """
    Returns the application-wide singleton instance of QdrantVectorStore.
    Guarantees a single storage lock on app/data/qdrant_storage during server runtime.
    """
    global _vector_store_instance
    with _instance_lock:
        is_closed = False
        if _vector_store_instance is not None:
            c = getattr(_vector_store_instance, "client", None)
            if c is not None:
                if getattr(c, "_closed", False):
                    is_closed = True
                elif hasattr(c, "_client") and getattr(c._client, "_closed", False):
                    is_closed = True

        if _vector_store_instance is None or force_new or is_closed:
            if "in_memory" not in kwargs and getattr(settings, "QDRANT_IN_MEMORY", False):
                kwargs["in_memory"] = True
            _vector_store_instance = QdrantVectorStore(**kwargs)
        return _vector_store_instance
