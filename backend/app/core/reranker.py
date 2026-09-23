"""
Reranking Engine using NVIDIA Llama Nemotron Rerank VL 1B V2 via OpenRouter.
Second-stage cross-encoder pipeline re-scoring Top 10 hybrid candidates into refined Top 3-5 chunks.

Model: nvidia/llama-nemotron-rerank-vl-1b-v2:free
Provider: OpenRouter (POST https://openrouter.ai/api/v1/rerank)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

from app.config import settings

logger = logging.getLogger("hotel_assistant.reranker")

OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
DEFAULT_NEMOTRON_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"


# =====================================================================
# Base Interface
# =====================================================================

class BaseReranker(ABC):
    """Abstract interface for document reranking models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the reranking model."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously re-score and sort candidate documents for a query."""
        pass

    @abstractmethod
    async def arerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously re-score and sort candidate documents for a query."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics."""
        pass


def _candidate_text(candidate: Dict[str, Any]) -> str:
    """Constructs an informative text representation of a chunk for a cross-encoder."""
    title = str(candidate.get("title", "")).strip()
    content = str(candidate.get("content", "")).strip()
    category = str(candidate.get("category", "")).strip().upper()

    if title and category:
        return f"[{category}] {title}: {content}"
    elif title:
        return f"{title}: {content}"
    return content


# Default local cross-encoder for the zero-cost pipeline (FlashRank, ONNX CPU).
DEFAULT_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"


class FlashRankReranker(BaseReranker):
    """
    Zero-cost local cross-encoder reranker powered by FlashRank (ONNX, CPU).

    No API key, no quota, no network calls after the one-time model download.
    Falls back to DeterministicLocalReranker if the library or model weights
    are unavailable, so reranking never crashes the pipeline.
    """

    provider_name = "flashrank-local"

    def __init__(
        self,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        fallback_reranker: Optional[BaseReranker] = None,
    ):
        self.model = model or getattr(settings, "RERANKER_LOCAL_MODEL", "") or DEFAULT_FLASHRANK_MODEL
        self.top_n = top_n if top_n is not None else getattr(settings, "RERANKER_TOP_N", 3)
        self._fallback = fallback_reranker or DeterministicLocalReranker(
            model_name=f"{self.model} (fallback)"
        )
        self._ranker: Any = None
        self._ranker_failed = False
        self._lock = threading.Lock()

        # Operational metrics
        self._stats_lock = threading.Lock()
        self._total_requests: int = 0
        self._successful_reranks: int = 0
        self._fallback_reranks: int = 0
        self._total_latency_ms: float = 0.0

    @property
    def model_name(self) -> str:
        return self.model

    def _get_ranker(self) -> Any:
        """Lazily loads the FlashRank ONNX ranker (cached per instance)."""
        with self._lock:
            if self._ranker is None and not self._ranker_failed:
                try:
                    from flashrank import Ranker
                    self._ranker = Ranker(model_name=self.model)
                except Exception as e:
                    self._ranker_failed = True
                    logger.warning(f"[FlashRank] Model load failed ({e}); using deterministic fallback.")
            return self._ranker

    def _record_metrics(self, latency_ms: float, is_fallback: bool) -> None:
        with self._stats_lock:
            self._total_requests += 1
            if is_fallback:
                self._fallback_reranks += 1
            else:
                self._successful_reranks += 1
            self._total_latency_ms += latency_ms

    def _enrich(
        self,
        candidates: Sequence[Dict[str, Any]],
        scored_ids: List[int],
        scores: Dict[int, float],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Attaches rerank_score / rerank_rank / rerank_delta to the Top-N."""
        reranked: List[Dict[str, Any]] = []
        for orig_idx in scored_ids[:limit]:
            cand = dict(candidates[orig_idx])
            cand["rerank_score"] = round(float(scores.get(orig_idx, 0.0)), 6)
            cand["rerank_model"] = self.model
            cand["rerank_fallback"] = False
            cand["_orig_idx"] = orig_idx
            reranked.append(cand)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        final_list = reranked[:limit]
        for rank_idx, item in enumerate(final_list, start=1):
            item["rerank_rank"] = rank_idx
            prior_rank = item.get("rrf_rank", item.pop("_orig_idx", rank_idx) + 1)
            item.pop("_orig_idx", None)
            item["rerank_delta"] = prior_rank - rank_idx
        return final_list

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        limit = top_n if top_n is not None else self.top_n
        start_time = time.perf_counter()
        ranker = self._get_ranker()

        passages: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            text = _candidate_text(cand)
            if text.strip():
                passages.append({"id": idx, "text": text})

        if ranker is None or not passages:
            res = self._fallback.rerank(query, candidates, top_n=limit)
            self._record_metrics((time.perf_counter() - start_time) * 1000.0, is_fallback=True)
            return res

        try:
            from flashrank import RerankRequest
            with self._lock:
                results = ranker.rerank(RerankRequest(query=query, passages=passages))
            scored_ids: List[int] = []
            scores: Dict[int, float] = {}
            for item in results or []:
                if not isinstance(item, dict):
                    continue
                orig_idx = item.get("id")
                if orig_idx is None or orig_idx < 0 or orig_idx >= len(candidates):
                    continue
                scored_ids.append(orig_idx)
                scores[orig_idx] = float(item.get("score", 0.0))
            if not scored_ids:
                raise ValueError("FlashRank returned no scored passages")
            res = self._enrich(candidates, scored_ids, scores, limit)
            self._record_metrics((time.perf_counter() - start_time) * 1000.0, is_fallback=False)
            return res
        except Exception as e:
            logger.warning(f"[FlashRank] Rerank failed ({e}); using deterministic fallback.")
            res = self._fallback.rerank(query, candidates, top_n=limit)
            self._record_metrics((time.perf_counter() - start_time) * 1000.0, is_fallback=True)
            return res

    async def arerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        # ONNX inference is CPU-bound: keep the event loop free.
        return await asyncio.to_thread(self.rerank, query, candidates, top_n)

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            avg_lat = (
                round(self._total_latency_ms / self._total_requests, 2)
                if self._total_requests > 0
                else 0.0
            )
            return {
                "model": self.model,
                "provider": self.provider_name,
                "default_top_n": self.top_n,
                "total_requests": self._total_requests,
                "successful_reranks": self._successful_reranks,
                "fallback_reranks": self._fallback_reranks,
                "avg_latency_ms": avg_lat,
                "model_loaded": self._ranker is not None,
            }


# =====================================================================
# Deterministic Local Fallback Cross-Scorer
# =====================================================================

class DeterministicLocalReranker(BaseReranker):
    """
    Fast offline cross-encoder approximation.
    Computes exact lexical-semantic cross-scoring between query terms
    and candidate content/title/keywords, providing resilient fallback.
    """

    provider_name = "deterministic-local"

    def __init__(self, model_name: str = "deterministic-local-cross-encoder"):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    _STOPWORDS = {
        "is", "the", "a", "an", "what", "does", "do", "for", "to", "of", "in",
        "at", "which", "are", "have", "you", "hotel", "resort", "grand", "azure",
        "tell", "me", "this", "that", "there", "and", "or", "on", "with", "can", "i",
        "my", "we", "our"
    }

    def _score_candidate(self, query: str, candidate: Dict[str, Any]) -> float:
        """Score candidate chunk using normalized cross-feature alignment."""
        raw_q_tokens = [t.lower() for t in re.findall(r"\b\w{2,}\b", query)]
        q_tokens = [t for t in raw_q_tokens if t not in self._STOPWORDS]
        if not q_tokens:
            q_tokens = raw_q_tokens
        if not q_tokens:
            return float(candidate.get("rrf_score", 0.0001))

        title = str(candidate.get("title", "")).lower()
        content = str(candidate.get("content", "")).lower()
        keywords = [k.lower() for k in candidate.get("keywords", [])]

        title_hits = sum(1 for t in q_tokens if t in title)
        content_hits = sum(1 for t in q_tokens if t in content)
        keyword_hits = sum(1 for t in q_tokens if any(t in kw for kw in keywords))

        # Weight title heavily (like cross-encoder focus), then keywords, then content
        token_score = (2.5 * title_hits + 1.5 * keyword_hits + 1.0 * content_hits) / (len(q_tokens) * 3.5)
        token_score = min(1.0, token_score)

        # Base RRF score provides tiny tie-breaker
        base_rrf = float(candidate.get("rrf_score", 0.0))
        final_score = 0.90 * token_score + 0.10 * min(1.0, base_rrf * 30.0)
        return round(min(0.9999, max(0.0001, final_score)), 6)

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        limit = top_n if top_n is not None else 3
        scored: List[Dict[str, Any]] = []

        for orig_idx, cand in enumerate(candidates):
            c_copy = dict(cand)
            score = self._score_candidate(query, cand)
            c_copy["rerank_score"] = round(score, 6)
            c_copy["rerank_model"] = self.model_name
            c_copy["rerank_fallback"] = True
            c_copy["_orig_idx"] = orig_idx
            scored.append(c_copy)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        results = scored[:limit]

        for rank_idx, item in enumerate(results, start=1):
            item["rerank_rank"] = rank_idx
            prior_rank = item.get("rrf_rank", item.pop("_orig_idx", rank_idx) + 1)
            item.pop("_orig_idx", None)
            item["rerank_delta"] = prior_rank - rank_idx

        return results

    async def arerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self.rerank(query, candidates, top_n=top_n)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "provider": "deterministic-local",
            "type": "fallback",
        }


# =====================================================================
# OpenRouter NVIDIA Llama Nemotron Reranker
# =====================================================================

class OpenRouterNemotronReranker(BaseReranker):
    """
    Reranker engine powered strictly by NVIDIA Llama Nemotron Rerank VL 1B V2
    via OpenRouter (POST https://openrouter.ai/api/v1/rerank).
    Re-scores candidate chunks jointly against the query with cross-attention.
    """

    provider_name = "openrouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        fallback_reranker: Optional[BaseReranker] = None,
    ):
        self.api_key = api_key or getattr(settings, "OPENROUTER_API_KEY", "")
        self.model = model or getattr(settings, "RERANKER_MODEL", DEFAULT_NEMOTRON_MODEL)
        self.top_n = top_n if top_n is not None else getattr(settings, "RERANKER_TOP_N", 3)
        self.timeout = timeout if timeout is not None else getattr(settings, "RERANKER_TIMEOUT", 6.0)
        self.max_retries = max_retries
        self.endpoint = OPENROUTER_RERANK_URL
        self._fallback = fallback_reranker or DeterministicLocalReranker(
            model_name=f"{self.model} (fallback)"
        )

        # Operational metrics
        self._stats_lock = threading.Lock()
        self._total_requests: int = 0
        self._successful_reranks: int = 0
        self._fallback_reranks: int = 0
        self._total_latency_ms: float = 0.0

    @property
    def model_name(self) -> str:
        return self.model

    def _prepare_document_text(self, candidate: Dict[str, Any]) -> str:
        """Constructs an informative text representation of the chunk for the cross-encoder."""
        return _candidate_text(candidate)

    def _build_headers(self) -> Dict[str, str]:
        """Constructs OpenRouter HTTP authorization and identification headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/n1dhruv/hotel_guest_assistant",
            "X-Title": "Hotel Guest Assistant RAG Reranker",
        }

    def _record_metrics(self, latency_ms: float, is_fallback: bool) -> None:
        """Thread-safe recording of rerank query performance metrics."""
        with self._stats_lock:
            self._total_requests += 1
            if is_fallback:
                self._fallback_reranks += 1
            else:
                self._successful_reranks += 1
            self._total_latency_ms += latency_ms

    def _parse_rerank_response(
        self,
        data: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Parses OpenRouter rerank response:
        Expects: {"results": [{"index": int, "relevance_score": float}, ...]}
        """
        results_list = data.get("results") or data.get("data")
        if not results_list or not isinstance(results_list, list):
            raise ValueError(f"OpenRouter rerank response missing 'results' list: {data}")

        reranked: List[Dict[str, Any]] = []
        for item in results_list:
            if not isinstance(item, dict):
                continue
            orig_idx = item.get("index")
            if orig_idx is None or orig_idx < 0 or orig_idx >= len(candidates):
                continue

            score = float(item.get("relevance_score", 0.0))
            cand = dict(candidates[orig_idx])
            cand["rerank_score"] = round(score, 6)
            cand["rerank_model"] = self.model
            cand["rerank_fallback"] = False
            cand["_orig_idx"] = orig_idx
            reranked.append(cand)

        # Sort descending by relevance score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        final_list = reranked[:limit]

        # Assign ranks and calculate delta (rank shift vs initial RRF rank)
        for rank_idx, item in enumerate(final_list, start=1):
            item["rerank_rank"] = rank_idx
            prior_rank = item.get("rrf_rank", item.pop("_orig_idx", rank_idx) + 1)
            item.pop("_orig_idx", None)
            item["rerank_delta"] = prior_rank - rank_idx

        return final_list

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronously reranks candidate chunks using OpenRouter Nemotron Rerank.
        Falls back seamlessly if API key is absent, rate-limited, or unavailable.
        """
        if not candidates:
            return []

        limit = top_n if top_n is not None else self.top_n
        start_time = time.perf_counter()

        # If offline mock mode or no API key, execute fallback directly
        if getattr(settings, "MOCK_LLM", False) or not self.api_key:
            res = self._fallback.rerank(query, candidates, top_n=limit)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self._record_metrics(latency_ms, is_fallback=True)
            return res

        docs = [self._prepare_document_text(c) for c in candidates]
        payload = {
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_n": limit,
        }
        headers = self._build_headers()

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.endpoint, json=payload, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    res = self._parse_rerank_response(data, candidates, limit)
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    self._record_metrics(latency_ms, is_fallback=False)
                    return res

                # Handle rate limits (429) or server errors
                if resp.status_code in (429, 502, 503, 504):
                    wait = 1.5 * (attempt + 1)
                    logger.warning(
                        f"OpenRouter rerank {resp.status_code} received (attempt {attempt + 1}). Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue

                # Non-retryable error
                logger.warning(f"OpenRouter rerank failed with status {resp.status_code}: {resp.text}")
                break

            except Exception as e:
                last_error = e
                wait = 1.0 * (attempt + 1)
                logger.warning(f"OpenRouter rerank exception (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        # Fallback if retries exhausted
        logger.warning(
            f"OpenRouter Nemotron reranker falling back to local cross-scorer: {last_error or 'request failed'}"
        )
        res = self._fallback.rerank(query, candidates, top_n=limit)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self._record_metrics(latency_ms, is_fallback=True)
        return res

    async def arerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously reranks candidate chunks using OpenRouter Nemotron Rerank.
        Falls back seamlessly if API key is absent, rate-limited, or unavailable.
        """
        if not candidates:
            return []

        limit = top_n if top_n is not None else self.top_n
        start_time = time.perf_counter()

        if getattr(settings, "MOCK_LLM", False) or not self.api_key:
            res = await self._fallback.arerank(query, candidates, top_n=limit)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self._record_metrics(latency_ms, is_fallback=True)
            return res

        docs = [self._prepare_document_text(c) for c in candidates]
        payload = {
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_n": limit,
        }
        headers = self._build_headers()

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.endpoint, json=payload, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    res = self._parse_rerank_response(data, candidates, limit)
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    self._record_metrics(latency_ms, is_fallback=False)
                    return res

                if resp.status_code in (429, 502, 503, 504):
                    wait = 1.5 * (attempt + 1)
                    logger.warning(
                        f"OpenRouter rerank {resp.status_code} async (attempt {attempt + 1}). Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                    continue

                logger.warning(f"OpenRouter rerank failed with status {resp.status_code}: {resp.text}")
                break

            except Exception as e:
                last_error = e
                wait = 1.0 * (attempt + 1)
                logger.warning(f"OpenRouter rerank async exception (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)

        logger.warning(
            f"OpenRouter Nemotron async reranker falling back to local cross-scorer: {last_error or 'request failed'}"
        )
        res = await self._fallback.arerank(query, candidates, top_n=limit)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self._record_metrics(latency_ms, is_fallback=True)
        return res

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            avg_lat = (
                round(self._total_latency_ms / self._total_requests, 2)
                if self._total_requests > 0
                else 0.0
            )
            return {
                "model": self.model,
                "provider": "openrouter",
                "endpoint": self.endpoint,
                "default_top_n": self.top_n,
                "timeout": self.timeout,
                "total_requests": self._total_requests,
                "successful_reranks": self._successful_reranks,
                "fallback_reranks": self._fallback_reranks,
                "avg_latency_ms": avg_lat,
                "has_api_key": bool(self.api_key),
            }


# =====================================================================
# Thread-Safe Singleton Factory
# =====================================================================

_reranker_lock = threading.Lock()
_reranker_instance: Optional[BaseReranker] = None


def get_reranker(force_new: bool = False, **kwargs: Any) -> BaseReranker:
    """
    Returns application-wide singleton reranker, routed by RERANKER_PROVIDER:
    - "flashrank" (default, zero-cost): FlashRankReranker (local ONNX cross-encoder).
    - "local" / "deterministic": DeterministicLocalReranker (lexical cross-scorer).
    - anything else: OpenRouterNemotronReranker (hosted, quota-metered).
    Thread-safe initialization.
    """
    global _reranker_instance
    provider = str(
        kwargs.pop("provider", None) or getattr(settings, "RERANKER_PROVIDER", "openrouter")
    ).lower().strip()
    with _reranker_lock:
        if _reranker_instance is None or force_new:
            if provider in ("flashrank", "local-flashrank"):
                _reranker_instance = FlashRankReranker(**kwargs)
            elif provider in ("local", "deterministic", "fallback"):
                _reranker_instance = DeterministicLocalReranker()
            else:
                _reranker_instance = OpenRouterNemotronReranker(**kwargs)
        return _reranker_instance
