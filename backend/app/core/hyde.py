"""
HyDE (Hypothetical Document Embeddings) Query Generation Pipeline.

Transforms conversational, terse, or implicit guest questions into structured
hypothetical passages from hotel documentation. Embedding these hypothetical
passages aligns guest intent with formal hotel policies, room specifications,
and amenity descriptions in the vector embedding space.

Architecture:
    Guest Query -> HyDE Generator (LiteLLM, max 80 tokens, temp=0.0) -> Hypothetical Passage -> Vector Embedding
    (Resilient fallback to raw query on timeout, offline mock mode, or error)
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

import litellm

from app.config import settings

logger = logging.getLogger("hotel_assistant.hyde")

# Canonical prompt template as specified in Phase 3 design
HYDE_SYSTEM_PROMPT = "You are an expert hospitality knowledge base generator."
HYDE_USER_PROMPT_TEMPLATE = (
    "Write a short, hypothetical paragraph (2-3 sentences) from a resort information document "
    "that directly addresses the following guest question:\n"
    'Question: "{query}"\n'
    "Hypothetical passage:"
)


class HyDEGenerator:
    """
    Hypothetical Document Embedding (HyDE) Generator.

    Generates short, hypothetical hotel handbook excerpts for guest questions to bridge
    the semantic and vocabulary discrepancy between conversational questions and formal
    hotel documentation.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        enabled: Optional[bool] = None,
        mock_mode: Optional[bool] = None,
    ):
        self.model = (
            model
            or getattr(settings, "HYDE_MODEL", "")
            or getattr(settings, "LLM_MODEL", "gpt-4o-mini")
        )
        self.temperature = (
            temperature
            if temperature is not None
            else getattr(settings, "HYDE_TEMPERATURE", 0.0)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else getattr(settings, "HYDE_MAX_TOKENS", 80)
        )
        self.timeout = (
            timeout
            if timeout is not None
            else getattr(settings, "HYDE_TIMEOUT", 1.5)
        )
        self.enabled = (
            enabled
            if enabled is not None
            else getattr(settings, "HYDE_ENABLED", True)
        )
        self.mock_mode = (
            mock_mode
            if mock_mode is not None
            else getattr(settings, "MOCK_LLM", False)
        )

        # Observability & metrics
        self._total_calls = 0
        self._total_fallbacks = 0
        self._last_latency_ms = 0.0
        self._lock = threading.Lock()

    def build_prompt(self, query: str) -> List[Dict[str, str]]:
        """Constructs LiteLLM chat messages for hypothetical passage generation."""
        cleaned_query = query.strip()
        return [
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": HYDE_USER_PROMPT_TEMPLATE.format(query=cleaned_query)},
        ]

    def clean_passage(self, raw_text: str, fallback_query: str) -> str:
        """
        Cleans and sanitizes the LLM-generated hypothetical passage.
        Strips markdown wrappers, redundant prefixes, and extra quotes.
        """
        if not raw_text or not raw_text.strip():
            return fallback_query

        text = raw_text.strip()

        # Remove markdown code blocks if wrapped
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        # Remove common prefixes output by LLMs
        prefix_patterns = [
            r"^(?:hypothetical\s+passage|passage|handbook\s+excerpt|resort\s+document|answer):\s*",
            r'^(?:hypothetical\s+paragraph):\s*',
        ]
        for pattern in prefix_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        # Strip surrounding quotation marks
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()

        # Remove internal double quotes if wrapped
        text = re.sub(r'^\s*"(.*?)"\s*$', r"\1", text)

        # Normalize redundant whitespaces
        text = re.sub(r"\s+", " ", text).strip()

        return text if text else fallback_query

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Synchronously generates a hypothetical document passage for the query.
        Returns the original query as fallback on timeout, error, or mock mode.
        """
        if not query or not query.strip():
            return query or ""

        trimmed_query = query.strip()

        # Fast path: disabled or mock/offline mode
        if not self.enabled or self.mock_mode:
            with self._lock:
                self._total_calls += 1
                self._total_fallbacks += 1
                self._last_latency_ms = 0.0
            return trimmed_query

        start_time = time.perf_counter()
        with self._lock:
            self._total_calls += 1

        try:
            messages = self.build_prompt(trimmed_query)
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )

            raw_content = ""
            if response and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, "message") and choice.message:
                    raw_content = getattr(choice.message, "content", "") or ""
                elif isinstance(choice, dict):
                    raw_content = choice.get("message", {}).get("content", "") or ""

            cleaned = self.clean_passage(raw_content, trimmed_query)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            with self._lock:
                self._last_latency_ms = latency_ms

            return cleaned

        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            with self._lock:
                self._total_fallbacks += 1
                self._last_latency_ms = latency_ms

            logger.warning(
                f"[HyDE] Generation failed or timed out ({type(e).__name__}: {e}) in {latency_ms}ms. "
                f"Falling back to raw query."
            )
            return trimmed_query

    async def agenerate_hypothetical_document(self, query: str) -> str:
        """
        Asynchronously generates a hypothetical document passage for the query.
        Non-blocking execution for FastAPI async endpoints.
        Returns original query on timeout, error, or mock mode.
        """
        if not query or not query.strip():
            return query or ""

        trimmed_query = query.strip()

        # Fast path: disabled or mock/offline mode
        if not self.enabled or self.mock_mode:
            with self._lock:
                self._total_calls += 1
                self._total_fallbacks += 1
                self._last_latency_ms = 0.0
            return trimmed_query

        start_time = time.perf_counter()
        with self._lock:
            self._total_calls += 1

        try:
            messages = self.build_prompt(trimmed_query)
            # Wrap in asyncio.wait_for to strictly enforce timeout
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                ),
                timeout=self.timeout + 0.2,
            )

            raw_content = ""
            if response and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, "message") and choice.message:
                    raw_content = getattr(choice.message, "content", "") or ""
                elif isinstance(choice, dict):
                    raw_content = choice.get("message", {}).get("content", "") or ""

            cleaned = self.clean_passage(raw_content, trimmed_query)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            with self._lock:
                self._last_latency_ms = latency_ms

            return cleaned

        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            with self._lock:
                self._total_fallbacks += 1
                self._last_latency_ms = latency_ms

            logger.warning(
                f"[HyDE] Async generation failed or timed out ({type(e).__name__}: {e}) in {latency_ms}ms. "
                f"Falling back to raw query."
            )
            return trimmed_query

    def get_stats(self) -> Dict[str, Any]:
        """Returns runtime performance and fallback statistics."""
        with self._lock:
            fallback_rate = (
                round((self._total_fallbacks / self._total_calls) * 100, 1)
                if self._total_calls > 0
                else 0.0
            )
            return {
                "enabled": self.enabled,
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "timeout_seconds": self.timeout,
                "mock_mode": self.mock_mode,
                "total_calls": self._total_calls,
                "total_fallbacks": self._total_fallbacks,
                "fallback_rate_pct": fallback_rate,
                "last_latency_ms": self._last_latency_ms,
            }

    def reset_stats(self) -> None:
        """Resets observability metrics."""
        with self._lock:
            self._total_calls = 0
            self._total_fallbacks = 0
            self._last_latency_ms = 0.0


# =====================================================================
# Singleton Management
# =====================================================================

_hyde_lock = threading.Lock()
_hyde_instance: Optional[HyDEGenerator] = None


def get_hyde_generator(force_new: bool = False, **kwargs: Any) -> HyDEGenerator:
    """
    Returns application-wide singleton instance of HyDEGenerator.
    Thread-safe initialization.
    """
    global _hyde_instance
    with _hyde_lock:
        if _hyde_instance is None or force_new:
            _hyde_instance = HyDEGenerator(**kwargs)
        return _hyde_instance
