import json
import time
from typing import Any, AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.chain import orchestrator
from app.core.stats import stats_tracker

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str = Field(description="Message role: 'user' or 'assistant'")
    content: str = Field(description="Message text content")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(description="List of conversation turns")


class ChatResponse(BaseModel):
    reply: str
    tool_called: bool
    availability: dict[str, Any] | None = None
    used_fallback: bool
    injection_blocked: bool = False
    needs_dates: bool = False
    retrieved_sources: list[str] = []
    # Phase 6: detailed citation pills, per-stage latency, pipeline provenance
    sources: list[dict[str, Any]] = []
    latency_breakdown: dict[str, float] = {}
    pipeline: dict[str, Any] = {}
    latency_ms: float = 0.0


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint:
    - Performs prompt injection verification
    - Performs semantic RAG retrieval from hotel ground truth
    - Dispatches to LangChain tool-calling agent (or deterministic fallback)
    - Records telemetry into StatsTracker
    - Returns structured answer and room availability cards if requested
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    start_time = time.perf_counter()
    messages_data = [{"role": m.role, "content": m.content} for m in request.messages]
    result = await orchestrator.execute(messages_data)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Track metrics
    stats_tracker.record_request(
        latency_ms=latency_ms,
        tool_called=result.get("tool_called", False),
        used_fallback=result.get("used_fallback", False),
        injection_blocked=result.get("injection_blocked", False)
    )

    breakdown = dict(result.get("latency_breakdown", {}))
    breakdown.setdefault("generation_ms", 0.0)
    # Total wall-clock always present alongside the per-stage breakdown
    breakdown["total_ms"] = latency_ms

    return ChatResponse(
        reply=result.get("reply", ""),
        tool_called=result.get("tool_called", False),
        availability=result.get("availability"),
        used_fallback=result.get("used_fallback", False),
        injection_blocked=result.get("injection_blocked", False),
        needs_dates=result.get("needs_dates", False),
        retrieved_sources=result.get("retrieved_sources", []),
        sources=result.get("sources", []),
        latency_breakdown=breakdown,
        pipeline=result.get("pipeline", {}),
        latency_ms=latency_ms
    )


async def _sse_generator(messages_data: list[dict]) -> AsyncGenerator[str, None]:
    """Server-Sent Events generator that streams tokens from the orchestrator."""
    start_time = time.perf_counter()
    metadata: dict = {}

    try:
        async for chunk in orchestrator.execute_stream(messages_data):
            chunk_type = chunk.get("type")
            if chunk_type == "token":
                # Stream each token as an SSE data event
                payload = json.dumps({"type": "token", "token": chunk["token"]})
                yield f"data: {payload}\n\n"
            elif chunk_type == "metadata":
                metadata = chunk
                # Record stats when we have metadata
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                stats_tracker.record_request(
                    latency_ms=latency_ms,
                    tool_called=chunk.get("tool_called", False),
                    used_fallback=chunk.get("used_fallback", False),
                    injection_blocked=chunk.get("injection_blocked", False)
                )
                # Send metadata event so frontend can update state
                chunk_breakdown = dict(chunk.get("latency_breakdown", {}))
                chunk_breakdown.setdefault("total_ms", latency_ms)
                payload = json.dumps({
                    "type": "metadata",
                    "tool_called": chunk.get("tool_called", False),
                    "availability": chunk.get("availability"),
                    "used_fallback": chunk.get("used_fallback", False),
                    "injection_blocked": chunk.get("injection_blocked", False),
                    "needs_dates": chunk.get("needs_dates", False),
                    "retrieved_sources": chunk.get("retrieved_sources", []),
                    "sources": chunk.get("sources", []),
                    "latency_breakdown": chunk_breakdown,
                    "pipeline": chunk.get("pipeline", {}),
                    "latency_ms": latency_ms
                })
                yield f"data: {payload}\n\n"
    except Exception as e:
        error_payload = json.dumps({"type": "error", "message": str(e)})
        yield f"data: {error_payload}\n\n"

    # Final [DONE] sentinel
    yield "data: [DONE]\n\n"


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    """
    Streaming conversational endpoint using Server-Sent Events (SSE).
    Yields tokens in real-time as the LLM generates them.
    Frontend should consume via EventSource or fetch with ReadableStream.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    messages_data = [{"role": m.role, "content": m.content} for m in request.messages]

    return StreamingResponse(
        _sse_generator(messages_data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
