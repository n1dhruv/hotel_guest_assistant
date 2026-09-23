import time
from typing import Any
from fastapi import APIRouter, HTTPException
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

    return ChatResponse(
        reply=result.get("reply", ""),
        tool_called=result.get("tool_called", False),
        availability=result.get("availability"),
        used_fallback=result.get("used_fallback", False),
        injection_blocked=result.get("injection_blocked", False),
        needs_dates=result.get("needs_dates", False),
        retrieved_sources=result.get("retrieved_sources", []),
        latency_ms=latency_ms
    )
