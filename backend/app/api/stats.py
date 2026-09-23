from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.stats import stats_tracker

router = APIRouter(prefix="/api/stats", tags=["stats"])

class StatsResponse(BaseModel):
    total_requests: int
    tool_calls_count: int
    tool_call_rate: float
    fallback_count: int
    fallback_rate: float
    injection_attempts_blocked: int
    avg_latency_ms: float
    p95_latency_ms: float
    status: str

@router.get("", response_model=StatsResponse)
def get_stats() -> dict[str, Any]:
    """
    Returns live operational statistics answering 'How would you measure usefulness?':
    - tool_call_rate: % of requests resolving guest booking intents via tools
    - fallback_rate: % of queries requiring human concierge escalation
    - injection_attempts_blocked: Security containment metrics
    - avg_latency_ms & p95_latency_ms: User response speed SLAs
    """
    return stats_tracker.get_summary()
