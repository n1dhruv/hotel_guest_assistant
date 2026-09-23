import time
from threading import Lock
from typing import Any

class StatsTracker:
    """
    Thread-safe operational telemetry and usefulness metrics tracker.
    Provides live statistics on response latencies, tool adoption rate,
    fallback frequency, and prompt injection blocks.
    """
    def __init__(self):
        self._lock = Lock()
        self.total_requests = 0
        self.tool_calls_count = 0
        self.fallback_count = 0
        self.injection_attempts_blocked = 0
        self.latencies_ms: list[float] = []

    def record_request(
        self,
        latency_ms: float,
        tool_called: bool,
        used_fallback: bool,
        injection_blocked: bool = False
    ):
        with self._lock:
            self.total_requests += 1
            if tool_called:
                self.tool_calls_count += 1
            if used_fallback:
                self.fallback_count += 1
            if injection_blocked:
                self.injection_attempts_blocked += 1

            self.latencies_ms.append(round(latency_ms, 2))
            # Bound memory to last 1000 requests
            if len(self.latencies_ms) > 1000:
                self.latencies_ms.pop(0)

    def reset(self):
        """Resets tracker for test isolation."""
        with self._lock:
            self.total_requests = 0
            self.tool_calls_count = 0
            self.fallback_count = 0
            self.injection_attempts_blocked = 0
            self.latencies_ms.clear()

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            total = self.total_requests
            avg_lat = round(sum(self.latencies_ms) / len(self.latencies_ms), 2) if self.latencies_ms else 0.0

            p95_lat = 0.0
            if self.latencies_ms:
                sorted_lat = sorted(self.latencies_ms)
                p95_idx = int(len(sorted_lat) * 0.95)
                p95_lat = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

            return {
                "total_requests": total,
                "tool_calls_count": self.tool_calls_count,
                "tool_call_rate": round(self.tool_calls_count / total, 3) if total > 0 else 0.0,
                "fallback_count": self.fallback_count,
                "fallback_rate": round(self.fallback_count / total, 3) if total > 0 else 0.0,
                "injection_attempts_blocked": self.injection_attempts_blocked,
                "avg_latency_ms": avg_lat,
                "p95_latency_ms": p95_lat,
                "status": "healthy"
            }

stats_tracker = StatsTracker()
