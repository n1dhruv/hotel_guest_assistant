import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.stats import stats_tracker

client = TestClient(app)

def test_stats_tracker_unit():
    stats_tracker.reset()
    assert stats_tracker.total_requests == 0

    # Record 4 simulated requests
    stats_tracker.record_request(latency_ms=100.0, tool_called=True, used_fallback=False)
    stats_tracker.record_request(latency_ms=200.0, tool_called=False, used_fallback=False)
    stats_tracker.record_request(latency_ms=300.0, tool_called=False, used_fallback=True)
    stats_tracker.record_request(latency_ms=400.0, tool_called=True, used_fallback=False, injection_blocked=True)

    summary = stats_tracker.get_summary()
    assert summary["total_requests"] == 4
    assert summary["tool_calls_count"] == 2
    assert summary["tool_call_rate"] == 0.5
    assert summary["fallback_count"] == 1
    assert summary["fallback_rate"] == 0.25
    assert summary["injection_attempts_blocked"] == 1
    assert summary["avg_latency_ms"] == 250.0
    assert summary["status"] == "healthy"

def test_stats_api_endpoint():
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_requests" in data
    assert "tool_call_rate" in data
    assert "fallback_rate" in data
    assert "avg_latency_ms" in data
    assert "p95_latency_ms" in data
    assert data["status"] == "healthy"

def test_chat_updates_stats_telemetry():
    stats_tracker.reset()

    # Send a chat query
    payload = {
        "messages": [
            {"role": "user", "content": "Does the hotel have a pool?"}
        ]
    }
    chat_res = client.post("/api/chat", json=payload)
    assert chat_res.status_code == 200
    assert "latency_ms" in chat_res.json()
    assert chat_res.json()["latency_ms"] > 0

    # Verify telemetry endpoint reflects the request
    stats_res = client.get("/api/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert stats_data["total_requests"] >= 1
    assert stats_data["avg_latency_ms"] > 0
