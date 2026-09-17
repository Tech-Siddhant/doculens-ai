"""Unit tests for Phase 9.3 Metrics & Observability Surface."""

import time
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.config import settings
from app.core.metrics import metrics_collector
from app.schemas.health import SystemMetricsResponse
from app.services.orchestrator import orchestrator


def setup_function() -> None:
    """Reset metrics before each test for clean isolation."""
    metrics_collector.reset()


def teardown_function() -> None:
    """Reset metrics after each test."""
    metrics_collector.reset()


def test_metrics_collector_request_recording() -> None:
    """Verify HTTP request telemetry recording and aggregation."""
    metrics_collector.record_request("GET", "/api/v1/health", 200)
    metrics_collector.record_request("POST", "/api/v1/documents/upload", 201)
    metrics_collector.record_request("GET", "/api/v1/documents/doc_nonexistent", 404)
    metrics_collector.record_request("POST", "/api/v1/documents/doc_1/ask", 500)

    snapshot = metrics_collector.get_snapshot()
    assert snapshot.request_telemetry.total_requests == 4
    assert snapshot.request_telemetry.successful_requests == 2
    assert snapshot.request_telemetry.failed_requests == 2
    assert snapshot.request_telemetry.status_codes["200"] == 1
    assert snapshot.request_telemetry.status_codes["201"] == 1
    assert snapshot.request_telemetry.status_codes["404"] == 1
    assert snapshot.request_telemetry.status_codes["500"] == 1
    assert snapshot.request_telemetry.requests_by_endpoint["GET /api/v1/health"] == 1
    assert snapshot.request_telemetry.requests_by_endpoint["POST /api/v1/documents/upload"] == 1


def test_metrics_collector_pipeline_recording() -> None:
    """Verify pipeline telemetry and AI quality metric recording."""
    # Ingestion
    metrics_collector.record_ingestion(duration_ms=120.5, pages=5, chunks=15)
    metrics_collector.record_ingestion(duration_ms=80.5, pages=3, chunks=9)
    metrics_collector.record_rendering_failure()

    # Retrieval
    metrics_collector.record_retrieval(duration_ms=25.0, result_count=6)
    metrics_collector.record_retrieval(duration_ms=15.0, result_count=0)  # Empty retrieval

    # Reranking
    metrics_collector.record_reranking(duration_ms=30.0)

    # Generation & Grounding
    metrics_collector.record_generation(
        duration_ms=250.0,
        is_grounded=True,
        valid_citations=3,
        rejected_citations=1,
        is_refusal=False,
    )
    metrics_collector.record_generation(
        duration_ms=5.0,
        is_grounded=False,
        valid_citations=0,
        rejected_citations=0,
        is_refusal=True,
    )

    # Provider errors
    metrics_collector.record_provider_timeout()
    metrics_collector.record_provider_rate_limit()

    snapshot = metrics_collector.get_snapshot()

    # Ingestion assertions
    assert snapshot.pipeline_telemetry.ingestion.ingestion_count == 2
    assert snapshot.pipeline_telemetry.ingestion.total_pages_processed == 8
    assert snapshot.pipeline_telemetry.ingestion.total_chunks_created == 24
    assert snapshot.pipeline_telemetry.ingestion.rendering_failures_count == 1
    assert snapshot.pipeline_telemetry.ingestion.avg_duration_ms == 100.5

    # Retrieval assertions
    assert snapshot.pipeline_telemetry.retrieval.total_queries == 2
    assert snapshot.pipeline_telemetry.retrieval.empty_queries == 1
    assert snapshot.pipeline_telemetry.retrieval.avg_latency_ms == 20.0
    assert snapshot.pipeline_telemetry.retrieval.avg_result_count == 3.0

    # Reranking assertions
    assert snapshot.pipeline_telemetry.reranking.total_queries == 1
    assert snapshot.pipeline_telemetry.reranking.avg_latency_ms == 30.0

    # Generation assertions
    assert snapshot.pipeline_telemetry.generation.total_queries == 2


def test_metrics_collector_reset_isolation() -> None:
    """Verify reset() clears all metrics and resets timestamps."""
    metrics_collector.record_request("GET", "/test", 200)
    metrics_collector.record_ingestion(100.0, 2, 4)
    metrics_collector.record_rendering_failure()
    metrics_collector.record_retrieval(10.0, 3)
    metrics_collector.record_reranking(20.0)
    metrics_collector.record_generation(50.0, True, 1, 0)
    metrics_collector.record_provider_timeout()
    metrics_collector.record_provider_rate_limit()

    metrics_collector.reset()
    snapshot = metrics_collector.get_snapshot()

    assert snapshot.request_telemetry.total_requests == 0
    assert snapshot.request_telemetry.status_codes == {}
    assert snapshot.pipeline_telemetry.ingestion.ingestion_count == 0
    assert snapshot.pipeline_telemetry.ingestion.rendering_failures_count == 0
    assert snapshot.pipeline_telemetry.retrieval.total_queries == 0
    assert snapshot.pipeline_telemetry.reranking.total_queries == 0
    assert snapshot.pipeline_telemetry.generation.total_queries == 0
    assert snapshot.pipeline_telemetry.generation.timeout_count == 0
    assert snapshot.pipeline_telemetry.generation.rate_limit_count == 0
    assert snapshot.ai_quality_metrics.total_answers_generated == 0


def test_metrics_api_endpoint_healthy(client: TestClient) -> None:
    """Verify GET /health/metrics and GET /metrics return structured metrics snapshot."""
    res = client.get(f"{settings.API_V1_STR}/health/metrics")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "healthy"
    assert data["version"] == settings.VERSION
    assert "timestamp" in data
    assert "system_health" in data
    assert "request_telemetry" in data
    assert "pipeline_telemetry" in data
    assert "ai_quality_metrics" in data

    # Verify alias /metrics works identically
    alias_res = client.get(f"{settings.API_V1_STR}/metrics")
    assert alias_res.status_code == 200
    alias_data = alias_res.json()
    assert alias_data["version"] == settings.VERSION


def test_metrics_api_endpoint_unhealthy_when_dependency_down(client: TestClient) -> None:
    """Verify GET /health/metrics returns 503 when critical dependency is unhealthy."""
    with patch("app.services.vector_store.vector_store.get_stats", side_effect=RuntimeError("Vector DB down")):
        res = client.get(f"{settings.API_V1_STR}/health/metrics")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "unhealthy"
        assert data["system_health"]["status"] == "unhealthy"
        assert data["system_health"]["components"]["vector_store"]["status"] == "unhealthy"


def test_metrics_zero_sensitive_data_leakage(client: TestClient) -> None:
    """Verify metrics payload never leaks API keys, tokens, passwords, or secret prompts."""
    # Perform requests and pipeline operations
    client.get(f"{settings.API_V1_STR}/health")
    res = client.get(f"{settings.API_V1_STR}/health/metrics")
    assert res.status_code == 200
    metrics_str = res.text.lower()

    sensitive_patterns = [
        "api_key",
        "apikey",
        "secret",
        "password",
        "bearer ",
        "token=",
        "sk-",
        "ai_za",
    ]
    for pattern in sensitive_patterns:
        assert pattern not in metrics_str, f"Sensitive pattern '{pattern}' detected in metrics output!"


def test_metrics_orchestrator_execution_recording() -> None:
    """Verify orchestrator query execution automatically updates metrics collector."""
    orch = orchestrator
    res = orch.orchestrate_query(
        question="What is DocuLens AI architecture?",
        document_id="doc_metrics_test",
    )
    assert res.answer is not None

    snapshot = metrics_collector.get_snapshot()
    assert snapshot.pipeline_telemetry.retrieval.total_queries >= 1
    assert snapshot.pipeline_telemetry.reranking.total_queries >= 1
    assert snapshot.pipeline_telemetry.generation.total_queries >= 1
    assert snapshot.ai_quality_metrics.total_answers_generated >= 1
    assert snapshot.pipeline_telemetry.generation.avg_latency_ms is not None
    assert snapshot.pipeline_telemetry.generation.avg_latency_ms >= 0.0
    assert snapshot.pipeline_telemetry.generation.provider == settings.LLM_PROVIDER
    assert snapshot.pipeline_telemetry.generation.model == settings.LLM_MODEL
    assert snapshot.pipeline_telemetry.generation.timeout_count == 0
    assert snapshot.pipeline_telemetry.generation.rate_limit_count == 0

