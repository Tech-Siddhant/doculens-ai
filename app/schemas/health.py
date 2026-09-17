from typing import Any, Literal
from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    details: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Application liveness status")
    version: str = Field(..., description="Application version")


class ReadinessResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall readiness status")
    version: str = Field(..., description="Application version")
    components: dict[str, ComponentStatus] = Field(
        default_factory=dict,
        description="Health status breakdown of individual backend dependencies",
    )


class SystemHealthMetrics(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall system readiness status")
    uptime_seconds: float = Field(..., ge=0.0, description="Server process uptime in seconds")
    components: dict[str, ComponentStatus] = Field(default_factory=dict, description="Component status breakdown")


class RequestTelemetryMetrics(BaseModel):
    total_requests: int = Field(default=0, ge=0, description="Total HTTP requests received")
    successful_requests: int = Field(default=0, ge=0, description="Total successful requests (HTTP 2xx, 3xx)")
    failed_requests: int = Field(default=0, ge=0, description="Total failed requests (HTTP 4xx, 5xx)")
    status_codes: dict[str, int] = Field(default_factory=dict, description="Count of requests grouped by HTTP status code")
    requests_by_endpoint: dict[str, int] = Field(default_factory=dict, description="Count of requests grouped by endpoint")


class IngestionMetrics(BaseModel):
    ingestion_count: int = Field(default=0, ge=0, description="Total ingestion runs executed")
    total_pages_processed: int = Field(default=0, ge=0, description="Total PDF pages extracted and processed")
    total_chunks_created: int = Field(default=0, ge=0, description="Total document chunks created")
    rendering_failures_count: int = Field(default=0, ge=0, description="Total page rendering failures")
    avg_duration_ms: float | None = Field(default=None, ge=0.0, description="Average ingestion duration in milliseconds")


class RetrievalMetrics(BaseModel):
    total_queries: int = Field(default=0, ge=0, description="Total retrieval queries processed")
    empty_queries: int = Field(default=0, ge=0, description="Retrieval queries yielding 0 candidate chunks")
    avg_latency_ms: float | None = Field(default=None, ge=0.0, description="Average retrieval stage latency in ms")
    avg_result_count: float | None = Field(default=None, ge=0.0, description="Average retrieved candidate count per query")


class RerankingMetrics(BaseModel):
    total_queries: int = Field(default=0, ge=0, description="Total reranking passes executed")
    avg_latency_ms: float | None = Field(default=None, ge=0.0, description="Average reranking latency in ms")


class GenerationMetrics(BaseModel):
    total_queries: int = Field(default=0, ge=0, description="Total generation calls executed")
    avg_latency_ms: float | None = Field(default=None, ge=0.0, description="Average LLM generation latency in ms")
    provider: str = Field(..., description="Active LLM provider identifier")
    model: str = Field(..., description="Active LLM model identifier")
    timeout_count: int = Field(default=0, ge=0, description="Total provider timeout occurrences")
    rate_limit_count: int = Field(default=0, ge=0, description="Total provider 429 rate limit occurrences")


class PipelineTelemetryMetrics(BaseModel):
    ingestion: IngestionMetrics = Field(default_factory=IngestionMetrics, description="Ingestion pipeline metrics")
    retrieval: RetrievalMetrics = Field(default_factory=RetrievalMetrics, description="Retrieval stage metrics")
    reranking: RerankingMetrics = Field(default_factory=RerankingMetrics, description="Reranking stage metrics")
    generation: GenerationMetrics = Field(..., description="Generation stage metrics")


class AIQualityMetrics(BaseModel):
    total_answers_generated: int = Field(default=0, ge=0, description="Total answers synthesized")
    grounded_answers_count: int = Field(default=0, ge=0, description="Total grounded answers based on evidence")
    refusal_answers_count: int = Field(default=0, ge=0, description="Total polite refusal answers due to insufficient evidence")
    valid_citations_count: int = Field(default=0, ge=0, description="Total valid citations verified")
    rejected_citations_count: int = Field(default=0, ge=0, description="Total invalid/hallucinated citations dropped")


class SystemMetricsResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall system health status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of the metrics snapshot")
    system_health: SystemHealthMetrics = Field(..., description="Underlying infrastructure health")
    request_telemetry: RequestTelemetryMetrics = Field(..., description="HTTP layer request telemetry")
    pipeline_telemetry: PipelineTelemetryMetrics = Field(..., description="End-to-end RAG pipeline telemetry")
    ai_quality_metrics: AIQualityMetrics = Field(..., description="Groundedness and citation verification metrics")


__all__ = [
    "ComponentStatus",
    "HealthResponse",
    "ReadinessResponse",
    "SystemHealthMetrics",
    "RequestTelemetryMetrics",
    "IngestionMetrics",
    "RetrievalMetrics",
    "RerankingMetrics",
    "GenerationMetrics",
    "PipelineTelemetryMetrics",
    "AIQualityMetrics",
    "SystemMetricsResponse",
]


