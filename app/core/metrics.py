import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.schemas.health import (
    AIQualityMetrics,
    ComponentStatus,
    GenerationMetrics,
    IngestionMetrics,
    PipelineTelemetryMetrics,
    RequestTelemetryMetrics,
    RerankingMetrics,
    RetrievalMetrics,
    SystemHealthMetrics,
    SystemMetricsResponse,
)


class MetricsCollector:
    """Thread-safe, lightweight in-memory metrics collector for DocuLens AI.

    Resource footprint: strictly bounded (maxlen=100 deques, <50KB RAM, zero external daemons).
    """

    def __init__(self, max_samples: int = 100) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._max_samples = max_samples
        self._init_metrics()

    def _init_metrics(self) -> None:
        # Request Telemetry
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.status_codes: dict[str, int] = {}
        self.requests_by_endpoint: dict[str, int] = {}

        # Pipeline Telemetry: Ingestion
        self.ingestion_count = 0
        self.total_pages_processed = 0
        self.total_chunks_created = 0
        self.rendering_failures_count = 0
        self.ingestion_durations_ms: deque[float] = deque(maxlen=self._max_samples)

        # Pipeline Telemetry: Retrieval
        self.total_retrieval_queries = 0
        self.empty_retrieval_count = 0
        self.retrieval_durations_ms: deque[float] = deque(maxlen=self._max_samples)
        self.retrieval_result_counts: deque[int] = deque(maxlen=self._max_samples)

        # Pipeline Telemetry: Reranking
        self.total_rerankings = 0
        self.reranking_durations_ms: deque[float] = deque(maxlen=self._max_samples)

        # Pipeline Telemetry: Generation
        self.total_generations = 0
        self.generation_durations_ms: deque[float] = deque(maxlen=self._max_samples)
        self.timeout_count = 0
        self.rate_limit_count = 0

        # AI Quality / Groundedness Outcomes
        self.total_answers_generated = 0
        self.grounded_answers_count = 0
        self.refusal_answers_count = 0
        self.valid_citations_count = 0
        self.rejected_citations_count = 0

    def reset(self) -> None:
        """Reset all metrics (primarily for testing and isolation)."""
        with self._lock:
            self._start_time = time.time()
            self._init_metrics()

    def record_request(self, method: str, path: str, status_code: int) -> None:
        """Record HTTP request outcome."""
        with self._lock:
            self.total_requests += 1
            if status_code < 400:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            code_key = str(status_code)
            self.status_codes[code_key] = self.status_codes.get(code_key, 0) + 1

            endpoint_key = f"{method.upper()} {path}"
            self.requests_by_endpoint[endpoint_key] = (
                self.requests_by_endpoint.get(endpoint_key, 0) + 1
            )

    def record_ingestion(self, duration_ms: float, pages: int, chunks: int) -> None:
        """Record document ingestion execution."""
        with self._lock:
            self.ingestion_count += 1
            self.total_pages_processed += max(0, pages)
            self.total_chunks_created += max(0, chunks)
            self.ingestion_durations_ms.append(max(0.0, duration_ms))

    def record_rendering_failure(self) -> None:
        """Record a page rendering failure event."""
        with self._lock:
            self.rendering_failures_count += 1

    def record_retrieval(self, duration_ms: float, result_count: int) -> None:
        """Record retrieval stage execution."""
        with self._lock:
            self.total_retrieval_queries += 1
            if result_count == 0:
                self.empty_retrieval_count += 1
            self.retrieval_durations_ms.append(max(0.0, duration_ms))
            self.retrieval_result_counts.append(max(0, result_count))

    def record_reranking(self, duration_ms: float) -> None:
        """Record reranker stage execution."""
        with self._lock:
            self.total_rerankings += 1
            self.reranking_durations_ms.append(max(0.0, duration_ms))

    def record_generation(
        self,
        duration_ms: float,
        is_grounded: bool,
        valid_citations: int = 0,
        rejected_citations: int = 0,
        is_refusal: bool = False,
    ) -> None:
        """Record answer generation and grounding verification outcome."""
        with self._lock:
            self.total_generations += 1
            self.generation_durations_ms.append(max(0.0, duration_ms))
            self.total_answers_generated += 1
            if is_grounded and not is_refusal:
                self.grounded_answers_count += 1
            else:
                self.refusal_answers_count += 1
            self.valid_citations_count += max(0, valid_citations)
            self.rejected_citations_count += max(0, rejected_citations)

    def record_provider_timeout(self) -> None:
        """Record provider timeout event."""
        with self._lock:
            self.timeout_count += 1

    def record_provider_rate_limit(self) -> None:
        """Record provider rate limit event."""
        with self._lock:
            self.rate_limit_count += 1

    def _safe_avg(self, seq: deque[Any]) -> float | None:
        if not seq:
            return None
        return round(sum(seq) / len(seq), 2)

    def get_snapshot(
        self,
        overall_status: str = "healthy",
        components: dict[str, ComponentStatus] | None = None,
    ) -> SystemMetricsResponse:
        """Assemble structured, typed metrics response."""
        with self._lock:
            uptime = round(max(0.0, time.time() - self._start_time), 2)

            system_health = SystemHealthMetrics(
                status=overall_status,  # type: ignore[arg-type]
                uptime_seconds=uptime,
                components=components or {},
            )

            request_telemetry = RequestTelemetryMetrics(
                total_requests=self.total_requests,
                successful_requests=self.successful_requests,
                failed_requests=self.failed_requests,
                status_codes=dict(self.status_codes),
                requests_by_endpoint=dict(self.requests_by_endpoint),
            )

            ingestion_metrics = IngestionMetrics(
                ingestion_count=self.ingestion_count,
                total_pages_processed=self.total_pages_processed,
                total_chunks_created=self.total_chunks_created,
                rendering_failures_count=self.rendering_failures_count,
                avg_duration_ms=self._safe_avg(self.ingestion_durations_ms),
            )

            retrieval_metrics = RetrievalMetrics(
                total_queries=self.total_retrieval_queries,
                empty_queries=self.empty_retrieval_count,
                avg_latency_ms=self._safe_avg(self.retrieval_durations_ms),
                avg_result_count=self._safe_avg(self.retrieval_result_counts),
            )

            reranking_metrics = RerankingMetrics(
                total_queries=self.total_rerankings,
                avg_latency_ms=self._safe_avg(self.reranking_durations_ms),
            )

            generation_metrics = GenerationMetrics(
                total_queries=self.total_generations,
                avg_latency_ms=self._safe_avg(self.generation_durations_ms),
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
                timeout_count=self.timeout_count,
                rate_limit_count=self.rate_limit_count,
            )

            pipeline_telemetry = PipelineTelemetryMetrics(
                ingestion=ingestion_metrics,
                retrieval=retrieval_metrics,
                reranking=reranking_metrics,
                generation=generation_metrics,
            )

            ai_quality = AIQualityMetrics(
                total_answers_generated=self.total_answers_generated,
                grounded_answers_count=self.grounded_answers_count,
                refusal_answers_count=self.refusal_answers_count,
                valid_citations_count=self.valid_citations_count,
                rejected_citations_count=self.rejected_citations_count,
            )

            return SystemMetricsResponse(
                status=overall_status,  # type: ignore[arg-type]
                version=settings.VERSION,
                timestamp=datetime.now(timezone.utc).isoformat(),
                system_health=system_health,
                request_telemetry=request_telemetry,
                pipeline_telemetry=pipeline_telemetry,
                ai_quality_metrics=ai_quality,
            )


# Global singleton instance
metrics_collector = MetricsCollector()

