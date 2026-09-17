import logging
from pathlib import Path

from fastapi import APIRouter, Response, status

from app.core.config import settings
from app.core.metrics import metrics_collector
from app.schemas.health import (
    ComponentStatus,
    HealthResponse,
    ReadinessResponse,
    SystemMetricsResponse,
)
from app.services.vector_store import vector_store
from app.services.visual_vector_store import visual_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


def check_dependencies() -> tuple[str, dict[str, ComponentStatus]]:
    """Inspect critical backend dependencies and evaluate overall readiness status."""
    components: dict[str, ComponentStatus] = {}
    is_ready = True

    # 1. Storage check (upload directory accessible and writable)
    try:
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        probe_file = upload_dir / ".health_probe"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
        components["storage"] = ComponentStatus(status="healthy", details="Upload directory is writable")
    except Exception as exc:
        is_ready = False
        components["storage"] = ComponentStatus(status="unhealthy", details=f"Storage unavailable: {type(exc).__name__}")

    # 2. Vector store (Qdrant chunk collection)
    try:
        stats = vector_store.get_stats()
        components["vector_store"] = ComponentStatus(
            status="healthy",
            details=f"Collection '{stats.collection_name}' ready ({stats.total_points} chunks)",
        )
    except Exception as exc:
        is_ready = False
        components["vector_store"] = ComponentStatus(
            status="unhealthy",
            details=f"Vector store unavailable: {type(exc).__name__}",
        )

    # 3. Visual vector store (Qdrant visual page collection)
    try:
        vstats = visual_vector_store.get_stats()
        components["visual_vector_store"] = ComponentStatus(
            status="healthy",
            details=f"Collection '{vstats.collection_name}' ready ({vstats.total_points} pages)",
        )
    except Exception as exc:
        is_ready = False
        components["visual_vector_store"] = ComponentStatus(
            status="unhealthy",
            details=f"Visual vector store unavailable: {type(exc).__name__}",
        )

    # 4. LLM provider configuration check
    try:
        provider = settings.LLM_PROVIDER
        if provider == "mock":
            components["llm_provider"] = ComponentStatus(status="healthy", details="Mock provider active")
        elif provider == "gemini":
            if settings.GEMINI_API_KEY or settings.LLM_API_KEY:
                components["llm_provider"] = ComponentStatus(status="healthy", details="Gemini provider configured")
            else:
                is_ready = False
                components["llm_provider"] = ComponentStatus(status="unhealthy", details="Missing GEMINI_API_KEY")
        elif provider == "openai":
            if settings.LLM_API_KEY:
                components["llm_provider"] = ComponentStatus(status="healthy", details="OpenAI provider configured")
            else:
                is_ready = False
                components["llm_provider"] = ComponentStatus(status="unhealthy", details="Missing LLM_API_KEY")
        else:
            components["llm_provider"] = ComponentStatus(status="degraded", details=f"Unknown provider: {provider}")
    except Exception as exc:
        is_ready = False
        components["llm_provider"] = ComponentStatus(status="unhealthy", details=f"LLM config check failed: {type(exc).__name__}")

    overall_status = "healthy" if is_ready else "unhealthy"
    return overall_status, components


@router.get("/health", response_model=HealthResponse)
@router.get("/health/live", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Fast liveness check indicating the application process is running."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
    )


@router.get("/health/ready", response_model=ReadinessResponse)
@router.get("/health/readiness", response_model=ReadinessResponse)
async def readiness_check(response: Response) -> ReadinessResponse:
    """Deep readiness check validating storage, vector stores, and LLM provider."""
    overall_status, components = check_dependencies()
    if overall_status == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status=overall_status,  # type: ignore[arg-type]
        version=settings.VERSION,
        components=components,
    )


@router.get("/health/metrics", response_model=SystemMetricsResponse)
@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_metrics(response: Response) -> SystemMetricsResponse:
    """Unified system health, HTTP request telemetry, and RAG pipeline metrics."""
    overall_status, components = check_dependencies()
    if overall_status == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return metrics_collector.get_snapshot(
        overall_status=overall_status,
        components=components,
    )



