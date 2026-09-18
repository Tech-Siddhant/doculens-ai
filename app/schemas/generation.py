from typing import Any, Literal
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.citation import (
    Citation,
    CitationValidationIssue,
    CitationValidationResult,
    EvidenceCitation,
)

class QuestionRequest(BaseModel):
    """User question payload for document question answering."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question to answer (1-2000 characters)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of evidence chunks to retrieve (1-50)",
    )
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0, description="Optional minimum cosine similarity score threshold")
    answer_style: Literal["concise", "balanced", "detailed"] = Field(
        default="balanced",
        description="Target answer style and verbosity: concise (short direct facts), balanced (conversational explanation), or detailed (in-depth structured breakdown)"
    )

class PipelineStageStatus(str, Enum):
    SUCCESS = "success"
    FALLBACK = "fallback"
    FAILED = "failed"
    SKIPPED = "skipped"

class PipelineStageTrace(BaseModel):
    stage_id: str = Field(..., description="Unique machine-readable stage ID")
    stage_name: str = Field(..., description="Human-readable stage title")
    order: int = Field(..., ge=1, description="1-based sequence order in pipeline execution")
    status: Literal["success", "fallback", "failed", "skipped"] = Field(default="success", description="Outcome status of the stage")
    duration_ms: float = Field(default=0.0, ge=0.0, description="Measured execution duration in milliseconds")
    input_count: int | None = Field(default=None, ge=0, description="Input entity or item count")
    output_count: int | None = Field(default=None, ge=0, description="Output entity or item count")
    error_category: str | None = Field(default=None, description="Category or classification of error or fallback")
    description: str = Field(..., description="Concise observable summary of stage execution")
    fallback_used: bool = Field(default=False, description="Whether fallback logic was activated")
    fallback_reason: str | None = Field(default=None, description="Concise explanation if fallback was used")
    error_message: str | None = Field(default=None, description="Safe user-facing error description if stage failed")
    details: dict[str, Any] = Field(default_factory=dict, description="Observable telemetry metrics")

class PipelineSummary(BaseModel):
    total_duration_ms: float = Field(default=0.0, ge=0.0, description="Total measured pipeline wall-clock time in ms")
    retrieval_duration_ms: float | None = Field(default=None, description="Retrieval stage duration in ms")
    reranking_duration_ms: float | None = Field(default=None, description="Reranking stage duration in ms")
    generation_duration_ms: float | None = Field(default=None, description="Generation stage duration in ms")
    citation_validation_duration_ms: float | None = Field(default=None, description="Citation validation stage duration in ms")
    total_stages: int = Field(default=0, ge=0, description="Total executed stages")
    successful_stages: int = Field(default=0, ge=0, description="Number of successful stages")
    failed_stages: int = Field(default=0, ge=0, description="Number of failed stages")
    fallback_stages: int = Field(default=0, ge=0, description="Number of stages using fallback")

class PipelineTrace(BaseModel):
    pipeline_id: str = Field(..., description="Unique identifier for this pipeline execution")
    stages: list[PipelineStageTrace] = Field(default_factory=list, description="Ordered execution trace of each stage")
    summary: PipelineSummary = Field(default_factory=PipelineSummary, description="Summary latency and operational rollups")

class GenerationResult(BaseModel):
    """Final grounded answer with provenance citations and model metadata."""
    question: str = Field(..., description="User query")
    answer: str = Field(..., description="Generated answer text")
    document_id: str | None = Field(default=None, description="Document ID queried, if scoped")
    model: str = Field(..., description="LLM model identifier used for generation")
    provider: str = Field(..., description="LLM provider name (e.g. mock, openai)")
    citations: list[Citation] = Field(default_factory=list, description="Validated evidence citations for the answer")
    is_grounded: bool = Field(default=True, description="Whether answer was generated from retrieved evidence")
    usage: dict[str, Any] = Field(default_factory=dict, description="Token usage and generation metadata")
    pipeline_trace: PipelineTrace | None = Field(default=None, description="Observable system execution telemetry")

__all__ = [
    "Citation",
    "EvidenceCitation",
    "CitationValidationIssue",
    "CitationValidationResult",
    "QuestionRequest",
    "PipelineStageStatus",
    "PipelineStageTrace",
    "PipelineSummary",
    "PipelineTrace",
    "GenerationResult"
]
