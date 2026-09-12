from typing import Any

from pydantic import BaseModel, Field

from app.schemas.citation import (
    Citation,
    CitationValidationIssue,
    CitationValidationResult,
    EvidenceCitation,
)


class QuestionRequest(BaseModel):
    """User question payload for document question answering."""

    question: str = Field(..., min_length=1, description="User question to answer")
    top_k: int = Field(
        default=5, ge=1, description="Maximum number of evidence chunks to retrieve"
    )
    score_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity score threshold",
    )


class GenerationResult(BaseModel):
    """Final grounded answer with provenance citations and model metadata."""

    question: str = Field(..., description="User query")
    answer: str = Field(..., description="Generated answer text")
    document_id: str | None = Field(
        default=None, description="Document ID queried, if scoped"
    )
    model: str = Field(..., description="LLM model identifier used for generation")
    provider: str = Field(..., description="LLM provider name (e.g. mock, openai)")
    citations: list[Citation] = Field(
        default_factory=list, description="Validated evidence citations for the answer"
    )
    is_grounded: bool = Field(
        default=True, description="Whether answer was generated from retrieved evidence"
    )
    usage: dict[str, Any] = Field(
        default_factory=dict, description="Token usage and generation metadata"
    )


__all__ = [
    "Citation",
    "EvidenceCitation",
    "CitationValidationIssue",
    "CitationValidationResult",
    "QuestionRequest",
    "GenerationResult",
]


