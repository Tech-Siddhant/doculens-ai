from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Structured citation referencing a specific retrieved document chunk."""

    reference: str = Field(..., description="Evidence reference tag, e.g. [Evidence 1]")
    rank: int = Field(..., ge=1, description="1-based retrieval ranking position")
    score: float = Field(..., description="Cosine similarity score")
    chunk_id: str = Field(..., description="Deterministic chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: int = Field(..., ge=1, description="1-based page number citation")
    chunk_index: int = Field(..., ge=0, description="0-based chunk index within page")
    evidence_text: str = Field(..., description="Text excerpt used as grounding evidence")
    text: str | None = Field(default=None, description="Alias for evidence_text")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional chunk metadata"
    )

    def model_post_init(self, __context: Any) -> None:
        if self.text is None:
            self.text = self.evidence_text


# Backward compatibility alias
EvidenceCitation = Citation


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

