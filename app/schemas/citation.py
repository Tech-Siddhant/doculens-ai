from typing import Any
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Structured citation referencing a specific retrieved document chunk or visual page."""

    reference: str = Field(..., description="Evidence reference tag, e.g. [Evidence 1]")
    rank: int = Field(..., ge=1, description="1-based retrieval ranking position")
    score: float = Field(..., description="Cosine similarity score or reranker relevance score")
    chunk_id: str | None = Field(
        default=None, description="Deterministic chunk identifier if text chunk"
    )
    document_id: str = Field(..., description="Parent document identifier")
    page_number: int = Field(..., ge=1, description="1-based page number citation")
    chunk_index: int | None = Field(
        default=None, ge=0, description="0-based chunk index within page"
    )
    evidence_text: str = Field(
        default="", description="Text excerpt used as grounding evidence"
    )
    text: str | None = Field(default=None, description="Alias for evidence_text")
    image_url: str | None = Field(
        default=None, description="Relative URL for visual page evidence"
    )
    retrieval_type: str = Field(
        default="evidence", description="Evidence classification identifier"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Original retrieval channels (e.g. ['dense', 'bm25', 'visual'])",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk/page payload and citation metadata",
    )

    def model_post_init(self, __context: Any) -> None:
        if self.text is None:
            self.text = self.evidence_text
        elif not self.evidence_text and self.text:
            self.evidence_text = self.text


# Backward compatibility alias
EvidenceCitation = Citation


class CitationValidationIssue(BaseModel):
    """Describes a single validation issue with a citation."""

    reference: str = Field(
        ..., description="The citation reference tag that caused the issue"
    )
    issue_type: str = Field(
        ...,
        description="Issue type: 'fabricated_id', 'out_of_range', 'wrong_document_id', 'invalid_page_number', 'corrupt_evidence', 'path_leakage', 'duplicate_citation'",
    )
    message: str = Field(..., description="Human-readable issue description")
    raw_index: int | None = Field(
        default=None, description="Extracted index if numeric"
    )


class CitationValidationResult(BaseModel):
    """Result of citation validation with valid citations and issues."""

    citations: list[Citation] = Field(
        default_factory=list, description="Validated citations"
    )
    is_grounded: bool = Field(
        default=False,
        description="Whether the answer is considered grounded in valid evidence",
    )
    valid_count: int = Field(
        default=0, description="Count of valid citations accepted"
    )
    invalid_count: int = Field(
        default=0, description="Count of invalid citations rejected"
    )
    issues: list[CitationValidationIssue] = Field(
        default_factory=list, description="List of citation validation issues"
    )
