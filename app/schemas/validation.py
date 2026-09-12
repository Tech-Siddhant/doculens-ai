from pydantic import BaseModel, Field
from app.schemas.retrieval import EvidenceItem


class EvidenceValidationIssue(BaseModel):
    """Describes a single validation issue with an evidence item."""
    item_rank: int
    issue_type: str = Field(..., description="Type of validation issue (e.g., 'missing_document_id', 'invalid_page_number')")
    message: str = Field(..., description="Human-readable issue description")


class EvidenceValidationResult(BaseModel):
    """Result of evidence validation with valid items and any issues found."""
    valid_evidence: list[EvidenceItem]
    invalid_count: int
    issues: list[EvidenceValidationIssue]
    total_validated: int
