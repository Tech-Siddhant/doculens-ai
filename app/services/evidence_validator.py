from typing import Sequence
from app.schemas.retrieval import EvidenceItem
from app.schemas.validation import EvidenceValidationIssue, EvidenceValidationResult


class EvidenceValidator:
    """
    Validates evidence items before context assembly or generation.
    Enforces document isolation, structural validity, and content traceability.
    """

    def _validate_item(self, item: EvidenceItem, target_document_id: str | None = None) -> EvidenceValidationIssue | None:
        """
        Validate a single evidence item. Returns an issue if invalid, None if valid.
        """
        # 1. Document ID presence and format
        if not item.document_id or not isinstance(item.document_id, str) or not item.document_id.strip():
            return EvidenceValidationIssue(
                item_rank=item.rank,
                issue_type="missing_document_id",
                message=f"Evidence item rank {item.rank} has missing or empty document_id"
            )

        # 2. Document isolation - cross-document evidence rejection
        if target_document_id and item.document_id != target_document_id:
            return EvidenceValidationIssue(
                item_rank=item.rank,
                issue_type="wrong_document_id",
                message=f"Evidence item rank {item.rank} belongs to document '{item.document_id}' but expected '{target_document_id}'"
            )

        # 3. Page number validity
        if item.page_number < 1:
            return EvidenceValidationIssue(
                item_rank=item.rank,
                issue_type="invalid_page_number",
                message=f"Evidence item rank {item.rank} has invalid page_number {item.page_number} (must be >= 1)"
            )

        # 4. Chunk ID format validation (if present)
        if item.chunk_id is not None:
            if not isinstance(item.chunk_id, str) or not item.chunk_id.strip():
                return EvidenceValidationIssue(
                    item_rank=item.rank,
                    issue_type="malformed_chunk_id",
                    message=f"Evidence item rank {item.rank} has malformed chunk_id"
                )

        # 5. Content validation - must have text or image_url
        has_text = item.text and isinstance(item.text, str) and item.text.strip()
        has_image = item.image_url and isinstance(item.image_url, str) and item.image_url.strip()
        
        if not has_text and not has_image:
            return EvidenceValidationIssue(
                item_rank=item.rank,
                issue_type="empty_evidence",
                message=f"Evidence item rank {item.rank} has no usable text or image_url"
            )

        # 6. Sources validation
        if not item.sources or not isinstance(item.sources, list):
            return EvidenceValidationIssue(
                item_rank=item.rank,
                issue_type="missing_sources",
                message=f"Evidence item rank {item.rank} has missing or invalid sources metadata"
            )

        return None

    def validate_evidence(
        self,
        evidence: Sequence[EvidenceItem],
        target_document_id: str | None = None,
        strict: bool = False
    ) -> EvidenceValidationResult:
        """
        Validate a sequence of evidence items.
        
        Args:
            evidence: Sequence of EvidenceItem objects to validate
            target_document_id: Optional document ID for document isolation enforcement
            strict: If True, raises exception on first validation failure (default False)
        
        Returns:
            EvidenceValidationResult with valid items and any issues found
        """
        valid_items: list[EvidenceItem] = []
        issues: list[EvidenceValidationIssue] = []
        seen_keys: set[str] = set()

        for item in evidence:
            # Validate item
            issue = self._validate_item(item, target_document_id)
            
            if issue:
                issues.append(issue)
                if strict:
                    raise ValueError(f"Evidence validation failed: {issue.message}")
                continue

            # Deduplication check (should already be done by evidence selector, but defense in depth)
            key = f"{item.document_id}::{item.page_number}::{item.chunk_id or 'page'}"
            if key in seen_keys:
                issues.append(EvidenceValidationIssue(
                    item_rank=item.rank,
                    issue_type="duplicate_evidence",
                    message=f"Evidence item rank {item.rank} is a duplicate"
                ))
                continue
            seen_keys.add(key)

            valid_items.append(item)

        return EvidenceValidationResult(
            valid_evidence=valid_items,
            invalid_count=len(issues),
            issues=issues,
            total_validated=len(evidence)
        )


# Singleton instance
evidence_validator = EvidenceValidator()
