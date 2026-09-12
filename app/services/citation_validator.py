import re
from typing import Any, Sequence, Union

from app.schemas.citation import (
    Citation,
    CitationValidationIssue,
    CitationValidationResult,
)
from app.schemas.retrieval import EvidenceItem, RetrievedChunk, RetrievedVisualPage

EvidenceInput = Union[RetrievedChunk, EvidenceItem, RetrievedVisualPage, dict[str, Any]]

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I do not have sufficient information in the provided document to answer this question."
)

REFUSAL_PHRASES = (
    INSUFFICIENT_EVIDENCE_ANSWER.lower(),
    "insufficient information",
    "insufficient evidence",
    "i do not have sufficient information",
    "does not contain sufficient information",
    "no information provided",
    "not mentioned in the provided",
)

# Regex to match single citation brackets like [Evidence 1], [evidence 2], [Evidence: 1], [1]
CITATION_TAG_PATTERN = re.compile(
    r"\[(?:Evidence\s*:?\s*)?(\d+)\]", re.IGNORECASE
)

# Regex to find any bracket contents that might represent citation tags (including commas, e.g. [1, 2] or [Evidence 1, 2])
CITATION_BRACKET_PATTERN = re.compile(
    r"\[\s*(?:Evidence\s*:?\s*)?(\d+(?:\s*,\s*(?:Evidence\s*:?\s*)?\d+)*)\s*\]",
    re.IGNORECASE,
)

# Regex to detect suspicious filesystem paths
LOCAL_PATH_PATTERN = re.compile(
    r"(?:/workspaces/|/home/|/tmp/|/var/|/etc/|[A-Za-z]:\\|storage/)", re.IGNORECASE
)

SENSITIVE_METADATA_KEYS = {
    "file_path",
    "filepath",
    "full_path",
    "storage_path",
    "local_path",
    "path",
    "api_key",
    "secret",
    "token",
    "password",
}


def is_refusal_response(answer: str) -> bool:
    """Detect whether generated text is an explicit refusal or indicates insufficient evidence."""
    clean = answer.strip().lower()
    return any(phrase in clean for phrase in REFUSAL_PHRASES)


class CitationValidator:
    """
    Validates evidence citations produced during answer generation against server-side evidence.
    Enforces document isolation, structural validity, reference accuracy, and security isolation.
    """

    def _get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Helper to safely retrieve an attribute or dict value."""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Strip internal filesystem paths and sensitive credentials from metadata."""
        if not isinstance(metadata, dict):
            return {}

        clean_meta: dict[str, Any] = {}
        for key, value in metadata.items():
            if str(key).lower() in SENSITIVE_METADATA_KEYS:
                continue

            # If value is a string looking like an absolute local filesystem path, sanitize or omit
            if isinstance(value, str) and LOCAL_PATH_PATTERN.search(value):
                continue

            clean_meta[key] = value

        return clean_meta

    def _sanitize_image_url(
        self, image_url: str | None, document_id: str, page_number: int
    ) -> str | None:
        """Sanitize visual page image URL to prevent local filesystem path leakage."""
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            return None

        clean_url = image_url.strip()

        # If it's a relative API route or standard HTTP URL, it is safe
        if clean_url.startswith("/api/") or clean_url.startswith("http://") or clean_url.startswith("https://"):
            return clean_url

        # If it contains local filesystem path strings, map to the safe API route
        if LOCAL_PATH_PATTERN.search(clean_url):
            return f"/api/v1/documents/{document_id}/pages/{page_number}/image"

        return clean_url

    def extract_citation_indices(self, text: str) -> list[int]:
        """
        Extract referenced 1-based evidence indices from text in order of appearance.
        Preserves encounter order and deduplicates indices.

        Supports:
        - [Evidence 1], [evidence 2], [Evidence: 1]
        - Numeric brackets: [1], [2]
        - Multi-citation brackets: [1, 2], [Evidence 1, Evidence 2]
        """
        if not text or not isinstance(text, str):
            return []

        indices: list[int] = []
        seen: set[int] = set()

        for bracket_match in CITATION_BRACKET_PATTERN.finditer(text):
            inner = bracket_match.group(1)
            # Find all numbers within this bracket
            for num_match in re.finditer(r"\d+", inner):
                idx = int(num_match.group(0))
                if idx not in seen:
                    seen.add(idx)
                    indices.append(idx)

        # Fallback single tag pattern if bracket pattern didn't catch standard tags
        for match in CITATION_TAG_PATTERN.finditer(text):
            idx = int(match.group(1))
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)

        return indices

    def build_citation(self, index: int, item: EvidenceInput) -> Citation:
        """
        Construct a structured Citation object with full provenance metadata.
        """
        doc_id = str(self._get_attr(item, "document_id", "") or "")
        page_num = int(self._get_attr(item, "page_number", 1) or 1)
        chunk_id = self._get_attr(item, "chunk_id", None)
        chunk_index = self._get_attr(item, "chunk_index", None)
        
        # Raw text / evidence text
        text_val = self._get_attr(item, "text", None) or self._get_attr(item, "evidence_text", None) or ""
        raw_img = self._get_attr(item, "image_url", None)
        sanitized_img = self._sanitize_image_url(raw_img, doc_id, page_num)
        
        # Score & Rank
        score = float(self._get_attr(item, "score", 0.0) or 0.0)
        rank = int(self._get_attr(item, "rank", index) or index)
        retrieval_type = str(self._get_attr(item, "retrieval_type", "evidence") or "evidence")
        
        # Sources
        sources = self._get_attr(item, "sources", []) or []
        if isinstance(sources, list):
            sources_list = [str(s) for s in sources]
        else:
            sources_list = [str(sources)]

        raw_meta = self._get_attr(item, "metadata", {}) or {}
        sanitized_meta = self._sanitize_metadata(raw_meta)

        return Citation(
            reference=f"[Evidence {index}]",
            rank=rank,
            score=score,
            chunk_id=chunk_id,
            document_id=doc_id,
            page_number=page_num,
            chunk_index=chunk_index,
            evidence_text=text_val,
            text=text_val,
            image_url=sanitized_img,
            retrieval_type=retrieval_type,
            sources=sources_list,
            metadata=sanitized_meta,
        )

    def validate_citation_item(
        self,
        item: EvidenceInput,
        target_document_id: str | None = None,
    ) -> CitationValidationIssue | None:
        """
        Validate an evidence item's integrity and document isolation.
        Returns a CitationValidationIssue if invalid, None if valid.
        """
        doc_id = self._get_attr(item, "document_id", "")
        if not doc_id or not isinstance(doc_id, str) or not doc_id.strip():
            return CitationValidationIssue(
                reference="[Unknown]",
                issue_type="corrupt_evidence",
                message="Evidence item is missing document_id",
            )

        if target_document_id and doc_id != target_document_id:
            return CitationValidationIssue(
                reference="[Evidence]",
                issue_type="wrong_document_id",
                message=f"Evidence belongs to document '{doc_id}', expected '{target_document_id}'",
            )

        page_num = self._get_attr(item, "page_number", 0)
        if page_num is None or not isinstance(page_num, int) or page_num < 1:
            return CitationValidationIssue(
                reference="[Evidence]",
                issue_type="invalid_page_number",
                message=f"Evidence has invalid page_number: {page_num}",
            )

        # Must have either text or image_url
        text = self._get_attr(item, "text", "") or self._get_attr(item, "evidence_text", "") or ""
        image_url = self._get_attr(item, "image_url", "") or ""
        if not str(text).strip() and not str(image_url).strip():
            return CitationValidationIssue(
                reference="[Evidence]",
                issue_type="corrupt_evidence",
                message="Evidence item contains neither text nor image_url",
            )

        return None

    def validate_citations(
        self,
        answer: str,
        evidence: Sequence[EvidenceInput],
        target_document_id: str | None = None,
    ) -> CitationValidationResult:
        """
        Validate all citation references in the generated answer against server-side evidence.

        Args:
            answer: Generated answer text from LLM/VLM.
            evidence: Sequence of retrieved & validated evidence items.
            target_document_id: Optional document ID for strict document isolation.

        Returns:
            CitationValidationResult with validated citations, grounding flag, and issues.
        """
        if not evidence or is_refusal_response(answer):
            return CitationValidationResult(
                citations=[],
                is_grounded=False,
                valid_count=0,
                invalid_count=0,
                issues=[],
            )

        # Build 1-based server-side evidence map
        evidence_map: dict[int, EvidenceInput] = {
            idx: item for idx, item in enumerate(evidence, start=1)
        }

        cited_indices = self.extract_citation_indices(answer)
        valid_citations: list[Citation] = []
        issues: list[CitationValidationIssue] = []
        seen_citation_keys: set[str] = set()

        if cited_indices:
            for idx in cited_indices:
                ref_tag = f"[Evidence {idx}]"

                # Check if evidence index exists
                if idx not in evidence_map:
                    issues.append(
                        CitationValidationIssue(
                            reference=ref_tag,
                            issue_type="fabricated_id" if idx > len(evidence) else "out_of_range",
                            message=f"Citation {ref_tag} references nonexistent evidence index {idx} (available: 1..{len(evidence)})",
                            raw_index=idx,
                        )
                    )
                    continue

                item = evidence_map[idx]
                item_issue = self.validate_citation_item(item, target_document_id)
                if item_issue:
                    item_issue.reference = ref_tag
                    item_issue.raw_index = idx
                    issues.append(item_issue)
                    continue

                # Build and deduplicate citation
                citation = self.build_citation(idx, item)
                dedup_key = f"{citation.document_id}::{citation.page_number}::{citation.chunk_id or citation.image_url}"
                if dedup_key not in seen_citation_keys:
                    seen_citation_keys.add(dedup_key)
                    valid_citations.append(citation)

            # If all explicit citations were fabricated/invalid, answer is ungrounded
            is_grounded = len(valid_citations) > 0

            return CitationValidationResult(
                citations=valid_citations,
                is_grounded=is_grounded,
                valid_count=len(valid_citations),
                invalid_count=len(issues),
                issues=issues,
            )

        # Fallback: answer has no explicit citation tags and is not a refusal.
        # Deterministically validate all retrieved evidence items.
        for idx, item in enumerate(evidence, start=1):
            ref_tag = f"[Evidence {idx}]"
            item_issue = self.validate_citation_item(item, target_document_id)
            if item_issue:
                item_issue.reference = ref_tag
                item_issue.raw_index = idx
                issues.append(item_issue)
                continue

            citation = self.build_citation(idx, item)
            dedup_key = f"{citation.document_id}::{citation.page_number}::{citation.chunk_id or citation.image_url}"
            if dedup_key not in seen_citation_keys:
                seen_citation_keys.add(dedup_key)
                valid_citations.append(citation)

        is_grounded = len(valid_citations) > 0

        return CitationValidationResult(
            citations=valid_citations,
            is_grounded=is_grounded,
            valid_count=len(valid_citations),
            invalid_count=len(issues),
            issues=issues,
        )


# Aliases & Singletons
CitationBuilder = CitationValidator
citation_validator = CitationValidator()


def extract_citation_indices(text: str) -> list[int]:
    """Module-level helper to extract citation indices from text."""
    return citation_validator.extract_citation_indices(text)


def validate_citations(
    answer: str,
    evidence: Sequence[EvidenceInput],
    target_document_id: str | None = None,
) -> CitationValidationResult:
    """Module-level helper to validate citations against evidence."""
    return citation_validator.validate_citations(
        answer=answer,
        evidence=evidence,
        target_document_id=target_document_id,
    )
