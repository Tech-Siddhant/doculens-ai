"""
Unit tests for Phase 5.6 Citation Validation.
Ensures citations produced by grounded generation are valid, traceable, isolated,
and correspond to server-side retrieved evidence.
"""

from typing import Any
import pytest

from app.schemas.citation import (
    Citation,
    CitationValidationIssue,
    CitationValidationResult,
)
from app.schemas.retrieval import EvidenceItem, RetrievedChunk, RetrievedVisualPage
from app.services.citation_validator import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    CitationBuilder,
    CitationValidator,
    citation_validator,
    extract_citation_indices,
    is_refusal_response,
    validate_citations,
)
from app.services.generator import AnswerGenerator
from app.services.llm_provider import MockLLMProvider


@pytest.fixture
def sample_text_evidence() -> list[EvidenceItem]:
    """Sample text-based evidence items."""
    return [
        EvidenceItem(
            rank=1,
            score=0.92,
            initial_rank=1,
            initial_score=0.92,
            document_id="doc_valid_123",
            page_number=1,
            chunk_id="doc_valid_123_p1_c0",
            chunk_index=0,
            text="DocuLens AI utilizes dense embeddings and Qdrant for semantic search.",
            image_url=None,
            retrieval_type="evidence",
            sources=["dense"],
            raw_scores={"dense": 0.92},
            normalized_scores={"dense": 0.92},
            metadata={"char_count": 72, "section": "Introduction"},
        ),
        EvidenceItem(
            rank=2,
            score=0.85,
            initial_rank=2,
            initial_score=0.85,
            document_id="doc_valid_123",
            page_number=2,
            chunk_id="doc_valid_123_p2_c0",
            chunk_index=0,
            text="BM25 sparse search handles exact keyword matches effectively.",
            image_url=None,
            retrieval_type="evidence",
            sources=["bm25"],
            raw_scores={"bm25": 14.5},
            normalized_scores={"bm25": 0.85},
            metadata={"char_count": 62, "section": "Retrieval"},
        ),
    ]


@pytest.fixture
def sample_visual_evidence() -> list[EvidenceItem]:
    """Sample visual page evidence item."""
    return [
        EvidenceItem(
            rank=1,
            score=0.88,
            initial_rank=1,
            initial_score=0.88,
            document_id="doc_valid_123",
            page_number=3,
            chunk_id=None,
            chunk_index=None,
            text=None,
            image_url="/api/v1/documents/doc_valid_123/pages/3/image",
            retrieval_type="evidence",
            sources=["visual"],
            raw_scores={"visual": 0.88},
            normalized_scores={"visual": 0.88},
            metadata={"width": 800, "height": 1100, "dpi": 150},
        ),
    ]


@pytest.fixture
def sample_mixed_evidence() -> list[EvidenceItem]:
    """Sample mixed text and visual evidence items."""
    return [
        EvidenceItem(
            rank=1,
            score=0.95,
            initial_rank=1,
            initial_score=0.95,
            document_id="doc_valid_123",
            page_number=1,
            chunk_id="doc_valid_123_p1_c0",
            chunk_index=0,
            text="Cross-encoder reranking re-scores retrieved candidates for optimal precision.",
            image_url=None,
            retrieval_type="evidence",
            sources=["dense", "reranked"],
            raw_scores={"rerank": 0.95},
            normalized_scores={"rerank": 0.95},
            metadata={"char_count": 78},
        ),
        EvidenceItem(
            rank=2,
            score=0.89,
            initial_rank=2,
            initial_score=0.89,
            document_id="doc_valid_123",
            page_number=4,
            chunk_id=None,
            chunk_index=None,
            text=None,
            image_url="/api/v1/documents/doc_valid_123/pages/4/image",
            retrieval_type="evidence",
            sources=["visual"],
            raw_scores={"visual": 0.89},
            normalized_scores={"visual": 0.89},
            metadata={"figure": "Figure 2: Performance Graph"},
        ),
    ]


# 1. Valid Citation
def test_citation_validation_single_valid(sample_text_evidence: list[EvidenceItem]) -> None:
    answer = "DocuLens AI utilizes dense embeddings for semantic search [Evidence 1]."
    result = validate_citations(answer, sample_text_evidence, target_document_id="doc_valid_123")

    assert isinstance(result, CitationValidationResult)
    assert result.is_grounded is True
    assert result.valid_count == 1
    assert result.invalid_count == 0
    assert len(result.citations) == 1
    assert result.issues == []

    cit = result.citations[0]
    assert cit.reference == "[Evidence 1]"
    assert cit.rank == 1
    assert cit.score == 0.92
    assert cit.chunk_id == "doc_valid_123_p1_c0"
    assert cit.document_id == "doc_valid_123"
    assert cit.page_number == 1
    assert cit.chunk_index == 0
    assert "dense embeddings" in cit.evidence_text
    assert cit.metadata["section"] == "Introduction"


# 2. Multiple Valid Citations
def test_citation_validation_multiple_valid(sample_text_evidence: list[EvidenceItem]) -> None:
    answer = "Semantic search uses dense vectors [Evidence 1], while keyword search uses BM25 [Evidence 2]."
    result = validate_citations(answer, sample_text_evidence, target_document_id="doc_valid_123")

    assert result.is_grounded is True
    assert result.valid_count == 2
    assert result.invalid_count == 0
    assert len(result.citations) == 2

    assert result.citations[0].reference == "[Evidence 1]"
    assert result.citations[0].chunk_id == "doc_valid_123_p1_c0"
    assert result.citations[1].reference == "[Evidence 2]"
    assert result.citations[1].chunk_id == "doc_valid_123_p2_c0"


# 3. Fabricated Evidence ID
def test_citation_validation_fabricated_evidence_id(sample_text_evidence: list[EvidenceItem]) -> None:
    answer = "DocuLens performs magic hallucinated reasoning [Evidence 99]."
    result = validate_citations(answer, sample_text_evidence, target_document_id="doc_valid_123")

    assert result.is_grounded is False
    assert result.valid_count == 0
    assert result.invalid_count == 1
    assert len(result.citations) == 0
    assert len(result.issues) == 1
    assert result.issues[0].issue_type in ("fabricated_id", "out_of_range")
    assert result.issues[0].reference == "[Evidence 99]"


# 4. Out-of-Range Evidence Index
def test_citation_validation_out_of_range_index(sample_text_evidence: list[EvidenceItem]) -> None:
    answer = "DocuLens handles out-of-range citations [Evidence 0] and [Evidence 10]."
    result = validate_citations(answer, sample_text_evidence, target_document_id="doc_valid_123")

    assert result.is_grounded is False
    assert result.valid_count == 0
    assert result.invalid_count == 2
    assert len(result.citations) == 0
    assert len(result.issues) == 2


# 5. Nonexistent / Corrupt Evidence
def test_citation_validation_corrupt_evidence_item() -> None:
    corrupt_dict: dict[str, Any] = {
        "rank": 1,
        "score": 0.5,
        "document_id": "doc_valid_123",
        "page_number": 1,
        "chunk_id": None,
        "text": "",  # Empty text
        "image_url": None,  # No image
        "retrieval_type": "evidence",
        "sources": [],
    }
    answer = "Corrupted item claim [Evidence 1]."
    result = validate_citations(answer, [corrupt_dict], target_document_id="doc_valid_123")

    assert result.is_grounded is False
    assert result.valid_count == 0
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "corrupt_evidence"


# 6. Wrong Document ID (Document Isolation)
def test_citation_validation_wrong_document_id(sample_text_evidence: list[EvidenceItem]) -> None:
    answer = "DocuLens semantic search is verified [Evidence 1]."
    # Requested target_document_id is doc_another_456, but evidence belongs to doc_valid_123
    result = validate_citations(answer, sample_text_evidence, target_document_id="doc_another_456")

    assert result.is_grounded is False
    assert result.valid_count == 0
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "wrong_document_id"


# 7. Mismatched / Invalid Page Number
def test_citation_validation_invalid_page_number() -> None:
    invalid_page_dict: dict[str, Any] = {
        "rank": 1,
        "score": 0.8,
        "document_id": "doc_valid_123",
        "page_number": 0,  # Invalid: page < 1
        "chunk_id": "doc_valid_123_p0_c0",
        "chunk_index": 0,
        "text": "Invalid page text content.",
        "image_url": None,
        "retrieval_type": "evidence",
        "sources": ["dense"],
    }
    answer = "Claim on page 0 [Evidence 1]."
    result = validate_citations(answer, [invalid_page_dict], target_document_id="doc_valid_123")

    assert result.is_grounded is False
    assert result.valid_count == 0
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "invalid_page_number"


# 8. Malformed Citation
def test_citation_validation_malformed_citation(sample_text_evidence: list[EvidenceItem]) -> None:
    answer = (
        "DocuLens handles malformed [Evidence abc], non-numeric [Evidence NaN], "
        "and valid brackets [Evidence 1]."
    )
    result = validate_citations(answer, sample_text_evidence, target_document_id="doc_valid_123")

    assert result.is_grounded is True
    assert result.valid_count == 1
    assert len(result.citations) == 1
    assert result.citations[0].reference == "[Evidence 1]"


# 9. Duplicate Citations
def test_citation_validation_duplicate_citations(sample_text_evidence: list[EvidenceItem]) -> None:
    answer = (
        "DocuLens uses dense embeddings [Evidence 1], repeated reference [Evidence 1], "
        "and numeric reference [1]."
    )
    result = validate_citations(answer, sample_text_evidence, target_document_id="doc_valid_123")

    assert result.is_grounded is True
    assert result.valid_count == 1
    assert len(result.citations) == 1
    assert result.citations[0].reference == "[Evidence 1]"


# 10. Answer with No Citations
def test_citation_validation_no_citations_fallback(sample_text_evidence: list[EvidenceItem]) -> None:
    answer = "DocuLens AI provides grounded answers across complex technical PDFs."
    result = validate_citations(answer, sample_text_evidence, target_document_id="doc_valid_123")

    assert result.is_grounded is True
    assert result.valid_count == 2
    assert len(result.citations) == 2
    assert result.citations[0].reference == "[Evidence 1]"
    assert result.citations[1].reference == "[Evidence 2]"


# 11. Insufficient-Evidence Answer (Refusal)
def test_citation_validation_insufficient_evidence_refusal(sample_text_evidence: list[EvidenceItem]) -> None:
    answer = INSUFFICIENT_EVIDENCE_ANSWER
    result = validate_citations(answer, sample_text_evidence, target_document_id="doc_valid_123")

    assert result.is_grounded is False
    assert result.valid_count == 0
    assert result.citations == []
    assert result.issues == []


# 12. Visual Page Citation
def test_citation_validation_visual_page_citation(sample_visual_evidence: list[EvidenceItem]) -> None:
    answer = "The architecture diagram illustrates the multimodal pipeline [Evidence 1]."
    result = validate_citations(answer, sample_visual_evidence, target_document_id="doc_valid_123")

    assert result.is_grounded is True
    assert result.valid_count == 1
    assert len(result.citations) == 1

    cit = result.citations[0]
    assert cit.reference == "[Evidence 1]"
    assert cit.document_id == "doc_valid_123"
    assert cit.page_number == 3
    assert cit.chunk_id is None
    assert cit.image_url == "/api/v1/documents/doc_valid_123/pages/3/image"
    assert cit.sources == ["visual"]
    assert cit.metadata["dpi"] == 150


# 13. Mixed Text / Visual Evidence
def test_citation_validation_mixed_text_and_visual_evidence(sample_mixed_evidence: list[EvidenceItem]) -> None:
    answer = (
        "Reranking re-scores candidates [Evidence 1], and Figure 2 displays the benchmark curve [Evidence 2]."
    )
    result = validate_citations(answer, sample_mixed_evidence, target_document_id="doc_valid_123")

    assert result.is_grounded is True
    assert result.valid_count == 2
    assert len(result.citations) == 2

    # Text citation
    assert result.citations[0].reference == "[Evidence 1]"
    assert result.citations[0].chunk_id == "doc_valid_123_p1_c0"
    assert result.citations[0].page_number == 1
    assert "Cross-encoder reranking" in result.citations[0].evidence_text

    # Visual citation
    assert result.citations[1].reference == "[Evidence 2]"
    assert result.citations[1].chunk_id is None
    assert result.citations[1].page_number == 4
    assert result.citations[1].image_url == "/api/v1/documents/doc_valid_123/pages/4/image"
    assert result.citations[1].sources == ["visual"]


# 14. Deterministic Citation Mapping
def test_citation_validation_deterministic_mapping(sample_mixed_evidence: list[EvidenceItem]) -> None:
    answer = "Claim from visual page [Evidence 2] and claim from text [Evidence 1]."

    result1 = validate_citations(answer, sample_mixed_evidence, target_document_id="doc_valid_123")
    result2 = validate_citations(answer, sample_mixed_evidence, target_document_id="doc_valid_123")

    assert result1.is_grounded == result2.is_grounded
    assert result1.valid_count == result2.valid_count
    assert len(result1.citations) == len(result2.citations)

    for c1, c2 in zip(result1.citations, result2.citations):
        assert c1.reference == c2.reference
        assert c1.score == c2.score
        assert c1.chunk_id == c2.chunk_id
        assert c1.document_id == c2.document_id
        assert c1.page_number == c2.page_number
        assert c1.image_url == c2.image_url


# 15. No Filesystem Path Leakage
def test_citation_validation_no_filesystem_path_leakage() -> None:
    unsafe_item = EvidenceItem(
        rank=1,
        score=0.9,
        initial_rank=1,
        initial_score=0.9,
        document_id="doc_secure_789",
        page_number=5,
        chunk_id="doc_secure_789_p5_c0",
        chunk_index=0,
        text="Secure document snippet.",
        image_url="/workspaces/doculens-ai/storage/pages/doc_secure_789_p5.png",
        retrieval_type="evidence",
        sources=["visual"],
        metadata={
            "file_path": "/workspaces/doculens-ai/storage/uploads/secret.pdf",
            "storage_path": "/var/data/storage/doc.pdf",
            "api_key": "ai-secret-token-value",
            "token": "bearer-12345",
            "dpi": 200,
            "heading": "Security Analysis",
        },
    )

    answer = "Security information is presented [Evidence 1]."
    result = validate_citations(answer, [unsafe_item], target_document_id="doc_secure_789")

    assert result.is_grounded is True
    assert len(result.citations) == 1

    cit = result.citations[0]
    # Check that filesystem path in image_url was sanitized to the safe API endpoint
    assert cit.image_url == "/api/v1/documents/doc_secure_789/pages/5/image"
    assert "/workspaces/" not in (cit.image_url or "")

    # Check that sensitive keys and local paths in metadata were stripped
    assert "file_path" not in cit.metadata
    assert "storage_path" not in cit.metadata
    assert "api_key" not in cit.metadata
    assert "token" not in cit.metadata
    # Safe metadata preserved
    assert cit.metadata["dpi"] == 200
    assert cit.metadata["heading"] == "Security Analysis"


# End-to-End Generator Integration with Citation Validation
def test_generator_with_citation_validation_and_traceability(
    sample_mixed_evidence: list[EvidenceItem],
) -> None:
    mock_text = "DocuLens uses reranking [Evidence 1] and visual graphs [Evidence 2]."
    generator = AnswerGenerator(
        provider=MockLLMProvider(mock_response=mock_text, model="mock-validator-llm")
    )

    gen_result = generator.generate_answer(
        question="How does reranking and visual evidence work?",
        evidence=sample_mixed_evidence,
        document_id="doc_valid_123",
    )

    assert gen_result.is_grounded is True
    assert gen_result.document_id == "doc_valid_123"
    assert len(gen_result.citations) == 2

    # Verify complete provenance traceability
    cit1 = gen_result.citations[0]
    assert cit1.reference == "[Evidence 1]"
    assert cit1.chunk_id == "doc_valid_123_p1_c0"
    assert cit1.page_number == 1
    assert "reranked" in cit1.sources

    cit2 = gen_result.citations[1]
    assert cit2.reference == "[Evidence 2]"
    assert cit2.chunk_id is None
    assert cit2.page_number == 4
    assert cit2.image_url == "/api/v1/documents/doc_valid_123/pages/4/image"
    assert "visual" in cit2.sources
