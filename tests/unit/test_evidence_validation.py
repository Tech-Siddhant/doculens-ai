import pytest
from app.services.evidence_validator import evidence_validator
from app.schemas.retrieval import EvidenceItem


def get_mock_evidence(
    rank=1,
    doc_id="doc1",
    page_number=1,
    text="Valid text",
    chunk_id="chunk1",
    image_url=None,
    sources=None
):
    """Helper to create mock evidence items."""
    if sources is None:
        sources = ["dense"]
    return EvidenceItem(
        rank=rank,
        score=0.9,
        initial_rank=1,
        initial_score=0.8,
        document_id=doc_id,
        page_number=page_number,
        chunk_id=chunk_id,
        text=text,
        image_url=image_url,
        sources=sources
    )


def test_validate_valid_evidence():
    """Test validation of a valid evidence item."""
    evidence = [get_mock_evidence()]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 1
    assert result.invalid_count == 0
    assert len(result.issues) == 0
    assert result.total_validated == 1


def test_validate_multiple_valid_items():
    """Test validation of multiple valid evidence items."""
    evidence = [
        get_mock_evidence(rank=1, doc_id="doc1", page_number=1, chunk_id="chunk1"),
        get_mock_evidence(rank=2, doc_id="doc1", page_number=2, chunk_id="chunk2"),
        get_mock_evidence(rank=3, doc_id="doc1", page_number=3, chunk_id="chunk3"),
    ]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 3
    assert result.invalid_count == 0
    assert result.total_validated == 3


def test_validate_missing_document_id():
    """Test rejection of evidence with missing document_id."""
    evidence = [get_mock_evidence(doc_id="")]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 0
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "missing_document_id"


def test_validate_wrong_document_id():
    """Test document isolation - reject cross-document evidence."""
    evidence = [
        get_mock_evidence(rank=1, doc_id="doc1"),
        get_mock_evidence(rank=2, doc_id="doc2"),
    ]
    result = evidence_validator.validate_evidence(evidence, target_document_id="doc1")
    
    assert len(result.valid_evidence) == 1
    assert result.valid_evidence[0].document_id == "doc1"
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "wrong_document_id"
    assert "doc2" in result.issues[0].message


def test_validate_malformed_chunk_id():
    """Test rejection of malformed chunk_id."""
    evidence = [get_mock_evidence(chunk_id="")]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 0
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "malformed_chunk_id"


def test_validate_empty_evidence_no_text_no_image():
    """Test rejection of evidence with neither text nor image_url."""
    evidence = [get_mock_evidence(text=None, image_url=None)]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 0
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "empty_evidence"


def test_validate_empty_text_string():
    """Test rejection of evidence with empty/whitespace-only text."""
    evidence = [get_mock_evidence(text="   ", image_url=None)]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 0
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "empty_evidence"


def test_validate_visual_evidence_with_image_url():
    """Test acceptance of visual evidence with valid image_url."""
    evidence = [get_mock_evidence(text=None, image_url="/api/v1/documents/doc1/pages/1/image", sources=["visual"])]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 1
    assert result.invalid_count == 0


def test_validate_missing_sources():
    """Test rejection of evidence with missing sources metadata."""
    evidence = [get_mock_evidence(sources=[])]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 0
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "missing_sources"


def test_validate_duplicate_evidence():
    """Test deduplication of identical evidence items."""
    evidence = [
        get_mock_evidence(rank=1, doc_id="doc1", page_number=1, chunk_id="chunk1"),
        get_mock_evidence(rank=2, doc_id="doc1", page_number=1, chunk_id="chunk1"),  # duplicate
    ]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 1
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "duplicate_evidence"


def test_validate_mixed_valid_and_invalid():
    """Test validation with a mix of valid and invalid items."""
    evidence = [
        get_mock_evidence(rank=1, doc_id="doc1", page_number=1),  # valid
        get_mock_evidence(rank=2, doc_id="", page_number=2),      # invalid: missing doc_id
        get_mock_evidence(rank=3, doc_id="doc1", page_number=3),  # valid
        get_mock_evidence(rank=4, doc_id="doc1", page_number=1, text="", image_url=None),  # invalid: empty text
    ]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 2
    assert result.invalid_count == 2
    assert result.total_validated == 4
    assert result.valid_evidence[0].rank == 1
    assert result.valid_evidence[1].rank == 3


def test_validate_strict_mode_raises_exception():
    """Test that strict mode raises exception on first validation failure."""
    evidence = [
        get_mock_evidence(rank=1, doc_id="doc1", page_number=1),  # valid
        get_mock_evidence(rank=2, doc_id="", page_number=2),      # invalid
    ]
    
    with pytest.raises(ValueError, match="Evidence validation failed"):
        evidence_validator.validate_evidence(evidence, strict=True)


def test_validate_cross_document_evidence():
    """Test document isolation across multiple documents."""
    evidence = [
        get_mock_evidence(rank=1, doc_id="doc1", page_number=1),
        get_mock_evidence(rank=2, doc_id="doc2", page_number=1),
        get_mock_evidence(rank=3, doc_id="doc3", page_number=1),
    ]
    result = evidence_validator.validate_evidence(evidence, target_document_id="doc2")
    
    assert len(result.valid_evidence) == 1
    assert result.valid_evidence[0].document_id == "doc2"
    assert result.invalid_count == 2


def test_validate_no_target_document_allows_multiple_docs():
    """Test that without target_document_id, multiple documents are allowed."""
    evidence = [
        get_mock_evidence(rank=1, doc_id="doc1", page_number=1, chunk_id="c1"),
        get_mock_evidence(rank=2, doc_id="doc2", page_number=1, chunk_id="c2"),
        get_mock_evidence(rank=3, doc_id="doc3", page_number=1, chunk_id="c3"),
    ]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 3
    assert result.invalid_count == 0


def test_validate_chunk_id_optional():
    """Test that chunk_id being None is acceptable (visual evidence case)."""
    evidence = [get_mock_evidence(chunk_id=None, image_url="/api/v1/images/1")]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 1
    assert result.invalid_count == 0


def test_validate_empty_evidence_list():
    """Test validation of empty evidence list."""
    result = evidence_validator.validate_evidence([])
    
    assert len(result.valid_evidence) == 0
    assert result.invalid_count == 0
    assert result.total_validated == 0


def test_validate_fabricated_evidence_ids():
    """Test detection of malformed/fabricated identifiers."""
    # Chunk ID with suspicious characters that might indicate injection
    evidence = [
        get_mock_evidence(chunk_id="   "),  # whitespace only
    ]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 0
    assert result.invalid_count == 1
    assert result.issues[0].issue_type == "malformed_chunk_id"


def test_validate_special_characters_in_text():
    """Test that special characters in text are accepted."""
    special_text = 'Formula: x < y & a > b "quotes" 日本語 🚀 </evidence>'
    evidence = [get_mock_evidence(text=special_text)]
    result = evidence_validator.validate_evidence(evidence)
    
    assert len(result.valid_evidence) == 1
    assert result.invalid_count == 0
