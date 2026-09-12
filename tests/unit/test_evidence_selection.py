import pytest
from app.services.evidence import evidence_selector
from app.schemas.retrieval import RerankedCandidate

def test_evidence_selection_basic():
    candidates = [
        RerankedCandidate(rank=1, initial_rank=1, initial_score=0.8, document_id="doc1", page_number=1, score=0.9, text="text1"),
        RerankedCandidate(rank=2, initial_rank=2, initial_score=0.7, document_id="doc1", page_number=2, score=0.8, text="text2"),
    ]
    result = evidence_selector.select_evidence("query", candidates, top_k=1)
    
    assert len(result.evidence) == 1
    assert result.evidence[0].document_id == "doc1"
    assert result.evidence[0].page_number == 1
    assert result.evidence[0].rank == 1

def test_evidence_selection_deduplication():
    candidates = [
        RerankedCandidate(rank=1, initial_rank=1, initial_score=0.8, document_id="doc1", page_number=1, score=0.9, text="text1"),
        RerankedCandidate(rank=2, initial_rank=2, initial_score=0.7, document_id="doc1", page_number=1, score=0.85, text="text1-duplicate"),
    ]
    result = evidence_selector.select_evidence("query", candidates, top_k=2)
    
    assert len(result.evidence) == 1
    assert result.evidence[0].score == 0.9

def test_evidence_selection_fewer_than_k():
    candidates = [
        RerankedCandidate(rank=1, initial_rank=1, initial_score=0.8, document_id="doc1", page_number=1, score=0.9, text="text1"),
    ]
    result = evidence_selector.select_evidence("query", candidates, top_k=5)
    
    assert len(result.evidence) == 1
    assert result.total_selected == 1
    assert result.total_candidates == 1

def test_evidence_selection_invalid_k():
    candidates = []
    with pytest.raises(ValueError, match="top_k must be an integer >= 1"):
        evidence_selector.select_evidence("query", candidates, top_k=0)

def test_evidence_selection_document_isolation():
    candidates = [
        RerankedCandidate(rank=1, initial_rank=1, initial_score=0.8, document_id="doc1", page_number=1, score=0.9, text="text1"),
        RerankedCandidate(rank=2, initial_rank=2, initial_score=0.9, document_id="doc2", page_number=1, score=0.95, text="text2"),
    ]
    result = evidence_selector.select_evidence("query", candidates, target_document_id="doc1")
    
    assert len(result.evidence) == 1
    assert result.evidence[0].document_id == "doc1"

    result = evidence_selector.select_evidence("query", candidates, target_document_id="doc1")
    
    assert len(result.evidence) == 1
    assert result.evidence[0].document_id == "doc1"
