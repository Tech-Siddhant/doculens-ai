"""Unit tests for CrossEncoderReranker service — Phase 5.1 baseline."""

from unittest.mock import MagicMock, patch
import pytest

from app.core.config import settings
from app.schemas.retrieval import (
    FusedCandidate,
    RerankedCandidate,
    RerankResult,
    RetrievedChunk,
    RetrievedVisualPage,
)
from app.services.reranker import CrossEncoderReranker, _candidate_text, reranker


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_encoder():
    """Mock TextCrossEncoder to isolate unit tests from downloading/running ONNX."""
    mock = MagicMock()
    # Default return value is set in tests or dynamically generated
    return mock


@pytest.fixture
def reranker_with_mock(mock_encoder):
    svc = CrossEncoderReranker(model_name="test-model", batch_size=32)
    svc._model = mock_encoder
    return svc


def make_fused_candidate(
    doc_id: str = "doc_1",
    chunk_id: str = "doc_1_0_0",
    rank: int = 1,
    score: float = 0.9,
    text: str = "Sample chunk text",
    sources: list[str] | None = None,
) -> FusedCandidate:
    return FusedCandidate(
        rank=rank,
        score=score,
        document_id=doc_id,
        page_number=1,
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        image_url=None,
        retrieval_type="fused",
        sources=sources or ["dense", "sparse"],
        raw_scores={"dense": 0.85, "sparse": 12.5},
        normalized_scores={"dense": 0.9, "sparse": 0.8},
        metadata={"section": "intro"},
    )


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_reranker_init_defaults():
    svc = CrossEncoderReranker()
    assert svc.model_name == settings.RERANKER_MODEL_NAME
    assert svc.batch_size == settings.RERANKER_BATCH_SIZE
    assert svc._model is None


def test_reranker_lazy_load_failure():
    svc = CrossEncoderReranker(model_name="nonexistent/invalid_model_12345")
    with patch("app.services.reranker.TextCrossEncoder", side_effect=Exception("Download failed")):
        with pytest.raises(RuntimeError, match="Failed to load reranker model"):
            _ = svc.model


def test_candidate_text_extraction():
    # Fused candidate with text
    fc = make_fused_candidate(text="Hello world")
    assert _candidate_text(fc) == "Hello world"

    # Visual page with metadata caption
    vp = RetrievedVisualPage(
        rank=1,
        score=0.8,
        document_id="doc_v",
        page_number=3,
        image_url="http://example.com/p3.png",
        metadata={"caption": "Architecture diagram"},
    )
    assert _candidate_text(vp) == "Architecture diagram"

    # Visual page without caption fallback
    vp_empty = RetrievedVisualPage(
        rank=1,
        score=0.8,
        document_id="doc_v",
        page_number=2,
        image_url="http://example.com/p2.png",
    )
    assert _candidate_text(vp_empty) == "Document doc_v Page 2"

    # Dict candidate
    d_cand = {"document_id": "doc_d", "page_number": 5, "text": ""}
    assert _candidate_text(d_cand) == "Document doc_d Page 5"


def test_score_pairs_validation(reranker_with_mock):
    # Empty or whitespace queries
    with pytest.raises(ValueError, match="non-empty string"):
        reranker_with_mock.score_pairs("", ["text 1"])
    with pytest.raises(ValueError, match="non-empty string"):
        reranker_with_mock.score_pairs("   ", ["text 1"])

    # Empty texts returns empty list
    assert reranker_with_mock.score_pairs("query", []) == []


def test_score_pairs_inference_error(reranker_with_mock, mock_encoder):
    mock_encoder.rerank.side_effect = Exception("ONNX runtime failure")
    with pytest.raises(RuntimeError, match="Reranker inference failed"):
        reranker_with_mock.score_pairs("query", ["text 1"])


def test_rerank_candidates_basic(reranker_with_mock, mock_encoder):
    mock_encoder.rerank.return_value = [1.2, 5.8, 3.4]

    c1 = make_fused_candidate(doc_id="doc_1", chunk_id="c1", rank=1, score=0.9, text="Text 1")
    c2 = make_fused_candidate(doc_id="doc_1", chunk_id="c2", rank=2, score=0.7, text="Text 2")
    c3 = make_fused_candidate(doc_id="doc_1", chunk_id="c3", rank=3, score=0.5, text="Text 3")

    results = reranker_with_mock.rerank_candidates("query", [c1, c2, c3], top_k=2)

    assert len(results) == 2
    # c2 had score 5.8 -> rank 1
    assert results[0].chunk_id == "c2"
    assert results[0].rank == 1
    assert results[0].score == 5.8
    assert results[0].initial_rank == 2
    assert results[0].initial_score == 0.7
    assert results[0].retrieval_type == "reranked"
    assert results[0].metadata == {"section": "intro"}

    # c3 had score 3.4 -> rank 2
    assert results[1].chunk_id == "c3"
    assert results[1].rank == 2
    assert results[1].score == 3.4
    assert results[1].initial_rank == 3


def test_rerank_candidates_document_isolation(reranker_with_mock, mock_encoder):
    mock_encoder.rerank.return_value = [4.0]

    c_target = make_fused_candidate(doc_id="doc_allowed", chunk_id="c_target", text="Allowed")
    c_other = make_fused_candidate(doc_id="doc_forbidden", chunk_id="c_other", text="Forbidden")

    results = reranker_with_mock.rerank_candidates(
        "query", [c_target, c_other], target_document_id="doc_allowed"
    )

    assert len(results) == 1
    assert results[0].document_id == "doc_allowed"
    assert results[0].chunk_id == "c_target"
    # Ensure forbidden candidate was never sent to model
    mock_encoder.rerank.assert_called_once_with("query", ["Allowed"], batch_size=32)


def test_rerank_candidates_score_threshold(reranker_with_mock, mock_encoder):
    mock_encoder.rerank.return_value = [10.0, -2.5, 3.0]

    c1 = make_fused_candidate(chunk_id="c1", rank=1, text="Text 1")
    c2 = make_fused_candidate(chunk_id="c2", rank=2, text="Text 2")
    c3 = make_fused_candidate(chunk_id="c3", rank=3, text="Text 3")

    results = reranker_with_mock.rerank_candidates(
        "query", [c1, c2, c3], score_threshold=0.0
    )

    # c2 (-2.5) should be filtered out
    assert len(results) == 2
    assert [r.chunk_id for r in results] == ["c1", "c3"]


def test_rerank_candidates_mixed_candidate_types(reranker_with_mock, mock_encoder):
    mock_encoder.rerank.return_value = [2.0, 5.0, 1.0]

    chunk = RetrievedChunk(
        rank=1,
        score=0.8,
        document_id="doc_1",
        page_number=1,
        chunk_id="chunk_1",
        chunk_index=0,
        text="Dense text",
        retrieval_type="dense",
    )
    visual = RetrievedVisualPage(
        rank=2,
        score=0.7,
        document_id="doc_1",
        page_number=2,
        image_url="http://example.com/page2.png",
        retrieval_type="visual",
        metadata={"caption": "Chart of revenue"},
    )
    dict_cand = {
        "rank": 3,
        "score": 0.6,
        "document_id": "doc_1",
        "page_number": 3,
        "chunk_id": "chunk_dict",
        "chunk_index": 2,
        "text": "Dict chunk",
        "sources": ["sparse"],
    }

    results = reranker_with_mock.rerank_candidates("query", [chunk, visual, dict_cand])

    assert len(results) == 3
    # Top score 5.0 is the visual page
    assert results[0].page_number == 2
    assert results[0].sources == ["visual"]
    assert results[0].image_url == "http://example.com/page2.png"

    # Second is chunk (2.0)
    assert results[1].chunk_id == "chunk_1"
    assert results[1].sources == ["dense"]

    # Third is dict_cand (1.0)
    assert results[2].chunk_id == "chunk_dict"
    assert results[2].sources == ["sparse"]


def test_rerank_structured_result(reranker_with_mock, mock_encoder):
    mock_encoder.rerank.return_value = [5.5]

    c = make_fused_candidate(doc_id="doc_abc", chunk_id="c_abc", text="Target text")
    res = reranker_with_mock.rerank("query string", [c], target_document_id="doc_abc", top_k=5)

    assert isinstance(res, RerankResult)
    assert res.query == "query string"
    assert res.document_id == "doc_abc"
    assert res.model_name == "test-model"
    assert res.top_k == 5
    assert res.total_results == 1
    assert len(res.results) == 1
    assert res.results[0].chunk_id == "c_abc"


def test_rerank_empty_inputs(reranker_with_mock):
    # Empty candidates list
    assert reranker_with_mock.rerank_candidates("query", []) == []
    res = reranker_with_mock.rerank("query", [])
    assert res.total_results == 0
    assert res.results == []


# ---------------------------------------------------------------------------
# Integration Test with Live FastEmbed Model
# ---------------------------------------------------------------------------

def test_live_reranker_relevance_ordering():
    """Live integration test verifying the actual ONNX cross-encoder ranks
    semantically relevant chunks ahead of irrelevant chunks.
    """
    svc = CrossEncoderReranker()
    query = "What is the capital of France?"

    relevant = make_fused_candidate(
        doc_id="doc_1",
        chunk_id="rel",
        rank=2,  # deliberately given worse initial rank
        score=0.4,
        text="Paris is the capital and most populous city of France.",
    )
    irrelevant = make_fused_candidate(
        doc_id="doc_1",
        chunk_id="irrel",
        rank=1,  # initially higher
        score=0.9,
        text="The quick brown fox jumps over the lazy dog in the forest.",
    )

    results = svc.rerank_candidates(query, [irrelevant, relevant])

    assert len(results) == 2
    # The relevant chunk about Paris must be promoted to rank 1
    assert results[0].chunk_id == "rel"
    assert results[0].rank == 1
    assert results[0].initial_rank == 2
    assert results[0].score > results[1].score

    # The irrelevant chunk is demoted to rank 2
    assert results[1].chunk_id == "irrel"
    assert results[1].rank == 2
    assert results[1].initial_rank == 1
