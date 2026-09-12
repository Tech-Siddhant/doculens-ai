"""Unit tests for score normalization utilities."""

import pytest

from app.schemas.retrieval import RetrievedChunk, RetrievedVisualPage
from app.services.normalizer import min_max_scale_scores, normalize_retrieval_results


def test_min_max_scale_scores_standard() -> None:
    scores = [10.0, 20.0, 30.0]
    scaled = min_max_scale_scores(scores)
    assert scaled == [0.0, 0.5, 1.0]


def test_min_max_scale_scores_identical_positive() -> None:
    scores = [5.0, 5.0, 5.0]
    scaled = min_max_scale_scores(scores)
    assert scaled == [1.0, 1.0, 1.0]


def test_min_max_scale_scores_identical_zero_or_negative() -> None:
    assert min_max_scale_scores([0.0, 0.0]) == [0.0, 0.0]
    assert min_max_scale_scores([-2.5, -2.5]) == [0.0, 0.0]


def test_min_max_scale_scores_single_element() -> None:
    assert min_max_scale_scores([12.5]) == [1.0]
    assert min_max_scale_scores([0.0]) == [0.0]
    assert min_max_scale_scores([-5.0]) == [0.0]


def test_min_max_scale_scores_empty() -> None:
    assert min_max_scale_scores([]) == []


def test_min_max_scale_scores_negative_to_positive() -> None:
    scores = [-10.0, 0.0, 10.0]
    scaled = min_max_scale_scores(scores)
    assert scaled == [0.0, 0.5, 1.0]


def test_min_max_scale_extreme_values() -> None:
    scores = [1e-9, 2e-9, 3e-9]
    scaled = min_max_scale_scores(scores)
    assert scaled == [0.0, 0.5, 1.0]

    large_scores = [1_000_000.0, 2_000_000.0, 3_000_000.0]
    large_scaled = min_max_scale_scores(large_scores)
    assert large_scaled == [0.0, 0.5, 1.0]


def test_normalize_retrieved_chunks_immutability_and_preservation() -> None:
    chunks = [
        RetrievedChunk(
            rank=1,
            score=15.0,
            chunk_id="doc1_p1_c0",
            document_id="doc1",
            page_number=1,
            chunk_index=0,
            text="First highest scoring chunk.",
            metadata={"char_count": 28},
            retrieval_type="bm25",
        ),
        RetrievedChunk(
            rank=2,
            score=5.0,
            chunk_id="doc1_p2_c0",
            document_id="doc1",
            page_number=2,
            chunk_index=0,
            text="Second chunk.",
            metadata={"char_count": 13},
            retrieval_type="bm25",
        ),
    ]

    normalized = normalize_retrieval_results(chunks)

    # Verify original items were NOT mutated
    assert chunks[0].normalized_score is None
    assert chunks[1].normalized_score is None

    # Verify normalized results
    assert len(normalized) == 2
    assert normalized[0].raw_score == 15.0
    assert normalized[0].normalized_score == 1.0
    assert normalized[1].raw_score == 5.0
    assert normalized[1].normalized_score == 0.0
    assert normalized[0].chunk_id == "doc1_p1_c0"
    assert normalized[0].rank == 1
    assert normalized[0].metadata == {"char_count": 28}


def test_normalize_retrieved_visual_pages() -> None:
    pages = [
        RetrievedVisualPage(
            rank=1,
            score=0.85,
            document_id="doc_vis",
            page_number=1,
            image_url="/documents/doc_vis/pages/1/image",
            metadata={"width": 612},
            retrieval_type="visual",
        ),
        RetrievedVisualPage(
            rank=2,
            score=0.65,
            document_id="doc_vis",
            page_number=2,
            image_url="/documents/doc_vis/pages/2/image",
            metadata={"width": 612},
            retrieval_type="visual",
        ),
    ]

    normalized = normalize_retrieval_results(pages)

    assert len(normalized) == 2
    assert normalized[0].raw_score == 0.85
    assert normalized[0].normalized_score == 1.0
    assert normalized[1].raw_score == 0.65
    assert normalized[1].normalized_score == 0.0
    assert normalized[0].image_url == "/documents/doc_vis/pages/1/image"


def test_normalize_dicts() -> None:
    data = [
        {"score": 100.0, "id": "a"},
        {"score": 50.0, "id": "b"},
        {"score": 0.0, "id": "c"},
    ]
    normalized = normalize_retrieval_results(data)

    assert data[0].get("normalized_score") is None
    assert normalized[0]["normalized_score"] == 1.0
    assert normalized[1]["normalized_score"] == 0.5
    assert normalized[2]["normalized_score"] == 0.0
    assert normalized[0]["raw_score"] == 100.0


def test_normalize_empty_list() -> None:
    assert normalize_retrieval_results([]) == []
