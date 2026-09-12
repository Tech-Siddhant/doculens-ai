"""Unit tests for Phase 4.3 & 4.4 Hybrid Retrieval Fusion and Strategy Experiments."""

from typing import Any

import pytest

from app.schemas.retrieval import (
    FusedCandidate,
    HybridRetrievalResult,
    ModalityWeights,
    RetrievedChunk,
    RetrievedVisualPage,
)
from app.services.hybrid import (
    HybridRetriever,
    collect_and_merge_candidates,
    fuse_rrf,
    fuse_weighted,
    get_candidate_key,
)


@pytest.fixture
def sample_dense_results() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=1,
            score=0.90,
            raw_score=0.90,
            normalized_score=1.0,
            chunk_id="doc_a_p1_c0",
            document_id="doc_a",
            page_number=1,
            chunk_index=0,
            text="Machine learning models and neural networks.",
            metadata={"source": "dense"},
        ),
        RetrievedChunk(
            rank=2,
            score=0.70,
            raw_score=0.70,
            normalized_score=0.0,
            chunk_id="doc_a_p1_c1",
            document_id="doc_a",
            page_number=1,
            chunk_index=1,
            text="Deep learning architectures and transformer layers.",
            metadata={"source": "dense"},
        ),
    ]


@pytest.fixture
def sample_bm25_results() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=1,
            score=12.0,
            raw_score=12.0,
            normalized_score=1.0,
            chunk_id="doc_a_p1_c0",  # Same candidate as dense rank 1
            document_id="doc_a",
            page_number=1,
            chunk_index=0,
            text="Machine learning models and neural networks.",
            metadata={"source": "bm25"},
        ),
        RetrievedChunk(
            rank=2,
            score=6.0,
            raw_score=6.0,
            normalized_score=0.0,
            chunk_id="doc_a_p2_c0",  # BM25-only candidate
            document_id="doc_a",
            page_number=2,
            chunk_index=0,
            text="Convolutional filters in computer vision.",
            metadata={"source": "bm25"},
        ),
    ]


@pytest.fixture
def sample_visual_results() -> list[RetrievedVisualPage]:
    return [
        RetrievedVisualPage(
            rank=1,
            score=0.85,
            raw_score=0.85,
            normalized_score=1.0,
            document_id="doc_a",
            page_number=2,
            image_url="/api/v1/documents/doc_a/pages/2/image",
            metadata={"visual_aspect": "chart"},
        )
    ]


# ---------------------------------------------------------------------------
# 1. Hybrid Candidate Collection and Deduplication
# ---------------------------------------------------------------------------

def test_get_candidate_key_deterministic() -> None:
    chunk = RetrievedChunk(
        rank=1,
        score=0.8,
        chunk_id="doc1_p1_c0",
        document_id="doc1",
        page_number=1,
        chunk_index=0,
        text="Sample text",
    )
    assert get_candidate_key(chunk) == "chunk::doc1::doc1_p1_c0"

    page = RetrievedVisualPage(
        rank=1,
        score=0.9,
        document_id="doc1",
        page_number=3,
        image_url="/api/v1/documents/doc1/pages/3/image",
    )
    assert get_candidate_key(page) == "page::doc1::3"


def test_collect_and_merge_candidates_duplicate_merging(
    sample_dense_results: list[RetrievedChunk],
    sample_bm25_results: list[RetrievedChunk],
    sample_visual_results: list[RetrievedVisualPage],
) -> None:
    candidates = collect_and_merge_candidates(
        dense_results=sample_dense_results,
        bm25_results=sample_bm25_results,
        visual_results=sample_visual_results,
    )

    # 4 distinct candidate keys expected:
    # 1) doc_a_p1_c0 (both dense & bm25)
    # 2) doc_a_p1_c1 (dense only)
    # 3) doc_a_p2_c0 (bm25 only)
    # 4) page_2 (visual only)
    assert len(candidates) == 4

    merged = candidates["chunk::doc_a::doc_a_p1_c0"]
    assert merged["sources"] == ["dense", "bm25"]
    assert merged["raw_scores"] == {"dense": 0.90, "bm25": 12.0}
    assert merged["normalized_scores"] == {"dense": 1.0, "bm25": 1.0}
    assert merged["ranks"] == {"dense": 1, "bm25": 1}
    assert merged["text"] == "Machine learning models and neural networks."

    visual_cand = candidates["page::doc_a::2"]
    assert visual_cand["sources"] == ["visual"]
    assert visual_cand["raw_scores"] == {"visual": 0.85}
    assert visual_cand["image_url"] == "/api/v1/documents/doc_a/pages/2/image"


def test_collect_candidates_document_isolation() -> None:
    dense_a = [
        RetrievedChunk(
            rank=1,
            score=0.9,
            raw_score=0.9,
            normalized_score=1.0,
            chunk_id="doc_A_p1_c0",
            document_id="doc_A",
            page_number=1,
            chunk_index=0,
            text="Text A",
        )
    ]
    dense_b = [
        RetrievedChunk(
            rank=1,
            score=0.95,
            raw_score=0.95,
            normalized_score=1.0,
            chunk_id="doc_B_p1_c0",
            document_id="doc_B",
            page_number=1,
            chunk_index=0,
            text="Text B",
        )
    ]

    # Target only doc_A
    candidates_a = collect_and_merge_candidates(
        dense_results=dense_a + dense_b,
        bm25_results=[],
        visual_results=[],
        target_document_id="doc_A",
    )

    assert len(candidates_a) == 1
    assert "chunk::doc_A::doc_A_p1_c0" in candidates_a
    assert "chunk::doc_B::doc_B_p1_c0" not in candidates_a


def test_collect_candidates_empty_results() -> None:
    candidates = collect_and_merge_candidates([], [], [])
    assert len(candidates) == 0


def test_top_k_behavior_and_partial_modality_failure() -> None:
    # Simulating dense returns 2, bm25 returns 0, visual returns 1
    candidates = collect_and_merge_candidates(
        dense_results=[
            RetrievedChunk(rank=1, score=0.9, chunk_id="c1", document_id="doc1", page_number=1, chunk_index=0, text="A"),
            RetrievedChunk(rank=2, score=0.8, chunk_id="c2", document_id="doc1", page_number=1, chunk_index=1, text="B"),
            RetrievedChunk(rank=3, score=0.7, chunk_id="c3", document_id="doc1", page_number=2, chunk_index=0, text="C"),
        ],
        bm25_results=[],  # BM25 failed to match anything
        visual_results=[], # Visual disabled
    )
    # top_k=2 truncates the 3 candidates to 2
    fused = fuse_weighted(candidates, weights={"dense": 1.0, "bm25": 0.0, "visual": 0.0}, top_k=2)
    assert len(fused) == 2
    assert fused[0].rank == 1
    assert fused[1].rank == 2
    assert fused[0].chunk_id in ("c1", "c2", "c3")
    assert fused[1].chunk_id in ("c1", "c2", "c3")


# ---------------------------------------------------------------------------
# 2. Weighted Score Fusion
# ---------------------------------------------------------------------------

def test_weighted_fusion_score_calculation(
    sample_dense_results: list[RetrievedChunk],
    sample_bm25_results: list[RetrievedChunk],
    sample_visual_results: list[RetrievedVisualPage],
) -> None:
    candidates = collect_and_merge_candidates(
        dense_results=sample_dense_results,
        bm25_results=sample_bm25_results,
        visual_results=sample_visual_results,
    )

    # Weights: dense=0.5, bm25=0.3, visual=0.2 (sums to 1.0)
    weights = ModalityWeights(dense=0.5, bm25=0.3, visual=0.2)
    results = fuse_weighted(candidates, weights=weights, top_k=5)

    assert len(results) == 4
    # Rank 1 must be doc_a_p1_c0 (dense norm=1.0 * 0.5 + bm25 norm=1.0 * 0.3 = 0.8)
    top = results[0]
    assert top.rank == 1
    assert top.chunk_id == "doc_a_p1_c0"
    assert top.score == 0.8
    assert top.sources == ["dense", "bm25"]

    # Rank 2 is visual page 2 (visual norm=1.0 * 0.2 = 0.2)
    second = results[1]
    assert second.rank == 2
    assert second.page_number == 2
    assert second.image_url == "/api/v1/documents/doc_a/pages/2/image"
    assert second.score == 0.2


def test_weighted_fusion_custom_weight_normalization() -> None:
    candidates = {
        "c1": {
            "candidate_key": "c1",
            "document_id": "doc1",
            "page_number": 1,
            "chunk_id": "c1",
            "chunk_index": 0,
            "text": "T1",
            "image_url": None,
            "sources": ["dense"],
            "raw_scores": {"dense": 0.8},
            "normalized_scores": {"dense": 1.0},
            "ranks": {"dense": 1},
            "metadata": {},
        }
    }

    # Weights: dense=3.0, bm25=1.0, visual=0.0 -> total 4.0 -> normalized dense=0.75
    results = fuse_weighted(candidates, weights={"dense": 3.0, "bm25": 1.0, "visual": 0.0}, top_k=1)
    assert len(results) == 1
    assert results[0].score == 0.75


def test_weighted_fusion_invalid_weights() -> None:
    candidates = {}
    with pytest.raises(ValueError, match="Modality weights must be non-negative"):
        fuse_weighted(candidates, weights={"dense": -0.5, "bm25": 0.5, "visual": 0.5})

    with pytest.raises(ValueError, match="At least one modality weight must be strictly positive"):
        fuse_weighted(candidates, weights={"dense": 0.0, "bm25": 0.0, "visual": 0.0})


def test_weighted_fusion_score_threshold() -> None:
    candidates = collect_and_merge_candidates(
        dense_results=[
            RetrievedChunk(
                rank=1,
                score=0.9,
                raw_score=0.9,
                normalized_score=1.0,
                chunk_id="c1",
                document_id="doc1",
                page_number=1,
                chunk_index=0,
                text="T1",
            )
        ],
        bm25_results=[],
        visual_results=[],
    )
    # dense_w = 0.5 -> score = 0.5
    # threshold 0.6 filters it out
    res = fuse_weighted(candidates, score_threshold=0.6)
    assert len(res) == 0

    # threshold 0.4 keeps it
    res2 = fuse_weighted(candidates, score_threshold=0.4)
    assert len(res2) == 1


# ---------------------------------------------------------------------------
# 3. Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def test_rrf_score_calculation(
    sample_dense_results: list[RetrievedChunk],
    sample_bm25_results: list[RetrievedChunk],
    sample_visual_results: list[RetrievedVisualPage],
) -> None:
    candidates = collect_and_merge_candidates(
        dense_results=sample_dense_results,
        bm25_results=sample_bm25_results,
        visual_results=sample_visual_results,
    )

    # RRF with k=60
    # doc_a_p1_c0: rank 1 in dense (1/61) + rank 1 in bm25 (1/61) = 2/61 ≈ 0.032787
    # visual page 2: rank 1 in visual (1/61) = 1/61 ≈ 0.016393
    # doc_a_p1_c1: rank 2 in dense (1/62) ≈ 0.016129
    # doc_a_p2_c0: rank 2 in bm25 (1/62) ≈ 0.016129
    results = fuse_rrf(candidates, rrf_k=60, top_k=5)

    assert len(results) == 4
    top = results[0]
    assert top.rank == 1
    assert top.chunk_id == "doc_a_p1_c0"
    assert top.score == round(2.0 / 61.0, 6)
    assert top.sources == ["dense", "bm25"]

    second = results[1]
    assert second.rank == 2
    assert second.page_number == 2
    assert second.score == round(1.0 / 61.0, 6)


def test_rrf_configurable_k() -> None:
    candidates = {
        "c1": {
            "candidate_key": "c1",
            "document_id": "doc1",
            "page_number": 1,
            "chunk_id": "c1",
            "chunk_index": 0,
            "text": "T1",
            "image_url": None,
            "sources": ["dense"],
            "raw_scores": {"dense": 0.8},
            "normalized_scores": {"dense": 1.0},
            "ranks": {"dense": 1},
            "metadata": {},
        }
    }

    # With k=10: score = 1 / (10 + 1) = 1/11 ≈ 0.090909
    res = fuse_rrf(candidates, rrf_k=10)
    assert res[0].score == round(1.0 / 11.0, 6)

    with pytest.raises(ValueError, match="rrf_k must be >= 1"):
        fuse_rrf(candidates, rrf_k=0)


def test_rrf_deterministic_ordering_on_tie() -> None:
    # Two candidates with identical rank 1 in single modalities
    c_a = {
        "candidate_key": "chunk::doc1::c_a",
        "document_id": "doc1",
        "page_number": 1,
        "chunk_id": "c_a",
        "chunk_index": 0,
        "text": "A",
        "image_url": None,
        "sources": ["dense"],
        "raw_scores": {"dense": 0.8},
        "normalized_scores": {"dense": 1.0},
        "ranks": {"dense": 1},
        "metadata": {},
    }
    c_b = {
        "candidate_key": "chunk::doc1::c_b",
        "document_id": "doc1",
        "page_number": 1,
        "chunk_id": "c_b",
        "chunk_index": 1,
        "text": "B",
        "image_url": None,
        "sources": ["bm25"],
        "raw_scores": {"bm25": 10.0},
        "normalized_scores": {"bm25": 1.0},
        "ranks": {"bm25": 1},
        "metadata": {},
    }

    res = fuse_rrf({"chunk::doc1::c_b": c_b, "chunk::doc1::c_a": c_a}, rrf_k=60)
    # Both have score = 1/61 ≈ 0.016393
    assert res[0].score == res[1].score
    # Tie broken deterministically by candidate_key alphabetical ascending
    assert res[0].chunk_id == "c_a"
    assert res[1].chunk_id == "c_b"


# ---------------------------------------------------------------------------
# 4. Strategy Selection and Comparison
# ---------------------------------------------------------------------------

class MockDenseRetriever:
    def retrieve(self, query: str, top_k: int = 5, document_id: str | None = None) -> Any:
        class FakeRes:
            results = [
                RetrievedChunk(
                    rank=1,
                    score=0.9,
                    chunk_id="c_dense_1",
                    document_id=document_id or "doc1",
                    page_number=1,
                    chunk_index=0,
                    text="Dense matched content",
                )
            ]
        return FakeRes()


class MockBM25Retriever:
    def retrieve(self, query: str, top_k: int = 5, document_id: str | None = None) -> Any:
        class FakeRes:
            results = [
                RetrievedChunk(
                    rank=1,
                    score=15.0,
                    chunk_id="c_bm25_1",
                    document_id=document_id or "doc1",
                    page_number=2,
                    chunk_index=0,
                    text="BM25 matched keyword content",
                )
            ]
        return FakeRes()


class MockVisualEmbedder:
    def embed_visual_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class MockVisualStore:
    def search_visual(self, query_vector: list[float], top_k: int = 5, document_id: str | None = None) -> list[RetrievedVisualPage]:
        return [
            RetrievedVisualPage(
                rank=1,
                score=0.88,
                document_id=document_id or "doc1",
                page_number=3,
                image_url=f"/api/v1/documents/{document_id or 'doc1'}/pages/3/image",
            )
        ]


def test_hybrid_retriever_service_weighted_and_rrf() -> None:
    hybrid = HybridRetriever(
        dense_retriever_inst=MockDenseRetriever(),  # type: ignore[arg-type]
        bm25_retriever_inst=MockBM25Retriever(),    # type: ignore[arg-type]
        visual_embedder_inst=MockVisualEmbedder(),  # type: ignore[arg-type]
        visual_store_inst=MockVisualStore(),        # type: ignore[arg-type]
    )

    # 1. Weighted Retrieval
    res_w = hybrid.retrieve(query="neural networks", strategy="weighted", top_k=3)
    assert isinstance(res_w, HybridRetrievalResult)
    assert res_w.strategy == "weighted"
    assert res_w.total_results == 3
    assert res_w.weights is not None

    # 2. RRF Retrieval
    res_rrf = hybrid.retrieve(query="neural networks", strategy="rrf", rrf_k=60, top_k=3)
    assert isinstance(res_rrf, HybridRetrievalResult)
    assert res_rrf.strategy == "rrf"
    assert res_rrf.rrf_k == 60
    assert res_rrf.total_results == 3

    # 3. Invalid Strategy
    with pytest.raises(ValueError, match="Invalid fusion strategy"):
        hybrid.retrieve(query="test", strategy="unknown_strategy")


def test_compare_strategies() -> None:
    hybrid = HybridRetriever(
        dense_retriever_inst=MockDenseRetriever(),  # type: ignore[arg-type]
        bm25_retriever_inst=MockBM25Retriever(),    # type: ignore[arg-type]
        visual_embedder_inst=MockVisualEmbedder(),  # type: ignore[arg-type]
        visual_store_inst=MockVisualStore(),        # type: ignore[arg-type]
    )

    comparisons = hybrid.compare_strategies(query="machine learning", top_k=5)
    assert "dense_only" in comparisons
    assert "dense_bm25_weighted" in comparisons
    assert "dense_bm25_rrf" in comparisons
    assert "dense_bm25_visual_weighted" in comparisons
    assert "dense_bm25_visual_rrf" in comparisons

    assert comparisons["dense_only"].total_results == 1
    assert comparisons["dense_bm25_weighted"].total_results == 2
    assert comparisons["dense_bm25_visual_weighted"].total_results == 3
