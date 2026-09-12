"""Unit tests for Phase 6.3 Retrieval Evaluation — metrics, evaluator, failure classification, and runner."""

from datetime import datetime, timezone
import pytest

from app.evaluation.metrics import (
    compute_hit_at_k,
    compute_mrr_at_k,
    compute_recall_at_k,
    compute_retrieval_metrics,
    extract_candidate_info,
    is_candidate_relevant,
)
from app.evaluation.runner import (
    RetrievalEvaluator,
    classify_retrieval_failures,
    retrieval_evaluator,
)
from app.schemas.evaluation import (
    EvaluationThresholds,
    ExpectedModality,
    ExperimentResult,
    FailureCategory,
    GoldDataset,
    GoldDocument,
    GoldQuery,
    PipelineConfig,
    QuestionCategory,
)
from app.schemas.retrieval import (
    EvidenceItem,
    FusedCandidate,
    RerankedCandidate,
    RetrievedChunk,
    RetrievedVisualPage,
)
from app.services.dataset import GoldBenchmark


# ──────────────────────────── Helper Fixtures & Objects ────────────────────────────

def _make_chunk(doc_id: str, page: int, chunk_idx: int, rank: int = 1, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        score=score,
        chunk_id=f"{doc_id}_p{page}_c{chunk_idx}",
        document_id=doc_id,
        page_number=page,
        chunk_index=chunk_idx,
        text=f"Sample text on page {page}",
    )


def _make_visual_page(doc_id: str, page: int, rank: int = 1, score: float = 0.85) -> RetrievedVisualPage:
    return RetrievedVisualPage(
        rank=rank,
        score=score,
        document_id=doc_id,
        page_number=page,
        image_url=f"/api/v1/documents/{doc_id}/pages/{page}/image",
    )


def _make_fused_candidate(doc_id: str, page: int, chunk_idx: int | None = None, rank: int = 1, score: float = 0.8) -> FusedCandidate:
    return FusedCandidate(
        candidate_key=f"item::{doc_id}::{page}",
        document_id=doc_id,
        page_number=page,
        chunk_id=f"{doc_id}_p{page}_c{chunk_idx}" if chunk_idx is not None else None,
        chunk_index=chunk_idx,
        text=f"Text on page {page}",
        rank=rank,
        score=score,
        sources=["dense", "visual"],
        raw_scores={"dense": 0.8, "visual": 0.7},
        normalized_scores={"dense": 0.8, "visual": 0.7},
    )


def _make_reranked_candidate(doc_id: str, page: int, rank: int = 1, score: float = 2.5) -> RerankedCandidate:
    return RerankedCandidate(
        rank=rank,
        score=score,
        initial_rank=rank,
        initial_score=0.8,
        document_id=doc_id,
        page_number=page,
        chunk_id=f"{doc_id}_p{page}_c0",
        chunk_index=0,
        text=f"Reranked text on page {page}",
    )


def _make_evidence_item(doc_id: str, page: int, rank: int = 1, score: float = 2.5) -> EvidenceItem:
    return EvidenceItem(
        rank=rank,
        score=score,
        initial_rank=rank,
        initial_score=0.8,
        document_id=doc_id,
        page_number=page,
        chunk_id=f"{doc_id}_p{page}_c0",
        chunk_index=0,
        text=f"Evidence text on page {page}",
    )


def _sample_query(**overrides) -> GoldQuery:
    base = {
        "query_id": "q-001",
        "document_id": "doc-1",
        "question": "What is the attention formula?",
        "category": QuestionCategory.FACTOID_TEXT,
        "expected_sources": [ExpectedModality.TEXT],
        "ground_truth_pages": [4],
        "ground_truth_chunks": ["doc-1_p4_c0"],
        "ground_truth_evidence_text": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V",
        "ground_truth_answer": "The attention formula is softmax(QK^T / sqrt(d_k))V.",
        "key_reference_facts": ["softmax scaling by sqrt(d_k)"],
        "is_answerable": True,
    }
    base.update(overrides)
    return GoldQuery(**base)


# ──────────────────────────── Candidate Extraction & Matching ────────────────────────────

def test_extract_candidate_info_across_types():
    # RetrievedChunk
    chunk = _make_chunk("doc-1", 3, 0)
    assert extract_candidate_info(chunk) == ("doc-1", 3, "doc-1_p3_c0")

    # RetrievedVisualPage
    vpage = _make_visual_page("doc-1", 5)
    assert extract_candidate_info(vpage) == ("doc-1", 5, None)

    # FusedCandidate
    fused = _make_fused_candidate("doc-2", 6, chunk_idx=1)
    assert extract_candidate_info(fused) == ("doc-2", 6, "doc-2_p6_c1")

    # RerankedCandidate
    reranked = _make_reranked_candidate("doc-1", 7)
    assert extract_candidate_info(reranked) == ("doc-1", 7, "doc-1_p7_c0")

    # EvidenceItem
    evidence = _make_evidence_item("doc-1", 8)
    assert extract_candidate_info(evidence) == ("doc-1", 8, "doc-1_p8_c0")

    # Dict representation
    d = {"document_id": "doc-1", "page_number": 4, "chunk_id": "c4"}
    assert extract_candidate_info(d) == ("doc-1", 4, "c4")

    # Raw integer page
    assert extract_candidate_info(2) == ("", 2, None)


def test_is_candidate_relevant():
    # Same doc, matching page
    assert is_candidate_relevant(_make_chunk("doc-1", 4, 0), ground_truth_pages=[4], target_document_id="doc-1")
    # Same doc, non-matching page
    assert not is_candidate_relevant(_make_chunk("doc-1", 5, 0), ground_truth_pages=[4], target_document_id="doc-1")
    # Document mismatch
    assert not is_candidate_relevant(_make_chunk("doc-2", 4, 0), ground_truth_pages=[4], target_document_id="doc-1")
    # Chunk ID match
    assert is_candidate_relevant(
        _make_chunk("doc-1", 5, 0),
        ground_truth_pages=[],
        target_document_id="doc-1",
        ground_truth_chunks=["doc-1_p5_c0"],
    )


# ──────────────────────────── Recall@K Tests ────────────────────────────

def test_recall_at_k_perfect_match():
    retrieved = [_make_chunk("doc-1", 1, 0), _make_chunk("doc-1", 2, 0), _make_chunk("doc-1", 3, 0)]
    score = compute_recall_at_k(retrieved, ground_truth_pages=[1, 2], k=3, target_document_id="doc-1")
    assert score == 1.0


def test_recall_at_k_no_match():
    retrieved = [_make_chunk("doc-1", 5, 0), _make_chunk("doc-1", 6, 0)]
    score = compute_recall_at_k(retrieved, ground_truth_pages=[1, 2], k=5, target_document_id="doc-1")
    assert score == 0.0


def test_recall_at_k_partial_match():
    # Ground truth requires pages 1, 2, 3 (3 pages). Top-3 has pages 1, 5, 6.
    retrieved = [_make_chunk("doc-1", 1, 0), _make_chunk("doc-1", 5, 0), _make_chunk("doc-1", 6, 0)]
    score = compute_recall_at_k(retrieved, ground_truth_pages=[1, 2, 3], k=3, target_document_id="doc-1")
    assert score == pytest.approx(1.0 / 3.0, rel=1e-5)


def test_recall_at_k_cutoff_boundary():
    # Ground truth is page 4. Top-3 has [1, 2, 3], page 4 is at rank 4.
    retrieved = [
        _make_chunk("doc-1", 1, 0, rank=1),
        _make_chunk("doc-1", 2, 0, rank=2),
        _make_chunk("doc-1", 3, 0, rank=3),
        _make_chunk("doc-1", 4, 0, rank=4),
    ]
    assert compute_recall_at_k(retrieved, [4], k=3, target_document_id="doc-1") == 0.0
    assert compute_recall_at_k(retrieved, [4], k=4, target_document_id="doc-1") == 1.0


# ──────────────────────────── MRR@K Tests ────────────────────────────

def test_mrr_at_k_first_relevant_result():
    retrieved = [_make_chunk("doc-1", 4, 0, rank=1), _make_chunk("doc-1", 2, 0, rank=2)]
    score = compute_mrr_at_k(retrieved, ground_truth_pages=[4], k=5, target_document_id="doc-1")
    assert score == 1.0


def test_mrr_at_k_later_relevant_result():
    # Relevant item at rank 2 -> MRR = 1/2 = 0.5
    retrieved = [_make_chunk("doc-1", 1, 0), _make_chunk("doc-1", 4, 0), _make_chunk("doc-1", 5, 0)]
    score = compute_mrr_at_k(retrieved, ground_truth_pages=[4], k=5, target_document_id="doc-1")
    assert score == 0.5

    # Relevant item at rank 4 -> MRR = 1/4 = 0.25
    retrieved_4 = [
        _make_chunk("doc-1", 1, 0),
        _make_chunk("doc-1", 2, 0),
        _make_chunk("doc-1", 3, 0),
        _make_chunk("doc-1", 4, 0),
        _make_chunk("doc-1", 5, 0),
    ]
    assert compute_mrr_at_k(retrieved_4, ground_truth_pages=[4], k=5, target_document_id="doc-1") == 0.25


def test_mrr_at_k_no_relevant_result():
    retrieved = [_make_chunk("doc-1", 1, 0), _make_chunk("doc-1", 2, 0)]
    score = compute_mrr_at_k(retrieved, ground_truth_pages=[4], k=5, target_document_id="doc-1")
    assert score == 0.0


def test_compute_hit_at_k():
    retrieved = [_make_chunk("doc-1", 1, 0), _make_chunk("doc-1", 4, 0)]
    assert compute_hit_at_k(retrieved, ground_truth_pages=[4], k=2, target_document_id="doc-1") is True
    assert compute_hit_at_k(retrieved, ground_truth_pages=[4], k=1, target_document_id="doc-1") is False


# ──────────────────────────── Edge Cases: Duplicates, Isolation, Empty GT ────────────────────────────

def test_duplicate_retrieved_evidence():
    # Multiple chunks from the same page 2 in top candidates
    retrieved = [
        _make_chunk("doc-1", 2, 0, rank=1),
        _make_chunk("doc-1", 2, 1, rank=2),
        _make_chunk("doc-1", 3, 0, rank=3),
    ]
    # Ground truth is [2, 3]
    # Page 2 appears twice but should count as 1 distinct retrieved page
    recall = compute_recall_at_k(retrieved, ground_truth_pages=[2, 3], k=3, target_document_id="doc-1")
    assert recall == 1.0  # Both page 2 and 3 are in top 3
    # MRR should be 1.0 (first relevant at rank 1)
    mrr = compute_mrr_at_k(retrieved, ground_truth_pages=[2, 3], k=3, target_document_id="doc-1")
    assert mrr == 1.0


def test_document_isolation_mismatch():
    # Retrieved chunk is from doc-2, query targets doc-1
    retrieved = [_make_chunk("doc-2", 4, 0, rank=1), _make_chunk("doc-1", 4, 0, rank=2)]
    # Rank 1 must be ignored because of document isolation
    mrr = compute_mrr_at_k(retrieved, ground_truth_pages=[4], k=5, target_document_id="doc-1")
    assert mrr == 0.5  # doc-1 chunk is at rank 2


def test_empty_or_invalid_ground_truth():
    retrieved = [_make_chunk("doc-1", 1, 0)]
    # Empty ground truth pages
    assert compute_recall_at_k(retrieved, ground_truth_pages=[], k=5) == 0.0
    assert compute_mrr_at_k(retrieved, ground_truth_pages=[], k=5) == 0.0
    assert compute_hit_at_k(retrieved, ground_truth_pages=[], k=5) is False

    # Empty retrieved
    assert compute_recall_at_k([], ground_truth_pages=[1], k=5) == 0.0
    assert compute_mrr_at_k([], ground_truth_pages=[1], k=5) == 0.0

    # k <= 0
    assert compute_recall_at_k(retrieved, ground_truth_pages=[1], k=0) == 0.0
    assert compute_mrr_at_k(retrieved, ground_truth_pages=[1], k=-1) == 0.0


# ──────────────────────────── Multi-K RetrievalEvalMetrics ────────────────────────────

def test_compute_retrieval_metrics_multi_k():
    retrieved = [
        _make_chunk("doc-1", 10, 0, rank=1),
        _make_chunk("doc-1", 4, 0, rank=2),   # GT page 4 hit at rank 2
        _make_chunk("doc-1", 6, 0, rank=3),   # GT page 6 hit at rank 3
        _make_chunk("doc-1", 11, 0, rank=4),
        _make_chunk("doc-1", 12, 0, rank=5),
    ]
    gt_pages = [4, 6]
    metrics = compute_retrieval_metrics(
        retrieved=retrieved,
        ground_truth_pages=gt_pages,
        k_values=[1, 2, 3, 5],
        default_k=5,
        target_document_id="doc-1",
    )

    # default_k = 5
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr_at_k == 0.5
    assert metrics.hit_at_k is True
    assert metrics.retrieved_pages == [10, 4, 6, 11, 12]

    # k=1: recall=0.0, mrr=0.0
    assert metrics.k_metrics[1]["recall"] == 0.0
    assert metrics.k_metrics[1]["mrr"] == 0.0

    # k=2: recall=0.5 (page 4 found), mrr=0.5
    assert metrics.k_metrics[2]["recall"] == 0.5
    assert metrics.k_metrics[2]["mrr"] == 0.5

    # k=3: recall=1.0 (pages 4 and 6 found), mrr=0.5
    assert metrics.k_metrics[3]["recall"] == 1.0
    assert metrics.k_metrics[3]["mrr"] == 0.5


# ──────────────────────────── Failure Classification ────────────────────────────

def test_classify_retrieval_failures():
    q_text = _sample_query(expected_sources=[ExpectedModality.TEXT], ground_truth_pages=[4])
    q_vis = _sample_query(
        category=QuestionCategory.FIGURE_CHART_ANALYSIS,
        expected_sources=[ExpectedModality.VISUAL, ExpectedModality.FIGURE],
        ground_truth_pages=[5],
    )
    q_unans = _sample_query(
        category=QuestionCategory.NEGATIVE_UNANSWERABLE,
        is_answerable=False,
        ground_truth_pages=[],
    )

    # 1. Perfect retrieval -> No failures
    perfect_metrics = compute_retrieval_metrics([_make_chunk("doc-1", 4, 0)], [4], default_k=5)
    assert classify_retrieval_failures(q_text, perfect_metrics) == []

    # 2. Text retrieval miss
    miss_metrics = compute_retrieval_metrics([_make_chunk("doc-1", 1, 0)], [4], default_k=5)
    assert classify_retrieval_failures(q_text, miss_metrics) == [FailureCategory.RETRIEVAL_MISS]

    # 3. Visual query miss -> RETRIEVAL_MISS + VISUAL_MISS
    vis_miss_metrics = compute_retrieval_metrics([_make_chunk("doc-1", 1, 0)], [5], default_k=5)
    failures = classify_retrieval_failures(q_vis, vis_miss_metrics)
    assert FailureCategory.RETRIEVAL_MISS in failures
    assert FailureCategory.VISUAL_MISS in failures

    # 4. Low rank retrieval (first hit at rank 3, MRR = 0.333 < 0.5)
    low_rank_candidates = [_make_chunk("doc-1", 1, 0), _make_chunk("doc-1", 2, 0), _make_chunk("doc-1", 4, 0)]
    low_rank_metrics = compute_retrieval_metrics(low_rank_candidates, [4], default_k=5)
    assert classify_retrieval_failures(q_text, low_rank_metrics) == [FailureCategory.RETRIEVAL_LOW_RANK]

    # 5. Unanswerable query -> No retrieval failures
    unans_metrics = compute_retrieval_metrics([_make_chunk("doc-1", 1, 0)], [], default_k=5)
    assert classify_retrieval_failures(q_unans, unans_metrics) == []


# ──────────────────────────── Evaluator & Runner Tests ────────────────────────────

def test_evaluator_single_query():
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10], default_k=5)
    query = _sample_query(ground_truth_pages=[4])
    candidates = [_make_chunk("doc-1", 4, 0, rank=1)]

    result = evaluator.evaluate_query(query, candidates)
    assert result.query_id == "q-001"
    assert result.retrieval_metrics.recall_at_k == 1.0
    assert result.retrieval_metrics.mrr_at_k == 1.0
    assert result.passed_thresholds is True
    assert len(result.failure_categories) == 0


def test_evaluator_queries_aggregation():
    evaluator = RetrievalEvaluator(
        k_values=[1, 3, 5],
        default_k=5,
        thresholds=EvaluationThresholds(min_recall_at_k=0.70, min_mrr_at_k=0.60),
    )

    q1 = _sample_query(query_id="q-001", category=QuestionCategory.FACTOID_TEXT, ground_truth_pages=[1])
    q2 = _sample_query(query_id="q-002", category=QuestionCategory.TABLE_LOOKUP, ground_truth_pages=[6])
    q3 = _sample_query(
        query_id="q-003",
        category=QuestionCategory.NEGATIVE_UNANSWERABLE,
        is_answerable=False,
        ground_truth_pages=[],
    )

    # Precomputed retrieval candidates
    precomputed = {
        "q-001": [_make_chunk("doc-1", 1, 0)],                      # Recall=1.0, MRR=1.0
        "q-002": [_make_chunk("doc-1", 2, 0), _make_chunk("doc-1", 6, 0)], # Recall=1.0, MRR=0.5
        "q-003": [_make_chunk("doc-1", 9, 0)],                      # Unanswerable
    }

    benchmark = GoldBenchmark(queries=[q1, q2, q3])
    exp_result = evaluator.evaluate_queries(
        dataset=benchmark,
        precomputed_results=precomputed,
        dataset_id="test-dataset-v1",
    )

    assert exp_result.summary.total_queries == 3
    # Means calculated across the 2 answerable queries: (1.0 + 1.0) / 2 = 1.0
    assert exp_result.summary.mean_recall_at_k == 1.0
    # Mean MRR across 2 answerable queries: (1.0 + 0.5) / 2 = 0.75
    assert exp_result.summary.mean_mrr_at_k == 0.75
    assert exp_result.summary.thresholds_passed is True

    # Multi-K check
    assert 1 in exp_result.summary.k_metrics
    assert 3 in exp_result.summary.k_metrics
    assert 5 in exp_result.summary.k_metrics

    # Category breakdown check
    assert "factoid_text" in exp_result.summary.category_metrics
    assert "table_lookup" in exp_result.summary.category_metrics
    assert exp_result.summary.category_metrics["factoid_text"]["mean_recall_at_k"] == 1.0


def test_evaluator_thresholds_failure():
    # Strict thresholds that fail
    strict_evaluator = RetrievalEvaluator(
        default_k=5,
        thresholds=EvaluationThresholds(min_recall_at_k=0.99, min_mrr_at_k=0.99),
    )
    q1 = _sample_query(ground_truth_pages=[1])
    # Hit at rank 2 -> MRR = 0.5 < 0.99
    precomputed = {"q-001": [_make_chunk("doc-1", 9, 0), _make_chunk("doc-1", 1, 0)]}

    exp_result = strict_evaluator.evaluate_queries(
        dataset=[q1],
        precomputed_results=precomputed,
    )
    assert exp_result.summary.thresholds_passed is False
    assert exp_result.query_results[0].passed_thresholds is False


def test_compare_retrieval_configurations():
    evaluator = RetrievalEvaluator(default_k=5)
    q1 = _sample_query(query_id="q-001", ground_truth_pages=[2])

    configs = {
        "dense_baseline": lambda q: [_make_chunk("doc-1", 2, 0, rank=1)],  # Perfect
        "bm25_baseline": lambda q: [_make_chunk("doc-1", 9, 0, rank=1)],   # Miss
    }

    results = evaluator.compare_retrieval_configurations(dataset=[q1], configurations=configs)

    assert "dense_baseline" in results
    assert "bm25_baseline" in results
    assert results["dense_baseline"].summary.mean_recall_at_k == 1.0
    assert results["bm25_baseline"].summary.mean_recall_at_k == 0.0
    assert results["bm25_baseline"].summary.failure_counts.get("retrieval_miss") == 1




def test_evaluator_with_real_seed_dataset():
    from pathlib import Path
    from app.services.dataset import load_gold_dataset

    seed_path = Path("data/gold_dataset.jsonl")
    if not seed_path.exists():
        pytest.skip("data/gold_dataset.jsonl not found")

    benchmark = load_gold_dataset(seed_path)
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10], default_k=5)

    # Mock retrieval function matching ground truth pages for answerable queries
    def mock_retriever(q: GoldQuery) -> list[RetrievedChunk]:
        if not q.ground_truth_pages:
            return [_make_chunk(q.document_id, 99, 0)]
        return [_make_chunk(q.document_id, p, 0, rank=idx + 1) for idx, p in enumerate(q.ground_truth_pages)]

    result = evaluator.evaluate_queries(
        dataset=benchmark,
        retrieval_fn=mock_retriever,
        dataset_id="gold-benchmark-v1",
        pipeline_config=PipelineConfig(name="mock_baseline"),
    )

    assert result.summary.total_queries == benchmark.total
    assert result.summary.mean_recall_at_k == 1.0
    assert result.summary.mean_mrr_at_k == 1.0
    assert result.summary.thresholds_passed is True
    assert len(result.query_results) == benchmark.total

def test_deterministic_serialization():
    evaluator = RetrievalEvaluator(default_k=5)
    q1 = _sample_query(query_id="q-001", ground_truth_pages=[1])
    exp_res = evaluator.evaluate_queries(dataset=[q1], precomputed_results={"q-001": [_make_chunk("doc-1", 1, 0)]})

    # Serialize to JSON and parse back
    raw_json = exp_res.model_dump_json()
    reloaded = ExperimentResult.model_validate_json(raw_json)

    assert reloaded.experiment_id == exp_res.experiment_id
    assert reloaded.summary.mean_recall_at_k == exp_res.summary.mean_recall_at_k
    assert reloaded.query_results[0].query_id == "q-001"
