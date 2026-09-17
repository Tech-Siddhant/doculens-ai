"""Unit tests for Generation and RAG Evaluation — Phase 6.4.

Tests deterministic and judge-based evaluation of:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- Citation Precision & Recall
- Abstention Correctness
- Failure Taxonomy Classification
- Seed Benchmark Dataset Evaluation
"""

from pathlib import Path
import pytest

from app.evaluation import (
    DeterministicGenerationJudge,
    GenerationEvaluator,
    LLMJudgeEvaluator,
    MockLLMGenerationJudge,
    RAGEvaluator,
    RetrievalEvaluator,
    classify_generation_failures,
    classify_retrieval_failures,
    compute_abstention_correctness,
    compute_answer_relevancy,
    compute_citation_precision,
    compute_citation_recall,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_generation_metrics,
)
from app.schemas.citation import Citation
from app.schemas.evaluation import (
    EvaluationThresholds,
    ExpectedModality,
    FailureCategory,
    GoldQuery,
    PipelineConfig,
    QuestionCategory,
)
from app.schemas.generation import GenerationResult
from app.schemas.retrieval import EvidenceItem, RetrievedChunk
from app.services.citation_validator import INSUFFICIENT_EVIDENCE_ANSWER
from app.services.dataset import load_gold_dataset
from app.services.llm_provider import MockLLMProvider


@pytest.fixture
def sample_query() -> GoldQuery:
    return GoldQuery(
        query_id="q-001",
        document_id="doc-test-1",
        question="What is the computational complexity per layer of Self-Attention?",
        category=QuestionCategory.TABLE_LOOKUP,
        expected_sources=[ExpectedModality.TEXT, ExpectedModality.TABLE],
        ground_truth_pages=[6],
        ground_truth_chunks=["doc-test-1_p6_c2"],
        ground_truth_evidence_text="Table 1: Self-Attention complexity per layer is O(n^2 * d) with O(1) sequential operations.",
        ground_truth_answer="Self-Attention has a per-layer complexity of O(n^2 * d) with O(1) sequential operations.",
        key_reference_facts=[
            "Self-Attention complexity per layer is O(n^2 * d)",
            "O(1) sequential operations",
        ],
        is_answerable=True,
    )


@pytest.fixture
def unanswerable_query() -> GoldQuery:
    return GoldQuery(
        query_id="q-unans",
        document_id="doc-test-1",
        question="What was the author's favorite coffee brand mentioned in the paper?",
        category=QuestionCategory.NEGATIVE_UNANSWERABLE,
        expected_sources=[ExpectedModality.TEXT],
        ground_truth_pages=[],
        ground_truth_chunks=[],
        ground_truth_evidence_text="",
        ground_truth_answer="The provided document does not contain information regarding coffee brands.",
        key_reference_facts=[],
        is_answerable=False,
    )


# 1. Fully grounded answer test
def test_fully_grounded_answer(sample_query: GoldQuery) -> None:
    evidence = [
        RetrievedChunk(
            rank=1,
            score=0.95,
            chunk_id="doc-test-1_p6_c2",
            document_id="doc-test-1",
            page_number=6,
            chunk_index=2,
            text="Table 1: Self-Attention complexity per layer is O(n^2 * d) with O(1) sequential operations.",
        )
    ]
    answer = "Self-Attention has a per-layer computational complexity of O(n^2 * d) with O(1) sequential operations [Evidence 1]."
    citations = [
        Citation(
            reference="[Evidence 1]",
            rank=1,
            document_id="doc-test-1",
            page_number=6,
            chunk_id="doc-test-1_p6_c2",
            score=0.95,
        )
    ]

    faithfulness = compute_faithfulness(answer=answer, evidence=evidence, question=sample_query.question)
    assert faithfulness == 1.0

    evaluator = GenerationEvaluator()
    res = evaluator.evaluate_query(
        query=sample_query,
        answer=answer,
        evidence=evidence,
        citations=citations,
    )

    assert res.generation_metrics.faithfulness == 1.0
    assert res.generation_metrics.citation_precision == 1.0
    assert res.generation_metrics.citation_recall == 1.0
    assert res.generation_metrics.abstention_correct is True
    assert res.is_grounded is True
    assert len(res.failure_categories) == 0
    assert res.passed_thresholds is True


# 2. Unsupported claim / Hallucination test
def test_unsupported_claim_hallucination(sample_query: GoldQuery) -> None:
    evidence = [
        RetrievedChunk(
            rank=1,
            score=0.90,
            chunk_id="doc-test-1_p6_c2",
            document_id="doc-test-1",
            page_number=6,
            chunk_index=2,
            text="The architecture uses multi-head attention with 8 attention heads.",
        )
    ]
    hallucinated_answer = (
        "Self-Attention uses quantum blockchain teleportation requiring O(2^n) quantum entanglement cycles."
    )

    faithfulness = compute_faithfulness(answer=hallucinated_answer, evidence=evidence, question=sample_query.question)
    assert faithfulness < 0.5

    evaluator = GenerationEvaluator()
    res = evaluator.evaluate_query(
        query=sample_query,
        answer=hallucinated_answer,
        evidence=evidence,
        citations=[],
    )

    assert res.is_grounded is False
    assert FailureCategory.GROUNDING_HALLUCINATION in res.failure_categories
    assert res.passed_thresholds is False


# 3. Irrelevant answer test
def test_irrelevant_answer(sample_query: GoldQuery) -> None:
    evidence = [
        RetrievedChunk(
            rank=1,
            score=0.90,
            chunk_id="doc-test-1_p6_c2",
            document_id="doc-test-1",
            page_number=6,
            chunk_index=2,
            text="Table 1: Self-Attention complexity per layer is O(n^2 * d).",
        )
    ]
    irrelevant_answer = "The weather in Seattle during winter is mostly rainy and overcast."

    relevancy = compute_answer_relevancy(
        question=sample_query.question,
        answer=irrelevant_answer,
        ground_truth_answer=sample_query.ground_truth_answer,
        is_answerable=True,
    )
    assert relevancy < 0.3


# 4. Missing relevant context / Empty retrieval test
def test_empty_retrieval_and_missing_context(sample_query: GoldQuery) -> None:
    empty_evidence: list[RetrievedChunk] = []

    cp = compute_context_precision(empty_evidence, sample_query.ground_truth_pages, k=5)
    cr = compute_context_recall(empty_evidence, sample_query.key_reference_facts)
    assert cp == 0.0
    assert cr == 0.0

    ret_evaluator = RetrievalEvaluator()
    ret_res = ret_evaluator.evaluate_query(sample_query, empty_evidence)
    assert ret_res.retrieval_metrics.recall_at_k == 0.0
    assert FailureCategory.RETRIEVAL_MISS in ret_res.failure_categories

    abstention_answer = INSUFFICIENT_EVIDENCE_ANSWER
    faithfulness = compute_faithfulness(abstention_answer, empty_evidence, is_refusal=True)
    assert faithfulness == 1.0


# 5. Correct context and context precision/recall test
def test_correct_and_partial_context(sample_query: GoldQuery) -> None:
    chunk1 = RetrievedChunk(
        rank=1,
        score=0.95,
        chunk_id="doc-test-1_p6_c2",
        document_id="doc-test-1",
        page_number=6,
        chunk_index=2,
        text="Table 1: Self-Attention complexity per layer is O(n^2 * d) with O(1) sequential operations.",
    )
    chunk2 = RetrievedChunk(
        rank=2,
        score=0.50,
        chunk_id="doc-test-1_p1_c1",
        document_id="doc-test-1",
        page_number=1,
        chunk_index=0,
        text="Introduction: Deep learning models have achieved great success.",
    )

    cp = compute_context_precision([chunk1, chunk2], sample_query.ground_truth_pages, k=2)
    assert cp == 1.0

    cp_low = compute_context_precision([chunk2, chunk1], sample_query.ground_truth_pages, k=2)
    assert cp_low == 0.5

    cr_full = compute_context_recall([chunk1], sample_query.key_reference_facts)
    assert cr_full == 1.0

    partial_chunk = RetrievedChunk(
        rank=1,
        score=0.80,
        chunk_id="doc-test-1_p6_c1",
        document_id="doc-test-1",
        page_number=6,
        chunk_index=1,
        text="Self-Attention complexity per layer is O(n^2 * d).",
    )
    cr_partial = compute_context_recall([partial_chunk], sample_query.key_reference_facts)
    assert cr_partial == 0.5


# 6. Negative / Unanswerable query and abstention correctness
def test_abstention_evaluation(unanswerable_query: GoldQuery) -> None:
    refusal_answer = INSUFFICIENT_EVIDENCE_ANSWER
    correct_abstention = compute_abstention_correctness(refusal_answer, is_answerable=False)
    assert correct_abstention is True

    rel = compute_answer_relevancy(
        unanswerable_query.question,
        refusal_answer,
        is_answerable=False,
    )
    assert rel == 1.0

    gen_eval = GenerationEvaluator()
    res_correct = gen_eval.evaluate_query(
        query=unanswerable_query,
        answer=refusal_answer,
        evidence=[],
    )
    assert res_correct.generation_metrics.abstention_correct is True
    assert FailureCategory.ABSTENTION_FAILURE not in res_correct.failure_categories
    assert res_correct.passed_thresholds is True

    hallucinated_answer = "The authors drink Starbucks Pike Place Roast every morning."
    res_fail = gen_eval.evaluate_query(
        query=unanswerable_query,
        answer=hallucinated_answer,
        evidence=[],
    )
    assert res_fail.generation_metrics.abstention_correct is False
    assert FailureCategory.ABSTENTION_FAILURE in res_fail.failure_categories
    assert res_fail.passed_thresholds is False


# 7. Missing ground truth handling
def test_missing_ground_truth_handling() -> None:
    query_no_gt = GoldQuery(
        query_id="q-no-gt",
        document_id="doc-x",
        question="What is X?",
        category=QuestionCategory.FACTOID_TEXT,
        ground_truth_pages=[],
        key_reference_facts=[],
        is_answerable=True,
    )

    cp = compute_context_precision([], query_no_gt.ground_truth_pages)
    cr = compute_context_recall([], query_no_gt.key_reference_facts)
    assert cp == 0.0
    assert cr == 0.0


# 8. Deterministic aggregation test across multiple queries
def test_deterministic_generation_aggregation() -> None:
    q1 = GoldQuery(
        query_id="q-001",
        document_id="doc-1",
        question="What is the secret code Alpha?",
        category=QuestionCategory.FACTOID_TEXT,
        ground_truth_pages=[1],
        ground_truth_evidence_text="The secret code is Alpha.",
        ground_truth_answer="The secret code is Alpha.",
        is_answerable=True,
    )
    q2 = GoldQuery(
        query_id="q-002",
        document_id="doc-1",
        question="Question 2",
        category=QuestionCategory.NEGATIVE_UNANSWERABLE,
        ground_truth_pages=[],
        is_answerable=False,
    )

    ev1 = [
        RetrievedChunk(
            rank=1,
            score=0.9,
            chunk_id="doc-1_p1_c1",
            document_id="doc-1",
            page_number=1,
            chunk_index=0,
            text="The secret code is Alpha.",
        )
    ]
    cit1 = [
        Citation(
            reference="[Evidence 1]",
            rank=1,
            document_id="doc-1",
            page_number=1,
            score=0.9,
        )
    ]

    precomputed = {
        "q-001": ("The secret code is Alpha [Evidence 1].", ev1, cit1),
        "q-002": (INSUFFICIENT_EVIDENCE_ANSWER, [], []),
    }

    evaluator = GenerationEvaluator()
    exp_result = evaluator.evaluate_queries(
        dataset=[q1, q2],
        precomputed_generations=precomputed,
        pipeline_config=PipelineConfig(name="test_pipeline"),
    )

    assert exp_result.summary.total_queries == 2
    assert exp_result.summary.mean_faithfulness == 1.0
    assert exp_result.summary.abstention_accuracy == 1.0
    assert exp_result.summary.mean_citation_precision == 1.0
    assert exp_result.summary.thresholds_passed is True


# 9. Judges test (Deterministic, Mock, and LLM judge with mock provider)
def test_judges(sample_query: GoldQuery) -> None:
    evidence = [
        RetrievedChunk(
            rank=1,
            score=0.95,
            chunk_id="doc-test-1_p6_c2",
            document_id="doc-test-1",
            page_number=6,
            chunk_index=2,
            text="Table 1: Self-Attention complexity per layer is O(n^2 * d).",
        )
    ]
    answer = "Self-Attention complexity per layer is O(n^2 * d) [Evidence 1]."

    mock_judge = MockLLMGenerationJudge(faithfulness=0.99, answer_relevancy=0.95)
    mock_metrics = mock_judge.evaluate(sample_query, answer, evidence)
    assert mock_metrics.faithfulness == 0.99
    assert mock_metrics.answer_relevancy == 0.95

    mock_provider = MockLLMProvider(
        mock_response='{"faithfulness": 0.92, "answer_relevancy": 0.88}'
    )
    llm_judge = LLMJudgeEvaluator(provider=mock_provider)
    llm_metrics = llm_judge.evaluate(sample_query, answer, evidence)
    assert llm_metrics.faithfulness == 0.92
    assert llm_metrics.answer_relevancy == 0.88


# 10. End-to-end RAG Evaluator on real Gold Dataset
def test_rag_evaluator_with_seed_dataset() -> None:
    seed_path = Path("data/gold_dataset.jsonl")
    if not seed_path.exists():
        seed_path = Path(__file__).resolve().parent.parent.parent / "data/gold_dataset.jsonl"
    if not seed_path.exists():
        pytest.skip("data/gold_dataset.jsonl not found")

    dataset = load_gold_dataset(seed_path)
    assert len(dataset.queries) > 0

    rag_eval = RAGEvaluator()

    for query in dataset.queries:
        if query.is_answerable:
            candidates = [
                RetrievedChunk(
                    rank=idx,
                    score=0.95 - (idx * 0.05),
                    chunk_id=(
                        query.ground_truth_chunks[idx - 1]
                        if idx - 1 < len(query.ground_truth_chunks)
                        else f"{query.document_id}_p{p}_c0"
                    ),
                    document_id=query.document_id,
                    page_number=p,
                    chunk_index=0,
                    text=query.ground_truth_evidence_text,
                )
                for idx, p in enumerate(query.ground_truth_pages, start=1)
            ]
            answer = f"{query.ground_truth_answer} [Evidence 1]"
            citations = [
                Citation(
                    reference=f"[Evidence {idx}]",
                    rank=idx,
                    document_id=query.document_id,
                    page_number=p,
                    score=0.95 - (idx * 0.05),
                )
                for idx, p in enumerate(query.ground_truth_pages, start=1)
            ]
        else:
            candidates = []
            answer = INSUFFICIENT_EVIDENCE_ANSWER
            citations = []

        res = rag_eval.evaluate_query(
            query=query,
            candidates=candidates,
            answer=answer,
            citations=citations,
        )

        assert res.query_id == query.query_id
        if query.is_answerable:
            assert res.retrieval_metrics.recall_at_k == 1.0
            assert res.generation_metrics.faithfulness is not None
            assert res.generation_metrics.faithfulness >= 0.80
        else:
            assert res.generation_metrics.abstention_correct is True
