import json
import pytest
from pydantic import ValidationError

from app.schemas.citation import Citation
from app.schemas.evaluation import (
    EvaluationThresholds,
    ExpectedModality,
    ExperimentResult,
    ExperimentSummary,
    FailureCategory,
    GenerationEvalMetrics,
    GoldDataset,
    GoldDocument,
    GoldQuery,
    OperationalMetrics,
    PipelineConfig,
    QuestionCategory,
    QueryEvaluationResult,
    RetrievalEvalMetrics,
)


def test_question_category_enum():
    assert QuestionCategory.FACTOID_TEXT == "factoid_text"
    assert QuestionCategory.MULTI_PAGE_REASONING == "multi_page_reasoning"
    assert QuestionCategory.TABLE_LOOKUP == "table_lookup"
    assert QuestionCategory.FIGURE_CHART_ANALYSIS == "figure_chart_analysis"
    assert QuestionCategory.METHODOLOGY_SUMMARY == "methodology_summary"
    assert QuestionCategory.NEGATIVE_UNANSWERABLE == "negative_unanswerable"


def test_expected_modality_enum():
    assert ExpectedModality.TEXT == "text"
    assert ExpectedModality.VISUAL == "visual"
    assert ExpectedModality.HYBRID == "hybrid"
    assert ExpectedModality.TABLE == "table"
    assert ExpectedModality.FIGURE == "figure"


def test_failure_category_enum():
    assert FailureCategory.RETRIEVAL_MISS == "retrieval_miss"
    assert FailureCategory.RETRIEVAL_LOW_RANK == "retrieval_low_rank"
    assert FailureCategory.VISUAL_MISS == "visual_miss"
    assert FailureCategory.PARSING_FAILURE == "parsing_failure"
    assert FailureCategory.GROUNDING_HALLUCINATION == "grounding_hallucination"
    assert FailureCategory.CITATION_FABRICATION == "citation_fabrication"
    assert FailureCategory.ANSWER_INCOMPLETENESS == "answer_incompleteness"
    assert FailureCategory.ABSTENTION_FAILURE == "abstention_failure"
    assert FailureCategory.FALSE_ABSTENTION == "false_abstention"
    assert FailureCategory.SYSTEM_TIMEOUT_ERROR == "system_timeout_error"


def test_pipeline_config_defaults():
    config = PipelineConfig(name="test_pipeline")
    assert config.name == "test_pipeline"
    assert config.retrieval_types == ["dense", "bm25", "visual"]
    assert config.weights == {"dense": 0.5, "bm25": 0.3, "visual": 0.2}
    assert config.reranker_enabled is True
    assert config.reranker_model == "BAAI/bge-reranker-base"
    assert config.top_k_retrieve == 10
    assert config.top_k_rerank == 5
    assert config.evidence_score_threshold is None
    assert config.llm_provider == "mock"
    assert config.llm_model == "gpt-4o-mini"
    assert config.temperature == 0.0


def test_pipeline_config_custom_and_validation():
    config = PipelineConfig(
        name="custom_dense",
        retrieval_types=["dense"],
        weights={"dense": 1.0},
        reranker_enabled=False,
        top_k_retrieve=20,
        top_k_rerank=10,
        evidence_score_threshold=0.75,
        llm_provider="openai",
        llm_model="gpt-4o",
        temperature=0.2,
    )
    assert config.name == "custom_dense"
    assert config.temperature == 0.2
    assert config.evidence_score_threshold == 0.75

    with pytest.raises(ValidationError):
        PipelineConfig(name="invalid_temp", temperature=2.5)

    with pytest.raises(ValidationError):
        PipelineConfig(name="invalid_top_k", top_k_retrieve=0)


def test_gold_document_validation():
    doc = GoldDocument(
        document_id="doc-1",
        document_title="Attention Is All You Need",
        file_name="attention.pdf",
        total_pages=15,
        domain="machine_learning",
        metadata={"year": 2017},
    )
    assert doc.document_id == "doc-1"
    assert doc.total_pages == 15
    assert doc.metadata["year"] == 2017

    with pytest.raises(ValidationError):
        GoldDocument(
            document_id="",
            document_title="Title",
            file_name="file.pdf",
            total_pages=10,
        )

    with pytest.raises(ValidationError):
        GoldDocument(
            document_id="doc-1",
            document_title="Title",
            file_name="file.pdf",
            total_pages=0,
        )


def test_gold_query_validation():
    query = GoldQuery(
        query_id="q-001",
        document_id="doc-1",
        question="What is the complexity of Self-Attention?",
        category=QuestionCategory.TABLE_LOOKUP,
        expected_sources=[ExpectedModality.TEXT, ExpectedModality.TABLE],
        ground_truth_pages=[6],
        ground_truth_chunks=["doc-1_p6_c1"],
        ground_truth_evidence_text="Table 1: Self-Attention has O(n^2 * d) complexity.",
        ground_truth_answer="Self-Attention has O(n^2 * d) complexity per layer.",
        key_reference_facts=["Complexity is O(n^2 * d)"],
        is_answerable=True,
        difficulty="medium",
        metadata={"table_id": "Table 1"},
    )
    assert query.query_id == "q-001"
    assert query.category == QuestionCategory.TABLE_LOOKUP
    assert query.expected_sources == [ExpectedModality.TEXT, ExpectedModality.TABLE]
    assert query.ground_truth_pages == [6]
    assert query.is_answerable is True

    with pytest.raises(ValidationError):
        GoldQuery(
            query_id="",
            document_id="doc-1",
            question="Question",
            category=QuestionCategory.FACTOID_TEXT,
        )


def test_gold_dataset_operations_and_serialization():
    doc = GoldDocument(
        document_id="doc-1",
        document_title="Test Doc",
        file_name="test.pdf",
        total_pages=5,
    )
    q1 = GoldQuery(
        query_id="q-001",
        document_id="doc-1",
        question="What is X?",
        category=QuestionCategory.FACTOID_TEXT,
        ground_truth_pages=[1],
    )
    q2 = GoldQuery(
        query_id="q-002",
        document_id="doc-1",
        question="What is shown in Figure 2?",
        category=QuestionCategory.FIGURE_CHART_ANALYSIS,
        ground_truth_pages=[3],
    )
    q3 = GoldQuery(
        query_id="q-003",
        document_id="doc-1",
        question="What is the capital of Mars?",
        category=QuestionCategory.NEGATIVE_UNANSWERABLE,
        is_answerable=False,
    )

    dataset = GoldDataset(
        dataset_id="benchmark-v1",
        name="Test Benchmark",
        version="1.0.0",
        description="A unit test benchmark dataset",
        documents=[doc],
        queries=[q1, q2, q3],
    )

    assert dataset.total_queries == 3
    assert dataset.get_query("q-002") == q2
    assert dataset.get_query("q-999") is None
    assert dataset.get_document("doc-1") == doc
    assert dataset.get_document("doc-999") is None

    factoid_queries = dataset.filter_by_category(QuestionCategory.FACTOID_TEXT)
    assert len(factoid_queries) == 1
    assert factoid_queries[0].query_id == "q-001"

    # Test JSON serialization roundtrip
    raw_json = dataset.model_dump_json()
    loaded_dataset = GoldDataset.model_validate_json(raw_json)
    assert loaded_dataset.dataset_id == dataset.dataset_id
    assert loaded_dataset.total_queries == 3
    assert loaded_dataset.queries[1].category == QuestionCategory.FIGURE_CHART_ANALYSIS


def test_retrieval_eval_metrics():
    metrics = RetrievalEvalMetrics(
        recall_at_k=1.0,
        mrr_at_k=0.5,
        context_precision=0.8,
        context_recall=1.0,
        hit_at_k=True,
        retrieved_pages=[2, 1, 4],
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
    )
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr_at_k == 0.5
    assert metrics.hit_at_k is True
    assert metrics.retrieved_pages == [2, 1, 4]

    with pytest.raises(ValidationError):
        RetrievalEvalMetrics(recall_at_k=1.5)

    with pytest.raises(ValidationError):
        RetrievalEvalMetrics(mrr_at_k=-0.1)


def test_generation_eval_metrics():
    gen_metrics = GenerationEvalMetrics(
        faithfulness=0.95,
        answer_relevancy=0.90,
        citation_precision=1.0,
        citation_recall=0.8,
        abstention_correct=True,
    )
    assert gen_metrics.faithfulness == 0.95
    assert gen_metrics.citation_precision == 1.0
    assert gen_metrics.abstention_correct is True

    with pytest.raises(ValidationError):
        GenerationEvalMetrics(faithfulness=1.2)


def test_operational_metrics():
    ops = OperationalMetrics(
        retrieval_latency_ms=120.5,
        reranking_latency_ms=80.2,
        generation_latency_ms=650.0,
        total_latency_ms=850.7,
        prompt_tokens=450,
        completion_tokens=120,
        total_tokens=570,
    )
    assert ops.total_latency_ms == 850.7
    assert ops.total_tokens == 570


def test_query_evaluation_result_and_failure_classification():
    citation = Citation(
        reference="[Evidence 1]",
        rank=1,
        score=0.95,
        chunk_id="chunk-1",
        document_id="doc-1",
        page_number=6,
        evidence_text="Table 1: Self-Attention...",
    )
    result = QueryEvaluationResult(
        query_id="q-001",
        document_id="doc-1",
        category=QuestionCategory.TABLE_LOOKUP,
        question="What is Self-Attention complexity?",
        is_answerable=True,
        retrieval_metrics=RetrievalEvalMetrics(recall_at_k=1.0, mrr_at_k=1.0, hit_at_k=True),
        generation_metrics=GenerationEvalMetrics(faithfulness=1.0, citation_precision=1.0),
        operational_metrics=OperationalMetrics(total_latency_ms=450.0),
        generated_answer="Self-Attention complexity is O(n^2 * d) [Evidence 1].",
        citations=[citation],
        is_grounded=True,
        failure_categories=[],
        passed_thresholds=True,
    )
    assert result.query_id == "q-001"
    assert len(result.citations) == 1
    assert result.passed_thresholds is True
    assert len(result.failure_categories) == 0

    # Test failure result
    failure_result = QueryEvaluationResult(
        query_id="q-002",
        document_id="doc-1",
        category=QuestionCategory.FIGURE_CHART_ANALYSIS,
        question="Explain Figure 3",
        is_answerable=True,
        retrieval_metrics=RetrievalEvalMetrics(recall_at_k=0.0, mrr_at_k=0.0, hit_at_k=False),
        failure_categories=[FailureCategory.VISUAL_MISS, FailureCategory.RETRIEVAL_MISS],
        passed_thresholds=False,
        notes="Visual embedding failed to retrieve page 4",
    )
    assert failure_result.passed_thresholds is False
    assert FailureCategory.VISUAL_MISS in failure_result.failure_categories


def test_experiment_summary_and_result():
    thresholds = EvaluationThresholds(
        min_recall_at_k=0.80,
        min_mrr_at_k=0.70,
        min_faithfulness=0.85,
    )
    summary = ExperimentSummary(
        total_queries=10,
        mean_recall_at_k=0.85,
        mean_mrr_at_k=0.75,
        mean_faithfulness=0.90,
        mean_citation_precision=0.95,
        abstention_accuracy=1.0,
        mean_total_latency_ms=520.0,
        p95_total_latency_ms=1100.0,
        category_metrics={"table_lookup": {"recall_at_k": 0.88}},
        failure_counts={"retrieval_miss": 1},
        thresholds=thresholds,
        thresholds_passed=True,
    )
    assert summary.total_queries == 10
    assert summary.thresholds_passed is True

    pipeline_config = PipelineConfig(name="hybrid_reranked")
    experiment = ExperimentResult(
        experiment_id="exp-001",
        name="Hybrid Multimodal Benchmark Run",
        pipeline_config=pipeline_config,
        dataset_id="benchmark-v1",
        dataset_version="1.0.0",
        summary=summary,
        query_results=[],
    )

    assert experiment.experiment_id == "exp-001"
    assert experiment.pipeline_config.name == "hybrid_reranked"

    # Test full JSON roundtrip
    raw_json = experiment.model_dump_json()
    loaded_exp = ExperimentResult.model_validate_json(raw_json)
    assert loaded_exp.experiment_id == "exp-001"
    assert loaded_exp.summary.mean_recall_at_k == 0.85


def test_schema_exports():
    import app.schemas as schemas

    assert hasattr(schemas, "QuestionCategory")
    assert hasattr(schemas, "ExpectedModality")
    assert hasattr(schemas, "FailureCategory")
    assert hasattr(schemas, "PipelineConfig")
    assert hasattr(schemas, "GoldDocument")
    assert hasattr(schemas, "GoldQuery")
    assert hasattr(schemas, "GoldDataset")
    assert hasattr(schemas, "RetrievalEvalMetrics")
    assert hasattr(schemas, "GenerationEvalMetrics")
    assert hasattr(schemas, "OperationalMetrics")
    assert hasattr(schemas, "QueryEvaluationResult")
    assert hasattr(schemas, "EvaluationThresholds")
    assert hasattr(schemas, "ExperimentSummary")
    assert hasattr(schemas, "ExperimentResult")
