from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.citation import Citation


class QuestionCategory(str, Enum):
    """Categorical taxonomy for evaluation questions."""

    FACTOID_TEXT = "factoid_text"
    MULTI_PAGE_REASONING = "multi_page_reasoning"
    TABLE_LOOKUP = "table_lookup"
    FIGURE_CHART_ANALYSIS = "figure_chart_analysis"
    METHODOLOGY_SUMMARY = "methodology_summary"
    NEGATIVE_UNANSWERABLE = "negative_unanswerable"


class ExpectedModality(str, Enum):
    """Expected retrieval modalities required to answer a gold query."""

    TEXT = "text"
    VISUAL = "visual"
    HYBRID = "hybrid"
    TABLE = "table"
    FIGURE = "figure"


class FailureCategory(str, Enum):
    """Granular failure taxonomy for root-cause error analysis."""

    RETRIEVAL_MISS = "retrieval_miss"
    RETRIEVAL_LOW_RANK = "retrieval_low_rank"
    VISUAL_MISS = "visual_miss"
    PARSING_FAILURE = "parsing_failure"
    GROUNDING_HALLUCINATION = "grounding_hallucination"
    CITATION_FABRICATION = "citation_fabrication"
    ANSWER_INCOMPLETENESS = "answer_incompleteness"
    ABSTENTION_FAILURE = "abstention_failure"
    FALSE_ABSTENTION = "false_abstention"
    SYSTEM_TIMEOUT_ERROR = "system_timeout_error"


class PipelineConfig(BaseModel):
    """Configuration definition for a specific pipeline evaluation variant."""

    name: str = Field(..., description="Pipeline variant identifier, e.g. 'hybrid_reranked'")
    retrieval_types: list[str] = Field(
        default_factory=lambda: ["dense", "bm25", "visual"],
        description="Active retrieval channels",
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {"dense": 0.5, "bm25": 0.3, "visual": 0.2},
        description="Modality weights used during hybrid fusion",
    )
    reranker_enabled: bool = Field(
        default=True, description="Whether cross-encoder reranker is enabled"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base", description="Reranker model identifier"
    )
    top_k_retrieve: int = Field(
        default=10, ge=1, description="Number of candidates to retrieve per modality"
    )
    top_k_rerank: int = Field(
        default=5, ge=1, description="Number of top candidates preserved after reranking"
    )
    evidence_score_threshold: float | None = Field(
        default=None, description="Optional minimum score cutoff for evidence validation"
    )
    llm_provider: str = Field(
        default="mock", description="LLM provider name (e.g. mock, openai, anthropic)"
    )
    llm_model: str = Field(
        default="gpt-4o-mini", description="LLM model identifier"
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature"
    )


class GoldDocument(BaseModel):
    """Metadata for a document referenced in the gold evaluation benchmark."""

    document_id: str = Field(..., min_length=1, description="Unique document identifier")
    document_title: str = Field(..., min_length=1, description="Human-readable document title")
    file_name: str = Field(..., min_length=1, description="Source PDF filename")
    total_pages: int = Field(..., ge=1, description="Total number of pages in the document")
    domain: str = Field(default="technical_research", description="Document domain/category")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional document metadata (year, authors, arxiv ID)"
    )


class GoldQuery(BaseModel):
    """Single benchmark evaluation question with ground-truth references."""

    query_id: str = Field(..., min_length=1, description="Unique query identifier (e.g. q-001)")
    document_id: str = Field(..., min_length=1, description="Target document identifier")
    question: str = Field(..., min_length=1, description="User question text")
    category: QuestionCategory = Field(
        ..., description="Taxonomy classification for the question"
    )
    expected_sources: list[ExpectedModality] = Field(
        default_factory=lambda: [ExpectedModality.TEXT],
        description="Modalities expected to resolve this query",
    )
    ground_truth_pages: list[int] = Field(
        default_factory=list,
        description="1-based page numbers containing essential evidence",
    )
    ground_truth_chunks: list[str] = Field(
        default_factory=list,
        description="Optional specific chunk IDs containing supporting evidence",
    )
    ground_truth_evidence_text: str = Field(
        default="",
        description="Reference excerpt/snippet containing the necessary evidence",
    )
    ground_truth_answer: str = Field(
        default="", description="Canonical ground-truth answer"
    )
    key_reference_facts: list[str] = Field(
        default_factory=list,
        description="Atomic factual points required for a complete answer",
    )
    is_answerable: bool = Field(
        default=True,
        description="Whether the document contains sufficient evidence to answer (False for negative tests)",
    )
    difficulty: str = Field(
        default="medium", description="Subjective difficulty rating: easy, medium, hard"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary query metadata (section, table id, notes)"
    )


class GoldDataset(BaseModel):
    """Collection of benchmark evaluation queries and document specifications."""

    dataset_id: str = Field(..., min_length=1, description="Unique dataset identifier")
    name: str = Field(..., min_length=1, description="Human-readable dataset name")
    version: str = Field(default="1.0.0", description="Semantic version of the dataset")
    description: str = Field(default="", description="Dataset description and scope")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of creation",
    )
    documents: list[GoldDocument] = Field(
        default_factory=list, description="Documents covered in this benchmark"
    )
    queries: list[GoldQuery] = Field(
        default_factory=list, description="Curated evaluation queries"
    )

    @property
    def total_queries(self) -> int:
        """Return total number of queries in the dataset."""
        return len(self.queries)

    def get_query(self, query_id: str) -> GoldQuery | None:
        """Find query by unique query_id."""
        for q in self.queries:
            if q.query_id == query_id:
                return q
        return None

    def get_document(self, document_id: str) -> GoldDocument | None:
        """Find document metadata by document_id."""
        for d in self.documents:
            if d.document_id == document_id:
                return d
        return None

    def filter_by_category(self, category: QuestionCategory) -> list[GoldQuery]:
        """Filter queries belonging to a specific taxonomy category."""
        return [q for q in self.queries if q.category == category]


class RetrievalEvalMetrics(BaseModel):
    """Retrieval evaluation metrics computed for a single query."""

    recall_at_k: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of ground-truth pages retrieved in top-k"
    )
    mrr_at_k: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Mean reciprocal rank of first relevant page"
    )
    context_precision: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Precision of retrieved evidence ranking"
    )
    context_recall: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of reference facts covered in retrieved context"
    )
    hit_at_k: bool = Field(
        default=False, description="Whether at least one ground-truth page was retrieved in top-k"
    )
    retrieved_pages: list[int] = Field(
        default_factory=list, description="Ordered list of page numbers retrieved"
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list, description="Ordered list of chunk IDs retrieved"
    )
    k_metrics: dict[int, dict[str, float]] = Field(
        default_factory=dict,
        description="Metric values broken down by specific K value (e.g. {1: {'recall': 0.5, 'mrr': 0.5}})",
    )


class GenerationEvalMetrics(BaseModel):
    """Generation and grounding evaluation metrics computed for a single query."""

    faithfulness: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Ratio of claims entailed by context (1.0 = 0 hallucination)"
    )
    answer_relevancy: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Relevance of generated answer to the question"
    )
    citation_precision: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Ratio of valid citations to total citations"
    )
    citation_recall: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Ratio of ground-truth pages cited in answer"
    )
    abstention_correct: bool | None = Field(
        default=None, description="Whether model correctly abstained on unanswerable query"
    )


class OperationalMetrics(BaseModel):
    """Operational latency and token usage metrics."""

    retrieval_latency_ms: float = Field(default=0.0, ge=0.0, description="Retrieval duration in ms")
    reranking_latency_ms: float = Field(default=0.0, ge=0.0, description="Reranking duration in ms")
    generation_latency_ms: float = Field(default=0.0, ge=0.0, description="LLM generation duration in ms")
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="Total pipeline latency in ms")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt tokens consumed")
    completion_tokens: int = Field(default=0, ge=0, description="Completion tokens generated")
    total_tokens: int = Field(default=0, ge=0, description="Total token count")


class QueryEvaluationResult(BaseModel):
    """Complete evaluation outcome for a single query."""

    query_id: str = Field(..., description="Evaluated query identifier")
    document_id: str = Field(..., description="Document identifier")
    category: QuestionCategory = Field(..., description="Question taxonomy category")
    question: str = Field(..., description="Question text")
    is_answerable: bool = Field(default=True, description="Whether query was answerable")
    retrieval_metrics: RetrievalEvalMetrics = Field(
        default_factory=RetrievalEvalMetrics, description="Retrieval quality metrics"
    )
    generation_metrics: GenerationEvalMetrics = Field(
        default_factory=GenerationEvalMetrics, description="Generation quality metrics"
    )
    operational_metrics: OperationalMetrics = Field(
        default_factory=OperationalMetrics, description="Latency and token metrics"
    )
    generated_answer: str = Field(default="", description="Generated answer text")
    citations: list[Citation] = Field(
        default_factory=list, description="Validated citations produced"
    )
    is_grounded: bool = Field(default=True, description="Whether answer was marked as grounded")
    failure_categories: list[FailureCategory] = Field(
        default_factory=list, description="Classified failure categories if any"
    )
    passed_thresholds: bool = Field(
        default=True, description="Whether this query passed target evaluation criteria"
    )
    notes: str = Field(default="", description="Diagnostic notes or error messages")


class EvaluationThresholds(BaseModel):
    """Planned target thresholds and minimum pass criteria for pipeline evaluation."""

    min_recall_at_k: float = Field(default=0.75, ge=0.0, le=1.0)
    min_mrr_at_k: float = Field(default=0.65, ge=0.0, le=1.0)
    min_context_precision: float = Field(default=0.70, ge=0.0, le=1.0)
    min_context_recall: float = Field(default=0.75, ge=0.0, le=1.0)
    min_faithfulness: float = Field(default=0.85, ge=0.0, le=1.0)
    min_answer_relevancy: float = Field(default=0.80, ge=0.0, le=1.0)
    min_citation_precision: float = Field(default=0.90, ge=0.0, le=1.0)
    min_citation_recall: float = Field(default=0.80, ge=0.0, le=1.0)
    min_abstention_accuracy: float = Field(default=0.85, ge=0.0, le=1.0)
    max_p95_latency_ms: float = Field(default=5000.0, ge=0.0)


class ExperimentSummary(BaseModel):
    """Aggregated evaluation metrics summary across an entire benchmark dataset."""

    total_queries: int = Field(default=0, ge=0)
    mean_recall_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_mrr_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_context_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_context_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_citation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_citation_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    abstention_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_total_latency_ms: float = Field(default=0.0, ge=0.0)
    p95_total_latency_ms: float = Field(default=0.0, ge=0.0)
    k_metrics: dict[int, dict[str, float]] = Field(
        default_factory=dict,
        description="Mean metric values for each configured K (e.g. {1: {'recall': 0.5, 'mrr': 0.5}})",
    )
    category_metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metrics broken down per QuestionCategory"
    )
    failure_counts: dict[str, int] = Field(
        default_factory=dict, description="Occurrence count per FailureCategory"
    )
    thresholds: EvaluationThresholds = Field(
        default_factory=EvaluationThresholds, description="Target thresholds applied"
    )
    thresholds_passed: bool = Field(
        default=False, description="Whether aggregate metrics satisfy all target thresholds"
    )


class ExperimentResult(BaseModel):
    """Complete evaluation experiment artifact with configuration, summary, and query-level traces."""

    experiment_id: str = Field(..., description="Unique experiment run identifier")
    name: str = Field(..., description="Human-readable experiment name")
    pipeline_config: PipelineConfig = Field(..., description="Evaluated pipeline configuration")
    dataset_id: str = Field(..., description="Evaluated dataset identifier")
    dataset_version: str = Field(..., description="Evaluated dataset version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of experiment completion",
    )
    summary: ExperimentSummary = Field(
        default_factory=ExperimentSummary, description="Aggregated experiment summary"
    )
    query_results: list[QueryEvaluationResult] = Field(
        default_factory=list, description="Per-query evaluation records"
    )


class MetricDelta(BaseModel):
    """Delta comparison between a current evaluation metric and a baseline."""

    metric_name: str = Field(..., description="Identifier for the evaluated metric")
    baseline_value: float | None = Field(default=None, description="Metric value from baseline run")
    current_value: float | None = Field(default=None, description="Metric value from current run")
    delta: float | None = Field(default=None, description="Absolute difference (current - baseline)")
    relative_change_pct: float | None = Field(
        default=None, description="Percentage change relative to baseline"
    )
    is_regression: bool = Field(
        default=False, description="Whether delta breaches regression tolerance threshold"
    )
    is_improvement: bool = Field(
        default=False, description="Whether delta represents significant positive improvement"
    )


class RegressionReport(BaseModel):
    """Structured report comparing current evaluation run against a designated baseline."""

    status: str = Field(
        ..., description="Overall status: 'PASSED', 'REGRESSION_DETECTED', 'IMPROVED', or 'NEUTRAL'"
    )
    passed: bool = Field(
        ..., description="True if no metric experienced unacceptable regression"
    )
    baseline_id: str | None = Field(default=None, description="Baseline experiment identifier")
    current_id: str | None = Field(default=None, description="Current experiment identifier")
    total_metrics_evaluated: int = Field(default=0, ge=0)
    degradations: list[str] = Field(
        default_factory=list, description="List of metric degradation descriptions"
    )
    improvements: list[str] = Field(
        default_factory=list, description="List of metric improvement descriptions"
    )
    metric_deltas: dict[str, MetricDelta] = Field(
        default_factory=dict, description="Detailed per-metric delta records"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of report generation",
    )


__all__ = [
    "QuestionCategory",
    "ExpectedModality",
    "FailureCategory",
    "PipelineConfig",
    "GoldDocument",
    "GoldQuery",
    "GoldDataset",
    "RetrievalEvalMetrics",
    "GenerationEvalMetrics",
    "OperationalMetrics",
    "QueryEvaluationResult",
    "EvaluationThresholds",
    "ExperimentSummary",
    "ExperimentResult",
    "MetricDelta",
    "RegressionReport",
]
