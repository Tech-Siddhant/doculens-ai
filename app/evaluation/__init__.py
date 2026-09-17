"""Evaluation module for DocuLens AI."""

from app.evaluation.harness import (
    BASELINE_NAMES,
    EvaluationStores,
    create_isolated_stores,
    get_retrieval_function_for_baseline,
    get_shared_stores,
    index_evaluation_document,
    run_all_baselines,
    run_baseline_evaluation,
)
from app.evaluation.judges import (
    BaseGenerationJudge,
    DeterministicGenerationJudge,
    LLMJudgeEvaluator,
    MockLLMGenerationJudge,
)
from app.evaluation.metrics import (
    compute_abstention_correctness,
    compute_answer_relevancy,
    compute_citation_precision,
    compute_citation_recall,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_generation_metrics,
    compute_hit_at_k,
    compute_mrr_at_k,
    compute_recall_at_k,
    compute_retrieval_metrics,
    extract_candidate_info,
    extract_evidence_text,
    is_candidate_relevant,
)
from app.evaluation.regression import (
    RegressionThresholds,
    compare_against_baseline,
)
from app.evaluation.runner import (
    GenerationEvaluator,
    RAGEvaluator,
    RetrievalEvaluator,
    classify_generation_failures,
    classify_retrieval_failures,
)

# Singletons / default instances
retrieval_evaluator = RetrievalEvaluator()
generation_evaluator = GenerationEvaluator()
rag_evaluator = RAGEvaluator()

__all__ = [
    # Metrics
    "extract_candidate_info",
    "extract_evidence_text",
    "is_candidate_relevant",
    "compute_recall_at_k",
    "compute_mrr_at_k",
    "compute_hit_at_k",
    "compute_context_precision",
    "compute_context_recall",
    "compute_faithfulness",
    "compute_answer_relevancy",
    "compute_citation_precision",
    "compute_citation_recall",
    "compute_abstention_correctness",
    "compute_retrieval_metrics",
    "compute_generation_metrics",
    # Judges
    "BaseGenerationJudge",
    "DeterministicGenerationJudge",
    "MockLLMGenerationJudge",
    "LLMJudgeEvaluator",
    # Classifiers & Runners
    "classify_retrieval_failures",
    "classify_generation_failures",
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "RAGEvaluator",
    "retrieval_evaluator",
    "generation_evaluator",
    "rag_evaluator",
    # Regression
    "RegressionThresholds",
    "compare_against_baseline",
    # Harness & Baselines
    "BASELINE_NAMES",
    "EvaluationStores",
    "get_shared_stores",
    "create_isolated_stores",
    "index_evaluation_document",
    "get_retrieval_function_for_baseline",
    "run_baseline_evaluation",
    "run_all_baselines",
]
