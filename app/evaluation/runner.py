"""Evaluation runner and experiment harness for DocuLens AI — Phase 6.3 & 6.4.

Executes deterministic retrieval, generation, and end-to-end RAG evaluations over
gold benchmark datasets. Computes per-query metrics, performs root-cause failure
classification, and aggregates results for configuration comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import math
import time
from typing import Any
import uuid

from app.evaluation.judges import BaseGenerationJudge, DeterministicGenerationJudge
from app.evaluation.metrics import (
    compute_generation_metrics,
    compute_retrieval_metrics,
)
from app.schemas.citation import Citation
from app.schemas.evaluation import (
    EvaluationThresholds,
    ExpectedModality,
    ExperimentResult,
    ExperimentSummary,
    FailureCategory,
    GenerationEvalMetrics,
    GoldDataset,
    GoldQuery,
    OperationalMetrics,
    PipelineConfig,
    QuestionCategory,
    QueryEvaluationResult,
    RetrievalEvalMetrics,
)
from app.schemas.generation import GenerationResult
from app.services.citation_validator import is_refusal_response


def classify_retrieval_failures(
    query: GoldQuery,
    metrics: RetrievalEvalMetrics,
) -> list[FailureCategory]:
    """Deterministically classify retrieval failure categories for a query."""
    if not query.is_answerable or not query.ground_truth_pages:
        return []

    failures: list[FailureCategory] = []

    # 1. Complete retrieval miss in top-k
    if metrics.recall_at_k == 0.0:
        failures.append(FailureCategory.RETRIEVAL_MISS)
        visual_modalities = {ExpectedModality.VISUAL, ExpectedModality.FIGURE}
        if any(src in visual_modalities for src in query.expected_sources):
            failures.append(FailureCategory.VISUAL_MISS)
    # 2. Retrieved but at low rank (first hit after rank 2: MRR < 0.5)
    elif metrics.mrr_at_k < 0.5:
        failures.append(FailureCategory.RETRIEVAL_LOW_RANK)

    return failures


def classify_generation_failures(
    query: GoldQuery,
    answer: str,
    gen_metrics: GenerationEvalMetrics,
    ret_metrics: RetrievalEvalMetrics | None = None,
    thresholds: EvaluationThresholds | None = None,
) -> list[FailureCategory]:
    """Deterministically classify generation and grounding failure categories for a query."""
    failures: list[FailureCategory] = []
    thresh = thresholds or EvaluationThresholds()
    is_refusal = is_refusal_response(answer)

    # 1. Negative / Unanswerable query failures
    if not query.is_answerable:
        if not is_refusal or gen_metrics.abstention_correct is False:
            failures.append(FailureCategory.ABSTENTION_FAILURE)
        return failures

    # 2. Answerable query failures
    if is_refusal:
        # If retrieval had hits (recall > 0), this is a false abstention
        if ret_metrics is None or ret_metrics.recall_at_k > 0:
            failures.append(FailureCategory.FALSE_ABSTENTION)
        return failures

    # 3. Grounding / Hallucination failure
    if gen_metrics.faithfulness is not None and gen_metrics.faithfulness < thresh.min_faithfulness:
        failures.append(FailureCategory.GROUNDING_HALLUCINATION)

    # 4. Citation Fabrication failure
    if (
        gen_metrics.citation_precision is not None
        and gen_metrics.citation_precision < thresh.min_citation_precision
    ):
        failures.append(FailureCategory.CITATION_FABRICATION)

    # 5. Answer Incompleteness / Low citation recall
    if (
        gen_metrics.citation_recall is not None
        and gen_metrics.citation_recall < thresh.min_citation_recall
    ):
        failures.append(FailureCategory.ANSWER_INCOMPLETENESS)

    return failures


class RetrievalEvaluator:
    """Evaluates retrieval quality independently from LLM generation."""

    def __init__(
        self,
        k_values: Sequence[int] = (1, 3, 5, 10),
        default_k: int = 5,
        thresholds: EvaluationThresholds | None = None,
    ) -> None:
        self.k_values = list(k_values)
        self.default_k = default_k
        self.thresholds = thresholds or EvaluationThresholds()

    def evaluate_query(
        self,
        query: GoldQuery,
        candidates: Sequence[Any],
        operational_metrics: OperationalMetrics | None = None,
    ) -> QueryEvaluationResult:
        """Evaluate retrieval candidates for a single gold query."""
        metrics = compute_retrieval_metrics(
            retrieved=candidates,
            ground_truth_pages=query.ground_truth_pages,
            k_values=self.k_values,
            default_k=self.default_k,
            target_document_id=query.document_id,
            ground_truth_chunks=query.ground_truth_chunks,
            key_reference_facts=query.key_reference_facts,
            ground_truth_evidence_text=query.ground_truth_evidence_text,
        )

        failures = classify_retrieval_failures(query, metrics)

        # Check per-query thresholds
        if query.is_answerable and query.ground_truth_pages:
            passed = (
                metrics.recall_at_k >= self.thresholds.min_recall_at_k
                and metrics.mrr_at_k >= self.thresholds.min_mrr_at_k
            )
        else:
            passed = True

        note_str = f"Failures: {', '.join(f.value for f in failures)}" if failures else ""

        return QueryEvaluationResult(
            query_id=query.query_id,
            document_id=query.document_id,
            category=query.category,
            question=query.question,
            is_answerable=query.is_answerable,
            retrieval_metrics=metrics,
            operational_metrics=operational_metrics or OperationalMetrics(),
            failure_categories=failures,
            passed_thresholds=passed,
            notes=note_str,
        )

    def evaluate_queries(
        self,
        dataset: Sequence[GoldQuery] | Any,
        retrieval_fn: Callable[[GoldQuery], Sequence[Any]] | None = None,
        precomputed_results: Mapping[str, Sequence[Any]] | None = None,
        pipeline_config: PipelineConfig | None = None,
        dataset_id: str = "gold_benchmark",
        dataset_version: str = "1.0.0",
        experiment_id: str | None = None,
        experiment_name: str | None = None,
    ) -> ExperimentResult:
        """Execute evaluation across a collection of gold queries."""
        queries: list[GoldQuery] = getattr(dataset, "queries", dataset)
        if not isinstance(queries, list):
            queries = list(queries)

        query_results: list[QueryEvaluationResult] = []
        failure_counts: dict[str, int] = {}

        for query in queries:
            if precomputed_results and query.query_id in precomputed_results:
                candidates = precomputed_results[query.query_id]
            elif retrieval_fn:
                candidates = retrieval_fn(query)
            else:
                candidates = []

            result = self.evaluate_query(query=query, candidates=candidates)
            query_results.append(result)

            for failure in result.failure_categories:
                failure_counts[failure.value] = failure_counts.get(failure.value, 0) + 1

        eval_queries = [
            r
            for r in query_results
            if r.is_answerable
            and len(getattr(self._find_query(queries, r.query_id), "ground_truth_pages", [])) > 0
        ]

        total_queries = len(queries)
        eval_count = len(eval_queries)

        if eval_count > 0:
            mean_rec = sum(r.retrieval_metrics.recall_at_k for r in eval_queries) / eval_count
            mean_mrr = sum(r.retrieval_metrics.mrr_at_k for r in eval_queries) / eval_count
            valid_cp = [
                r.retrieval_metrics.context_precision
                for r in eval_queries
                if r.retrieval_metrics.context_precision is not None
            ]
            mean_cp = sum(valid_cp) / len(valid_cp) if valid_cp else None
            valid_cr = [
                r.retrieval_metrics.context_recall
                for r in eval_queries
                if r.retrieval_metrics.context_recall is not None
            ]
            mean_cr = sum(valid_cr) / len(valid_cr) if valid_cr else None
        else:
            mean_rec = 0.0
            mean_mrr = 0.0
            mean_cp = None
            mean_cr = None

        # Multi-K aggregations
        k_metrics: dict[int, dict[str, float]] = {}
        for k in self.k_values:
            if eval_count > 0:
                k_rec = (
                    sum(
                        r.retrieval_metrics.k_metrics.get(k, {}).get("recall", 0.0)
                        for r in eval_queries
                    )
                    / eval_count
                )
                k_mrr = (
                    sum(
                        r.retrieval_metrics.k_metrics.get(k, {}).get("mrr", 0.0)
                        for r in eval_queries
                    )
                    / eval_count
                )
            else:
                k_rec = 0.0
                k_mrr = 0.0
            k_metrics[k] = {"recall": round(k_rec, 6), "mrr": round(k_mrr, 6)}

        # Category breakdown
        category_metrics: dict[str, dict[str, float]] = {}
        for cat in QuestionCategory:
            cat_results = [r for r in eval_queries if r.category == cat]
            if cat_results:
                c_len = len(cat_results)
                c_rec = sum(r.retrieval_metrics.recall_at_k for r in cat_results) / c_len
                c_mrr = sum(r.retrieval_metrics.mrr_at_k for r in cat_results) / c_len
                c_hit = sum(1.0 for r in cat_results if r.retrieval_metrics.hit_at_k) / c_len
                category_metrics[cat.value] = {
                    "count": float(c_len),
                    "mean_recall_at_k": round(c_rec, 6),
                    "mean_mrr_at_k": round(c_mrr, 6),
                    "hit_rate_at_k": round(c_hit, 6),
                }

        thresholds_passed = (
            mean_rec >= self.thresholds.min_recall_at_k
            and mean_mrr >= self.thresholds.min_mrr_at_k
        )

        summary = ExperimentSummary(
            total_queries=total_queries,
            mean_recall_at_k=round(mean_rec, 6),
            mean_mrr_at_k=round(mean_mrr, 6),
            mean_context_precision=round(mean_cp, 6) if mean_cp is not None else None,
            mean_context_recall=round(mean_cr, 6) if mean_cr is not None else None,
            k_metrics=k_metrics,
            category_metrics=category_metrics,
            failure_counts=failure_counts,
            thresholds=self.thresholds,
            thresholds_passed=thresholds_passed,
        )

        config = pipeline_config or PipelineConfig(name="default_retrieval")
        exp_id = experiment_id or f"exp-{uuid.uuid4().hex[:8]}"
        exp_name = experiment_name or f"Retrieval Evaluation ({config.name})"

        return ExperimentResult(
            experiment_id=exp_id,
            name=exp_name,
            pipeline_config=config,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            summary=summary,
            query_results=query_results,
        )

    def _find_query(self, queries: Sequence[GoldQuery], query_id: str) -> GoldQuery | None:
        for q in queries:
            if q.query_id == query_id:
                return q
        return None

    def compare_retrieval_configurations(
        self,
        dataset: Sequence[GoldQuery] | Any,
        configurations: Mapping[str, Callable[[GoldQuery], Sequence[Any]]],
        pipeline_configs: Mapping[str, PipelineConfig] | None = None,
        dataset_id: str = "gold_benchmark",
        dataset_version: str = "1.0.0",
    ) -> dict[str, ExperimentResult]:
        """Run and compare multiple retrieval pipeline configurations on the same dataset."""
        results: dict[str, ExperimentResult] = {}
        pipeline_configs = pipeline_configs or {}

        for config_name, ret_fn in configurations.items():
            p_config = pipeline_configs.get(
                config_name,
                PipelineConfig(name=config_name),
            )
            exp_res = self.evaluate_queries(
                dataset=dataset,
                retrieval_fn=ret_fn,
                pipeline_config=p_config,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                experiment_name=f"Evaluation - {config_name}",
            )
            results[config_name] = exp_res

        return results


class GenerationEvaluator:
    """Evaluates answer generation and grounding quality independently from retrieval quality."""

    def __init__(
        self,
        thresholds: EvaluationThresholds | None = None,
        judge: BaseGenerationJudge | None = None,
    ) -> None:
        self.thresholds = thresholds or EvaluationThresholds()
        self.judge = judge or DeterministicGenerationJudge()

    def evaluate_query(
        self,
        query: GoldQuery,
        answer: str,
        evidence: Sequence[Any],
        citations: Sequence[Citation] | None = None,
        operational_metrics: OperationalMetrics | None = None,
        retrieval_metrics: RetrievalEvalMetrics | None = None,
    ) -> QueryEvaluationResult:
        """Evaluate generation quality for a single query."""
        gen_metrics = self.judge.evaluate(
            query=query,
            answer=answer,
            evidence=evidence,
            citations=citations,
        )

        failures = classify_generation_failures(
            query=query,
            answer=answer,
            gen_metrics=gen_metrics,
            ret_metrics=retrieval_metrics,
            thresholds=self.thresholds,
        )

        # Evaluate threshold criteria
        passed = True
        if not query.is_answerable:
            passed = gen_metrics.abstention_correct is True
        elif is_refusal_response(answer):
            passed = False
        else:
            if (
                gen_metrics.faithfulness is not None
                and gen_metrics.faithfulness < self.thresholds.min_faithfulness
            ):
                passed = False
            if (
                gen_metrics.answer_relevancy is not None
                and gen_metrics.answer_relevancy < self.thresholds.min_answer_relevancy
            ):
                passed = False
            if (
                gen_metrics.citation_precision is not None
                and gen_metrics.citation_precision < self.thresholds.min_citation_precision
            ):
                passed = False

        note_str = f"Failures: {', '.join(f.value for f in failures)}" if failures else ""

        return QueryEvaluationResult(
            query_id=query.query_id,
            document_id=query.document_id,
            category=query.category,
            question=query.question,
            is_answerable=query.is_answerable,
            generated_answer=answer,
            citations=list(citations or []),
            is_grounded=(
                gen_metrics.faithfulness is not None
                and gen_metrics.faithfulness >= self.thresholds.min_faithfulness
            ),
            retrieval_metrics=retrieval_metrics or RetrievalEvalMetrics(),
            generation_metrics=gen_metrics,
            operational_metrics=operational_metrics or OperationalMetrics(),
            failure_categories=failures,
            passed_thresholds=passed,
            notes=note_str,
        )

    def evaluate_queries(
        self,
        dataset: Sequence[GoldQuery] | Any,
        generation_fn: (
            Callable[
                [GoldQuery, Sequence[Any]],
                GenerationResult | tuple[str, list[Citation]],
            ]
            | None
        ) = None,
        precomputed_generations: (
            Mapping[str, tuple[str, Sequence[Any], Sequence[Citation]]] | None
        ) = None,
        default_evidence: Sequence[Any] | None = None,
        pipeline_config: PipelineConfig | None = None,
        dataset_id: str = "gold_benchmark",
        dataset_version: str = "1.0.0",
        experiment_id: str | None = None,
        experiment_name: str | None = None,
    ) -> ExperimentResult:
        """Run generation evaluation across a gold dataset."""
        queries: list[GoldQuery] = getattr(dataset, "queries", dataset)
        if not isinstance(queries, list):
            queries = list(queries)

        query_results: list[QueryEvaluationResult] = []
        failure_counts: dict[str, int] = {}
        latencies: list[float] = []

        for query in queries:
            t0 = time.perf_counter()
            if precomputed_generations and query.query_id in precomputed_generations:
                ans_text, ev_items, cits = precomputed_generations[query.query_id]
                op_met = OperationalMetrics()
            elif generation_fn:
                ev_items = default_evidence or [
                    {
                        "document_id": query.document_id,
                        "page_number": p,
                        "text": query.ground_truth_evidence_text,
                    }
                    for p in (query.ground_truth_pages or [1])
                ]
                gen_output = generation_fn(query, ev_items)
                dur_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(dur_ms)
                if isinstance(gen_output, GenerationResult):
                    ans_text = gen_output.answer
                    cits = gen_output.citations
                    op_met = OperationalMetrics(
                        generation_latency_ms=dur_ms,
                        total_latency_ms=dur_ms,
                    )
                else:
                    ans_text, cits = gen_output
                    op_met = OperationalMetrics(
                        generation_latency_ms=dur_ms,
                        total_latency_ms=dur_ms,
                    )
            else:
                ans_text = ""
                ev_items = []
                cits = []
                op_met = OperationalMetrics()

            result = self.evaluate_query(
                query=query,
                answer=ans_text,
                evidence=ev_items,
                citations=cits,
                operational_metrics=op_met,
            )
            query_results.append(result)

            for failure in result.failure_categories:
                failure_counts[failure.value] = failure_counts.get(failure.value, 0) + 1

        total_queries = len(queries)
        answerable_results = [r for r in query_results if r.is_answerable]
        unanswerable_results = [r for r in query_results if not r.is_answerable]

        def _mean_metric(vals: list[float | None]) -> float | None:
            clean = [v for v in vals if v is not None]
            return round(sum(clean) / len(clean), 6) if clean else None

        mean_faith = _mean_metric([r.generation_metrics.faithfulness for r in answerable_results])
        mean_rel = _mean_metric([r.generation_metrics.answer_relevancy for r in query_results])
        mean_cit_prec = _mean_metric(
            [r.generation_metrics.citation_precision for r in answerable_results]
        )
        mean_cit_rec = _mean_metric(
            [r.generation_metrics.citation_recall for r in answerable_results]
        )

        abstention_vals = [
            1.0 if r.generation_metrics.abstention_correct else 0.0 for r in query_results
        ]
        abstention_acc = _mean_metric(abstention_vals)

        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_lat = sorted(latencies)
        p95_lat = (
            sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)]
            if sorted_lat
            else 0.0
        )

        category_metrics: dict[str, dict[str, float]] = {}
        for cat in QuestionCategory:
            cat_results = [r for r in query_results if r.category == cat]
            if cat_results:
                c_len = len(cat_results)
                c_faith = _mean_metric([r.generation_metrics.faithfulness for r in cat_results])
                c_rel = _mean_metric([r.generation_metrics.answer_relevancy for r in cat_results])
                category_metrics[cat.value] = {
                    "count": float(c_len),
                    "faithfulness": c_faith if c_faith is not None else 0.0,
                    "answer_relevancy": c_rel if c_rel is not None else 0.0,
                }

        thresholds_passed = (
            (mean_faith is None or mean_faith >= self.thresholds.min_faithfulness)
            and (mean_rel is None or mean_rel >= self.thresholds.min_answer_relevancy)
            and (mean_cit_prec is None or mean_cit_prec >= self.thresholds.min_citation_precision)
            and (abstention_acc is None or abstention_acc >= self.thresholds.min_abstention_accuracy)
        )

        summary = ExperimentSummary(
            total_queries=total_queries,
            mean_faithfulness=mean_faith,
            mean_answer_relevancy=mean_rel,
            mean_citation_precision=mean_cit_prec,
            mean_citation_recall=mean_cit_rec,
            abstention_accuracy=abstention_acc,
            mean_total_latency_ms=round(mean_lat, 2),
            p95_total_latency_ms=round(p95_lat, 2),
            category_metrics=category_metrics,
            failure_counts=failure_counts,
            thresholds=self.thresholds,
            thresholds_passed=thresholds_passed,
        )

        config = pipeline_config or PipelineConfig(name="default_generation")
        exp_id = experiment_id or f"exp-gen-{uuid.uuid4().hex[:8]}"
        exp_name = experiment_name or f"Generation Evaluation ({config.name})"

        return ExperimentResult(
            experiment_id=exp_id,
            name=exp_name,
            pipeline_config=config,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            summary=summary,
            query_results=query_results,
        )


class RAGEvaluator:
    """Unified end-to-end evaluator combining retrieval, reranking, and generation boundaries."""

    def __init__(
        self,
        thresholds: EvaluationThresholds | None = None,
        generation_judge: BaseGenerationJudge | None = None,
        k_values: Sequence[int] = (1, 3, 5, 10),
        default_k: int = 5,
    ) -> None:
        self.thresholds = thresholds or EvaluationThresholds()
        self.retrieval_evaluator = RetrievalEvaluator(
            k_values=k_values, default_k=default_k, thresholds=self.thresholds
        )
        self.generation_evaluator = GenerationEvaluator(
            thresholds=self.thresholds, judge=generation_judge
        )

    def evaluate_query(
        self,
        query: GoldQuery,
        candidates: Sequence[Any],
        answer: str,
        citations: Sequence[Citation] | None = None,
        operational_metrics: OperationalMetrics | None = None,
    ) -> QueryEvaluationResult:
        """Evaluate both retrieval and generation for a single query."""
        ret_result = self.retrieval_evaluator.evaluate_query(
            query=query,
            candidates=candidates,
            operational_metrics=operational_metrics,
        )

        gen_result = self.generation_evaluator.evaluate_query(
            query=query,
            answer=answer,
            evidence=candidates,
            citations=citations,
            operational_metrics=operational_metrics,
            retrieval_metrics=ret_result.retrieval_metrics,
        )

        combined_failures = list(
            dict.fromkeys(ret_result.failure_categories + gen_result.failure_categories)
        )
        passed = ret_result.passed_thresholds and gen_result.passed_thresholds
        note_str = (
            f"Failures: {', '.join(f.value for f in combined_failures)}"
            if combined_failures
            else ""
        )

        return QueryEvaluationResult(
            query_id=query.query_id,
            document_id=query.document_id,
            category=query.category,
            question=query.question,
            is_answerable=query.is_answerable,
            generated_answer=answer,
            citations=gen_result.citations,
            is_grounded=gen_result.is_grounded,
            retrieval_metrics=ret_result.retrieval_metrics,
            generation_metrics=gen_result.generation_metrics,
            operational_metrics=operational_metrics or OperationalMetrics(),
            failure_categories=combined_failures,
            passed_thresholds=passed,
            notes=note_str,
        )


retrieval_evaluator = RetrievalEvaluator()
generation_evaluator = GenerationEvaluator()
rag_evaluator = RAGEvaluator()
