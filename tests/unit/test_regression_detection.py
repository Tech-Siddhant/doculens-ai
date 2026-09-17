"""Unit tests for regression detection and baseline comparison — Phase 8.7."""

import pytest

from app.evaluation.regression import RegressionThresholds, compare_against_baseline
from app.schemas.evaluation import (
    EvaluationThresholds,
    ExperimentResult,
    ExperimentSummary,
    PipelineConfig,
    RegressionReport,
)


def _make_summary(**kwargs) -> ExperimentSummary:
    defaults = {
        "total_queries": 5,
        "mean_recall_at_k": 0.85,
        "mean_mrr_at_k": 0.75,
        "mean_context_precision": 0.80,
        "mean_context_recall": 0.80,
        "mean_faithfulness": 0.90,
        "mean_answer_relevancy": 0.85,
        "mean_citation_precision": 0.95,
        "mean_citation_recall": 0.85,
        "abstention_accuracy": 0.90,
        "p95_total_latency_ms": 100.0,
        "thresholds_passed": True,
    }
    defaults.update(kwargs)
    return ExperimentSummary(**defaults)


def test_identical_summaries_pass_regression_check():
    base = _make_summary()
    curr = _make_summary()

    report = compare_against_baseline(current=curr, baseline=base)

    assert report.passed is True
    assert report.status == "PASSED"
    assert len(report.degradations) == 0
    assert len(report.improvements) == 0
    assert report.total_metrics_evaluated > 0


def test_quality_improvement_detected():
    base = _make_summary(mean_recall_at_k=0.70, mean_mrr_at_k=0.60)
    curr = _make_summary(mean_recall_at_k=0.85, mean_mrr_at_k=0.75)

    report = compare_against_baseline(current=curr, baseline=base)

    assert report.passed is True
    assert report.status == "IMPROVED"
    assert len(report.degradations) == 0
    assert len(report.improvements) >= 2
    assert "mean_recall_at_k" in report.metric_deltas
    assert report.metric_deltas["mean_recall_at_k"].is_improvement is True


def test_recall_regression_breaching_threshold():
    base = _make_summary(mean_recall_at_k=0.90)
    curr = _make_summary(mean_recall_at_k=0.80)  # 0.10 drop > max 0.05

    report = compare_against_baseline(
        current=curr, baseline=base, thresholds=RegressionThresholds(max_recall_drop=0.05)
    )

    assert report.passed is False
    assert report.status == "REGRESSION_DETECTED"
    assert len(report.degradations) == 1
    assert "mean_recall_at_k" in report.degradations[0]
    assert report.metric_deltas["mean_recall_at_k"].is_regression is True


def test_minor_drop_within_tolerance_passes():
    base = _make_summary(mean_recall_at_k=0.90)
    curr = _make_summary(mean_recall_at_k=0.88)  # 0.02 drop < max 0.05

    report = compare_against_baseline(
        current=curr, baseline=base, thresholds=RegressionThresholds(max_recall_drop=0.05)
    )

    assert report.passed is True
    assert report.status == "PASSED"
    assert len(report.degradations) == 0


def test_latency_regression_exceeding_ratio():
    base = _make_summary(p95_total_latency_ms=100.0)
    curr = _make_summary(p95_total_latency_ms=150.0)  # +50% > allowed +25%

    report = compare_against_baseline(
        current=curr, baseline=base, thresholds=RegressionThresholds(max_latency_increase_ratio=0.25)
    )

    assert report.passed is False
    assert report.status == "REGRESSION_DETECTED"
    assert any("p95_total_latency_ms" in d for d in report.degradations)
    assert report.metric_deltas["p95_total_latency_ms"].is_regression is True


def test_latency_speedup_recorded_as_improvement():
    base = _make_summary(p95_total_latency_ms=200.0)
    curr = _make_summary(p95_total_latency_ms=100.0)  # -50% speedup

    report = compare_against_baseline(
        current=curr, baseline=base, thresholds=RegressionThresholds(max_latency_increase_ratio=0.25)
    )

    assert report.passed is True
    assert report.status == "IMPROVED"
    assert report.metric_deltas["p95_total_latency_ms"].is_improvement is True


def test_generation_metrics_regression():
    base = _make_summary(mean_faithfulness=0.95, mean_answer_relevancy=0.90)
    curr = _make_summary(mean_faithfulness=0.80, mean_answer_relevancy=0.75)

    report = compare_against_baseline(current=curr, baseline=base)

    assert report.passed is False
    assert report.status == "REGRESSION_DETECTED"
    assert any("mean_faithfulness" in d for d in report.degradations)
    assert any("mean_answer_relevancy" in d for d in report.degradations)


def test_experiment_result_wrapper_comparison():
    config = PipelineConfig(name="dense_bm25")
    base_res = ExperimentResult(
        experiment_id="exp-baseline-001",
        name="Baseline",
        pipeline_config=config,
        dataset_id="gold-v1",
        dataset_version="1.0",
        summary=_make_summary(mean_recall_at_k=0.85),
    )
    curr_res = ExperimentResult(
        experiment_id="exp-curr-002",
        name="Current",
        pipeline_config=config,
        dataset_id="gold-v1",
        dataset_version="1.0",
        summary=_make_summary(mean_recall_at_k=0.90),
    )

    report = compare_against_baseline(current=curr_res, baseline=base_res)

    assert report.baseline_id == "exp-baseline-001"
    assert report.current_id == "exp-curr-002"
    assert report.passed is True
    assert report.status == "IMPROVED"

    # Verify serialization
    json_str = report.model_dump_json()
    reloaded = RegressionReport.model_validate_json(json_str)
    assert reloaded.status == report.status
    assert reloaded.passed == report.passed
