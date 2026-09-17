"""Regression detection engine for DocuLens AI — Phase 8.7.

Compares current evaluation experiment metrics against a designated baseline run
to detect statistically meaningful degradations in retrieval and generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas.evaluation import (
    ExperimentResult,
    ExperimentSummary,
    MetricDelta,
    RegressionReport,
)


@dataclass
class RegressionThresholds:
    """Configurable thresholds for detecting performance regressions."""

    max_recall_drop: float = 0.05
    max_mrr_drop: float = 0.05
    max_context_precision_drop: float = 0.05
    max_context_recall_drop: float = 0.05
    max_faithfulness_drop: float = 0.05
    max_answer_relevancy_drop: float = 0.05
    max_citation_precision_drop: float = 0.05
    max_citation_recall_drop: float = 0.05
    max_abstention_accuracy_drop: float = 0.05
    max_latency_increase_ratio: float = 0.25  # Max 25% latency increase


def compare_against_baseline(
    current: ExperimentSummary | ExperimentResult,
    baseline: ExperimentSummary | ExperimentResult,
    thresholds: RegressionThresholds | None = None,
    current_id: str | None = None,
    baseline_id: str | None = None,
) -> RegressionReport:
    """Compare a current evaluation against a baseline and flag regressions."""
    thresh = thresholds or RegressionThresholds()

    curr_summary = current.summary if isinstance(current, ExperimentResult) else current
    base_summary = baseline.summary if isinstance(baseline, ExperimentResult) else baseline

    c_id = current_id or (
        current.experiment_id if isinstance(current, ExperimentResult) else "current"
    )
    b_id = baseline_id or (
        baseline.experiment_id if isinstance(baseline, ExperimentResult) else "baseline"
    )

    quality_checks: list[tuple[str, float | None, float | None, float]] = [
        ("mean_recall_at_k", curr_summary.mean_recall_at_k, base_summary.mean_recall_at_k, thresh.max_recall_drop),
        ("mean_mrr_at_k", curr_summary.mean_mrr_at_k, base_summary.mean_mrr_at_k, thresh.max_mrr_drop),
        ("mean_context_precision", curr_summary.mean_context_precision, base_summary.mean_context_precision, thresh.max_context_precision_drop),
        ("mean_context_recall", curr_summary.mean_context_recall, base_summary.mean_context_recall, thresh.max_context_recall_drop),
        ("mean_faithfulness", curr_summary.mean_faithfulness, base_summary.mean_faithfulness, thresh.max_faithfulness_drop),
        ("mean_answer_relevancy", curr_summary.mean_answer_relevancy, base_summary.mean_answer_relevancy, thresh.max_answer_relevancy_drop),
        ("mean_citation_precision", curr_summary.mean_citation_precision, base_summary.mean_citation_precision, thresh.max_citation_precision_drop),
        ("mean_citation_recall", curr_summary.mean_citation_recall, base_summary.mean_citation_recall, thresh.max_citation_recall_drop),
        ("abstention_accuracy", curr_summary.abstention_accuracy, base_summary.abstention_accuracy, thresh.max_abstention_accuracy_drop),
    ]

    deltas: dict[str, MetricDelta] = {}
    degradations: list[str] = []
    improvements: list[str] = []

    for metric_name, c_val, b_val, max_drop in quality_checks:
        if c_val is None or b_val is None:
            continue

        diff = round(c_val - b_val, 6)
        rel_pct = round((diff / b_val) * 100, 2) if b_val > 0 else (0.0 if diff == 0.0 else 100.0)
        is_reg = diff < -max_drop
        is_imp = diff > 0.0001

        deltas[metric_name] = MetricDelta(
            metric_name=metric_name,
            baseline_value=b_val,
            current_value=c_val,
            delta=diff,
            relative_change_pct=rel_pct,
            is_regression=is_reg,
            is_improvement=is_imp,
        )

        if is_reg:
            degradations.append(
                f"{metric_name}: dropped by {abs(diff):.4f} (baseline={b_val:.4f}, current={c_val:.4f}, allowed={max_drop:.4f})"
            )
        elif is_imp:
            improvements.append(
                f"{metric_name}: improved by +{diff:.4f} (baseline={b_val:.4f}, current={c_val:.4f})"
            )

    # Operational latency check (lower is better)
    if curr_summary.p95_total_latency_ms > 0 and base_summary.p95_total_latency_ms > 0:
        c_lat = curr_summary.p95_total_latency_ms
        b_lat = base_summary.p95_total_latency_ms
        lat_diff = round(c_lat - b_lat, 2)
        lat_rel = round((lat_diff / b_lat) * 100, 2)
        lat_reg = lat_rel > (thresh.max_latency_increase_ratio * 100)
        lat_imp = lat_rel < -(thresh.max_latency_increase_ratio * 100)

        deltas["p95_total_latency_ms"] = MetricDelta(
            metric_name="p95_total_latency_ms",
            baseline_value=b_lat,
            current_value=c_lat,
            delta=lat_diff,
            relative_change_pct=lat_rel,
            is_regression=lat_reg,
            is_improvement=lat_imp,
        )
        if lat_reg:
            degradations.append(
                f"p95_total_latency_ms: increased by {lat_rel:.1f}% ({b_lat:.1f}ms -> {c_lat:.1f}ms)"
            )
        elif lat_imp:
            improvements.append(
                f"p95_total_latency_ms: speedup of {abs(lat_rel):.1f}% ({b_lat:.1f}ms -> {c_lat:.1f}ms)"
            )

    passed = len(degradations) == 0
    status = "REGRESSION_DETECTED" if not passed else ("IMPROVED" if improvements else "PASSED")

    return RegressionReport(
        status=status,
        passed=passed,
        baseline_id=b_id,
        current_id=c_id,
        total_metrics_evaluated=len(deltas),
        degradations=degradations,
        improvements=improvements,
        metric_deltas=deltas,
    )
