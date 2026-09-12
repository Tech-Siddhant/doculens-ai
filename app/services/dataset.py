"""
Gold benchmark dataset loader and validator for Phase 6.2.

Responsibilities:
- Load and parse JSONL files into GoldQuery objects (one per line).
- Validate structural integrity: duplicate IDs, missing required fields, page bounds.
- Expose a thin GoldBenchmark container for downstream harness use.
- No heavy dependencies — stdlib + Pydantic only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from app.schemas.evaluation import GoldQuery, QuestionCategory


@dataclass
class ValidationIssue:
    line: int  # 1-based
    query_id: str | None
    message: str


@dataclass
class GoldBenchmark:
    """Loaded, validated gold benchmark dataset."""

    queries: list[GoldQuery] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    source_path: Path | None = None

    # ------------------------------------------------------------------ counts

    @property
    def total(self) -> int:
        return len(self.queries)

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    def by_category(self, category: QuestionCategory) -> list[GoldQuery]:
        return [q for q in self.queries if q.category == category]

    def answerable(self) -> list[GoldQuery]:
        return [q for q in self.queries if q.is_answerable]

    def unanswerable(self) -> list[GoldQuery]:
        return [q for q in self.queries if not q.is_answerable]


def load_gold_dataset(path: str | Path) -> GoldBenchmark:
    """
    Load and validate a JSONL gold benchmark file.

    Each non-blank, non-comment line must be a valid JSON object that
    can be parsed into a GoldQuery.  Lines starting with '#' are treated
    as comments and skipped.

    Validation rules applied:
    1. JSON parse error  → issue recorded, line skipped.
    2. Pydantic schema error  → issue recorded, line skipped.
    3. Duplicate query_id  → issue recorded for the second occurrence.
    4. Visually-dependent query without ground_truth_pages  → issue recorded.
    5. is_answerable=True but no ground_truth_answer  → issue recorded.
    6. Queries sorted deterministically by query_id before return.
    """
    path = Path(path)
    benchmark = GoldBenchmark(source_path=path)
    seen_ids: dict[str, int] = {}  # query_id -> first line number

    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # --- JSON parse ---
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                benchmark.issues.append(
                    ValidationIssue(line=lineno, query_id=None, message=f"JSON parse error: {exc}")
                )
                continue

            # --- Schema validation ---
            try:
                query = GoldQuery(**obj)
            except ValidationError as exc:
                qid = obj.get("query_id")
                benchmark.issues.append(
                    ValidationIssue(
                        line=lineno, query_id=qid, message=f"Schema validation error: {exc}"
                    )
                )
                continue

            # --- Duplicate ID ---
            if query.query_id in seen_ids:
                benchmark.issues.append(
                    ValidationIssue(
                        line=lineno,
                        query_id=query.query_id,
                        message=(
                            f"Duplicate query_id '{query.query_id}' "
                            f"(first seen on line {seen_ids[query.query_id]})"
                        ),
                    )
                )
                continue  # discard duplicate; keep first occurrence
            seen_ids[query.query_id] = lineno

            # --- Content consistency warnings ---
            _check_consistency(benchmark, query, lineno)

            benchmark.queries.append(query)

    # Deterministic ordering by query_id
    benchmark.queries.sort(key=lambda q: q.query_id)
    return benchmark


def _check_consistency(benchmark: GoldBenchmark, query: GoldQuery, lineno: int) -> None:
    """Record content-level consistency issues (does not discard the query)."""
    from app.schemas.evaluation import ExpectedModality

    # Visually-dependent questions must reference at least one page
    visual_modalities = {ExpectedModality.VISUAL, ExpectedModality.FIGURE, ExpectedModality.TABLE}
    needs_pages = any(src in visual_modalities for src in query.expected_sources)
    if needs_pages and not query.ground_truth_pages:
        benchmark.issues.append(
            ValidationIssue(
                line=lineno,
                query_id=query.query_id,
                message=(
                    f"Query '{query.query_id}' expects visual/table/figure evidence "
                    "but has no ground_truth_pages."
                ),
            )
        )

    # Answerable queries should have a reference answer
    if query.is_answerable and not query.ground_truth_answer.strip():
        benchmark.issues.append(
            ValidationIssue(
                line=lineno,
                query_id=query.query_id,
                message=(
                    f"Query '{query.query_id}' is marked is_answerable=True "
                    "but ground_truth_answer is empty."
                ),
            )
        )
