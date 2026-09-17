"""
Unit tests for app/services/dataset.py — Phase 6.2 gold dataset loader and validator.
Tests use in-memory tmp files instead of the real data/gold_dataset.jsonl
so they are hermetic and portable.
"""

import json
from pathlib import Path

import pytest

from app.schemas.evaluation import ExpectedModality, QuestionCategory
from app.services.dataset import GoldBenchmark, ValidationIssue, load_gold_dataset


# ──────────────────────────── helpers ────────────────────────────────────────

def _write_jsonl(tmp_path: Path, lines: list) -> Path:
    """Write a list of objects (or raw strings) to a JSONL file."""
    p = tmp_path / "dataset.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write((line if isinstance(line, str) else json.dumps(line)) + "\n")
    return p


def _minimal_query(**overrides) -> dict:
    base = {
        "query_id": "q-001",
        "document_id": "doc-1",
        "question": "What is X?",
        "category": "factoid_text",
        "expected_sources": ["text"],
        "ground_truth_pages": [1],
        "ground_truth_answer": "X is Y.",
        "is_answerable": True,
    }
    base.update(overrides)
    return base


# ──────────────────────────── basic loading ──────────────────────────────────

def test_load_empty_file(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [])
    bm = load_gold_dataset(p)
    assert bm.total == 0
    assert bm.is_valid


def test_load_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [
        "# This is a comment",
        "",
        json.dumps(_minimal_query()),
    ])
    bm = load_gold_dataset(p)
    assert bm.total == 1
    assert bm.is_valid


def test_load_single_valid_query(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_query()])
    bm = load_gold_dataset(p)
    assert bm.total == 1
    assert bm.is_valid
    assert bm.queries[0].query_id == "q-001"


def test_load_multiple_valid_queries(tmp_path: Path) -> None:
    rows = [
        _minimal_query(query_id="q-003"),
        _minimal_query(query_id="q-001"),
        _minimal_query(query_id="q-002"),
    ]
    p = _write_jsonl(tmp_path, rows)
    bm = load_gold_dataset(p)
    assert bm.total == 3
    # deterministic sort by query_id
    assert [q.query_id for q in bm.queries] == ["q-001", "q-002", "q-003"]


def test_source_path_recorded(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [_minimal_query()])
    bm = load_gold_dataset(p)
    assert bm.source_path == p


# ──────────────────────────── json / schema errors ───────────────────────────

def test_invalid_json_line_records_issue_and_skips(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [
        _minimal_query(query_id="q-001"),
        "this is not json {{{}",
        _minimal_query(query_id="q-002"),
    ])
    bm = load_gold_dataset(p)
    assert bm.total == 2
    assert len(bm.issues) == 1
    assert "JSON parse error" in bm.issues[0].message
    assert bm.issues[0].line == 2


def test_schema_validation_error_records_issue_and_skips(tmp_path: Path) -> None:
    bad = _minimal_query()
    bad["query_id"] = ""  # min_length=1 violation
    p = _write_jsonl(tmp_path, [bad, _minimal_query(query_id="q-002")])
    bm = load_gold_dataset(p)
    assert bm.total == 1
    assert len(bm.issues) == 1
    assert "Schema validation error" in bm.issues[0].message


def test_missing_required_field_records_issue(tmp_path: Path) -> None:
    bad = {"document_id": "doc-1", "question": "Q?", "category": "factoid_text"}
    p = _write_jsonl(tmp_path, [bad])
    bm = load_gold_dataset(p)
    assert bm.total == 0
    assert len(bm.issues) == 1


# ──────────────────────────── duplicate ID ───────────────────────────────────

def test_duplicate_query_id_records_issue_keeps_first(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [
        _minimal_query(query_id="q-001", question="First version"),
        _minimal_query(query_id="q-001", question="Duplicate version"),
    ])
    bm = load_gold_dataset(p)
    assert bm.total == 1
    assert bm.queries[0].question == "First version"
    assert len(bm.issues) == 1
    issue = bm.issues[0]
    assert "Duplicate" in issue.message
    assert issue.query_id == "q-001"


# ──────────────────────────── consistency checks ─────────────────────────────

def test_visual_query_without_pages_records_consistency_issue(tmp_path: Path) -> None:
    row = _minimal_query(
        query_id="q-001",
        expected_sources=["visual"],
        ground_truth_pages=[],   # missing pages for visual query
    )
    p = _write_jsonl(tmp_path, [row])
    bm = load_gold_dataset(p)
    # query still loads — consistency issue is non-fatal
    assert bm.total == 1
    assert len(bm.issues) == 1
    assert "ground_truth_pages" in bm.issues[0].message


def test_figure_query_without_pages_records_consistency_issue(tmp_path: Path) -> None:
    row = _minimal_query(
        query_id="q-001",
        expected_sources=["figure"],
        ground_truth_pages=[],
    )
    p = _write_jsonl(tmp_path, [row])
    bm = load_gold_dataset(p)
    assert bm.total == 1
    assert len(bm.issues) == 1


def test_table_query_without_pages_records_consistency_issue(tmp_path: Path) -> None:
    row = _minimal_query(
        query_id="q-001",
        expected_sources=["table"],
        ground_truth_pages=[],
    )
    p = _write_jsonl(tmp_path, [row])
    bm = load_gold_dataset(p)
    assert bm.total == 1
    assert len(bm.issues) == 1


def test_answerable_query_without_answer_records_consistency_issue(tmp_path: Path) -> None:
    row = _minimal_query(
        query_id="q-001",
        is_answerable=True,
        ground_truth_answer="",   # missing answer
    )
    p = _write_jsonl(tmp_path, [row])
    bm = load_gold_dataset(p)
    assert bm.total == 1
    assert len(bm.issues) == 1
    assert "ground_truth_answer" in bm.issues[0].message


def test_unanswerable_query_empty_answer_is_valid(tmp_path: Path) -> None:
    row = _minimal_query(
        query_id="q-001",
        is_answerable=False,
        ground_truth_answer="",
    )
    p = _write_jsonl(tmp_path, [row])
    bm = load_gold_dataset(p)
    assert bm.total == 1
    assert bm.is_valid


def test_text_query_without_pages_is_valid(tmp_path: Path) -> None:
    """Text-only queries don't require ground_truth_pages."""
    row = _minimal_query(
        query_id="q-001",
        expected_sources=["text"],
        ground_truth_pages=[],
    )
    p = _write_jsonl(tmp_path, [row])
    bm = load_gold_dataset(p)
    assert bm.total == 1
    assert bm.is_valid


# ──────────────────────────── GoldBenchmark helpers ──────────────────────────

def test_by_category(tmp_path: Path) -> None:
    rows = [
        _minimal_query(query_id="q-001", category="factoid_text"),
        _minimal_query(query_id="q-002", category="table_lookup"),
        _minimal_query(query_id="q-003", category="factoid_text"),
    ]
    p = _write_jsonl(tmp_path, rows)
    bm = load_gold_dataset(p)
    factoids = bm.by_category(QuestionCategory.FACTOID_TEXT)
    assert len(factoids) == 2
    tables = bm.by_category(QuestionCategory.TABLE_LOOKUP)
    assert len(tables) == 1


def test_answerable_unanswerable_split(tmp_path: Path) -> None:
    rows = [
        _minimal_query(query_id="q-001", is_answerable=True),
        _minimal_query(
            query_id="q-002",
            is_answerable=False,
            ground_truth_answer="",
            category="negative_unanswerable",
        ),
    ]
    p = _write_jsonl(tmp_path, rows)
    bm = load_gold_dataset(p)
    assert len(bm.answerable()) == 1
    assert len(bm.unanswerable()) == 1


# ──────────────────────────── real seed file smoke test ──────────────────────

def test_real_seed_file_loads_cleanly() -> None:
    """Smoke test: the committed seed file must load without structural errors."""
    root = Path(__file__).resolve().parent.parent.parent
    seed = root / "data/gold_dataset.jsonl"
    if not seed.exists():
        pytest.skip("data/gold_dataset.jsonl not present in this environment")
    bm = load_gold_dataset(seed)
    # Structural errors (JSON parse / schema / duplicate ID) must be zero.
    structural_issues = [
        i for i in bm.issues
        if any(kw in i.message for kw in ("JSON parse error", "Schema validation error", "Duplicate"))
    ]
    assert structural_issues == [], structural_issues
    assert bm.total >= 1
