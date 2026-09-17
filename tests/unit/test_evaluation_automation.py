"""Unit tests for evaluation harness, store lifecycle, and CLI automation — Phase 8.7."""

from pathlib import Path
import pytest

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
from app.schemas.evaluation import ExperimentResult, GoldQuery, QuestionCategory
from app.services.dataset import load_gold_dataset
from scripts.run_evaluation import format_markdown_report, format_summary_table


def test_evaluation_stores_lifecycle_identity():
    """Verify that stores and retrievers share the EXACT same vector store and BM25 instances."""
    stores = create_isolated_stores()

    # Verify object identity (same memory address)
    assert stores.dense_retriever.vector_store is stores.vector_store
    assert stores.hybrid_retriever.dense_retriever.vector_store is stores.vector_store
    assert stores.hybrid_retriever.bm25_retriever is stores.bm25_retriever
    assert stores.hybrid_retriever.visual_store is stores.visual_vector_store


def test_create_isolated_stores_independence():
    """Verify isolated stores do not mutate or leak data into shared singletons."""
    shared = get_shared_stores()
    isolated = create_isolated_stores()

    assert isolated.vector_store is not shared.vector_store
    assert isolated.bm25_retriever is not shared.bm25_retriever


def test_index_evaluation_document_and_retrieval(tmp_path: Path):
    """Verify indexing an evaluation PDF document populates both dense and BM25 stores."""
    stores = create_isolated_stores()
    pdf_path = Path("data/samples/doculens-architecture-v1.pdf")
    if not pdf_path.exists():
        pdf_path = Path(__file__).resolve().parent.parent.parent / "data/samples/doculens-architecture-v1.pdf"
    if not pdf_path.exists():
        pytest.skip("doculens-architecture-v1.pdf not found")

    stats = index_evaluation_document(
        stores=stores,
        document_id="doculens-architecture-v1",
        pdf_path=pdf_path,
    )

    assert stats["dense_chunks"] >= 2
    assert stats["bm25_chunks"] >= 2
    assert stores.vector_store.count_chunks("doculens-architecture-v1") >= 2
    assert stores.bm25_retriever.count_chunks("doculens-architecture-v1") >= 2


def test_baseline_configurations_definitions():
    stores = create_isolated_stores()
    for name in BASELINE_NAMES:
        config, ret_fn = get_retrieval_function_for_baseline(stores, name, top_k=5)
        assert config.name == name
        assert callable(ret_fn)
        if name == "dense":
            assert config.retrieval_types == ["dense"]
            assert config.reranker_enabled is False
        elif name == "dense_bm25":
            assert "bm25" in config.retrieval_types
            assert config.reranker_enabled is False
        elif name == "hybrid_reranked":
            assert config.reranker_enabled is True


def test_unknown_baseline_name_raises():
    stores = create_isolated_stores()
    with pytest.raises(ValueError, match="Unknown baseline name"):
        get_retrieval_function_for_baseline(stores, "non_existent_baseline")


def test_run_all_baselines_execution():
    stores = create_isolated_stores()
    pdf_path = Path("data/samples/doculens-architecture-v1.pdf")
    if not pdf_path.exists():
        pdf_path = Path(__file__).resolve().parent.parent.parent / "data/samples/doculens-architecture-v1.pdf"
    if not pdf_path.exists():
        pytest.skip("doculens-architecture-v1.pdf not found")

    index_evaluation_document(stores, "doculens-architecture-v1", pdf_path)

    seed_path = Path("data/gold_dataset.jsonl")
    if not seed_path.exists():
        seed_path = Path(__file__).resolve().parent.parent.parent / "data/gold_dataset.jsonl"
    if not seed_path.exists():
        pytest.skip("gold_dataset.jsonl not found")

    dataset = load_gold_dataset(seed_path)
    all_results = run_all_baselines(stores=stores, dataset=dataset, default_k=5)

    assert len(all_results) == 4
    for name in BASELINE_NAMES:
        assert name in all_results
        res = all_results[name]
        assert isinstance(res, ExperimentResult)
        assert res.summary.total_queries == dataset.total
        assert res.summary.mean_recall_at_k > 0.0
        assert res.summary.mean_mrr_at_k > 0.0
        assert len(res.query_results) == dataset.total


def test_table_and_markdown_formatters():
    stores = create_isolated_stores()
    pdf_path = Path("data/samples/doculens-architecture-v1.pdf")
    if not pdf_path.exists():
        pdf_path = Path(__file__).resolve().parent.parent.parent / "data/samples/doculens-architecture-v1.pdf"
    if not pdf_path.exists():
        pytest.skip("doculens-architecture-v1.pdf not found")

    index_evaluation_document(stores, "doculens-architecture-v1", pdf_path)

    q = GoldQuery(
        query_id="q-001",
        document_id="doculens-architecture-v1",
        question="What are the architectural layers?",
        category=QuestionCategory.FACTOID_TEXT,
        expected_sources=["text"],
        ground_truth_pages=[1],
        ground_truth_answer="Five layers",
        is_answerable=True,
    )
    results = run_all_baselines(stores=stores, dataset=[q], default_k=5)

    table_str = format_summary_table(results)
    assert "CONFIGURATION" in table_str
    assert "dense" in table_str
    assert "hybrid_reranked" in table_str

    md_str = format_markdown_report(results)
    assert "# DocuLens AI — Automated Evaluation Report" in md_str
    assert "| `dense` |" in md_str
