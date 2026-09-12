"""Unit tests for BM25 sparse lexical retrieval service and tokenization."""

import pytest

from app.schemas.document import DocumentChunk
from app.schemas.retrieval import RetrievalResult, RetrievedChunk
from app.services.bm25 import BM25OkapiIndex, BM25Retriever, tokenize


def test_tokenize_basic_and_technical_terms() -> None:
    text = "Model ABC-123 achieved a BLEU score of 42.7 on ISO-27001 compliance."
    tokens = tokenize(text)
    assert "abc-123" in tokens
    assert "bleu" in tokens
    assert "score" in tokens
    assert "42.7" in tokens
    assert "iso-27001" in tokens
    assert "compliance" in tokens


def test_tokenize_edge_cases() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("!@#$%^&*()") == []
    assert tokenize("GPT-4.5_turbo/v2") == ["gpt-4.5_turbo/v2"]
    assert tokenize("Upper and LOWER case MiXeD") == ["upper", "and", "lower", "case", "mixed"]


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="doc_a_p1_c0",
            document_id="doc_a",
            page_number=1,
            chunk_index=0,
            text="Model ABC-123 achieved a state-of-the-art BLEU score of 42.7 in machine translation.",
            char_count=83,
        ),
        DocumentChunk(
            chunk_id="doc_a_p2_c0",
            document_id="doc_a",
            page_number=2,
            chunk_index=0,
            text="The latency and throughput benchmark for ABC-123 was measured on NVIDIA H100 GPUs.",
            char_count=82,
        ),
        DocumentChunk(
            chunk_id="doc_b_p1_c0",
            document_id="doc_b",
            page_number=1,
            chunk_index=0,
            text="Model XYZ-900 achieved a BLEU score of 38.1 with lower memory consumption.",
            char_count=73,
        ),
        DocumentChunk(
            chunk_id="doc_b_p2_c0",
            document_id="doc_b",
            page_number=2,
            chunk_index=0,
            text="Compliance guidelines follow ISO-27001 security standards and SOC-2 auditing.",
            char_count=77,
        ),
    ]


def test_index_and_stats(sample_chunks: list[DocumentChunk]) -> None:
    index = BM25OkapiIndex()
    count = index.index_chunks(sample_chunks)
    assert count == 4

    stats = index.get_stats()
    assert stats.total_documents == 2
    assert stats.total_chunks == 4
    assert stats.total_terms > 0
    assert stats.avg_chunk_length > 0


def test_exact_technical_keyword_search(sample_chunks: list[DocumentChunk]) -> None:
    index = BM25OkapiIndex()
    index.index_chunks(sample_chunks)

    # Search for specific model identifier
    results = index.search(query="ABC-123", top_k=5)
    assert len(results) == 2
    assert results[0].chunk_id in ("doc_a_p1_c0", "doc_a_p2_c0")
    assert all(r.retrieval_type == "bm25" for r in results)
    assert all(r.document_id == "doc_a" for r in results)


def test_discriminating_multi_term_search(sample_chunks: list[DocumentChunk]) -> None:
    index = BM25OkapiIndex()
    index.index_chunks(sample_chunks)

    # Query matching both docs on 'BLEU', but '42.7' only matches doc_a_p1_c0
    results = index.search(query="BLEU 42.7", top_k=5)
    assert len(results) >= 2
    assert results[0].chunk_id == "doc_a_p1_c0"
    assert results[0].score > results[1].score


def test_document_isolation_filtering(sample_chunks: list[DocumentChunk]) -> None:
    index = BM25OkapiIndex()
    index.index_chunks(sample_chunks)

    # Search for common term 'BLEU' restricted to doc_b
    results_b = index.search(query="BLEU", top_k=5, document_id="doc_b")
    assert len(results_b) == 1
    assert results_b[0].chunk_id == "doc_b_p1_c0"
    assert results_b[0].document_id == "doc_b"

    # Search restricted to doc_a
    results_a = index.search(query="BLEU", top_k=5, document_id="doc_a")
    assert len(results_a) == 1
    assert results_a[0].chunk_id == "doc_a_p1_c0"
    assert results_a[0].document_id == "doc_a"


def test_score_threshold_filtering(sample_chunks: list[DocumentChunk]) -> None:
    index = BM25OkapiIndex()
    index.index_chunks(sample_chunks)

    results_all = index.search(query="ISO-27001", top_k=5)
    assert len(results_all) == 1
    top_score = results_all[0].score

    # Threshold slightly higher than top_score returns empty
    results_filtered = index.search(query="ISO-27001", top_k=5, score_threshold=top_score + 1.0)
    assert len(results_filtered) == 0

    # Threshold lower than top_score returns result
    results_allowed = index.search(query="ISO-27001", top_k=5, score_threshold=top_score - 0.1)
    assert len(results_allowed) == 1


def test_delete_by_document(sample_chunks: list[DocumentChunk]) -> None:
    index = BM25OkapiIndex()
    index.index_chunks(sample_chunks)

    assert index.count_chunks() == 4
    assert index.count_chunks(document_id="doc_a") == 2

    deleted = index.delete_by_document("doc_a")
    assert deleted == 2
    assert index.count_chunks(document_id="doc_a") == 0
    assert index.count_chunks(document_id="doc_b") == 2
    assert index.count_chunks() == 2

    # Query for ABC-123 should now yield no results
    results = index.search(query="ABC-123", top_k=5)
    assert len(results) == 0


def test_reindexing_existing_chunk(sample_chunks: list[DocumentChunk]) -> None:
    index = BM25OkapiIndex()
    index.index_chunks(sample_chunks)

    updated_chunk = DocumentChunk(
        chunk_id="doc_a_p1_c0",
        document_id="doc_a",
        page_number=1,
        chunk_index=0,
        text="Updated text with only ROUGE metrics and no previous keywords.",
        char_count=63,
    )
    index.index_chunks([updated_chunk])

    assert index.count_chunks() == 4
    # Searching for BLEU should no longer match doc_a_p1_c0
    results_bleu = index.search(query="BLEU", top_k=5, document_id="doc_a")
    assert len(results_bleu) == 0

    # Searching for ROUGE matches doc_a_p1_c0
    results_rouge = index.search(query="ROUGE", top_k=5, document_id="doc_a")
    assert len(results_rouge) == 1
    assert results_rouge[0].chunk_id == "doc_a_p1_c0"


def test_clear_index(sample_chunks: list[DocumentChunk]) -> None:
    index = BM25OkapiIndex()
    index.index_chunks(sample_chunks)
    assert index.count_chunks() == 4

    index.clear()
    assert index.count_chunks() == 0
    assert index.search(query="ABC-123") == []


def test_empty_corpus_and_no_match(sample_chunks: list[DocumentChunk]) -> None:
    index = BM25OkapiIndex()
    assert index.search(query="anything") == []

    index.index_chunks(sample_chunks)
    assert index.search(query="nonexistentqueryterm12345") == []


def test_validation_errors() -> None:
    index = BM25OkapiIndex()

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        index.search(query="")

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        index.search(query="   ")

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        index.search(query="test", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        index.search(query="test", top_k=-2)

    with pytest.raises(ValueError, match="document_id filter must be a non-empty string"):
        index.search(query="test", document_id="")

    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        index.delete_by_document("")


def test_bm25_retriever_service(sample_chunks: list[DocumentChunk]) -> None:
    retriever = BM25Retriever()
    retriever.index_chunks(sample_chunks)

    res = retriever.retrieve(query="ISO-27001", top_k=2)
    assert isinstance(res, RetrievalResult)
    assert res.query == "ISO-27001"
    assert res.total_results == 1
    assert res.results[0].chunk_id == "doc_b_p2_c0"
    assert res.results[0].raw_score is not None
    assert res.results[0].raw_score > 0.0
    assert res.results[0].retrieval_type == "bm25"
    assert res.results[0].rank == 1
