"""Unit tests for Phase 4.5 Hybrid Retrieval API endpoints."""

from fastapi.testclient import TestClient

from app.core.config import settings
from app.schemas.retrieval import HybridRetrievalResult


def test_hybrid_document_retrieve_api_weighted(client: TestClient, valid_pdf_bytes: bytes) -> None:
    # 1. Upload valid document
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("test_hybrid.pdf", valid_pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # 2. Index document (generates chunks, dense embeddings, and BM25 index)
    index_res = client.post(f"{settings.API_V1_STR}/documents/{doc_id}/index")
    assert index_res.status_code == 200

    # 3. Retrieve using weighted hybrid endpoint
    retrieval_res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/hybrid",
        json={
            "query": "introduction text",
            "top_k": 3,
            "strategy": "weighted",
            "weights": {"dense": 0.6, "bm25": 0.4, "visual": 0.0},
        },
    )
    assert retrieval_res.status_code == 200
    data = retrieval_res.json()
    assert data["query"] == "introduction text"
    assert data["document_id"] == doc_id
    assert data["strategy"] == "weighted"
    assert data["total_results"] >= 1
    assert data["weights"]["dense"] == 0.6
    assert data["weights"]["bm25"] == 0.4

    first = data["results"][0]
    assert first["rank"] == 1
    assert isinstance(first["score"], float)
    assert "dense" in first["sources"] or "bm25" in first["sources"]
    assert "raw_scores" in first
    assert "normalized_scores" in first
    assert first["document_id"] == doc_id


def test_hybrid_document_retrieve_api_rrf(client: TestClient, valid_pdf_bytes: bytes) -> None:
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("test_rrf.pdf", valid_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]
    client.post(f"{settings.API_V1_STR}/documents/{doc_id}/index")

    retrieval_res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/hybrid",
        json={
            "query": "content details",
            "top_k": 2,
            "strategy": "rrf",
            "rrf_k": 60,
        },
    )
    assert retrieval_res.status_code == 200
    data = retrieval_res.json()
    assert data["strategy"] == "rrf"
    assert data["rrf_k"] == 60
    assert data["total_results"] >= 1


def test_hybrid_collection_retrieve_api(client: TestClient, valid_pdf_bytes: bytes) -> None:
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("test_coll_hybrid.pdf", valid_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]
    client.post(f"{settings.API_V1_STR}/documents/{doc_id}/index")

    retrieval_res = client.post(
        f"{settings.API_V1_STR}/documents/retrieve/hybrid",
        json={"query": "introduction", "top_k": 2},
    )
    assert retrieval_res.status_code == 200
    data = retrieval_res.json()
    assert data["total_results"] >= 1


def test_hybrid_retrieve_not_found(client: TestClient) -> None:
    fake_id = "doc_000000000000"
    res = client.post(
        f"{settings.API_V1_STR}/documents/{fake_id}/retrieve/hybrid",
        json={"query": "nonexistent query"},
    )
    assert res.status_code == 404


def test_hybrid_retrieve_invalid_inputs(client: TestClient, valid_pdf_bytes: bytes) -> None:
    # Invalid document ID format
    res_bad_id = client.post(
        f"{settings.API_V1_STR}/documents/invalid-id/retrieve/hybrid",
        json={"query": "test"},
    )
    assert res_bad_id.status_code == 400

    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("doc_val_hybrid.pdf", valid_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]

    # Empty query string (422)
    res_empty = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/hybrid",
        json={"query": ""},
    )
    assert res_empty.status_code == 422

    # Whitespace-only query (400)
    res_space = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/hybrid",
        json={"query": "    "},
    )
    assert res_space.status_code == 400

    # Invalid top_k (422)
    res_top_k = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/hybrid",
        json={"query": "test", "top_k": 0},
    )
    assert res_top_k.status_code == 422

    # Invalid strategy (422)
    res_strat = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/hybrid",
        json={"query": "test", "strategy": "invalid_strat"},
    )
    assert res_strat.status_code == 422

    # Negative weights (422)
    res_neg_w = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/hybrid",
        json={"query": "test", "weights": {"dense": -0.1, "bm25": 0.5, "visual": 0.5}},
    )
    assert res_neg_w.status_code == 422

    # All-zero weights (400)
    res_zero_w = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/hybrid",
        json={"query": "test", "weights": {"dense": 0.0, "bm25": 0.0, "visual": 0.0}},
    )
    assert res_zero_w.status_code == 400


def test_hybrid_document_isolation_between_two_docs(
    client: TestClient,
    valid_pdf_bytes: bytes,
    multi_page_pdf_bytes: bytes,
) -> None:
    # 1. Upload Doc A and Doc B
    res_a = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("doc_a.pdf", valid_pdf_bytes, "application/pdf")},
    )
    doc_a_id = res_a.json()["document_id"]
    client.post(f"{settings.API_V1_STR}/documents/{doc_a_id}/index")

    res_b = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("doc_b.pdf", multi_page_pdf_bytes, "application/pdf")},
    )
    doc_b_id = res_b.json()["document_id"]
    client.post(f"{settings.API_V1_STR}/documents/{doc_b_id}/index")

    # 2. Query Doc A specifically
    res_query_a = client.post(
        f"{settings.API_V1_STR}/documents/{doc_a_id}/retrieve/hybrid",
        json={"query": "DocuLens architecture multimodal", "top_k": 5},
    )
    assert res_query_a.status_code == 200
    data_a = res_query_a.json()
    for candidate in data_a["results"]:
        assert candidate["document_id"] == doc_a_id
        assert candidate["document_id"] != doc_b_id

    # 3. Query Doc B specifically
    res_query_b = client.post(
        f"{settings.API_V1_STR}/documents/{doc_b_id}/retrieve/hybrid",
        json={"query": "DocuLens architecture multimodal", "top_k": 5},
    )
    assert res_query_b.status_code == 200
    data_b = res_query_b.json()
    for candidate in data_b["results"]:
        assert candidate["document_id"] == doc_b_id
        assert candidate["document_id"] != doc_a_id


def test_hybrid_no_secret_or_internal_path_leakage(client: TestClient, valid_pdf_bytes: bytes) -> None:
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("secret_test.pdf", valid_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]
    client.post(f"{settings.API_V1_STR}/documents/{doc_id}/index")

    res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/hybrid",
        json={"query": "introduction", "top_k": 3},
    )
    assert res.status_code == 200
    raw_response_text = res.text

    # Verify no secret key, API keys, or raw root server paths leaked
    assert "SECRET_KEY" not in raw_response_text
    assert "GEMINI_API_KEY" not in raw_response_text
    assert "OPENAI_API_KEY" not in raw_response_text
    assert "/root" not in raw_response_text
    assert "/etc" not in raw_response_text
