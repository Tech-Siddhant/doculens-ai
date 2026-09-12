"""Unit tests for BM25 retrieval API endpoints."""

from fastapi.testclient import TestClient

from app.core.config import settings
from app.schemas.document import DocumentChunk
from app.services.bm25 import bm25_retriever


def test_bm25_document_retrieve_api(client: TestClient, valid_pdf_bytes: bytes) -> None:
    # 1. Upload valid document
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("test_bm25.pdf", valid_pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # 2. Index document (this triggers extraction, chunking, and both dense + BM25 indexing)
    index_res = client.post(f"{settings.API_V1_STR}/documents/{doc_id}/index")
    assert index_res.status_code == 200

    # 3. Retrieve using BM25 endpoint
    retrieval_res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/bm25",
        json={"query": "introduction text", "top_k": 3},
    )
    assert retrieval_res.status_code == 200
    data = retrieval_res.json()
    assert data["query"] == "introduction text"
    assert data["document_id"] == doc_id
    assert data["total_results"] >= 1
    assert data["results"][0]["retrieval_type"] == "bm25"
    assert data["results"][0]["raw_score"] is not None


def test_bm25_collection_retrieve_api(client: TestClient, valid_pdf_bytes: bytes) -> None:
    # Upload and index
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("test_bm25_coll.pdf", valid_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]
    client.post(f"{settings.API_V1_STR}/documents/{doc_id}/index")

    # Global BM25 retrieve
    retrieval_res = client.post(
        f"{settings.API_V1_STR}/documents/retrieve/bm25",
        json={"query": "details", "top_k": 2},
    )
    assert retrieval_res.status_code == 200
    data = retrieval_res.json()
    assert data["total_results"] >= 1
    assert data["results"][0]["retrieval_type"] == "bm25"


def test_bm25_retrieve_not_found(client: TestClient) -> None:
    fake_id = "doc_000000000000"
    res = client.post(
        f"{settings.API_V1_STR}/documents/{fake_id}/retrieve/bm25",
        json={"query": "test query"},
    )
    assert res.status_code == 404


def test_bm25_retrieve_invalid_inputs(client: TestClient, valid_pdf_bytes: bytes) -> None:
    # Invalid document ID format
    res = client.post(
        f"{settings.API_V1_STR}/documents/invalid-id/retrieve/bm25",
        json={"query": "test"},
    )
    assert res.status_code == 400

    # Upload document for payload validation tests
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("doc_val.pdf", valid_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]

    # Empty string query (fails pydantic min_length=1 with 422)
    res_empty = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/bm25",
        json={"query": ""},
    )
    assert res_empty.status_code == 422

    # Whitespace-only query (fails route validation with 400)
    res_space = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/bm25",
        json={"query": "   "},
    )
    assert res_space.status_code == 400

    # Invalid top_k (fails pydantic ge=1 with 422)
    res_k = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/retrieve/bm25",
        json={"query": "test", "top_k": 0},
    )
    assert res_k.status_code == 422
