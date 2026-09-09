from fastapi.testclient import TestClient

from app.core.config import settings


def test_upload_valid_pdf(client: TestClient, valid_pdf_bytes: bytes) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("sample.pdf", valid_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["document_id"].startswith("doc_")
    assert len(data["document_id"]) == 16
    assert data["filename"] == "sample.pdf"
    assert data["size_bytes"] == len(valid_pdf_bytes)
    assert data["status"] == "uploaded"


def test_upload_empty_file(client: TestClient) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_invalid_extension(client: TestClient, valid_pdf_bytes: bytes) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("document.txt", valid_pdf_bytes, "text/plain")},
    )
    assert response.status_code == 400
    assert "extension" in response.json()["detail"].lower()


def test_upload_fake_pdf_magic_bytes(client: TestClient) -> None:
    fake_content = b"This is plain text with a .pdf extension"
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("fake.pdf", fake_content, "application/pdf")},
    )
    assert response.status_code == 400
    assert "magic bytes" in response.json()["detail"].lower() or "header" in response.json()["detail"].lower()


def test_upload_corrupted_pdf(client: TestClient) -> None:
    corrupt_content = b"%PDF-1.4\nCorrupted content without valid PDF trailer"
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("corrupt.pdf", corrupt_content, "application/pdf")},
    )
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower() or "unreadable" in response.json()["detail"].lower()


def test_upload_oversized_file(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 50)
    pdf_bytes = b"%PDF-" + b"0" * 100
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("large.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()


def test_extract_endpoint(client: TestClient, multi_page_pdf_bytes: bytes) -> None:
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("arch.pdf", multi_page_pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    extract_res = client.post(f"{settings.API_V1_STR}/documents/{doc_id}/extract")
    assert extract_res.status_code == 200
    data = extract_res.json()
    assert data["document_id"] == doc_id
    assert data["metadata"]["total_pages"] == 3
    assert data["metadata"]["title"] == "DocuLens Architecture"
    assert data["metadata"]["author"] == "DocuLens Team"
    assert len(data["pages"]) == 3
    assert data["pages"][0]["page_number"] == 1
    assert "DocuLens is a Multimodal" in data["pages"][0]["text"]


def test_extract_nonexistent_document(client: TestClient) -> None:
    response = client.post(f"{settings.API_V1_STR}/documents/doc_000000000000/extract")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_extract_invalid_document_id_format(client: TestClient) -> None:
    response = client.post(f"{settings.API_V1_STR}/documents/invalid_doc_id/extract")
    assert response.status_code == 400
    assert "invalid document identifier format" in response.json()["detail"].lower()


def test_chunk_endpoint(client: TestClient, multi_page_pdf_bytes: bytes) -> None:
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("arch.pdf", multi_page_pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    chunk_res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/chunk?chunk_size=100&chunk_overlap=20"
    )
    assert chunk_res.status_code == 200
    data = chunk_res.json()
    assert data["document_id"] == doc_id
    assert data["chunk_size"] == 100
    assert data["chunk_overlap"] == 20
    assert data["total_chunks"] > 0
    assert len(data["chunks"]) == data["total_chunks"]

    # Verify chunk structure
    first_chunk = data["chunks"][0]
    assert first_chunk["document_id"] == doc_id
    assert first_chunk["page_number"] == 1
    assert first_chunk["chunk_index"] == 0
    assert first_chunk["chunk_id"] == f"{doc_id}_p1_c0"
    assert len(first_chunk["text"]) <= 100


def test_chunk_invalid_parameters(client: TestClient, valid_pdf_bytes: bytes) -> None:
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("doc.pdf", valid_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]

    # chunk_size <= 0
    res1 = client.post(f"{settings.API_V1_STR}/documents/{doc_id}/chunk?chunk_size=0")
    assert res1.status_code == 400

    # chunk_overlap >= chunk_size
    res2 = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/chunk?chunk_size=100&chunk_overlap=100"
    )
    assert res2.status_code == 400

    # chunk_overlap < 0
    res3 = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/chunk?chunk_size=100&chunk_overlap=-5"
    )
    assert res3.status_code == 400


def test_chunk_nonexistent_document(client: TestClient) -> None:
    response = client.post(f"{settings.API_V1_STR}/documents/doc_000000000000/chunk")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_chunk_invalid_document_id_format(client: TestClient) -> None:
    response = client.post(f"{settings.API_V1_STR}/documents/bad_id/chunk")
    assert response.status_code == 400
    assert "invalid document identifier format" in response.json()["detail"].lower()


def test_index_and_query_document_lifecycle(
    client: TestClient, multi_page_pdf_bytes: bytes
) -> None:
    # Upload multi-page document
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("arch.pdf", multi_page_pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # Index the document
    index_res = client.post(f"{settings.API_V1_STR}/documents/{doc_id}/index")
    assert index_res.status_code == 200
    index_data = index_res.json()
    assert index_data["document_id"] == doc_id
    assert index_data["total_pages"] == 3
    assert index_data["total_chunks"] > 0
    assert index_data["total_embeddings"] == index_data["total_chunks"]
    assert index_data["status"] == "indexed"

    # Query the document via /{document_id}/query
    query_res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/query",
        json={"query": "multimodal architecture and document analysis", "top_k": 2},
    )
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert query_data["query"] == "multimodal architecture and document analysis"
    assert query_data["document_id"] == doc_id
    assert query_data["top_k"] == 2
    assert query_data["total_results"] > 0
    assert len(query_data["results"]) <= 2

    first_result = query_data["results"][0]
    assert first_result["rank"] == 1
    assert first_result["document_id"] == doc_id
    assert first_result["page_number"] in [1, 2, 3]
    assert "chunk_id" in first_result
    assert "text" in first_result
    assert first_result["score"] > 0.0

    # Cross-document / general collection query endpoint
    general_query_res = client.post(
        f"{settings.API_V1_STR}/documents/query",
        json={"query": "multimodal architecture", "top_k": 1},
    )
    assert general_query_res.status_code == 200
    assert general_query_res.json()["total_results"] >= 1


def test_index_nonexistent_document(client: TestClient) -> None:
    response = client.post(f"{settings.API_V1_STR}/documents/doc_000000000000/index")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_index_invalid_document_id(client: TestClient) -> None:
    response = client.post(f"{settings.API_V1_STR}/documents/invalid_doc_id/index")
    assert response.status_code == 400


def test_query_nonexistent_document(client: TestClient) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/documents/doc_000000000000/query",
        json={"query": "test query"},
    )
    assert response.status_code == 404


def test_query_empty_or_invalid_payload(
    client: TestClient, valid_pdf_bytes: bytes
) -> None:
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("doc.pdf", valid_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]

    # Empty string query (fails pydantic min_length=1 with 422)
    empty_res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/query",
        json={"query": ""},
    )
    assert empty_res.status_code == 422

    # Whitespace-only query (fails custom validation with 400)
    whitespace_res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/query",
        json={"query": "    "},
    )
    assert whitespace_res.status_code == 400

    # Invalid top_k (fails pydantic ge=1 with 422)
    invalid_k_res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/query",
        json={"query": "valid query", "top_k": 0},
    )
    assert invalid_k_res.status_code == 422


def test_ask_document_endpoint_success(
    client: TestClient, multi_page_pdf_bytes: bytes
) -> None:
    # 1. Upload
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("arch.pdf", multi_page_pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # 2. Index
    index_res = client.post(f"{settings.API_V1_STR}/documents/{doc_id}/index")
    assert index_res.status_code == 200

    # 3. Ask document
    ask_res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/ask",
        json={"question": "What is DocuLens AI and how does it work?", "top_k": 3},
    )
    assert ask_res.status_code == 200
    data = ask_res.json()

    assert data["question"] == "What is DocuLens AI and how does it work?"
    assert data["document_id"] == doc_id
    assert "answer" in data
    assert data["is_grounded"] is True
    assert data["provider"] == "mock"
    assert len(data["citations"]) > 0
    assert data["citations"][0]["page_number"] in [1, 2, 3]


def test_ask_nonexistent_document(client: TestClient) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/documents/doc_000000000000/ask",
        json={"question": "What is the summary?"},
    )
    assert response.status_code == 404


def test_ask_invalid_document_id(client: TestClient) -> None:
    response = client.post(
        f"{settings.API_V1_STR}/documents/invalid_doc/ask",
        json={"question": "What is the summary?"},
    )
    assert response.status_code == 400


def test_ask_empty_question(client: TestClient, valid_pdf_bytes: bytes) -> None:
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("doc.pdf", valid_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]

    res = client.post(
        f"{settings.API_V1_STR}/documents/{doc_id}/ask",
        json={"question": ""},
    )
    assert res.status_code == 422


def test_ask_collection_endpoint(
    client: TestClient, multi_page_pdf_bytes: bytes
) -> None:
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("arch.pdf", multi_page_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]
    client.post(f"{settings.API_V1_STR}/documents/{doc_id}/index")

    # Ask collection
    res = client.post(
        f"{settings.API_V1_STR}/documents/ask",
        json={"question": "Explain system architecture.", "top_k": 2},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_grounded"] is True
    assert len(data["citations"]) > 0
