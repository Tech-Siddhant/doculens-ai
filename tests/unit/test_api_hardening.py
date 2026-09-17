import io
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.api.main import create_app
from app.core.config import settings
from app.services.validator import PDFValidationError, validate_pdf_file


def test_document_list_pagination(client: TestClient) -> None:
    """Verify document list supports limit and offset pagination."""
    # Test valid defaults
    res = client.get(f"{settings.API_V1_STR}/documents?limit=10&offset=0")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # Test invalid limit (<=0 or >200) -> 422
    res_zero = client.get(f"{settings.API_V1_STR}/documents?limit=0")
    assert res_zero.status_code == 422

    res_huge = client.get(f"{settings.API_V1_STR}/documents?limit=500")
    assert res_huge.status_code == 422

    # Test invalid offset (<0) -> 422
    res_neg_offset = client.get(f"{settings.API_V1_STR}/documents?offset=-1")
    assert res_neg_offset.status_code == 422


def test_chunk_size_and_page_bounds(client: TestClient) -> None:
    """Verify upper and lower bounds on chunk size and page number."""
    doc_id = "doc_1234567890ab"
    
    # Chunk size > 10000 -> 400
    res_chunk = client.post(f"{settings.API_V1_STR}/documents/{doc_id}/chunk?chunk_size=20000")
    assert res_chunk.status_code == 400
    assert "chunk_size" in res_chunk.json()["detail"]

    # Page number > 5000 -> 400
    res_page = client.get(f"{settings.API_V1_STR}/documents/{doc_id}/pages/9999/image")
    assert res_page.status_code == 400
    assert "page_number" in res_page.json()["detail"]


def test_pdf_bomb_page_count_limit() -> None:
    """Verify PDF files exceeding MAX_PAGE_COUNT (1000) are rejected safely."""
    writer = PdfWriter()
    for _ in range(1001):
        writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    try:
        validate_pdf_file(filename="bomb.pdf", content=pdf_bytes, max_size_bytes=20 * 1024 * 1024)
        assert False, "Should have raised PDFValidationError for > 1000 pages"
    except PDFValidationError as exc:
        assert exc.status_code == 400
        assert "maximum permitted page limit" in exc.message


def test_global_unhandled_exception_does_not_leak_internals() -> None:
    """Verify unhandled 500 exceptions return safe generic message and no stack trace."""
    app = create_app()

    @app.get("/api/v1/test-server-error")
    def trigger_error():
        raise RuntimeError("Sensitive secret: sk-12345678901234567890 /home/keshav/secrets.env")

    test_client = TestClient(app, raise_server_exceptions=False)
    res = test_client.get("/api/v1/test-server-error")
    assert res.status_code == 500
    data = res.json()
    assert "detail" in data
    assert data["detail"] == "An internal server error occurred. Please contact support or retry."
    assert "sk-" not in str(data)
    assert "/home/keshav" not in str(data)
    assert "Traceback" not in str(data)


def test_global_validation_error_formatting() -> None:
    """Verify request validation errors return structured clean JSON with status 422."""
    app = create_app()
    test_client = TestClient(app)

    # Malformed question request (missing required 'question' field)
    res = test_client.post("/api/v1/documents/ask", json={})
    assert res.status_code == 422
    data = res.json()
    assert "detail" in data
    assert isinstance(data["detail"], list)
    assert any("question" in err["loc"] for err in data["detail"])
