import io
import pytest
from fastapi.testclient import TestClient
from app.api.main import app
from app.core.config import settings
from app.ingestion.extractor import extract_text_and_metadata
from app.ingestion.storage import save_uploaded_pdf
from tests.conftest import create_sample_pdf_with_text

client = TestClient(app)


def test_extract_text_and_metadata_success(tmp_path):
    pdf_bytes = create_sample_pdf_with_text(
        ["First page content", "Second page content"],
        title="Research Paper",
        author="Alice Smith",
    )
    doc_id, pdf_path = save_uploaded_pdf(pdf_bytes, "paper.pdf", str(tmp_path))

    result = extract_text_and_metadata(pdf_path, doc_id)

    assert result.document_id == doc_id
    assert result.metadata.title == "Research Paper"
    assert result.metadata.author == "Alice Smith"
    assert result.metadata.total_pages == 2

    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert "First page content" in result.pages[0].text
    assert result.pages[0].character_count > 0

    assert result.pages[1].page_number == 2
    assert "Second page content" in result.pages[1].text


def test_extract_non_existent_file():
    with pytest.raises(FileNotFoundError):
        extract_text_and_metadata("data/storage/non_existent_doc.pdf", "doc_missing")


def test_extract_endpoint_success(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    pdf_bytes = create_sample_pdf_with_text(["API extraction content"], title="API Doc", author="Bob")

    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("api.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]

    extract_res = client.post(f"{settings.API_V1_STR}/documents/{doc_id}/extract")
    assert extract_res.status_code == 200
    data = extract_res.json()
    assert data["document_id"] == doc_id
    assert data["metadata"]["title"] == "API Doc"
    assert data["metadata"]["total_pages"] == 1
    assert len(data["pages"]) == 1
    assert "API extraction content" in data["pages"][0]["text"]


def test_extract_endpoint_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    response = client.post(f"{settings.API_V1_STR}/documents/doc_nonexistent/extract")
    assert response.status_code == 404
