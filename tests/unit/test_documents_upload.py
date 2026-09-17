import io
from fastapi.testclient import TestClient
from app.api.main import app
from app.core.config import settings

client = TestClient(app)


def test_upload_valid_pdf(valid_pdf_bytes, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))

    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("sample.pdf", io.BytesIO(valid_pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["document_id"].startswith("doc_")
    assert data["filename"] == "sample.pdf"
    assert data["status"] == "uploaded"
    assert data["size_bytes"] == len(valid_pdf_bytes)


def test_upload_invalid_file_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))

    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"Hello world"), "text/plain")},
    )

    assert response.status_code == 400
    assert "Only .pdf files are accepted" in response.json()["detail"]


def test_upload_corrupted_pdf_content(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))

    # Fake PDF extension but bad bytes header
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("corrupted.pdf", io.BytesIO(b"NOT_A_PDF_HEADER"), "application/pdf")},
    )

    assert response.status_code == 400
    assert "magic bytes do not match PDF format" in response.json()["detail"]


def test_upload_oversized_file(valid_pdf_bytes, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 100)  # Set tiny 100 bytes limit

    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("large.pdf", io.BytesIO(valid_pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 413
    assert "exceeds maximum permitted limit" in response.json()["detail"]
