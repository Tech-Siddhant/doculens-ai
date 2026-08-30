import io
import pytest
from fastapi.testclient import TestClient
from app.api.main import app
from app.core.config import settings
from app.ingestion.chunker import chunk_document
from app.schemas.documents import DocumentMetadata, ExtractedPage, ExtractionResult
from tests.conftest import create_sample_pdf_with_text

client = TestClient(app)


def test_chunking_normal_and_boundary():
    # Long text on page 1
    page1_text = "A" * 120
    extraction = ExtractionResult(
        document_id="doc_test123",
        metadata=DocumentMetadata(total_pages=1),
        pages=[ExtractedPage(page_number=1, text=page1_text, character_count=len(page1_text))],
    )

    # chunk_size=50, chunk_overlap=10 -> step = 40
    result = chunk_document(extraction, chunk_size=50, chunk_overlap=10)

    assert result.document_id == "doc_test123"
    assert result.total_chunks == 3  # 0..50, 40..90, 80..120
    assert result.chunks[0].chunk_id == "doc_test123_p1_c0"
    assert result.chunks[0].page_number == 1
    assert result.chunks[0].chunk_index == 0
    assert len(result.chunks[0].text) == 50

    assert result.chunks[1].chunk_id == "doc_test123_p1_c1"
    assert result.chunks[1].chunk_index == 1
    assert len(result.chunks[1].text) == 50

    assert result.chunks[2].chunk_id == "doc_test123_p1_c2"
    assert result.chunks[2].chunk_index == 2
    assert len(result.chunks[2].text) == 40  # Remaining chars


def test_chunking_short_text():
    short_text = "Short page text."
    extraction = ExtractionResult(
        document_id="doc_short",
        metadata=DocumentMetadata(total_pages=1),
        pages=[ExtractedPage(page_number=1, text=short_text, character_count=len(short_text))],
    )

    result = chunk_document(extraction, chunk_size=500, chunk_overlap=50)

    assert result.total_chunks == 1
    assert result.chunks[0].chunk_id == "doc_short_p1_c0"
    assert result.chunks[0].text == short_text
    assert result.chunks[0].character_count == len(short_text)


def test_chunking_multi_page_and_empty_page():
    extraction = ExtractionResult(
        document_id="doc_multi",
        metadata=DocumentMetadata(total_pages=3),
        pages=[
            ExtractedPage(page_number=1, text="Page 1 content text.", character_count=20),
            ExtractedPage(page_number=2, text="", character_count=0),  # Empty page
            ExtractedPage(page_number=3, text="Page 3 content text.", character_count=20),
        ],
    )

    result = chunk_document(extraction, chunk_size=100, chunk_overlap=10)

    assert result.total_chunks == 2
    assert result.chunks[0].page_number == 1
    assert result.chunks[0].chunk_id == "doc_multi_p1_c0"

    assert result.chunks[1].page_number == 3
    assert result.chunks[1].chunk_id == "doc_multi_p3_c0"


def test_chunking_invalid_parameters():
    extraction = ExtractionResult(
        document_id="doc_invalid",
        metadata=DocumentMetadata(total_pages=1),
        pages=[ExtractedPage(page_number=1, text="Some text", character_count=9)],
    )

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        chunk_document(extraction, chunk_size=0, chunk_overlap=10)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        chunk_document(extraction, chunk_size=100, chunk_overlap=100)


def test_chunking_endpoint_success(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    pdf_bytes = create_sample_pdf_with_text(
        ["Paragraph one of document content.", "Paragraph two of document content."],
        title="Doc Title",
    )

    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("chunk_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    doc_id = upload_res.json()["document_id"]

    chunk_res = client.post(f"{settings.API_V1_STR}/documents/{doc_id}/chunk?chunk_size=100&chunk_overlap=20")
    assert chunk_res.status_code == 200
    data = chunk_res.json()
    assert data["document_id"] == doc_id
    assert data["total_chunks"] >= 2
    assert data["chunks"][0]["document_id"] == doc_id
    assert "chunk_id" in data["chunks"][0]
