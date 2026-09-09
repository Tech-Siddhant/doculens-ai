from pathlib import Path

import pytest

from app.services.extractor import extract_text_and_metadata, normalize_whitespace
from app.services.storage import generate_document_id, save_uploaded_pdf


def test_normalize_whitespace() -> None:
    raw_text = "  Hello   world \t from   DocuLens!\r\n\r\n\r\nSecond  line  here.  \n"
    normalized = normalize_whitespace(raw_text)
    assert normalized == "Hello world from DocuLens!\n\nSecond line here."


def test_normalize_empty_whitespace() -> None:
    assert normalize_whitespace("") == ""
    assert normalize_whitespace("   \n\t  ") == ""


def test_extract_multi_page_pdf(multi_page_pdf_bytes: bytes) -> None:
    doc_id = generate_document_id()
    pdf_path = save_uploaded_pdf(doc_id, multi_page_pdf_bytes)

    result = extract_text_and_metadata(doc_id, pdf_path)

    assert result.document_id == doc_id
    assert result.metadata.total_pages == 3
    assert result.metadata.title == "DocuLens Architecture"
    assert result.metadata.author == "DocuLens Team"
    assert result.metadata.file_size_bytes == len(multi_page_pdf_bytes)
    assert len(result.pages) == 3

    # Check 1-based page numbering
    assert result.pages[0].page_number == 1
    assert "DocuLens is a Multimodal" in result.pages[0].text
    assert result.pages[0].char_count == len(result.pages[0].text)

    assert result.pages[1].page_number == 2
    assert "Page two covers" in result.pages[1].text
    assert result.pages[1].char_count == len(result.pages[1].text)

    assert result.pages[2].page_number == 3
    assert "Page three discusses" in result.pages[2].text
    assert result.pages[2].char_count == len(result.pages[2].text)


def test_extract_empty_page(empty_page_pdf_bytes: bytes) -> None:
    doc_id = generate_document_id()
    pdf_path = save_uploaded_pdf(doc_id, empty_page_pdf_bytes)

    result = extract_text_and_metadata(doc_id, pdf_path)
    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert result.pages[0].text == "Valid page text"
    assert result.pages[1].page_number == 2
    assert result.pages[1].text == ""
    assert result.pages[1].char_count == 0


def test_extract_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="Document file not found"):
        extract_text_and_metadata("doc_1234567890ab", Path("/non/existent/path.pdf"))
