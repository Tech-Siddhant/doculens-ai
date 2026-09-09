from pathlib import Path

import pytest

from app.services.storage import (
    generate_document_id,
    get_document_path,
    get_upload_dir,
    is_valid_document_id,
    save_uploaded_pdf,
)


def test_generate_document_id() -> None:
    doc_id = generate_document_id()
    assert doc_id.startswith("doc_")
    assert len(doc_id) == 16  # "doc_" (4) + 12 hex chars (12)
    assert is_valid_document_id(doc_id)


def test_is_valid_document_id() -> None:
    assert is_valid_document_id("doc_0123456789ab") is True
    assert is_valid_document_id("doc_abcdef012345") is True
    assert is_valid_document_id("doc_ABCDEF012345") is False  # lowercase only
    assert is_valid_document_id("doc_123") is False  # too short
    assert is_valid_document_id("doc_12345678901234567") is False  # too long
    assert is_valid_document_id("../doc_123456789abc") is False  # path traversal
    assert is_valid_document_id("document_123456") is False


def test_save_and_retrieve_document(valid_pdf_bytes: bytes) -> None:
    doc_id = generate_document_id()
    saved_path = save_uploaded_pdf(doc_id, valid_pdf_bytes)
    assert saved_path.exists()
    assert saved_path.is_file()
    assert saved_path.read_bytes() == valid_pdf_bytes

    retrieved_path = get_document_path(doc_id)
    assert retrieved_path == saved_path


def test_save_invalid_doc_id_raises(valid_pdf_bytes: bytes) -> None:
    with pytest.raises(ValueError, match="Invalid document identifier"):
        save_uploaded_pdf("invalid_id", valid_pdf_bytes)


def test_get_nonexistent_document_path() -> None:
    assert get_document_path("doc_000000000000") is None
    assert get_document_path("invalid_format") is None
