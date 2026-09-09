import pytest

from app.services.validator import PDFValidationError, validate_pdf_file


def test_validate_valid_pdf(valid_pdf_bytes: bytes) -> None:
    # Should not raise
    validate_pdf_file(
        filename="document.pdf",
        content=valid_pdf_bytes,
        max_size_bytes=10 * 1024 * 1024,
    )


def test_validate_case_insensitive_pdf_extension(valid_pdf_bytes: bytes) -> None:
    # Should accept .PDF
    validate_pdf_file(
        filename="document.PDF",
        content=valid_pdf_bytes,
        max_size_bytes=10 * 1024 * 1024,
    )


def test_validate_invalid_extension(valid_pdf_bytes: bytes) -> None:
    with pytest.raises(PDFValidationError, match="Invalid file extension"):
        validate_pdf_file(
            filename="document.txt",
            content=valid_pdf_bytes,
            max_size_bytes=10 * 1024 * 1024,
        )


def test_validate_none_filename(valid_pdf_bytes: bytes) -> None:
    with pytest.raises(PDFValidationError, match="Invalid file extension"):
        validate_pdf_file(
            filename=None,
            content=valid_pdf_bytes,
            max_size_bytes=10 * 1024 * 1024,
        )


def test_validate_empty_content() -> None:
    with pytest.raises(PDFValidationError, match="empty"):
        validate_pdf_file(
            filename="empty.pdf",
            content=b"",
            max_size_bytes=10 * 1024 * 1024,
        )


def test_validate_oversized_content(valid_pdf_bytes: bytes) -> None:
    with pytest.raises(PDFValidationError, match="exceeds maximum permitted limit") as exc_info:
        validate_pdf_file(
            filename="large.pdf",
            content=valid_pdf_bytes,
            max_size_bytes=10,  # lower than content size
        )
    assert exc_info.value.status_code == 413


def test_validate_invalid_magic_bytes() -> None:
    fake_content = b"This is not a real PDF file but has text."
    with pytest.raises(PDFValidationError, match="Invalid PDF header"):
        validate_pdf_file(
            filename="fake.pdf",
            content=fake_content,
            max_size_bytes=10 * 1024 * 1024,
        )


def test_validate_corrupted_pdf_structure() -> None:
    corrupted_content = b"%PDF-1.4\nCorrupted garbage body with no xref table or trailer"
    with pytest.raises(PDFValidationError, match="Corrupted or unreadable PDF"):
        validate_pdf_file(
            filename="corrupt.pdf",
            content=corrupted_content,
            max_size_bytes=10 * 1024 * 1024,
        )
