import io
from pypdf import PdfReader
from pypdf.errors import PyPdfError


class PDFValidationError(Exception):
    """Custom exception raised when PDF validation fails."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def validate_pdf_file(content: bytes, filename: str, max_size_bytes: int) -> None:
    """Validates uploaded PDF file size, filename extension, magic bytes, and PDF structure.

    Raises PDFValidationError on validation failure.
    """
    if not content:
        raise PDFValidationError("Uploaded file is empty.", status_code=400)

    if len(content) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        raise PDFValidationError(
            f"File size exceeds maximum permitted limit of {max_mb:.1f} MB.",
            status_code=413,
        )

    if not filename.lower().endswith(".pdf"):
        raise PDFValidationError(
            "Invalid file type. Only PDF documents (.pdf) are supported.",
            status_code=400,
        )

    if not content.startswith(b"%PDF-"):
        raise PDFValidationError(
            "Invalid file format. File does not contain valid PDF magic bytes.",
            status_code=400,
        )

    try:
        reader = PdfReader(io.BytesIO(content))
        # Ensure pages can be accessed
        _ = len(reader.pages)
    except Exception as exc:
        raise PDFValidationError(
            f"Corrupted or unreadable PDF document: {str(exc)}",
            status_code=400,
        ) from exc
