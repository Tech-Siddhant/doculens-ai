import io
from pypdf import PdfReader
from pypdf.errors import PyPdfError

MAX_PAGE_COUNT: int = 1000


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
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise PDFValidationError("PDF contains no readable pages.", status_code=400)
        if total_pages > MAX_PAGE_COUNT:
            raise PDFValidationError(
                f"PDF exceeds maximum permitted page limit of {MAX_PAGE_COUNT} pages (found {total_pages}).",
                status_code=400,
            )
    except PDFValidationError:
        raise
    except (PyPdfError, Exception) as exc:
        err_type = type(exc).__name__
        raise PDFValidationError(
            f"Corrupted or unreadable PDF document ({err_type}).",
            status_code=400,
        ) from exc

