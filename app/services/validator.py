import io
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PAGE_COUNT: int = 1000


class PDFValidationError(Exception):
    """Custom exception for PDF validation failures."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def validate_pdf_file(filename: str | None, content: bytes, max_size_bytes: int) -> None:
    """Validate PDF file name, size, magic header, and structural readability."""
    if not filename or not filename.lower().endswith(".pdf"):
        raise PDFValidationError("Invalid file extension. Only .pdf files are accepted.", status_code=400)

    # Reject path separators / control characters: the original filename is echoed
    # in API responses and must never be usable as a path or reflected payload.
    if "/" in filename or "\\" in filename or "\x00" in filename:
        raise PDFValidationError(
            "Invalid filename. Filename must not contain path separators or control characters.",
            status_code=400,
        )

    if not content or len(content) == 0:
        raise PDFValidationError("Uploaded file is empty.", status_code=400)

    if len(content) > max_size_bytes:
        raise PDFValidationError(
            f"File size ({len(content)} bytes) exceeds maximum permitted limit ({max_size_bytes} bytes).",
            status_code=413,
        )

    # Magic byte check: must start with %PDF-
    if not content.startswith(b"%PDF-"):
        raise PDFValidationError(
            "Invalid PDF header. File magic bytes do not match PDF format.",
            status_code=400,
        )

    # Structural readability check with pypdf
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
    except (PdfReadError, Exception) as err:
        err_type = type(err).__name__
        raise PDFValidationError(
            f"Corrupted or unreadable PDF structure ({err_type}).",
            status_code=400,
        ) from err

