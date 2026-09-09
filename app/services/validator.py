import io
from pypdf import PdfReader
from pypdf.errors import PdfReadError


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
        if len(reader.pages) == 0:
            raise PDFValidationError("PDF contains no readable pages.", status_code=400)
    except PDFValidationError:
        raise
    except (PdfReadError, Exception) as err:
        raise PDFValidationError(
            f"Corrupted or unreadable PDF structure: {str(err)}",
            status_code=400,
        ) from err
