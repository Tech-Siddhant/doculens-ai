import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.schemas.document import DocumentMetadata, ExtractedPage, ExtractionResult


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure:

    - Convert CRLF and CR newlines to standard LF.
    - Replace horizontal whitespace sequences with a single space.
    - Strip leading and trailing spaces around lines.
    - Collapse three or more newlines to two newlines (paragraph boundary).
    - Strip leading/trailing whitespace.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_and_metadata(document_id: str, file_path: Path | str) -> ExtractionResult:
    """Extract structured text page-by-page and metadata from a stored PDF file.

    Preserves 1-based page numbers and strict traceability.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    file_size_bytes = file_path.stat().st_size

    try:
        reader = PdfReader(str(file_path))
    except (PdfReadError, Exception) as err:
        raise ValueError(f"Failed to read PDF file: {str(err)}") from err

    total_pages = len(reader.pages)
    pdf_meta = reader.metadata

    title = None
    author = None
    creation_date = None

    if pdf_meta:
        try:
            if pdf_meta.title:
                title = str(pdf_meta.title).strip() or None
        except Exception:
            title = None

        try:
            if pdf_meta.author:
                author = str(pdf_meta.author).strip() or None
        except Exception:
            author = None

        try:
            if pdf_meta.creation_date:
                creation_date = str(pdf_meta.creation_date).strip() or None
        except Exception:
            try:
                raw_date = pdf_meta.get("/CreationDate")
                creation_date = str(raw_date).strip() if raw_date else None
            except Exception:
                creation_date = None

    metadata = DocumentMetadata(
        title=title,
        author=author,
        creation_date=creation_date,
        total_pages=total_pages,
        file_size_bytes=file_size_bytes,
    )

    pages: list[ExtractedPage] = []
    for idx, page in enumerate(reader.pages):
        page_number = idx + 1  # 1-based page numbering
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        cleaned_text = normalize_whitespace(raw_text)
        pages.append(
            ExtractedPage(
                page_number=page_number,
                text=cleaned_text,
                char_count=len(cleaned_text),
            )
        )

    return ExtractionResult(
        document_id=document_id,
        metadata=metadata,
        pages=pages,
    )
