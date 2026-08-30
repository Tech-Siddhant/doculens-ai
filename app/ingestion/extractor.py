import os
from typing import Optional
from pypdf import PdfReader
from app.schemas.documents import DocumentMetadata, ExtractedPage, ExtractionResult


def extract_text_and_metadata(file_path: str, document_id: str) -> ExtractionResult:
    """Extracts text by page and metadata from a PDF file.

    Preserves traceability: document_id -> page_number -> extracted text.
    Handles pages with no text gracefully.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF document for extraction: {str(exc)}") from exc

    raw_meta = reader.metadata or {}
    title = getattr(raw_meta, "title", None) or raw_meta.get("/Title")
    author = getattr(raw_meta, "author", None) or raw_meta.get("/Author")
    creation_date = getattr(raw_meta, "creation_date", None) or raw_meta.get("/CreationDate")

    total_pages = len(reader.pages)
    metadata = DocumentMetadata(
        title=str(title) if title else None,
        author=str(author) if author else None,
        creation_date=str(creation_date) if creation_date else None,
        total_pages=total_pages,
    )

    extracted_pages = []
    for idx, page in enumerate(reader.pages):
        page_number = idx + 1
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        # Clean trailing whitespace per line while preserving structural content
        cleaned_text = "\n".join(line.rstrip() for line in page_text.splitlines())
        extracted_pages.append(
            ExtractedPage(
                page_number=page_number,
                text=cleaned_text,
                character_count=len(cleaned_text),
            )
        )

    return ExtractionResult(
        document_id=document_id,
        metadata=metadata,
        pages=extracted_pages,
    )
