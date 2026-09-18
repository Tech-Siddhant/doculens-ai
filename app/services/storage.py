import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from app.core.config import settings
from app.schemas.document import DocumentListItem

DOC_ID_PATTERN = re.compile(r"^doc_[0-9a-f]{12}$")


def generate_document_id() -> str:
    """Generate a traceable, deterministic-format document identifier: doc_<12 hex chars>."""
    return f"doc_{secrets.token_hex(6)}"


def is_valid_document_id(document_id: str) -> bool:
    """Validate document_id format to avoid path traversal and malformed inputs."""
    return bool(DOC_ID_PATTERN.match(document_id))


def get_upload_dir() -> Path:
    """Ensure upload directory exists and return Path."""
    path = Path(settings.UPLOAD_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_document_path(document_id: str) -> Path | None:
    """Resolve storage path for a given document_id if valid and existing."""
    if not is_valid_document_id(document_id):
        return None
    file_path = get_upload_dir() / f"{document_id}.pdf"
    if file_path.is_file():
        return file_path
    return None


def save_uploaded_pdf(document_id: str, content: bytes) -> Path:
    """Save validated PDF bytes to controlled storage path."""
    if not is_valid_document_id(document_id):
        raise ValueError(f"Invalid document identifier format: {document_id}")
    file_path = get_upload_dir() / f"{document_id}.pdf"
    file_path.write_bytes(content)
    return file_path


def get_stored_document(document_id: str) -> DocumentListItem | None:
    """Retrieve metadata and status for a single document."""
    file_path = get_document_path(document_id)
    if not file_path:
        return None

    size_bytes = file_path.stat().st_size
    mtime = file_path.stat().st_mtime
    uploaded_at = datetime.fromtimestamp(mtime, timezone.utc).isoformat()

    total_pages = 0
    filename = f"{document_id}.pdf"
    status = "ready"

    try:
        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)
        try:
            if reader.metadata and reader.metadata.title:
                meta_title = str(reader.metadata.title).strip()
                if meta_title:
                    filename = meta_title if meta_title.endswith(".pdf") else f"{meta_title}.pdf"
        except Exception:
            pass
    except Exception:
        status = "failed"

    return DocumentListItem(
        document_id=document_id,
        filename=filename,
        content_type="application/pdf",
        size_bytes=size_bytes,
        status=status,
        total_pages=total_pages,
        uploaded_at=uploaded_at,
    )


def list_stored_documents() -> list[DocumentListItem]:
    """List all stored PDF documents in upload directory, sorted newest first."""
    upload_dir = get_upload_dir()
    if not upload_dir.exists():
        return []

    documents: list[DocumentListItem] = []
    for file_path in upload_dir.glob("doc_*.pdf"):
        doc_id = file_path.stem
        if not is_valid_document_id(doc_id):
            continue
        doc_item = get_stored_document(doc_id)
        if doc_item:
            documents.append(doc_item)

    # Sort newest first
    documents.sort(key=lambda d: d.uploaded_at or "", reverse=True)
    return documents

