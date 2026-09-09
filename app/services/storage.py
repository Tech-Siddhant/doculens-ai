import re
import secrets
from pathlib import Path

from app.core.config import settings

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
