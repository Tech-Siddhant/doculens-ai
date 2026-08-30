import os
import uuid


def save_uploaded_pdf(content: bytes, filename: str, storage_dir: str) -> tuple[str, str]:
    """Saves PDF file content to storage directory with a unique document ID.

    Returns tuple of (document_id, file_path).
    """
    os.makedirs(storage_dir, exist_ok=True)
    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    file_path = os.path.join(storage_dir, f"{document_id}.pdf")

    with open(file_path, "wb") as f:
        f.write(content)

    return document_id, file_path


def get_pdf_path(document_id: str, storage_dir: str) -> str:
    """Returns absolute/relative path to stored PDF for document_id.

    Raises FileNotFoundError if document file does not exist.
    """
    file_path = os.path.join(storage_dir, f"{document_id}.pdf")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document with ID '{document_id}' not found in storage.")
    return file_path
