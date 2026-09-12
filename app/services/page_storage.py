"""Page image storage and metadata service.

Manages stored rendered page images with deterministic document/page structure.
Provides retrieval of page metadata without re-rendering.
"""

import logging
from pathlib import Path

from app.core.config import settings
from app.schemas.document import RenderedPage
from app.services.storage import is_valid_document_id

logger = logging.getLogger(__name__)


class PageStorageError(Exception):
    """Raised when page storage operations fail."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def get_page_storage_dir(document_id: str, base_dir: str | None = None) -> Path:
    """Get document-specific page storage directory path.
    
    Args:
        document_id: Document identifier (doc_<12 hex>).
        base_dir: Optional override for base storage directory.
        
    Returns:
        Path to document's page storage directory.
        
    Raises:
        ValueError: If document_id format is invalid.
    """
    if not is_valid_document_id(document_id):
        raise ValueError(f"Invalid document identifier format: {document_id}")
    path = Path(base_dir or settings.RENDER_OUTPUT_DIR) / document_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_stored_page_path(
    document_id: str,
    page_number: int,
    fmt: str | None = None,
    base_dir: str | None = None,
) -> Path | None:
    """Get path to a stored rendered page image if it exists.
    
    Args:
        document_id: Document identifier.
        page_number: 1-based page number.
        fmt: Optional image format ("png" or "jpeg"). If None, checks png, jpg, jpeg.
        base_dir: Optional override for base storage directory.
        
    Returns:
        Path to image file if it exists, None otherwise.
        
    Raises:
        ValueError: If document_id format is invalid or page_number < 1.
    """
    if not is_valid_document_id(document_id):
        raise ValueError(f"Invalid document identifier format: {document_id}")
    if page_number < 1:
        raise ValueError(f"page_number must be >= 1, got {page_number}")
    
    page_dir = get_page_storage_dir(document_id, base_dir)

    if fmt is not None:
        ext = "jpg" if fmt == "jpeg" else fmt
        page_path = page_dir / f"{document_id}_p{page_number}.{ext}"
        return page_path if page_path.is_file() else None
    
    for ext in ["png", "jpg", "jpeg"]:
        page_path = page_dir / f"{document_id}_p{page_number}.{ext}"
        if page_path.is_file():
            return page_path
    
    return None


def get_stored_page_metadata(
    document_id: str,
    page_number: int,
    fmt: str = "png",
    base_dir: str | None = None,
) -> RenderedPage | None:
    """Get metadata for a stored rendered page.
    
    Args:
        document_id: Document identifier.
        page_number: 1-based page number.
        fmt: Image format ("png" or "jpeg").
        base_dir: Optional override for base storage directory.
        
    Returns:
        RenderedPage with metadata if page exists, None otherwise.
        
    Raises:
        ValueError: If document_id format is invalid or page_number < 1.
    """
    page_path = get_stored_page_path(document_id, page_number, fmt, base_dir)
    if not page_path:
        return None
    
    try:
        # Read image metadata without loading full image into memory
        from PIL import Image
        with Image.open(page_path) as img:
            width, height = img.size
        
        size_bytes = page_path.stat().st_size
        
        return RenderedPage(
            document_id=document_id,
            page_number=page_number,
            image_path=str(page_path),
            width=width,
            height=height,
            format=fmt,
            size_bytes=size_bytes,
        )
    except Exception as err:
        logger.warning(
            "Failed to read metadata for %s page %d: %s",
            document_id, page_number, err
        )
        return None


def list_stored_pages(
    document_id: str,
    base_dir: str | None = None,
) -> list[RenderedPage]:
    """List all stored rendered pages for a document.
    
    Args:
        document_id: Document identifier.
        base_dir: Optional override for base storage directory.
        
    Returns:
        List of RenderedPage metadata for all stored pages, sorted by page_number.
        
    Raises:
        ValueError: If document_id format is invalid.
    """
    if not is_valid_document_id(document_id):
        raise ValueError(f"Invalid document identifier format: {document_id}")
    
    page_dir = get_page_storage_dir(document_id, base_dir)
    if not page_dir.exists():
        return []
    
    pages: list[RenderedPage] = []
    
    # Scan for image files matching pattern: {doc_id}_p{N}.{ext}
    for ext in ["png", "jpg", "jpeg"]:
        for img_path in page_dir.glob(f"{document_id}_p*.{ext}"):
            try:
                # Extract page number from filename
                stem = img_path.stem  # e.g., "doc_xxx_p5"
                page_num_str = stem.split("_p")[-1]
                page_number = int(page_num_str)
                
                if page_number < 1:
                    continue
                
                from PIL import Image
                with Image.open(img_path) as img:
                    width, height = img.size
                
                fmt = "jpeg" if ext in ["jpg", "jpeg"] else ext
                
                pages.append(
                    RenderedPage(
                        document_id=document_id,
                        page_number=page_number,
                        image_path=str(img_path),
                        width=width,
                        height=height,
                        format=fmt,
                        size_bytes=img_path.stat().st_size,
                    )
                )
            except (ValueError, OSError) as err:
                logger.warning("Skipping invalid page image %s: %s", img_path, err)
                continue
    
    # Sort by page number
    pages.sort(key=lambda p: p.page_number)
    return pages


def delete_stored_pages(
    document_id: str,
    base_dir: str | None = None,
) -> int:
    """Delete all stored rendered pages for a document.
    
    Args:
        document_id: Document identifier.
        base_dir: Optional override for base storage directory.
        
    Returns:
        Number of page images deleted.
        
    Raises:
        ValueError: If document_id format is invalid.
    """
    if not is_valid_document_id(document_id):
        raise ValueError(f"Invalid document identifier format: {document_id}")
    
    page_dir = get_page_storage_dir(document_id, base_dir)
    if not page_dir.exists():
        return 0
    
    deleted_count = 0
    for img_path in page_dir.iterdir():
        if img_path.is_file():
            try:
                img_path.unlink()
                deleted_count += 1
            except OSError as err:
                logger.warning("Failed to delete %s: %s", img_path, err)
    
    # Remove empty directory
    try:
        page_dir.rmdir()
    except OSError:
        pass  # Directory not empty or other issue
    
    logger.info("Deleted %d page images for document %s", deleted_count, document_id)
    return deleted_count
