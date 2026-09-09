"""PDF page renderer using PyMuPDF.

Renders PDF pages to images (PNG/JPEG) with configurable DPI.
Preserves document_id and 1-based page_number association.
"""

import logging
from pathlib import Path

import pymupdf

from app.core.config import settings
from app.schemas.document import RenderedPage, RenderingResult
from app.services.storage import is_valid_document_id

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"png", "jpeg"}


class RenderingError(Exception):
    """Raised when PDF page rendering fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _get_render_dir(document_id: str, base_dir: str | None = None) -> Path:
    """Ensure document-specific render output directory exists and return Path."""
    # Security: validate document_id to prevent path traversal
    if not is_valid_document_id(document_id):
        raise ValueError(f"Invalid document identifier format: {document_id}")
    path = Path(base_dir or settings.RENDER_OUTPUT_DIR) / document_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_render_inputs(
    document_id: str,
    file_path: Path,
    dpi: int,
    fmt: str,
) -> None:
    """Validate inputs before rendering."""
    if not is_valid_document_id(document_id):
        raise ValueError(f"Invalid document identifier format: {document_id}")
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if dpi < 36 or dpi > 600:
        raise ValueError(f"DPI must be between 36 and 600, got {dpi}")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{fmt}', must be one of {SUPPORTED_FORMATS}")


def render_page(
    document_id: str,
    file_path: Path,
    page_number: int,
    *,
    dpi: int | None = None,
    fmt: str | None = None,
    output_dir: str | None = None,
) -> RenderedPage:
    """Render a single PDF page to an image file.

    Args:
        document_id: Document identifier (doc_<12 hex>).
        file_path: Path to the PDF file.
        page_number: 1-based page number to render.
        dpi: Resolution in dots per inch. Defaults to settings.RENDER_DPI.
        fmt: Output format ("png" or "jpeg"). Defaults to settings.RENDER_FORMAT.
        output_dir: Override output directory.

    Returns:
        RenderedPage with image path, dimensions, and size.
    """
    dpi = dpi or settings.RENDER_DPI
    fmt = fmt or settings.RENDER_FORMAT
    _validate_render_inputs(document_id, file_path, dpi, fmt)

    if page_number < 1:
        raise ValueError(f"page_number must be >= 1, got {page_number}")

    try:
        doc = pymupdf.Document(str(file_path))
    except Exception as err:
        raise RenderingError(f"Failed to open PDF: {err}") from err

    try:
        total_pages = len(doc)
        if page_number > total_pages:
            raise ValueError(
                f"page_number {page_number} exceeds document page count ({total_pages})"
            )

        page = doc[page_number - 1]  # 0-based index
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        out_dir = _get_render_dir(document_id, output_dir)
        ext = "jpg" if fmt == "jpeg" else fmt
        out_file = out_dir / f"{document_id}_p{page_number}.{ext}"
        img_bytes = pix.tobytes(fmt)
        out_file.write_bytes(img_bytes)

        rendered = RenderedPage(
            document_id=document_id,
            page_number=page_number,
            image_path=str(out_file),
            width=pix.width,
            height=pix.height,
            format=fmt,
            size_bytes=len(img_bytes),
        )
        logger.info(
            "Rendered page %d of %s: %dx%d %s (%d bytes)",
            page_number, document_id, pix.width, pix.height, fmt, len(img_bytes),
        )
        return rendered
    finally:
        doc.close()


def render_document(
    document_id: str,
    file_path: Path,
    *,
    dpi: int | None = None,
    fmt: str | None = None,
    output_dir: str | None = None,
) -> RenderingResult:
    """Render all pages of a PDF document to image files.

    Args:
        document_id: Document identifier.
        file_path: Path to the PDF file.
        dpi: Resolution in dots per inch. Defaults to settings.RENDER_DPI.
        fmt: Output format ("png" or "jpeg"). Defaults to settings.RENDER_FORMAT.
        output_dir: Override output directory.

    Returns:
        RenderingResult with all rendered pages.
    """
    dpi = dpi or settings.RENDER_DPI
    fmt = fmt or settings.RENDER_FORMAT
    _validate_render_inputs(document_id, file_path, dpi, fmt)

    try:
        doc = pymupdf.Document(str(file_path))
    except Exception as err:
        raise RenderingError(f"Failed to open PDF: {err}") from err

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise RenderingError("PDF contains no pages to render")

        out_dir = _get_render_dir(document_id, output_dir)
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        ext = "jpg" if fmt == "jpeg" else fmt

        pages: list[RenderedPage] = []
        for idx in range(total_pages):
            page_number = idx + 1
            page = doc[idx]
            pix = page.get_pixmap(matrix=mat)
            out_file = out_dir / f"{document_id}_p{page_number}.{ext}"
            img_bytes = pix.tobytes(fmt)
            out_file.write_bytes(img_bytes)
            pages.append(
                RenderedPage(
                    document_id=document_id,
                    page_number=page_number,
                    image_path=str(out_file),
                    width=pix.width,
                    height=pix.height,
                    format=fmt,
                    size_bytes=len(img_bytes),
                )
            )

        logger.info(
            "Rendered %d pages of %s at %d DPI (%s)",
            total_pages, document_id, dpi, fmt,
        )
        return RenderingResult(
            document_id=document_id,
            total_pages_rendered=total_pages,
            dpi=dpi,
            format=fmt,
            pages=pages,
        )
    finally:
        doc.close()
