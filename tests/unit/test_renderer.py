import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from pypdf import PdfWriter

from app.schemas.document import RenderedPage, RenderingResult
from app.services.renderer import RenderingError, render_document, render_page


@pytest.fixture
def dummy_pdf_path(tmp_path: Path):
    """Create a minimal real PDF with 2 empty pages using pure pypdf."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=200)
    
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


def test_render_page_success(dummy_pdf_path: Path, tmp_path: Path):
    doc_id = "doc_1234567890ab"
    
    result = render_page(
        document_id=doc_id,
        file_path=dummy_pdf_path,
        page_number=1,
        dpi=72,  # 72 dpi = 1x zoom
        fmt="png",
        output_dir=str(tmp_path)
    )
    
    assert isinstance(result, RenderedPage)
    assert result.document_id == doc_id
    assert result.page_number == 1
    assert result.format == "png"
    assert result.width == 100
    assert result.height == 100
    assert result.size_bytes > 0
    
    expected_path = tmp_path / doc_id / f"{doc_id}_p1.png"
    assert result.image_path == str(expected_path)
    assert expected_path.is_file()
    assert expected_path.stat().st_size == result.size_bytes


def test_render_document_success(dummy_pdf_path: Path, tmp_path: Path):
    doc_id = "doc_1234567890ab"
    
    result = render_document(
        document_id=doc_id,
        file_path=dummy_pdf_path,
        dpi=72,
        fmt="jpeg",
        output_dir=str(tmp_path)
    )
    
    assert isinstance(result, RenderingResult)
    assert result.document_id == doc_id
    assert result.total_pages_rendered == 2
    assert result.dpi == 72
    assert result.format == "jpeg"
    assert len(result.pages) == 2
    
    assert result.pages[0].page_number == 1
    assert result.pages[0].format == "jpeg"
    assert result.pages[0].height == 100
    assert tmp_path.joinpath(doc_id, f"{doc_id}_p1.jpg").is_file()
    
    assert result.pages[1].page_number == 2
    assert result.pages[1].format == "jpeg"
    assert result.pages[1].height == 200
    assert tmp_path.joinpath(doc_id, f"{doc_id}_p2.jpg").is_file()


def test_render_page_invalid_inputs(dummy_pdf_path: Path):
    with pytest.raises(ValueError, match="Invalid document identifier format"):
        render_page("bad-id", dummy_pdf_path, 1)

    doc_id = "doc_1234567890ab"
        
    with pytest.raises(FileNotFoundError):
        render_page(doc_id, Path("missing.pdf"), 1)

    with pytest.raises(ValueError, match="DPI must be between"):
        render_page(doc_id, dummy_pdf_path, 1, dpi=10)

    with pytest.raises(ValueError, match="Unsupported format"):
        render_page(doc_id, dummy_pdf_path, 1, fmt="gif")

    with pytest.raises(ValueError, match="page_number must be >= 1"):
        render_page(doc_id, dummy_pdf_path, 0)
        
    with pytest.raises(ValueError, match="exceeds document page count"):
        render_page(doc_id, dummy_pdf_path, 99)


def test_render_corrupt_pdf(tmp_path: Path):
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"not a pdf")
    
    with pytest.raises(RenderingError):
        render_document("doc_1234567890ab", corrupt_pdf)
