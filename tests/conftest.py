import io
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest


def create_sample_pdf_with_text(pages_text: list[str], title: str = "Test Document", author: str = "Test Author") -> bytes:
    """Generates PDF bytes with real extractable text on each page."""
    writer = PdfWriter()

    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)

        # Build content stream with PDF text operators
        content_bytes = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET".encode("latin-1")
        stream = DecodedStreamObject()
        stream.set_data(content_bytes)

        # Font object definition
        font_dict = DictionaryObject()
        font_dict[NameObject("/Type")] = NameObject("/Font")
        font_dict[NameObject("/Subtype")] = NameObject("/Type1")
        font_dict[NameObject("/BaseFont")] = NameObject("/Helvetica")

        resources = DictionaryObject()
        resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font_dict})

        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = stream

    writer.add_metadata({"/Title": title, "/Author": author})
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def valid_pdf_bytes() -> bytes:
    """Returns valid minimal PDF bytes."""
    return create_sample_pdf_with_text(["Hello DocuLens AI"], title="Test Document", author="Test Author")

