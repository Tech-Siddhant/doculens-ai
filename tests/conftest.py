import io
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.api.main import create_app
from app.core.config import settings


def generate_pdf_bytes(pages_text: list[str], metadata: dict | None = None) -> bytes:
    """Helper to generate in-memory valid PDF bytes with custom text and metadata."""
    writer = PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        if text:
            stream_obj = DecodedStreamObject()
            content = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode("latin1")
            stream_obj.set_data(content)
            page[NameObject("/Contents")] = writer._add_object(stream_obj)
            font_dict = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            res = DictionaryObject({
                NameObject("/Font"): DictionaryObject({
                    NameObject("/F1"): writer._add_object(font_dict)
                })
            })
            page[NameObject("/Resources")] = writer._add_object(res)

    if metadata:
        meta_dict = {f"/{k}": v for k, v in metadata.items()}
        writer.add_metadata(meta_dict)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def valid_pdf_bytes() -> bytes:
    return generate_pdf_bytes(
        pages_text=["Page 1 introduction text", "Page 2 content details"],
        metadata={"Title": "Sample Document", "Author": "DocuLens Tester"},
    )


@pytest.fixture
def multi_page_pdf_bytes() -> bytes:
    return generate_pdf_bytes(
        pages_text=[
            "DocuLens is a Multimodal Document Intelligence Platform designed for high-accuracy RAG.",
            "Page two covers page-level chunking and preservation of complete source traceability.",
            "Page three discusses multimodal visual retrieval and evaluation against ground truth.",
        ],
        metadata={"Title": "DocuLens Architecture", "Author": "DocuLens Team"},
    )


@pytest.fixture
def empty_page_pdf_bytes() -> bytes:
    return generate_pdf_bytes(pages_text=["Valid page text", ""])


@pytest.fixture(autouse=True)
def setup_test_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    test_upload_dir = tmp_path / "uploads"
    test_upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(test_upload_dir))
    return test_upload_dir


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

